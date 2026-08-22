"""Benchmark cases.

Each case module exposes ``build(...) -> Simulation`` and ``run(...) -> CaseResult``
so the CLI can treat them uniformly.  ``build`` is separated out because the live
viewer needs an un-run simulation to animate.

A case may also declare ``CONVERGENCE_METRICS``, mapping the metrics worth
refining against to what kind of quantity each is.  That is what lets
:func:`grid_study` run a grid-refinement study on any case rather than only on
the one with an exact solution.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..config import BCKind, BCSpec
from ..core.timestepper import DivergenceError
from ..physics.incompressible import Simulation

log = logging.getLogger(__name__)

#: Boundary kinds that count as an outflow, and so can be retyped by
#: :func:`override_outlet`.
OUTLET_KINDS = (BCKind.OUTLET, BCKind.PRESSURE_OUTLET)


def override_outlet(boundary_config: dict[str, BCSpec],
                    outlet_type: str | BCKind | None = None,
                    p_ref: float | None = None) -> dict[str, BCSpec]:
    """Retype a case's outflow boundary without editing the case file.

    Backs the ``--outlet-type`` and ``--p-ref`` command-line flags.  Every wall
    currently carrying an outflow condition is replaced; walls of any other kind
    are untouched, so an inlet or a symmetry plane can never be clobbered by
    accident.

    Passing neither argument returns ``boundary_config`` unchanged, which is what
    keeps each case's own choice as the default.

    Raises
    ------
    ValueError
        If the case has no outflow boundary at all -- a closed cavity or a
        streamwise-periodic channel has nothing to retype, and silently doing
        nothing there would be worse than saying so.
    """
    if outlet_type is None and p_ref is None:
        return boundary_config

    kind = None if outlet_type is None else BCKind(outlet_type)
    if kind is not None and kind not in OUTLET_KINDS:
        raise ValueError(
            f"outlet_type must be one of "
            f"{[k.value for k in OUTLET_KINDS]}, got {kind.value!r}"
        )

    walls = [w for w, spec in boundary_config.items() if spec.kind in OUTLET_KINDS]
    if not walls:
        current = ", ".join(f"{w}={s.kind.value}" for w, s in boundary_config.items())
        raise ValueError(
            "this case has no outflow boundary to retype, so --outlet-type and "
            f"--p-ref do not apply to it (its walls are: {current})"
        )

    updated = dict(boundary_config)
    for wall in walls:
        spec = updated[wall]
        new_kind = kind if kind is not None else spec.kind
        new_ref = spec.p_ref if p_ref is None else p_ref
        if new_kind is BCKind.PRESSURE_OUTLET:
            updated[wall] = BCSpec(new_kind, p_ref=new_ref)
        else:
            updated[wall] = BCSpec(new_kind)
            if p_ref is not None:
                log.warning(
                    "p_ref=%g is ignored on the %s wall: a velocity outlet lets "
                    "the pressure float. Use --outlet-type pressure_outlet to "
                    "anchor it.", p_ref, wall,
                )
    return updated


@dataclass
class CaseResult:
    """Outcome of a benchmark run: the state, the metrics and the files written."""

    name: str
    simulation: Simulation
    metrics: dict[str, float] = field(default_factory=dict)
    outputs: list[Path] = field(default_factory=list)
    #: Human-readable pass/fail lines produced by the case's own validation.
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every validation check succeeded."""
        return all(ok for _, ok, _ in self.checks)

    def report(self) -> str:
        """Formatted summary for the console."""
        lines = [f"=== {self.name} ==="]
        for key, value in self.metrics.items():
            lines.append(f"  {key:<28s} {value:.6g}" if isinstance(value, float)
                         else f"  {key:<28s} {value}")
        if self.checks:
            lines.append("  validation:")
            for label, ok, detail in self.checks:
                lines.append(f"    [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if self.outputs:
            lines.append("  files:")
            lines.extend(f"    {p}" for p in self.outputs)
        return "\n".join(lines)


def available_cases() -> dict[str, str]:
    """Map case name -> one-line description, for ``--list-cases``."""
    return {
        "cavity": "Lid-driven cavity; validated against Ghia et al. (1982), Re = 100/400/1000",
        "channel": "Plane Poiseuille channel flow; validated against the analytical parabola",
        "cylinder": "Flow past a circular cylinder (immersed boundary); Cd and vortex shedding",
        "taylor_green": "Taylor-Green vortex; exact unsteady solution used for convergence studies",
    }


def load_case(name: str):
    """Import and return the module implementing ``name``."""
    from importlib import import_module

    modules = {
        "cavity": "lid_driven_cavity",
        "channel": "channel_flow",
        "cylinder": "cylinder_flow",
        "taylor_green": "taylor_green",
    }
    if name not in modules:
        raise ValueError(
            f"unknown case {name!r}; available: {sorted(modules)}"
        )
    return import_module(f".{modules[name]}", package=__package__)


# --------------------------------------------------------------------------- #
# Grid-refinement studies
# --------------------------------------------------------------------------- #
#: Resolutions used when none are named.
DEFAULT_RESOLUTIONS = (32, 64, 128)

#: Relative change between the two finest grids below which a metric is called
#: settled.  Two per cent is an engineering judgement, not a theorem: it is
#: small enough that a quantity still drifting at the 5-10% level -- which is
#: what an under-resolved body looks like -- is reported as unconverged.
SETTLED_TOLERANCE = 0.02

#: A metric this small was never a measurement -- it is a quantity that is
#: structurally zero, like the lift on a symmetric steady wake.  Comparing
#: successive values of one relatively is meaningless: two different roundings
#: of zero differ by tens of per cent and would be reported as a grid that
#: refuses to converge.
NOISE_FLOOR = 1.0e-9

#: Observed order below which an error metric is not really converging.  Half
#: of first order: a sequence decreasing more slowly than that is decreasing
#: rather than converging, and refining it further will not buy an answer.
MIN_CONVERGENCE_ORDER = 0.5


def parse_resolutions(text: str) -> tuple[int, ...]:
    """Parse ``"64,128,256"`` into validated, increasing cell counts."""
    parts = [p.strip() for p in str(text).replace(" ", ",").split(",") if p.strip()]
    try:
        values = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"resolutions must be a comma-separated list of integers, got {text!r}"
        ) from None
    if len(values) < 2:
        raise ValueError(
            f"a refinement study needs at least two resolutions, got {list(values)}"
        )
    if any(n < 4 for n in values):
        raise ValueError(f"every resolution must be at least 4, got {list(values)}")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(
            f"resolutions must increase, got {list(values)}; the study reports the "
            "order between successive grids and reads the finest pair as the "
            "asymptotic estimate"
        )
    return values


def case_aspect(case: str) -> float | None:
    """The case's own ``ny / nx``, or ``None`` when it derives ``ny`` itself.

    Refining a study has to preserve the shape of the domain the case chose --
    a channel is 1:2 and a cylinder 2:1, and squaring either of them would be a
    different problem, not a finer one.
    """
    params = inspect.signature(load_case(case).run).parameters
    nx = params["nx"].default if "nx" in params else None
    ny = params["ny"].default if "ny" in params else None
    if not isinstance(nx, int) or not isinstance(ny, int):
        return None
    return ny / nx


def case_resolution(case: str) -> int | None:
    """The case's own default ``nx``, or ``None`` if it does not take one.

    What a diagnostic study measures itself against: the grid the case would
    have used had nobody asked for a study, so a pass run below it says
    something about the run the user was actually going to do.
    """
    params = inspect.signature(load_case(case).run).parameters
    nx = params["nx"].default if "nx" in params else None
    return nx if isinstance(nx, int) else None


@dataclass
class GridStudy:
    """How a case's reported metrics behave under grid refinement.

    ``values`` holds one number per resolution for each tracked metric, and
    ``kinds`` says what each metric is: an ``"error"`` against a known reference
    -- for which an observed order of accuracy is meaningful -- or a
    ``"quantity"`` the run measures with nothing to compare against, which can
    only be watched to stop moving.
    """

    case: str
    resolutions: list[int]
    values: dict[str, list[float]]
    kinds: dict[str, str]
    #: The full outcome of each run, in the same order as ``resolutions``.
    #: ``values`` holds only the metrics being refined against; everything else
    #: a run reported -- its divergence, its blockage, its own pass/fail checks
    #: -- lives here, which is what lets :mod:`pycfd.analysis.diagnose` ask
    #: questions the study itself was not set up to answer.
    results: list[CaseResult] = field(default_factory=list)

    def changes(self, metric: str) -> list[float]:
        """Relative change between each successive pair of grids.

        A pair that is structurally zero on both grids counts as no change at
        all -- see :data:`NOISE_FLOOR`.
        """
        series = self.values[metric]
        out = []
        for a, b in zip(series, series[1:]):
            if abs(a) < NOISE_FLOOR and abs(b) < NOISE_FLOOR:
                out.append(0.0)
            elif a:
                out.append(abs(b - a) / abs(a))
            else:
                out.append(float("nan"))
        return out

    def observed_order(self, metric: str) -> float:
        """Order of accuracy from the two finest grids, for an error metric.

        ``nan`` for a measured quantity: without a reference there is no error
        to halve, and fitting an order to one anyway is how a sequence that
        never entered its asymptotic regime gets extrapolated as though it had.
        """
        if self.kinds.get(metric) != "error":
            return float("nan")
        from ..analysis.validation import convergence_order

        return convergence_order(self.resolutions, self.values[metric],
                                 metric).observed_order

    def settled(self, metric: str, tol: float = SETTLED_TOLERANCE) -> bool:
        """Whether the finest pair of grids agree to within ``tol``."""
        changes = self.changes(metric)
        return bool(changes) and math.isfinite(changes[-1]) and changes[-1] <= tol

    def converging(self, metric: str, tol: float = SETTLED_TOLERANCE) -> bool:
        """Whether ``metric`` is doing what refinement is supposed to do to it.

        Which question that is depends on what the metric *is*, and asking the
        wrong one inverts the answer.  A measured quantity converges by ceasing
        to move, so :meth:`settled` is the whole test.  An error converges by
        moving -- shrinking, at a rate the scheme's order predicts -- so an
        error that has settled has *stopped* converging, and reporting it as
        healthy would flag a second-order solver working perfectly as a failure.
        """
        if self.kinds.get(metric) != "error":
            return self.settled(metric, tol)

        series = self.values[metric]
        if len(series) < 2:
            return False
        if abs(series[-1]) <= NOISE_FLOOR:
            return True                   # exact to round-off; nothing left to do
        if abs(series[-1]) >= abs(series[-2]):
            return False                  # refining made it worse, or no better
        order = self.observed_order(metric)
        return math.isfinite(order) and order >= MIN_CONVERGENCE_ORDER

    def verdict(self, metric: str) -> str:
        """One line saying whether ``metric`` behaved, in its own terms."""
        healthy = self.converging(metric)
        if self.kinds.get(metric) == "error":
            order = self.observed_order(metric)
            rate = f" at order {order:.2f}" if math.isfinite(order) else ""
            return (f"converging{rate}" if healthy else
                    "NOT CONVERGING: refining is not reducing the error")
        return (f"settled at {SETTLED_TOLERANCE * 100:g}% between the two "
                f"finest grids" if healthy else
                f"STILL MOVING at more than {SETTLED_TOLERANCE * 100:g}% "
                f"between the two finest grids")

    def richardson(self, metric: str):
        """Richardson/GCI estimate for ``metric``, or ``None`` under three grids.

        Two grids say whether a number moved; three say whether it is *going*
        anywhere, and that is a different question -- see
        :mod:`pycfd.analysis.richardson`.
        """
        if len(self.resolutions) < 3:
            return None
        from ..analysis.richardson import richardson

        return richardson(self.resolutions, self.values[metric])

    @property
    def passed(self) -> bool:
        """True when every tracked metric behaved under refinement."""
        return all(self.converging(m) for m in self.values)

    def table(self) -> str:
        """One block per metric: value at each grid, change, and order."""
        lines = []
        for metric, series in self.values.items():
            kind = self.kinds.get(metric, "quantity")
            lines.append(f"  {metric}  ({kind})")
            lines.append(f"    {'N':>6} {'value':>14} {'change':>10}")
            changes = self.changes(metric)
            for k, (n, value) in enumerate(zip(self.resolutions, series)):
                change = "" if k == 0 else f"{changes[k - 1] * 100:9.2f}%"
                lines.append(f"    {n:6d} {value:14.6g} {change:>10}")
            order = self.observed_order(metric)
            if math.isfinite(order):
                # Named for how it was obtained, because the Richardson block
                # below reports an order too and the two are different
                # estimates: this one is the ratio of the error magnitudes
                # themselves, which assumes the error is a clean C*h^p. Where
                # they disagree, that assumption is what is failing.
                lines.append(f"    order from the error magnitudes: {order:.3f}")
            lines.append(f"    {self.verdict(metric)}")
            estimate = self.richardson(metric)
            if estimate is not None:
                lines.append(estimate.report())
        return "\n".join(lines)

    def extrapolated(self) -> dict[str, float]:
        """The zero-cell-size limit of each metric the study could extrapolate."""
        out = {}
        for metric in self.values:
            estimate = self.richardson(metric)
            if estimate is not None and estimate.trustworthy:
                out[metric] = estimate.extrapolated
        return out

    def report(self) -> str:
        """Formatted summary for the console."""
        head = (f"=== {self.case} grid study "
                f"({', '.join(str(n) for n in self.resolutions)}) ===")
        lines = [head, self.table()]
        if len(self.resolutions) < 3:
            lines.append("  two grids show whether a number moved; a third "
                         "would show whether it is going anywhere")
        lines.append(
            "  every tracked metric behaved under refinement" if self.passed else
            "  at least one metric did not: refine further before trusting it"
        )
        return "\n".join(lines)


def grid_study(case: str, resolutions=DEFAULT_RESOLUTIONS,
               progress: bool = False, **kwargs) -> GridStudy:
    """Run ``case`` at each resolution and track how its metrics move.

    This is the general form of a convergence study: the Taylor--Green vortex
    has an exact solution and so can report a true error, but every other case
    can still be refined and watched.  Which metrics are worth watching is the
    case's own declaration -- ``CONVERGENCE_METRICS`` in its module -- since
    only the case knows which of its numbers are results and which are
    bookkeeping.
    """
    module = load_case(case)
    kinds = dict(getattr(module, "CONVERGENCE_METRICS", {}))
    if not kinds:
        raise ValueError(
            f"case '{case}' does not declare CONVERGENCE_METRICS, so there is "
            "nothing to refine it against; add the mapping to its module to "
            "make it studiable"
        )

    aspect = case_aspect(case)
    values: dict[str, list[float]] = {m: [] for m in kinds}
    results: list[CaseResult] = []
    for n in resolutions:
        extra = {} if aspect is None else {"ny": max(4, round(n * aspect))}
        try:
            result = module.run(nx=n, make_plots=False, progress=progress,
                                **extra, **kwargs)
        except DivergenceError as exc:
            # Which grid blew up is the whole content of the message: a study
            # that fails only on its coarsest grid is a different problem from
            # one that fails everywhere, and the bare exception says neither.
            raise DivergenceError(
                f"the '{case}' case diverged at N={n}: {exc}"
            ) from exc
        for metric in kinds:
            if metric not in result.metrics:
                raise ValueError(
                    f"case '{case}' declares '{metric}' as a convergence metric "
                    f"but did not report it at N={n}"
                )
            values[metric].append(float(result.metrics[metric]))
        results.append(result)
        log.info("N=%d: %s", n,
                 "  ".join(f"{m}={values[m][-1]:.6g}" for m in kinds))

    return GridStudy(case, [int(n) for n in resolutions], values, kinds, results)
