"""One command that asks whether a setup is sound and its answer believable.

Every question this module asks was asked by hand at some point, in a different
order each time, and forgotten at least once.  Is the body resolved?  Does the
domain confine it?  Has the projection actually enforced continuity?  Did the
force coefficient stop moving under refinement, or is it merely moving slowly?
None of them is hard.  The failure mode is that a run produces a plausible
number regardless, and nothing about the number says which of these went
unasked.

So the pass runs the case on a coarse and a medium grid -- both *below* the one
the case would otherwise use -- and reports a verdict per question:

``ok``
    Nothing to say.
``warn``
    The run is usable but biased, unconverged, or resting on fewer independent
    observations than it looks.  A judgement call, and the caller's to make.
``fail``
    The run cannot support a quantitative answer: it blew up, continuity is
    materially violated, or the body is a handful of cells wide.

Why the grids sit below the production one
------------------------------------------
Refining is the only way to find out whether a number has converged, and the
cheapest place to look is underneath the grid you were going to use anyway: two
runs at half and three-quarters resolution cost a fraction of one at full, and
they bracket the trend that says whether full is enough.  Reading it the other
way round -- running the production grid and one finer -- answers the same
question at several times the price.

What this pass deliberately does not do
---------------------------------------
It does not shorten the run.  A truncated record makes a different flow, not a
cheaper look at the same one, and the force coefficient it reports would be an
answer to a question nobody asked.  The saving comes from the grid alone; pass
``--t-end`` explicitly if a shorter record is what is wanted, and read the
stationarity verdict knowing that is what was asked for.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from ..cases import (
    DEFAULT_RESOLUTIONS,
    GridStudy,
    SETTLED_TOLERANCE,
    case_resolution,
    grid_study,
)
from ..core.timestepper import DivergenceError

log = logging.getLogger(__name__)

#: Verdict levels, least severe first.
OK, WARN, FAIL = "ok", "warn", "fail"

_SEVERITY = {OK: 0, WARN: 1, FAIL: 2}

#: Fractions of the case's own default resolution used when none are named.
#: Two grids, because two is what it takes to see whether a number moves and
#: this pass is meant to be affordable enough that people actually run it.  A
#: third -- ``--resolutions`` -- buys Richardson extrapolation on top.
DIAGNOSE_FRACTIONS = (0.5, 0.75)

#: No diagnostic grid below this, however small the case's default.
MIN_DIAGNOSTIC_N = 8

#: Discrete divergence the projection is expected to leave behind.  It solves
#: the Poisson equation to :data:`~pycfd.config.DEFAULT_POISSON_TOL` (1e-10) and
#: in practice lands near round-off, so anything at 1e-8 means the linear solve
#: is not converging, and 1e-6 means continuity is violated by enough to make
#: the pressure -- and therefore the pressure drag -- meaningless.
DIVERGENCE_WARN = 1.0e-8
DIVERGENCE_FAIL = 1.0e-6

#: Blockage ratio ``d / H``.  Below 5% the confining walls leave the drag
#: coefficient essentially alone; from there it climbs, and by 10% the
#: correction is several per cent of Cd -- larger than the spread between
#: published values for the unbounded cylinder.
BLOCKAGE_NOTICEABLE = 0.05
BLOCKAGE_SUBSTANTIAL = 0.10

#: Cells spanning the body.  A staircase mask needs about 16 before surface
#: forces are quantitative; below 4 the "body" is a few pixels and its drag is
#: an artefact of which cells happened to be flagged solid.
CELLS_QUANTITATIVE = 16.0
CELLS_MINIMUM = 4.0


@dataclass(frozen=True)
class Finding:
    """One question, its verdict, and what to do about it.

    ``advice`` is separated from ``detail`` because the two are read at
    different moments: the detail says what was measured, and only somebody who
    has decided the measurement is a problem goes looking for the remedy.
    """

    name: str
    level: str
    detail: str
    advice: str = ""

    @property
    def severity(self) -> int:
        return _SEVERITY[self.level]

    def lines(self) -> list[str]:
        """The finding as console lines, indented under a level tag."""
        tag = {OK: "  OK  ", WARN: " WARN ", FAIL: " FAIL "}[self.level]
        out = [f"  [{tag}] {self.name}", f"           {self.detail}"]
        if self.advice:
            out.append(f"           -> {self.advice}")
        return out


@dataclass
class Diagnosis:
    """Every verdict on one setup, and whether it holds together.

    Returned whatever happened, including when the solver blew up -- a
    diagnostic that raises is one people route around.
    """

    case: str
    resolutions: list[int]
    findings: list[Finding] = field(default_factory=list)
    study: GridStudy | None = None
    #: The grid the case would have used unprompted, for context in the header.
    production: int | None = None

    @property
    def level(self) -> str:
        """The worst verdict returned, or ``ok`` when nothing was flagged."""
        if not self.findings:
            return OK
        return max(self.findings, key=lambda f: f.severity).level

    @property
    def usable(self) -> bool:
        """True when nothing failed outright -- warnings are the caller's call."""
        return self.level != FAIL

    def counts(self) -> dict[str, int]:
        """How many findings landed at each level."""
        return {level: sum(1 for f in self.findings if f.level == level)
                for level in (OK, WARN, FAIL)}

    def report(self) -> str:
        """Formatted verdict, safe to print whatever the run turned out to be."""
        grids = ", ".join(str(n) for n in self.resolutions)
        head = f"=== {self.case} diagnosis: grids {grids} ==="
        if self.production is not None:
            head = (f"=== {self.case} diagnosis: grids {grids} "
                    f"(the case's own default is {self.production}) ===")

        lines = [head, ""]
        # Worst first: the reason somebody ran this is at the top of the output
        # rather than at the bottom of a list of things that were fine.
        for finding in sorted(self.findings, key=lambda f: -f.severity):
            lines.extend(finding.lines())
            lines.append("")

        counts = self.counts()
        lines.append(
            f"  {counts[FAIL]} failed, {counts[WARN]} warned, {counts[OK]} passed"
        )
        if counts[FAIL]:
            lines.append("  this setup will not support a quantitative answer")
        elif counts[WARN]:
            lines.append("  usable, with the caveats above")
        else:
            lines.append("  nothing to flag")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Picking the grids
# --------------------------------------------------------------------------- #
def diagnostic_resolutions(case: str,
                           fractions=DIAGNOSE_FRACTIONS) -> tuple[int, ...]:
    """A coarse/medium pair sitting below the case's own default grid.

    Anchoring on the case's default is what makes the verdict transferable: a
    metric that has settled between half and three-quarters resolution has
    settled by the time the production grid runs, and one still moving there
    will still be moving on it.
    """
    nx = case_resolution(case)
    if nx is None:
        return DEFAULT_RESOLUTIONS[:2]

    chosen: list[int] = []
    for fraction in fractions:
        n = max(MIN_DIAGNOSTIC_N, int(round(fraction * nx)))
        if n not in chosen:
            chosen.append(n)
    if len(chosen) < 2:
        # A case whose default is small enough that the fractions collide.
        # Refining upward from the floor still brackets a trend.
        chosen = [MIN_DIAGNOSTIC_N, MIN_DIAGNOSTIC_N * 2]
    return tuple(sorted(chosen))


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #
def _ladder(study: GridStudy, metric: str) -> list[float]:
    """One value of ``metric`` per grid, or ``[]`` if the case never reports it.

    Every check is written against this, so a case that has no body in the flow
    simply has no blockage finding rather than a special case somewhere.
    """
    values = [r.metrics.get(metric) for r in study.results]
    if not values or any(v is None for v in values):
        return []
    return [float(v) for v in values]


def _check_continuity(study: GridStudy) -> list[Finding]:
    """Whether the projection actually drove the divergence out."""
    ladder = _ladder(study, "max_divergence")
    if not ladder:
        return []

    worst = max(ladder)
    detail = ("max|div u| = " + ", ".join(f"{v:.1e}" for v in ladder)
              + " across the grids")
    if worst > DIVERGENCE_FAIL:
        return [Finding("continuity", FAIL, detail, advice=(
            "the pressure solve is not enforcing incompressibility; rerun with "
            "--pressure-solver direct to separate a failing iterative solve "
            "from a genuinely ill-posed configuration"
        ))]
    if worst > DIVERGENCE_WARN:
        return [Finding("continuity", WARN, detail, advice=(
            f"above {DIVERGENCE_WARN:.0e} the Poisson solve is stopping short; "
            f"lower --cfl or switch --pressure-solver"
        ))]
    return [Finding("continuity", OK, detail)]


def _check_convergence(study: GridStudy) -> list[Finding]:
    """Whether each refined metric did what refinement should do to it.

    This is the check the whole pass exists for.  A drag coefficient that moves
    5% between grids is not a drag coefficient with a 5% error bar; it is a
    number that has not converged, and the difference matters because the first
    reading invites you to quote it and the second does not.

    The verdict is the case's own -- see
    :meth:`~pycfd.cases.GridStudy.converging` -- because a measured quantity and
    an error want opposite answers to "did it move?".
    """
    findings = []
    fine = study.resolutions[-1]
    coarse = study.resolutions[-2] if len(study.resolutions) > 1 else None
    for metric in study.values:
        findings.append(_one_metric(study, metric, coarse, fine))
    return findings


def _one_metric(study: GridStudy, metric: str,
                coarse: int | None, fine: int) -> Finding:
    """The verdict on a single refined metric."""
    name = f"convergence of {metric}"
    changes = study.changes(metric)
    moved = changes[-1] if changes else float("nan")
    healthy = study.converging(metric)
    estimate = study.richardson(metric)
    is_error = study.kinds.get(metric) == "error"

    if coarse is None or not math.isfinite(moved):
        detail = "not enough grids to say whether it moved"
    else:
        detail = (f"{'fell' if is_error else 'moved'} {moved * 100:.2f}% "
                  f"between N={coarse} and N={fine}")
        if is_error:
            order = study.observed_order(metric)
            if math.isfinite(order):
                detail += f", an observed order of {order:.2f}"

    # A sequence whose successive differences are not shrinking is not on its
    # way anywhere, and that is worth saying even when the last step happened
    # to be small enough to look settled.
    if estimate is not None and estimate.regime == "diverging":
        return Finding(name, WARN,
                       detail + f"; R = {estimate.convergence_ratio:+.3f}", advice=(
            "refining is not reducing the change, so the sequence is not "
            "converging: the grids are still too coarse to be in the asymptotic "
            "regime, or something other than resolution is setting this number"
        ))

    if healthy:
        if estimate is not None and estimate.trustworthy:
            detail += (f"; extrapolates to {estimate.extrapolated:.6g} "
                       f"(GCI {estimate.gci * 100:.2f}%)")
        return Finding(name, OK, detail)

    if is_error:
        return Finding(name, WARN, detail, advice=(
            "the error is not falling under refinement, so the discretisation "
            "is not what limits this run; look for a boundary condition, a time "
            "step or a model that is setting the error instead"
        ))

    advice = (f"still moving at more than {SETTLED_TOLERANCE * 100:g}%; refine "
              f"past N={fine} before quoting it")
    if estimate is not None and estimate.trustworthy:
        advice = (f"extrapolates to {estimate.extrapolated:.6g}, "
                  f"{estimate.relative_error * 100:.1f}% beyond the finest "
                  f"grid's {study.values[metric][-1]:.6g}")
    elif len(study.resolutions) < 3:
        advice += "; a third grid would say where it is heading"
    return Finding(name, WARN, detail, advice)


def _check_blockage(study: GridStudy) -> list[Finding]:
    """Whether the domain walls are close enough to change the answer."""
    ladder = _ladder(study, "blockage_ratio")
    if not ladder:
        return []

    ratio = ladder[-1]
    span = (_ladder(study, "characteristic_length") or [1.0])[-1]
    detail = f"the body spans {ratio * 100:.1f}% of the cross-stream extent"
    advice = (f"--domain-height {span / BLOCKAGE_NOTICEABLE:.6g} would bring it "
              f"under {BLOCKAGE_NOTICEABLE * 100:g}%")

    if ratio >= BLOCKAGE_SUBSTANTIAL:
        return [Finding("blockage", WARN, detail, advice=(
            f"above {BLOCKAGE_SUBSTANTIAL * 100:g}% the walls accelerate the "
            f"flow past the body and bias Cd upward by several per cent -- more "
            f"than the spread between published unbounded values. " + advice
        ))]
    if ratio >= BLOCKAGE_NOTICEABLE:
        return [Finding("blockage", WARN, detail, advice=(
            f"past {BLOCKAGE_NOTICEABLE * 100:g}% confinement starts to show in "
            f"Cd; " + advice
        ))]
    return [Finding("blockage", OK, detail)]


def _check_body_resolution(study: GridStudy) -> list[Finding]:
    """Whether the grid resolves the body well enough to measure a force on it."""
    ladder = _ladder(study, "cells_across_body")
    if not ladder:
        return []

    finest = ladder[-1]
    detail = ("the body spans "
              + " -> ".join(f"{v:.1f}" for v in ladder)
              + " cells across the study's grids")
    if finest <= 0:
        return [Finding("body resolution", FAIL, detail, advice=(
            "the body does not cover a single cell; scale the geometry up or "
            "refine the grid"
        ))]

    # Cells across the body scale with the grid, so the requirement converts
    # straight into a resolution rather than an abstract "refine more".
    needed = int(math.ceil(study.resolutions[-1] * CELLS_QUANTITATIVE / finest))
    if finest < CELLS_MINIMUM:
        return [Finding("body resolution", FAIL, detail, advice=(
            f"below {CELLS_MINIMUM:g} cells the surface is a handful of pixels "
            f"and its drag is an artefact of which ones were flagged solid; "
            f"about N={needed} is needed for a quantitative force"
        ))]
    if finest < CELLS_QUANTITATIVE:
        return [Finding("body resolution", WARN, detail, advice=(
            f"a staircase immersed boundary needs about {CELLS_QUANTITATIVE:g} "
            f"cells before forces are quantitative -- roughly N={needed} here"
        ))]
    return [Finding("body resolution", OK, detail)]


def _check_compressibility(study: GridStudy) -> list[Finding]:
    """Whether the flight conditions given are ones this solver can represent."""
    ladder = _ladder(study, "mach")
    if not ladder:
        return []

    from ..units import INCOMPRESSIBLE_MACH_LIMIT

    mach = ladder[-1]
    detail = f"M = {mach:.3f} at the stated conditions"
    if mach > INCOMPRESSIBLE_MACH_LIMIT:
        return [Finding("compressibility", WARN, detail, advice=(
            f"this solver has no density equation at all, so above "
            f"M = {INCOMPRESSIBLE_MACH_LIMIT:g} the result approximates a "
            f"different flow rather than the same one slightly less well"
        ))]
    return [Finding("compressibility", OK, detail)]


def _check_case_validation(study: GridStudy) -> list[Finding]:
    """Surface the case's own pass/fail checks, run on the finest grid.

    The case is the authority on what its numbers should look like -- whether
    the average was stationary, whether a wake shed, how far Cd may sit from the
    literature.  Restating those judgements here would create a second place to
    maintain them, so this reports rather than re-derives.
    """
    if not study.results:
        return []
    result = study.results[-1]
    if not result.checks:
        return []

    fine = study.resolutions[-1]
    failed = [(label, detail) for label, ok, detail in result.checks if not ok]
    passed = len(result.checks) - len(failed)
    if not failed:
        return [Finding("case validation", OK,
                        f"all {passed} of the case's own checks passed at N={fine}")]

    detail = (f"{passed} of {len(result.checks)} of the case's own checks "
              f"passed at N={fine}; " + "; ".join(f"{lbl}: {d}" for lbl, d in failed))
    return [Finding("case validation", WARN, detail, advice=(
        f"N={fine} is a diagnostic grid, below the one this case runs at, so a "
        f"check that only just failed here may pass at full resolution -- but a "
        f"non-stationary average or an undetected wake will not fix itself"
    ))]


#: Every question the pass asks, in the order they are computed.  A check that
#: finds the case does not report what it needs returns nothing at all, which is
#: what lets one list serve a closed cavity and an aircraft silhouette alike.
CHECKS = (
    _check_continuity,
    _check_convergence,
    _check_blockage,
    _check_body_resolution,
    _check_compressibility,
    _check_case_validation,
)


def diagnose(case: str, resolutions=None, progress: bool = False,
             **kwargs) -> Diagnosis:
    """Run ``case`` on a coarse/medium pair and report what is worth knowing.

    Parameters
    ----------
    case:
        Case name, as understood by :func:`~pycfd.cases.load_case`.
    resolutions:
        Cell counts, coarsest first.  Defaults to
        :func:`diagnostic_resolutions`; three or more enable Richardson
        extrapolation on every refined metric.
    **kwargs:
        Passed to the case's ``run``, so the diagnosis is of the configuration
        actually intended -- same geometry, same Reynolds number, same domain.

    Returns
    -------
    Diagnosis
        Always returned.  A solver that blew up comes back as a ``fail``
        finding, because "it diverged on the coarse grid" is an answer to the
        question that was asked.
    """
    grids = (diagnostic_resolutions(case) if not resolutions
             else tuple(int(n) for n in resolutions))
    production = case_resolution(case)
    log.info("diagnosing '%s' on grids %s", case,
             ", ".join(str(n) for n in grids))

    try:
        study = grid_study(case, grids, progress=progress, **kwargs)
    except DivergenceError as exc:
        return Diagnosis(case, list(grids), production=production, findings=[
            Finding("stability", FAIL, str(exc), advice=(
                "a run that blows up has no diagnosis beyond this one; lower "
                "--cfl, shorten --dt, or check that the boundary conditions "
                "leave the flow somewhere to go"
            )),
        ])

    findings: list[Finding] = []
    for check in CHECKS:
        findings.extend(check(study))
    for finding in findings:
        logger = {OK: log.info, WARN: log.warning, FAIL: log.error}[finding.level]
        logger("%s: %s", finding.name, finding.detail)

    return Diagnosis(case, list(grids), findings, study, production)
