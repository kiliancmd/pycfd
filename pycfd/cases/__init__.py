"""Benchmark cases.

Each case module exposes ``build(...) -> Simulation`` and ``run(...) -> CaseResult``
so the CLI can treat them uniformly.  ``build`` is separated out because the live
viewer needs an un-run simulation to animate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..config import BCKind, BCSpec
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
