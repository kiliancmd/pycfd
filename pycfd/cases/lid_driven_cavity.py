"""Lid-driven cavity -- the standard incompressible benchmark.

A square box with no-slip walls and a lid sliding at constant speed.  The flow
is steady for the Reynolds numbers considered here, and the centreline velocity
profiles are tabulated by Ghia, Ghia & Shin (1982), which makes it the usual
first check on any new incompressible solver.

Run with::

    python main.py --case cavity --re 400
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..analysis.postprocess import centerline_profiles
from ..analysis.validation import (
    GHIA_REYNOLDS,
    ghia_reference,
    l2_error,
    linf_error,
)
from ..config import BCKind, BCSpec, SimulationConfig
from ..physics.incompressible import Simulation

log = logging.getLogger(__name__)

#: Lid speed; also the reference velocity for the Reynolds number.
LID_SPEED = 1.0

#: Rate-of-change threshold below which the flow is declared steady.
STEADY_TOLERANCE = 1.0e-6

#: Agreement with Ghia et al. that counts as a pass, as a fraction of the lid
#: speed.  Ghia's own data come from a 129x129 grid, so sub-1% agreement is
#: about the most a uniform-grid second-order code can be asked for.
GHIA_L2_TOLERANCE = 0.02


def default_end_time(re: float) -> float:
    """Integration time that comfortably reaches steady state at this Reynolds number.

    Higher Re means a weaker viscous return to equilibrium, so the transient
    lasts longer.  The steady-state detector normally stops the run well before
    this bound.
    """
    return float(max(25.0, 0.1 * re + 15.0))


def build(re: float = 100.0, nx: int = 128, ny: int = 128,
          t_end: float | None = None, dt: float = 0.01, cfl_max: float = 0.4,
          **overrides) -> Simulation:
    """Construct the cavity simulation without running it."""
    # A benchmark has to pin the physics it validates.  ``use_les`` is a package
    # default that legitimately changes with whatever case is being worked on,
    # and an eddy-viscosity model silently switched on turns an exact laminar
    # solution into something else -- it cost this benchmark its second-order
    # convergence once already.  ``setdefault`` keeps ``--les`` working.
    overrides.setdefault("use_les", False)
    overrides.setdefault("name", f"cavity_Re{re:g}")

    cfg = SimulationConfig(
        nx=nx, ny=ny, lx=1.0, ly=1.0,
        re=re, u_ref=LID_SPEED, l_ref=1.0,
        dt=dt, t_end=default_end_time(re) if t_end is None else t_end,
        cfl_max=cfl_max, steady_tol=STEADY_TOLERANCE,
        boundary_config={
            "left": BCSpec(BCKind.NO_SLIP),
            "right": BCSpec(BCKind.NO_SLIP),
            "bottom": BCSpec(BCKind.NO_SLIP),
            "top": BCSpec(BCKind.MOVING_WALL, velocity=LID_SPEED),
        },
        **overrides,
    )
    return Simulation(cfg)


def validate(sim: Simulation) -> tuple[dict, list]:
    """Compare centreline profiles with Ghia et al. where reference data exist."""
    y, u_line, x, v_line = centerline_profiles(sim.fields)
    metrics: dict[str, float] = {}
    checks: list[tuple[str, bool, str]] = []

    re = int(round(sim.config.re))
    if re not in GHIA_REYNOLDS:
        checks.append((
            "Ghia comparison", True,
            f"skipped -- no published data at Re={re} (have {list(GHIA_REYNOLDS)})",
        ))
        return metrics, checks

    ref = ghia_reference(re)
    u_at = np.interp(ref["y"], y, u_line)
    v_at = np.interp(ref["x"], x, v_line)

    metrics["ghia_u_L2"] = l2_error(u_at, ref["u"])
    metrics["ghia_u_Linf"] = linf_error(u_at, ref["u"])
    metrics["ghia_v_L2"] = l2_error(v_at, ref["v"])
    metrics["ghia_v_Linf"] = linf_error(v_at, ref["v"])

    worst = max(metrics["ghia_u_L2"], metrics["ghia_v_L2"])
    checks.append((
        f"centreline profiles vs Ghia et al. Re={re}",
        worst < GHIA_L2_TOLERANCE,
        f"max L2 = {worst:.4f} (tolerance {GHIA_L2_TOLERANCE})",
    ))
    return metrics, checks


def run(re: float = 100.0, nx: int = 128, ny: int = 128, t_end: float | None = None,
        dt: float = 0.01, outdir: str | Path = "results/cavity",
        make_plots: bool = True, progress: bool = False, **overrides):
    """Run the cavity, validate it and write the figures."""
    from . import CaseResult

    sim = build(re=re, nx=nx, ny=ny, t_end=t_end, dt=dt, **overrides)
    result = sim.run(progress=progress)

    metrics, checks = validate(sim)
    metrics.update({
        "steps": result.steps,
        "final_time": result.time,
        "wall_time_s": result.wall_time,
        "reached_steady_state": float(result.converged),
        "max_divergence": sim.solver.max_divergence(sim.fields),
    })

    outputs: list[Path] = []
    if make_plots:
        from ..visualization import static_plot as sp

        outdir = Path(outdir)
        tag = f"Re{re:g}_{nx}x{ny}"
        p1 = outdir / f"cavity_{tag}_fields.png"
        sp.four_panel_figure(
            sim.fields, title=f"Lid-driven cavity, Re = {re:g}, {nx}x{ny}", path=p1,
        )
        ref = ghia_reference(int(round(re))) if int(round(re)) in GHIA_REYNOLDS else None
        p2 = outdir / f"cavity_{tag}_centerlines.png"
        sp.centerline_comparison_figure(
            sim.fields, reference=ref,
            title=f"Cavity centreline profiles, Re = {re:g}, {nx}x{ny}", path=p2,
        )
        outputs += [p1, p2]

    return CaseResult(f"Lid-driven cavity (Re={re:g}, {nx}x{ny})", sim,
                      metrics, outputs, checks)
