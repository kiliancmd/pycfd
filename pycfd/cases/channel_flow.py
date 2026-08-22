"""Plane Poiseuille flow between parallel plates.

Two configurations are provided.

``periodic`` (default)
    Streamwise-periodic channel driven by a constant body force.  The steady
    state is *exactly* the analytical parabola, so the L2 error measures nothing
    but discretisation error -- which makes this the clean quantitative check
    the benchmark is meant to be.

``developing``
    Uniform inflow at the left, zero-gradient outflow at the right.  This
    exercises the inlet/outlet boundary machinery and shows the entrance length
    over which the profile develops; the outlet profile is then compared with the
    parabola.

Run with::

    python main.py --case channel --re 10
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..analysis.validation import l2_error, linf_error, poiseuille_profile
from ..config import BCKind, BCSpec, SimulationConfig
from ..physics.incompressible import Simulation

log = logging.getLogger(__name__)

#: Target centreline speed; also the reference velocity for the Reynolds number.
U_MAX = 1.0

#: Channel height, and the reference length for the Reynolds number.
CHANNEL_HEIGHT = 1.0

#: Streamwise extent, in channel heights.
CHANNEL_LENGTH = 2.0

STEADY_TOLERANCE = 1.0e-9

#: What a grid study on this case should watch: the shape error against the
#: analytical Poiseuille parabola, which is a genuine discretisation error and
#: converges at order 2.
#:
#: ``centerline_error_pct`` is deliberately *not* here, although it looks like
#: the obvious second entry.  It measures the peak against a fixed target, and
#: what actually sets it is :data:`STEADY_TOLERANCE` -- the point at which the
#: time integration is allowed to stop -- rather than the grid.  Measured, at
#: nx = 16: ``steady_tol`` 1e-7 / 1e-9 / 1e-11 gives 1.01e-5 / 1.01e-7 / 1.01e-9
#: per cent, exactly proportional, while doubling the grid at fixed tolerance
#: moves it from 1.009e-7 to 1.013e-7 -- not at all.  Refinement has nothing to
#: act on, so a study that tracked it would report the channel as failing to
#: converge for a reason that has nothing to do with convergence.  The number is
#: still measured, reported and checked below; it just is not a grid metric.
CONVERGENCE_METRICS = {
    "profile_L2_relative": "error",
}

#: Relative L2 agreement with the analytical parabola that counts as a pass.
POISEUILLE_TOLERANCE = 0.02


def driving_force(nu: float, height: float = CHANNEL_HEIGHT,
                  u_max: float = U_MAX) -> float:
    """Body force per unit mass that produces the requested centreline speed.

    Inverting ``u_max = f h^2 / (8 nu)`` for the steady balance
    ``nu d2u/dy2 + f = 0`` with no slip at both walls.
    """
    return 8.0 * nu * u_max / height ** 2


def build(re: float = 10.0, nx: int = 32, ny: int = 64, t_end: float | None = None,
          dt: float = 0.01, cfl_max: float = 0.4, mode: str = "periodic",
          outlet_type: str | None = None, p_ref: float | None = None,
          **overrides) -> Simulation:
    """Construct the channel simulation without running it.

    ``outlet_type`` and ``p_ref`` retype the downstream boundary of the
    ``developing`` configuration.  The ``periodic`` one has no outflow boundary,
    so passing either there raises rather than doing nothing.
    """
    from . import override_outlet
    if mode not in ("periodic", "developing"):
        raise ValueError(f"mode must be 'periodic' or 'developing', got {mode!r}")

    # A benchmark has to pin the physics it validates.  ``use_les`` is a package
    # default that legitimately changes with whatever case is being worked on,
    # and an eddy-viscosity model silently switched on turns an exact laminar
    # solution into something else -- it cost this benchmark its second-order
    # convergence once already.  ``setdefault`` keeps ``--les`` working.
    overrides.setdefault("use_les", False)
    overrides.setdefault("name", f"channel_{mode}_Re{re:g}")

    nu = U_MAX * CHANNEL_HEIGHT / re
    # The transient decays on the viscous time scale h^2 / nu.
    default_t_end = 8.0 * CHANNEL_HEIGHT ** 2 / nu

    if mode == "periodic":
        bcs = {
            "left": BCSpec(BCKind.PERIODIC),
            "right": BCSpec(BCKind.PERIODIC),
            "bottom": BCSpec(BCKind.NO_SLIP),
            "top": BCSpec(BCKind.NO_SLIP),
        }
        force = (driving_force(nu), 0.0)
        length = CHANNEL_LENGTH
        u0 = 0.0
    else:
        bcs = {
            "left": BCSpec(BCKind.INLET, velocity=U_MAX, profile="uniform"),
            "right": BCSpec(BCKind.OUTLET),
            "bottom": BCSpec(BCKind.NO_SLIP),
            "top": BCSpec(BCKind.NO_SLIP),
        }
        force = (0.0, 0.0)
        # Entrance length scales as ~0.05 Re h; leave room for it plus a margin.
        length = max(6.0, 0.09 * re * CHANNEL_HEIGHT)
        u0 = U_MAX

    cfg = SimulationConfig(
        nx=nx, ny=ny, lx=length, ly=CHANNEL_HEIGHT,
        re=re, u_ref=U_MAX, l_ref=CHANNEL_HEIGHT,
        dt=dt, t_end=default_t_end if t_end is None else t_end,
        cfl_max=cfl_max, steady_tol=STEADY_TOLERANCE,
        body_force=force, boundary_config=override_outlet(bcs, outlet_type, p_ref),
        **overrides,
    )
    return Simulation(cfg, u_init=u0)


#: Peak-to-mean velocity ratio of a parabolic profile, ``u_max / u_bar = 3/2``.
PARABOLA_PEAK_OVER_MEAN = 1.5


def expected_peak_velocity(mode: str) -> float:
    """Centreline speed the fully developed profile must reach.

    The two configurations fix the flow rate in different ways, and comparing
    against the wrong one is an easy way to "discover" a 50% error that is not
    there.

    ``periodic``
        The body force sets the amplitude directly: the steady balance
        ``nu u'' + f = 0`` gives ``u_max = f h^2 / (8 nu)``, which
        :func:`driving_force` was chosen to make equal to ``U_MAX``.

    ``developing``
        A *uniform* inlet at ``U_MAX`` fixes the mass flux instead.  The
        developed profile must carry that same flux, and a parabola with mean
        ``U_MAX`` peaks at ``1.5 * U_MAX``.
    """
    return U_MAX if mode == "periodic" else PARABOLA_PEAK_OVER_MEAN * U_MAX


def validate(sim: Simulation, mode: str = "periodic"):
    """Compare a fully developed station with the analytical parabola."""
    uc, _ = sim.fields.cell_velocities()
    mesh = sim.mesh

    # Sample where the flow is fully developed: anywhere for the periodic case,
    # at the last station for the developing one.
    station = mesh.nx // 2 if mode == "periodic" else mesh.nx - 1
    profile = uc[station, :]

    # Shape check: the parabola carrying the *same* mass flux as the computed
    # profile.  This isolates the shape from the amplitude, which is checked
    # separately against the theory appropriate to each configuration.
    flux_peak = PARABOLA_PEAK_OVER_MEAN * float(profile.mean())
    exact = poiseuille_profile(mesh.yc, CHANNEL_HEIGHT, flux_peak)

    expected_peak = expected_peak_velocity(mode)
    metrics = {
        "u_max_numerical": float(profile.max()),
        "u_max_analytical": expected_peak,
        "mean_velocity": float(profile.mean()),
        "centerline_error_pct": float(
            abs(profile.max() - expected_peak) / expected_peak * 100.0
        ),
        "profile_L2": l2_error(profile, exact),
        "profile_L2_relative": l2_error(profile, exact, relative=True),
        "profile_Linf": linf_error(profile, exact),
    }
    checks = [(
        "centreline velocity vs Poiseuille",
        metrics["centerline_error_pct"] < POISEUILLE_TOLERANCE * 100,
        f"{metrics['centerline_error_pct']:.4f}% error vs the expected peak "
        f"{expected_peak:g} (tolerance {POISEUILLE_TOLERANCE * 100:g}%)",
    ), (
        "profile shape vs Poiseuille",
        metrics["profile_L2_relative"] < POISEUILLE_TOLERANCE,
        f"relative L2 = {metrics['profile_L2_relative']:.5f} against the parabola "
        "of equal mass flux",
    )]
    return metrics, checks, profile, exact


def run(re: float = 10.0, nx: int = 32, ny: int = 64, t_end: float | None = None,
        dt: float = 0.01, mode: str = "periodic",
        outdir: str | Path = "results/channel", make_plots: bool = True,
        progress: bool = False, outlet_type: str | None = None,
        p_ref: float | None = None, **overrides):
    """Run the channel, validate it and write the figures."""
    from . import CaseResult

    sim = build(re=re, nx=nx, ny=ny, t_end=t_end, dt=dt, mode=mode,
                outlet_type=outlet_type, p_ref=p_ref, **overrides)
    result = sim.run(progress=progress)
    metrics, checks, profile, exact = validate(sim, mode)
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
        tag = f"{mode}_Re{re:g}_{nx}x{ny}"
        p1 = outdir / f"channel_{tag}_profile.png"
        sp.profile_comparison_figure(
            sim.mesh.yc, profile, exact, xlabel="y", ylabel="u",
            title=f"Poiseuille profile, Re = {re:g} ({mode}), {nx}x{ny}",
            analytical_label="analytical parabola", path=p1,
        )
        p2 = outdir / f"channel_{tag}_fields.png"
        sp.four_panel_figure(
            sim.fields, title=f"Channel flow, Re = {re:g} ({mode})", path=p2,
        )
        outputs += [p1, p2]

    return CaseResult(f"Poiseuille channel ({mode}, Re={re:g}, {nx}x{ny})",
                      sim, metrics, outputs, checks)
