"""Taylor--Green vortex: the quantitative accuracy benchmark.

This is an *exact* unsteady solution of the full nonlinear incompressible
Navier--Stokes equations on a doubly periodic domain, so the numerical error can
be measured directly rather than inferred from grid refinement alone.

Why this case is the right one for measuring spatial order
----------------------------------------------------------
On a domain with walls, the fractional-step method commits a splitting error
concentrated in a thin pressure boundary layer, which pollutes any measured
order of accuracy.  A periodic domain has no such layer: the discrete projection
is exact, so what remains is purely the spatial truncation error of the
convective and viscous stencils.

The refinement study holds the Courant number fixed, so the time step shrinks in
proportion to the mesh.  With the third-order SSP-RK3 integrator the temporal
error is then O(h^3) and the second-order spatial error dominates cleanly.

Run with::

    python main.py --case taylor_green --nx 64
    python main.py --convergence
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..analysis.validation import ConvergenceStudy, convergence_order, taylor_green
from ..config import BCKind, BCSpec, SimulationConfig
from ..physics.incompressible import Simulation

log = logging.getLogger(__name__)

#: The classic Taylor--Green domain.
DOMAIN = 2.0 * np.pi

#: Default kinematic viscosity: large enough to decay visibly, small enough that
#: the nonlinear terms still matter.
DEFAULT_NU = 0.05

#: Default stop time, about one viscous decay e-folding at DEFAULT_NU.
DEFAULT_T_END = 0.5

#: Courant number held fixed across the refinement study.
STUDY_COURANT = 0.4

#: Order that counts as a pass for the second-order scheme.
MIN_OBSERVED_ORDER = 1.8

PERIODIC_BCS = {w: BCSpec(BCKind.PERIODIC) for w in ("left", "right", "bottom", "top")}


def exact_fields(mesh, t: float, nu: float):
    """Analytical ``(u, v)`` sampled at the staggered face locations."""
    Xu, Yu = mesh.u_grid()
    Xv, Yv = mesh.v_grid()
    return (taylor_green(Xu, Yu, t, nu, "u"), taylor_green(Xv, Yv, t, nu, "v"))


def resolve_viscosity(nu: float | None = None, re: float | None = None) -> float:
    """Viscosity from either spelling.

    The case is naturally parameterised by ``nu``, but the CLI speaks Reynolds
    numbers.  With ``u_ref = l_ref = 1`` the two are reciprocals; ``re`` wins if
    both are supplied.
    """
    if re is not None:
        if re <= 0:
            raise ValueError(f"Reynolds number must be positive, got {re}")
        return 1.0 / re
    return DEFAULT_NU if nu is None else nu


def build(nx: int = 64, ny: int | None = None, nu: float | None = None,
          re: float | None = None, t_end: float = DEFAULT_T_END,
          courant: float = STUDY_COURANT, dt: float | None = None,
          **overrides) -> Simulation:
    """Construct the vortex simulation, initialised with the exact solution.

    ``dt`` defaults to holding the Courant number fixed at ``courant``, which is
    what the refinement study needs; pass it explicitly to override.
    """
    ny = nx if ny is None else ny
    nu = resolve_viscosity(nu, re)
    # A benchmark has to pin the physics it validates.  ``use_les`` is a package
    # default that legitimately changes with whatever case is being worked on,
    # and an eddy-viscosity model silently switched on turns an exact laminar
    # solution into something else -- it cost this benchmark its second-order
    # convergence once already.  ``setdefault`` keeps ``--les`` working.
    overrides.setdefault("use_les", False)
    overrides.setdefault("name", f"taylor_green_{nx}")

    cfg = SimulationConfig(
        nx=nx, ny=ny, lx=DOMAIN, ly=DOMAIN,
        re=1.0 / nu, u_ref=1.0, l_ref=1.0,
        dt=courant * DOMAIN / nx if dt is None else dt,
        t_end=t_end, adaptive_dt=False,
        boundary_config=dict(PERIODIC_BCS),
        **overrides,
    )
    from ..core.mesh import StructuredMesh

    u0, v0 = exact_fields(StructuredMesh.from_config(cfg), 0.0, nu)
    return Simulation(cfg, u_init=u0, v_init=v0)


def measure_error(sim: Simulation, nu: float) -> dict[str, float]:
    """L2 and Linf velocity errors against the exact solution at the current time."""
    t = sim.fields.t
    ue, ve = exact_fields(sim.mesh, t, nu)
    # Drop the far-end faces: on a periodic axis they duplicate the near-end ones
    # and would be counted twice.
    du = sim.fields.u_phys[:-1, :] - ue[:-1, :]
    dv = sim.fields.v_phys[:, :-1] - ve[:, :-1]
    n = du.size + dv.size
    return {
        "L2": float(np.sqrt((np.sum(du ** 2) + np.sum(dv ** 2)) / n)),
        "Linf": float(max(np.abs(du).max(), np.abs(dv).max())),
        "max_divergence": sim.solver.max_divergence(sim.fields),
        "kinetic_energy_decay": float(np.exp(-4.0 * nu * t)),
    }


def convergence_study(resolutions=(16, 32, 64, 128), nu: float = DEFAULT_NU,  # noqa: D417
                      t_end: float = DEFAULT_T_END, courant: float = STUDY_COURANT,
                      progress: bool = False) -> tuple[ConvergenceStudy, ConvergenceStudy, list]:
    """Refine the grid and report the observed order in both norms."""
    l2, linf, rows = [], [], []
    for n in resolutions:
        sim = build(nx=n, nu=nu, t_end=t_end, courant=courant)
        result = sim.run(progress=progress)
        err = measure_error(sim, nu)
        l2.append(err["L2"])
        linf.append(err["Linf"])
        rows.append({"n": n, "steps": result.steps, "wall_time": result.wall_time, **err})
        log.info("N=%d: L2=%.4e Linf=%.4e maxdiv=%.2e", n, err["L2"], err["Linf"],
                 err["max_divergence"])
    return (convergence_order(resolutions, l2, "L2"),
            convergence_order(resolutions, linf, "Linf"), rows)


def run(nx: int = 64, ny: int | None = None, nu: float | None = None,
        re: float | None = None, t_end: float = DEFAULT_T_END,
        dt: float | None = None, outdir: str | Path = "results/taylor_green",
        make_plots: bool = True, progress: bool = False, **overrides):
    """Run a single Taylor--Green vortex and report its error."""
    from . import CaseResult

    nu = resolve_viscosity(nu, re)
    sim = build(nx=nx, ny=ny, nu=nu, t_end=t_end, dt=dt, **overrides)
    result = sim.run(progress=progress)
    err = measure_error(sim, nu)

    metrics = {
        **err,
        "steps": result.steps,
        "final_time": result.time,
        "wall_time_s": result.wall_time,
    }
    checks = [(
        "divergence-free after projection",
        err["max_divergence"] < 1e-9,
        f"max|div u| = {err['max_divergence']:.2e}",
    )]

    outputs: list[Path] = []
    if make_plots:
        from ..visualization import static_plot as sp

        outdir = Path(outdir)
        p1 = outdir / f"taylor_green_{nx}_fields.png"
        sp.four_panel_figure(
            sim.fields, title=f"Taylor-Green vortex, nu = {nu:g}, t = {result.time:g}",
            path=p1,
        )
        outputs.append(p1)

    return CaseResult(f"Taylor-Green vortex ({nx}x{ny or nx})", sim,
                      metrics, outputs, checks)


def run_convergence(resolutions=(16, 32, 64, 128), nu: float | None = None,
                    re: float | None = None, t_end: float = DEFAULT_T_END,
                    outdir: str | Path = "results/convergence",
                    make_plots: bool = True, progress: bool = False):
    """Full grid-convergence study with a table, a figure and a pass/fail check."""
    from . import CaseResult

    nu = resolve_viscosity(nu, re)
    study_l2, study_linf, rows = convergence_study(resolutions, nu, t_end,
                                                   progress=progress)
    metrics = {
        "observed_order_L2": study_l2.observed_order,
        "observed_order_Linf": study_linf.observed_order,
        "finest_L2": study_l2.errors[-1],
        "finest_Linf": study_linf.errors[-1],
        "max_divergence": max(r["max_divergence"] for r in rows),
    }
    checks = [(
        "second-order spatial convergence",
        study_l2.observed_order >= MIN_OBSERVED_ORDER,
        f"observed L2 order = {study_l2.observed_order:.3f} "
        f"(threshold {MIN_OBSERVED_ORDER})",
    )]

    outputs: list[Path] = []
    if make_plots:
        from ..visualization import static_plot as sp

        outdir = Path(outdir)
        p1 = outdir / "taylor_green_convergence.png"
        sp.convergence_figure(
            study_l2,
            title=f"Taylor-Green grid convergence: observed order "
                  f"{study_l2.observed_order:.2f}",
            path=p1,
        )
        outputs.append(p1)

    result = CaseResult("Taylor-Green convergence study",
                        build(nx=resolutions[-1], nu=nu, t_end=t_end),
                        metrics, outputs, checks)
    log.info("L2 convergence\n%s", study_l2.table())
    log.info("Linf convergence\n%s", study_linf.table())
    result.tables = {"L2": study_l2, "Linf": study_linf}   # type: ignore[attr-defined]
    return result
