"""Projection solver: discrete conservation, benchmark physics, failure modes.

The central invariant is that the projected velocity is divergence-free to
*machine* precision, not to truncation error.  That holds because the pressure
Poisson operator is assembled over exactly the face set the projection corrects,
and it is the property that most distinguishes a correct staggered projection
solver from a plausible-looking one.
"""

import numpy as np
import pytest

from pycfd.config import (
    AdvectionScheme,
    BCKind,
    BCSpec,
    PressureSolver,
    SimulationConfig,
    SolverType,
    TimeScheme,
)
from pycfd.analysis.postprocess import centerline_profiles, kinetic_energy
from pycfd.analysis.validation import poiseuille_profile
from pycfd.core.mesh import NonUniformMeshError
from pycfd.core.solver import ProjectionSolver
from pycfd.core.timestepper import DivergenceError, TimeStepper
from pycfd.geometry.obstacles import circle_mask

from .conftest import (
    make_config,
    DIVERGENCE_TOL,
    cavity_config,
    periodic_walls,
    taylor_green_fields,
    walls,
)


# --------------------------------------------------------------------------- #
# Discrete incompressibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bcs,label", [
    (walls(top=(BCKind.MOVING_WALL, 1.0)), "cavity"),
    (periodic_walls(), "doubly periodic"),
    (walls(left=BCKind.PERIODIC, right=BCKind.PERIODIC), "channel"),
    (walls(left=BCSpec(BCKind.INLET, velocity=1.0), right=BCKind.OUTLET), "inlet/outlet"),
    (walls(top=BCKind.SYMMETRY, bottom=BCKind.SYMMETRY,
           left=BCSpec(BCKind.INLET, velocity=1.0), right=BCKind.OUTLET), "symmetry"),
])
def test_velocity_is_divergence_free_after_every_step(bcs, label):
    cfg = make_config(nx=16, ny=16, re=50, dt=1e-3, boundary_config=bcs)
    solver = ProjectionSolver(cfg)
    # A noisy, deliberately non-solenoidal initial condition: initialize() must
    # project it, and every subsequent step must keep it divergence-free.
    rng = np.random.default_rng(0)
    fields = solver.initialize(
        u_init=0.1 * rng.standard_normal((cfg.nx + 1, cfg.ny)),
        v_init=0.1 * rng.standard_normal((cfg.nx, cfg.ny + 1)),
    )
    assert solver.max_divergence(fields) < DIVERGENCE_TOL, f"{label} (initial)"

    for _ in range(6):
        fields = solver.step(fields, 1e-3)
        assert solver.max_divergence(fields) < DIVERGENCE_TOL, label


def test_divergence_free_with_an_obstacle():
    cfg = make_config(
        nx=48, ny=32, lx=3.0, ly=2.0, re=50, dt=2e-3,
        boundary_config=walls(left=BCSpec(BCKind.INLET, velocity=1.0),
                              right=BCKind.OUTLET,
                              top=BCKind.SYMMETRY, bottom=BCKind.SYMMETRY),
    )
    obstacle = circle_mask(ProjectionSolver(cfg).mesh, (1.0, 1.0), 0.25)
    solver = ProjectionSolver(cfg, obstacle=obstacle.mask)
    fields = solver.initialize(1.0)
    for _ in range(10):
        fields = solver.step(fields, 2e-3)
    assert solver.max_divergence(fields) < DIVERGENCE_TOL


def test_obstacle_faces_hold_exactly_zero_velocity():
    cfg = make_config(nx=40, ny=40, lx=2.0, ly=2.0, re=50, dt=2e-3,
                           boundary_config=walls(top=(BCKind.MOVING_WALL, 1.0)))
    obstacle = circle_mask(ProjectionSolver(cfg).mesh, (1.0, 1.0), 0.3)
    solver = ProjectionSolver(cfg, obstacle=obstacle.mask)
    fields = solver.initialize()
    for _ in range(8):
        fields = solver.step(fields, 2e-3)

    assert np.all(fields.u[solver.u_upd][solver.u_face_solid] == 0.0)
    assert np.all(fields.v[solver.v_upd][solver.v_face_solid] == 0.0)


@pytest.mark.parametrize("scheme", list(TimeScheme))
def test_every_time_scheme_stays_divergence_free(scheme):
    cfg = cavity_config(nx=16, ny=16, time_scheme=scheme)
    solver = ProjectionSolver(cfg)
    fields = solver.initialize()
    for _ in range(5):
        fields = solver.step(fields, 1e-3)
    assert solver.max_divergence(fields) < DIVERGENCE_TOL
    assert fields.is_finite()


@pytest.mark.parametrize("kind", list(PressureSolver))
def test_every_pressure_solver_gives_the_same_step(kind):
    """Swapping the linear solver must not change the physics."""
    reference = None
    for chosen in (PressureSolver.DIRECT, kind):
        cfg = cavity_config(nx=16, ny=16, pressure_solver=chosen,
                            poisson_tol=1e-13, poisson_maxiter=200_000)
        solver = ProjectionSolver(cfg)
        fields = solver.initialize()
        for _ in range(3):
            fields = solver.step(fields, 1e-3)
        if reference is None:
            reference = fields.u.copy()
    assert np.abs(fields.u - reference).max() < 1e-9


# --------------------------------------------------------------------------- #
# Exactly preserved states
# --------------------------------------------------------------------------- #
def test_quiescent_flow_stays_quiescent():
    """With no forcing and no moving wall, nothing may appear from nowhere."""
    cfg = make_config(nx=16, ny=16, re=100, dt=1e-3, boundary_config=walls())
    solver = ProjectionSolver(cfg)
    fields = solver.initialize()
    for _ in range(20):
        fields = solver.step(fields, 1e-3)
    assert np.abs(fields.u).max() == 0.0
    assert np.abs(fields.v).max() == 0.0


def test_uniform_periodic_flow_is_preserved_exactly():
    """Uniform flow is an exact solution; advection and diffusion must both vanish."""
    cfg = make_config(nx=16, ny=16, re=100, dt=1e-3,
                           boundary_config=periodic_walls())
    solver = ProjectionSolver(cfg)
    fields = solver.initialize(u_init=2.0, v_init=-1.0)
    for _ in range(10):
        fields = solver.step(fields, 1e-3)
    assert np.abs(fields.u_phys - 2.0).max() < 1e-13
    assert np.abs(fields.v_phys + 1.0).max() < 1e-13


def test_pressure_gradient_not_pressure_level_drives_the_flow():
    """The Poisson solution is fixed only up to a constant; it must be removed."""
    cfg = cavity_config(nx=16, ny=16)
    solver = ProjectionSolver(cfg)
    fields = solver.step(solver.initialize(), 1e-3)
    assert abs(fields.p_phys.mean()) < 1e-12


# --------------------------------------------------------------------------- #
# Accuracy against exact solutions
# --------------------------------------------------------------------------- #
def test_taylor_green_is_second_order_in_space():
    nu, t_end = 0.05, 0.3
    errors = []
    for n in (16, 32):
        cfg = make_config(
            nx=n, ny=n, lx=2 * np.pi, ly=2 * np.pi, re=1.0 / nu,
            dt=0.4 * 2 * np.pi / n, t_end=t_end, adaptive_dt=False,
            boundary_config=periodic_walls(),
        )
        solver = ProjectionSolver(cfg)
        u0, v0 = taylor_green_fields(solver.mesh, 0.0, nu)
        fields = solver.initialize(u0, v0)
        result = TimeStepper(solver).run(fields, t_end=t_end)

        ue, ve = taylor_green_fields(solver.mesh, result.time, nu)
        du = result.fields.u_phys[:-1] - ue[:-1]
        dv = result.fields.v_phys[:, :-1] - ve[:, :-1]
        errors.append(np.sqrt((np.sum(du ** 2) + np.sum(dv ** 2)) / (du.size + dv.size)))

    assert np.log2(errors[0] / errors[1]) > 1.9


def test_taylor_green_energy_decays_at_the_analytical_rate():
    """Kinetic energy must follow exp(-4 nu t) -- a check on the viscous term."""
    nu, t_end = 0.1, 0.4
    n = 48
    cfg = make_config(
        nx=n, ny=n, lx=2 * np.pi, ly=2 * np.pi, re=1.0 / nu,
        dt=0.2 * 2 * np.pi / n, t_end=t_end, adaptive_dt=False,
        boundary_config=periodic_walls(),
    )
    solver = ProjectionSolver(cfg)
    u0, v0 = taylor_green_fields(solver.mesh, 0.0, nu)
    fields = solver.initialize(u0, v0)
    e0 = kinetic_energy(fields)
    result = TimeStepper(solver).run(fields, t_end=t_end)

    ratio = kinetic_energy(result.fields) / e0
    assert ratio == pytest.approx(np.exp(-4.0 * nu * result.time), rel=2e-3)


def test_channel_reproduces_the_poiseuille_parabola():
    """Gate check: centreline velocity within 2% of the analytical solution."""
    nu = 0.1
    cfg = make_config(
        nx=8, ny=48, lx=1.0, ly=1.0, re=1.0 / nu, dt=5e-3, t_end=40.0,
        cfl_max=0.4, steady_tol=1e-9, body_force=(8.0 * nu, 0.0),
        boundary_config=walls(left=BCKind.PERIODIC, right=BCKind.PERIODIC),
    )
    solver = ProjectionSolver(cfg)
    result = TimeStepper(solver).run(solver.initialize(), t_end=40.0)
    assert result.converged

    uc, _ = result.fields.cell_velocities()
    profile = uc[cfg.nx // 2, :]
    exact = poiseuille_profile(solver.mesh.yc, 1.0, 1.0)
    assert abs(profile.max() - 1.0) < 0.02                     # within 2%
    assert np.abs(profile - exact).max() < 0.01
    # The profile is symmetric about the channel centreline.
    assert np.abs(profile - profile[::-1]).max() < 1e-12


def test_cavity_develops_a_primary_vortex():
    """Gate check: a recognisable recirculation driven by the lid."""
    cfg = cavity_config(nx=48, ny=48, re=100, t_end=15.0, dt=5e-3,
                        cfl_max=0.4, steady_tol=1e-6)
    solver = ProjectionSolver(cfg)
    result = TimeStepper(solver).run(solver.initialize(), t_end=15.0)

    y, u_line, x, v_line = centerline_profiles(result.fields)
    # The lid drags fluid forward at the top and the return flow is negative below.
    assert u_line[-1] > 0.4
    assert u_line.min() < -0.15
    # The return flow sits in the lower half of the cavity.
    assert y[np.argmin(u_line)] < 0.5
    # Down-flow on the right of the box, up-flow on the left: one primary vortex.
    assert v_line[int(0.8 * len(x))] < -0.1
    assert v_line[int(0.2 * len(x))] > 0.1


# --------------------------------------------------------------------------- #
# Numerical options
# --------------------------------------------------------------------------- #
def test_upwind_blending_runs_and_stays_bounded():
    cfg = cavity_config(nx=32, ny=32, re=1000,
                        advection_scheme=AdvectionScheme.UPWIND, t_end=2.0)
    solver = ProjectionSolver(cfg)
    result = TimeStepper(solver).run(solver.initialize(), t_end=2.0)
    umax, vmax = result.fields.max_velocity()
    assert max(umax, vmax) <= 1.0 + 1e-9        # nothing exceeds the lid speed
    assert solver.max_divergence(result.fields) < DIVERGENCE_TOL


def test_upwind_blend_is_zero_for_the_central_scheme():
    solver = ProjectionSolver(cavity_config(nx=16, ny=16))
    assert solver.upwind_blend_for(solver.initialize(), 1e-3) == 0.0


def test_les_runs_and_reduces_to_the_laminar_operator_at_zero_strain():
    cfg = cavity_config(nx=24, ny=24, use_les=True, t_end=0.5)
    solver = ProjectionSolver(cfg)
    assert solver.turbulence is not None
    result = TimeStepper(solver).run(solver.initialize(), t_end=0.5)
    assert result.fields.is_finite()
    assert solver.max_divergence(result.fields) < DIVERGENCE_TOL

    # A uniform flow has zero strain rate, hence zero eddy viscosity.  This
    # needs a periodic domain: no-slip walls shear a uniform field by design.
    periodic = ProjectionSolver(
        make_config(nx=24, ny=24, re=100, use_les=True,
                         boundary_config=periodic_walls())
    )
    uniform = periodic.initialize(u_init=1.0, v_init=-0.5)
    nu_c, _ = periodic.turbulence.eddy_viscosity(uniform.u, uniform.v, periodic.nu)
    assert np.abs(nu_c - periodic.nu).max() < 1e-14


def test_variable_viscosity_matches_the_laplacian_when_viscosity_is_constant():
    """The stress-divergence form must reduce to nu*lap(u) on a solenoidal field."""
    cfg = make_config(nx=24, ny=24, lx=2 * np.pi, ly=2 * np.pi, re=20,
                           boundary_config=periodic_walls())
    solver = ProjectionSolver(cfg)
    u0, v0 = taylor_green_fields(solver.mesh, 0.0, solver.nu)
    fields = solver.initialize(u0, v0)

    du_c, dv_c = solver._diffusion(fields.u, fields.v)
    nu_c = np.full(solver.mesh.p_shape, solver.nu)
    nu_k = np.full((solver.mesh.nx + 1, solver.mesh.ny + 1), solver.nu)
    du_v, dv_v = solver._diffusion_variable(fields.u, fields.v, nu_c, nu_k)
    assert np.abs(du_c - du_v).max() < 1e-14
    assert np.abs(dv_c - dv_v).max() < 1e-14


# --------------------------------------------------------------------------- #
# Time-step control
# --------------------------------------------------------------------------- #
def test_adaptive_step_respects_both_stability_limits():
    cfg = cavity_config(nx=32, ny=32, re=10, dt=1.0)     # low Re: viscous limit binds
    solver = ProjectionSolver(cfg)
    stepper = TimeStepper(solver)
    fields = solver.initialize(u_init=2.0)

    dt = stepper.compute_dt(fields)
    dx, dy = solver.mesh.dx, solver.mesh.dy
    viscous = 0.8 / (2.0 * solver.nu * (1 / dx ** 2 + 1 / dy ** 2))
    assert dt <= viscous + 1e-15
    assert stepper.cfl_number(fields, dt) <= cfg.cfl_max + 1e-12


def test_quiescent_start_falls_back_to_the_configured_step():
    cfg = cavity_config(nx=16, ny=16, re=1e6, dt=0.01)
    stepper = TimeStepper(ProjectionSolver(cfg))
    fields = ProjectionSolver(cfg).initialize()
    assert stepper.compute_dt(fields) == pytest.approx(0.01)


def test_fixed_time_step_is_honoured():
    cfg = cavity_config(nx=16, ny=16, adaptive_dt=False, dt=2e-3)
    stepper = TimeStepper(ProjectionSolver(cfg))
    assert stepper.compute_dt(ProjectionSolver(cfg).initialize()) == 2e-3


# --------------------------------------------------------------------------- #
# Failure modes -- these must be loud
# --------------------------------------------------------------------------- #
def test_blow_up_raises_instead_of_returning_garbage():
    cfg = make_config(nx=16, ny=16, re=1e6, dt=5.0, adaptive_dt=False,
                           t_end=1e4, boundary_config=walls(top=(BCKind.MOVING_WALL, 1e3)))
    solver = ProjectionSolver(cfg)
    with pytest.raises(DivergenceError, match="diverged|non-finite"):
        TimeStepper(solver).run(solver.initialize(), t_end=1e4, max_steps=400)


def test_landing_exactly_on_t_end_is_not_mistaken_for_divergence():
    """A benign floating-point residual at the end of a run must not raise.

    A run whose adaptive step stays ceiling-bound at a fixed dt for many steps
    accumulates float round-off in the running sum of ``fields.t``.  Landing
    fractionally short of ``t_end`` then forces one more, very small step to
    close the gap -- entirely normal, and unrelated to numerical stability.
    Feeding that *clamped* step into the divergence check used to raise a false
    positive: five thousand additions of ``dt = 0.02`` leave a residual of
    ``2.757e-12`` before reaching ``t = 100``, comfortably under
    ``MIN_TIME_STEP`` (``1e-8``), which used to be reported as a collapsed step.
    """
    cfg = make_config(nx=8, ny=8, re=1000.0, dt=0.02, adaptive_dt=True,
                      cfl_max=0.9, boundary_config=walls())
    solver = ProjectionSolver(cfg)
    result = TimeStepper(solver).run(solver.initialize(), t_end=100.0)

    assert result.time == pytest.approx(100.0)
    assert result.fields.is_finite()
    assert solver.max_divergence(result.fields) < 1e-11


def test_genuine_collapse_is_still_caught_after_the_fix():
    """The fix must not loosen detection of an actually collapsing step.

    Unlike the benign end-of-run residual above, a *genuinely* shrinking
    adaptive step -- the un-clamped value returned by ``compute_dt`` itself --
    must still raise.  Forced here with a viscosity high enough that the
    viscous stability limit alone is already below ``MIN_TIME_STEP``.
    """
    cfg = make_config(nx=512, ny=512, re=1e-6, dt=1.0, adaptive_dt=True,
                      cfl_max=0.9, boundary_config=walls())
    solver = ProjectionSolver(cfg)
    with pytest.raises(DivergenceError, match="collapsed"):
        TimeStepper(solver).run(solver.initialize(), t_end=1.0)


def test_simple_solver_is_reported_as_unimplemented():
    cfg = cavity_config(solver_type=SolverType.SIMPLE)
    with pytest.raises(NotImplementedError, match="simple"):
        ProjectionSolver(cfg)


def test_a_stretched_mesh_is_accepted_by_the_solver():
    cfg = cavity_config(stretch_y=1.05)
    solver = ProjectionSolver(cfg)
    assert not solver.mesh.is_uniform


def test_a_stretched_periodic_axis_is_refused():
    """Stretching and periodicity are contradictory, not merely unimplemented.

    Geometric growth leaves the first and last cell different widths, so the
    spacing is discontinuous across the seam and the domain does not repeat.
    The operators would happily divide a flux there by a spacing from the far
    end of the domain and return a plausible wrong number.
    """
    cfg = cavity_config(stretch_x=1.05, boundary_config=periodic_walls())
    with pytest.raises(NonUniformMeshError, match="periodic and stretched"):
        ProjectionSolver(cfg)


def test_obstacle_shape_is_validated():
    cfg = cavity_config(nx=16, ny=16)
    with pytest.raises(ValueError, match="shape"):
        ProjectionSolver(cfg, obstacle=np.zeros((4, 4), dtype=bool))


def test_max_steps_caps_the_run():
    cfg = cavity_config(nx=16, ny=16, t_end=1e6)
    solver = ProjectionSolver(cfg)
    result = TimeStepper(solver).run(solver.initialize(), t_end=1e6, max_steps=7)
    assert result.steps == 7


# --------------------------------------------------------------------------- #
# Numba kernels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("blend", [0.0, 0.7])
@pytest.mark.parametrize("bcs", [walls(top=(BCKind.MOVING_WALL, 1.0)), periodic_walls()])
def test_numba_kernel_matches_the_numpy_stencils_exactly(blend, bcs):
    """The fused kernel is an optimisation, not an approximation."""
    from pycfd.core import kernels

    if not kernels.NUMBA_AVAILABLE:
        pytest.skip("numba is not installed")

    u0 = np.random.default_rng(3).standard_normal((21, 16))
    v0 = np.random.default_rng(4).standard_normal((20, 17))
    results = []
    for use_numba in (True, False):
        cfg = make_config(nx=20, ny=16, re=77.0, body_force=(0.3, -0.2),
                               use_numba=use_numba, boundary_config=bcs)
        solver = ProjectionSolver(cfg)
        fields = solver.initialize(u0, v0)
        results.append([a.copy() for a in solver.momentum_rhs(fields.u, fields.v, blend)])

    assert np.array_equal(results[0][0], results[1][0])
    assert np.array_equal(results[0][1], results[1][1])


def test_both_kernel_builds_agree():
    """The threaded build must reproduce the serial one bit for bit."""
    from pycfd.core import kernels

    if not kernels.NUMBA_AVAILABLE:
        pytest.skip("numba is not installed")

    nx = ny = 24
    u = np.random.default_rng(0).standard_normal((nx + 3, ny + 2))
    v = np.random.default_rng(1).standard_normal((nx + 2, ny + 3))
    out = []
    for kernel in (kernels._kernel_serial, kernels._kernel_parallel):
        ru, rv = np.empty((nx + 1, ny)), np.empty((nx, ny + 1))
        kernel(u, v, 0.01, 0.02, 0.001, 0.6, 0.1, -0.2, ru, rv)
        out.append((ru, rv))
    assert np.array_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1])


def test_les_falls_back_to_the_array_path():
    """The fused kernel assumes constant viscosity; LES must not use it."""
    solver = ProjectionSolver(cavity_config(nx=16, ny=16, use_les=True))
    assert solver._use_kernel is False
