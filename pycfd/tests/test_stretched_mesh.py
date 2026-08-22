"""Running the solver on a mesh whose cells are not all the same size.

What these tests are actually guarding
--------------------------------------
A stretched-mesh bug is unusually quiet.  Every operator still returns an array
of the right shape, the run still completes, and the answer is still *close* --
just first-order close instead of second-order close.  Nothing raises.  So the
tests here are mostly not "does it run" but "is it still the operator it claims
to be":

* the projection still removes divergence to round-off (:func:`test_projection_*`),
  which only holds if divergence, gradient and the assembled Poisson operator
  all agree about which of the two spacings each of them uses;
* the assembled operator is still symmetric, which conjugate gradients and the
  multigrid V-cycle both assume and neither checks;
* the solved field is still second order, which is the entire point of the
  feature and the one thing a wrong corner weight silently destroys.

The uniform mesh is tested here too, because the interesting claim is that it
did not change: same scalars, same kernel, same numbers.
"""

import numpy as np
import pytest

from pycfd.core.fields import FlowField
from pycfd.core.mesh import (MeshMetrics, NonUniformMeshError,
                            StructuredMesh, _geometric_widths)
from pycfd.core.pressure import assemble_poisson_matrix, make_pressure_solver
from pycfd.core.solver import ProjectionSolver
from pycfd.core.timestepper import TimeStepper
from pycfd.physics.turbulence import SmagorinskyModel

from .conftest import cavity_config, periodic_walls


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_a_uniform_mesh_keeps_scalar_spacings():
    """The reason the uniform path did not get slower or move its baselines.

    Every operator is written once, against the metrics.  It stays exactly the
    arithmetic it was only because those metrics are plain floats here, so
    ``a / m.hx`` is still a scalar division.
    """
    m = StructuredMesh(8, 8, 2.0, 3.0).metrics
    for name in ("hx", "hy", "hxu", "hyv", "hx_ext", "hy_ext",
                 "wx_corner", "wy_corner"):
        assert np.isscalar(getattr(m, name)), f"{name} should be a scalar"
    assert m.wx_corner == 0.5 and m.wy_corner == 0.5


def test_a_stretched_mesh_gets_arrays_shaped_to_broadcast():
    mesh = StructuredMesh(8, 6, stretch_x=1.1, stretch_y=1.2)
    m = mesh.metrics
    assert m.hx.shape == (8, 1) and m.hy.shape == (1, 6)
    assert m.hxu.shape == (9, 1) and m.hyv.shape == (1, 7)
    assert m.hx_ext.shape == (10, 1) and m.hy_ext.shape == (1, 8)
    assert m.wx_corner.shape == (9, 1) and m.wy_corner.shape == (1, 7)


def test_the_face_spacing_is_the_distance_between_cell_centres():
    """``hxu`` is the u control volume, and it is *not* the cell width.

    Confusing the two is the classic staggered-grid stretching bug: on a
    uniform mesh they are equal, so the confusion passes every uniform test.
    """
    mesh = StructuredMesh(10, 4, stretch_x=1.3)
    hxu = mesh.metrics.hxu[:, 0]
    assert hxu[1:-1] == pytest.approx(np.diff(mesh.xc))
    # The ghost mirrors its neighbour, so the end faces span one whole cell.
    assert hxu[0] == pytest.approx(mesh.dx_cells[0])
    assert hxu[-1] == pytest.approx(mesh.dx_cells[-1])


def test_the_corner_weight_is_where_the_corner_actually_is():
    """A cell face is not the midpoint of its two neighbouring cell centres."""
    mesh = StructuredMesh(4, 10, stretch_y=1.4)
    w = mesh.metrics.wy_corner[0, :]
    expected = (mesh.yf[1:-1] - mesh.yc[:-1]) / np.diff(mesh.yc)
    assert w[1:-1] == pytest.approx(expected)
    assert np.all(w[1:-1] != pytest.approx(0.5, abs=1e-3))


def test_integrating_over_a_stretched_mesh_gives_the_domain_area():
    mesh = StructuredMesh(16, 12, 3.0, 5.0, stretch_x=1.15, stretch_y=1.07)
    ones = np.ones(mesh.shape)
    assert mesh.metrics.integrate(ones) == pytest.approx(15.0)


def test_cell_widths_still_sum_to_the_domain():
    mesh = StructuredMesh(32, 32, 7.0, 11.0, stretch_x=1.2, stretch_y=1.05)
    assert mesh.dx_cells.sum() == pytest.approx(7.0)
    assert mesh.dy_cells.sum() == pytest.approx(11.0)


# --------------------------------------------------------------------------- #
# The Poisson operator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sx,sy", [(1.2, 1.0), (1.0, 1.2), (1.1, 1.15)])
def test_the_stretched_poisson_operator_is_symmetric(sx, sy):
    system = assemble_poisson_matrix(StructuredMesh(10, 10, stretch_x=sx, stretch_y=sy))
    assert abs(system.A - system.A.T).max() < 1.0e-14


def test_symmetry_survives_an_obstacle_and_an_outlet():
    """The two things that edit the stencil after it is built."""
    solid = np.zeros((12, 12), dtype=bool)
    solid[4:7, 5:8] = True
    system = assemble_poisson_matrix(
        StructuredMesh(12, 12, stretch_x=1.12, stretch_y=1.08),
        solid_mask=solid, dirichlet={"right": 0.0},
    )
    assert abs(system.A - system.A.T).max() < 1.0e-14


def test_the_solved_field_is_second_order_on_a_stretched_mesh():
    """The claim the whole item rests on.

    Refinement holds the stretching *function* fixed -- a constant total growth
    -- rather than the per-cell ratio, or the mesh would get more distorted as
    it got finer and the study would measure nothing.

    Note this is the order of the *solution*.  The non-uniform three-point
    Laplacian is only first order pointwise; the solution is second order
    anyway (supraconvergence), so measuring the truncation error here would
    give a misleadingly pessimistic answer.
    """
    a, b = np.pi, 2.0 * np.pi
    exact = lambda x, y: np.cos(a * x) * np.cos(b * y)      # noqa: E731

    def error(n, total_growth):
        ratio = total_growth ** (1.0 / (n - 1))
        mesh = StructuredMesh(n, n, 1.0, 1.0, stretch_x=ratio, stretch_y=ratio)
        solver = make_pressure_solver("direct", assemble_poisson_matrix(mesh))
        X, Y = mesh.cell_center_grid()
        p = solver.solve(-(a * a + b * b) * exact(X, Y))
        e = exact(X, Y)
        return np.sqrt(np.mean((p - (e - e.mean())) ** 2))

    for growth in (1.0, 8.0):
        errs = [error(n, growth) for n in (32, 64, 128)]
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
        assert min(orders) > 1.9, f"growth {growth}: orders {orders}"


def test_a_uniform_mesh_needs_no_row_scaling_at_all():
    system = assemble_poisson_matrix(StructuredMesh(8, 8))
    assert np.isscalar(system.row_scale) and system.row_scale == 1.0


# --------------------------------------------------------------------------- #
# The solver
# --------------------------------------------------------------------------- #
def test_a_stretched_periodic_axis_is_refused():
    cfg = cavity_config(stretch_y=1.05, boundary_config=periodic_walls())
    with pytest.raises(NonUniformMeshError, match="periodic and stretched"):
        ProjectionSolver(cfg)


@pytest.mark.parametrize("sx,sy", [(1.0, 1.0), (1.05, 1.0), (1.0, 1.05), (1.08, 1.06)])
def test_projection_removes_divergence_to_round_off(sx, sy):
    """The solver's central invariant, on every spacing.

    It holds only if the divergence, the pressure gradient and the assembled
    Poisson stencil all use the *same* spacing for the same face.  Any
    disagreement between the three shows up here and essentially nowhere else.
    """
    solver = ProjectionSolver(cavity_config(nx=24, ny=24, stretch_x=sx, stretch_y=sy))
    rng = np.random.default_rng(0)
    u, v = solver.mesh.zeros_u(), solver.mesh.zeros_v()
    u[solver.u_upd] = rng.standard_normal(u[solver.u_upd].shape) * 0.1
    v[solver.v_upd] = rng.standard_normal(v[solver.v_upd].shape) * 0.1

    before = np.abs(solver.divergence(u, v)).max()
    solver.project(u, v, dt=1.0)
    after = np.abs(solver.divergence(u, v)).max()
    assert after / before < 1.0e-11


def test_the_operators_agree_with_the_matrix_that_was_assembled_from_them():
    """``div(grad(p))`` must be the assembled operator, up to the row scaling.

    This is the same invariant as the projection test, checked directly rather
    than through a solve, so a failure says *which* half disagrees.
    """
    solver = ProjectionSolver(cavity_config(nx=20, ny=20, stretch_x=1.1, stretch_y=1.07))
    nx, ny = solver.mesh.shape
    system = solver.pressure_solver.system
    rng = np.random.default_rng(1)
    p_int = rng.standard_normal((nx, ny))

    p = solver.mesh.zeros_p()
    p[1:nx + 1, 1:ny + 1] = p_int
    u, v = solver.mesh.zeros_u(), solver.mesh.zeros_v()
    m = solver.metrics
    lo_u, hi_u = solver.u_upd[0].start, solver.u_upd[0].stop
    lo_v, hi_v = solver.v_upd[1].start, solver.v_upd[1].stop
    u[solver.u_upd] = -(p[lo_u:hi_u, 1:ny + 1] - p[lo_u - 1:hi_u - 1, 1:ny + 1]) \
        / MeshMetrics._slice(m.hxu, lo_u - 1, hi_u - 1, 0)
    v[solver.v_upd] = -(p[1:nx + 1, lo_v:hi_v] - p[1:nx + 1, lo_v - 1:hi_v - 1]) \
        / MeshMetrics._slice(m.hyv, lo_v - 1, hi_v - 1, 1)

    from_operators = -solver.divergence(u, v).ravel()
    from_matrix = (system.A @ p_int.ravel()) / system.row_scale
    rel = np.abs(from_operators - from_matrix).max() / np.abs(from_operators).max()
    assert rel < 1.0e-13


def test_the_fused_kernel_steps_aside_for_a_stretched_mesh():
    """It takes a scalar dx and dy, which a stretched mesh does not have."""
    stretched = ProjectionSolver(cavity_config(stretch_x=1.1, use_numba=True))
    assert not stretched._use_kernel


def test_a_stretched_run_stays_finite_and_divergence_free():
    solver = ProjectionSolver(cavity_config(nx=24, ny=24, stretch_x=1.06,
                                            stretch_y=1.06))
    dt = 0.4 * TimeStepper(solver, solver.config).viscous_dt_limit
    fields = solver.initialize(u_init=0.0, v_init=0.0)
    fields.u[:, -1] = 1.0                      # drag the lid
    for _ in range(15):
        fields = solver.step(fields, dt)
    assert np.isfinite(fields.u).all() and np.isfinite(fields.v).all()
    assert solver.max_divergence(fields) < 1.0e-10


def test_stretching_costs_quadratically_in_time_step():
    """Worth knowing before reaching for it: the price is not linear.

    The explicit viscous limit goes as ``h**2`` while the convective one goes
    as ``h``, so halving the smallest cell to resolve a boundary layer
    quarters the stable step.  A run that clusters aggressively pays for it in
    steps, not in cells.
    """
    limits = {}
    for stretch in (1.0, 1.05, 1.10):
        solver = ProjectionSolver(cavity_config(nx=32, ny=32, stretch_y=stretch))
        limits[stretch] = TimeStepper(solver, solver.config).viscous_dt_limit

    assert limits[1.05] < limits[1.0]
    assert limits[1.10] < limits[1.05]
    # The limit tracks the smallest cell squared, not the average cell.
    solver = ProjectionSolver(cavity_config(nx=32, ny=32, stretch_y=1.10))
    m = solver.mesh.metrics
    assert m.min_hy < solver.mesh.ly / 32


# --------------------------------------------------------------------------- #
# Everything downstream of the operators
# --------------------------------------------------------------------------- #
def test_the_filter_width_follows_the_local_cell():
    """A single filter width would over-damp exactly where the mesh was refined.

    The Smagorinsky length scale *is* the local grid spacing, so on a stretched
    mesh it has to vary with it -- otherwise clustering cells near a body adds
    resolution and then throws it away again as eddy viscosity.
    """
    uniform = SmagorinskyModel(StructuredMesh(16, 16))
    assert np.isscalar(uniform.delta)

    stretched = SmagorinskyModel(StructuredMesh(16, 16, stretch_y=1.2))
    assert not np.isscalar(stretched.delta)
    assert stretched.delta.max() / stretched.delta.min() > 2.0


def test_integral_quantities_weight_each_cell_by_its_own_area():
    """A plain sum would count a thin near-wall cell the same as a fat one."""
    from pycfd.analysis.postprocess import kinetic_energy

    mesh = StructuredMesh(16, 16, stretch_x=1.2, stretch_y=1.2)
    solver = ProjectionSolver(cavity_config(nx=16, ny=16, stretch_x=1.2,
                                            stretch_y=1.2))
    fields = FlowField.zeros(solver.mesh)
    fields.u_phys[...] = 1.0
    fields.v_phys[...] = 0.0
    # 0.5 * u^2 * area over a unit square with u = 1 everywhere.
    assert kinetic_energy(fields) == pytest.approx(0.5 * mesh.lx * mesh.ly, rel=1e-12)


def test_the_stream_function_walks_cell_by_cell():
    """Each step of the integration crosses one cell, so it is weighted by it.

    A uniform ``dx`` here would drift the stream function across the domain on
    a stretched mesh, which looks like a physical asymmetry rather than a bug.
    """
    from pycfd.analysis.postprocess import stream_function

    solver = ProjectionSolver(cavity_config(nx=20, ny=20, stretch_x=1.1))
    fields = FlowField.zeros(solver.mesh)
    fields.u_phys[...] = 1.0                     # psi = y
    psi = stream_function(fields)
    expected = solver.mesh.yf[None, :] * np.ones((solver.mesh.nx + 1, 1))
    assert psi == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- #
# Where the small cells go
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cluster", ["low", "walls", "centre"])
@pytest.mark.parametrize("n", [8, 9])
def test_every_layout_tiles_the_domain_exactly(cluster, n):
    """Whatever the shape, the cells still have to add up to the domain."""
    widths = _geometric_widths(3.0, n, 1.3, cluster)
    assert widths.shape == (n,)
    assert widths.sum() == pytest.approx(3.0, rel=1e-14)
    assert (widths > 0).all()


@pytest.mark.parametrize("n", [8, 9])
def test_the_two_sided_layouts_are_symmetric(n):
    """A channel and a far field are both symmetric; the mesh has to be too.

    An asymmetric mesh on a symmetric problem shows up as an asymmetric
    *solution*, which is indistinguishable from a physical result until someone
    thinks to check the grid.
    """
    for cluster in ("walls", "centre"):
        widths = _geometric_widths(1.0, n, 1.25, cluster)
        assert widths == pytest.approx(widths[::-1], rel=1e-14)


@pytest.mark.parametrize("n", [8, 9])
def test_the_ratio_stays_the_per_cell_growth_rate(n):
    """``stretch`` means the same thing in every mode: neighbouring cells differ
    by that factor.  The middle is where the growth turns around, so it is the
    one place the ratio inverts."""
    ratio = 1.25
    for cluster, expect_first in (("walls", ratio), ("centre", 1.0 / ratio)):
        widths = _geometric_widths(1.0, n, ratio, cluster)
        assert widths[1] / widths[0] == pytest.approx(expect_first, rel=1e-12)


def test_the_smallest_cells_land_where_the_mode_says():
    """The whole point of the mode: it decides which cells are the fine ones."""
    n = 40
    where_min = {
        c: int(np.argmin(_geometric_widths(1.0, n, 1.1, c)))
        for c in ("low", "walls", "centre")
    }
    assert where_min["low"] == 0                       # at the low wall
    assert where_min["walls"] in (0, n - 1)            # at a wall
    assert where_min["centre"] in (n // 2 - 1, n // 2)  # in the middle


def test_the_one_sided_layout_is_bit_for_bit_what_it_always_was():
    """``low`` is the default, so every existing stretched result must be
    untouched by the arrival of the other two modes."""
    n, ratio, length = 12, 1.2, 2.5
    w0 = length * (ratio - 1.0) / (ratio ** n - 1.0)
    previous = w0 * ratio ** np.arange(n, dtype=float)
    assert np.array_equal(_geometric_widths(length, n, ratio, "low"), previous)
    assert StructuredMesh(n, 4, length, 1.0, stretch_x=ratio).cluster_x == "low"


@pytest.mark.parametrize("cluster", ["low", "walls", "centre"])
def test_a_ratio_of_one_is_uniform_in_every_mode(cluster):
    mesh = StructuredMesh(16, 16, 1.0, 1.0, stretch_y=1.0, cluster_y=cluster)
    assert mesh.is_uniform


def test_the_american_spelling_is_accepted():
    """``centre`` matches the prose in this package, but nobody should have to
    guess which spelling a flag wants."""
    assert StructuredMesh(8, 8, cluster_x="center").cluster_x == "centre"
    with pytest.raises(ValueError, match="cluster mode"):
        StructuredMesh(8, 8, cluster_x="middle")


@pytest.mark.slow
def test_clustering_at_both_walls_beats_clustering_at_one():
    """The reason the other modes exist.

    Poiseuille flow is symmetric, so a one-sided mesh spends its cells refining
    one wall and starves the other -- and the starved wall costs more than the
    refined one wins.  This is a *measured* comparison rather than an assertion
    about the discretisation: all three meshes here are second order, they just
    put the cells in different places.
    """
    from pycfd.analysis.validation import poiseuille_profile
    from pycfd.cases import channel_flow

    def wall_error(**kwargs):
        sim = channel_flow.build(re=10.0, nx=8, ny=48, dt=0.002, **kwargs)
        sim.run()
        uc, _ = sim.fields.cell_velocities()
        profile, mesh = uc[4, :], sim.mesh
        mean = float(profile @ mesh.dy_cells) / float(mesh.dy_cells.sum())
        exact = poiseuille_profile(mesh.yc, 1.0, 1.5 * mean)
        return float(np.abs(profile - exact).max())

    uniform = wall_error(stretch_y=1.0)
    one_sided = wall_error(stretch_y=1.05, cluster_y="low")
    two_sided = wall_error(stretch_y=1.05, cluster_y="walls")

    assert one_sided > 3.0 * uniform, "expected one-sided clustering to hurt here"
    assert two_sided < uniform, "expected two-sided clustering to help here"


def test_a_body_in_mid_domain_is_better_resolved_by_centre_clustering():
    """The case the roadmap actually wanted stretching for: a body in the middle
    of a domain that only has to be large, not resolved."""
    import logging

    from pycfd.cases.cylinder_flow import _cell_height_at_body, build

    def cells_across(**kwargs):
        logging.disable(logging.WARNING)
        try:
            sim = build(re=100, nx=128, **kwargs)
        finally:
            logging.disable(logging.NOTSET)
        return (sim.obstacle.characteristic_length
                / _cell_height_at_body(sim.mesh, sim.obstacle))

    coarse = cells_across(ny=64)
    clustered = cells_across(ny=64, stretch_y=1.06, cluster_y="centre")
    finer_uniform = cells_across(ny=96)
    assert clustered > finer_uniform > coarse


def test_the_body_lands_in_the_same_place_on_a_stretched_mesh():
    """The obstacle is rasterised before the configuration exists, so it is easy
    for the mask and the mesh that gets run to disagree about cell spacing --
    which does not fail, it just quietly moves the body."""
    import logging

    from pycfd.cases.cylinder_flow import build

    logging.disable(logging.WARNING)
    try:
        sims = [build(re=100, nx=128, ny=64, **kw)
                for kw in (dict(), dict(stretch_y=1.06, cluster_y="centre"))]
    finally:
        logging.disable(logging.NOTSET)
    for sim in sims:
        rows = np.flatnonzero(np.asarray(sim.obstacle.mask).any(axis=0))
        span = sim.mesh.yc[rows]
        assert 0.5 * (span.min() + span.max()) == pytest.approx(
            sim.mesh.ly / 2.0, abs=0.06
        )
