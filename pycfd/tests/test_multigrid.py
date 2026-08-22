"""Multigrid hierarchy, its V-cycle, and the two solvers built on it.

Multigrid is easy to get *nearly* right: a hierarchy with a subtly wrong
prolongator, an asymmetric coarse solve or a mishandled null space still
converges on an easy problem and quietly falls apart on a real one.  So these
tests check the algebraic properties the method depends on -- the constant is
represented exactly at every level, coarse row sums vanish, the preconditioner
is symmetric, the face set is inherited unchanged -- rather than only checking
that some answer comes back.

The property that makes it multigrid at all is the last section: the iteration
count does not grow with the grid.  A method that solved everything correctly
but needed four times the iterations on four times the cells would be a
correct method and a pointless one.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pycfd.config import BCKind, BCSpec, PressureSolver
from pycfd.core.mesh import StructuredMesh
from pycfd.core.multigrid import (
    ANISOTROPY_THRESHOLD,
    COARSEST_UNKNOWNS,
    MultigridHierarchy,
    aggregate_blocks,
    coarse_shape,
    coarsening_factors,
    spectral_radius,
    tentative_prolongator,
)
from pycfd.core.pressure import (
    CGSolver,
    DirectSolver,
    MultigridCGSolver,
    MultigridSolver,
    assemble_poisson_matrix,
    make_pressure_solver,
)
from pycfd.core.solver import ProjectionSolver

from .conftest import DIVERGENCE_TOL, make_config, walls

MG_SOLVERS = ("multigrid", "mgcg")


def system_for(nx=64, ny=64, lx=1.0, ly=1.0, **kw):
    """An assembled Poisson system plus the shape its unknowns live on."""
    mesh = StructuredMesh(nx, ny, lx, ly)
    return assemble_poisson_matrix(mesh, **kw), mesh


def hierarchy_for(system, mesh, **kw):
    return MultigridHierarchy(system.A, mesh.shape, system.fluid,
                              system.singular, **kw)


def compatible_rhs(system, seed=0):
    """A random right-hand side the singular operator can actually solve."""
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(system.n)
    b[~system.fluid] = 0.0
    if system.singular:
        b[system.fluid] -= b[system.fluid].mean()
    return b


def dense_operator(apply, n):
    """Materialise a linear operator column by column."""
    M = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        M[:, k] = apply(e)
    return M


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_every_active_cell_lands_in_exactly_one_block():
    active = np.ones(8 * 6, dtype=bool)
    labels, coarse_active = aggregate_blocks((8, 6), active, (2, 2))
    assert (labels >= 0).all()
    assert coarse_active.all()
    # One column of P per block, one entry per row: a partition, not a covering.
    P = tentative_prolongator(labels, coarse_active.size)
    assert P.shape == (48, 12)
    assert np.array_equal(P.sum(axis=1).A1, np.ones(48))


def test_a_block_holds_the_four_cells_that_are_actually_neighbours():
    """Aggregation must group cells that touch, not cells that share an index."""
    labels, _ = aggregate_blocks((4, 4), np.ones(16, dtype=bool), (2, 2))
    grid = labels.reshape(4, 4)
    assert grid[0, 0] == grid[0, 1] == grid[1, 0] == grid[1, 1]
    assert grid[0, 0] != grid[2, 2]
    assert len(set(grid.ravel())) == 4


def test_an_odd_extent_leaves_a_block_of_one():
    """The leftover cell gets its own block; measured to beat widening the last."""
    labels, coarse_active = aggregate_blocks((5, 4), np.ones(20, dtype=bool), (2, 2))
    assert coarse_shape((5, 4), (2, 2)) == (3, 2)
    assert coarse_active.size == 6 and coarse_active.all()
    grid = labels.reshape(5, 4)
    # The last row of cells is alone in its blocks, not folded into row 2/3.
    assert grid[4, 0] != grid[3, 0]
    assert len(np.unique(grid[4, :])) == 2


def test_a_solid_cell_joins_no_block_and_leaves_an_empty_one():
    active = np.ones((4, 4), dtype=bool)
    active[0:2, 0:2] = False              # exactly one whole 2x2 block
    labels, coarse_active = aggregate_blocks((4, 4), active, (2, 2))
    assert (labels.reshape(4, 4)[0:2, 0:2] == -1).all()
    assert coarse_active.sum() == 3, "the fully solid block must not survive"
    P = tentative_prolongator(labels, coarse_active.size)
    assert P[0].nnz == 0, "a solid cell must have no prolongation row"


def test_a_partly_solid_block_survives_with_its_fluid_cells():
    active = np.ones((4, 4), dtype=bool)
    active[0, 0] = False
    labels, coarse_active = aggregate_blocks((4, 4), active, (2, 2))
    assert coarse_active.all()
    P = tentative_prolongator(labels, coarse_active.size)
    assert P[0].nnz == 0
    assert P.getcol(labels[1]).nnz == 3, "the block keeps its three fluid cells"


def test_the_coarse_grid_keeps_its_shape_so_the_next_level_can_aggregate():
    """A hole must stay in the same place, not renumber the grid around it."""
    active = np.ones((8, 8), dtype=bool)
    active[2:4, 2:4] = False
    _, coarse_active = aggregate_blocks((8, 8), active, (2, 2))
    assert coarse_active.size == 16
    assert not coarse_active.reshape(4, 4)[1, 1]
    assert coarse_active.sum() == 15


# --------------------------------------------------------------------------- #
# Anisotropy
# --------------------------------------------------------------------------- #
def test_a_square_cell_coarsens_both_ways():
    system, mesh = system_for(32, 32)
    assert coarsening_factors(system.A, mesh.shape) == (2, 2)


@pytest.mark.parametrize("lx,ly,expected", [
    (4.0, 1.0, (1, 2)),      # dx >> dy: coupling is strong in y, so coarsen y
    (1.0, 4.0, (2, 1)),      # and the other way round
])
def test_a_stretched_cell_coarsens_along_the_strong_axis(lx, ly, expected):
    system, mesh = system_for(32, 32, lx, ly)
    assert coarsening_factors(system.A, mesh.shape) == expected


def test_semi_coarsening_walks_the_aspect_ratio_back_to_square():
    """A 16:1 mesh must not stay 16:1 all the way down the hierarchy.

    Each pass divides the strength ratio by four, so a spacing ratio of 16 --
    a strength ratio of 256 -- takes four y-only passes before the cells are
    square and both axes coarsen together.  The point is that it converges,
    not how many steps it takes.
    """
    system, mesh = system_for(64, 64, 16.0, 1.0)
    shapes = [lv.shape for lv in hierarchy_for(system, mesh).levels]

    semi = [(a, b) for a, b in zip(shapes, shapes[1:]) if a[0] == b[0]]
    full = [(a, b) for a, b in zip(shapes, shapes[1:]) if a[0] != b[0]]
    assert len(semi) == 4, f"expected four y-only passes, got {shapes}"
    assert full, f"the hierarchy never recovered full coarsening: {shapes}"
    # Once it does, the cells are square: 64 x 4 cells over 16 x 1.
    first_full = full[0][0]
    assert first_full == (64, 4)


def test_the_threshold_is_where_semi_coarsening_starts_to_help():
    """Below the threshold, full coarsening; above it, one axis only."""
    assert ANISOTROPY_THRESHOLD == 2.0
    # Coupling scales as 1/h^2, so a spacing ratio r gives a strength ratio r^2.
    just_under, just_over = 1.35, 1.5      # squared: 1.82 and 2.25
    assert coarsening_factors(*_axes(just_under)) == (2, 2)
    assert coarsening_factors(*_axes(just_over)) == (1, 2)


def _axes(spacing_ratio):
    system, mesh = system_for(32, 32, spacing_ratio, 1.0)
    return system.A, mesh.shape


def test_semi_coarsening_is_what_keeps_a_stretched_mesh_cheap():
    """The whole point: without it, a 4:1 mesh costs several times the work."""
    system, mesh = system_for(64, 64, 4.0, 1.0)
    smart = _preconditioned_iterations(system, mesh)

    # Force full coarsening by claiming the operator is isotropic.
    import pycfd.core.multigrid as mgmod
    original = mgmod.coarsening_factors
    mgmod.coarsening_factors = lambda A, shape: (2, 2)
    try:
        naive = _preconditioned_iterations(system, mesh)
    finally:
        mgmod.coarsening_factors = original

    assert smart < naive / 3, (
        f"semi-coarsening took {smart} iterations against {naive} for full "
        "coarsening; the gap should be several-fold"
    )


def _preconditioned_iterations(system, mesh, tol=1e-10, **kw):
    solver = MultigridCGSolver(system, tol=tol, maxiter=5000, **kw)
    solver.solve(compatible_rhs(system).reshape(system.shape))
    return solver.last_iterations


# --------------------------------------------------------------------------- #
# The hierarchy's algebraic invariants
# --------------------------------------------------------------------------- #
def test_the_constant_is_interpolated_exactly_at_every_level():
    """``P 1_coarse = 1_fine`` is what keeps the constant the null vector below.

    Lose it and each level needs its own null vector carried alongside; keep it
    and 'subtract the mean' means the same thing all the way down.  It survives
    prolongator smoothing only because the operator annihilates the constant,
    so this is a real check on the smoothing pass, not a restatement of the 0/1
    tentative prolongator.
    """
    system, mesh = system_for(64, 64)
    mg = hierarchy_for(system, mesh)
    assert mg.n_levels >= 3
    for depth in range(1, mg.n_levels):
        level = mg.levels[depth]
        coarse_one = level.active.astype(float)
        fine_one = mg.levels[depth - 1].active.astype(float)
        assert np.abs(level.P @ coarse_one - fine_one).max() < 1e-12


def test_coarse_operators_annihilate_the_constant():
    system, mesh = system_for(64, 64)
    mg = hierarchy_for(system, mesh)
    for depth, level in enumerate(mg.levels):
        one = level.active.astype(float)
        residual = np.abs(level.A @ one).max() / np.abs(level.A).max()
        assert residual < 1e-12, f"level {depth} does not annihilate the constant"


def test_coarse_operators_stay_symmetric():
    """``A_c = P^T A P`` is symmetric by construction; assembly bugs are not."""
    system, mesh = system_for(48, 48)
    mg = hierarchy_for(system, mesh)
    for depth, level in enumerate(mg.levels):
        asymmetry = abs(level.A - level.A.T).max()
        assert asymmetry < 1e-9 * abs(level.A).max(), f"level {depth} is asymmetric"


def test_the_hierarchy_actually_coarsens_and_then_stops():
    system, mesh = system_for(128, 128)
    mg = hierarchy_for(system, mesh)
    counts = mg.unknowns
    assert counts[0] == 128 * 128
    for coarse, finer in zip(counts[1:], counts):
        assert coarse < finer
    assert counts[-1] <= COARSEST_UNKNOWNS
    assert mg.n_levels >= 4


def test_a_grid_too_small_to_coarsen_is_just_a_direct_solve():
    system, mesh = system_for(8, 8)
    mg = hierarchy_for(system, mesh)
    assert mg.n_levels == 1
    # One "cycle" is then exact, so the standalone solver converges immediately.
    solver = MultigridSolver(system, tol=1e-12, maxiter=50)
    reference = DirectSolver(system).solve(compatible_rhs(system).reshape((8, 8)))
    p = solver.solve(compatible_rhs(system).reshape((8, 8)))
    assert np.abs(p - reference).max() < 1e-10
    assert solver.last_iterations == 1


def test_operator_complexity_stays_affordable():
    """The hierarchy is the memory cost; a blown-up coarse stencil is the way it
    goes wrong.  Under 3 keeps it well below the direct solver's fill."""
    system, mesh = system_for(128, 128)
    mg = hierarchy_for(system, mesh)
    assert 1.0 < mg.operator_complexity < 3.0


def test_a_body_stays_a_hole_on_every_level():
    solid = np.zeros((64, 64), dtype=bool)
    solid[24:40, 24:40] = True
    system, mesh = system_for(64, 64, solid_mask=solid)
    mg = hierarchy_for(system, mesh)

    assert not mg.levels[0].active[system.fluid == False].any()  # noqa: E712
    for depth in range(1, mg.n_levels):
        level = mg.levels[depth]
        if level.n <= COARSEST_UNKNOWNS:
            break
        assert not level.active.all(), (
            f"the body has been averaged away by level {depth}"
        )


def test_building_the_same_hierarchy_twice_gives_the_same_numbers():
    """The setup estimates a spectral radius by power iteration, from a random
    start.  That start is seeded, and it has to stay seeded: this module is the
    only randomness anywhere in the solve path, and the package promises runs
    are bit-identical on one machine."""
    system, mesh = system_for(48, 48)
    first, second = hierarchy_for(system, mesh), hierarchy_for(system, mesh)
    for a, b in zip(first.levels, second.levels):
        assert np.array_equal(a.weighted_dinv, b.weighted_dinv)
        assert np.array_equal(a.A.data, b.A.data)

    rhs = compatible_rhs(system, seed=11).reshape(system.shape)
    left = MultigridCGSolver(system, tol=1e-12).solve(rhs)
    right = MultigridCGSolver(system, tol=1e-12).solve(rhs)
    assert np.array_equal(left, right)


def test_the_spectral_radius_estimate_brackets_the_true_one():
    system, mesh = system_for(32, 32)
    A = system.A
    dinv = 1.0 / A.diagonal()
    estimate = spectral_radius(A, dinv)
    exact = np.abs(np.linalg.eigvals(
        (sp.diags(dinv) @ A).toarray()
    )).max()
    assert estimate == pytest.approx(exact, rel=0.1)
    assert estimate >= exact, "a short estimate would over-relax the smoother"


# --------------------------------------------------------------------------- #
# The V-cycle as an operator
# --------------------------------------------------------------------------- #
def test_the_preconditioner_is_symmetric():
    """Conjugate gradient assumes it; an asymmetric V-cycle stalls unpredictably.

    Asymmetry is easy to introduce by accident -- unequal pre- and
    post-smoothing, a restriction that is not exactly ``P^T``, or a pinned
    coarse solve instead of the pseudo-inverse -- and none of those show up as
    a wrong answer on an easy problem.
    """
    system, mesh = system_for(32, 32)
    mg = hierarchy_for(system, mesh)
    M = dense_operator(mg.apply, system.n)
    assert np.abs(M - M.T).max() < 1e-10 * np.abs(M).max()


def test_the_preconditioner_is_symmetric_with_a_body_and_an_outlet():
    solid = np.zeros((32, 32), dtype=bool)
    solid[12:20, 12:20] = True
    system, mesh = system_for(32, 32, solid_mask=solid, dirichlet={"right": 0.0})
    mg = hierarchy_for(system, mesh)
    M = dense_operator(mg.apply, system.n)
    assert np.abs(M - M.T).max() < 1e-10 * np.abs(M).max()


def test_unequal_smoothing_is_refused_rather_than_silently_asymmetric():
    system, mesh = system_for(32, 32)
    with pytest.raises(ValueError, match="symmetric"):
        hierarchy_for(system, mesh, presmooth=2, postsmooth=1)


def test_a_cycle_with_no_smoothing_is_refused():
    system, mesh = system_for(32, 32)
    with pytest.raises(ValueError, match="at least one"):
        hierarchy_for(system, mesh, presmooth=0, postsmooth=0)


def test_the_preconditioner_maps_into_the_range_of_the_operator():
    """It must return something the singular operator can act on consistently."""
    system, mesh = system_for(48, 48)
    mg = hierarchy_for(system, mesh)
    out = mg.apply(compatible_rhs(system))
    assert out[~system.fluid].max(initial=0.0) == 0.0
    assert abs(out[system.fluid].mean()) < 1e-12


def test_one_cycle_removes_most_of_the_error():
    """A V-cycle that merely converges is not enough; it has to converge fast."""
    system, mesh = system_for(64, 64)
    mg = hierarchy_for(system, mesh)
    b = compatible_rhs(system)
    scale = np.linalg.norm(b)

    x = np.zeros_like(b)
    residuals = []
    for _ in range(4):
        mg._cycle(0, b, x)
        mg._project(mg.levels[0], x)
        residuals.append(float(np.linalg.norm(b - system.A @ x)) / scale)

    for previous, current in zip(residuals, residuals[1:]):
        assert current < 0.6 * previous, (
            f"convergence factor {current / previous:.2f} is too slow for a "
            f"V-cycle; residual history {residuals}"
        )


def test_the_convergence_factor_does_not_depend_on_the_grid():
    """The defining property.  Without it this is an expensive smoother."""
    factors = {}
    for n in (32, 64, 128):
        system, mesh = system_for(n, n)
        mg = hierarchy_for(system, mesh)
        b = compatible_rhs(system)
        scale = np.linalg.norm(b)
        x = np.zeros_like(b)
        history = []
        for _ in range(6):
            mg._cycle(0, b, x)
            mg._project(mg.levels[0], x)
            history.append(float(np.linalg.norm(b - system.A @ x)) / scale)
        # Skip the first cycle: it also clears the initial transient.
        factors[n] = float(np.mean([
            history[i + 1] / history[i] for i in range(1, len(history) - 1)
        ]))

    spread = max(factors.values()) - min(factors.values())
    assert spread < 0.08, f"convergence factor drifts with the grid: {factors}"
    assert max(factors.values()) < 0.6, factors


# --------------------------------------------------------------------------- #
# The solvers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", MG_SOLVERS)
@pytest.mark.parametrize("assembly,label", [
    ({}, "neumann"),
    ({"dirichlet": {"right": 1.0}}, "pressure outlet"),
    ({"periodic_x": True, "periodic_y": True}, "doubly periodic"),
])
def test_the_multigrid_solvers_agree_with_the_direct_one(kind, assembly, label):
    system, _ = system_for(48, 48, **assembly)
    rhs = compatible_rhs(system, seed=5).reshape(system.shape)
    reference = DirectSolver(system).solve(rhs)
    solver = make_pressure_solver(PressureSolver(kind), system,
                                  tol=1e-12, maxiter=5000)
    assert np.abs(solver.solve(rhs) - reference).max() < 1e-9, label


@pytest.mark.parametrize("kind", MG_SOLVERS)
def test_the_multigrid_solvers_agree_around_an_obstacle(kind):
    solid = np.zeros((64, 48), dtype=bool)
    solid[20:32, 16:32] = True
    system, _ = system_for(64, 48, 2.0, 1.5, solid_mask=solid)
    rhs = compatible_rhs(system, seed=6).reshape(system.shape)
    reference = DirectSolver(system).solve(rhs)
    solver = make_pressure_solver(PressureSolver(kind), system,
                                  tol=1e-12, maxiter=5000)
    assert np.abs(solver.solve(rhs) - reference).max() < 1e-9


@pytest.mark.parametrize("kind", MG_SOLVERS)
@pytest.mark.parametrize("shape", [(33, 33), (48, 31), (17, 65)])
def test_odd_grids_are_solved_as_well_as_even_ones(kind, shape):
    """Nothing in the hierarchy may assume a power of two, or even an even number."""
    nx, ny = shape
    system, _ = system_for(nx, ny, nx / 32, ny / 32)
    rhs = compatible_rhs(system, seed=7).reshape(system.shape)
    reference = DirectSolver(system).solve(rhs)
    solver = make_pressure_solver(PressureSolver(kind), system,
                                  tol=1e-12, maxiter=5000)
    assert np.abs(solver.solve(rhs) - reference).max() < 1e-9


def test_the_manufactured_solution_comes_back():
    """``p = cos(pi x) cos(pi y)`` satisfies dp/dn = 0 on the unit box."""
    system, mesh = system_for(64, 64)
    X, Y = mesh.cell_center_grid()
    exact = np.cos(np.pi * X) * np.cos(np.pi * Y)
    solver = MultigridCGSolver(system, tol=1e-11, maxiter=5000)
    p = solver.solve(-2.0 * np.pi ** 2 * exact)
    assert np.abs(p - (exact - exact.mean())).max() < 2e-3


def test_an_incompatible_rhs_is_projected_not_rejected():
    system, _ = system_for(32, 32)
    p = MultigridCGSolver(system, tol=1e-12).solve(np.ones(system.shape))
    assert np.abs(p).max() < 1e-10


def test_a_bad_guess_still_reaches_the_right_answer():
    """A guess may cost iterations; it may never change what comes back.

    Random noise is the *worst* possible start for a Laplacian, not a
    near-miss: the operator amplifies high frequencies by ``1/h**2``, so
    scattering 1e-3 over the exact solution leaves a larger residual than
    starting from zero.  The answer still has to be the same one.
    """
    system, _ = system_for(64, 64)
    rhs = compatible_rhs(system, seed=8).reshape(system.shape)
    solver = MultigridCGSolver(system, tol=1e-12, maxiter=5000)

    cold = solver.solve(rhs)
    rng = np.random.default_rng(9)
    noisy = solver.solve(rhs, cold + 1e-3 * rng.standard_normal(system.shape))
    assert np.abs(noisy - cold).max() < 1e-9


def test_the_previous_solution_of_a_nearby_problem_saves_iterations():
    """The case warm starting is actually for: the field barely moved."""
    system, mesh = system_for(64, 64)
    X, Y = mesh.cell_center_grid()
    first = compatible_rhs(system, seed=10).reshape(system.shape)
    # A small, smooth change -- what one substep does to div(u*)/dt.
    nudge = 0.01 * np.cos(np.pi * X) * np.cos(np.pi * Y)
    second = first + nudge - nudge.mean()

    solver = MultigridCGSolver(system, tol=1e-12, maxiter=5000)
    previous = solver.solve(first)
    cold = solver.solve(second)
    cold_iterations = solver.last_iterations
    warm = solver.solve(second, previous)

    assert np.abs(warm - cold).max() < 1e-9
    assert solver.last_iterations < cold_iterations, (
        f"warm start took {solver.last_iterations} iterations against "
        f"{cold_iterations} cold; it should have saved several"
    )


def test_a_stalled_multigrid_solve_says_so_instead_of_grinding():
    """``poisson_maxiter`` counts Jacobi sweeps; on V-cycles it is absurdly large."""
    system, _ = system_for(32, 32)
    solver = MultigridSolver(system, tol=1e-300, maxiter=1_000_000)
    with pytest.raises(RuntimeError, match="stalled"):
        solver.solve(compatible_rhs(system).reshape(system.shape))
    assert solver.last_iterations < 200, (
        "the stall guard must trip long before maxiter"
    )


@pytest.mark.parametrize("kind", MG_SOLVERS)
def test_the_multigrid_solvers_are_warm_startable(kind):
    system, _ = system_for(32, 32)
    solver = make_pressure_solver(PressureSolver(kind), system)
    assert solver.warm_startable
    assert not DirectSolver(system).warm_startable


def test_more_sweeps_cost_fewer_iterations():
    system, mesh = system_for(64, 64)
    one = _preconditioned_iterations(system, mesh, smooth_sweeps=1)
    two = _preconditioned_iterations(system, mesh, smooth_sweeps=2)
    assert two < one


def test_the_iteration_count_does_not_grow_with_the_grid():
    """What multigrid buys, stated as the number it is bought in.

    Jacobi-preconditioned CG needs iterations proportional to the number of
    cells across the domain; that is the cost this replaces.
    """
    counts = {}
    for n in (32, 64, 128, 256):
        system, mesh = system_for(n, n)
        counts[n] = _preconditioned_iterations(system, mesh)
    assert max(counts.values()) <= min(counts.values()) + 2, (
        f"iteration count is drifting with the grid: {counts}"
    )
    assert max(counts.values()) < 25, counts


def test_the_v_cycle_beats_jacobi_preconditioning_by_an_order_of_magnitude():
    """The comparison the docs quote, checked at one grid so it stays true.

    Jacobi-CG needs 552 iterations here against 13 -- and the gap widens with
    the grid, because that column doubles every time the grid does.
    """
    import scipy.sparse.linalg as spla

    system, mesh = system_for(128, 128)
    jacobi = CGSolver(system, tol=1e-10, maxiter=200_000)
    b = jacobi._prepare_rhs(compatible_rhs(system))
    counted = [0]
    spla.cg(system.A, b, rtol=1e-10, atol=0.0, maxiter=200_000, M=jacobi._M,
            callback=lambda _x: counted.__setitem__(0, counted[0] + 1))

    multigrid = _preconditioned_iterations(system, mesh)
    assert multigrid * 10 < counted[0], (
        f"multigrid took {multigrid} iterations against Jacobi's {counted[0]}; "
        "the whole point is that this is an order of magnitude"
    )


# --------------------------------------------------------------------------- #
# In the solver
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", MG_SOLVERS)
def test_a_step_stays_divergence_free(kind):
    cfg = make_config(nx=32, ny=32, pressure_solver=kind, poisson_tol=1e-13,
                      boundary_config=walls(top=(BCKind.MOVING_WALL, 1.0)))
    solver = ProjectionSolver(cfg)
    fields = solver.initialize()
    for _ in range(5):
        fields = solver.step(fields, 1e-3)
    assert solver.max_divergence(fields) < DIVERGENCE_TOL
    assert fields.is_finite()


def test_the_projection_reuses_the_previous_pressure():
    """Warm starting is wired up, not merely available on the solver."""
    cfg = make_config(nx=32, ny=32, pressure_solver=PressureSolver.MGCG,
                      boundary_config=walls(top=(BCKind.MOVING_WALL, 1.0)))
    solver = ProjectionSolver(cfg)
    assert solver._p_guess is None
    fields = solver.step(solver.initialize(), 1e-3)
    assert solver._p_guess is not None
    first = solver.pressure_solver.last_iterations
    for _ in range(4):
        fields = solver.step(fields, 1e-3)
    assert solver.pressure_solver.last_iterations <= first


def test_a_direct_solver_keeps_no_pressure_guess():
    """Nothing would read it, and the field is large."""
    cfg = make_config(nx=32, ny=32, pressure_solver=PressureSolver.DIRECT,
                      boundary_config=walls(top=(BCKind.MOVING_WALL, 1.0)))
    solver = ProjectionSolver(cfg)
    solver.step(solver.initialize(), 1e-3)
    assert solver._p_guess is None


def test_a_pressure_outlet_is_still_anchored_under_multigrid():
    cfg = make_config(
        nx=32, ny=32, lx=4.0, ly=4.0, dt=1e-3,
        pressure_solver=PressureSolver.MGCG, poisson_tol=1e-13,
        boundary_config={
            "left": BCSpec(BCKind.INLET, velocity=1.0),
            "right": BCSpec(BCKind.PRESSURE_OUTLET, p_ref=0.75),
            "bottom": BCSpec(BCKind.SYMMETRY),
            "top": BCSpec(BCKind.SYMMETRY),
        },
    )
    solver = ProjectionSolver(cfg)
    fields = solver.initialize(u_init=1.0)
    for _ in range(5):
        fields = solver.step(fields, 1e-3)
    nx, ny = solver.mesh.shape
    face = 0.5 * (fields.p[nx, 1:ny + 1] + fields.p[nx + 1, 1:ny + 1])
    assert np.abs(face - 0.75).max() < 1e-8


def test_mg_sweeps_must_be_at_least_one():
    with pytest.raises(ValueError, match="mg_sweeps"):
        make_config(nx=16, ny=16, mg_sweeps=0)


def test_the_cli_passes_the_sweep_count_through():
    from pycfd.main import build_parser

    args = build_parser().parse_args(
        ["--case", "cavity", "--pressure-solver", "mgcg", "--mg-sweeps", "2"]
    )
    assert args.pressure_solver == "mgcg"
    assert args.mg_sweeps == 2
