"""Poisson assembly and linear-solver correctness.

The pressure solve is where an incompressible code most often goes quietly
wrong, so these tests check the operator's algebraic structure (symmetry, null
space, mask consistency) as well as the numbers it produces.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pycfd.config import BCKind, BCSpec, PressureSolver
from pycfd.core.mesh import StructuredMesh
from pycfd.core.pressure import (
    DirectSolver,
    SORSolver,
    assemble_poisson_matrix,
    make_pressure_solver,
)
from pycfd.core.solver import ProjectionSolver

from .conftest import make_config

ALL_SOLVERS = ("direct", "cg", "sor", "jacobi")


@pytest.fixture
def unit_mesh():
    return StructuredMesh(16, 16)


# --------------------------------------------------------------------------- #
# Algebraic structure
# --------------------------------------------------------------------------- #
def test_operator_is_symmetric(unit_mesh):
    A = assemble_poisson_matrix(unit_mesh).A
    assert abs(A - A.T).max() == 0.0


def _row_sum_residual(A) -> float:
    """Largest row sum of ``A``, relative to its largest entry.

    Vanishing row sums are exactly the statement that no flux crosses the
    boundary.  The cancellation is only exact to round-off: the diagonal is
    accumulated face by face while the off-diagonals are each a single
    ``1/h**2``, and those agree bit-for-bit only when ``1/h**2`` happens to be a
    power of two.
    """
    scale = np.abs(A).max()
    return float(np.abs(np.asarray(A.sum(axis=1))).max() / scale)


def test_neumann_operator_annihilates_constants(unit_mesh):
    """Zero row sums are exactly the statement that dp/dn = 0 on every wall."""
    system = assemble_poisson_matrix(unit_mesh)
    assert _row_sum_residual(system.A) < 1e-14
    ones = np.ones(system.n)
    assert np.abs(system.A @ ones).max() / np.abs(system.A).max() < 1e-14
    assert system.singular


def test_periodic_operator_is_symmetric_and_singular():
    mesh = StructuredMesh(12, 12, 2 * np.pi, 2 * np.pi)
    system = assemble_poisson_matrix(mesh, periodic_x=True, periodic_y=True)
    assert abs(system.A - system.A.T).max() == 0.0
    assert _row_sum_residual(system.A) < 1e-14
    # Periodic wrapping adds the corner couplings that Neumann walls lack.
    neumann = assemble_poisson_matrix(mesh)
    assert system.A.nnz > neumann.A.nnz


def test_interior_stencil_is_the_five_point_laplacian():
    mesh = StructuredMesh(8, 8, 1.0, 1.0)
    A = assemble_poisson_matrix(mesh).A.tolil()
    h2 = mesh.dx ** 2
    k = 3 * mesh.ny + 3                       # a cell well inside the domain
    row = dict(zip(A.rows[k], A.data[k]))
    assert row.pop(k) == pytest.approx(-4.0 / h2)
    assert sorted(row.values()) == pytest.approx([1.0 / h2] * 4)
    assert set(row) == {k - 1, k + 1, k - mesh.ny, k + mesh.ny}


def test_anisotropic_spacing_enters_the_stencil():
    mesh = StructuredMesh(8, 16, 2.0, 1.0)      # dx = 0.25, dy = 0.0625
    A = assemble_poisson_matrix(mesh).A.tolil()
    k = 3 * mesh.ny + 5
    row = dict(zip(A.rows[k], A.data[k]))
    assert row[k + 1] == pytest.approx(1.0 / mesh.dy ** 2)
    assert row[k + mesh.ny] == pytest.approx(1.0 / mesh.dx ** 2)


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ALL_SOLVERS)
def test_solvers_reproduce_a_manufactured_solution(unit_mesh, kind):
    """``p = cos(pi x) cos(pi y)`` satisfies dp/dn = 0 on the unit box."""
    system = assemble_poisson_matrix(unit_mesh)
    X, Y = unit_mesh.cell_center_grid()
    exact = np.cos(np.pi * X) * np.cos(np.pi * Y)

    solver = make_pressure_solver(PressureSolver(kind), system, tol=1e-11, maxiter=200_000)
    p = solver.solve(-2.0 * np.pi ** 2 * exact)

    # Only the discretisation error remains; the mean is arbitrary and removed.
    assert np.abs(p - (exact - exact.mean())).max() < 5e-3
    assert solver.last_residual < 1e-9


def test_second_order_discretisation():
    errors = []
    for n in (16, 32, 64):
        mesh = StructuredMesh(n, n)
        X, Y = mesh.cell_center_grid()
        exact = np.cos(np.pi * X) * np.cos(np.pi * Y)
        solver = DirectSolver(assemble_poisson_matrix(mesh))
        p = solver.solve(-2.0 * np.pi ** 2 * exact)
        errors.append(np.abs(p - (exact - exact.mean())).max())

    for coarse, fine in zip(errors, errors[1:]):
        assert np.log2(coarse / fine) > 1.9


def test_all_solvers_agree_with_the_direct_one(unit_mesh):
    system = assemble_poisson_matrix(unit_mesh)
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal(system.shape)
    rhs -= rhs.mean()                      # make it compatible

    reference = DirectSolver(system).solve(rhs)
    for kind in ("cg", "sor", "jacobi"):
        p = make_pressure_solver(PressureSolver(kind), system,
                                 tol=1e-12, maxiter=200_000).solve(rhs)
        assert np.abs(p - reference).max() < 1e-8


def test_solution_has_zero_mean_and_solves_the_system(unit_mesh):
    system = assemble_poisson_matrix(unit_mesh)
    rng = np.random.default_rng(1)
    rhs = rng.standard_normal(system.shape)
    p = DirectSolver(system).solve(rhs)

    assert p.mean() == pytest.approx(0.0, abs=1e-12)
    # The compatible part of the right-hand side is reproduced exactly.
    residual = (system.A @ p.ravel()) - (rhs.ravel() - rhs.mean())
    assert np.abs(residual).max() < 1e-9


def test_incompatible_rhs_is_projected_not_rejected(unit_mesh):
    """A constant right-hand side has no solution; the mean must be removed."""
    system = assemble_poisson_matrix(unit_mesh)
    p = DirectSolver(system).solve(np.ones(system.shape))
    assert np.abs(p).max() < 1e-12


# --------------------------------------------------------------------------- #
# Obstacle masking
# --------------------------------------------------------------------------- #
def test_solid_cells_are_decoupled_and_trivial():
    mesh = StructuredMesh(10, 10)
    solid = np.zeros((10, 10), dtype=bool)
    solid[4:6, 4:6] = True
    system = assemble_poisson_matrix(mesh, solid_mask=solid)
    A = system.A.tolil()

    for i, j in [(4, 4), (5, 5)]:
        k = i * mesh.ny + j
        assert A.rows[k] == [k]
        assert A.data[k] == [1.0]
        assert not system.fluid[k]

    # A fluid cell adjacent to the block loses exactly the face it shares with it.
    k = 3 * mesh.ny + 4                     # cell (3, 4), solid neighbour at (4, 4)
    row = dict(zip(A.rows[k], A.data[k]))
    assert (4 * mesh.ny + 4) not in row
    assert row[k] == pytest.approx(-3.0 / mesh.dx ** 2)


def test_masked_operator_still_annihilates_constants_on_the_fluid():
    mesh = StructuredMesh(12, 12)
    solid = np.zeros((12, 12), dtype=bool)
    solid[5:8, 5:8] = True
    system = assemble_poisson_matrix(mesh, solid_mask=solid)

    x = np.zeros(system.n)
    x[system.fluid] = 1.0
    residual = np.abs((system.A @ x)[system.fluid]).max() / np.abs(system.A).max()
    assert residual < 1e-14


def test_enclosed_fluid_cell_is_made_trivial():
    """A cell walled off on all four sides would otherwise leave a zero row."""
    mesh = StructuredMesh(8, 8)
    solid = np.zeros((8, 8), dtype=bool)
    solid[3, 4] = solid[5, 4] = solid[4, 3] = solid[4, 5] = True
    system = assemble_poisson_matrix(mesh, solid_mask=solid)
    k = 4 * mesh.ny + 4
    assert not system.fluid[k]
    assert system.A[k, k] == 1.0


# --------------------------------------------------------------------------- #
# Solver-specific behaviour
# --------------------------------------------------------------------------- #
def test_sor_rejects_an_invalid_two_colouring():
    """An odd periodic extent breaks the red/black ordering; it must not go silent."""
    mesh = StructuredMesh(9, 9)
    system = assemble_poisson_matrix(mesh, periodic_x=True, periodic_y=True)
    with pytest.raises(ValueError, match="red/black"):
        SORSolver(system)


def test_iterative_solver_reports_stalling(unit_mesh):
    system = assemble_poisson_matrix(unit_mesh)
    rng = np.random.default_rng(2)
    solver = make_pressure_solver(PressureSolver.JACOBI, system, tol=1e-14, maxiter=5)
    with pytest.raises(RuntimeError, match="stalled"):
        solver.solve(rng.standard_normal(system.shape))


def test_mask_shape_is_validated():
    mesh = StructuredMesh(8, 8)
    with pytest.raises(ValueError, match="shape"):
        assemble_poisson_matrix(mesh, solid_mask=np.zeros((4, 4), dtype=bool))


def test_stretched_mesh_is_refused():
    from pycfd.core.mesh import NonUniformMeshError

    with pytest.raises(NonUniformMeshError):
        assemble_poisson_matrix(StructuredMesh(8, 8, stretch_x=1.1))


# --------------------------------------------------------------------------- #
# Pressure outlet (Dirichlet pressure)
# --------------------------------------------------------------------------- #
def external_flow_config(**kw):
    """Uniform inflow, pressure outlet, slip side walls -- the external-flow setup."""
    return make_config(
        nx=kw.pop("nx", 32), ny=kw.pop("ny", 32), lx=4.0, ly=4.0,
        re=100.0, dt=1e-3, t_end=1e-3,
        boundary_config={
            "left": BCSpec(BCKind.INLET, velocity=1.0),
            "right": BCSpec(BCKind.PRESSURE_OUTLET, p_ref=kw.pop("p_ref", 0.0)),
            "bottom": BCSpec(BCKind.SYMMETRY),
            "top": BCSpec(BCKind.SYMMETRY),
        },
        **kw,
    )


def outlet_face_pressure(fields, mesh):
    """Pressure interpolated onto the outlet face itself.

    The Dirichlet value is imposed on the boundary *face*, which on a staggered
    grid sits half a cell beyond the last cell centre -- the face value is the
    mean of the last interior cell and its ghost.
    """
    nx, ny = mesh.shape
    return 0.5 * (fields.p[nx, 1:ny + 1] + fields.p[nx + 1, 1:ny + 1])


def test_pressure_outlet_anchors_the_face_at_p_ref():
    """After a step, the outlet face must sit at ``p_ref`` to round-off."""
    solver = ProjectionSolver(external_flow_config())
    fields = solver.step(solver.initialize(u_init=1.0), 1e-3)

    face = outlet_face_pressure(fields, solver.mesh)
    assert np.abs(face).max() < 1e-10, (
        f"outlet pressure not anchored: max deviation = {np.abs(face).max():.2e}"
    )


@pytest.mark.parametrize("p_ref", [0.0, 2.5, -1.25])
def test_pressure_outlet_honours_a_non_zero_reference(p_ref):
    solver = ProjectionSolver(external_flow_config(p_ref=p_ref))
    fields = solver.step(solver.initialize(u_init=1.0), 1e-3)
    face = outlet_face_pressure(fields, solver.mesh)
    assert np.abs(face - p_ref).max() < 1e-10


def test_pressure_outlet_makes_the_operator_non_singular():
    """A Dirichlet wall removes the constant null space, so nothing is pinned."""
    mesh = StructuredMesh(16, 16)
    neumann = assemble_poisson_matrix(mesh)
    dirichlet = assemble_poisson_matrix(mesh, dirichlet={"right": 0.0})

    assert neumann.singular and not dirichlet.singular
    # The constant is annihilated by the Neumann operator but not the Dirichlet one.
    ones = np.ones(dirichlet.n)
    assert np.abs(neumann.A @ ones).max() / np.abs(neumann.A).max() < 1e-14
    assert np.abs(dirichlet.A @ ones).max() > 0.0


def test_dirichlet_row_keeps_its_divergence_equation():
    """The outlet row is a real Laplacian row, not an identity row.

    Overwriting it with ``p = p_ref`` would discard the continuity equation of
    the cell next to the outlet, and that column would stop being
    divergence-free.  The condition is imposed through the ghost instead, which
    only changes the diagonal.
    """
    mesh = StructuredMesh(16, 16)
    system = assemble_poisson_matrix(mesh, dirichlet={"right": 0.0})
    A = system.A.tolil()
    h2 = mesh.dx ** 2
    k = 15 * mesh.ny + 5                    # last column, mid height

    row = dict(zip(A.rows[k], A.data[k]))
    # Three interior faces at -1/h^2 each, plus -2/h^2 from the eliminated ghost.
    assert row.pop(k) == pytest.approx(-5.0 / h2)
    assert len(row) == 3, "the outlet row must keep its three interior neighbours"
    assert sorted(row.values()) == pytest.approx([1.0 / h2] * 3)
    assert system.bc_rhs[k] == pytest.approx(0.0)     # p_ref = 0


def test_dirichlet_reference_enters_the_right_hand_side():
    mesh = StructuredMesh(16, 16)
    system = assemble_poisson_matrix(mesh, dirichlet={"right": 3.0})
    k = 15 * mesh.ny + 5
    assert system.bc_rhs[k] == pytest.approx(-2.0 * 3.0 / mesh.dx ** 2)
    # A uniform p_ref everywhere is the exact solution of the homogeneous problem.
    p = DirectSolver(system).solve(np.zeros(mesh.shape))
    assert np.abs(p - 3.0).max() < 1e-10


def test_dirichlet_poisson_is_second_order():
    """``p = cos(pi x/2) cos(pi y)`` satisfies Neumann on three walls, p=0 on the right."""
    kx, ky = np.pi / 2, np.pi
    errors = []
    for n in (16, 32, 64):
        mesh = StructuredMesh(n, n)
        X, Y = mesh.cell_center_grid()
        exact = np.cos(kx * X) * np.cos(ky * Y)
        system = assemble_poisson_matrix(mesh, dirichlet={"right": 0.0})
        p = DirectSolver(system).solve(-(kx ** 2 + ky ** 2) * exact)
        errors.append(np.abs(p - exact).max())
    for coarse, fine in zip(errors, errors[1:]):
        assert np.log2(coarse / fine) > 1.9


@pytest.mark.parametrize("kind", ["direct", "cg", "sor", "jacobi"])
def test_all_solvers_agree_on_the_non_singular_system(kind):
    mesh = StructuredMesh(16, 16)
    system = assemble_poisson_matrix(mesh, dirichlet={"right": 1.0})
    rng = np.random.default_rng(7)
    rhs = rng.standard_normal(system.shape)

    reference = DirectSolver(system).solve(rhs)
    solver = make_pressure_solver(PressureSolver(kind), system,
                                  tol=1e-12, maxiter=300_000)
    assert np.abs(solver.solve(rhs) - reference).max() < 1e-8


def test_inlet_stays_neumann_alongside_a_pressure_outlet():
    """``dp/dn = 0`` at the inlet: the ghost tracks the first interior cell.

    Dirichlet pressure at the inlet as well would fix the level at both ends and
    impose a pressure drop the flow did not ask for.
    """
    solver = ProjectionSolver(external_flow_config())
    fields = solver.initialize(u_init=1.0)
    for _ in range(5):
        fields = solver.step(fields, 1e-3)

    dp_dn = np.abs(fields.p[0, :] - fields.p[1, :])
    assert dp_dn.max() < 1e-12, (
        f"inlet Neumann condition violated: max |dp/dn| = {dp_dn.max():.2e}"
    )
    assert solver.boundaries.dirichlet_pressure() == {"right": 0.0}


def test_pressure_outlet_stays_divergence_free_including_the_outlet_column():
    solver = ProjectionSolver(external_flow_config(nx=24, ny=24))
    fields = solver.initialize(u_init=1.0)
    for _ in range(10):
        fields = solver.step(fields, 1e-3)

    divergence = solver.divergence(fields.u, fields.v)
    assert np.abs(divergence).max() < 1e-11
    # Explicitly including the column adjacent to the outlet.
    assert np.abs(divergence[-1, :]).max() < 1e-11


def test_pressure_outlet_conserves_mass_without_rescaling():
    """Zero divergence everywhere forces net boundary flux to zero, exactly."""
    solver = ProjectionSolver(external_flow_config(nx=24, ny=24))
    assert solver.boundaries._outlets == [], (
        "a pressure outlet must not be rescaled by the global mass balance"
    )
    fields = solver.initialize(u_init=1.0)
    for _ in range(10):
        fields = solver.step(fields, 1e-3)

    nx, ny = solver.mesh.shape
    inflow = fields.u[1, 1:ny + 1].sum() * solver.mesh.dy
    outflow = fields.u[nx + 1, 1:ny + 1].sum() * solver.mesh.dy
    assert outflow == pytest.approx(inflow, rel=1e-12)


def test_outlet_face_is_a_solved_unknown():
    """The projection must own the outlet face; nothing may overwrite it."""
    solver = ProjectionSolver(external_flow_config())
    nx = solver.mesh.nx
    assert solver.u_upd[0].stop == nx + 2, "outlet face missing from the solvable set"

    fields = solver.step(solver.initialize(u_init=1.0), 1e-3)
    before = fields.u[nx + 1, :].copy()
    solver.boundaries.apply_velocity(fields, predictor=False)
    assert np.array_equal(fields.u[nx + 1, :], before)


def test_pressure_outlet_works_on_the_top_wall():
    """The condition is wall-generic, not hard-coded to the right boundary."""
    cfg = make_config(
        nx=24, ny=24, lx=2.0, ly=2.0, re=100.0, dt=1e-3,
        boundary_config={
            "left": BCSpec(BCKind.NO_SLIP),
            "right": BCSpec(BCKind.NO_SLIP),
            "bottom": BCSpec(BCKind.INLET, velocity=1.0),
            "top": BCSpec(BCKind.PRESSURE_OUTLET, p_ref=0.5),
        },
    )
    solver = ProjectionSolver(cfg)
    assert not solver.poisson.singular
    assert solver.v_upd[1].stop == cfg.ny + 2

    fields = solver.initialize(v_init=1.0)
    for _ in range(5):
        fields = solver.step(fields, 1e-3)

    ny = cfg.ny
    face = 0.5 * (fields.p[1:cfg.nx + 1, ny] + fields.p[1:cfg.nx + 1, ny + 1])
    assert np.abs(face - 0.5).max() < 1e-10
    assert solver.max_divergence(fields) < 1e-11
