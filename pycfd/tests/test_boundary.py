"""Boundary conditions: every type gets a test, and every test checks the
*exact* value that was prescribed.

The staggered layout treats the three variables differently at a wall (see
:mod:`pycfd.core.boundary`), so these tests assert on the specific quantity each
condition actually controls: the face value for the normal component, the
half-sum of ghost and interior for the tangential one.
"""

import numpy as np
import pytest

from pycfd.config import BCKind, BCSpec, SimulationConfig
from pycfd.core.boundary import BoundaryManager, wall_index
from pycfd.core.fields import FlowField
from pycfd.core.mesh import StructuredMesh
from pycfd.core.solver import ProjectionSolver

from .conftest import periodic_walls, walls, make_config

NX = NY = 12


@pytest.fixture
def mesh():
    return StructuredMesh(NX, NY)


def noisy_field(mesh, seed=0):
    """A field of random values, so a condition cannot pass by acting on zeros."""
    rng = np.random.default_rng(seed)
    return FlowField(
        mesh,
        rng.standard_normal(mesh.u_shape),
        rng.standard_normal(mesh.v_shape),
        rng.standard_normal(mesh.p_shape),
    )


# Where two walls meet, the corner value belongs to whichever condition is
# applied last -- a genuine ambiguity of the staggered layout, not a defect.
# These helpers therefore return only the span a wall unambiguously owns.
def tangential_wall_value(field, wall, mesh):
    """The tangential velocity the wall actually sees: mean of ghost and interior."""
    wi = wall_index(wall, mesh)
    tang = field.v if wi.axis == 0 else field.u
    if wi.axis == 0:
        ghost, inner = tang[wi.t_ghost, 2:NY + 1], tang[wi.t_in, 2:NY + 1]
    else:
        ghost, inner = tang[2:NX + 1, wi.t_ghost], tang[2:NX + 1, wi.t_in]
    return 0.5 * (ghost + inner)


def normal_face_value(field, wall, mesh):
    """The normal velocity on the boundary face itself, excluding ghost rows."""
    wi = wall_index(wall, mesh)
    norm = field.u if wi.axis == 0 else field.v
    return (norm[wi.n_face, 1:NY + 1] if wi.axis == 0
            else norm[1:NX + 1, wi.n_face])


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("wall", ["left", "right", "bottom", "top"])
def test_no_slip_zeroes_both_components_on_every_wall(mesh, wall):
    bm = BoundaryManager(walls(**{wall: BCKind.NO_SLIP}), mesh)
    f = noisy_field(mesh)
    bm.apply_velocity(f)

    assert np.all(normal_face_value(f, wall, mesh) == 0.0)
    assert np.abs(tangential_wall_value(f, wall, mesh)).max() < 1e-15


@pytest.mark.parametrize("wall,speed", [("top", 1.0), ("bottom", -2.5),
                                        ("left", 0.75), ("right", 3.0)])
def test_moving_wall_imposes_the_tangential_speed_exactly(mesh, wall, speed):
    bm = BoundaryManager(walls(**{wall: (BCKind.MOVING_WALL, speed)}), mesh)
    f = noisy_field(mesh, seed=1)
    bm.apply_velocity(f)

    assert np.allclose(tangential_wall_value(f, wall, mesh), speed, atol=1e-15)
    assert np.all(normal_face_value(f, wall, mesh) == 0.0)


def test_uniform_inlet_prescribes_the_normal_velocity(mesh):
    speed = 1.7
    bm = BoundaryManager(
        walls(left=BCSpec(BCKind.INLET, velocity=speed), right=BCKind.OUTLET), mesh
    )
    f = noisy_field(mesh, seed=2)
    bm.apply_velocity(f)

    # Inflow points into the domain, i.e. +x on the left wall.
    assert np.allclose(normal_face_value(f, "left", mesh), speed)
    assert np.abs(tangential_wall_value(f, "left", mesh)).max() < 1e-15


def test_parabolic_inlet_matches_the_analytical_profile(mesh):
    peak = 2.0
    bm = BoundaryManager(
        walls(left=BCSpec(BCKind.INLET, velocity=peak, profile="parabolic"),
              right=BCKind.OUTLET), mesh
    )
    f = noisy_field(mesh, seed=3)
    bm.apply_velocity(f)

    got = normal_face_value(f, "left", mesh)
    expected = peak * 4.0 * mesh.yc * (mesh.ly - mesh.yc) / mesh.ly ** 2
    assert np.allclose(got, expected)
    assert got.max() == pytest.approx(peak, rel=1e-2)


def test_inlet_direction_flips_with_the_wall(mesh):
    """The same spec must push inward on either side of the domain."""
    speed = 1.0
    for wall, sign in (("left", +1.0), ("right", -1.0)):
        other = "right" if wall == "left" else "left"
        bm = BoundaryManager(
            walls(**{wall: BCSpec(BCKind.INLET, velocity=speed), other: BCKind.OUTLET}),
            mesh,
        )
        f = noisy_field(mesh, seed=4)
        bm.apply_velocity(f)
        assert np.allclose(normal_face_value(f, wall, mesh), sign * speed)


def test_outlet_is_zero_gradient(mesh):
    bm = BoundaryManager(
        walls(left=BCSpec(BCKind.INLET, velocity=1.0), right=BCKind.OUTLET), mesh
    )
    f = noisy_field(mesh, seed=5)
    bm.apply_velocity(f, predictor=True)

    wi = wall_index("right", mesh)
    assert np.allclose(f.u[wi.n_face, :], f.u[wi.n_in, :])       # du/dn = 0
    assert np.allclose(f.v[wi.t_ghost, :], f.v[wi.t_in, :])      # dv/dn = 0


def test_outlet_normal_is_frozen_after_the_projection(mesh):
    """Re-deriving the outlet velocity post-projection would reintroduce divergence."""
    bm = BoundaryManager(
        walls(left=BCSpec(BCKind.INLET, velocity=1.0), right=BCKind.OUTLET), mesh
    )
    f = noisy_field(mesh, seed=6)
    bm.apply_velocity(f, predictor=True)

    wi = wall_index("right", mesh)
    f.u[wi.n_in, :] += 10.0          # pretend the projection changed the interior
    frozen = f.u[wi.n_face, :].copy()
    bm.apply_velocity(f, predictor=False)
    assert np.array_equal(f.u[wi.n_face, :], frozen)


@pytest.mark.parametrize("wall", ["left", "right", "bottom", "top"])
def test_symmetry_blocks_flow_but_lets_the_tangential_component_slip(mesh, wall):
    bm = BoundaryManager(walls(**{wall: BCKind.SYMMETRY}), mesh)
    f = noisy_field(mesh, seed=7)
    bm.apply_velocity(f)

    wi = wall_index(wall, mesh)
    assert np.all(normal_face_value(f, wall, mesh) == 0.0)
    tang = f.v if wi.axis == 0 else f.u
    if wi.axis == 0:
        ghost, inner = tang[wi.t_ghost, 2:NY + 1], tang[wi.t_in, 2:NY + 1]
    else:
        ghost, inner = tang[2:NX + 1, wi.t_ghost], tang[2:NX + 1, wi.t_in]
    assert np.allclose(ghost, inner)              # zero normal gradient
    assert np.abs(inner).max() > 1e-3             # and it is genuinely non-zero


def test_periodic_wraps_faces_and_ghosts(mesh):
    bm = BoundaryManager(periodic_walls(), mesh)
    f = noisy_field(mesh, seed=8)
    bm.apply_velocity(f)
    bm.apply_pressure(f)
    u, v, p = f.u, f.v, f.p

    # The far-end face is the same physical face as the near-end one.
    assert np.array_equal(u[NX + 1, :], u[1, :])
    assert np.array_equal(v[:, NY + 1], v[:, 1])
    # Ghosts take the values from the opposite end of the domain.
    assert np.array_equal(u[0, :], u[NX, :])
    assert np.array_equal(u[NX + 2, :], u[2, :])
    assert np.array_equal(v[0, :], v[NX, :])
    assert np.array_equal(v[NX + 1, :], v[1, :])
    assert np.array_equal(p[0, :], p[NX, :])
    assert np.array_equal(p[NX + 1, :], p[1, :])
    assert np.array_equal(p[:, 0], p[:, NY])
    assert np.array_equal(p[:, NY + 1], p[:, 1])


def test_pressure_ghosts_are_neumann_at_solid_walls(mesh):
    bm = BoundaryManager(walls(), mesh)
    f = noisy_field(mesh, seed=9)
    bm.apply_pressure(f)
    assert np.array_equal(f.p[0, :], f.p[1, :])
    assert np.array_equal(f.p[NX + 1, :], f.p[NX, :])
    assert np.array_equal(f.p[:, 0], f.p[:, 1])
    assert np.array_equal(f.p[:, NY + 1], f.p[:, NY])


# --------------------------------------------------------------------------- #
def test_global_mass_balance_makes_outflow_match_inflow(mesh):
    bm = BoundaryManager(
        walls(left=BCSpec(BCKind.INLET, velocity=1.0), right=BCKind.OUTLET), mesh
    )
    f = noisy_field(mesh, seed=10)
    bm.apply_velocity(f, predictor=True)
    bm.enforce_global_mass_balance(f)

    inflow = f.u[1, 1:NY + 1].sum() * mesh.dy
    outflow = f.u[NX + 1, 1:NY + 1].sum() * mesh.dy
    assert outflow == pytest.approx(inflow, rel=1e-12)


def test_mass_balance_handles_a_quiescent_start(mesh):
    """With no outflow yet, the correction must be additive, not a division by zero."""
    bm = BoundaryManager(
        walls(left=BCSpec(BCKind.INLET, velocity=1.0), right=BCKind.OUTLET), mesh
    )
    f = FlowField.zeros(mesh)
    bm.apply_velocity(f, predictor=True)
    bm.enforce_global_mass_balance(f)

    inflow = f.u[1, 1:NY + 1].sum() * mesh.dy
    outflow = f.u[NX + 1, 1:NY + 1].sum() * mesh.dy
    assert np.isfinite(outflow)
    assert outflow == pytest.approx(inflow, rel=1e-12)


def test_boundary_conditions_survive_a_full_time_step():
    """The prescribed values must still hold exactly after predictor and projection."""
    cfg = make_config(
        nx=16, ny=16, re=100, dt=1e-3,
        boundary_config=walls(top=(BCKind.MOVING_WALL, 1.0),
                              left=BCKind.SYMMETRY),
    )
    solver = ProjectionSolver(cfg)
    fields = solver.initialize()
    for _ in range(5):
        fields = solver.step(fields, 1e-3)

    rows, cols = slice(1, cfg.ny + 1), slice(1, cfg.nx + 1)
    assert np.all(fields.u[1, rows] == 0.0)                    # symmetry, no through-flow
    assert np.all(fields.u[cfg.nx + 1, rows] == 0.0)           # no-slip right
    assert np.all(fields.v[cols, 1] == 0.0)                    # no-slip bottom
    assert np.all(fields.v[cols, cfg.ny + 1] == 0.0)           # lid: no penetration
    lid = 0.5 * (fields.u[2:cfg.nx + 1, cfg.ny] + fields.u[2:cfg.nx + 1, cfg.ny + 1])
    assert np.allclose(lid, 1.0, atol=1e-14)                   # lid speed exactly


def test_unknown_wall_name_is_rejected(mesh):
    with pytest.raises(ValueError, match="unknown wall"):
        wall_index("diagonal", mesh)


def test_half_periodic_axis_is_rejected():
    with pytest.raises(ValueError, match="periodicity"):
        make_config(nx=8, ny=8, boundary_config=walls(left=BCKind.PERIODIC))
