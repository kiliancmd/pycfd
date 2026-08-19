"""Reference solutions, published data and error metrics.

These tests guard the *yardsticks*.  A silently wrong analytical formula or a
mistyped reference table would make every downstream benchmark meaningless, so
the exact solutions are verified against the governing equations themselves
rather than taken on trust.
"""

import numpy as np
import pytest

from pycfd.analysis.export import (
    export_csv,
    export_profile_csv,
    export_vtk,
    load_checkpoint,
    save_checkpoint,
)
from pycfd.analysis.postprocess import (
    Probe,
    enstrophy,
    force_coefficients,
    kinetic_energy,
    sample_at,
    stream_function,
    strouhal_number,
    vorticity,
)
from pycfd.analysis.validation import (
    GHIA_REYNOLDS,
    GHIA_U,
    GHIA_V,
    GHIA_X,
    GHIA_Y,
    convergence_order,
    ghia_reference,
    l2_error,
    linf_error,
    poiseuille_profile,
    poiseuille_u_max,
    taylor_green,
)
from pycfd.config import BCKind, SimulationConfig
from pycfd.core.solver import ProjectionSolver

from .conftest import periodic_walls, walls, make_config


# --------------------------------------------------------------------------- #
# Analytical solutions
# --------------------------------------------------------------------------- #
def test_poiseuille_profile_satisfies_its_boundary_conditions():
    h, u_max = 2.0, 3.0
    y = np.linspace(0, h, 101)
    u = poiseuille_profile(y, h, u_max)
    assert u[0] == pytest.approx(0.0)
    assert u[-1] == pytest.approx(0.0)
    assert u.max() == pytest.approx(u_max)
    assert u[len(u) // 2] == pytest.approx(u_max)          # peak on the centreline
    assert np.allclose(u, u[::-1])                          # symmetric


def test_poiseuille_profile_solves_the_momentum_balance():
    """``nu u'' = dp/dx`` must hold for the profile and the amplitude formula."""
    h, nu, dpdx = 1.0, 0.1, -0.8
    u_max = poiseuille_u_max(dpdx, h, nu)
    y = np.linspace(0, h, 2001)
    u = poiseuille_profile(y, h, u_max)
    d2u = np.gradient(np.gradient(u, y), y)
    assert np.allclose(nu * d2u[5:-5], dpdx, rtol=1e-6)


def test_poiseuille_amplitude_matches_the_driving_force():
    # A body force f per unit mass is equivalent to dp/dx = -f.
    assert poiseuille_u_max(-0.8, 1.0, 0.1) == pytest.approx(1.0)


def test_taylor_green_is_divergence_free():
    x = np.linspace(0, 2 * np.pi, 400)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = taylor_green(X, Y, 0.3, 0.05, "u")
    v = taylor_green(X, Y, 0.3, 0.05, "v")
    dudx = np.gradient(u, x, axis=0)
    dvdy = np.gradient(v, x, axis=1)
    # Interior only: np.gradient switches to one-sided stencils at the array
    # edges, so the two terms stop cancelling there for numerical reasons alone.
    assert np.abs((dudx + dvdy)[2:-2, 2:-2]).max() < 1e-12


def test_taylor_green_solves_the_momentum_equation():
    """Check the full nonlinear residual, not just the decay factor."""
    nu, t = 0.05, 0.25
    x = np.linspace(0, 2 * np.pi, 600)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = taylor_green(X, Y, t, nu, "u")
    v = taylor_green(X, Y, t, nu, "v")
    p = taylor_green(X, Y, t, nu, "p")

    dudt = -2.0 * nu * u                       # exact time derivative
    dudx = np.gradient(u, x, axis=0)
    dudy = np.gradient(u, x, axis=1)
    lap_u = (np.gradient(dudx, x, axis=0) + np.gradient(dudy, x, axis=1))
    dpdx = np.gradient(p, x, axis=0)

    residual = dudt + u * dudx + v * dudy + dpdx - nu * lap_u
    assert np.abs(residual[3:-3, 3:-3]).max() < 1e-4


def test_taylor_green_decays_at_the_expected_rate():
    nu = 0.05
    x = np.linspace(0, 2 * np.pi, 64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u0 = taylor_green(X, Y, 0.0, nu, "u")
    u1 = taylor_green(X, Y, 1.0, nu, "u")
    assert np.allclose(u1, u0 * np.exp(-2.0 * nu))


def test_taylor_green_rejects_an_unknown_component():
    with pytest.raises(ValueError, match="component"):
        taylor_green(np.zeros(3), np.zeros(3), 0.0, 0.1, "w")


# --------------------------------------------------------------------------- #
# Ghia reference data
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("re", GHIA_REYNOLDS)
def test_ghia_tables_are_well_formed(re):
    data = ghia_reference(re)
    assert data["u"].shape == GHIA_Y.shape
    assert data["v"].shape == GHIA_X.shape
    # Sample coordinates are sorted and span the full cavity.
    assert np.all(np.diff(GHIA_Y) > 0)
    assert np.all(np.diff(GHIA_X) > 0)
    assert (GHIA_Y[0], GHIA_Y[-1]) == (0.0, 1.0)
    assert (GHIA_X[0], GHIA_X[-1]) == (0.0, 1.0)


@pytest.mark.parametrize("re", GHIA_REYNOLDS)
def test_ghia_data_satisfies_the_cavity_boundary_conditions(re):
    data = ghia_reference(re)
    assert data["u"][0] == 0.0        # no slip on the floor
    assert data["u"][-1] == 1.0       # lid speed
    assert data["v"][0] == 0.0        # no slip on the left wall
    assert data["v"][-1] == 0.0       # no slip on the right wall


@pytest.mark.parametrize("re", GHIA_REYNOLDS)
def test_ghia_profiles_show_the_expected_recirculation(re):
    data = ghia_reference(re)
    # The lid drags fluid forward; the return flow below it is negative.
    assert np.nanmin(data["u"]) < -0.15
    # Down-flow near the right wall, up-flow near the left one.
    assert np.nanmin(data["v"]) < -0.2
    assert np.nanmax(data["v"]) > 0.15


def test_ghia_boundary_layers_thin_as_reynolds_number_rises():
    """A physical consistency check across the three tabulated Reynolds numbers."""
    peaks = [GHIA_X[np.nanargmin(GHIA_V[re])] for re in (100, 400, 1000)]
    assert peaks == sorted(peaks)          # the v-minimum moves toward the wall
    depths = [abs(np.nanmin(GHIA_V[re])) for re in (100, 400, 1000)]
    assert depths == sorted(depths)        # and deepens


def test_unknown_reynolds_number_is_rejected():
    with pytest.raises(ValueError, match="no Ghia"):
        ghia_reference(250)


# --------------------------------------------------------------------------- #
# Error metrics
# --------------------------------------------------------------------------- #
def test_error_norms_are_zero_for_identical_inputs():
    a = np.array([1.0, -2.0, 3.5])
    assert l2_error(a, a) == 0.0
    assert linf_error(a, a) == 0.0


def test_error_norms_match_their_definitions():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 0.0])
    assert linf_error(a, b) == pytest.approx(3.0)
    assert l2_error(a, b) == pytest.approx(np.sqrt(9.0 / 3.0))
    assert l2_error(a, b, relative=True) == pytest.approx(
        np.sqrt(3.0) / np.sqrt(np.mean(b ** 2))
    )


def test_error_norms_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        l2_error(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="shape mismatch"):
        linf_error(np.zeros(3), np.zeros(4))


def test_convergence_order_recovers_a_known_rate():
    n = [16, 32, 64, 128]
    errors = [1.0 / k ** 2 for k in n]
    study = convergence_order(n, errors)
    assert study.observed_order == pytest.approx(2.0)
    assert all(o == pytest.approx(2.0) for o in study.orders)
    assert "order" in study.table()


def test_convergence_order_handles_non_doubling_refinement():
    study = convergence_order([10, 30], [1.0, 1.0 / 9.0])
    assert study.observed_order == pytest.approx(2.0)


def test_convergence_order_needs_at_least_two_grids():
    with pytest.raises(ValueError, match="at least two"):
        convergence_order([16], [1.0])


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def _solid_body_rotation(nx=32):
    """Field with uniform vorticity: u = -omega*y, v = omega*x about the centre."""
    cfg = make_config(nx=nx, ny=nx, lx=2.0, ly=2.0, re=100,
                           boundary_config=periodic_walls())
    solver = ProjectionSolver(cfg)
    Xu, Yu = solver.mesh.u_grid()
    Xv, Yv = solver.mesh.v_grid()
    return solver, -(Yu - 1.0), (Xv - 1.0)


def test_vorticity_of_solid_body_rotation_is_uniform():
    solver, u, v = _solid_body_rotation()
    # Build the field directly: the projection would fight the periodic wrap here.
    fields = solver.initialize()
    fields.u_phys[...] = u
    fields.v_phys[...] = v
    w = vorticity(fields)
    assert np.allclose(w[2:-2, 2:-2], 2.0)      # omega = dv/dx - du/dy = 2


def test_stream_function_recovers_a_uniform_shear():
    cfg = make_config(nx=16, ny=16, re=100, boundary_config=periodic_walls())
    solver = ProjectionSolver(cfg)
    fields = solver.initialize(u_init=1.0)
    psi = stream_function(fields)
    # u = dpsi/dy with u = 1 gives psi = y exactly.
    expected = np.tile(solver.mesh.yf, (solver.mesh.nx + 1, 1))
    assert np.abs(psi - expected).max() < 1e-13


def test_kinetic_energy_and_enstrophy_of_known_fields():
    cfg = make_config(nx=16, ny=16, re=100, boundary_config=periodic_walls())
    solver = ProjectionSolver(cfg)
    fields = solver.initialize(u_init=1.0, v_init=0.0)
    assert kinetic_energy(fields) == pytest.approx(0.5)     # 0.5 * 1^2 * area
    assert enstrophy(fields) == pytest.approx(0.0, abs=1e-20)


def test_force_coefficients_use_the_standard_normalisation():
    cd, cl = force_coefficients((2.0, 1.0), u_ref=2.0, l_ref=0.5)
    assert cd == pytest.approx(2.0 / (0.5 * 4.0 * 0.5))
    assert cl == pytest.approx(1.0 / (0.5 * 4.0 * 0.5))
    with pytest.raises(ValueError, match="positive"):
        force_coefficients((1.0, 0.0), u_ref=0.0, l_ref=1.0)


def test_strouhal_number_recovers_a_known_frequency():
    freq = 0.17
    t = np.linspace(0, 60, 3000)
    st = strouhal_number(t, np.sin(2 * np.pi * freq * t), l_ref=1.0, u_ref=1.0)
    assert st == pytest.approx(freq, rel=0.03)


def test_strouhal_number_is_nan_for_a_short_record():
    assert np.isnan(strouhal_number(np.arange(4.0), np.zeros(4), 1.0, 1.0))


def test_point_probe_interpolates_and_records():
    cfg = make_config(nx=16, ny=16, re=100, boundary_config=periodic_walls())
    solver = ProjectionSolver(cfg)
    fields = solver.initialize(u_init=2.0, v_init=-1.0)

    sample = sample_at(fields, 0.5, 0.5)
    assert sample["u"] == pytest.approx(2.0)
    assert sample["v"] == pytest.approx(-1.0)

    probe = Probe(0.5, 0.5, "centre")
    probe.record(fields)
    probe.record(fields)
    arrays = probe.as_arrays()
    assert arrays["u"].shape == (2,)
    assert np.allclose(arrays["u"], 2.0)


# --------------------------------------------------------------------------- #
# Export round-trips
# --------------------------------------------------------------------------- #
def test_vtk_and_csv_exports_are_well_formed(tmp_path):
    cfg = make_config(nx=8, ny=6, re=100,
                           boundary_config=walls(top=(BCKind.MOVING_WALL, 1.0)))
    solver = ProjectionSolver(cfg)
    fields = solver.step(solver.initialize(), 1e-3)

    vtk = export_vtk(fields, tmp_path / "f.vtk")
    text = vtk.read_text()
    assert text.startswith("# vtk DataFile Version")
    assert "RECTILINEAR_GRID" in text
    assert "DIMENSIONS 8 6 1" in text
    assert f"POINT_DATA {8 * 6}" in text
    assert "VECTORS velocity float" in text

    csv = export_csv(fields, tmp_path / "f.csv")
    lines = csv.read_text().splitlines()
    assert lines[0] == "x,y,u,v,speed,pressure,vorticity"
    assert len(lines) == 8 * 6 + 1


def test_profile_csv_requires_equal_column_lengths(tmp_path):
    export_profile_csv(tmp_path / "p.csv", {"y": np.arange(3.0), "u": np.arange(3.0)})
    with pytest.raises(ValueError, match="equal length"):
        export_profile_csv(tmp_path / "q.csv", {"y": np.arange(3.0), "u": np.arange(4.0)})


def test_checkpoint_round_trip_is_exact(tmp_path):
    cfg = make_config(nx=12, ny=10, re=250, dt=3e-3,
                           boundary_config=walls(top=(BCKind.MOVING_WALL, 1.0)))
    solver = ProjectionSolver(cfg)
    fields = solver.step(solver.initialize(), 3e-3)

    path = save_checkpoint(fields, cfg, tmp_path / "ck")
    restored, restored_cfg = load_checkpoint(path)

    assert np.array_equal(restored.u, fields.u)
    assert np.array_equal(restored.v, fields.v)
    assert np.array_equal(restored.p, fields.p)
    assert restored.t == fields.t and restored.step == fields.step
    assert restored_cfg == cfg
    assert restored_cfg.boundary_config["top"].velocity == 1.0


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.npz")


def test_withheld_reference_points_are_skipped_by_the_norms():
    """A reference entry stored as nan must not poison the whole error norm."""
    from pycfd.analysis.validation import GHIA_KNOWN_GAPS

    assert GHIA_KNOWN_GAPS, "the withheld-entry registry should not be empty"
    for (re, component, index), _ in GHIA_KNOWN_GAPS.items():
        table = ghia_reference(re)[component]
        assert np.isnan(table[index])

    reference = np.array([1.0, np.nan, 3.0])
    numerical = np.array([1.0, 99.0, 3.0])
    assert l2_error(numerical, reference) == 0.0
    assert linf_error(numerical, reference) == 0.0


def test_all_undefined_reference_raises():
    with pytest.raises(ValueError, match="no comparable points"):
        l2_error(np.zeros(3), np.full(3, np.nan))
