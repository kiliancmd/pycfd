"""Phase 1 gate check: mesh geometry, spacing and staggered array shapes."""

import numpy as np
import pytest

from pycfd.core.mesh import NonUniformMeshError, StructuredMesh

from .conftest import make_config


@pytest.mark.parametrize("n", [32, 128])
def test_uniform_mesh_shapes_and_bounds(n):
    m = StructuredMesh(n, n, 1.0, 1.0)

    assert m.xc.shape == (n,)
    assert m.yc.shape == (n,)
    assert m.xf.shape == (n + 1,)
    assert m.yf.shape == (n + 1,)

    # Domain is covered exactly, with no round-off leaking into the end faces.
    assert m.xf[0] == 0.0 and m.xf[-1] == 1.0
    assert m.yf[0] == 0.0 and m.yf[-1] == 1.0

    # Cell centres sit strictly inside, half a cell from each end face.
    assert m.xc[0] == pytest.approx(0.5 / n)
    assert m.xc[-1] == pytest.approx(1.0 - 0.5 / n)
    assert np.all(np.diff(m.xc) > 0)
    assert np.all(np.diff(m.yc) > 0)


def test_uniform_spacing_is_exactly_uniform():
    m = StructuredMesh(64, 32, 2.0, 1.0)
    assert m.is_uniform
    assert m.dx == pytest.approx(2.0 / 64)
    assert m.dy == pytest.approx(1.0 / 32)
    assert np.allclose(np.diff(m.xf), m.dx)
    assert np.allclose(np.diff(m.yf), m.dy)
    assert m.cell_area == pytest.approx(m.dx * m.dy)
    assert m.n_cells == 64 * 32


def test_non_square_domain_coordinates():
    m = StructuredMesh(8, 4, 4.0, 1.0)
    assert m.xc[0] == pytest.approx(0.25)
    assert m.xc[-1] == pytest.approx(3.75)
    assert m.yc[0] == pytest.approx(0.125)
    assert m.yc[-1] == pytest.approx(0.875)


def test_staggered_array_shapes_include_one_ghost_layer():
    nx, ny = 16, 12
    m = StructuredMesh(nx, ny)
    # u: nx+1 physical x-faces + 2 ghost columns; ny cell rows + 2 ghost rows.
    assert m.u_shape == (nx + 3, ny + 2)
    assert m.v_shape == (nx + 2, ny + 3)
    assert m.p_shape == (nx + 2, ny + 2)
    assert m.zeros_u().shape == m.u_shape
    assert m.zeros_v().shape == m.v_shape
    assert m.zeros_p().shape == m.p_shape


def test_variable_location_coordinate_generators():
    nx, ny = 8, 8
    m = StructuredMesh(nx, ny)
    xu, yu = m.u_coords()
    xv, yv = m.v_coords()
    xp, yp = m.cell_centers()

    # u sits on x-faces at cell-centre heights; v is the transpose arrangement.
    assert xu.shape == (nx + 1,) and yu.shape == (ny,)
    assert xv.shape == (nx,) and yv.shape == (ny + 1,)
    assert xp.shape == (nx,) and yp.shape == (ny,)
    assert np.allclose(xu, m.xf)
    assert np.allclose(yv, m.yf)

    X, Y = m.cell_center_grid()
    assert X.shape == (nx, ny) and Y.shape == (nx, ny)
    assert X[3, 5] == pytest.approx(m.xc[3])
    assert Y[3, 5] == pytest.approx(m.yc[5])

    Xu, Yu = m.u_grid()
    assert Xu.shape == (nx + 1, ny)
    Xv, Yv = m.v_grid()
    assert Xv.shape == (nx, ny + 1)


def test_geometric_stretching_sums_to_domain_length():
    ratio = 1.05
    m = StructuredMesh(20, 20, 1.0, 2.0, stretch_x=1.0, stretch_y=ratio)

    assert m.dy_cells.sum() == pytest.approx(2.0)
    assert m.yf[-1] == pytest.approx(2.0)
    # Successive cells grow by exactly the requested ratio.
    assert np.allclose(m.dy_cells[1:] / m.dy_cells[:-1], ratio)
    assert not m.is_uniform


def test_stretched_mesh_refuses_uniform_spacing_accessors():
    m = StructuredMesh(10, 10, stretch_x=1.1)
    assert not m.is_uniform
    with pytest.raises(NonUniformMeshError, match="uniform"):
        _ = m.dx
    with pytest.raises(NonUniformMeshError):
        m.require_uniform("test operator")


def test_stretch_ratio_one_is_exactly_uniform():
    m = StructuredMesh(16, 16, stretch_x=1.0, stretch_y=1.0)
    assert m.is_uniform
    assert m.dx == pytest.approx(1.0 / 16)


def test_rejects_degenerate_meshes():
    with pytest.raises(ValueError, match="4x4"):
        StructuredMesh(2, 32)
    with pytest.raises(ValueError, match="positive"):
        StructuredMesh(8, 8, lx=-1.0)
    with pytest.raises(ValueError, match="positive"):
        StructuredMesh(8, 8, stretch_x=-0.5)


def test_from_config_matches_direct_construction():
    cfg = make_config(nx=24, ny=48, lx=3.0, ly=1.5)
    m = StructuredMesh.from_config(cfg)
    assert m.shape == (24, 48)
    assert m.dx == pytest.approx(cfg.dx)
    assert m.dy == pytest.approx(cfg.dy)


def test_repr_is_informative_for_both_mesh_kinds():
    assert "uniform" in repr(StructuredMesh(8, 8))
    assert "stretched" in repr(StructuredMesh(8, 8, stretch_y=1.2))
