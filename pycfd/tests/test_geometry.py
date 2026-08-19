"""Obstacle construction: analytic primitives, custom outlines, bitmaps, predicates.

The volume-fraction machinery is shared by every entry point, so these tests
mostly check that each one hands it the right region -- verified against areas
and centroids that are known in closed form.
"""

import numpy as np
import pytest

from pycfd.core.mesh import StructuredMesh
from pycfd.geometry.obstacles import (
    Obstacle,
    circle_mask,
    load_polygon,
    mask_from_function,
    mask_from_image,
    polygon_mask,
    rectangle_mask,
    transform_polygon,
)


@pytest.fixture
def mesh():
    return StructuredMesh(200, 200, 2.0, 2.0)


def area_of(obstacle: Obstacle, mesh: StructuredMesh) -> float:
    """Physical area implied by the volume fractions."""
    return obstacle.area * mesh.cell_area


def centroid_of(obstacle: Obstacle, mesh: StructuredMesh):
    """Area-weighted centroid, for checking placement."""
    total = obstacle.fraction.sum()
    return (float((obstacle.fraction.sum(1) * mesh.xc).sum() / total),
            float((obstacle.fraction.sum(0) * mesh.yc).sum() / total))


# --------------------------------------------------------------------------- #
# Analytic primitives
# --------------------------------------------------------------------------- #
def test_circle_area_and_placement(mesh):
    obstacle = circle_mask(mesh, (1.0, 1.0), 0.4)
    assert area_of(obstacle, mesh) == pytest.approx(np.pi * 0.16, rel=2e-3)
    assert centroid_of(obstacle, mesh) == pytest.approx((1.0, 1.0), abs=1e-3)
    assert obstacle.characteristic_length == pytest.approx(0.8)


def test_rectangle_area_and_length(mesh):
    obstacle = rectangle_mask(mesh, (0.5, 0.75), (1.5, 1.25))
    assert area_of(obstacle, mesh) == pytest.approx(1.0 * 0.5, rel=1e-6)
    assert obstacle.characteristic_length == pytest.approx(0.5)


def test_circle_outside_the_domain_is_rejected(mesh):
    with pytest.raises(ValueError, match="outside the domain"):
        circle_mask(mesh, (5.0, 1.0), 0.2)


def test_obstacle_too_small_for_the_grid_is_rejected():
    coarse = StructuredMesh(8, 8, 2.0, 2.0)
    with pytest.raises(ValueError, match="single cell"):
        circle_mask(coarse, (1.0, 1.0), 0.005)


# --------------------------------------------------------------------------- #
# Polygons
# --------------------------------------------------------------------------- #
def test_polygon_reproduces_a_square_exactly(mesh):
    square = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
    obstacle = polygon_mask(mesh, square)
    # The outline is grid-aligned, so the fractions should be exact.
    assert area_of(obstacle, mesh) == pytest.approx(1.0, rel=1e-9)
    assert obstacle.characteristic_length == pytest.approx(1.0)


def test_polygon_converges_to_the_circle_it_approximates(mesh):
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    circle = np.column_stack([1 + 0.4 * np.cos(theta), 1 + 0.4 * np.sin(theta)])
    obstacle = polygon_mask(mesh, circle)
    assert area_of(obstacle, mesh) == pytest.approx(np.pi * 0.16, rel=3e-3)
    assert centroid_of(obstacle, mesh) == pytest.approx((1.0, 1.0), abs=1e-3)


def test_polygon_winding_order_does_not_matter(mesh):
    square = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
    clockwise = polygon_mask(mesh, square[::-1])
    anticlockwise = polygon_mask(mesh, square)
    assert np.array_equal(clockwise.mask, anticlockwise.mask)


def test_polygon_handles_a_concave_outline(mesh):
    """An L-shape: the even-odd rule must exclude the notch."""
    ell = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.0],
                    [1.0, 1.0], [1.0, 1.5], [0.5, 1.5]])
    obstacle = polygon_mask(mesh, ell)
    assert area_of(obstacle, mesh) == pytest.approx(0.75, rel=1e-9)
    # The notch corner must be fluid, the arm beside it solid.
    assert not obstacle.mask[np.searchsorted(mesh.xc, 1.25),
                             np.searchsorted(mesh.yc, 1.25)]
    assert obstacle.mask[np.searchsorted(mesh.xc, 0.75),
                         np.searchsorted(mesh.yc, 1.25)]


def test_polygon_rejects_degenerate_input(mesh):
    with pytest.raises(ValueError, match=r"n >= 3"):
        polygon_mask(mesh, np.array([[0.5, 0.5], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="outside the domain"):
        polygon_mask(mesh, np.array([[9.0, 9.0], [9.5, 9.0], [9.5, 9.5]]))


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def test_transform_scales_and_recentres():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    moved = transform_polygon(square, scale=2.0, center=(5.0, 3.0))
    assert moved.mean(axis=0) == pytest.approx([5.0, 3.0])
    assert (moved.max(axis=0) - moved.min(axis=0)) == pytest.approx([2.0, 2.0])


def test_transform_rotation_preserves_area(mesh):
    square = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
    upright = polygon_mask(mesh, square)
    turned = polygon_mask(mesh, transform_polygon(square, rotate_deg=30.0))
    assert area_of(turned, mesh) == pytest.approx(area_of(upright, mesh), rel=5e-3)


def test_transform_rejects_bad_input():
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        transform_polygon(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="scale must be positive"):
        transform_polygon(np.zeros((4, 2)), scale=-1.0)


# --------------------------------------------------------------------------- #
# Vertex files
# --------------------------------------------------------------------------- #
def test_load_polygon_accepts_comments_and_mixed_separators(tmp_path):
    path = tmp_path / "shape.csv"
    path.write_text(
        "# an outline\n"
        "0.0, 0.0\n"
        "1.0 0.0\n"
        "\n"
        "1.0,1.0   # trailing comment\n"
        "0.0, 1.0\n"
    )
    verts = load_polygon(path)
    assert verts.shape == (4, 2)
    assert verts[2] == pytest.approx([1.0, 1.0])


def test_load_polygon_drops_a_repeated_closing_vertex(tmp_path):
    path = tmp_path / "closed.txt"
    path.write_text("0 0\n1 0\n1 1\n0 1\n0 0\n")
    assert load_polygon(path).shape == (4, 2)


def test_load_polygon_reports_bad_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="geometry file not found"):
        load_polygon(tmp_path / "missing.csv")

    short = tmp_path / "short.csv"
    short.write_text("0 0\n1 1\n")
    with pytest.raises(ValueError, match="at least 3 vertices"):
        load_polygon(short)

    bad = tmp_path / "bad.csv"
    bad.write_text("0 0\n1 0\nnot a number here\n0 1\n")
    with pytest.raises(ValueError, match="could not parse"):
        load_polygon(bad)

    thin = tmp_path / "thin.csv"
    thin.write_text("0 0\n1\n2 2\n")
    with pytest.raises(ValueError, match="expected two numbers"):
        load_polygon(thin)


# --------------------------------------------------------------------------- #
# Bitmaps
# --------------------------------------------------------------------------- #
def _write_disc_image(path, height=200, width=400, radius=50):
    """White background with a black disc at the centre."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.image as mpimg

    yy, xx = np.mgrid[0:height, 0:width]
    img = np.ones((height, width))
    img[((xx - width // 2) ** 2 + (yy - height // 2) ** 2) < radius ** 2] = 0.0
    mpimg.imsave(path, img, cmap="gray", vmin=0, vmax=1)


def test_image_silhouette_area_orientation_and_length(tmp_path):
    path = tmp_path / "disc.png"
    _write_disc_image(path)
    mesh = StructuredMesh(400, 200, 4.0, 2.0)          # same 2:1 aspect
    obstacle = mask_from_image(mesh, path)

    # 50 px of 400 across a 4.0-long domain gives radius 0.5.
    assert area_of(obstacle, mesh) == pytest.approx(np.pi * 0.25, rel=1e-2)
    # Centred, which also confirms the image was flipped to y-up.
    assert centroid_of(obstacle, mesh) == pytest.approx((2.0, 1.0), abs=1e-2)
    assert obstacle.characteristic_length == pytest.approx(1.0, rel=5e-2)


def test_image_invert_selects_the_background(tmp_path):
    path = tmp_path / "disc.png"
    _write_disc_image(path)
    mesh = StructuredMesh(200, 100, 4.0, 2.0)
    normal = mask_from_image(mesh, path)
    flipped = mask_from_image(mesh, path, invert=True)

    # The two are complements wherever the cell is not an exact tie.  A cell
    # covered by precisely half a sub-sample count has fraction 0.5, and the
    # strict ``> 0.5`` threshold calls it fluid in *both* orientations -- an
    # inherent ambiguity of thresholding, confined to the surface.
    decided = normal.fraction != 0.5
    assert np.array_equal(flipped.mask[decided], ~normal.mask[decided])
    assert (~decided).sum() < 0.05 * normal.mask.size

    # Away from the surface the two are unambiguously opposite.
    assert normal.mask[100, 50] and not flipped.mask[100, 50]        # disc centre
    assert flipped.mask[5, 5] and not normal.mask[5, 5]              # far corner


def test_image_threshold_extremes_are_reported(tmp_path):
    path = tmp_path / "disc.png"
    _write_disc_image(path)
    mesh = StructuredMesh(100, 50, 4.0, 2.0)
    with pytest.raises(ValueError, match="no cell"):
        mask_from_image(mesh, path, threshold=0.0)
    with pytest.raises(ValueError, match="every cell is solid"):
        mask_from_image(mesh, path, threshold=1.01)


def test_missing_image_is_reported(tmp_path):
    mesh = StructuredMesh(32, 32)
    with pytest.raises(FileNotFoundError, match="geometry image not found"):
        mask_from_image(mesh, tmp_path / "nope.png")


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #
def test_predicate_builds_an_ellipse(mesh):
    obstacle = mask_from_function(
        mesh, lambda x, y: ((x - 1) / 0.5) ** 2 + ((y - 1) / 0.25) ** 2 <= 1.0,
        characteristic_length=0.5,
    )
    assert area_of(obstacle, mesh) == pytest.approx(np.pi * 0.5 * 0.25, rel=2e-3)
    assert obstacle.characteristic_length == pytest.approx(0.5)


def test_predicate_that_selects_nothing_is_reported(mesh):
    with pytest.raises(ValueError, match="no cell solid"):
        mask_from_function(mesh, lambda x, y: np.zeros(x.shape, dtype=bool))


# --------------------------------------------------------------------------- #
# Integration with the solver
# --------------------------------------------------------------------------- #
def test_custom_geometry_runs_and_stays_divergence_free():
    """A custom body must behave exactly like a built-in one in the solver."""
    from pycfd.config import BCKind, BCSpec

    from .conftest import make_config
    from pycfd.core.solver import ProjectionSolver

    mesh = StructuredMesh(64, 32, 4.0, 2.0)
    wedge = np.array([[1.0, 0.8], [1.6, 1.0], [1.0, 1.2]])
    obstacle = polygon_mask(mesh, wedge, name="wedge")

    cfg = make_config(
        nx=64, ny=32, lx=4.0, ly=2.0, re=50.0, dt=2e-3,
        boundary_config={
            "left": BCSpec(BCKind.INLET, velocity=1.0),
            "right": BCSpec(BCKind.OUTLET),
            "bottom": BCSpec(BCKind.SYMMETRY),
            "top": BCSpec(BCKind.SYMMETRY),
        },
    )
    solver = ProjectionSolver(cfg, obstacle=obstacle.mask)
    fields = solver.initialize(1.0)
    for _ in range(10):
        fields = solver.step(fields, 2e-3)

    assert solver.max_divergence(fields) < 1e-11
    assert np.all(fields.u[solver.u_upd][solver.u_face_solid] == 0.0)
    assert np.all(fields.v[solver.v_upd][solver.v_face_solid] == 0.0)
