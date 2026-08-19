"""Mask-based immersed boundaries.

An obstacle is represented by a boolean cell mask.  The solver enforces it by
*direct forcing*: every staggered face touching a solid cell is reset to zero
velocity after the predictor and again after the projection, and the pressure
Poisson stencil drops those same faces so the two operators stay consistent.

Boundary resolution
-------------------
A plain "is the cell centre inside?" test makes the surface jump by a whole cell
and gives noticeably grid-dependent forces.  :func:`circle_mask` instead
computes the *solid volume fraction* of each cell by supersampling and
thresholds at one half, so the staircase follows the true surface as closely as
a cell-centred mask can.  The fraction itself is returned as well, since it is a
better estimate of the enclosed area than the mask is.

The representation remains first-order accurate at the surface -- that is the
price of a non-conforming Cartesian mesh, and it is why the drag coefficients
this code reports should be read as engineering estimates.

Custom geometry
---------------
Any 2D shape can be embedded, not just the analytic primitives.  Four entry
points share the same volume-fraction machinery:

* :func:`circle_mask` and :func:`rectangle_mask` -- analytic primitives;
* :func:`polygon_mask` -- an arbitrary closed outline, typically loaded from a
  vertex file with :func:`load_polygon`;
* :func:`mask_from_image` -- a black-on-white bitmap silhouette;
* :func:`mask_from_function` -- any vectorised ``inside(x, y)`` predicate, for
  shapes that are easier to define by formula than by outline.

All four return an :class:`Obstacle`, which is what
:class:`~pycfd.physics.incompressible.Simulation` accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.mesh import StructuredMesh

#: Sub-samples per cell edge used when estimating solid volume fractions.
DEFAULT_SUBSAMPLES = 8

#: Volume fraction above which a cell is considered solid.
SOLID_FRACTION_THRESHOLD = 0.5


@dataclass
class Obstacle:
    """A solid body embedded in the flow.

    Attributes
    ----------
    mask:
        Boolean ``(nx, ny)`` array, ``True`` inside the body.
    fraction:
        Solid volume fraction of each cell, in ``[0, 1]``.
    characteristic_length:
        Length used to non-dimensionalise forces (a cylinder's diameter).
    name:
        Label used in plots and logs.
    """

    mask: np.ndarray
    fraction: np.ndarray
    characteristic_length: float
    name: str = "obstacle"

    @property
    def area(self) -> float:
        """Solid area in cell units (multiply by the cell area for a true area)."""
        return float(self.fraction.sum())

    def __repr__(self) -> str:
        return (
            f"Obstacle({self.name!r}, {int(self.mask.sum())} solid cells, "
            f"L={self.characteristic_length:g})"
        )


def _cell_solid_fraction(mesh: StructuredMesh, inside,
                         subsamples: int = DEFAULT_SUBSAMPLES) -> np.ndarray:
    """Fraction of each cell lying inside the region defined by ``inside(x, y)``.

    ``inside`` is a vectorised predicate over coordinate arrays.
    """
    if subsamples < 1:
        raise ValueError(f"subsamples must be >= 1, got {subsamples}")
    offs = (np.arange(subsamples) + 0.5) / subsamples   # mid-point sub-samples

    xs = (mesh.xf[:-1, None] + offs[None, :] * mesh.dx_cells[:, None]).ravel()
    ys = (mesh.yf[:-1, None] + offs[None, :] * mesh.dy_cells[:, None]).ravel()
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    hits = inside(X, Y).astype(float)
    hits = hits.reshape(mesh.nx, subsamples, mesh.ny, subsamples)
    return hits.mean(axis=(1, 3))


def circle_mask(mesh: StructuredMesh, center: tuple[float, float], radius: float,
                subsamples: int = DEFAULT_SUBSAMPLES, name: str = "cylinder") -> Obstacle:
    """Circular obstacle with a sub-cell-resolved boundary.

    Parameters
    ----------
    center:
        ``(cx, cy)`` centre of the circle in domain coordinates.
    radius:
        Circle radius; the characteristic length is the diameter.
    subsamples:
        Sub-samples per cell edge used to estimate the solid fraction.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    cx, cy = center
    if not (0 <= cx <= mesh.lx and 0 <= cy <= mesh.ly):
        raise ValueError(f"circle centre {center} lies outside the domain")

    frac = _cell_solid_fraction(
        mesh, lambda X, Y: (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2, subsamples
    )
    mask = frac > SOLID_FRACTION_THRESHOLD
    if not mask.any():
        raise ValueError(
            f"circle of radius {radius:g} does not cover a single cell on a "
            f"{mesh.nx}x{mesh.ny} grid (dx={mesh.dx:g}); refine the mesh or "
            "enlarge the obstacle"
        )
    return Obstacle(mask, frac, 2.0 * radius, name)


def rectangle_mask(mesh: StructuredMesh, lower_left: tuple[float, float],
                   upper_right: tuple[float, float],
                   subsamples: int = DEFAULT_SUBSAMPLES,
                   name: str = "rectangle") -> Obstacle:
    """Axis-aligned rectangular obstacle."""
    x0, y0 = lower_left
    x1, y1 = upper_right
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"rectangle corners are degenerate: {lower_left} -> {upper_right}")

    frac = _cell_solid_fraction(
        mesh, lambda X, Y: (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1), subsamples
    )
    mask = frac > SOLID_FRACTION_THRESHOLD
    if not mask.any():
        raise ValueError("rectangle does not cover a single cell; refine the mesh")
    return Obstacle(mask, frac, y1 - y0, name)


# --------------------------------------------------------------------------- #
# Custom geometry
# --------------------------------------------------------------------------- #
def _point_in_polygon(X: np.ndarray, Y: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Vectorised even-odd (ray-casting) point-in-polygon test.

    A horizontal ray is cast from each point towards ``+x`` and the crossings
    with each edge are counted; an odd count means the point is inside.  The
    polygon is treated as closed, so the last vertex joins back to the first.
    Points are pre-filtered by the polygon's bounding box, which is what keeps
    this affordable at the sub-sample densities :func:`_cell_solid_fraction`
    uses.
    """
    inside = np.zeros(X.shape, dtype=bool)
    xv, yv = verts[:, 0], verts[:, 1]

    box = ((X >= xv.min()) & (X <= xv.max()) & (Y >= yv.min()) & (Y <= yv.max()))
    if not box.any():
        return inside

    xs, ys = X[box], Y[box]
    hit = np.zeros(xs.shape, dtype=bool)
    xa, ya = xv, yv
    xb, yb = np.roll(xv, -1), np.roll(yv, -1)

    for x0, y0, x1, y1 in zip(xa, ya, xb, yb):
        # Only edges that straddle the ray's height can be crossed.  Horizontal
        # edges never straddle (the test is False), so the division below is
        # harmless even when it produces an infinity.
        straddles = (y0 > ys) != (y1 > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x0 + (ys - y0) * (x1 - x0) / (y1 - y0)
        hit ^= straddles & (xs < x_cross)

    inside[box] = hit
    return inside

# vertex file
def load_polygon(path: str | Path) -> np.ndarray:
    """Read a closed 2D outline from a text file of ``x, y`` vertex pairs.

    Accepts comma- or whitespace-separated columns; blank lines and lines
    starting with ``#`` are ignored, so CSV exported from a CAD tool and a
    hand-written table both work.  The outline is treated as closed, so a
    repeated final vertex is optional and is dropped if present.

    Returns an ``(n, 2)`` array of vertices in domain coordinates.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"geometry file not found: {path}")

    rows: list[tuple[float, float]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 2:
            raise ValueError(
                f"{path}:{lineno}: expected two numbers 'x y', got {raw.strip()!r}"
            )
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            raise ValueError(
                f"{path}:{lineno}: could not parse coordinates from {raw.strip()!r}"
            ) from None

    if len(rows) < 3:
        raise ValueError(
            f"{path}: a polygon needs at least 3 vertices, found {len(rows)}"
        )
    verts = np.asarray(rows, dtype=float)
    # Drop an explicitly repeated closing vertex; the outline is always closed.
    if np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]
    return verts


def transform_polygon(verts: np.ndarray, scale: float = 1.0,
                      center: tuple[float, float] | None = None,
                      rotate_deg: float = 0.0) -> np.ndarray:
    """Scale, rotate and reposition an outline.

    The shape is scaled and rotated about its own centroid, then translated so
    that centroid sits at ``center``.  This is what lets one geometry file be
    reused across domains of different size without editing the file.
    """
    verts = np.asarray(verts, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError(f"vertices must have shape (n, 2), got {verts.shape}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    centroid = verts.mean(axis=0)
    local = (verts - centroid) * scale
    if rotate_deg:
        theta = np.radians(rotate_deg)
        c, s = np.cos(theta), np.sin(theta)
        local = local @ np.array([[c, s], [-s, c]])
    return local + (centroid if center is None else np.asarray(center, dtype=float))


def polygon_mask(mesh: StructuredMesh, vertices: np.ndarray,
                 characteristic_length: float | None = None,
                 subsamples: int = DEFAULT_SUBSAMPLES,
                 name: str = "polygon") -> Obstacle:
    """Obstacle from an arbitrary closed 2D outline.

    Parameters
    ----------
    vertices:
        ``(n, 2)`` array of outline vertices in domain coordinates, in either
        winding order.  Load one with :func:`load_polygon`.
    characteristic_length:
        Reference length for force coefficients.  Defaults to the outline's
        height (its extent normal to the flow for the usual left-to-right
        configuration), matching the convention used for a cylinder diameter.
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
        raise ValueError(
            f"vertices must be an (n, 2) array with n >= 3, got {verts.shape}"
        )

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    if hi[0] < 0 or lo[0] > mesh.lx or hi[1] < 0 or lo[1] > mesh.ly:
        raise ValueError(
            f"geometry bounding box x=[{lo[0]:g}, {hi[0]:g}] y=[{lo[1]:g}, {hi[1]:g}] "
            f"lies entirely outside the domain [0, {mesh.lx:g}] x [0, {mesh.ly:g}]; "
            "reposition it with transform_polygon()"
        )

    frac = _cell_solid_fraction(
        mesh, lambda X, Y: _point_in_polygon(X, Y, verts), subsamples
    )
    mask = frac > SOLID_FRACTION_THRESHOLD
    if not mask.any():
        raise ValueError(
            f"the outline does not cover a single cell on a {mesh.nx}x{mesh.ny} "
            f"grid (dx={mesh.dx:g}, dy={mesh.dy:g}); refine the mesh or scale the "
            "geometry up with transform_polygon()"
        )

    length = float(hi[1] - lo[1]) if characteristic_length is None \
        else float(characteristic_length)
    return Obstacle(mask, frac, length, name)

# bitmap silhouette
def mask_from_image(mesh: StructuredMesh, path: str | Path, threshold: float = 0.5,
                    invert: bool = False, characteristic_length: float | None = None,
                    subsamples: int = DEFAULT_SUBSAMPLES,
                    name: str = "image") -> Obstacle:
    """Obstacle from a bitmap silhouette stretched over the whole domain.

    Dark pixels are solid by default, so a black shape on a white background
    works with no preparation; pass ``invert=True`` for the opposite convention.
    Colour images are reduced to luminance. The image is stretched to cover the
    full domain, so its aspect ratio should match ``lx : ly`` if the shape is
    not to be distorted.

    Parameters
    ----------
    threshold:
        Luminance in ``[0, 1]`` below which a pixel counts as solid.
    """
    import matplotlib.image as mpimg      # matplotlib is a hard dependency

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"geometry image not found: {path}")

    img = np.asarray(mpimg.imread(path), dtype=float)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)   # RGB(A) -> luminance, alpha ignored
    if img.max() > 1.0:
        img = img / 255.0                 # 8-bit images arrive in [0, 255]
    # Image row 0 is the top of the picture; flip so the row index grows with y.
    img = img[::-1, :]
    rows, cols = img.shape

    def inside(X, Y):
        col = np.clip((X / mesh.lx * cols).astype(int), 0, cols - 1)
        row = np.clip((Y / mesh.ly * rows).astype(int), 0, rows - 1)
        solid = img[row, col] < threshold
        return ~solid if invert else solid

    frac = _cell_solid_fraction(mesh, inside, subsamples)
    mask = frac > SOLID_FRACTION_THRESHOLD
    if not mask.any():
        raise ValueError(
            f"no cell of the {mesh.nx}x{mesh.ny} grid is solid after thresholding "
            f"{path.name} at {threshold}; try a different threshold or invert=True"
        )
    if mask.all():
        raise ValueError(
            f"every cell is solid after thresholding {path.name} at {threshold}; "
            "the image is probably inverted (try invert=True)"
        )

    if characteristic_length is None:
        rows_used = np.flatnonzero(mask.any(axis=0))
        characteristic_length = float((rows_used[-1] - rows_used[0] + 1) * mesh.dy)
    return Obstacle(mask, frac, float(characteristic_length), name)

# formula
def mask_from_function(mesh: StructuredMesh, inside,
                       characteristic_length: float = 1.0,
                       subsamples: int = DEFAULT_SUBSAMPLES,
                       name: str = "custom") -> Obstacle:
    """Obstacle from a vectorised predicate ``inside(x, y) -> bool array``.

    The most direct entry point when the shape has a closed-form description::

        # an ellipse with semi-axes a and b, centred at (cx, cy)
        mask_from_function(
            mesh, lambda x, y: ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0,
            characteristic_length=2 * b,
        )

    ``inside`` must accept and return NumPy arrays of matching shape.
    """
    frac = _cell_solid_fraction(mesh, inside, subsamples)
    mask = frac > SOLID_FRACTION_THRESHOLD
    if not mask.any():
        raise ValueError(
            f"the predicate marks no cell solid on a {mesh.nx}x{mesh.ny} grid; "
            "check the coordinates it expects are in domain units"
        )
    return Obstacle(mask, frac, float(characteristic_length), name)
