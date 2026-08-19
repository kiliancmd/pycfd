"""Structured 2D Cartesian mesh with a staggered (MAC) variable layout.

Why staggered
-------------
On a *collocated* grid the second-order central pressure gradient has a
two-cell stencil, which leaves the odd and even pressure modes decoupled and
admits the classic chequerboard pressure oscillation.  On the staggered
Marker-and-Cell grid of Harlow & Welch (1965) the discrete divergence and the
discrete gradient are exact negative adjoints of one another, so the projection
step removes the divergence *to machine precision* and no pressure smoothing or
Rhie--Chow interpolation is needed.

Index convention
----------------
The domain holds ``nx x ny`` cells.  Every field carries exactly one ghost
layer on each side, which makes boundary conditions and periodic wrapping
uniform for all three variables::

    p[i, j]   shape (nx+2, ny+2)   centre  ( (i-0.5)dx , (j-0.5)dy )
                                   interior i = 1..nx , j = 1..ny

    u[i, j]   shape (nx+3, ny+2)   centre  ( (i-1)dx   , (j-0.5)dy )
                                   physical x-faces i = 1..nx+1
                                   (i=1 is x=0, i=nx+1 is x=lx)
                                   ghost columns i = 0 and i = nx+2

    v[i, j]   shape (nx+2, ny+3)   centre  ( (i-0.5)dx , (j-1)dy   )
                                   physical y-faces j = 1..ny+1
                                   (j=1 is y=0, j=ny+1 is y=ly)
                                   ghost rows j = 0 and j = ny+2

With this numbering the two staggered operators are one-liners::

    div[i, j] = (u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dy
    u[i, j]  -= dt * (p[i, j] - p[i-1, j]) / dx
    v[i, j]  -= dt * (p[i, j] - p[i, j-1]) / dy

i.e. the x-face with index ``i`` separates cells ``i-1`` and ``i``.

Stretching
----------
Geometric stretching is supported by the mesh so that non-uniform grids can be
generated and inspected, but the finite-difference operators in
:mod:`pycfd.core.solver` are derived for uniform spacing and will refuse to run
on a stretched mesh (see :meth:`StructuredMesh.require_uniform`).
"""

from __future__ import annotations

import numpy as np


class NonUniformMeshError(RuntimeError):
    """Raised when a uniform-spacing operator is handed a stretched mesh."""


#: Relative tolerance used when deciding whether cell widths are all equal.
_UNIFORMITY_RTOL = 1.0e-12


def _geometric_widths(length: float, n: int, ratio: float) -> np.ndarray:
    """Return ``n`` cell widths growing geometrically by ``ratio`` and summing to ``length``.

    For ``ratio == 1`` this is a uniform distribution.  Otherwise the first cell
    width follows from the geometric series ``w0 * (r**n - 1) / (r - 1) = length``.
    """
    if ratio <= 0:
        raise ValueError(f"stretch ratio must be positive, got {ratio}")
    if abs(ratio - 1.0) < 1.0e-14:
        return np.full(n, length / n, dtype=float)
    w0 = length * (ratio - 1.0) / (ratio ** n - 1.0)
    return w0 * ratio ** np.arange(n, dtype=float)


class StructuredMesh:
    """Uniform-or-stretched structured Cartesian mesh on ``[0, lx] x [0, ly]``.

    Parameters
    ----------
    nx, ny:
        Number of *cells* in each direction.
    lx, ly:
        Domain extents.
    stretch_x, stretch_y:
        Geometric cell-growth ratios; ``1.0`` gives uniform spacing.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        lx: float = 1.0,
        ly: float = 1.0,
        stretch_x: float = 1.0,
        stretch_y: float = 1.0,
    ) -> None:
        if nx < 4 or ny < 4:
            raise ValueError(f"mesh must be at least 4x4 cells, got {nx}x{ny}")
        if lx <= 0 or ly <= 0:
            raise ValueError(f"domain extents must be positive, got {lx}x{ly}")

        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.stretch_x = float(stretch_x)
        self.stretch_y = float(stretch_y)

        # Cell widths, then faces by cumulative sum (exact endpoints enforced).
        self.dx_cells = _geometric_widths(self.lx, self.nx, self.stretch_x)
        self.dy_cells = _geometric_widths(self.ly, self.ny, self.stretch_y)

        self.xf = np.concatenate(([0.0], np.cumsum(self.dx_cells)))
        self.yf = np.concatenate(([0.0], np.cumsum(self.dy_cells)))
        self.xf[-1] = self.lx      # kill accumulated round-off at the far face
        self.yf[-1] = self.ly

        self.xc = 0.5 * (self.xf[:-1] + self.xf[1:])
        self.yc = 0.5 * (self.yf[:-1] + self.yf[1:])

        # Uniformity is a property of the construction arguments, so it is
        # decided once here: the spacing accessors are on the solver's hot path
        # and must not re-scan the width arrays on every call.
        self._is_uniform = bool(
            np.allclose(self.dx_cells, self.dx_cells[0], rtol=_UNIFORMITY_RTOL)
            and np.allclose(self.dy_cells, self.dy_cells[0], rtol=_UNIFORMITY_RTOL)
        )
        self._dx = float(self.dx_cells[0])
        self._dy = float(self.dy_cells[0])
        self._cell_area = self._dx * self._dy

    # ------------------------------------------------------------------ #
    # Uniformity
    # ------------------------------------------------------------------ #
    @property
    def is_uniform(self) -> bool:
        """True when every cell in both directions has the same size."""
        return self._is_uniform

    def require_uniform(self, who: str = "operator") -> None:
        """Raise :class:`NonUniformMeshError` unless the mesh is uniform."""
        if not self._is_uniform:
            raise NonUniformMeshError(
                f"{who} is discretised for uniform spacing but the mesh is "
                f"stretched (stretch_x={self.stretch_x}, stretch_y={self.stretch_y}). "
                "Use stretch_x = stretch_y = 1.0."
            )

    @property
    def dx(self) -> float:
        """Uniform cell width.  Raises on a stretched mesh."""
        if not self._is_uniform:
            self.require_uniform("StructuredMesh.dx")
        return self._dx

    @property
    def dy(self) -> float:
        """Uniform cell height.  Raises on a stretched mesh."""
        if not self._is_uniform:
            self.require_uniform("StructuredMesh.dy")
        return self._dy

    @property
    def cell_area(self) -> float:
        """Area of one cell (uniform mesh only)."""
        if not self._is_uniform:
            self.require_uniform("StructuredMesh.cell_area")
        return self._cell_area

    @property
    def shape(self) -> tuple[int, int]:
        """``(nx, ny)`` cell counts."""
        return (self.nx, self.ny)

    @property
    def n_cells(self) -> int:
        """Total number of interior cells."""
        return self.nx * self.ny

    # ------------------------------------------------------------------ #
    # Coordinate generators for the staggered variable locations
    # ------------------------------------------------------------------ #
    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """1D cell-centre coordinates ``(xc, yc)`` -- where pressure lives."""
        return self.xc, self.yc

    def face_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """1D face coordinates ``(xf, yf)``."""
        return self.xf, self.yf

    def u_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the u degrees of freedom: x-faces by cell-centre rows."""
        return self.xf, self.yc

    def v_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the v degrees of freedom: cell-centre columns by y-faces."""
        return self.xc, self.yf

    def cell_center_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """2D ``(X, Y)`` cell-centre meshgrid with ``ij`` indexing, shape ``(nx, ny)``."""
        return np.meshgrid(self.xc, self.yc, indexing="ij")

    def u_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """2D ``(X, Y)`` meshgrid at u locations, shape ``(nx+1, ny)``."""
        return np.meshgrid(self.xf, self.yc, indexing="ij")

    def v_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """2D ``(X, Y)`` meshgrid at v locations, shape ``(nx, ny+1)``."""
        return np.meshgrid(self.xc, self.yf, indexing="ij")

    # ------------------------------------------------------------------ #
    # Allocation helpers -- the single source of truth for field shapes
    # ------------------------------------------------------------------ #
    @property
    def u_shape(self) -> tuple[int, int]:
        """Ghosted shape of the u array."""
        return (self.nx + 3, self.ny + 2)

    @property
    def v_shape(self) -> tuple[int, int]:
        """Ghosted shape of the v array."""
        return (self.nx + 2, self.ny + 3)

    @property
    def p_shape(self) -> tuple[int, int]:
        """Ghosted shape of the p array."""
        return (self.nx + 2, self.ny + 2)

    def zeros_u(self) -> np.ndarray:
        """Zero-initialised ghosted u array."""
        return np.zeros(self.u_shape, dtype=float)

    def zeros_v(self) -> np.ndarray:
        """Zero-initialised ghosted v array."""
        return np.zeros(self.v_shape, dtype=float)

    def zeros_p(self) -> np.ndarray:
        """Zero-initialised ghosted p array."""
        return np.zeros(self.p_shape, dtype=float)

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        kind = "uniform" if self.is_uniform else "stretched"
        if self.is_uniform:
            spacing = f"dx={self.dx:.6g}, dy={self.dy:.6g}"
        else:
            spacing = (
                f"dx in [{self.dx_cells.min():.6g}, {self.dx_cells.max():.6g}], "
                f"dy in [{self.dy_cells.min():.6g}, {self.dy_cells.max():.6g}]"
            )
        return (
            f"StructuredMesh({self.nx}x{self.ny} cells, "
            f"domain [0,{self.lx:g}]x[0,{self.ly:g}], {kind}, {spacing})"
        )

    @classmethod
    def from_config(cls, cfg) -> "StructuredMesh":
        """Build a mesh from a :class:`~pycfd.config.SimulationConfig`."""
        return cls(cfg.nx, cfg.ny, cfg.lx, cfg.ly, cfg.stretch_x, cfg.stretch_y)
