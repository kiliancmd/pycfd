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
Geometric stretching lets a grid cluster cells where the flow needs them --
near a body, or against a wall -- without paying that resolution across the
whole domain.  The operators in :mod:`pycfd.core.solver` are written against
:class:`MeshMetrics` rather than a single ``dx``, so they run on either kind of
mesh.

Two numbers describe a stretched axis: ``stretch``, the growth factor between
neighbouring cells, and ``cluster``, which end of the axis the growth starts
from (:data:`CLUSTER_MODES`).  The second matters more than it looks.  Growth
has to start *somewhere*, and starting it at the low wall -- the only option
this mesh had at first -- refines that wall and coarsens everything else, which
is right for one wall and wrong for anything symmetric.  Measured on Poiseuille
flow at ratio 1.05, 48 cells: ``low`` is 6x *worse* than a uniform mesh, because
the wall it starves costs more than the wall it refines wins, while ``walls``
is better than uniform on the same cells.  Pick the mode from where the flow
has its gradients, not from a default.

What clustering buys, when the mode matches the problem: an interior layer of
half-width 0.03 reaches 1% error on 48 uniform cells per axis, 24 at a 4x total
growth and 16 at 10x -- 4x and 9x fewer cells in 2D, at second order throughout.

The time step does not pay for this.  It follows the *smallest* cell, and the
smallest cell is whatever resolution the problem demanded in the first place;
clustering changes how many cells are carried alongside it, not how big it is.
Refining a wall by stretching costs the same step as refining it uniformly and
far fewer cells -- but it is still that step, so a mesh clustered 100x takes
100x smaller steps than its coarse cells suggest.

The spacings in :class:`MeshMetrics` are deliberately *scalars* on a uniform
mesh and arrays only on a stretched one.  Every operator is then one
expression rather than a uniform branch and a stretched branch, and the uniform
path still divides by a Python float, so it neither slows down nor changes its
answer in the last bits.

Two spacings, not one
~~~~~~~~~~~~~~~~~~~~~
A staggered grid has two interleaved sets of nodes, so it has two spacings, and
using the wrong one is the classic way to silently lose an order of accuracy:

* ``hx`` -- the *cell* widths ``xf[i+1] - xf[i]``.  This is the control volume
  of a pressure cell, so it divides anything differenced across a cell: the
  divergence, and the transverse convective flux of a momentum equation.
* ``hxu`` -- the spacing between neighbouring cell *centres*, which is the
  control volume of a ``u`` face.  This divides anything differenced across a
  face: the pressure gradient, and the streamwise convective flux.

On a uniform mesh the two coincide, which is exactly why code written only for
uniform spacing can confuse them and still pass every test.

What is *not* second order
~~~~~~~~~~~~~~~~~~~~~~~~~~
A cell centre is the midpoint of its own two faces by construction, so
interpolating a face variable to a cell centre stays a plain average even when
stretched.  A cell *face* is not the midpoint of its two neighbouring centres,
so interpolating a cell variable to a corner needs the geometric weight in
:attr:`MeshMetrics.wy_corner`.  Using ``0.5`` there is first-order on a
stretched mesh and second-order on a uniform one -- a bug that only ever
appears once someone stretches the grid.

Periodicity
~~~~~~~~~~~
A geometrically stretched axis cannot also be periodic: the first and last
cells differ in width, so the spacing is discontinuous across the seam and the
domain does not actually repeat.  :class:`~pycfd.core.solver.ProjectionSolver`
refuses that combination rather than quietly computing a wrong flux there.
"""

from __future__ import annotations

import numpy as np


class NonUniformMeshError(RuntimeError):
    """Raised when a uniform-spacing operator is handed a stretched mesh."""


#: Relative tolerance used when deciding whether cell widths are all equal.
_UNIFORMITY_RTOL = 1.0e-12


class MeshMetrics:
    """The spacings the staggered operators divide by, shaped to broadcast.

    Built once per mesh and cached, because these are on the solver's hot path.

    Every attribute is a plain ``float`` when the mesh is uniform and an
    ``ndarray`` when it is stretched.  Callers never test which: ``a / m.hx``
    is a scalar division on a uniform mesh and a broadcast one on a stretched
    mesh, and reads the same either way.  The uniform case therefore produces
    bit-for-bit the numbers it produced before this class existed.

    Arrays carry a length-1 axis so they broadcast against the ``(nx, ny)``
    interior blocks the operators work in -- ``(n, 1)`` for an x-spacing,
    ``(1, n)`` for a y-spacing.

    Attributes
    ----------
    hx, hy:
        Cell widths/heights, ``(nx, 1)`` and ``(1, ny)``.  The control volume
        of a pressure cell.
    hxu, hyv:
        Distance between neighbouring cell centres, ``(nx+1, 1)`` and
        ``(1, ny+1)``, indexed by face.  The control volume of a velocity face.
        Extended by one ghost at each end, where the ghost cell mirrors the
        width of the cell it borders -- so ``hxu[0] == hx[0]``.
    hx_ext, hy_ext:
        Cell widths including that ghost at each end, ``(nx+2, 1)`` and
        ``(1, ny+2)``.  A face's two flanking cell widths are ``hx_ext[m]`` and
        ``hx_ext[m+1]``, which is what the viscous stencil needs.
    wx_corner, wy_corner:
        Weight of the *upper* node when interpolating a cell-centred variable
        to a corner, ``(nx+1, 1)`` and ``(1, ny+1)``.  Exactly ``0.5`` on a
        uniform mesh; see the module docstring for why it cannot be assumed.
    cell_areas:
        ``(nx, ny)`` cell areas, for integrating anything over the domain.
    min_hx, min_hy:
        Smallest cell width/height, which is what a stability limit answers to.
    """

    __slots__ = ("hx", "hy", "hxu", "hyv", "hx_ext", "hy_ext",
                 "wx_corner", "wy_corner", "cell_areas", "min_hx", "min_hy",
                 "uniform_area")

    def __init__(self, mesh: "StructuredMesh") -> None:
        dx_cells, dy_cells = mesh.dx_cells, mesh.dy_cells

        # The ghost cell mirrors the width of the cell it borders.  That is the
        # convention the ghost *values* already follow, and it makes hxu[0]
        # come out as hx[0] rather than something that depends on a cell
        # outside the domain.
        hx_ext = np.concatenate(([dx_cells[0]], dx_cells, [dx_cells[-1]]))
        hy_ext = np.concatenate(([dy_cells[0]], dy_cells, [dy_cells[-1]]))

        # Centre-to-centre distance across face m, including the two ghosts.
        hxu = 0.5 * (hx_ext[:-1] + hx_ext[1:])
        hyv = 0.5 * (hy_ext[:-1] + hy_ext[1:])

        # A corner sits half a cell above the centre below it, so the weight of
        # the upper node is that half-cell over the full centre-to-centre span.
        wx_corner = 0.5 * hx_ext[:-1] / hxu
        wy_corner = 0.5 * hy_ext[:-1] / hyv

        self.cell_areas = dx_cells[:, None] * dy_cells[None, :]
        self.min_hx = float(dx_cells.min())
        self.min_hy = float(dy_cells.min())

        self.uniform_area = (
            float(dx_cells[0] * dy_cells[0]) if mesh.is_uniform else None
        )

        if mesh.is_uniform:
            # Scalars, so every operator stays exactly the arithmetic it was
            # before stretching was supported.
            self.hx = float(dx_cells[0])
            self.hy = float(dy_cells[0])
            self.hxu = float(dx_cells[0])
            self.hyv = float(dy_cells[0])
            self.hx_ext = float(dx_cells[0])
            self.hy_ext = float(dy_cells[0])
            self.wx_corner = 0.5
            self.wy_corner = 0.5
        else:
            self.hx = dx_cells[:, None]
            self.hy = dy_cells[None, :]
            self.hxu = hxu[:, None]
            self.hyv = hyv[None, :]
            self.hx_ext = hx_ext[:, None]
            self.hy_ext = hy_ext[None, :]
            self.wx_corner = wx_corner[:, None]
            self.wy_corner = wy_corner[None, :]

    def integrate(self, field: np.ndarray) -> float:
        """Domain integral of a cell-centred ``(nx, ny)`` field.

        One scalar multiply after the reduction on a uniform mesh -- cheaper,
        and the exact arithmetic the recorded energies were measured with --
        and a weighted reduction when the cells differ in size.
        """
        if self.uniform_area is not None:
            return float(np.sum(field)) * self.uniform_area
        return float(np.sum(field * self.cell_areas))

    @staticmethod
    def _slice(value, lo: int, hi: int, axis: int):
        """``value[lo:hi]`` along ``axis``, or the scalar itself unchanged.

        Lets an operator take the part of a spacing it needs without first
        asking whether the mesh is stretched.
        """
        if np.isscalar(value):
            return value
        return value[lo:hi, :] if axis == 0 else value[:, lo:hi]


#: Where a stretched axis puts its *smallest* cells.  The stretch ratio is a
#: per-cell growth rate in every mode; what changes is which end of the axis the
#: growth starts from.
#:
#: ``"low"``
#:     Smallest cell at the low-coordinate end, growing monotonically across the
#:     axis.  Right for a single wall or a shear layer pinned to one side.
#: ``"walls"``
#:     Smallest cells at *both* ends, largest in the middle.  Right for a channel
#:     or any domain bounded by two walls -- ``"low"`` refines one wall and
#:     starves the other, which costs more accuracy at the starved wall than the
#:     clustering wins at the refined one.
#: ``"centre"``
#:     Smallest cells in the middle, growing outward to both ends.  Right for a
#:     body held in the interior of a domain whose far field only has to be far,
#:     not resolved.
CLUSTER_MODES = ("low", "walls", "centre")


def _normalise_cluster(mode: str) -> str:
    """Accept the American spelling of ``"centre"`` and reject anything else."""
    mode = str(mode).lower()
    if mode == "center":
        return "centre"
    if mode not in CLUSTER_MODES:
        raise ValueError(
            f"cluster mode must be one of {CLUSTER_MODES}, got {mode!r}"
        )
    return mode


def _geometric_widths(length: float, n: int, ratio: float,
                      cluster: str = "low") -> np.ndarray:
    """Return ``n`` cell widths growing by ``ratio`` per cell and summing to ``length``.

    ``ratio == 1`` is uniform in every mode.  Otherwise ``cluster`` selects one
    of :data:`CLUSTER_MODES`; in each case the amplitude is fixed by requiring
    the widths to sum to ``length`` exactly, so the ratio stays a pure shape
    parameter and refining the mesh does not change the shape.

    An axis with an odd cell count gets a single unpaired middle cell, which
    continues the geometric series rather than interrupting it: the largest cell
    for ``"walls"``, the smallest for ``"centre"``.
    """
    if ratio <= 0:
        raise ValueError(f"stretch ratio must be positive, got {ratio}")
    if cluster not in CLUSTER_MODES:
        raise ValueError(
            f"cluster mode must be one of {CLUSTER_MODES}, got {cluster!r}"
        )
    if abs(ratio - 1.0) < 1.0e-14:
        return np.full(n, length / n, dtype=float)

    if cluster == "low":
        w0 = length * (ratio - 1.0) / (ratio ** n - 1.0)
        return w0 * ratio ** np.arange(n, dtype=float)

    half, odd = divmod(n, 2)
    if cluster == "walls":
        # [small .. large] (largest) [large .. small]
        run = ratio ** np.arange(half, dtype=float)
        w0 = length / (2.0 * run.sum() + odd * ratio ** half)
        run = w0 * run
        middle = [w0 * ratio ** half] if odd else []
        return np.concatenate([run, middle, run[::-1]])

    # "centre": [large .. small] (smallest) [small .. large]
    if odd:
        out = ratio ** np.arange(1, half + 1, dtype=float)
        w0 = length / (1.0 + 2.0 * out.sum())
        out = w0 * out
        return np.concatenate([out[::-1], [w0], out])
    run = ratio ** np.arange(half, dtype=float)
    run = (length / (2.0 * run.sum())) * run
    return np.concatenate([run[::-1], run])


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
    cluster_x, cluster_y:
        Which end of each axis the growth starts from, one of
        :data:`CLUSTER_MODES`.  Ignored when the corresponding ratio is ``1.0``.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        lx: float = 1.0,
        ly: float = 1.0,
        stretch_x: float = 1.0,
        stretch_y: float = 1.0,
        cluster_x: str = "low",
        cluster_y: str = "low",
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
        self.cluster_x = _normalise_cluster(cluster_x)
        self.cluster_y = _normalise_cluster(cluster_y)

        # Cell widths, then faces by cumulative sum (exact endpoints enforced).
        self.dx_cells = _geometric_widths(self.lx, self.nx, self.stretch_x,
                                          self.cluster_x)
        self.dy_cells = _geometric_widths(self.ly, self.ny, self.stretch_y,
                                          self.cluster_y)

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
        self._metrics: MeshMetrics | None = None

    # ------------------------------------------------------------------ #
    # Uniformity
    # ------------------------------------------------------------------ #
    @property
    def is_uniform(self) -> bool:
        """True when every cell in both directions has the same size."""
        return self._is_uniform

    @property
    def stretched_x(self) -> bool:
        """True when the x cell widths are not all equal."""
        return not bool(
            np.allclose(self.dx_cells, self.dx_cells[0], rtol=_UNIFORMITY_RTOL)
        )

    @property
    def stretched_y(self) -> bool:
        """True when the y cell heights are not all equal."""
        return not bool(
            np.allclose(self.dy_cells, self.dy_cells[0], rtol=_UNIFORMITY_RTOL)
        )

    @property
    def metrics(self) -> MeshMetrics:
        """The spacings the staggered operators divide by; built once, cached."""
        if self._metrics is None:
            self._metrics = MeshMetrics(self)
        return self._metrics

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
        return cls(cfg.nx, cfg.ny, cfg.lx, cfg.ly, cfg.stretch_x, cfg.stretch_y,
                   cfg.cluster_x, cfg.cluster_y)
