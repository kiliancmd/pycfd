"""Incompressible Navier--Stokes solver: Chorin's projection (fractional step) method.

Algorithm
---------
Each substep advances

    u*        = u^n + dt * ( -A(u^n) + D(u^n) + f )        predictor
    lap(p)    = div(u*) / dt                                pressure Poisson
    u^{n+1}   = u* - dt * grad(p)                           projection

with ``A`` the convective term, ``D`` the viscous term and ``f`` a body force.

Discretisation
--------------
Everything lives on the staggered MAC grid described in :mod:`pycfd.core.mesh`.

*Convection* uses the conservative (divergence) form ``d(uu)/dx + d(uv)/dy``.
The mixed product ``uv`` is evaluated once at each cell **corner** and shared by
both momentum equations, which is what makes the scheme discretely
momentum- and (in the inviscid limit) energy-conserving.  An optional donor-cell
blend in the style of Griebel et al. adds controllable upwinding for high
Reynolds numbers.

*Diffusion* is the standard 5-point Laplacian.  When the Smagorinsky model is
active the viscous term is instead written in full stress-divergence form,
``d/dx(2 nu_e du/dx) + d/dy(nu_e (du/dy + dv/dx))``, with the normal stresses at
cell centres and the shear stress at corners.  On a divergence-free field with
constant viscosity the two forms agree to machine precision.

*Projection* corrects exactly those faces that the Poisson stencil in
:mod:`pycfd.core.pressure` includes, which is what makes the final velocity
divergence-free to solver tolerance rather than to truncation error.

Time integration is delegated to :mod:`pycfd.core.timestepper`; this module
exposes the single-substep operator that the Runge--Kutta stages compose.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from ..config import (
    AdvectionScheme,
    BCKind,
    PressureSolver,
    SimulationConfig,
    SolverType,
    TimeScheme,
)
from . import kernels
from .boundary import BoundaryManager
from .fields import FlowField
from .mesh import MeshMetrics, NonUniformMeshError, StructuredMesh
from .pressure import assemble_poisson_matrix, make_pressure_solver

log = logging.getLogger(__name__)


def _face_solid_mask(solid: np.ndarray, lo: int, hi: int, axis: int,
                     periodic: bool) -> np.ndarray:
    """Which faces in ``[lo, hi)`` touch a solid cell.

    A face with array index ``m`` separates the zero-based cells ``m-2`` and
    ``m-1``.  Off-domain neighbours are fluid by convention (padded ``False``),
    which is what lets a pressure-outlet face -- whose outward neighbour does
    not exist -- be treated like any other.
    """
    m = np.arange(lo, hi)
    if periodic:
        n = solid.shape[axis]
        return (np.take(solid, (m - 2) % n, axis=axis)
                | np.take(solid, (m - 1) % n, axis=axis))
    pad = [(0, 0), (0, 0)]
    pad[axis] = (1, 1)
    padded = np.pad(solid, pad, constant_values=False)
    return np.take(padded, m - 1, axis=axis) | np.take(padded, m, axis=axis)


class ProjectionSolver:
    """Fractional-step solver for the 2D incompressible Navier--Stokes equations.

    Parameters
    ----------
    config:
        Validated :class:`~pycfd.config.SimulationConfig`.
    mesh:
        Optional pre-built mesh; created from ``config`` when omitted.
    obstacle:
        Optional boolean ``(nx, ny)`` mask, ``True`` inside a solid body.
    bodies:
        Optional per-body masks whose union is ``obstacle``.  Supplying them
        makes the solver report the reaction force on each body separately in
        :attr:`body_force_reactions`, which is the whole point of putting more
        than one body in a domain.  Omitting them costs nothing and reports the
        total only.
    """

    def __init__(
        self,
        config: SimulationConfig,
        mesh: StructuredMesh | None = None,
        obstacle: np.ndarray | None = None,
        bodies: Sequence[np.ndarray] | None = None,
    ) -> None:
        if config.solver_type is SolverType.SIMPLE:
            raise NotImplementedError(
                "solver_type='simple' is reserved for a future SIMPLE "
                "(pressure-correction) implementation; only 'projection' is "
                "available. Set solver_type='projection'."
            )

        self.config = config
        self.mesh = mesh if mesh is not None else StructuredMesh.from_config(config)
        self.metrics = self.mesh.metrics
        self.nu = config.nu

        self.boundaries = BoundaryManager(config.boundary_config, self.mesh)
        self.periodic_x = self.boundaries.periodic_x
        self.periodic_y = self.boundaries.periodic_y
        self._reject_stretched_periodic()

        nx, ny = self.mesh.shape
        if obstacle is None:
            self.solid = np.zeros((nx, ny), dtype=bool)
        else:
            self.solid = np.asarray(obstacle, dtype=bool)
            if self.solid.shape != (nx, ny):
                raise ValueError(
                    f"obstacle mask must have shape {(nx, ny)}, got {self.solid.shape}"
                )
        self.has_obstacle = bool(self.solid.any())

        # Per-body masks are kept only when they would tell us something the
        # union does not -- with a single body the two are the same array and
        # the extra reduction on every substep would buy nothing.
        self.body_masks: tuple[np.ndarray, ...] = ()
        if bodies is not None and len(bodies) > 1:
            self.body_masks = tuple(np.asarray(b, dtype=bool) for b in bodies)
            for k, b in enumerate(self.body_masks):
                if b.shape != (nx, ny):
                    raise ValueError(
                        f"body {k} mask must have shape {(nx, ny)}, got {b.shape}"
                    )

        self._build_slices()

        #: Walls holding the pressure fixed; non-empty makes the operator
        #: non-singular and the reported pressure absolute.
        self.dirichlet_pressure = self.boundaries.dirichlet_pressure()
        self.poisson = assemble_poisson_matrix(
            self.mesh, self.periodic_x, self.periodic_y,
            self.solid if self.has_obstacle else None,
            dirichlet=self.dirichlet_pressure or None,
        )
        self.pressure_solver = make_pressure_solver(
            config.pressure_solver, self.poisson,
            tol=config.poisson_tol, maxiter=config.poisson_maxiter,
            sor_omega=config.sor_omega, mg_sweeps=config.mg_sweeps,
        )

        self.turbulence = None
        if config.use_les:
            from ..physics.turbulence import SmagorinskyModel
            self.turbulence = SmagorinskyModel(self.mesh, config.smagorinsky_cs)

        #: Previous substep's pressure, offered to the next solve as a starting
        #: guess.  Left at ``None`` for a direct solver, which cannot use one.
        self._p_guess: np.ndarray | None = None

        #: Immersed-boundary reaction force from the most recent substep, (fx, fy).
        self.body_force_reaction = (0.0, 0.0)
        #: The same force resolved per body, empty unless several were supplied.
        self.body_force_reactions: tuple[tuple[float, float], ...] = tuple(
            (0.0, 0.0) for _ in self.body_masks
        )
        self._blend = 0.0    # donor-cell blending factor in force for this substep

        # The fused kernel covers the constant-viscosity case only; the
        # Smagorinsky closure needs the variable-viscosity stress form.  It
        # also takes a scalar dx/dy, so a stretched mesh falls back to the
        # array stencils rather than being handed a spacing that is a lie.
        self._use_kernel = (
            config.use_numba and kernels.NUMBA_AVAILABLE
            and self.turbulence is None and self.mesh.is_uniform
        )
        if self._use_kernel:
            self._rhs_u = np.empty((nx + 1, ny))
            self._rhs_v = np.empty((nx, ny + 1))
            self._kernel = kernels.select_kernel(nx * ny)
            kernels.warmup(nx * ny)
        elif config.use_numba and not kernels.NUMBA_AVAILABLE:
            log.info("numba is not installed; using the vectorised NumPy stencils")

    @staticmethod
    def _momentum_sum(faces: np.ndarray, mask: np.ndarray, area) -> float:
        """Momentum carried by the masked faces, given their control volumes.

        Weighting before the reduction is what a varying area requires; on a
        uniform mesh one scalar multiply after it is both cheaper and the exact
        arithmetic the recorded forces were measured with.
        """
        if np.isscalar(area):
            return float(faces[mask].sum()) * area
        return float((faces * area)[mask].sum())

    def _reject_stretched_periodic(self) -> None:
        """Refuse a stretched axis that is also periodic.

        Geometric stretching makes the first and last cell different widths, so
        the spacing jumps across the seam and the domain does not repeat.  The
        operators would still produce numbers there -- wrong ones, from a flux
        divided by a spacing that belongs to the other end of the domain -- so
        this is caught at construction instead of at the seam.
        """
        for axis, stretched, periodic, ratio in (
            ("x", self.mesh.stretched_x, self.periodic_x, self.mesh.stretch_x),
            ("y", self.mesh.stretched_y, self.periodic_y, self.mesh.stretch_y),
        ):
            if stretched and periodic:
                raise NonUniformMeshError(
                    f"the {axis} axis is periodic and stretched "
                    f"(stretch_{axis}={ratio}): the first and last cells differ "
                    "in width, so the spacing is discontinuous across the seam "
                    "and the domain does not actually repeat. Use "
                    f"stretch_{axis}=1.0 on a periodic axis, or a "
                    "non-periodic boundary condition."
                )

    # ------------------------------------------------------------------ #
    # Index bookkeeping
    # ------------------------------------------------------------------ #
    def _build_slices(self) -> None:
        """Pre-compute the slices of solvable faces and their obstacle masks.

        A face is *solvable* when the projection is allowed to change it:

        * every interior face;
        * the first face of a periodic axis, whose partner at the far end is its
          own image;
        * the boundary face of a **pressure outlet**, where the velocity is not
          prescribed at all but follows from the pressure field.

        Everywhere else the normal velocity is imposed, the Poisson stencil
        carries no flux through that face, and the projection must leave it
        untouched -- the two have to agree exactly or the divergence stops
        vanishing.
        """
        nx, ny = self.mesh.shape
        kinds = {w: bc.kind for w, bc in self.boundaries.conditions.items()}
        outlet = BCKind.PRESSURE_OUTLET

        u_lo = 1 if (self.periodic_x or kinds["left"] is outlet) else 2
        u_hi = nx + 2 if (kinds["right"] is outlet and not self.periodic_x) else nx + 1
        self.u_upd = (slice(u_lo, u_hi), slice(1, ny + 1))
        self.u_rhs_sel = slice(u_lo - 1, u_hi - 1)   # into the (nx+1)-long RHS block

        v_lo = 1 if (self.periodic_y or kinds["bottom"] is outlet) else 2
        v_hi = ny + 2 if (kinds["top"] is outlet and not self.periodic_y) else ny + 1
        self.v_upd = (slice(1, nx + 1), slice(v_lo, v_hi))
        self.v_rhs_sel = slice(v_lo - 1, v_hi - 1)

        # Control-volume area behind each updated velocity face, for turning a
        # velocity into the momentum it carries.  A u face owns a centre-to-
        # centre width and a full cell height; a v face the other way round.
        # Both are the scalar cell area on a uniform mesh.
        m = self.metrics
        if self.mesh.is_uniform:
            self._u_cv_area = self._v_cv_area = self.mesh.cell_area
        else:
            self._u_cv_area = (m.hxu[u_lo - 1:u_hi - 1, :] * m.hy)
            self._v_cv_area = (m.hx * m.hyv[:, v_lo - 1:v_hi - 1])

        if self.has_obstacle:
            self.u_face_solid = _face_solid_mask(
                self.solid, u_lo, u_hi, axis=0, periodic=self.periodic_x)
            self.v_face_solid = _face_solid_mask(
                self.solid, v_lo, v_hi, axis=1, periodic=self.periodic_y)
        else:
            self.u_face_solid = None
            self.v_face_solid = None

        # One face set per body, so the momentum removed at each face can be
        # charged to the body that removed it.  The sets are disjoint and their
        # union is the total: a face touching two bodies would land in two of
        # them, which is exactly the geometry ObstacleGroup refuses.
        self.u_face_body = tuple(
            _face_solid_mask(b, u_lo, u_hi, axis=0, periodic=self.periodic_x)
            for b in self.body_masks
        )
        self.v_face_body = tuple(
            _face_solid_mask(b, v_lo, v_hi, axis=1, periodic=self.periodic_y)
            for b in self.body_masks
        )

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def initialize(
        self,
        u_init: np.ndarray | float | None = None,
        v_init: np.ndarray | float | None = None,
    ) -> FlowField:
        """Create a :class:`FlowField` with boundary conditions already applied.

        ``u_init``/``v_init`` may be scalars or arrays shaped like the physical
        face layouts, ``(nx+1, ny)`` and ``(nx, ny+1)`` respectively.
        """
        fields = FlowField.zeros(self.mesh)
        if u_init is not None:
            fields.u_phys[...] = u_init
        if v_init is not None:
            fields.v_phys[...] = v_init
        self.boundaries.apply_velocity(fields, predictor=True)
        self.boundaries.enforce_global_mass_balance(fields)
        self._mask_obstacle(fields.u, fields.v, dt=None)
        self._wrap_periodic_faces(fields.u, fields.v)

        # An arbitrary initial condition need not be discretely solenoidal, and
        # the fractional-step method assumes that it is: the Runge-Kutta stages
        # are convex combinations, so any initial divergence would decay only
        # geometrically instead of being removed. Project it out once, here.
        fields.p = self.project(fields.u, fields.v, dt=1.0)
        self._mask_obstacle(fields.u, fields.v, dt=None)
        self._wrap_periodic_faces(fields.u, fields.v)
        self.boundaries.apply_velocity(fields, predictor=False)
        return fields

    # ------------------------------------------------------------------ #
    # Spatial operators
    # ------------------------------------------------------------------ #
    def _corner_products(self, u: np.ndarray, v: np.ndarray):
        """Velocity components interpolated to cell corners.

        Returns ``(u_c, v_c)``, both of shape ``(nx+1, ny+1)`` and indexed
        ``[m-1, n-1]`` for the corner at ``x=xf[m-1]``, ``y=yf[n-1]``.  Sharing
        one corner evaluation between the two momentum equations is what makes
        the convective discretisation conservative.

        ``u`` is interpolated along y and ``v`` along x, each across a pair of
        cell centres.  A corner is *not* the midpoint of that pair once the
        mesh is stretched, so the weights come from the geometry; they collapse
        to the plain average when it is not.
        """
        nx, ny = self.mesh.shape
        m = self.metrics
        wy, wx = m.wy_corner, m.wx_corner

        u_lo, u_hi = u[1:nx + 2, 0:ny + 1], u[1:nx + 2, 1:ny + 2]
        v_lo, v_hi = v[0:nx + 1, 1:ny + 2], v[1:nx + 2, 1:ny + 2]
        if self.mesh.is_uniform:
            # Algebraically the same as the weighted form below at w = 1/2, but
            # written so the uniform mesh keeps the exact rounding it had
            # before stretching existed and the recorded baselines still hold.
            return 0.5 * (u_lo + u_hi), 0.5 * (v_lo + v_hi)
        return u_lo + wy * (u_hi - u_lo), v_lo + wx * (v_hi - v_lo)

    def _advection(self, u: np.ndarray, v: np.ndarray, blend: float):
        """Convective terms ``(A_u, A_v)`` in conservative form.

        Shapes are ``(nx+1, ny)`` for ``A_u`` (x-faces 1..nx+1) and
        ``(nx, ny+1)`` for ``A_v`` (y-faces 1..ny+1).  ``blend`` in ``(0, 1]``
        adds the donor-cell (upwind) correction of Griebel et al.; ``0`` is pure
        second-order central differencing.
        """
        nx, ny = self.mesh.shape
        m = self.metrics

        u_c, v_c = self._corner_products(u, v)
        uv = u_c * v_c
        # Cell-centred interpolants of each component along its own direction.
        # A cell centre *is* the midpoint of its own two faces however the mesh
        # is stretched, so this stays a plain average.
        u_cc = 0.5 * (u[0:nx + 2, :] + u[1:nx + 3, :])       # [m, j], m = 0..nx+1
        v_cc = 0.5 * (v[:, 0:ny + 2] + v[:, 1:ny + 3])       # [i, n], n = 0..ny+1

        # -- u momentum, faces m = 1..nx+1, rows j = 1..ny ------------------ #
        # Streamwise flux is differenced across the u face's own control
        # volume (centre to centre, hxu); the transverse one across the cell.
        uu_r = u_cc[1:nx + 2, 1:ny + 1]      # cell to the right of face m
        uu_l = u_cc[0:nx + 1, 1:ny + 1]      # cell to the left  of face m
        duudx = (uu_r ** 2 - uu_l ** 2) / m.hxu
        duvdy = (uv[0:nx + 1, 1:ny + 1] - uv[0:nx + 1, 0:ny]) / m.hy

        # -- v momentum, faces n = 1..ny+1, cols i = 1..nx ------------------ #
        vv_t = v_cc[1:nx + 1, 1:ny + 2]      # cell above face n
        vv_b = v_cc[1:nx + 1, 0:ny + 1]      # cell below face n
        dvvdy = (vv_t ** 2 - vv_b ** 2) / m.hyv
        duvdx = (uv[1:nx + 1, 0:ny + 1] - uv[0:nx, 0:ny + 1]) / m.hx

        if blend > 0.0:
            # Donor-cell correction: upwind-weighted differences of the
            # transported quantity, scaled by the magnitude of the transporting
            # velocity.  Reduces to first-order upwinding at blend = 1.
            # Each correction is differenced over the same control volume as
            # the central flux it corrects, so the spacings match those above.
            um, u0, up = u[0:nx + 1, 1:ny + 1], u[1:nx + 2, 1:ny + 1], u[2:nx + 3, 1:ny + 1]
            duudx += blend / m.hxu * (
                np.abs(uu_r) * (u0 - up) / 2.0 - np.abs(uu_l) * (um - u0) / 2.0
            )
            ud, uc0, uu_ = u[1:nx + 2, 0:ny], u[1:nx + 2, 1:ny + 1], u[1:nx + 2, 2:ny + 2]
            duvdy += blend / m.hy * (
                np.abs(v_c[0:nx + 1, 1:ny + 1]) * (uc0 - uu_) / 2.0
                - np.abs(v_c[0:nx + 1, 0:ny]) * (ud - uc0) / 2.0
            )

            vm, v0, vp = v[1:nx + 1, 0:ny + 1], v[1:nx + 1, 1:ny + 2], v[1:nx + 1, 2:ny + 3]
            dvvdy += blend / m.hyv * (
                np.abs(vv_t) * (v0 - vp) / 2.0 - np.abs(vv_b) * (vm - v0) / 2.0
            )
            vl, vc0, vr = v[0:nx, 1:ny + 2], v[1:nx + 1, 1:ny + 2], v[2:nx + 2, 1:ny + 2]
            duvdx += blend / m.hx * (
                np.abs(u_c[1:nx + 1, 0:ny + 1]) * (vc0 - vr) / 2.0
                - np.abs(u_c[0:nx, 0:ny + 1]) * (vl - vc0) / 2.0
            )

        return duudx + duvdy, duvdx + dvvdy

    def _diffusion(self, u: np.ndarray, v: np.ndarray):
        """Viscous terms ``(D_u, D_v)`` for constant kinematic viscosity.

        Written as a difference of two first derivatives rather than the
        familiar three-point second difference, because on a stretched mesh
        those are not the same operator: the inner gradients span the two cells
        flanking the face, and the outer difference spans the face's own
        control volume.  Collapsing that to ``(a - 2b + c)/h**2`` is what makes
        a stretched-grid viscous term first order.
        """
        nx, ny = self.mesh.shape
        m = self.metrics
        if self.mesh.is_uniform:
            # Same operator, kept in its original grouping so a uniform run
            # reproduces its recorded baselines bit for bit.
            dx2, dy2 = m.hx ** 2, m.hy ** 2
            u0 = u[1:nx + 2, 1:ny + 1]
            d2u = (
                (u[0:nx + 1, 1:ny + 1] - 2.0 * u0 + u[2:nx + 3, 1:ny + 1]) / dx2
                + (u[1:nx + 2, 0:ny] - 2.0 * u0 + u[1:nx + 2, 2:ny + 2]) / dy2
            )
            v0 = v[1:nx + 1, 1:ny + 2]
            d2v = (
                (v[0:nx, 1:ny + 2] - 2.0 * v0 + v[2:nx + 2, 1:ny + 2]) / dx2
                + (v[1:nx + 1, 0:ny + 1] - 2.0 * v0 + v[1:nx + 1, 2:ny + 3]) / dy2
            )
            return self.nu * d2u, self.nu * d2v

        sl = MeshMetrics._slice
        # u lives on x-faces m = 0..nx and on cell rows j = 0..ny-1.
        u0 = u[1:nx + 2, 1:ny + 1]
        gx_hi = (u[2:nx + 3, 1:ny + 1] - u0) / sl(m.hx_ext, 1, nx + 2, 0)
        gx_lo = (u0 - u[0:nx + 1, 1:ny + 1]) / sl(m.hx_ext, 0, nx + 1, 0)
        gy_hi = (u[1:nx + 2, 2:ny + 2] - u0) / sl(m.hyv, 1, ny + 1, 1)
        gy_lo = (u0 - u[1:nx + 2, 0:ny]) / sl(m.hyv, 0, ny, 1)
        d2u = (gx_hi - gx_lo) / m.hxu + (gy_hi - gy_lo) / m.hy

        # v lives on cell columns i = 0..nx-1 and on y-faces n = 0..ny.
        v0 = v[1:nx + 1, 1:ny + 2]
        hx_hi = (v[2:nx + 2, 1:ny + 2] - v0) / sl(m.hxu, 1, nx + 1, 0)
        hx_lo = (v0 - v[0:nx, 1:ny + 2]) / sl(m.hxu, 0, nx, 0)
        hy_hi = (v[1:nx + 1, 2:ny + 3] - v0) / sl(m.hy_ext, 1, ny + 2, 1)
        hy_lo = (v0 - v[1:nx + 1, 0:ny + 1]) / sl(m.hy_ext, 0, ny + 1, 1)
        d2v = (hx_hi - hx_lo) / m.hx + (hy_hi - hy_lo) / m.hyv
        return self.nu * d2u, self.nu * d2v

    def _diffusion_variable(self, u: np.ndarray, v: np.ndarray, nu_c: np.ndarray,
                            nu_corner: np.ndarray):
        """Viscous terms in stress-divergence form for a variable eddy viscosity.

        ``nu_c`` is cell-centred with the ghosted pressure shape; ``nu_corner``
        is corner-based with shape ``(nx+1, ny+1)``.
        """
        nx, ny = self.mesh.shape
        m = self.metrics
        sl = MeshMetrics._slice

        # Shear stress tau = nu * (du/dy + dv/dx), evaluated at the corners.
        # Both derivatives cross a face, so both use a centre-to-centre span.
        dudy_c = (u[1:nx + 2, 1:ny + 2] - u[1:nx + 2, 0:ny + 1]) / m.hyv
        dvdx_c = (v[1:nx + 2, 1:ny + 2] - v[0:nx + 1, 1:ny + 2]) / m.hxu
        tau = nu_corner * (dudy_c + dvdx_c)

        # Normal stresses live at cell centres, so they are differenced over a
        # cell width; the ghost columns extend that by one at each end.
        dudx_cc = (u[1:nx + 3, :] - u[0:nx + 2, :]) / m.hx_ext   # [m, j], m = 0..nx+1
        dvdy_cc = (v[:, 1:ny + 3] - v[:, 0:ny + 2]) / m.hy_ext   # [i, n], n = 0..ny+1

        d_u = (
            2.0 * (nu_c[1:nx + 2, 1:ny + 1] * dudx_cc[1:nx + 2, 1:ny + 1]
                   - nu_c[0:nx + 1, 1:ny + 1] * dudx_cc[0:nx + 1, 1:ny + 1]) / m.hxu
            + (tau[0:nx + 1, 1:ny + 1] - tau[0:nx + 1, 0:ny]) / m.hy
        )
        d_v = (
            (tau[1:nx + 1, 0:ny + 1] - tau[0:nx, 0:ny + 1]) / m.hx
            + 2.0 * (nu_c[1:nx + 1, 1:ny + 2] * dvdy_cc[1:nx + 1, 1:ny + 2]
                     - nu_c[1:nx + 1, 0:ny + 1] * dvdy_cc[1:nx + 1, 0:ny + 1]) / m.hyv
        )
        return d_u, d_v

    def momentum_rhs(self, u: np.ndarray, v: np.ndarray, blend: float = 0.0):
        """Right-hand side of the momentum equation, ``-A(u) + D(u) + f``.

        Dispatches to the fused Numba kernel when it is available and the
        viscosity is constant; the array path is used otherwise and produces
        identical numbers.

        .. note::
           On the Numba path the returned arrays are internal buffers that the
           next call overwrites.  :meth:`substep` consumes them immediately, so
           this is invisible there; any caller that needs to keep the values
           across calls must copy them.
        """
        if self._use_kernel:
            fx, fy = self.config.body_force
            self._kernel(
                u, v, self.mesh.dx, self.mesh.dy, self.nu, blend, fx, fy,
                self._rhs_u, self._rhs_v,
            )
            return self._rhs_u, self._rhs_v

        adv_u, adv_v = self._advection(u, v, blend)
        if self.turbulence is None:
            dif_u, dif_v = self._diffusion(u, v)
        else:
            nu_c, nu_corner = self.turbulence.eddy_viscosity(u, v, self.nu)
            dif_u, dif_v = self._diffusion_variable(u, v, nu_c, nu_corner)
        fx, fy = self.config.body_force
        return -adv_u + dif_u + fx, -adv_v + dif_v + fy

    # ------------------------------------------------------------------ #
    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Discrete divergence at cell centres, shape ``(nx, ny)``.

        A flux balance over the cell, so it divides by the cell's own width and
        height -- exact on any spacing, stretched or not.
        """
        nx, ny = self.mesh.shape
        m = self.metrics
        return (
            (u[2:nx + 2, 1:ny + 1] - u[1:nx + 1, 1:ny + 1]) / m.hx
            + (v[1:nx + 1, 2:ny + 2] - v[1:nx + 1, 1:ny + 1]) / m.hy
        )

    def max_divergence(self, fields: FlowField) -> float:
        """Largest absolute divergence over the fluid cells."""
        d = self.divergence(fields.u, fields.v)
        if self.has_obstacle:
            d = d[~self.solid]
        return float(np.abs(d).max()) if d.size else 0.0

    # ------------------------------------------------------------------ #
    def _mask_obstacle(self, u: np.ndarray, v: np.ndarray, dt: float | None,
                       reset: bool = False) -> None:
        """Zero the velocity on every face touching a solid cell (direct forcing).

        When ``dt`` is given, the momentum that had to be removed is accumulated
        as the reaction force the fluid exerts on the body.  Both maskings in a
        substep must be counted, because they carry different halves of the load:

        * after the predictor, the removed momentum is the **advective and
          viscous** flux that would have entered the body;
        * after the projection, the removed momentum is
          ``dt * grad(p)`` across each interface face.  Since solid cells hold
          ``p = 0``, that reduces to the neighbouring fluid pressure times the
          face area, with the sign of the outward normal -- i.e. exactly the
          discrete **pressure** surface integral.

        Counting only the second term drops the entire viscous contribution,
        which is roughly half the drag on a cylinder at Re = 20.
        """
        if not self.has_obstacle:
            return
        u_faces = u[self.u_upd]
        v_faces = v[self.v_upd]
        if dt is not None:
            u_area, v_area = self._u_cv_area, self._v_cv_area
            fx = self._momentum_sum(u_faces, self.u_face_solid, u_area) / dt
            fy = self._momentum_sum(v_faces, self.v_face_solid, v_area) / dt
            if reset:
                self.body_force_reaction = (fx, fy)
            else:
                prev = self.body_force_reaction
                self.body_force_reaction = (prev[0] + fx, prev[1] + fy)
            if self.body_masks:
                per_body = tuple(
                    (self._momentum_sum(u_faces, um, u_area) / dt,
                     self._momentum_sum(v_faces, vm, v_area) / dt)
                    for um, vm in zip(self.u_face_body, self.v_face_body)
                )
                if reset:
                    self.body_force_reactions = per_body
                else:
                    self.body_force_reactions = tuple(
                        (a[0] + b[0], a[1] + b[1])
                        for a, b in zip(self.body_force_reactions, per_body)
                    )
        u_faces[self.u_face_solid] = 0.0
        v_faces[self.v_face_solid] = 0.0

    def _wrap_periodic_faces(self, u: np.ndarray, v: np.ndarray) -> None:
        """Keep the duplicated far-end face of a periodic axis in sync."""
        nx, ny = self.mesh.shape
        if self.periodic_x:
            u[nx + 1, :] = u[1, :]
        if self.periodic_y:
            v[:, ny + 1] = v[:, 1]

    # ------------------------------------------------------------------ #
    # One projected forward-Euler substep -- the building block of every scheme
    # ------------------------------------------------------------------ #
    def substep(self, fields: FlowField, dt: float, blend: float = 0.0):
        """Advance ``(u, v)`` by one projected explicit Euler substep.

        Returns ``(u_new, v_new, p)`` as fresh arrays; ``fields`` is untouched.
        """
        u, v = fields.u, fields.v
        rhs_u, rhs_v = self.momentum_rhs(u, v, blend)

        us, vs = u.copy(), v.copy()
        us[self.u_upd] += dt * rhs_u[self.u_rhs_sel, :]
        vs[self.v_upd] += dt * rhs_v[:, self.v_rhs_sel]

        # Predictor boundary pass: prescribed faces reset, outflow extrapolated.
        star = FlowField(self.mesh, us, vs, fields.p, fields.t, fields.step)
        self.boundaries.apply_velocity(star, predictor=True)
        self.boundaries.enforce_global_mass_balance(star)
        self._mask_obstacle(us, vs, dt=dt, reset=True)
        self._wrap_periodic_faces(us, vs)

        # Pressure Poisson and projection.
        p = self.project(us, vs, dt)

        # Corrector boundary pass: ghosts refreshed, prescribed faces untouched.
        self._mask_obstacle(us, vs, dt=dt, reset=False)
        self._wrap_periodic_faces(us, vs)
        self.boundaries.apply_velocity(
            FlowField(self.mesh, us, vs, p, fields.t, fields.step), predictor=False
        )
        return us, vs, p

    # ------------------------------------------------------------------ #
    def project(self, u: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        """Remove the divergence from ``(u, v)`` in place; return the pressure.

        Solves ``lap(p) = div(u)/dt`` and subtracts ``dt*grad(p)`` from exactly
        the faces the Poisson stencil includes.  ``dt`` only sets the scaling of
        ``p``: the velocity correction ``dt*grad(p)`` is independent of it.

        An iterative solver is handed the previous substep's pressure to start
        from.  The field barely moves between substeps, so the guess is already
        close: measured on the cylinder at 256x128, multigrid-preconditioned CG
        needs 10.4 iterations from the previous pressure against 13.0 from zero.
        This cannot change the answer -- the solve runs to the same tolerance on
        the same right-hand side either way -- only how long it takes to get
        there.
        """
        nx, ny = self.mesh.shape
        p_interior = self.pressure_solver.solve(
            self.divergence(u, v) / dt, self._p_guess,
        )
        if self.pressure_solver.warm_startable:
            self._p_guess = p_interior
        p = self.mesh.zeros_p()
        p[1:nx + 1, 1:ny + 1] = p_interior
        # The pressure ghost layer must be filled *before* projecting: on a
        # periodic axis the first face reads the ghost cell, and it has to hold
        # the value from the opposite end of the domain rather than zero.
        self.boundaries.apply_pressure(FlowField(self.mesh, u, v, p))

        m = self.metrics
        lo_u, hi_u = self.u_upd[0].start, self.u_upd[0].stop
        lo_v, hi_v = self.v_upd[1].start, self.v_upd[1].stop
        # The gradient crosses a face, so it divides by the centre-to-centre
        # distance -- the same spacing the Poisson stencil was assembled with,
        # which is what keeps the projected field divergence-free to round-off.
        u[self.u_upd] -= dt * (
            p[lo_u:hi_u, 1:ny + 1] - p[lo_u - 1:hi_u - 1, 1:ny + 1]
        ) / MeshMetrics._slice(m.hxu, lo_u - 1, hi_u - 1, 0)
        v[self.v_upd] -= dt * (
            p[1:nx + 1, lo_v:hi_v] - p[1:nx + 1, lo_v - 1:hi_v - 1]
        ) / MeshMetrics._slice(m.hyv, lo_v - 1, hi_v - 1, 1)
        return p

    # ------------------------------------------------------------------ #
    def step(self, fields: FlowField, dt: float) -> FlowField:
        """Advance the flow by ``dt`` using the configured time scheme.

        The Runge--Kutta stages are convex combinations of projected states.
        Since the projection is affine and the stage weights sum to one, every
        intermediate state is itself divergence-free and satisfies the (linear)
        boundary conditions -- so no extra projection of the combination is
        needed.
        """
        blend = self._blend
        scheme = self.config.time_scheme
        u0, v0 = fields.u, fields.v

        if scheme is TimeScheme.EULER:
            u, v, p = self.substep(fields, dt, blend)

        elif scheme is TimeScheme.RK2:
            u1, v1, _ = self.substep(fields, dt, blend)
            stage = FlowField(self.mesh, u1, v1, fields.p, fields.t + dt, fields.step)
            u2, v2, p = self.substep(stage, dt, blend)
            u = 0.5 * u0 + 0.5 * u2
            v = 0.5 * v0 + 0.5 * v2

        elif scheme is TimeScheme.RK3:
            # SSP Runge--Kutta 3 (Shu & Osher).  Unlike Euler and RK2 its
            # stability region contains a genuine interval of the imaginary
            # axis, which is what makes central-difference advection stable.
            u1, v1, _ = self.substep(fields, dt, blend)
            s1 = FlowField(self.mesh, u1, v1, fields.p, fields.t + dt, fields.step)
            ue, ve, _ = self.substep(s1, dt, blend)
            u2 = 0.75 * u0 + 0.25 * ue
            v2 = 0.75 * v0 + 0.25 * ve
            s2 = FlowField(self.mesh, u2, v2, fields.p, fields.t + 0.5 * dt, fields.step)
            ue2, ve2, p = self.substep(s2, dt, blend)
            u = (1.0 / 3.0) * u0 + (2.0 / 3.0) * ue2
            v = (1.0 / 3.0) * v0 + (2.0 / 3.0) * ve2
        else:  # pragma: no cover - TimeScheme is exhaustive
            raise ValueError(f"unknown time scheme {scheme!r}")

        out = FlowField(self.mesh, u, v, p, fields.t + dt, fields.step + 1)
        if scheme is not TimeScheme.EULER:
            # Convex combinations preserve both properties exactly; re-applying
            # them costs little and guards against round-off drift.
            self.boundaries.apply_velocity(out, predictor=False)
            self._wrap_periodic_faces(out.u, out.v)
        return out

    # ------------------------------------------------------------------ #
    def set_upwind_blend(self, blend: float) -> None:
        """Set the donor-cell blending factor used by the next substep."""
        self._blend = float(np.clip(blend, 0.0, 1.0))

    def upwind_blend_for(self, fields: FlowField, dt: float) -> float:
        """Blending factor for the current state.

        With ``advection_scheme='central'`` this is zero.  With ``'upwind'`` and
        no explicit ``upwind_blend`` in the configuration it follows the usual
        recommendation of tracking the local Courant number, which keeps the
        added numerical diffusion at the minimum the stability of the scheme
        requires.
        """
        if self.config.advection_scheme is AdvectionScheme.CENTRAL:
            return 0.0
        if self.config.upwind_blend is not None:
            return self.config.upwind_blend
        umax, vmax = fields.max_velocity()
        # The blend has to cover the worst cell, which on a stretched mesh is
        # the smallest one.
        m = self.metrics
        courant = max(umax * dt / m.min_hx, vmax * dt / m.min_hy)
        return float(min(1.0, 1.2 * courant))
