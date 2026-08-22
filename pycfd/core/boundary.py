"""Boundary conditions for the staggered (MAC) grid.

On a staggered grid a wall touches the three variables differently, and getting
this wrong is the most common source of a plausible-looking but incorrect
solution.  The three cases are:

*Normal velocity* sits exactly **on** the boundary face, so a Dirichlet
condition is applied by assignment -- no interpolation, no ghost value.

*Tangential velocity* sits half a cell **outside** and half a cell **inside**
the wall.  A Dirichlet value ``t_wall`` is therefore imposed through the ghost
value::

    0.5 * (t_ghost + t_interior) = t_wall   ==>   t_ghost = 2*t_wall - t_interior

which keeps the wall-adjacent viscous stencil second-order accurate.  A
zero-gradient (Neumann) condition is ``t_ghost = t_interior``.

*Pressure* gets a homogeneous Neumann condition ``dp/dn = 0`` at every
non-periodic boundary.  This is not a modelling choice but a consistency
requirement: the projection step never corrects a boundary face (its normal
velocity is prescribed), so the discrete Poisson operator must likewise carry
no flux through that face.  The assembly in :mod:`pycfd.core.pressure` simply
drops the corresponding coefficient.

Because every boundary is Neumann (or periodic), the pressure system is
singular by one constant mode.  That is handled in :mod:`pycfd.core.pressure`,
and it requires the *net mass flux through the domain boundary to vanish* --
enforced here by :meth:`BoundaryManager.enforce_global_mass_balance`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BCKind, BCSpec, WALLS
from .fields import FlowField
from .mesh import StructuredMesh

#: Below this total outflow the outlet rescaling is replaced by a uniform
#: additive correction (avoids dividing by a vanishing flux at start-up).
_MIN_OUTFLOW_FLUX = 1.0e-12


# --------------------------------------------------------------------------- #
# Wall index bookkeeping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WallIndex:
    """Array indices describing one wall of the staggered grid.

    ``normal`` refers to the velocity component perpendicular to the wall (``u``
    for left/right, ``v`` for bottom/top); ``tangential`` to the other one.
    """

    axis: int        # 0 for left/right, 1 for bottom/top
    n_face: int      # index of the boundary face in the normal-velocity array
    n_in: int        # first interior face of the normal-velocity array
    n_ghost: int     # ghost index of the normal-velocity array
    t_ghost: int     # ghost index of the tangential-velocity array
    t_in: int        # first interior index of the tangential-velocity array
    p_ghost: int     # ghost index of the pressure array
    p_in: int        # first interior cell index of the pressure array
    outward: int     # +1 if the outward normal points along +x/+y, else -1


def wall_index(wall: str, mesh: StructuredMesh) -> WallIndex:
    """Build the :class:`WallIndex` for ``wall`` on ``mesh``."""
    nx, ny = mesh.shape
    if wall == "left":
        return WallIndex(0, 1, 2, 0, 0, 1, 0, 1, -1)
    if wall == "right":
        return WallIndex(0, nx + 1, nx, nx + 2, nx + 1, nx, nx + 1, nx, +1)
    if wall == "bottom":
        return WallIndex(1, 1, 2, 0, 0, 1, 0, 1, -1)
    if wall == "top":
        return WallIndex(1, ny + 1, ny, ny + 2, ny + 1, ny, ny + 1, ny, +1)
    raise ValueError(f"unknown wall {wall!r}, expected one of {WALLS}")


def _line(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """View of one row/column of ``arr`` along ``axis`` -- assignable."""
    return arr[idx, :] if axis == 0 else arr[:, idx]


def _set(arr: np.ndarray, axis: int, idx: int, value) -> None:
    """Assign one row/column of ``arr`` along ``axis``."""
    if axis == 0:
        arr[idx, :] = value
    else:
        arr[:, idx] = value


# --------------------------------------------------------------------------- #
# Boundary condition classes
# --------------------------------------------------------------------------- #
class BoundaryCondition:
    """Base class.  One instance is bound to one wall of the domain.

    Sub-classes override :meth:`normal_velocity` and :meth:`tangential_ghost`;
    the shared :meth:`apply` then writes the staggered arrays.
    """

    kind: BCKind
    #: Pressure treatment implied by this condition, consumed by the Poisson
    #: assembly.  Every physical wall here is ``"neumann"``.
    pressure_kind: str = "neumann"
    #: True for conditions whose normal velocity is derived from the interior
    #: rather than prescribed; those must not be refreshed after the projection.
    derives_normal: bool = False
    #: False when the boundary face is a *solved* degree of freedom rather than
    #: an imposed value -- true only of :class:`PressureOutlet`, where the
    #: outflow velocity is whatever the pressure field produces.
    prescribes_normal: bool = True

    def __init__(self, wall: str, spec: BCSpec | None = None) -> None:
        if wall not in WALLS:
            raise ValueError(f"unknown wall {wall!r}, expected one of {WALLS}")
        self.wall = wall
        self.spec = spec or BCSpec()

    # -- hooks --------------------------------------------------------- #
    def normal_velocity(self, mesh: StructuredMesh, wi: WallIndex,
                        normal: np.ndarray) -> np.ndarray | float:
        """Value to impose on the boundary face of the normal component."""
        return 0.0

    def tangential_ghost(self, mesh: StructuredMesh, wi: WallIndex,
                         tangential: np.ndarray) -> np.ndarray:
        """Ghost values of the tangential component (default: no-slip, zero)."""
        return -_line(tangential, wi.axis, wi.t_in)

    def normal_ghost(self, mesh: StructuredMesh, wi: WallIndex,
                     normal: np.ndarray) -> np.ndarray:
        """Ghost of the normal component: reflect about the boundary face value."""
        return 2.0 * _line(normal, wi.axis, wi.n_face) - _line(normal, wi.axis, wi.n_in)

    # -- driver -------------------------------------------------------- #
    def apply(self, fields: FlowField, mesh: StructuredMesh,
              predictor: bool = True) -> None:
        """Write this wall's contribution into ``fields``.

        Parameters
        ----------
        predictor:
            ``True`` before the pressure solve, ``False`` after it.  Conditions
            that *derive* their normal velocity from the interior (outflow) are
            only allowed to do so in the predictor pass -- recomputing them
            after the projection would reintroduce a divergence in the
            boundary-adjacent cell.
        """
        wi = wall_index(self.wall, mesh)
        normal, tangential = (fields.u, fields.v) if wi.axis == 0 else (fields.v, fields.u)

        if self.prescribes_normal and (predictor or not self.derives_normal):
            _set(normal, wi.axis, wi.n_face, self.normal_velocity(mesh, wi, normal))

        # Ghost of the normal component, so that any stencil reaching outside the
        # domain sees a consistent extension of the field.
        _set(normal, wi.axis, wi.n_ghost, self.normal_ghost(mesh, wi, normal))

        _set(tangential, wi.axis, wi.t_ghost, self.tangential_ghost(mesh, wi, tangential))

    def apply_pressure(self, fields: FlowField, mesh: StructuredMesh) -> None:
        """Fill the pressure ghost layer (homogeneous Neumann)."""
        wi = wall_index(self.wall, mesh)
        _set(fields.p, wi.axis, wi.p_ghost, _line(fields.p, wi.axis, wi.p_in))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(wall={self.wall!r})"


class NoSlip(BoundaryCondition):
    """Stationary solid wall: both velocity components vanish at the wall."""

    kind = BCKind.NO_SLIP


class MovingWall(BoundaryCondition):
    """Solid wall translating in its own plane (the lid-driven-cavity lid).

    The tangential wall speed is ``spec.velocity``: ``+x`` for the horizontal
    walls, ``+y`` for the vertical ones.  The normal component is still zero.
    """

    kind = BCKind.MOVING_WALL

    def tangential_ghost(self, mesh, wi, tangential):
        """Ghost value that makes the wall-tangential velocity the wall speed."""
        return 2.0 * self.spec.velocity - _line(tangential, wi.axis, wi.t_in)


class Inlet(BoundaryCondition):
    """Prescribed inflow: uniform or parabolic normal velocity, no slip tangentially.

    ``spec.velocity`` is the uniform speed, or the *centreline peak* for a
    parabolic profile.  The sign is interpreted as "into the domain", so the
    same spec works on any wall.
    """

    kind = BCKind.INLET

    def normal_velocity(self, mesh, wi, normal):
        """Prescribed inflow profile on the boundary face."""
        # Inflow points inward, i.e. opposite to the outward wall normal.
        speed = -wi.outward * self.spec.velocity
        if self.spec.profile == "uniform":
            return speed
        # Parabolic: zero at the two walls bounding the inlet face.
        coord, extent = (mesh.yc, mesh.ly) if wi.axis == 0 else (mesh.xc, mesh.lx)
        prof = 4.0 * coord * (extent - coord) / extent ** 2
        full = np.zeros(normal.shape[1] if wi.axis == 0 else normal.shape[0])
        full[1:1 + coord.size] = speed * prof     # interior rows/cols only
        return full

    def tangential_ghost(self, mesh, wi, tangential):
        """Zero tangential velocity at the inlet."""
        return -_line(tangential, wi.axis, wi.t_in)


class Outlet(BoundaryCondition):
    """Zero-gradient (convective) outflow.

    Both components are extrapolated from the interior.  The resulting outflow
    is then rescaled globally by
    :meth:`BoundaryManager.enforce_global_mass_balance` so that the all-Neumann
    pressure problem stays compatible.
    """

    kind = BCKind.OUTLET
    derives_normal = True

    def normal_velocity(self, mesh, wi, normal):
        """Zero-gradient extrapolation of the normal component."""
        return _line(normal, wi.axis, wi.n_in).copy()

    def tangential_ghost(self, mesh, wi, tangential):
        """Zero-gradient extrapolation of the tangential component."""
        return _line(tangential, wi.axis, wi.t_in).copy()


class PressureOutlet(Outlet):
    """Outflow that holds the **pressure** fixed instead of the velocity.

    ``Outlet`` prescribes the outflow velocity by extrapolation and leaves the
    pressure to float (``dp/dn = 0``); the pressure level of the whole domain is
    then arbitrary and has to be pinned at some reference cell.  A pressure
    outlet inverts that: it imposes ``p = p_ref`` on the boundary face and lets
    the outflow velocity be whatever the pressure field produces.  That is the
    physically meaningful condition for external aerodynamics and any open
    domain -- the far field is at a known pressure, not at a known velocity.

    Three consequences follow, all handled automatically:

    *The Poisson operator stops being singular.*  With a Dirichlet value
    anywhere the constant null space disappears, so no reference cell is pinned
    and no mean is subtracted: the pressure returned is an absolute field, and
    differences across the body are directly meaningful.

    *The outlet face becomes an unknown.*  It is added to the set of faces the
    projection corrects, which is why :attr:`prescribes_normal` is ``False`` --
    nothing may overwrite it once the pressure solve has set it.

    *Mass conservation becomes automatic.*  Every fluid cell is driven to zero
    divergence, so by the divergence theorem the net boundary flux is exactly
    zero; the outflow matches the inflow without the global rescaling that
    :class:`Outlet` needs.

    The condition is imposed at the boundary **face**, not at the last cell
    centre, through the ghost value ``p_ghost = 2*p_ref - p_interior``.  That
    keeps it second-order accurate and -- more importantly -- leaves the
    divergence equation of the last cell intact, so the cell adjacent to the
    outlet is as divergence-free as any other.
    """

    kind = BCKind.PRESSURE_OUTLET
    pressure_kind = "dirichlet"
    #: The outflow velocity is solved for, not imposed.
    prescribes_normal = False
    derives_normal = False

    @property
    def p_ref(self) -> float:
        """Pressure held on the boundary face."""
        return self.spec.p_ref

    def normal_ghost(self, mesh, wi, normal):
        """Zero-gradient extension: the face itself is a solved unknown."""
        return _line(normal, wi.axis, wi.n_face).copy()

    def apply_pressure(self, fields: FlowField, mesh: StructuredMesh) -> None:
        """Ghost value that places ``p_ref`` exactly on the boundary face.

        The face value is the mean of the ghost and the first interior cell, so
        ``p_ghost = 2*p_ref - p_interior`` puts ``p_ref`` on the face itself.
        This is the same relation the Poisson assembly encodes, which is what
        makes the projection and the pressure solve consistent.
        """
        wi = wall_index(self.wall, mesh)
        _set(fields.p, wi.axis, wi.p_ghost,
             2.0 * self.p_ref - _line(fields.p, wi.axis, wi.p_in))


class Symmetry(BoundaryCondition):
    """Slip / symmetry plane: zero normal velocity, zero normal gradient of the tangential one."""

    kind = BCKind.SYMMETRY

    def tangential_ghost(self, mesh, wi, tangential):
        """Even reflection: zero normal gradient of the tangential component."""
        return _line(tangential, wi.axis, wi.t_in).copy()


class Periodic(BoundaryCondition):
    """Marker for a periodic wall.

    Periodicity couples two opposite walls at once, so it cannot be applied wall
    by wall; :class:`BoundaryManager` handles it for the whole axis.  Calling
    :meth:`apply` on this class is a no-op by design.
    """

    kind = BCKind.PERIODIC
    pressure_kind = "periodic"

    def apply(self, fields, mesh, predictor=True):
        """No-op: periodic wrapping is applied for the whole axis at once."""
        return

    def apply_pressure(self, fields, mesh):
        """No-op: see :meth:`apply`."""
        return


_BC_REGISTRY: dict[BCKind, type[BoundaryCondition]] = {
    BCKind.NO_SLIP: NoSlip,
    BCKind.MOVING_WALL: MovingWall,
    BCKind.INLET: Inlet,
    BCKind.OUTLET: Outlet,
    BCKind.PRESSURE_OUTLET: PressureOutlet,
    BCKind.PERIODIC: Periodic,
    BCKind.SYMMETRY: Symmetry,
}


def make_boundary(wall: str, spec: BCSpec) -> BoundaryCondition:
    """Instantiate the boundary condition described by ``spec`` for ``wall``."""
    try:
        cls = _BC_REGISTRY[spec.kind]
    except KeyError:  # pragma: no cover - BCKind is exhaustive
        raise ValueError(f"unsupported boundary kind {spec.kind!r}") from None
    return cls(wall, spec)


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #
class BoundaryManager:
    """Applies all four wall conditions, periodic wrapping and mass balance."""

    def __init__(self, boundary_config: dict[str, BCSpec], mesh: StructuredMesh) -> None:
        self.mesh = mesh
        self.conditions = {w: make_boundary(w, boundary_config[w]) for w in WALLS}
        self.periodic_x = self.conditions["left"].kind is BCKind.PERIODIC
        self.periodic_y = self.conditions["bottom"].kind is BCKind.PERIODIC
        # Only a velocity Outlet needs the global flux correction.  A pressure
        # outlet conserves mass by construction (see :class:`PressureOutlet`),
        # and rescaling its face would fight the projection that just set it.
        self._outlets = [w for w, bc in self.conditions.items()
                         if bc.kind is BCKind.OUTLET]
        self._inlets = [w for w, bc in self.conditions.items() if isinstance(bc, Inlet)]

    # ------------------------------------------------------------------ #
    def apply_velocity(self, fields: FlowField, predictor: bool = True) -> None:
        """Fill every velocity boundary face and ghost layer."""
        for bc in self.conditions.values():
            bc.apply(fields, self.mesh, predictor=predictor)
        self._wrap_velocity(fields)

    def apply_pressure(self, fields: FlowField) -> None:
        """Fill the pressure ghost layer (used by post-processing and plotting)."""
        for bc in self.conditions.values():
            bc.apply_pressure(fields, self.mesh)
        self._wrap_pressure(fields)

    # ------------------------------------------------------------------ #
    def _wrap_velocity(self, fields: FlowField) -> None:
        """Copy interior data into the ghosts of a periodic axis.

        On a periodic axis the first and last physical faces are the *same*
        face, so they are kept identical; the ghosts then take the values of the
        opposite end of the domain.
        """
        nx, ny = self.mesh.shape
        u, v = fields.u, fields.v

        if self.periodic_x:
            u[nx + 1, :] = u[1, :]      # x=lx is the same face as x=0
            u[0, :] = u[nx, :]          # ghost at x=-dx  <- face at x=lx-dx
            u[nx + 2, :] = u[2, :]      # ghost at x=lx+dx <- face at x=dx
            v[0, :] = v[nx, :]          # ghost column <- last interior column
            v[nx + 1, :] = v[1, :]

        if self.periodic_y:
            v[:, ny + 1] = v[:, 1]
            v[:, 0] = v[:, ny]
            v[:, ny + 2] = v[:, 2]
            u[:, 0] = u[:, ny]
            u[:, ny + 1] = u[:, 1]

    def _wrap_pressure(self, fields: FlowField) -> None:
        nx, ny = self.mesh.shape
        p = fields.p
        if self.periodic_x:
            p[0, :] = p[nx, :]
            p[nx + 1, :] = p[1, :]
        if self.periodic_y:
            p[:, 0] = p[:, ny]
            p[:, ny + 1] = p[:, 1]

    # ------------------------------------------------------------------ #
    def enforce_global_mass_balance(self, fields: FlowField) -> None:
        """Rescale outflow so the net boundary flux vanishes.

        The pressure problem is pure Neumann, hence solvable only if the
        divergence integrates to zero over the domain -- equivalently, if total
        inflow equals total outflow.  With a zero-gradient outlet that balance
        does not hold automatically, so the outlet faces are corrected here.
        Domains without an outlet (closed boxes, fully periodic) need nothing.
        """
        if not self._outlets:
            return

        nx, ny = self.mesh.shape
        u, v = fields.u, fields.v
        uniform = self.mesh.is_uniform
        # A face on a left/right wall is one cell tall, and on a stretched mesh
        # the cells differ, so the flux through a wall is a height-weighted sum
        # rather than a sum times one height.
        if uniform:
            widths = (self.mesh.dy, self.mesh.dx)
        else:
            widths = (self.mesh.dy_cells, self.mesh.dx_cells)

        influx = 0.0        # volume flux entering the domain, positive inward
        outflux = 0.0       # volume flux leaving through the outlets
        outlet_faces: list[tuple[np.ndarray, np.ndarray | float, int]] = []

        for wall, bc in self.conditions.items():
            if bc.kind is BCKind.PERIODIC:
                continue
            wi = wall_index(wall, self.mesh)
            normal = u if wi.axis == 0 else v
            face = _line(normal, wi.axis, wi.n_face)
            face = face[1:ny + 1] if wi.axis == 0 else face[1:nx + 1]
            width = widths[wi.axis]
            flux_out = wi.outward * (                          # >0 leaves domain
                float(face.sum()) * width if uniform
                else float((face * width).sum())
            )

            if wall in self._outlets:
                outflux += flux_out
                outlet_faces.append((face, width, wi.outward))
            else:
                influx -= flux_out

        if not outlet_faces:
            return

        if outflux > _MIN_OUTFLOW_FLUX:
            # Multiplicative correction preserves the shape of the outflow profile.
            scale = influx / outflux
            for face, _, _ in outlet_faces:
                face *= scale
        else:
            # Degenerate start-up (little or no outflow yet, so a multiplicative
            # rescale would be ill-conditioned or sign-flipping): close the
            # remaining deficit with a uniform additive correction instead.
            total_area = sum(
                face.size * width if uniform else float(width.sum())
                for face, width, _ in outlet_faces
            )
            deficit = influx - outflux
            for face, _, outward in outlet_faces:
                face += outward * deficit / total_area

    # ------------------------------------------------------------------ #
    def pressure_boundary_kinds(self) -> dict[str, str]:
        """Map wall -> pressure treatment, consumed by the Poisson assembly."""
        return {w: bc.pressure_kind for w, bc in self.conditions.items()}

    def dirichlet_pressure(self) -> dict[str, float]:
        """Walls holding the pressure fixed, mapped to their reference value.

        Non-empty means the Poisson operator is non-singular.
        """
        return {w: bc.p_ref for w, bc in self.conditions.items()
                if bc.pressure_kind == "dirichlet"}

    def __repr__(self) -> str:
        inner = ", ".join(f"{w}={type(bc).__name__}" for w, bc in self.conditions.items())
        return f"BoundaryManager({inner})"
