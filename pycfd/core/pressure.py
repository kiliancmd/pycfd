"""Pressure Poisson equation: assembly and linear solvers.

The projection step needs the solution of

    lap(p) = div(u*) / dt

subject to ``dp/dn = 0`` on every solid or inflow boundary, ``p = p_ref`` on a
pressure outlet, and periodic wrapping where requested.

Discrete consistency
--------------------
The single most important property of this module is that the assembled
Laplacian is *exactly* ``div . grad`` for the same face set that the divergence
and projection operators in :mod:`pycfd.core.solver` use.  Concretely, a face is
included in the stencil **iff** the projection is allowed to correct the
velocity on it.  A face is excluded when it lies on a non-periodic domain
boundary (its normal velocity is prescribed) or when it borders an obstacle
cell (its normal velocity is masked to zero).  Because the two operators share
the same face set, the corrected velocity is divergence-free to the accuracy of
the linear solve rather than to the accuracy of the discretisation.

Singularity
-----------
With Neumann conditions everywhere the operator has a one-dimensional null
space (the constant).  Two things follow:

1. The right-hand side must be *compatible*: it has to integrate to zero over
   the fluid.  (The operator's row sums vanish only to round-off -- the
   diagonal is accumulated face by face while each off-diagonal is a single
   ``1/h**2`` -- but that discrepancy is ~1e-16 relative and far below any
   solver tolerance.)  This is guaranteed physically by the global mass balance
   enforced in :mod:`pycfd.core.boundary`, and enforced numerically here by
   subtracting the fluid mean of the right-hand side.
2. One degree of freedom must be fixed.  The direct solver pins a single
   reference cell.  That does not perturb the answer: the rows of the Neumann
   Laplacian sum to zero, so once the other ``N-1`` equations hold and the
   right-hand side sums to zero, the pinned cell's equation is satisfied
   automatically.  Only gradients of ``p`` ever enter the momentum equation, so
   the additive constant is irrelevant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..config import (
    DEFAULT_MG_SWEEPS,
    DEFAULT_POISSON_MAXITER,
    DEFAULT_POISSON_TOL,
    DEFAULT_SOR_OMEGA,
    PressureSolver,
)
from .mesh import StructuredMesh

log = logging.getLogger(__name__)

#: Right-hand-side norm below which a solve is treated as trivially zero.
_ZERO_RHS_NORM = 1.0e-300


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
@dataclass
class PoissonSystem:
    """Assembled Poisson operator plus the metadata the solvers need."""

    A: sp.csr_matrix           #: symmetric (unpinned) operator, shape (N, N)
    shape: tuple[int, int]     #: (nx, ny) cell counts
    fluid: np.ndarray          #: flat bool mask of solvable cells, length N
    singular: bool             #: True when the operator has a constant null space
    pin: int                   #: reference cell index used to fix the constant
    #: Constant contribution of Dirichlet boundaries, added to the right-hand
    #: side at solve time.  Zero everywhere for an all-Neumann problem.
    bc_rhs: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.bc_rhs is None:
            self.bc_rhs = np.zeros(self.A.shape[0])

    @property
    def n(self) -> int:
        """Number of unknowns."""
        return self.A.shape[0]


def _neighbor_indices(idx: np.ndarray, axis: int, shift: int, periodic: bool) -> np.ndarray:
    """Flat index of the neighbour ``shift`` cells away along ``axis``.

    Entries that would leave a non-periodic domain are marked ``-1``.
    """
    nb = np.roll(idx, -shift, axis=axis)
    if not periodic:
        edge = [slice(None), slice(None)]
        edge[axis] = -1 if shift > 0 else 0
        nb[tuple(edge)] = -1
    return nb


def assemble_poisson_matrix(
    mesh: StructuredMesh,
    periodic_x: bool = False,
    periodic_y: bool = False,
    solid_mask: np.ndarray | None = None,
    dirichlet: dict[str, float] | None = None,
) -> PoissonSystem:
    """Assemble the 5-point Neumann/periodic Laplacian once, for re-use every step.

    Parameters
    ----------
    mesh:
        Uniform structured mesh.
    periodic_x, periodic_y:
        Whether the corresponding axis wraps.
    solid_mask:
        Boolean ``(nx, ny)`` array, ``True`` inside an obstacle.  Solid cells
        are given the trivial equation ``p = 0`` and every face they share with
        a fluid cell is dropped from the stencil, matching the zero normal
        velocity imposed there.
    dirichlet:
        Walls holding the pressure fixed, mapped to their reference value, e.g.
        ``{"right": 0.0}`` for a pressure outlet.  Such a face is *kept* in the
        stencil with the ghost relation ``p_ghost = 2*p_ref - p_cell``, which
        places ``p_ref`` on the face itself and contributes ``-2/h^2`` to the
        diagonal and ``-2*p_ref/h^2`` to the right-hand side.

        Note what this deliberately does **not** do: it does not overwrite the
        row with an identity ``p = p_ref``.  Doing so would anchor the pressure
        half a cell inside the domain instead of on the boundary, drop to first
        order, and -- decisively -- discard that cell's divergence equation, so
        the cells along the outlet would no longer be divergence-free.  Keeping
        the row intact preserves the solver's central invariant.
    """
    mesh.require_uniform("pressure Poisson assembly")
    nx, ny = mesh.shape
    dx, dy = mesh.dx, mesh.dy
    n = nx * ny

    solid = (
        np.zeros((nx, ny), dtype=bool)
        if solid_mask is None
        else np.asarray(solid_mask, dtype=bool)
    )
    if solid.shape != (nx, ny):
        raise ValueError(f"solid_mask must have shape {(nx, ny)}, got {solid.shape}")
    fluid = ~solid

    idx = np.arange(n, dtype=np.int64).reshape(nx, ny)
    diag = np.zeros((nx, ny), dtype=float)
    bc = np.zeros((nx, ny), dtype=float)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    dirichlet = dict(dirichlet or {})
    # Each (axis, shift) leaves the domain through exactly one wall, so the
    # boundary faces of that direction are precisely that wall's faces.
    directions = (
        (0, -1, dx, periodic_x, "left"),
        (0, +1, dx, periodic_x, "right"),
        (1, -1, dy, periodic_y, "bottom"),
        (1, +1, dy, periodic_y, "top"),
    )
    for axis, shift, h, periodic, wall in directions:
        nb = _neighbor_indices(idx, axis, shift, periodic)
        # A face contributes only if both sides are fluid cells inside the domain.
        nb_is_fluid = np.zeros_like(nb, dtype=bool)
        inside = nb >= 0
        nb_is_fluid[inside] = fluid.ravel()[nb[inside]]
        active = fluid & inside & nb_is_fluid

        coeff = 1.0 / (h * h)
        rows.append(idx[active])
        cols.append(nb[active])
        data.append(np.full(int(active.sum()), coeff))
        diag -= coeff * active          # one -1/h^2 per active face

        if wall in dirichlet:
            # Domain-boundary faces of a Dirichlet wall stay in the stencil,
            # with the ghost eliminated as p_ghost = 2*p_ref - p_cell.
            on_wall = fluid & ~inside
            diag -= 2.0 * coeff * on_wall
            bc -= 2.0 * coeff * dirichlet[wall] * on_wall

    # Fluid cells with no active face at all (fully enclosed by obstacle/boundary)
    # would leave a zero row; give them the trivial equation p = 0.
    isolated = fluid & (diag == 0.0)
    if isolated.any():
        log.warning(
            "%d fluid cell(s) are completely enclosed by solid or boundary faces; "
            "their pressure is set to zero.", int(isolated.sum()),
        )

    solvable = fluid & ~isolated
    diag = np.where(solvable, diag, 1.0)     # solid + isolated rows: identity
    bc = np.where(solvable, bc, 0.0)

    rows.append(idx.ravel())
    cols.append(idx.ravel())
    data.append(diag.ravel())

    A = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    A.sum_duplicates()

    # A single Dirichlet wall removes the constant null space: the pressure level
    # is then set by physics, so no reference cell is pinned and no mean is
    # subtracted, and the field that comes back is absolute rather than relative.
    singular = not dirichlet
    pin = int(np.flatnonzero(solvable.ravel())[0]) if solvable.any() else 0

    return PoissonSystem(A, (nx, ny), solvable.ravel().copy(), singular, pin,
                         bc.ravel().copy())


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
class PoissonSolverBase:
    """Common right-hand-side conditioning and result normalisation."""

    name = "base"

    #: Whether :meth:`solve` gets anything out of the ``p0`` argument.  The
    #: pressure moves little from one substep to the next, so an iterative
    #: solver handed the previous field starts much closer than it would from
    #: zero; a direct solve takes the same two triangular sweeps either way, and
    #: :meth:`~pycfd.core.solver.ProjectionSolver.project` skips keeping the
    #: previous field around when nothing will read it.  Accuracy does not
    #: depend on this: every solver iterates to the same tolerance on the same
    #: right-hand side, so the guess changes only how long that takes.
    warm_startable = False

    def __init__(self, system: PoissonSystem, tol: float = DEFAULT_POISSON_TOL,
                 maxiter: int = DEFAULT_POISSON_MAXITER) -> None:
        self.system = system
        self.tol = tol
        self.maxiter = maxiter
        self.last_iterations = 0
        self.last_residual = 0.0

    # -- helpers ------------------------------------------------------- #
    def _prepare_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """Flatten, zero the non-solvable rows and project onto the compatible subspace."""
        b = np.asarray(rhs, dtype=float).ravel() + self.system.bc_rhs
        fluid = self.system.fluid
        b[~fluid] = 0.0
        if self.system.singular:
            b[fluid] -= b[fluid].mean()
        return b

    def _normalize(self, p: np.ndarray) -> np.ndarray:
        """Remove the arbitrary additive constant and zero the non-solvable rows."""
        fluid = self.system.fluid
        p[~fluid] = 0.0
        if self.system.singular and fluid.any():
            p[fluid] -= p[fluid].mean()
        return p.reshape(self.system.shape)

    def residual_norm(self, p: np.ndarray, b: np.ndarray) -> float:
        """Relative 2-norm of ``b - A p`` (absolute if ``b`` is essentially zero)."""
        r = b - self.system.A @ p
        nb = float(np.linalg.norm(b))
        return float(np.linalg.norm(r)) / (nb if nb > _ZERO_RHS_NORM else 1.0)

    # -- interface ----------------------------------------------------- #
    def solve(self, rhs: np.ndarray, p0: np.ndarray | None = None) -> np.ndarray:
        """Solve ``lap(p) = rhs`` and return ``p`` with shape ``(nx, ny)``."""
        raise NotImplementedError


class DirectSolver(PoissonSolverBase):
    """Sparse LU factorisation computed once at construction and re-used.

    The task specification calls for ``spsolve`` with a pre-assembled matrix;
    ``spsolve`` re-factorises on every call, so the factorisation is hoisted out
    with :func:`scipy.sparse.linalg.splu` instead.  The answer is identical, the
    per-step cost is a pair of triangular solves.
    """

    name = "direct"

    def __init__(self, system, tol=DEFAULT_POISSON_TOL, maxiter=DEFAULT_POISSON_MAXITER):
        super().__init__(system, tol, maxiter)
        self._lu = spla.splu(self._pinned_matrix().tocsc())

    def _pinned_matrix(self) -> sp.csr_matrix:
        """Copy of ``A`` with the reference row replaced by the identity."""
        if not self.system.singular:
            return self.system.A
        A = self.system.A.tolil(copy=True)
        A.rows[self.system.pin] = [self.system.pin]
        A.data[self.system.pin] = [1.0]
        return A.tocsr()

    def solve(self, rhs, p0=None):
        """Solve by back-substitution through the stored LU factors."""
        b = self._prepare_rhs(rhs)
        b_pinned = b.copy()
        if self.system.singular:
            b_pinned[self.system.pin] = 0.0
        p = self._lu.solve(b_pinned)
        self.last_iterations = 1
        self.last_residual = self.residual_norm(p, b)
        return self._normalize(p)


class CGSolver(PoissonSolverBase):
    """Conjugate gradient on the symmetric (singular but consistent) operator.

    No pinning is needed: CG stays in the range space provided the right-hand
    side is compatible, which :meth:`_prepare_rhs` guarantees.
    """

    name = "cg"
    warm_startable = True

    def __init__(self, system, tol=DEFAULT_POISSON_TOL, maxiter=DEFAULT_POISSON_MAXITER):
        super().__init__(system, tol, maxiter)
        # Jacobi preconditioning: cheap and helpful for the 5-point Laplacian.
        d = self.system.A.diagonal()
        self._M = spla.LinearOperator(
            self.system.A.shape, matvec=lambda x: x / d, dtype=float
        )

    def solve(self, rhs, p0=None):
        """Solve with Jacobi-preconditioned conjugate gradient."""
        b = self._prepare_rhs(rhs)
        x0 = None if p0 is None else np.asarray(p0, dtype=float).ravel().copy()
        p, info = spla.cg(
            self.system.A, b, x0=x0, rtol=self.tol, atol=0.0,
            maxiter=self.maxiter, M=self._M,
        )
        self.last_residual = self.residual_norm(p, b)
        if info != 0 and self.last_residual > max(self.tol * 10, 1e-8):
            raise RuntimeError(
                f"pressure CG failed to converge (info={info}, "
                f"relative residual={self.last_residual:.3e} after {self.maxiter} iterations)"
            )
        return self._normalize(p)


#: Damping factor for Jacobi.  Undamped Jacobi does **not** converge on this
#: operator: for the Neumann Laplacian assembled by dropping boundary
#: coefficients, the chequerboard mode has eigenvalue exactly -1 (this holds at
#: interior, edge and corner cells alike, because the diagonal shrinks in step
#: with the number of neighbours).  Damping by ``omega`` maps that eigenvalue to
#: ``1 - 2*omega``; 2/3 is the classical optimal smoothing choice.
JACOBI_DAMPING = 2.0 / 3.0


class JacobiSolver(PoissonSolverBase):
    """Damped Jacobi iteration -- the transparent, fully explicit variant.

    Kept for teaching value: every sweep is ``p <- p + omega * D^-1 (b - A p)``
    with no hidden machinery.  Convergence is geometric but slow (the error
    contracts by roughly ``1 - O(omega * h^2)`` per sweep for the smoothest
    modes), so this is an educational option, not a production one.
    """

    name = "jacobi"
    warm_startable = True

    def __init__(self, system, tol=DEFAULT_POISSON_TOL, maxiter=DEFAULT_POISSON_MAXITER,
                 omega: float = JACOBI_DAMPING):
        super().__init__(system, tol, maxiter)
        self.omega = omega
        self._dinv = 1.0 / self.system.A.diagonal()
        #: Residual is only checked every few sweeps -- the matvec is the cost.
        self._check_every = 20

    def solve(self, rhs, p0=None):
        """Sweep damped Jacobi until the residual meets ``tol``."""
        b = self._prepare_rhs(rhs)
        A, fluid = self.system.A, self.system.fluid
        p = np.zeros_like(b) if p0 is None else np.asarray(p0, float).ravel().copy()
        p[~fluid] = 0.0

        res = self.residual_norm(p, b)
        for it in range(1, self.maxiter + 1):
            p += self.omega * self._dinv * (b - A @ p)
            if self.system.singular:
                p[fluid] -= p[fluid].mean()     # keep the iterate off the null space
            if it % self._check_every == 0 or it == self.maxiter:
                res = self.residual_norm(p, b)
                if res < self.tol:
                    break
        self.last_iterations = it
        self.last_residual = res
        if res > self.tol:
            raise RuntimeError(
                f"Jacobi pressure solve stalled: relative residual {res:.3e} > "
                f"tol {self.tol:.1e} after {it} sweeps. Use pressure_solver='direct' "
                "or raise poisson_maxiter."
            )
        return self._normalize(p)


class SORSolver(PoissonSolverBase):
    """Red/black successive over-relaxation.

    The 5-point stencil connects a cell only to cells of the opposite
    chequerboard colour, so a whole colour can be updated in one vectorised
    shot.  That makes this a genuine Gauss-Seidel/SOR sweep -- not a Jacobi
    approximation of one -- while remaining pure NumPy.
    """

    name = "sor"
    warm_startable = True

    def __init__(self, system, tol=DEFAULT_POISSON_TOL, maxiter=DEFAULT_POISSON_MAXITER,
                 omega: float = DEFAULT_SOR_OMEGA):
        super().__init__(system, tol, maxiter)
        self.omega = omega
        nx, ny = system.shape
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        colour = ((i + j) % 2).ravel()
        self._red = np.flatnonzero(colour == 0)
        self._black = np.flatnonzero(colour == 1)

        A = system.A.tocsr()
        self._A_rb = A[self._red][:, self._black]
        self._A_br = A[self._black][:, self._red]
        d = A.diagonal()
        self._d_red = d[self._red]
        self._d_black = d[self._black]

        # Validate the two-colouring: with an odd periodic extent the wrap-around
        # neighbour shares the colour of the cell and the sweep would be wrong.
        for label, block in (("red", A[self._red][:, self._red]),
                             ("black", A[self._black][:, self._black])):
            off = block - sp.diags(block.diagonal())
            if off.nnz and np.abs(off).max() > 0:
                raise ValueError(
                    f"red/black ordering is invalid for this grid ({label} cells are "
                    "coupled to each other). This happens with an odd number of cells "
                    "on a periodic axis; use an even nx/ny or pressure_solver='direct'."
                )
        self._check_every = 5

    def solve(self, rhs, p0=None):
        """Sweep red then black cells until the residual meets ``tol``."""
        b = self._prepare_rhs(rhs)
        fluid = self.system.fluid
        p = np.zeros_like(b) if p0 is None else np.asarray(p0, float).ravel().copy()
        p[~fluid] = 0.0
        red, black, w = self._red, self._black, self.omega
        b_red, b_black = b[red], b[black]

        res = self.residual_norm(p, b)
        for it in range(1, self.maxiter + 1):
            p[red] = (1 - w) * p[red] + w * (b_red - self._A_rb @ p[black]) / self._d_red
            p[black] = (1 - w) * p[black] + w * (b_black - self._A_br @ p[red]) / self._d_black
            if self.system.singular:
                p[fluid] -= p[fluid].mean()
            if it % self._check_every == 0 or it == self.maxiter:
                res = self.residual_norm(p, b)
                if res < self.tol:
                    break
        self.last_iterations = it
        self.last_residual = res
        if res > self.tol:
            raise RuntimeError(
                f"SOR pressure solve stalled: relative residual {res:.3e} > "
                f"tol {self.tol:.1e} after {it} sweeps. Use pressure_solver='direct' "
                "or raise poisson_maxiter."
            )
        return self._normalize(p)


class MultigridSolverBase(PoissonSolverBase):
    """Shared setup for the two solvers built on a V-cycle hierarchy."""

    def __init__(self, system, tol=DEFAULT_POISSON_TOL,
                 maxiter=DEFAULT_POISSON_MAXITER,
                 smooth_sweeps: int = DEFAULT_MG_SWEEPS) -> None:
        super().__init__(system, tol, maxiter)
        from .multigrid import MultigridHierarchy

        self.hierarchy = MultigridHierarchy(
            system.A, system.shape, system.fluid, system.singular,
            presmooth=smooth_sweeps, postsmooth=smooth_sweeps,
        )
        log.debug("%s: %r", self.name, self.hierarchy)


class MultigridSolver(MultigridSolverBase):
    """Standalone V-cycles, iterated until the residual meets ``tol``.

    Convergence is geometric with a factor that does not depend on the grid --
    measured at 0.42 per cycle from 32x32 up to 512x512, and 0.12 to 0.20 on a
    stretched mesh, where semi-coarsening leaves the coarse problem nearly
    one-dimensional.  Useful on its own, and the clearest way to see that the
    hierarchy is doing its job, but :class:`MultigridCGSolver` reaches the same
    tolerance in about half the work and is the one to reach for.
    """

    name = "multigrid"
    warm_startable = True

    def solve(self, rhs, p0=None):
        """V-cycle from ``p0`` (or zero) until the relative residual meets ``tol``."""
        b = self._prepare_rhs(rhs)
        x0 = None if p0 is None else np.asarray(p0, dtype=float).ravel()
        p, cycles, res = self.hierarchy.cycle_until(
            b, x0, self.tol, self.maxiter, self.residual_norm,
        )
        self.last_iterations = cycles
        self.last_residual = res
        if res > self.tol:
            raise RuntimeError(
                f"multigrid pressure solve stalled: relative residual {res:.3e} > "
                f"tol {self.tol:.1e} after {cycles} V-cycles. Use "
                "pressure_solver='direct' or raise poisson_maxiter."
            )
        return self._normalize(p)


class MultigridCGSolver(MultigridSolverBase):
    """Conjugate gradient preconditioned by one multigrid V-cycle.

    This is the solver the multigrid work exists to provide.  Jacobi-CG needs an
    iteration count that grows with the grid, because the preconditioner does
    nothing about the smooth part of the error; a V-cycle attacks exactly that
    part, and the count stops growing:

    ========  ============  =========
    grid      Jacobi-CG      MG-CG
    ========  ============  =========
    64x64        283 it       13 it
    128x128      552 it       13 it
    256x256     1128 it       13 it
    512x512     2219 it       13 it
    ========  ============  =========

    Measured, to a relative residual of 1e-10.  Jacobi-CG doubles with every
    doubling of the grid, which is the square root of the condition number
    doing exactly what theory says it will; the multigrid column does not move.

    Against the *direct* solver the trade is different and worth being plain
    about: sparse LU wins on time at small and moderate sizes and loses on
    memory as the grid grows, because its factors fill in while a hierarchy does
    not.  ``direct`` therefore stays the package default; this is what to use
    when the grid is large enough that the factorisation becomes the problem.
    """

    name = "mgcg"
    warm_startable = True

    def __init__(self, system, tol=DEFAULT_POISSON_TOL,
                 maxiter=DEFAULT_POISSON_MAXITER,
                 smooth_sweeps: int = DEFAULT_MG_SWEEPS) -> None:
        super().__init__(system, tol, maxiter, smooth_sweeps)
        self._M = self.hierarchy.as_linear_operator()

    def solve(self, rhs, p0=None):
        """Solve with multigrid-preconditioned conjugate gradient."""
        b = self._prepare_rhs(rhs)
        x0 = None if p0 is None else np.asarray(p0, dtype=float).ravel().copy()
        iterations = [0]
        p, info = spla.cg(
            self.system.A, b, x0=x0, rtol=self.tol, atol=0.0,
            maxiter=self.maxiter, M=self._M,
            callback=lambda _xk: iterations.__setitem__(0, iterations[0] + 1),
        )
        self.last_iterations = iterations[0]
        self.last_residual = self.residual_norm(p, b)
        if info != 0 and self.last_residual > max(self.tol * 10, 1e-8):
            raise RuntimeError(
                f"multigrid-preconditioned CG failed to converge (info={info}, "
                f"relative residual={self.last_residual:.3e} after "
                f"{self.last_iterations} iterations)"
            )
        return self._normalize(p)


_SOLVERS: dict[PressureSolver, type[PoissonSolverBase]] = {
    PressureSolver.DIRECT: DirectSolver,
    PressureSolver.CG: CGSolver,
    PressureSolver.JACOBI: JacobiSolver,
    PressureSolver.SOR: SORSolver,
    PressureSolver.MULTIGRID: MultigridSolver,
    PressureSolver.MGCG: MultigridCGSolver,
}


def make_pressure_solver(
    kind: PressureSolver,
    system: PoissonSystem,
    tol: float = DEFAULT_POISSON_TOL,
    maxiter: int = DEFAULT_POISSON_MAXITER,
    sor_omega: float = DEFAULT_SOR_OMEGA,
    mg_sweeps: int = DEFAULT_MG_SWEEPS,
) -> PoissonSolverBase:
    """Instantiate the requested pressure solver for ``system``."""
    kind = PressureSolver(kind)
    cls = _SOLVERS[kind]
    if cls is SORSolver:
        return SORSolver(system, tol, maxiter, omega=sor_omega)
    if issubclass(cls, MultigridSolverBase):
        return cls(system, tol, maxiter, smooth_sweeps=mg_sweeps)
    return cls(system, tol, maxiter)
