"""Aggregation multigrid for the pressure Poisson operator.

Why this exists
---------------
The projection step solves ``lap(p) = div(u*)/dt`` once per Runge--Kutta
substep, three times per step under the default RK3.  Measured on the cylinder
case that is 55% of the wall time at 128x64 and 82% at 512x256 -- the pressure
solve, not the momentum stencils, is what a large run waits for.

The default :class:`~pycfd.core.pressure.DirectSolver` is hard to beat on time
alone in 2D: the LU factors are computed once and every later solve is a pair of
triangular sweeps.  What it cannot do is *scale*, because the factors fill in::

    grid         unknowns   factorise   per solve   LU fill
    512 x 256     131_072      0.55 s      11 ms     166 MiB
    1024 x 512    524_288      4.07 s      59 ms     850 MiB
    1536 x 768  1_179_648     13.16 s     339 ms    2144 MiB

The fill grows faster than the problem and the triangular solves with it, so the
direct path runs into memory well before it runs into patience.  A V-cycle costs
``O(N)`` work and ``O(N)`` storage -- the whole hierarchy is under twice the fine
matrix -- and its convergence factor does not depend on the grid.  That is the
trade this module offers.  It is *not* a blanket speed-up: on small grids the
direct solver still wins comfortably, which is why
:mod:`pycfd.core.pressure` keeps ``direct`` as the default.

Why aggregation, and why Galerkin
---------------------------------
Textbook geometric multigrid re-discretises the Laplacian on each coarse grid.
That is fine on an empty box and wrong here: an immersed body is a *mask*, and
coarsening a mask loses thin features -- a body one or two cells thick simply
disappears a couple of levels down, leaving a coarse operator that no longer
describes the problem it is supposed to be correcting.

So the coarse operators are formed variationally instead, ``A_c = P^T A P``.
Whatever the fine operator says about the geometry -- dropped obstacle faces,
eliminated Dirichlet ghosts, periodic wrap-around -- is inherited exactly, with
no second opinion about where the body is.  :mod:`pycfd.core.pressure` builds
the fine operator so that its stencil matches the projection's face set exactly,
and Galerkin coarsening is what carries that guarantee down the hierarchy.

The aggregates themselves are geometric: blocks of neighbouring cells, normally
2x2.  That is what keeps setup cheap -- no strength-of-connection pass, no graph
matching -- and it suits an operator that really is a structured-grid Laplacian.
Cells inside a body join no aggregate, so a body punches a hole through every
level rather than being averaged away.  On a stretched mesh the block becomes
2x1 or 1x2 instead; see :func:`coarsening_factors` for why that is not optional.

The null space
--------------
With Neumann conditions everywhere the operator is singular and the constant is
its null vector.  Two properties keep that from causing trouble:

* The tentative prolongator has entries of exactly 0 or 1, which is the *only*
  weighting for which ``P 1_coarse = 1_fine``.  So the constant is represented
  exactly at every level, the coarse operators inherit vanishing row sums, and a
  coarse right-hand side ``P^T r`` is compatible whenever ``r`` is.  Scaling the
  columns -- the usual smoothed-aggregation normalisation -- would keep the
  constant in the *range* of ``P`` but stop it being the coarse null *vector*,
  and then every level would have to carry its own null vector around.  The
  price of not scaling is that coarse matrix entries grow by a fixed factor per
  level; every use of them divides by the diagonal, so nothing notices.
* The coarsest solve pins one unknown and then removes the mean.  Together those
  are exactly the pseudo-inverse, which matters for the reason below.

Symmetry
--------
Used as a preconditioner inside conjugate gradient, the V-cycle has to be a
*symmetric* operator, or CG loses the guarantee it is built on and stalls
unpredictably.  Every piece is picked to keep it so: damped Jacobi is a
symmetric smoother, the same number of sweeps runs before and after the coarse
correction, restriction is exactly ``P^T``, the coarsest solve is the
pseudo-inverse rather than a pinned solve (pinning is not symmetric), and the
projection onto the active subspace and off the null space is applied on the way
in and again on the way out.  ``test_multigrid.py`` builds the preconditioner's
matrix column by column and checks this, rather than trusting the paragraph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

log = logging.getLogger(__name__)

#: Stop coarsening once a level has at most this many unknowns; the coarsest
#: operator is factorised directly, so it wants to be small -- but one extra
#: level costs a setup pass and buys nothing below a few dozen unknowns.
COARSEST_UNKNOWNS = 96

#: Never build more levels than this, whatever the grid.  A safety rail: 2x2
#: aggregation quarters the count each time, so twenty levels covers grids far
#: beyond anything that fits in memory.
MAX_LEVELS = 20

#: Give up coarsening if a level fails to shrink the problem by this factor.
#: Only reachable on a degenerate domain -- a single row or column of cells,
#: where 2x2 blocks can only halve -- but there it stops an infinite descent.
MIN_COARSENING_RATIO = 1.3

#: Damped-Jacobi weight as a multiple of ``1 / rho(D^-1 A)``.  The classical
#: choice: it maps the upper half of the spectrum -- exactly the part no coarse
#: grid can represent -- onto factors of magnitude at most 1/3.
JACOBI_WEIGHT_FACTOR = 4.0 / 3.0

#: Power iterations used to estimate ``rho(D^-1 A)`` on each level.  The weight
#: is insensitive to the estimate -- being 10% short moves the smoothing factor
#: in the third decimal -- so a loose one is plenty.
POWER_ITERATIONS = 12

#: A cycle that leaves the residual above this multiple of its previous value
#: has made no useful progress.  Not 1.0: a converged iterate wanders by
#: round-off, and the last cycle before the tolerance is met often gains only a
#: few per cent.
STALL_FACTOR = 0.995

#: Consecutive unproductive cycles tolerated before giving up.  Three, because a
#: single flat cycle can be a plateau the next one breaks through.
STALL_PATIENCE = 3


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def spectral_radius(A: sp.csr_matrix, dinv: np.ndarray, seed: int = 0) -> float:
    """Estimate ``rho(D^-1 A)`` by power iteration.

    A Gershgorin bound would give 2 for free on the fine operator, whose row
    sums vanish.  It stops being free further down: smoothing the prolongator
    can put off-diagonals of the diagonal's own sign into a coarse operator, and
    then the bound no longer holds.  A dozen matrix-vector products at setup is
    cheaper than being wrong about it.
    """
    n = A.shape[0]
    if n == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    norm = float(np.linalg.norm(x))
    if norm == 0.0:
        return 1.0
    x /= norm
    rho = 1.0
    for _ in range(POWER_ITERATIONS):
        y = dinv * (A @ x)
        norm = float(np.linalg.norm(y))
        if norm == 0.0:
            return rho
        rho = norm
        x = y / norm
    # Power iteration approaches the radius from below.  Nudge up, so a short
    # estimate cannot push the damped smoother past its stable range.
    return rho * 1.05


#: Strength ratio between the two axes beyond which a level coarsens one axis
#: only.  Two is not a taste: coarsening an axis divides the ratio by four, so
#: semi-coarsening lands closer to isotropy than full coarsening exactly when
#: the ratio exceeds two, and the rule is simply "do whichever helps".
ANISOTROPY_THRESHOLD = 2.0


def coarsening_factors(A: sp.csr_matrix, shape: tuple[int, int]) -> tuple[int, int]:
    """Choose how much to coarsen each axis: ``(2, 2)``, ``(2, 1)`` or ``(1, 2)``.

    Damped Jacobi smooths error across *strong* connections and leaves it rough
    across weak ones.  On a stretched cell the two axes are not equally strong
    -- the Laplacian coefficient is ``1/h**2``, so halving one spacing
    quadruples that axis's coupling -- and a coarse grid that assumes the error
    is smooth in both directions then corrects something the smoother never
    made smooth.  The result degrades fast: measured on 64x64, a cell aspect
    ratio of 1 needs 13 preconditioned iterations, 4 needs 42, and 16 needs 165.

    Coarsening only along the strong axis is the standard answer, and it is
    self-correcting -- each pass moves the ratio back towards one, so a 16:1
    mesh semi-coarsens twice and then proceeds normally.

    The ratio is read off the operator rather than the mesh: on a grid of shape
    ``(nx, ny)`` laid out row-major, x-neighbours sit ``ny`` apart in the flat
    index and y-neighbours sit ``1`` apart, so their mean magnitudes are two
    diagonals of the matrix.  Taking it from the operator means the Galerkin
    coarse levels are judged on what they actually became, not on what the mesh
    spacing was several levels ago, and it costs no extra bookkeeping.
    """
    nx, ny = shape
    if nx < 2 and ny < 2:
        return 1, 1

    def strength(offset: int) -> float:
        if offset >= A.shape[0]:
            return 0.0
        band = np.abs(A.diagonal(offset))
        return float(band.mean()) if band.size else 0.0

    sx, sy = strength(ny), strength(1)
    if nx >= 2 and sx > ANISOTROPY_THRESHOLD * sy:
        return 2, 1
    if ny >= 2 and sy > ANISOTROPY_THRESHOLD * sx:
        return 1, 2
    return (2 if nx >= 2 else 1), (2 if ny >= 2 else 1)


def coarse_shape(shape: tuple[int, int], factors: tuple[int, int]) -> tuple[int, int]:
    """Grid shape after coarsening ``shape`` by ``factors``, rounding up."""
    return tuple(max(-(-n // f), 1) for n, f in zip(shape, factors))  # type: ignore[return-value]


def aggregate_blocks(shape: tuple[int, int], active: np.ndarray,
                     factors: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Group active cells into blocks; return block labels and the block mask.

    ``labels`` is ``-1`` on inactive cells and otherwise the flat index of the
    cell's block in the coarse grid of shape :func:`coarse_shape`.

    An extent that does not divide evenly leaves a cell over, and it becomes a
    block of its own rather than joining the last full one.  Both are legal
    aggregations, and the intuition that a lone cell must interpolate badly is
    simply wrong here -- measured on 65x65, 129x129 and 65x33, singleton blocks
    give a convergence factor of 0.42 against 0.59 for three-wide ones, and 13
    preconditioned iterations against 16.  Widening the last block coarsens
    three cells into one where the grid is least able to afford it; a singleton
    is reproduced exactly instead.  Odd extents compound -- 65 coarsens to 33 to
    17 to 9 -- so the choice is made on every level of such a grid, and with
    singletons an odd grid costs nothing at all against an even one.

    The coarse grid keeps its full rectangular shape, with blocks holding no
    active cell marked inactive rather than renumbered away.  Renumbering would
    be tidier by one array but would destroy the thing the *next* level needs: a
    coarse unknown's position on a grid, so that its own blocks pair up
    neighbours.  A body therefore stays a hole in the same place on every level.
    """
    nx, ny = shape
    fx, fy = factors
    ncx, ncy = coarse_shape(shape, factors)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    block = ((i // fx) * ncy + (j // fy)).ravel()
    labels = np.where(np.asarray(active).ravel(), block, -1)

    coarse_active = np.zeros(ncx * ncy, dtype=bool)
    coarse_active[labels[labels >= 0]] = True
    return labels, coarse_active


def tentative_prolongator(labels: np.ndarray, n_coarse: int) -> sp.csr_matrix:
    """The 0/1 aggregation matrix: one entry per active row, none per inactive one."""
    rows = np.flatnonzero(labels >= 0)
    cols = labels[rows]
    return sp.coo_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(labels.size, n_coarse)
    ).tocsr()


def smooth_prolongator(A: sp.csr_matrix, dinv: np.ndarray, weight: float,
                       P: sp.csr_matrix) -> sp.csr_matrix:
    """One damped-Jacobi pass over the tentative prolongator, ``P <- (I - w D^-1 A) P``.

    This is what separates usable multigrid from a two-level method that
    degrades as the grid refines.  Piecewise-constant interpolation delivers a
    coarse correction as a staircase, and the smoother then spends its sweeps
    removing the steps it was just handed.

    The null space survives the pass.  Where the operator is singular its row
    sums vanish, so ``A 1 = 0`` on the active set and ``P 1_coarse = 1_fine``
    still holds afterwards; where a Dirichlet wall makes the operator
    non-singular there is no null space to preserve.  Inactive rows stay empty
    either way, because the operator gives them the identity and ``P`` gives
    them nothing.
    """
    return (P - sp.diags(weight * dinv) @ (A @ P)).tocsr()


def _identity_on_inactive(A: sp.csr_matrix, active: np.ndarray) -> sp.csr_matrix:
    """Give every inactive row the trivial equation ``x = 0``.

    Galerkin coarsening leaves a dropped block as an all-zero row and column.
    Zero rows have no diagonal to smooth with and make the coarsest
    factorisation singular for a reason that has nothing to do with the physical
    null space, so they get the same treatment the fine assembly gives an
    obstacle cell.
    """
    if active.all():
        return A
    return (A + sp.diags((~active).astype(float))).tocsr()


# --------------------------------------------------------------------------- #
# Hierarchy
# --------------------------------------------------------------------------- #
@dataclass
class Level:
    """One rung: its operator, its smoother, and the way up from the next one."""

    A: sp.csr_matrix
    #: Reciprocal of the diagonal.
    dinv: np.ndarray
    #: The same times :attr:`weight`, stored pre-multiplied so that a smoothing
    #: sweep is one multiply and one add on top of the matrix-vector product.
    weighted_dinv: np.ndarray
    #: The damping weight on its own; prolongator smoothing needs it separately.
    weight: float
    #: Rows this level owns.  Inactive rows are obstacle cells at level 0 and
    #: dropped aggregates below it.
    active: np.ndarray
    #: Grid shape this level's unknowns are laid out on, for the next aggregation.
    shape: tuple[int, int]
    #: Prolongation *from* this level *to* the one above; ``None`` on the finest.
    P: sp.csr_matrix | None = None
    #: ``True`` when the constant (on the active set) is in the null space.
    singular: bool = False
    #: Factorisation of the pinned operator; only the coarsest level has one.
    lu: object = field(default=None, repr=False)
    #: Index of the unknown pinned by that factorisation.
    pin: int = 0

    @property
    def n(self) -> int:
        return self.A.shape[0]


class MultigridHierarchy:
    """A V-cycle for one assembled Poisson operator.

    Built once when the solver is constructed and applied every substep.  The
    hierarchy depends only on the operator, so geometry that moved would need it
    rebuilt; the solver is static-geometry today and assembles the operator
    once, so it does not.
    """

    def __init__(self, A: sp.csr_matrix, shape: tuple[int, int],
                 active: np.ndarray, singular: bool,
                 presmooth: int = 1, postsmooth: int = 1) -> None:
        if presmooth < 1 or postsmooth < 1:
            raise ValueError(
                "a V-cycle needs at least one smoothing sweep on each side, got "
                f"pre={presmooth}, post={postsmooth}"
            )
        if presmooth != postsmooth:
            raise ValueError(
                f"a V-cycle with pre={presmooth} and post={postsmooth} sweeps is "
                "not a symmetric operator, and conjugate gradient needs one; use "
                "an equal number on each side"
            )
        self.presmooth = presmooth
        self.postsmooth = postsmooth
        self.levels: list[Level] = []
        self._build(sp.csr_matrix(A), tuple(shape),
                    np.asarray(active, dtype=bool).ravel(), bool(singular))

    # -- construction ---------------------------------------------------- #
    def _make_level(self, A: sp.csr_matrix, shape: tuple[int, int],
                    active: np.ndarray, singular: bool,
                    P: sp.csr_matrix | None) -> Level:
        diag = A.diagonal()
        dinv = 1.0 / np.where(diag == 0.0, 1.0, diag)
        rho = spectral_radius(A, dinv)
        weight = JACOBI_WEIGHT_FACTOR / rho if rho > 0.0 else 1.0
        return Level(A=A, dinv=dinv, weighted_dinv=weight * dinv, weight=weight,
                     active=active, shape=shape, P=P, singular=singular)

    def _build(self, A: sp.csr_matrix, shape: tuple[int, int],
               active: np.ndarray, singular: bool) -> None:
        level = self._make_level(A, shape, active, singular, None)
        self.levels.append(level)

        while len(self.levels) < MAX_LEVELS and level.n > COARSEST_UNKNOWNS:
            fx, fy = coarsening_factors(level.A, level.shape)
            labels, coarse_active = aggregate_blocks(level.shape, level.active,
                                                     (fx, fy))
            next_shape = coarse_shape(level.shape, (fx, fy))
            n_coarse = coarse_active.size
            if n_coarse == 0 or level.n < MIN_COARSENING_RATIO * n_coarse:
                break

            P = tentative_prolongator(labels, n_coarse)
            P = smooth_prolongator(level.A, level.dinv, level.weight, P)
            coarse_A = _identity_on_inactive(
                sp.csr_matrix(P.T @ level.A @ P), coarse_active,
            )
            coarse_A.sum_duplicates()
            level = self._make_level(coarse_A, next_shape, coarse_active,
                                     singular, P)
            self.levels.append(level)

        coarsest = self.levels[-1]
        coarsest.pin = int(np.flatnonzero(coarsest.active)[0]) \
            if coarsest.active.any() else 0
        coarsest.lu = _factorise_coarsest(coarsest.A, coarsest.singular,
                                          coarsest.pin)
        log.debug("multigrid hierarchy: unknowns %s, non-zeros %s, complexity %.2f",
                  self.unknowns, [lv.A.nnz for lv in self.levels],
                  self.operator_complexity)

    # -- reporting ------------------------------------------------------- #
    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def unknowns(self) -> list[int]:
        """Unknown count per level, finest first."""
        return [lv.n for lv in self.levels]

    @property
    def operator_complexity(self) -> float:
        """Non-zeros across the whole hierarchy, over the fine operator's.

        The standard multigrid health number.  It is both the memory the
        hierarchy costs relative to the matrix it preconditions and, near
        enough, the cost of one V-cycle in fine matrix-vector products.
        """
        fine = self.levels[0].A.nnz
        return sum(lv.A.nnz for lv in self.levels) / fine if fine else 1.0

    def __repr__(self) -> str:
        return (f"MultigridHierarchy({self.n_levels} levels, "
                f"{self.unknowns[0]} -> {self.unknowns[-1]} unknowns, "
                f"complexity {self.operator_complexity:.2f})")

    # -- application ----------------------------------------------------- #
    def _smooth(self, level: Level, b: np.ndarray, x: np.ndarray,
                sweeps: int) -> None:
        """``sweeps`` damped-Jacobi passes over ``x``, in place."""
        for _ in range(sweeps):
            x += level.weighted_dinv * (b - level.A @ x)

    def _cycle(self, depth: int, b: np.ndarray, x: np.ndarray) -> None:
        """One V-cycle at ``depth``, refining ``x`` in place."""
        level = self.levels[depth]
        if depth == len(self.levels) - 1:
            x[:] = _solve_coarsest(level, b)
            return

        self._smooth(level, b, x, self.presmooth)
        residual = b - level.A @ x

        # ``P.T`` is a CSC view, and CSC matrix-vector products are about 2.4x
        # slower than CSR ones here.  Caching ``P.T.tocsr()`` per level fixes
        # that and is tempting; measured at 512x256 it is worth 1.07x on the
        # solve for 1.32x the hierarchy's memory.  Restriction is simply not
        # where the time goes -- the operator matrix-vector products are -- and
        # memory is the whole reason to prefer this solver, so it stays a view.
        P = self.levels[depth + 1].P
        coarse_x = np.zeros(P.shape[1])
        self._cycle(depth + 1, P.T @ residual, coarse_x)
        x += P @ coarse_x

        self._smooth(level, b, x, self.postsmooth)

    @staticmethod
    def _project(level: Level, v: np.ndarray) -> None:
        """Zero the inactive rows and, when singular, remove the constant."""
        v[~level.active] = 0.0
        if level.singular and level.active.any():
            v[level.active] -= v[level.active].mean()

    def apply(self, r: np.ndarray) -> np.ndarray:
        """One V-cycle applied to ``r`` from a zero start: the preconditioner.

        The projections bracket the cycle so that what this returns is a
        symmetric operator mapping the range of ``A`` into itself -- which is
        exactly what conjugate gradient needs of a preconditioner on a singular
        but consistent system.
        """
        top = self.levels[0]
        b = np.array(r, dtype=float).ravel()
        self._project(top, b)
        x = np.zeros_like(b)
        self._cycle(0, b, x)
        self._project(top, x)
        return x

    def as_linear_operator(self) -> spla.LinearOperator:
        """The preconditioner in the form :func:`scipy.sparse.linalg.cg` expects."""
        n = self.levels[0].n
        return spla.LinearOperator((n, n), matvec=self.apply, dtype=float)

    def cycle_until(self, b: np.ndarray, x0: np.ndarray | None, tol: float,
                    maxiter: int, residual_norm) -> tuple[np.ndarray, int, float]:
        """V-cycle from ``x0`` until ``residual_norm`` falls below ``tol``.

        Returns the iterate, the number of cycles taken, and the residual
        reached.  This is the standalone solver's loop; the preconditioned one
        goes through :meth:`apply` inside CG instead.

        Gives up early if the residual stops falling.  ``poisson_maxiter``
        defaults to twenty thousand, which is a reasonable bound on *Jacobi*
        sweeps and an absurd one on V-cycles -- a converging hierarchy needs
        about thirty.  Asking for a tolerance the arithmetic cannot reach would
        otherwise spend hours discovering it.
        """
        top = self.levels[0]
        b = np.array(b, dtype=float).ravel()
        self._project(top, b)
        x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float).ravel()
        self._project(top, x)

        res = residual_norm(x, b)
        cycles = 0
        stalled = 0
        while cycles < maxiter and res >= tol and stalled < STALL_PATIENCE:
            previous = res
            self._cycle(0, b, x)
            self._project(top, x)
            res = residual_norm(x, b)
            cycles += 1
            stalled = stalled + 1 if res > STALL_FACTOR * previous else 0
        return x, cycles, res


def _factorise_coarsest(A: sp.csr_matrix, singular: bool, pin: int):
    """LU of the coarsest operator, with one unknown pinned when it is singular."""
    if singular and A.shape[0]:
        M = A.tolil(copy=True)
        M.rows[pin] = [pin]
        M.data[pin] = [1.0]
        A = M.tocsr()
    return spla.splu(sp.csc_matrix(A))


def _solve_coarsest(level: Level, b: np.ndarray) -> np.ndarray:
    """Exact solve on the coarsest level -- the pseudo-inverse when singular.

    Pinning on its own is an asymmetric operator, and dropping an asymmetric
    step into the middle of a V-cycle costs conjugate gradient the guarantee it
    runs on.  Pinning followed by removing the mean is not asymmetric: for a
    compatible right-hand side the pair is exactly ``A^+``, and the
    pseudo-inverse of a symmetric matrix is symmetric.
    """
    rhs = np.array(b, dtype=float)
    active = level.active
    if level.singular:
        rhs[~active] = 0.0
        rhs[active] -= rhs[active].mean()
        rhs[level.pin] = 0.0
    x = level.lu.solve(rhs)
    x[~active] = 0.0
    if level.singular and active.any():
        x[active] -= x[active].mean()
    return x
