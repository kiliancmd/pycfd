"""Optional Numba kernels for the momentum right-hand side.

The NumPy implementation in :mod:`pycfd.core.solver` is already fully
vectorised, so this module is not about replacing Python loops -- it is about
*fusion*.  Evaluating the convective and viscous terms with array expressions
allocates roughly twenty temporaries the size of the grid and walks memory once
per temporary.  The kernel below computes the same quantity in a single pass
with no intermediate storage, which is where the speed-up comes from.

The kernels are numerically identical to the array version (same operations, same
order), not an approximation of it; :mod:`pycfd.tests.test_solver` pins them
against each other.  If Numba is unavailable the solver silently keeps using the
NumPy path.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised implicitly by whichever path is available
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        """No-op stand-in so the module imports without Numba installed."""
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap


def _momentum_rhs_body(u, v, dx, dy, nu, blend, fx, fy, rhs_u, rhs_v):
    """Fused convective + viscous + body-force right-hand side.

    Mirrors :meth:`pycfd.core.solver.ProjectionSolver.momentum_rhs` exactly:
    conservative advection with the shared corner flux, the five-point viscous
    Laplacian, and the optional donor-cell blend.  Results are written into
    ``rhs_u`` with shape ``(nx+1, ny)`` and ``rhs_v`` with shape ``(nx, ny+1)``.
    """
    nx = u.shape[0] - 3
    ny = u.shape[1] - 2
    dx2 = dx * dx
    dy2 = dy * dy

    # -- u momentum: x-faces m = 1..nx+1, cell rows j = 1..ny ---------------- #
    for m in prange(1, nx + 2):
        for j in range(1, ny + 1):
            u0 = u[m, j]

            # d(uu)/dx from the cell-centred interpolants either side of the face.
            u_right = 0.5 * (u0 + u[m + 1, j])
            u_left = 0.5 * (u[m - 1, j] + u0)
            duudx = (u_right * u_right - u_left * u_left) / dx

            # d(uv)/dy from the corner fluxes above and below the face.
            u_top = 0.5 * (u0 + u[m, j + 1])
            v_top = 0.5 * (v[m - 1, j + 1] + v[m, j + 1])
            u_bot = 0.5 * (u[m, j - 1] + u0)
            v_bot = 0.5 * (v[m - 1, j] + v[m, j])
            duvdy = (u_top * v_top - u_bot * v_bot) / dy

            if blend > 0.0:
                duudx += blend / dx * (
                    abs(u_right) * (u0 - u[m + 1, j]) * 0.5
                    - abs(u_left) * (u[m - 1, j] - u0) * 0.5
                )
                duvdy += blend / dy * (
                    abs(v_top) * (u0 - u[m, j + 1]) * 0.5
                    - abs(v_bot) * (u[m, j - 1] - u0) * 0.5
                )

            lap = ((u[m - 1, j] - 2.0 * u0 + u[m + 1, j]) / dx2
                   + (u[m, j - 1] - 2.0 * u0 + u[m, j + 1]) / dy2)
            rhs_u[m - 1, j - 1] = -(duudx + duvdy) + nu * lap + fx

    # -- v momentum: y-faces n = 1..ny+1, cell columns i = 1..nx ------------- #
    for i in prange(1, nx + 1):
        for n in range(1, ny + 2):
            v0 = v[i, n]

            v_top = 0.5 * (v0 + v[i, n + 1])
            v_bot = 0.5 * (v[i, n - 1] + v0)
            dvvdy = (v_top * v_top - v_bot * v_bot) / dy

            u_right = 0.5 * (u[i + 1, n - 1] + u[i + 1, n])
            v_right = 0.5 * (v0 + v[i + 1, n])
            u_left = 0.5 * (u[i, n - 1] + u[i, n])
            v_left = 0.5 * (v[i - 1, n] + v0)
            duvdx = (u_right * v_right - u_left * v_left) / dx

            if blend > 0.0:
                dvvdy += blend / dy * (
                    abs(v_top) * (v0 - v[i, n + 1]) * 0.5
                    - abs(v_bot) * (v[i, n - 1] - v0) * 0.5
                )
                duvdx += blend / dx * (
                    abs(u_right) * (v0 - v[i + 1, n]) * 0.5
                    - abs(u_left) * (v[i - 1, n] - v0) * 0.5
                )

            lap = ((v[i - 1, n] - 2.0 * v0 + v[i + 1, n]) / dx2
                   + (v[i, n - 1] - 2.0 * v0 + v[i, n + 1]) / dy2)
            rhs_v[i - 1, n - 1] = -(duvdx + dvvdy) + nu * lap + fy


# Two compilations of the same source.  Threading pays for itself only once the
# grid is large enough to amortise the fork/join cost -- below the threshold the
# serial build is measurably faster, above it the parallel build wins by a wide
# margin.  ``prange`` degrades to ``range`` when ``parallel=False``, so the two
# builds are numerically identical (and bitwise reproducible: every iteration
# writes its own output cell, with no reduction and no ordering dependence).
# NOTE: the parallel build must not set ``cache=True``.  Numba keys its on-disk
# cache on the source function, so two builds of the same function collide and
# the second silently loads the first one's artifact -- yielding a "parallel"
# kernel that runs serially.  Recompiling it each session costs about a second,
# which ``warmup`` absorbs.
_kernel_serial = njit(cache=True, fastmath=False, parallel=False)(_momentum_rhs_body)
_kernel_parallel = njit(cache=False, fastmath=False, parallel=True)(_momentum_rhs_body)

#: Cell count above which the threaded kernel is selected.  Measured on an
#: 8-thread machine, the two builds cross between 96x96 (~9k cells, serial
#: faster) and 128x128 (~16k cells, threaded faster); see README.md.
PARALLEL_CELL_THRESHOLD = 12_000


def select_kernel(n_cells: int):
    """Return the kernel build best suited to a grid of ``n_cells`` cells."""
    return _kernel_parallel if n_cells >= PARALLEL_CELL_THRESHOLD else _kernel_serial


def momentum_rhs_kernel(u, v, dx, dy, nu, blend, fx, fy, rhs_u, rhs_v):
    """Convenience wrapper that picks the kernel build automatically."""
    kernel = select_kernel(rhs_u.shape[0] * rhs_u.shape[1])
    kernel(u, v, dx, dy, nu, blend, fx, fy, rhs_u, rhs_v)


def warmup(n_cells: int = 0) -> None:
    """Trigger JIT compilation once so the first timed step is not the slowest."""
    if not NUMBA_AVAILABLE:
        return
    nx = ny = 8
    u = np.zeros((nx + 3, ny + 2))
    v = np.zeros((nx + 2, ny + 3))
    select_kernel(n_cells)(u, v, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                           np.zeros((nx + 1, ny)), np.zeros((nx, ny + 1)))
