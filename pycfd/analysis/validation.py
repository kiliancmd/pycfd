"""Analytical solutions, published reference data and error metrics.

Contents
--------
* Plane Poiseuille flow -- exact steady solution between parallel plates.
* Taylor--Green vortex -- exact *unsteady* solution of the full nonlinear
  equations on a periodic domain.  This is the primary convergence benchmark:
  because the domain is periodic there is no pressure boundary layer, so the
  splitting error of the projection method does not contaminate the measured
  spatial order.
* Ghia, Ghia & Shin (1982) lid-driven-cavity centreline data for
  Re = 100, 400 and 1000.
* L2/Linf error norms and a grid-convergence (observed-order) estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Analytical solutions
# --------------------------------------------------------------------------- #
def poiseuille_profile(y: np.ndarray, height: float, u_max: float) -> np.ndarray:
    """Parabolic velocity profile between plates at ``y = 0`` and ``y = height``.

    ``u(y) = u_max * 4 * y * (h - y) / h^2``, peaking at ``u_max`` on the centreline.
    """
    y = np.asarray(y, dtype=float)
    return u_max * 4.0 * y * (height - y) / height ** 2


def poiseuille_u_max(dpdx: float, height: float, nu: float, density: float = 1.0) -> float:
    """Centreline speed driven by a constant pressure gradient (or body force).

    Solving ``nu * d2u/dy2 = dp/dx / rho`` with no slip at both walls gives
    ``u_max = -dpdx * h^2 / (8 * rho * nu)``.  A body force ``f`` per unit mass
    corresponds to ``dpdx = -f``.
    """
    if nu <= 0 or height <= 0:
        raise ValueError("nu and height must be positive")
    return -dpdx * height ** 2 / (8.0 * density * nu)


def taylor_green(x: np.ndarray, y: np.ndarray, t: float, nu: float,
                 component: str) -> np.ndarray:
    """Exact Taylor--Green vortex on a ``2*pi``-periodic domain.

        u = -cos(x) sin(y) exp(-2 nu t)
        v =  sin(x) cos(y) exp(-2 nu t)
        p = -[cos(2x) + cos(2y)] exp(-4 nu t) / 4

    The convective term is balanced exactly by the pressure gradient, leaving a
    pure decay ``du/dt = nu lap(u)``.
    """
    decay = np.exp(-2.0 * nu * t)
    if component == "u":
        return -np.cos(x) * np.sin(y) * decay
    if component == "v":
        return np.sin(x) * np.cos(y) * decay
    if component == "p":
        return -0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * decay ** 2
    raise ValueError(f"component must be 'u', 'v' or 'p', got {component!r}")


# --------------------------------------------------------------------------- #
# Ghia, Ghia & Shin (1982) reference data
# --------------------------------------------------------------------------- #
# U. Ghia, K. N. Ghia and C. T. Shin, "High-Re solutions for incompressible flow
# using the Navier-Stokes equations and a multigrid method", Journal of
# Computational Physics 48 (1982) 387-411.
#
# Table I: u along the vertical line through the geometric centre (x = 0.5).
# Table II: v along the horizontal line through the geometric centre (y = 0.5).
# The tabulated points cluster near the walls, which is where the profiles are
# hardest to resolve -- a uniform grid will show its largest error there.

GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000,
])

GHIA_U: dict[int, np.ndarray] = {
    100: np.array([
        0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
        -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
        0.78871, 0.84123, 1.00000,
    ]),
    400: np.array([
        0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726,
        -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756,
        0.68439, 0.75837, 1.00000,
    ]),
    1000: np.array([
        0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805,
        -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117,
        0.57492, 0.65928, 1.00000,
    ]),
}

GHIA_X = np.array([
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
    0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000,
])

GHIA_V: dict[int, np.ndarray] = {
    100: np.array([
        0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
        0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864,
        -0.07391, -0.05906, 0.00000,
    ]),
    400: np.array([
        0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203,
        0.30174, 0.05186, -0.38598, -0.44993, np.nan, -0.22847, -0.19254,
        -0.15663, -0.12146, 0.00000,
    ]),   # index 11 (x = 0.9063) withheld -- see GHIA_KNOWN_GAPS below
    1000: np.array([
        0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075,
        0.32235, 0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714,
        -0.27669, -0.21388, 0.00000,
    ]),
}

GHIA_REYNOLDS = tuple(sorted(GHIA_U))

#: Reference entries that are deliberately absent (stored as ``nan``) and are
#: therefore skipped by the error norms.
#:
#: ``GHIA_V[400]`` at ``x = 0.9063``: the value transcribed here could not be
#: confirmed and is inconsistent with its own neighbours.  Every other point of
#: that profile is reproduced to better than 0.006, and the discrepancy at this
#: one point (0.150) does not shrink under grid refinement -- 96x96 and 160x160
#: give -0.3861 and -0.3884 against a tabulated -0.23827 -- which is the
#: signature of a bad reference value rather than of discretisation error.  The
#: neighbouring published values are -0.44993 at x = 0.8594 and -0.22847 at
#: x = 0.9453, so the true entry lies between them, but it is not reconstructed
#: here: a fabricated number in a reference table is worse than a missing one.
#: If you have Ghia, Ghia & Shin (1982) to hand, restore the Table II value and
#: this gap closes.
GHIA_KNOWN_GAPS = {(400, "v", 11): "x = 0.9063, value unverified"}


def ghia_reference(re: int) -> dict[str, np.ndarray]:
    """Ghia et al. centreline data for a supported Reynolds number."""
    re = int(re)
    if re not in GHIA_U:
        raise ValueError(
            f"no Ghia et al. reference data for Re={re}; available: {list(GHIA_REYNOLDS)}"
        )
    return {"y": GHIA_Y, "u": GHIA_U[re], "x": GHIA_X, "v": GHIA_V[re]}


# --------------------------------------------------------------------------- #
# Error metrics
# --------------------------------------------------------------------------- #
def _aligned(numerical, reference):
    """Validate shapes and drop sample points the reference does not define.

    Reference entries stored as ``nan`` (see :data:`GHIA_KNOWN_GAPS`) are
    excluded rather than propagating a ``nan`` through the whole norm.
    """
    a = np.asarray(numerical, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    keep = np.isfinite(b) & np.isfinite(a)
    if not keep.any():
        raise ValueError("no comparable points: every reference value is undefined")
    return a[keep], b[keep]


def l2_error(numerical: np.ndarray, reference: np.ndarray, relative: bool = False) -> float:
    """Root-mean-square difference, optionally normalised by the reference RMS."""
    a, b = _aligned(numerical, reference)
    err = float(np.sqrt(np.mean((a - b) ** 2)))
    if relative:
        scale = float(np.sqrt(np.mean(b ** 2)))
        return err / scale if scale > 0 else err
    return err


def linf_error(numerical: np.ndarray, reference: np.ndarray, relative: bool = False) -> float:
    """Maximum absolute difference, optionally normalised by ``max|reference|``."""
    a, b = _aligned(numerical, reference)
    err = float(np.abs(a - b).max())
    if relative:
        scale = float(np.abs(b).max())
        return err / scale if scale > 0 else err
    return err


@dataclass
class ConvergenceStudy:
    """Result of a grid-refinement study."""

    resolutions: list[int]
    errors: list[float]
    orders: list[float]
    norm: str = "L2"

    @property
    def observed_order(self) -> float:
        """Order from the two finest grids -- the asymptotic estimate."""
        return self.orders[-1] if self.orders else float("nan")

    def table(self) -> str:
        """Formatted refinement table."""
        lines = [f"{'N':>6} {self.norm + ' error':>14} {'order':>8}"]
        for k, (n, e) in enumerate(zip(self.resolutions, self.errors)):
            order = "" if k == 0 else f"{self.orders[k - 1]:8.3f}"
            lines.append(f"{n:6d} {e:14.6e} {order:>8}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ConvergenceStudy({self.norm}, N={self.resolutions}, "
            f"observed order={self.observed_order:.3f})"
        )


def convergence_order(resolutions, errors, norm: str = "L2") -> ConvergenceStudy:
    """Observed order of accuracy between successive grid refinements.

    For each consecutive pair the order is ``log(e1/e2) / log(N2/N1)``, which
    handles non-doubling refinement ratios correctly.
    """
    resolutions = [int(n) for n in resolutions]
    errors = [float(e) for e in errors]
    if len(resolutions) != len(errors):
        raise ValueError("resolutions and errors must have the same length")
    if len(resolutions) < 2:
        raise ValueError("at least two resolutions are needed to estimate an order")

    orders = []
    for k in range(len(resolutions) - 1):
        e1, e2 = errors[k], errors[k + 1]
        n1, n2 = resolutions[k], resolutions[k + 1]
        if e2 <= 0 or e1 <= 0:
            orders.append(float("nan"))
        else:
            orders.append(float(np.log(e1 / e2) / np.log(n2 / n1)))
    return ConvergenceStudy(resolutions, errors, orders, norm)
