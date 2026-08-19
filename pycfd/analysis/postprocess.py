"""Derived quantities: vorticity, stream function, forces, energy, probes.

Everything here consumes arrays and returns arrays -- no plotting, no file I/O.
Quantities that live naturally at cell corners on a staggered grid (vorticity,
stream function) are computed there and optionally averaged to cell centres for
plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.fields import FlowField


# --------------------------------------------------------------------------- #
# Vorticity
# --------------------------------------------------------------------------- #
def vorticity_corner(fields: FlowField) -> np.ndarray:
    """Vorticity ``omega = dv/dx - du/dy`` at cell corners, shape ``(nx+1, ny+1)``.

    Both derivatives are exact central differences of adjacent staggered values,
    so this is the natural second-order vorticity of the MAC grid -- no
    interpolation is involved.
    """
    nx, ny = fields.mesh.shape
    dx, dy = fields.mesh.dx, fields.mesh.dy
    u, v = fields.u, fields.v
    dvdx = (v[1:nx + 2, 1:ny + 2] - v[0:nx + 1, 1:ny + 2]) / dx
    dudy = (u[1:nx + 2, 1:ny + 2] - u[1:nx + 2, 0:ny + 1]) / dy
    return dvdx - dudy


def vorticity(fields: FlowField) -> np.ndarray:
    """Vorticity averaged to cell centres, shape ``(nx, ny)`` -- the plotting form."""
    w = vorticity_corner(fields)
    return 0.25 * (w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:])


# --------------------------------------------------------------------------- #
# Stream function
# --------------------------------------------------------------------------- #
def stream_function(fields: FlowField) -> np.ndarray:
    """Stream function at cell corners, shape ``(nx+1, ny+1)``, with ``psi = 0`` at the origin.

    Rather than solving ``lap(psi) = -omega``, this integrates the defining
    relations ``u = dpsi/dy`` and ``v = -dpsi/dx`` directly.  The integral is
    path-independent exactly when the discrete divergence vanishes, which the
    projection guarantees to machine precision -- so direct integration is both
    cheaper and more accurate than a Poisson solve here, and it reproduces the
    prescribed boundary values exactly.
    """
    nx, ny = fields.mesh.shape
    dx, dy = fields.mesh.dx, fields.mesh.dy

    psi = np.zeros((nx + 1, ny + 1))
    # Walk along the bottom edge using v, then up each column using u.
    v_bottom = fields.v[1:nx + 1, 1]                    # v on the y=0 face
    psi[1:, 0] = -np.cumsum(v_bottom) * dx
    u_col = fields.u[1:nx + 2, 1:ny + 1]                # u on every x-face
    psi[:, 1:] = psi[:, 0][:, None] + np.cumsum(u_col, axis=1) * dy
    return psi


# --------------------------------------------------------------------------- #
# Integral quantities
# --------------------------------------------------------------------------- #
def kinetic_energy(fields: FlowField, mask: np.ndarray | None = None) -> float:
    """``0.5 * sum(u^2 + v^2) * dA`` over the cell centres."""
    uc, vc = fields.cell_velocities()
    e = 0.5 * (uc ** 2 + vc ** 2)
    if mask is not None:
        e = np.where(mask, 0.0, e)
    return float(e.sum()) * fields.mesh.cell_area


def enstrophy(fields: FlowField, mask: np.ndarray | None = None) -> float:
    """``0.5 * sum(omega^2) * dA`` over the cell centres."""
    w = vorticity(fields)
    e = 0.5 * w ** 2
    if mask is not None:
        e = np.where(mask, 0.0, e)
    return float(e.sum()) * fields.mesh.cell_area


def divergence_norms(fields: FlowField, solver) -> tuple[float, float]:
    """``(L2, Linf)`` norms of the discrete divergence -- a solver health check."""
    d = solver.divergence(fields.u, fields.v)
    if getattr(solver, "has_obstacle", False):
        d = d[~solver.solid]
    if d.size == 0:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(d ** 2))), float(np.abs(d).max())


# --------------------------------------------------------------------------- #
# Forces
# --------------------------------------------------------------------------- #
def force_coefficients(force: tuple[float, float], u_ref: float,
                       l_ref: float, density: float = 1.0) -> tuple[float, float]:
    """Convert a force per unit depth to ``(Cd, Cl)``.

    Uses the standard 2D normalisation ``C = F / (0.5 * rho * u_ref^2 * l_ref)``.
    """
    if u_ref <= 0 or l_ref <= 0:
        raise ValueError("u_ref and l_ref must be positive to form a coefficient")
    q = 0.5 * density * u_ref ** 2 * l_ref
    return force[0] / q, force[1] / q


def strouhal_number(times: np.ndarray, signal: np.ndarray,
                    l_ref: float, u_ref: float) -> float:
    """Shedding Strouhal number ``St = f * L / U`` from an oscillating signal.

    The dominant frequency is taken from the FFT of the mean-removed signal,
    resampled onto a uniform time base first (the solver uses adaptive steps).
    Returns ``nan`` when the record is too short to resolve a peak.
    """
    times = np.asarray(times, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if times.size < 16:
        return float("nan")

    uniform_t = np.linspace(times[0], times[-1], times.size)
    s = np.interp(uniform_t, times, signal)
    s = s - s.mean()
    dt = uniform_t[1] - uniform_t[0]
    if dt <= 0:
        return float("nan")

    spectrum = np.abs(np.fft.rfft(s * np.hanning(s.size)))
    freqs = np.fft.rfftfreq(s.size, dt)
    spectrum[0] = 0.0                    # ignore the residual mean
    peak = int(np.argmax(spectrum))
    if peak == 0:
        return float("nan")
    return float(freqs[peak] * l_ref / u_ref)


# --------------------------------------------------------------------------- #
# Profiles and probes
# --------------------------------------------------------------------------- #
def centerline_profiles(fields: FlowField):
    """Centreline profiles used for the cavity benchmark.

    Returns ``(y, u_vertical, x, v_horizontal)``: ``u`` sampled along the
    vertical line ``x = lx/2`` and ``v`` along the horizontal line ``y = ly/2``,
    both interpolated to the exact centre so that odd and even grids are handled
    identically.
    """
    mesh = fields.mesh
    uc, vc = fields.cell_velocities()

    u_line = _interp_axis(uc, mesh.xc, 0.5 * mesh.lx, axis=0)
    v_line = _interp_axis(vc, mesh.yc, 0.5 * mesh.ly, axis=1)
    return mesh.yc, u_line, mesh.xc, v_line


def _interp_axis(arr: np.ndarray, coords: np.ndarray, target: float, axis: int) -> np.ndarray:
    """Linearly interpolate ``arr`` at ``target`` along ``axis``."""
    i = int(np.clip(np.searchsorted(coords, target) - 1, 0, coords.size - 2))
    w = (target - coords[i]) / (coords[i + 1] - coords[i])
    lo = arr[i] if axis == 0 else arr[:, i]
    hi = arr[i + 1] if axis == 0 else arr[:, i + 1]
    return (1.0 - w) * lo + w * hi


def sample_at(fields: FlowField, x: float, y: float) -> dict[str, float]:
    """Bilinearly interpolated ``u``, ``v`` and ``p`` at a point."""
    mesh = fields.mesh
    uc, vc = fields.cell_velocities()
    out = {}
    for name, arr in (("u", uc), ("v", vc), ("p", fields.p_phys)):
        out[name] = float(_bilinear(arr, mesh.xc, mesh.yc, x, y))
    return out


def _bilinear(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray, x: float, y: float) -> float:
    """Bilinear interpolation with clamping at the array edges."""
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, xs.size - 2))
    j = int(np.clip(np.searchsorted(ys, y) - 1, 0, ys.size - 2))
    tx = np.clip((x - xs[i]) / (xs[i + 1] - xs[i]), 0.0, 1.0)
    ty = np.clip((y - ys[j]) / (ys[j + 1] - ys[j]), 0.0, 1.0)
    return (
        (1 - tx) * (1 - ty) * arr[i, j] + tx * (1 - ty) * arr[i + 1, j]
        + (1 - tx) * ty * arr[i, j + 1] + tx * ty * arr[i + 1, j + 1]
    )


@dataclass
class Probe:
    """Time series recorder for a single point in the domain."""

    x: float
    y: float
    name: str = "probe"
    t: list[float] = field(default_factory=list)
    u: list[float] = field(default_factory=list)
    v: list[float] = field(default_factory=list)
    p: list[float] = field(default_factory=list)

    def record(self, fields: FlowField) -> None:
        """Append the current values at the probe location."""
        s = sample_at(fields, self.x, self.y)
        self.t.append(fields.t)
        self.u.append(s["u"])
        self.v.append(s["v"])
        self.p.append(s["p"])

    def as_arrays(self) -> dict[str, np.ndarray]:
        """The recorded series as NumPy arrays."""
        return {k: np.asarray(getattr(self, k), dtype=float)
                for k in ("t", "u", "v", "p")}

    def __repr__(self) -> str:
        return f"Probe({self.name!r} at ({self.x:g}, {self.y:g}), {len(self.t)} samples)"
