"""Publication-quality static figures.

This layer only ever *receives* arrays -- it never touches the solver.  Field
arrays arrive in the solver's ``(nx, ny)`` ``ij`` ordering and are transposed
once, here, for Matplotlib's ``(row, column)`` convention.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from ..analysis.postprocess import (
    centerline_profiles,
    stream_function,
    vorticity,
)
from ..core.fields import FlowField

#: Resolution for saved figures.
FIGURE_DPI = 300

#: Colour maps used consistently across every figure.
CMAP_SPEED = "viridis"
CMAP_PRESSURE = "RdBu_r"
CMAP_VORTICITY = "RdBu_r"

#: Number of filled contour levels.
N_LEVELS = 40


def _plt():
    """Import ``pyplot``, selecting a non-interactive backend when headless."""
    import matplotlib.pyplot as plt
    return plt


def use_headless_backend() -> None:
    """Force the Agg backend -- call before plotting in a script or on a server."""
    matplotlib.use("Agg", force=True)


def _mask(field: np.ndarray, solid: np.ndarray | None) -> np.ndarray:
    """Hide obstacle cells so they render as blank rather than as zero."""
    if solid is None:
        return field
    return np.ma.masked_where(solid, field)


def _symmetric_levels(field: np.ndarray, n: int = N_LEVELS, clip: float = 99.0):
    """Contour levels symmetric about zero, robust to isolated extremes.

    Vorticity near a moving lid is singular in the continuum limit, so the raw
    maximum is grid-dependent; clipping at a high percentile keeps the colour
    scale informative instead of being dominated by two corner cells.
    """
    data = np.ma.filled(np.ma.masked_invalid(field), 0.0)
    limit = float(np.percentile(np.abs(data), clip))
    if limit <= 0:
        limit = float(np.abs(data).max()) or 1.0
    return np.linspace(-limit, limit, n)


# --------------------------------------------------------------------------- #
# Single panels
# --------------------------------------------------------------------------- #
def plot_velocity_magnitude(ax, fields: FlowField, solid=None, streamlines: bool = True,
                            density: float = 1.2):
    """Filled speed contours with an optional streamline overlay."""
    mesh = fields.mesh
    uc, vc = fields.cell_velocities()
    speed = _mask(np.hypot(uc, vc), solid)

    cf = ax.contourf(mesh.xc, mesh.yc, speed.T, levels=N_LEVELS, cmap=CMAP_SPEED)
    if streamlines:
        # streamplot needs a strictly uniform grid and (row, col) ordering.
        ax.streamplot(mesh.xc, mesh.yc, uc.T, vc.T, color="white",
                      linewidth=0.6, density=density, arrowsize=0.7)
    _finish(ax, mesh, "Velocity magnitude", solid)
    return cf


def plot_pressure(ax, fields: FlowField, solid=None):
    """Filled pressure contours on a diverging scale."""
    mesh = fields.mesh
    p = _mask(fields.p_phys, solid)
    cf = ax.contourf(mesh.xc, mesh.yc, p.T, levels=_symmetric_levels(p),
                     cmap=CMAP_PRESSURE, extend="both")
    _finish(ax, mesh, "Pressure", solid)
    return cf


def plot_vorticity(ax, fields: FlowField, solid=None):
    """Filled vorticity contours on a diverging scale."""
    mesh = fields.mesh
    w = _mask(vorticity(fields), solid)
    cf = ax.contourf(mesh.xc, mesh.yc, w.T, levels=_symmetric_levels(w),
                     cmap=CMAP_VORTICITY, extend="both")
    _finish(ax, mesh, "Vorticity", solid)
    return cf


def plot_stream_function(ax, fields: FlowField, solid=None, n_levels: int = 24):
    """Stream-function contours -- the classic way to see cavity vortices."""
    mesh = fields.mesh
    psi = stream_function(fields)
    cf = ax.contourf(mesh.xf, mesh.yf, psi.T, levels=n_levels, cmap=CMAP_SPEED)
    ax.contour(mesh.xf, mesh.yf, psi.T, levels=n_levels, colors="k", linewidths=0.4)
    _finish(ax, mesh, "Stream function", solid)
    return cf


def plot_quiver(ax, fields: FlowField, solid=None, stride: int | None = None):
    """Velocity vectors, decimated so the arrows stay readable."""
    mesh = fields.mesh
    uc, vc = fields.cell_velocities()
    if stride is None:
        stride = max(1, min(mesh.nx, mesh.ny) // 24)
    X, Y = mesh.cell_center_grid()
    sl = (slice(None, None, stride), slice(None, None, stride))
    ax.quiver(X[sl], Y[sl], _mask(uc, solid)[sl], _mask(vc, solid)[sl], scale_units="xy")
    _finish(ax, mesh, "Velocity vectors", solid)


def _finish(ax, mesh, title: str, solid=None) -> None:
    """Shared axis cosmetics, including the obstacle outline."""
    if solid is not None and solid.any():
        ax.contour(mesh.xc, mesh.yc, solid.T.astype(float), levels=[0.5],
                   colors="k", linewidths=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0, mesh.lx)
    ax.set_ylim(0, mesh.ly)


# --------------------------------------------------------------------------- #
# Composite figures
# --------------------------------------------------------------------------- #
def four_panel_figure(fields: FlowField, solid=None, title: str | None = None,
                      path: str | Path | None = None):
    """The standard 2x2 overview: speed, pressure, vorticity, stream function."""
    plt = _plt()
    mesh = fields.mesh
    aspect = mesh.ly / mesh.lx
    fig, axes = plt.subplots(2, 2, figsize=(11, max(7.0, 10.0 * aspect)))

    for ax, fn in zip(
        axes.ravel(),
        (plot_velocity_magnitude, plot_pressure, plot_vorticity, plot_stream_function),
    ):
        cf = fn(ax, fields, solid)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def centerline_comparison_figure(fields: FlowField, reference: dict | None = None,
                                 title: str | None = None,
                                 path: str | Path | None = None,
                                 reference_label: str = "Ghia et al. (1982)"):
    """Side-by-side centreline profiles against reference data.

    Left: ``u`` along the vertical centreline.  Right: ``v`` along the
    horizontal centreline.  Passing ``reference=None`` plots the computed
    profiles alone.
    """
    plt = _plt()
    y, u_line, x, v_line = centerline_profiles(fields)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    ax1.plot(u_line, y, "-", color="C0", lw=1.8, label="pycfd")
    ax2.plot(x, v_line, "-", color="C0", lw=1.8, label="pycfd")
    if reference is not None:
        # Undefined reference entries are stored as nan; Matplotlib skips them.
        ax1.plot(reference["u"], reference["y"], "o", mfc="none", color="k",
                 ms=5, label=reference_label)
        ax2.plot(reference["x"], reference["v"], "o", mfc="none", color="k",
                 ms=5, label=reference_label)

    ax1.set_xlabel("u"); ax1.set_ylabel("y")
    ax1.set_title("u along the vertical centreline", fontsize=10)
    ax2.set_xlabel("x"); ax2.set_ylabel("v")
    ax2.set_title("v along the horizontal centreline", fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", lw=0.5) if ax is ax2 else ax.axvline(0, color="k", lw=0.5)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def profile_comparison_figure(x, numerical, analytical, xlabel: str, ylabel: str,
                              title: str | None = None, path: str | Path | None = None,
                              analytical_label: str = "analytical"):
    """Computed vs analytical profile with the point-wise error underneath."""
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(x, analytical, "k--", lw=1.6, label=analytical_label)
    ax1.plot(x, numerical, "C0-", lw=1.6, label="pycfd")
    ax1.set_ylabel(ylabel)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(x, np.asarray(numerical) - np.asarray(analytical), "C3-", lw=1.4)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("error")
    ax2.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def convergence_figure(study, title: str | None = None, path: str | Path | None = None):
    """Log-log error-versus-resolution plot with a second-order reference slope."""
    plt = _plt()
    n = np.asarray(study.resolutions, dtype=float)
    e = np.asarray(study.errors, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(n, e, "o-", color="C0", lw=1.8, ms=7, label=f"{study.norm} error")
    ref = e[0] * (n / n[0]) ** -2.0
    ax.loglog(n, ref, "k--", lw=1.2, label="2nd order reference")
    ax.set_xlabel("grid points per direction")
    ax.set_ylabel(f"{study.norm} error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_title(title or f"observed order = {study.observed_order:.2f}", fontsize=11)
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def time_series_figure(history: dict, keys=("kinetic_energy", "max_div", "dt"),
                       title: str | None = None, path: str | Path | None = None):
    """Diagnostic history from a :class:`~pycfd.core.timestepper.SimulationResult`."""
    plt = _plt()
    keys = [k for k in keys if history.get(k)]
    if not keys:
        raise ValueError("no plottable history keys were provided")

    fig, axes = plt.subplots(len(keys), 1, figsize=(7, 2.2 * len(keys)), sharex=True)
    axes = np.atleast_1d(axes)
    t = history["t"]
    for ax, key in zip(axes, keys):
        ax.plot(t, history[key], lw=1.3)
        ax.set_ylabel(key.replace("_", " "))
        ax.grid(alpha=0.3)
        if key in ("max_div", "dt"):
            ax.set_yscale("log")
    axes[-1].set_xlabel("time")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def _save(fig, path: str | Path, provenance: dict | None = None) -> Path:
    """Save at :data:`FIGURE_DPI` and close the figure.

    PNG output carries the run's provenance in standard ``tEXt`` chunks, so a
    figure that has drifted away from its results directory can still say which
    command produced it.  Any tool that reads PNG metadata will surface it.
    """
    from ..analysis.provenance import png_metadata, provenance_record

    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {"dpi": FIGURE_DPI, "bbox_inches": "tight"}
    if path.suffix.lower() == ".png":
        record = provenance_record() if provenance is None else provenance
        kwargs["metadata"] = png_metadata(record)
    fig.savefig(path, **kwargs)
    plt.close(fig)
    return path
