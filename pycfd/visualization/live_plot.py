"""Real-time animation of a running simulation.

The viewer owns the time loop: :class:`matplotlib.animation.FuncAnimation` calls
back every frame, the viewer advances the solver by ``plot_every`` steps and
redraws.  That keeps the solver itself free of any plotting code -- ``core/``
never imports Matplotlib.

Interaction
-----------
``space`` pauses and resumes, ``q`` closes the window.  While paused the solver
does no work, so a paused window costs nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analysis.postprocess import vorticity
from ..core.timestepper import DivergenceError

#: Fields the viewer can display.
DISPLAY_MODES = ("speed", "pressure", "vorticity", "streamlines")

#: Target delay between frames in milliseconds.
FRAME_INTERVAL_MS = 30

#: Colour-scale percentile, so a couple of singular cells cannot wash out the map.
COLOR_PERCENTILE = 99.0


class LiveViewer:
    """Animate a :class:`~pycfd.physics.incompressible.Simulation` as it runs.

    Parameters
    ----------
    simulation:
        A prepared simulation; the viewer advances it in place.
    mode:
        One of :data:`DISPLAY_MODES`.
    plot_every:
        Solver steps taken between rendered frames.
    t_end:
        Stop time; defaults to the configuration's ``t_end``.
    quiver:
        Overlay velocity vectors on the scalar field.
    """

    def __init__(self, simulation, mode: str = "speed", plot_every: int = 50,
                 t_end: float | None = None, quiver: bool = False,
                 rescale_every: int = 10) -> None:
        if mode not in DISPLAY_MODES:
            raise ValueError(f"mode must be one of {DISPLAY_MODES}, got {mode!r}")
        self.sim = simulation
        self.mode = mode
        self.plot_every = max(1, int(plot_every))
        self.t_end = simulation.config.t_end if t_end is None else t_end
        self.quiver = quiver
        self.rescale_every = max(1, int(rescale_every))

        self.paused = False
        self.finished = False
        self._frames = 0
        self._error: Exception | None = None

    # ------------------------------------------------------------------ #
    def _field(self) -> np.ndarray:
        """Scalar array currently being displayed, in ``(nx, ny)`` ordering."""
        f = self.sim.fields
        if self.mode == "pressure":
            return f.p_phys
        if self.mode == "vorticity":
            return vorticity(f)
        uc, vc = f.cell_velocities()
        return np.hypot(uc, vc)

    def _limits(self, data: np.ndarray) -> tuple[float, float]:
        """Colour limits: symmetric for signed fields, zero-based for magnitudes."""
        solid = self.sim.solid_mask
        vals = data[~solid] if solid is not None else data
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0, 1.0
        if self.mode in ("pressure", "vorticity"):
            lim = float(np.percentile(np.abs(vals), COLOR_PERCENTILE)) or 1.0
            return -lim, lim
        return 0.0, float(np.percentile(vals, COLOR_PERCENTILE)) or 1.0

    def _title(self) -> str:
        d = self.sim.fields.diagnostics or {}
        cfl = d.get("cfl", float("nan"))
        dt = d.get("dt", float("nan"))
        state = "  [PAUSED]" if self.paused else ""
        return (
            f"{self.sim.config.name}  Re={self.sim.config.re:g}  |  "
            f"step {self.sim.fields.step}   t = {self.sim.fields.t:.4f}   "
            f"dt = {dt:.2e}   CFL = {cfl:.3f}{state}"
        )

    # ------------------------------------------------------------------ #
    def _advance(self) -> None:
        """Run ``plot_every`` solver steps, stopping cleanly on error or t_end."""
        if self.paused or self.finished:
            return
        try:
            self.sim.run(t_end=self.t_end, max_steps=self.plot_every)
        except DivergenceError as exc:
            self._error = exc
            self.finished = True
            return
        if self.sim.fields.t >= self.t_end - 1e-12:
            self.finished = True

    # ------------------------------------------------------------------ #
    def start(self, save_path: str | Path | None = None, fps: int = 20,
              max_frames: int | None = None, show: bool = True):
        """Build the figure and run the animation.

        Parameters
        ----------
        save_path:
            Write the animation to this file instead of only displaying it.
            Requires a Matplotlib writer (``pillow`` handles ``.gif``).
        max_frames:
            Hard cap on rendered frames; required when ``save_path`` is given.
        show:
            Call ``plt.show()``.  Disable for headless rendering.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        mesh = self.sim.mesh
        solid = self.sim.solid_mask

        fig, ax = plt.subplots(figsize=(8.0, 8.0 * mesh.ly / mesh.lx + 1.0))
        data = self._field()
        lo, hi = self._limits(data)
        plotted = np.ma.masked_where(solid, data) if solid is not None else data

        cmap = "RdBu_r" if self.mode in ("pressure", "vorticity") else "viridis"
        # imshow is used rather than contourf: it can be updated in place, which
        # keeps the frame rate usable on a 128x128 grid and larger.
        image = ax.imshow(
            plotted.T, origin="lower", extent=(0, mesh.lx, 0, mesh.ly),
            cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear", aspect="equal",
        )
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label(self.mode)

        if solid is not None and solid.any():
            ax.contour(mesh.xc, mesh.yc, solid.T.astype(float), levels=[0.5],
                       colors="k", linewidths=1.2)

        quiver_artist = None
        if self.quiver:
            stride = max(1, min(mesh.nx, mesh.ny) // 20)
            X, Y = mesh.cell_center_grid()
            sl = (slice(None, None, stride), slice(None, None, stride))
            uc, vc = self.sim.fields.cell_velocities()
            quiver_artist = ax.quiver(X[sl], Y[sl], uc[sl], vc[sl], color="w", scale_units="xy")
            self._quiver_slice = sl

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(self._title(), fontsize=10)

        def on_key(event):
            if event.key == " ":
                self.paused = not self.paused
                ax.set_title(self._title(), fontsize=10)
                fig.canvas.draw_idle()
            elif event.key == "q":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

        def update(_frame):
            self._advance()
            self._frames += 1
            data = self._field()
            plotted = np.ma.masked_where(solid, data) if solid is not None else data
            image.set_data(plotted.T)
            # Rescaling every frame makes the colours flicker; do it periodically.
            if self._frames % self.rescale_every == 0:
                image.set_clim(*self._limits(data))
            if quiver_artist is not None:
                uc, vc = self.sim.fields.cell_velocities()
                quiver_artist.set_UVC(uc[self._quiver_slice], vc[self._quiver_slice])
            title = self._title()
            if self._error is not None:
                title = f"STOPPED: {self._error}"
            ax.set_title(title, fontsize=10)
            return (image,)

        frames = max_frames
        if save_path is not None and frames is None:
            raise ValueError("max_frames must be given when save_path is set")

        anim = FuncAnimation(
            fig, update, frames=frames, interval=FRAME_INTERVAL_MS,
            blit=False, cache_frame_data=False, repeat=False,
        )

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            anim.save(str(path), fps=fps)
            plt.close(fig)
            return anim
        if show:
            plt.show()
        return anim
