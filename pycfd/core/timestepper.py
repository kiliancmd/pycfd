"""Time integration driver: adaptive time-step control and the main loop.

Stability limits
----------------
Two explicit constraints act at once and the smaller one wins.

*Convective.*  The task specification writes this as
``dt = cfl * min(dx, dy) / max(|u|, |v|)``.  The form used here,

    dt_conv = cfl / ( max|u|/dx + max|v|/dy )

is the standard multi-dimensional generalisation: it accounts for both
directions transporting simultaneously, and is never less restrictive than the
per-direction minimum (the two agree to within a factor of two).

*Viscous.*  Explicit diffusion in two dimensions requires

    dt_visc = safety / ( 2 * nu * (1/dx^2 + 1/dy^2) )

This limit is absent from the specification but is not optional: on a fine grid
or at low Reynolds number it is by far the tighter of the two, and omitting it
makes the solver blow up in exactly the cases that should be easiest.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..config import (
    DIVERGENCE_VELOCITY_LIMIT,
    MIN_TIME_STEP,
    QUIESCENT_VELOCITY_FLOOR,
    SimulationConfig,
    VISCOUS_SAFETY_FACTOR,
)
from .fields import FlowField
from .solver import ProjectionSolver

log = logging.getLogger(__name__)


class DivergenceError(RuntimeError):
    """Raised when the solution blows up or produces non-finite values."""


@dataclass
class SimulationResult:
    """Final state plus the per-step diagnostic history."""

    fields: FlowField
    steps: int
    time: float
    wall_time: float
    converged: bool = False           #: True if the steady-state criterion was met
    history: dict[str, list] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line report suitable for logging."""
        rate = self.steps / self.wall_time if self.wall_time > 0 else float("nan")
        status = "steady state reached" if self.converged else "end time reached"
        return (
            f"{status}: {self.steps} steps, t = {self.time:.4f}, "
            f"{self.wall_time:.2f} s wall ({rate:.0f} steps/s)"
        )


class TimeStepper:
    """Runs a :class:`~pycfd.core.solver.ProjectionSolver` forward in time."""

    def __init__(self, solver: ProjectionSolver, config: SimulationConfig | None = None) -> None:
        self.solver = solver
        self.config = config if config is not None else solver.config
        self.mesh = solver.mesh
        self.viscous_dt_limit = self._viscous_limit()

    # ------------------------------------------------------------------ #
    def _viscous_limit(self) -> float:
        """Fixed part of the time-step limit imposed by explicit diffusion."""
        nu = self.solver.nu
        if nu <= 0:
            return float("inf")
        dx, dy = self.mesh.dx, self.mesh.dy
        return VISCOUS_SAFETY_FACTOR / (2.0 * nu * (1.0 / dx ** 2 + 1.0 / dy ** 2))

    def compute_dt(self, fields: FlowField) -> float:
        """Largest stable time step for the current state.

        Returns ``config.dt`` unchanged when adaptive stepping is disabled.
        """
        if not self.config.adaptive_dt:
            return self.config.dt

        umax, vmax = fields.max_velocity()
        dx, dy = self.mesh.dx, self.mesh.dy
        scale = umax / dx + vmax / dy
        dt_conv = (
            self.config.cfl_max / scale
            if scale > QUIESCENT_VELOCITY_FLOOR
            else float("inf")
        )
        # config.dt acts as an upper bound so a quiescent start cannot take an
        # arbitrarily large first step.
        return float(min(dt_conv, self.viscous_dt_limit, self.config.dt))

    def cfl_number(self, fields: FlowField, dt: float) -> float:
        """Courant number actually realised by ``dt``."""
        umax, vmax = fields.max_velocity()
        return float(dt * (umax / self.mesh.dx + vmax / self.mesh.dy))

    # ------------------------------------------------------------------ #
    def _check_health(self, fields: FlowField, dt: float) -> None:
        """Abort loudly on NaN, blow-up or a collapsed time step."""
        if not fields.is_finite():
            raise DivergenceError(
                f"non-finite values at step {fields.step} (t = {fields.t:.6g}). "
                "The most common causes are a time step above the stability "
                "limit and a Reynolds number too high for the grid; try "
                "adaptive_dt=True, a smaller cfl_max, or advection_scheme='upwind'."
            )
        umax, vmax = fields.max_velocity()
        if max(umax, vmax) > DIVERGENCE_VELOCITY_LIMIT:
            raise DivergenceError(
                f"solution diverged at step {fields.step} (t = {fields.t:.6g}): "
                f"max|u| = {umax:.3e}, max|v| = {vmax:.3e} exceeds the limit "
                f"{DIVERGENCE_VELOCITY_LIMIT:.0e}."
            )
        if dt < MIN_TIME_STEP:
            raise DivergenceError(
                f"adaptive time step collapsed to {dt:.3e} (< {MIN_TIME_STEP:.0e}) "
                f"at step {fields.step}; the solution is probably diverging."
            )

    # ------------------------------------------------------------------ #
    def run(
        self,
        fields: FlowField,
        t_end: float | None = None,
        max_steps: int | None = None,
        callback: Callable[[FlowField, dict], None] | None = None,
        callback_every: int = 50,
        progress: bool = False,
        record_every: int = 1,
    ) -> SimulationResult:
        """Integrate from the current state until ``t_end``.

        Parameters
        ----------
        callback:
            Called as ``callback(fields, info)`` every ``callback_every`` steps
            and once at the end.  ``info`` carries the live diagnostics.
        progress:
            Show a ``tqdm`` progress bar driven by simulated time.
        record_every:
            Store diagnostics every ``record_every`` steps.
        """
        t_end = self.config.t_end if t_end is None else t_end
        max_steps = self.config.max_steps if max_steps is None else max_steps
        history: dict[str, list] = {
            k: [] for k in ("step", "t", "dt", "cfl", "max_div", "kinetic_energy",
                            "residual", "fx", "fy")
        }

        bar = None
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=float(t_end), initial=float(fields.t),
                           unit="s", desc=self.config.name, dynamic_ncols=True)
            except ImportError:      # pragma: no cover - tqdm is a soft dependency
                log.warning("tqdm not installed; running without a progress bar")

        wall_start = time.perf_counter()
        converged = False
        steps_taken = 0

        try:
            while fields.t < t_end - 1e-12:
                if max_steps is not None and steps_taken >= max_steps:
                    break

                dt = self.compute_dt(fields)
                dt = min(dt, t_end - fields.t)      # land exactly on t_end
                self._check_health(fields, dt)

                self.solver.set_upwind_blend(self.solver.upwind_blend_for(fields, dt))
                previous = fields
                fields = self.solver.step(previous, dt)
                steps_taken += 1

                du = max(
                    float(np.abs(fields.u - previous.u).max()),
                    float(np.abs(fields.v - previous.v).max()),
                )
                rate_of_change = du / dt

                if steps_taken % record_every == 0:
                    info = self._diagnostics(fields, dt, rate_of_change)
                    for k, val in info.items():
                        if k in history:
                            history[k].append(val)
                    fields.diagnostics = info

                if callback is not None and steps_taken % callback_every == 0:
                    callback(fields, fields.diagnostics or
                             self._diagnostics(fields, dt, rate_of_change))

                if bar is not None:
                    bar.update(float(dt))

                if self.config.steady_tol is not None and rate_of_change < self.config.steady_tol:
                    converged = True
                    log.info(
                        "steady state at step %d (t = %.4f): max|du/dt| = %.3e < %.1e",
                        fields.step, fields.t, rate_of_change, self.config.steady_tol,
                    )
                    break

            self._check_health(fields, self.config.dt)
        finally:
            if bar is not None:
                bar.close()

        wall = time.perf_counter() - wall_start
        if callback is not None:
            callback(fields, fields.diagnostics or
                     self._diagnostics(fields, self.config.dt, float("nan")))

        result = SimulationResult(fields, steps_taken, fields.t, wall, converged, history)
        log.info(result.summary())
        return result

    # ------------------------------------------------------------------ #
    def _diagnostics(self, fields: FlowField, dt: float, rate_of_change: float) -> dict:
        """Collect the per-step diagnostic bundle."""
        uc, vc = fields.cell_velocities()
        cell_area = self.mesh.cell_area
        fx, fy = self.solver.body_force_reaction
        return {
            "step": fields.step,
            "t": fields.t,
            "dt": dt,
            "cfl": self.cfl_number(fields, dt),
            "max_div": self.solver.max_divergence(fields),
            "kinetic_energy": 0.5 * float((uc ** 2 + vc ** 2).sum()) * cell_area,
            "residual": self.solver.pressure_solver.last_residual,
            "rate_of_change": rate_of_change,
            "fx": fx,
            "fy": fy,
        }
