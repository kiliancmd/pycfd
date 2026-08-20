"""High-level driver for incompressible flow.

:class:`Simulation` owns the mesh, solver, time stepper and state, and is the
object the benchmark cases and the CLI actually work with.  It exists so that
callers never have to remember the order in which the pieces must be wired up.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from ..analysis import export as export_mod
from ..analysis.postprocess import Probe, kinetic_energy, vorticity
from ..config import SimulationConfig
from ..core.fields import FlowField
from ..core.mesh import StructuredMesh
from ..core.solver import ProjectionSolver
from ..core.timestepper import SimulationResult, TimeStepper
from ..geometry.obstacles import Obstacle

log = logging.getLogger(__name__)


class Simulation:
    """A configured, ready-to-run incompressible flow problem.

    Parameters
    ----------
    config:
        Validated configuration.
    obstacle:
        Optional immersed body.
    u_init, v_init:
        Optional initial condition on the physical face layouts,
        ``(nx+1, ny)`` and ``(nx, ny+1)``.
    """

    def __init__(
        self,
        config: SimulationConfig,
        obstacle: Obstacle | None = None,
        u_init: np.ndarray | float | None = None,
        v_init: np.ndarray | float | None = None,
    ) -> None:
        self.config = config
        self.mesh = StructuredMesh.from_config(config)
        self.obstacle = obstacle
        self.solver = ProjectionSolver(
            config, self.mesh, obstacle.mask if obstacle is not None else None
        )
        self.stepper = TimeStepper(self.solver, config)
        self.fields = self.solver.initialize(u_init, v_init)
        self.probes: list[Probe] = []
        self.result: SimulationResult | None = None
        log.info("simulation ready\n%s", config.summary())

    # ------------------------------------------------------------------ #
    @classmethod
    def from_checkpoint(cls, path: str | Path,
                        obstacle: Obstacle | None = None) -> "Simulation":
        """Rebuild a simulation from a checkpoint and restore its state."""
        fields, config = export_mod.load_checkpoint(path)
        sim = cls(config, obstacle)
        sim.fields = fields
        log.info("resumed from %s at t=%.6g (step %d)", path, fields.t, fields.step)
        return sim

    # ------------------------------------------------------------------ #
    def add_probe(self, x: float, y: float, name: str = "probe") -> Probe:
        """Register a point probe that records on every callback."""
        probe = Probe(x, y, name)
        self.probes.append(probe)
        return probe

    # ------------------------------------------------------------------ #
    def run(
        self,
        t_end: float | None = None,
        max_steps: int | None = None,
        callback: Callable[[FlowField, dict], None] | None = None,
        callback_every: int = 50,
        progress: bool = False,
    ) -> SimulationResult:
        """Integrate forward, recording probes as it goes."""

        def _callback(fields: FlowField, info: dict) -> None:
            for probe in self.probes:
                probe.record(fields)
            if callback is not None:
                callback(fields, info)

        cb = _callback if (self.probes or callback is not None) else None
        self.result = self.stepper.run(
            self.fields, t_end=t_end, max_steps=max_steps,
            callback=cb, callback_every=callback_every, progress=progress,
        )
        self.fields = self.result.fields
        return self.result

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    @property
    def solid_mask(self) -> np.ndarray | None:
        """Obstacle mask, or ``None`` for an unobstructed domain."""
        return self.obstacle.mask if self.obstacle is not None else None

    def force_coefficients(self) -> tuple[float, float]:
        """``(Cd, Cl)`` from the immersed-boundary reaction force.

        The force is the momentum the direct forcing had to remove to hold the
        body at rest, which is the discrete equivalent of integrating pressure
        and shear over the surface.  It is normalised with ``config.u_ref`` and
        ``config.l_ref`` -- the same two scales that define the Reynolds number,
        so the coefficient and the Reynolds number always refer to the same
        length.  A case that puts a body in the flow sets ``l_ref`` from that
        body (or from whatever convention the caller named instead).
        """
        if self.obstacle is None:
            return 0.0, 0.0
        from ..analysis.postprocess import force_coefficients
        return force_coefficients(
            self.solver.body_force_reaction, self.config.u_ref, self.config.l_ref,
        )

    def outlet_pressure_deviation(self) -> float:
        """Largest departure of a pressure-outlet face from its ``p_ref``.

        Zero to round-off whenever a pressure outlet is active; a growing value
        would mean the Dirichlet condition and the projection had come apart.
        Returns ``nan`` when no wall holds the pressure.
        """
        walls = self.solver.dirichlet_pressure
        if not walls:
            return float("nan")
        from ..core.boundary import wall_index

        worst = 0.0
        for wall, p_ref in walls.items():
            wi = wall_index(wall, self.mesh)
            p = self.fields.p
            ghost = p[wi.p_ghost, :] if wi.axis == 0 else p[:, wi.p_ghost]
            inner = p[wi.p_in, :] if wi.axis == 0 else p[:, wi.p_in]
            face = 0.5 * (ghost + inner)
            worst = max(worst, float(np.abs(face - p_ref).max()))
        return worst

    def diagnostics(self) -> dict[str, float]:
        """Current integral diagnostics."""
        mask = self.solid_mask
        cd, cl = self.force_coefficients()
        out = {
            "t": self.fields.t,
            "step": self.fields.step,
            "kinetic_energy": kinetic_energy(self.fields, mask),
            "max_divergence": self.solver.max_divergence(self.fields),
            "max_vorticity": float(np.abs(vorticity(self.fields)).max()),
            "cd": cd,
            "cl": cl,
        }
        if self.solver.dirichlet_pressure:
            out["outlet_pressure_dev"] = self.outlet_pressure_deviation()
        return out

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def provenance(self, **extra) -> dict:
        """Record describing this run, for embedding in anything it writes.

        Carries the configuration and the live diagnostics, so an exported file
        records not just how the run was set up but what it had converged to.
        """
        from ..analysis.provenance import provenance_record

        facts = {k: v for k, v in self.diagnostics().items()}
        if self.obstacle is not None:
            facts["obstacle"] = self.obstacle.name
            facts["characteristic_length"] = self.obstacle.characteristic_length
        facts.update(extra)
        return provenance_record(self.config, extra=facts)

    def save_checkpoint(self, path: str | Path) -> Path:
        """Write a restartable checkpoint."""
        return export_mod.save_checkpoint(self.fields, self.config, path,
                                          provenance=self.provenance())

    def export_vtk(self, path: str | Path, scaling=None) -> Path:
        """Write the current field as a legacy VTK file.

        ``scaling`` is a :class:`~pycfd.units.Scaling`; passing one writes the
        file in SI instead of solver units.
        """
        return export_mod.export_vtk(self.fields, path, self.config.name,
                                     provenance=self.provenance(),
                                     scaling=scaling)

    def export_csv(self, path: str | Path, scaling=None) -> Path:
        """Write the current field as CSV, in SI when given a ``scaling``."""
        return export_mod.export_csv(self.fields, path,
                                     provenance=self.provenance(),
                                     scaling=scaling)

    def __repr__(self) -> str:
        return (
            f"Simulation({self.config.name!r}, {self.mesh.nx}x{self.mesh.ny}, "
            f"Re={self.config.re:g}, t={self.fields.t:.4g})"
        )
