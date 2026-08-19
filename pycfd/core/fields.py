"""Container for the staggered flow variables.

Kept separate from :mod:`pycfd.core.solver` so that
:mod:`pycfd.core.boundary` can operate on a field bundle without importing the
solver (which would be circular).  See :mod:`pycfd.core.mesh` for the index
convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import StructuredMesh


@dataclass
class FlowField:
    """The staggered state ``(u, v, p)`` plus simulation clock.

    All three arrays carry one ghost layer per side; see the module docstring of
    :mod:`pycfd.core.mesh` for the exact numbering.
    """

    mesh: StructuredMesh
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    t: float = 0.0
    step: int = 0
    #: Populated by the solver each step for diagnostics/plotting.
    diagnostics: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def zeros(cls, mesh: StructuredMesh) -> "FlowField":
        """Quiescent field on ``mesh``."""
        return cls(mesh, mesh.zeros_u(), mesh.zeros_v(), mesh.zeros_p())

    def copy(self) -> "FlowField":
        """Deep copy of the state (mesh is shared, it is immutable in practice)."""
        return FlowField(
            self.mesh, self.u.copy(), self.v.copy(), self.p.copy(),
            self.t, self.step, dict(self.diagnostics),
        )

    # ------------------------------------------------------------------ #
    # Physical (non-ghost) views
    # ------------------------------------------------------------------ #
    @property
    def u_phys(self) -> np.ndarray:
        """u on the ``nx+1`` physical x-faces by ``ny`` cell rows."""
        nx, ny = self.mesh.shape
        return self.u[1:nx + 2, 1:ny + 1]

    @property
    def v_phys(self) -> np.ndarray:
        """v on ``nx`` cell columns by the ``ny+1`` physical y-faces."""
        nx, ny = self.mesh.shape
        return self.v[1:nx + 1, 1:ny + 2]

    @property
    def p_phys(self) -> np.ndarray:
        """Pressure on the ``nx x ny`` interior cells."""
        nx, ny = self.mesh.shape
        return self.p[1:nx + 1, 1:ny + 1]

    # ------------------------------------------------------------------ #
    def cell_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        """Velocity interpolated to cell centres, both of shape ``(nx, ny)``.

        This is the form every plotting and post-processing routine wants; the
        solver itself never uses it.
        """
        nx, ny = self.mesh.shape
        uc = 0.5 * (self.u[1:nx + 1, 1:ny + 1] + self.u[2:nx + 2, 1:ny + 1])
        vc = 0.5 * (self.v[1:nx + 1, 1:ny + 1] + self.v[1:nx + 1, 2:ny + 2])
        return uc, vc

    def speed(self) -> np.ndarray:
        """Cell-centred velocity magnitude, shape ``(nx, ny)``."""
        uc, vc = self.cell_velocities()
        return np.hypot(uc, vc)

    def max_velocity(self) -> tuple[float, float]:
        """``(max|u|, max|v|)`` over the physical faces -- used for the CFL limit."""
        return float(np.abs(self.u_phys).max()), float(np.abs(self.v_phys).max())

    def is_finite(self) -> bool:
        """False as soon as any NaN or Inf has appeared in the state."""
        return bool(
            np.all(np.isfinite(self.u))
            and np.all(np.isfinite(self.v))
            and np.all(np.isfinite(self.p))
        )
