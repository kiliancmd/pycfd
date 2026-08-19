"""pycfd -- a 2D incompressible Navier-Stokes solver.

A teaching and research-prototype CFD code built around Chorin's projection
method on a staggered (MAC) grid.  See ``README.md`` for a quickstart.
"""

from .config import (
    AdvectionScheme,
    BCKind,
    BCSpec,
    PressureSolver,
    SimulationConfig,
    SolverType,
    TimeScheme,
)

__version__ = "1.0.0"

__all__ = [
    "SimulationConfig", "BCSpec", "BCKind", "SolverType",
    "TimeScheme", "AdvectionScheme", "PressureSolver", "__version__",
]
