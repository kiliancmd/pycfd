"""Shared fixtures and helpers for the pycfd test suite."""

import numpy as np
import pytest

from pycfd.config import BCKind, BCSpec, SimulationConfig
from pycfd.core.solver import ProjectionSolver

#: Divergence this small is machine precision, not discretisation error --
#: the projection is expected to hit it exactly.
DIVERGENCE_TOL = 1.0e-11


def walls(**kinds) -> dict[str, BCSpec]:
    """Build a boundary_config, defaulting every unnamed wall to no-slip.

    ``walls(top=(BCKind.MOVING_WALL, 1.0))`` gives a lid-driven cavity.
    """
    out = {w: BCSpec(BCKind.NO_SLIP) for w in ("left", "right", "bottom", "top")}
    for wall, spec in kinds.items():
        if isinstance(spec, tuple):
            kind, velocity = spec
            out[wall] = BCSpec(kind, velocity=velocity)
        elif isinstance(spec, BCSpec):
            out[wall] = spec
        else:
            out[wall] = BCSpec(spec)
    return out


def periodic_walls() -> dict[str, BCSpec]:
    """Doubly periodic boundary configuration."""
    return {w: BCSpec(BCKind.PERIODIC) for w in ("left", "right", "bottom", "top")}


#: Physics the suite assumes unless a test says otherwise.
#:
#: ``SimulationConfig`` defaults belong to whoever is using the library and are
#: expected to change with the case being worked on.  A unit test must not
#: inherit them: a domain-dependent assertion silently becomes wrong when the
#: domain default moves, and a laminar test silently starts exercising the LES
#: path.  Every test therefore builds its configuration through
#: :func:`make_config`, which pins exactly the fields the suite reasons about.
TEST_DEFAULTS = {
    "lx": 1.0,
    "ly": 1.0,
    "re": 100.0,
    "dt": 1.0e-3,
    "t_end": 10.0,
    "cfl_max": 0.5,
    "use_les": False,
    "name": "test",
}


def make_config(**kw) -> SimulationConfig:
    """A :class:`SimulationConfig` on the suite's baseline, overridable per test."""
    return SimulationConfig(**{**TEST_DEFAULTS, **kw})


def cavity_config(nx=24, ny=24, re=100.0, **kw) -> SimulationConfig:
    """Small lid-driven cavity configuration for fast tests."""
    return make_config(
        nx=nx, ny=ny, re=re, dt=kw.pop("dt", 2.0e-3), t_end=kw.pop("t_end", 1.0),
        boundary_config=walls(top=(BCKind.MOVING_WALL, 1.0)), name="test_cavity", **kw
    )


@pytest.fixture
def cavity_solver():
    """A small cavity solver with its initialised field."""
    solver = ProjectionSolver(cavity_config())
    return solver, solver.initialize()


def taylor_green_fields(mesh, t=0.0, nu=0.05):
    """Exact Taylor--Green velocity at the staggered face locations."""
    Xu, Yu = mesh.u_grid()
    Xv, Yv = mesh.v_grid()
    decay = np.exp(-2.0 * nu * t)
    return -np.cos(Xu) * np.sin(Yu) * decay, np.sin(Xv) * np.cos(Yv) * decay
