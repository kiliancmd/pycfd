"""Typed configuration objects for the pycfd solver.

Everything the solver needs to know is expressed here as a dataclass, so a case
is fully described by a single serialisable object.  No numerical constant is
allowed to live inside the solver modules -- if a number influences the physics
or the discretisation it belongs in this file (or is derived from it).

Non-dimensionalisation
----------------------
The momentum equation that is actually discretised is

    du/dt + (u.grad)u = -grad p + nu * lap(u) + f

with the kinematic viscosity derived from the Reynolds number as

    nu = u_ref * l_ref / re

With the defaults ``u_ref = l_ref = 1`` this reduces to ``nu = 1/re``, matching
the classic non-dimensional form.  Cases whose reference length is *not* the
domain size (flow past a cylinder, where ``l_ref`` is the diameter) must set
``l_ref`` explicitly, otherwise the Reynolds number of the simulation is not the
Reynolds number that was asked for.

``u_ref`` is the velocity scale the fields are *already* expressed in, not an
annotation of how fast the real thing goes: force coefficients are divided by
its square.  Raising it to a flight speed while an inlet still drives the flow
at 1.0 therefore does not rescale the run, it only drives every reported
coefficient towards zero.  :meth:`SimulationConfig.validate` refuses that
combination outright; see :mod:`pycfd.units` for the supported route, which is
to leave the solver at ``u_ref = 1``, put the real speed into the Reynolds
number, and convert the results afterwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class SolverType(str, Enum):
    """Pressure--velocity coupling strategy."""

    PROJECTION = "projection"   # Chorin fractional step (implemented)
    SIMPLE = "simple"           # Semi-Implicit Method for Pressure-Linked Eqns


class TimeScheme(str, Enum):
    """Explicit time integrator for the momentum equation.

    ``EULER`` is the textbook baseline but is *linearly unstable* for pure
    central-difference advection (its stability region only touches the
    imaginary axis at the origin); it survives in practice solely because the
    viscous term damps the growth.  ``RK3`` (SSP Shu--Osher) contains a genuine
    interval of the imaginary axis and is the safe default.
    """

    EULER = "euler"
    RK2 = "rk2"
    RK3 = "rk3"


class AdvectionScheme(str, Enum):
    """Spatial discretisation of the convective term."""

    CENTRAL = "central"   # 2nd order, energy conserving, no numerical diffusion
    UPWIND = "upwind"     # central blended with donor-cell (1st order, stable)


class PressureSolver(str, Enum):
    """Linear solver used for the pressure Poisson equation."""

    DIRECT = "direct"           # sparse LU factorised once, re-used every step
    CG = "cg"                   # conjugate gradient (symmetric singular system)
    JACOBI = "jacobi"           # stationary, for educational transparency
    SOR = "sor"                 # red/black successive over-relaxation
    MULTIGRID = "multigrid"     # aggregation multigrid V-cycles, standalone
    MGCG = "mgcg"               # conjugate gradient preconditioned by a V-cycle


class BCKind(str, Enum):
    """Boundary condition types understood by :mod:`pycfd.core.boundary`."""

    NO_SLIP = "no_slip"
    MOVING_WALL = "moving_wall"
    INLET = "inlet"
    OUTLET = "outlet"
    PRESSURE_OUTLET = "pressure_outlet"
    PERIODIC = "periodic"
    SYMMETRY = "symmetry"


# --------------------------------------------------------------------------- #
# Boundary specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BCSpec:
    """Lightweight description of one wall's boundary condition.

    ``config`` deliberately stores a *specification* rather than a live
    :class:`~pycfd.core.boundary.BoundaryCondition` object: it keeps this module
    free of solver imports and makes configurations trivially serialisable.

    Parameters
    ----------
    kind:
        Which boundary condition to build.
    velocity:
        Wall/inlet velocity.  Interpretation depends on ``kind``:
        ``MOVING_WALL`` uses it as the *tangential* wall speed, ``INLET`` as the
        peak or uniform *normal* inflow speed.
    profile:
        Inlet profile shape, ``"uniform"`` or ``"parabolic"``.
    p_ref:
        Pressure held on the boundary face by ``PRESSURE_OUTLET``; ignored by
        every other kind.  Only pressure *differences* drive the flow, so the
        value is a datum -- it fixes the level of the reported field.
    """

    kind: BCKind = BCKind.NO_SLIP
    velocity: float = 0.0
    profile: str = "uniform"
    p_ref: float = 0.0

    def __post_init__(self) -> None:
        if self.profile not in ("uniform", "parabolic"):
            raise ValueError(
                f"BCSpec.profile must be 'uniform' or 'parabolic', got {self.profile!r}"
            )


WALLS = ("left", "right", "bottom", "top")


def _default_boundaries() -> dict[str, BCSpec]:
    """All four walls no-slip -- a closed box."""
    return {w: BCSpec(BCKind.NO_SLIP) for w in WALLS}


# --------------------------------------------------------------------------- #
# Numerical safety constants
# --------------------------------------------------------------------------- #
#: Fraction of the explicit-diffusion stability limit that the adaptive time
#: step is allowed to use.  The theoretical 2D limit for forward Euler is
#: dt <= 1 / (2*nu*(1/dx^2 + 1/dy^2)); we stay comfortably below it.
VISCOUS_SAFETY_FACTOR = 0.8

#: |u| above which the run is declared divergent and aborted.
DIVERGENCE_VELOCITY_LIMIT = 1.0e6

#: Adaptive time step below which the run is considered stalled.
MIN_TIME_STEP = 1.0e-8

#: Velocity scale used to keep the first adaptive dt finite when the field is
#: initialised to exactly zero everywhere.
QUIESCENT_VELOCITY_FLOOR = 1.0e-12

#: Default tolerance/iteration budget for the iterative pressure solvers.
DEFAULT_POISSON_TOL = 1.0e-10
DEFAULT_POISSON_MAXITER = 20_000

#: Over-relaxation factor for SOR.  1.0 recovers plain Gauss-Seidel.
DEFAULT_SOR_OMEGA = 1.8

#: Damped-Jacobi sweeps on each side of the coarse-grid correction in a
#: multigrid V-cycle.  One is the cheaper cycle and two is the better one; which
#: wins on wall time depends on the grid, so this is a knob rather than a
#: constant.  See :mod:`pycfd.core.multigrid`.
DEFAULT_MG_SWEEPS = 1


# --------------------------------------------------------------------------- #
# Main configuration object
# --------------------------------------------------------------------------- #
@dataclass
class SimulationConfig:
    """Complete description of one simulation run.

    Attributes are validated in :meth:`validate`, which is invoked
    automatically by ``__post_init__`` so an invalid configuration can never
    reach the solver.
    """

    # -- grid/mesh -------------------------------------------------------------- #
    nx: int = 128
    ny: int = 128
    lx: float = 1.0
    ly: float = 1.0
    stretch_x: float = 1.0          # geometric cell-growth ratio (1.0 = uniform)
    stretch_y: float = 1.0
    cluster_x: str = "low"          # where the small cells go: low/walls/centre
    cluster_y: str = "low"

    # -- time -------------------------------------------------------------- #
    dt: float = 1.0e-3              # initial / fixed time step
    t_end: float = 10.0
    max_steps: int | None = None    # optional hard cap on iteration count
    adaptive_dt: bool = True
    cfl_max: float = 0.5

    # -- physics ----------------------------------------------------------- #
    re: float = 100.0
    u_ref: float = 1.0              # reference velocity used to define nu
    l_ref: float = 1.0              # reference length used to define nu
    body_force: tuple[float, float] = (0.0, 0.0)

    # -- numerics ---------------------------------------------------------- #
    solver_type: SolverType = SolverType.PROJECTION
    time_scheme: TimeScheme = TimeScheme.RK3
    advection_scheme: AdvectionScheme = AdvectionScheme.CENTRAL
    upwind_blend: float | None = None   # None -> derived from the local CFL
    pressure_solver: PressureSolver = PressureSolver.DIRECT
    poisson_tol: float = DEFAULT_POISSON_TOL
    poisson_maxiter: int = DEFAULT_POISSON_MAXITER
    sor_omega: float = DEFAULT_SOR_OMEGA
    mg_sweeps: int = DEFAULT_MG_SWEEPS

    use_numba: bool = True          # use fused JIT stencils when numba is present

    # -- turbulence (optional) --------------------------------------------- #
    use_les: bool = False
    smagorinsky_cs: float = 0.17    # standard Smagorinsky constant

    # -- boundaries -------------------------------------------------------- #
    boundary_config: dict[str, BCSpec] = field(default_factory=_default_boundaries)

    # -- steady-state detection -------------------------------------------- #
    steady_tol: float | None = None  # stop when max|du/dt| falls below this

    # -- bookkeeping ------------------------------------------------------- #
    name: str = "simulation"

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        # Allow plain strings from the CLI / JSON to be promoted to enums.
        self.solver_type = SolverType(self.solver_type)
        self.time_scheme = TimeScheme(self.time_scheme)
        self.advection_scheme = AdvectionScheme(self.advection_scheme)
        self.pressure_solver = PressureSolver(self.pressure_solver)
        self.body_force = tuple(float(c) for c in self.body_force)  # type: ignore[assignment]
        self.validate()

    # ------------------------------------------------------------------ #
    @property
    def nu(self) -> float:
        """Kinematic viscosity implied by the Reynolds number."""
        return self.u_ref * self.l_ref / self.re

    @property
    def dx(self) -> float:
        """Uniform cell width (meaningless if ``stretch_x != 1``)."""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """Uniform cell height (meaningless if ``stretch_y != 1``)."""
        return self.ly / self.ny

    def is_periodic(self, axis: str) -> bool:
        """True if ``axis`` (``"x"`` or ``"y"``) is periodic."""
        pair = ("left", "right") if axis == "x" else ("bottom", "top")
        return all(self.boundary_config[w].kind is BCKind.PERIODIC for w in pair)

    def dirichlet_pressure_walls(self) -> dict[str, float]:
        """Walls that hold the pressure fixed, mapped to their ``p_ref``.

        Non-empty means the pressure Poisson operator is non-singular: the level
        is set by physics rather than by an arbitrary reference cell.
        """
        return {
            w: spec.p_ref for w, spec in self.boundary_config.items()
            if spec.kind is BCKind.PRESSURE_OUTLET
        }

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Raise :class:`ValueError` on any physically or numerically invalid input."""
        if self.nx < 4 or self.ny < 4:
            raise ValueError(f"grid must be at least 4x4, got {self.nx}x{self.ny}")
        if self.lx <= 0 or self.ly <= 0:
            raise ValueError(f"domain size must be positive, got {self.lx}x{self.ly}")
        if self.re <= 0:
            raise ValueError(f"Reynolds number must be positive, got {self.re}")
        if self.u_ref <= 0 or self.l_ref <= 0:
            raise ValueError("u_ref and l_ref must be positive")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.t_end <= 0:
            raise ValueError(f"t_end must be positive, got {self.t_end}")
        if not 0 < self.cfl_max <= 1.0:
            raise ValueError(f"cfl_max must lie in (0, 1], got {self.cfl_max}")
        if self.stretch_x <= 0 or self.stretch_y <= 0:
            raise ValueError("stretch ratios must be positive")
        # Normalise here rather than in the mesh so a bad spelling is caught
        # when the config is built, not several seconds into a run.
        from .core.mesh import _normalise_cluster
        object.__setattr__(self, "cluster_x", _normalise_cluster(self.cluster_x))
        object.__setattr__(self, "cluster_y", _normalise_cluster(self.cluster_y))
        if self.upwind_blend is not None and not 0.0 <= self.upwind_blend <= 1.0:
            raise ValueError(
                f"upwind_blend must lie in [0, 1], got {self.upwind_blend}"
            )
        if not 0 < self.sor_omega < 2:
            raise ValueError(f"sor_omega must lie in (0, 2), got {self.sor_omega}")
        if self.mg_sweeps < 1:
            raise ValueError(
                f"mg_sweeps must be at least 1, got {self.mg_sweeps}: a V-cycle "
                "with no smoothing does nothing but move error between grids"
            )
        if self.smagorinsky_cs < 0:
            raise ValueError("smagorinsky_cs must be non-negative")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1 when given")

        missing = set(WALLS) - set(self.boundary_config)
        if missing:
            raise ValueError(f"boundary_config is missing walls: {sorted(missing)}")
        unknown = set(self.boundary_config) - set(WALLS)
        if unknown:
            raise ValueError(f"boundary_config has unknown walls: {sorted(unknown)}")

        # An inlet *is* the velocity scale.  Letting the two drift apart is the
        # one configuration error that produces a plausible-looking run and a
        # silently wrong answer -- forces come back divided by the wrong square
        # -- so it is refused here rather than reported later.
        # ``u_ref`` is a magnitude, so an inlet blowing the other way (a negative
        # velocity, inflow from the right) is compared on magnitude too.
        for wall, spec in self.boundary_config.items():
            speed = abs(spec.velocity)
            if spec.kind is BCKind.INLET and speed > 0 and \
                    abs(speed - self.u_ref) > 1e-12 * max(1.0, self.u_ref):
                raise ValueError(
                    f"the {wall} inlet drives the flow at {spec.velocity:g} but "
                    f"u_ref is {self.u_ref:g}; forces are normalised by u_ref^2, "
                    "so the two must be the same speed. To run at a physical "
                    "speed, keep u_ref equal to the inlet velocity and put the "
                    "speed into 're' instead (see pycfd.units.reynolds_number "
                    "and the --wind-speed flag)"
                )

        # Periodicity must be declared on both walls of an axis or neither.
        for axis, (a, b) in (("x", ("left", "right")), ("y", ("bottom", "top"))):
            pa = self.boundary_config[a].kind is BCKind.PERIODIC
            pb = self.boundary_config[b].kind is BCKind.PERIODIC
            if pa != pb:
                raise ValueError(
                    f"periodicity on the {axis}-axis must be set on both "
                    f"'{a}' and '{b}', got {self.boundary_config[a].kind.value} / "
                    f"{self.boundary_config[b].kind.value}"
                )

    # ------------------------------------------------------------------ #
    def replace(self, **changes: Any) -> "SimulationConfig":
        """Return a validated copy with ``changes`` applied."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary representation."""
        d = asdict(self)
        d["boundary_config"] = {
            w: {"kind": s.kind.value, "velocity": s.velocity,
                "profile": s.profile, "p_ref": s.p_ref}
            for w, s in self.boundary_config.items()
        }
        d["solver_type"] = self.solver_type.value
        d["time_scheme"] = self.time_scheme.value
        d["advection_scheme"] = self.advection_scheme.value
        d["pressure_solver"] = self.pressure_solver.value
        d["body_force"] = list(self.body_force)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimulationConfig":
        """Inverse of :meth:`to_dict`."""
        d = dict(d)
        if "boundary_config" in d:
            d["boundary_config"] = {
                w: BCSpec(BCKind(s["kind"]), s.get("velocity", 0.0),
                          s.get("profile", "uniform"), s.get("p_ref", 0.0))
                for w, s in d["boundary_config"].items()
            }
        if "body_force" in d:
            d["body_force"] = tuple(d["body_force"])
        return cls(**d)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Human-readable one-block summary for logs."""
        bcs = ", ".join(f"{w}={self.boundary_config[w].kind.value}" for w in WALLS)
        return (
            f"case={self.name}  grid={self.nx}x{self.ny}  domain={self.lx}x{self.ly}\n"
            f"Re={self.re:g}  nu={self.nu:.4g}  u_ref={self.u_ref:g}  l_ref={self.l_ref:g}\n"
            f"time={self.time_scheme.value}  advection={self.advection_scheme.value}  "
            f"poisson={self.pressure_solver.value}\n"
            f"dt={self.dt:g} (adaptive={self.adaptive_dt}, CFL_max={self.cfl_max:g})  "
            f"t_end={self.t_end:g}\n"
            f"BCs: {bcs}"
        )
