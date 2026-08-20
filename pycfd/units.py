"""Dimensional quantities and the solver's internal scales.

The solver is non-dimensional: it never sees a metre, a second or a pascal.
Every field it produces is expressed in the scales the configuration declares --
``u_ref`` for velocity, ``l_ref`` for the length that forms the Reynolds number
-- and turning those back into real units is a step someone has to take by hand.
Doing it by hand is where the mistakes live.  This module is the bridge:

* :func:`atmosphere` gives the ISA properties of air at an altitude, so a
  Reynolds number can be formed from a speed and a size rather than looked up in
  a table;
* :func:`reynolds_number` and :func:`mach_number` form the two dimensionless
  groups that decide whether a run is meaningful at all;
* :class:`Scaling` converts a finished run's numbers back into m/s, Pa and
  seconds.

The mistake it exists to prevent
--------------------------------
``SimulationConfig.u_ref`` is *not* a free annotation of "how fast the real
thing goes".  It is the velocity scale the fields are already expressed in, and
force coefficients are divided by its square.  Setting it to a real flight speed
while the inlet still drives the flow at 1.0 does not rescale anything; it
divides every force coefficient by that speed squared and quietly reports a drag
coefficient near zero.  The correct route is the opposite one: leave the solver
at ``u_ref = 1``, put the real speed into the *Reynolds number*, and convert the
results afterwards with :class:`Scaling`.  :meth:`SimulationConfig.validate`
enforces the first half of that; this module supplies the second.

Everything here is plain floats and pure functions -- it imports nothing else
from the package, so it can be used to work out a configuration before there is
a configuration to work out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Physical constants (ISA)
# --------------------------------------------------------------------------- #
#: Standard gravity, m/s^2.
GRAVITY = 9.80665

#: Specific gas constant for dry air, J/(kg K).
GAS_CONSTANT_AIR = 287.05287

#: Ratio of specific heats for air.
HEAT_CAPACITY_RATIO = 1.4

#: Sutherland's law coefficients for the dynamic viscosity of air.
SUTHERLAND_BETA = 1.458e-6      # kg/(m s sqrt(K))
SUTHERLAND_S = 110.4            # K

#: ISA sea-level reference state.
SEA_LEVEL_TEMPERATURE = 288.15  # K
SEA_LEVEL_PRESSURE = 101325.0   # Pa

#: Temperature lapse rate in the troposphere, K/m (negative: it gets colder).
TROPOSPHERE_LAPSE_RATE = -0.0065

#: Top of the troposphere; above it the ISA holds the temperature constant.
TROPOPAUSE_ALTITUDE = 11000.0   # m

#: Altitudes outside this band are refused rather than extrapolated: the
#: two-layer model above is only the ISA up to the top of the lower stratosphere.
MIN_ALTITUDE = -5000.0          # m
MAX_ALTITUDE = 20000.0          # m

#: Mach number above which treating the flow as incompressible stops being
#: defensible -- the usual engineering rule, where the density change across the
#: stagnation point first passes a few per cent.
INCOMPRESSIBLE_MACH_LIMIT = 0.3


# --------------------------------------------------------------------------- #
# Atmosphere
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Atmosphere:
    """Air properties at one altitude, in SI units.

    Attributes
    ----------
    altitude:
        Geopotential altitude, m.
    temperature:
        Static temperature, K.
    pressure:
        Static pressure, Pa.
    density:
        Mass density, kg/m^3.
    dynamic_viscosity:
        Absolute viscosity ``mu``, Pa s.
    kinematic_viscosity:
        ``nu = mu / rho``, m^2/s -- the one the Reynolds number is formed with.
    speed_of_sound:
        ``sqrt(gamma R T)``, m/s.
    """

    altitude: float
    temperature: float
    pressure: float
    density: float
    dynamic_viscosity: float
    kinematic_viscosity: float
    speed_of_sound: float

    def summary(self) -> str:
        """One-line description for logs and reports."""
        return (
            f"ISA {self.altitude:g} m: T={self.temperature:.2f} K  "
            f"p={self.pressure:.0f} Pa  rho={self.density:.4f} kg/m^3  "
            f"nu={self.kinematic_viscosity:.4g} m^2/s  a={self.speed_of_sound:.1f} m/s"
        )


def _sutherland_viscosity(temperature: float) -> float:
    """Dynamic viscosity of air from Sutherland's law, Pa s."""
    return SUTHERLAND_BETA * temperature ** 1.5 / (temperature + SUTHERLAND_S)


def atmosphere(altitude: float = 0.0) -> Atmosphere:
    """ISA air properties at ``altitude`` metres.

    Implements the two lowest layers of the International Standard Atmosphere:
    a constant lapse rate up to the tropopause at 11 km, then an isothermal
    stratosphere.  That covers every altitude an incompressible 2D solver has
    any business being pointed at; anything outside
    ``[MIN_ALTITUDE, MAX_ALTITUDE]`` raises rather than extrapolating a model
    that has stopped applying.

    Examples
    --------
    >>> round(atmosphere().kinematic_viscosity, 9)
    1.4607e-05
    """
    if not MIN_ALTITUDE <= altitude <= MAX_ALTITUDE:
        raise ValueError(
            f"altitude {altitude:g} m lies outside the modelled band "
            f"[{MIN_ALTITUDE:g}, {MAX_ALTITUDE:g}] m; the two-layer ISA used here "
            "does not extend that far"
        )

    if altitude <= TROPOPAUSE_ALTITUDE:
        temperature = SEA_LEVEL_TEMPERATURE + TROPOSPHERE_LAPSE_RATE * altitude
        exponent = -GRAVITY / (TROPOSPHERE_LAPSE_RATE * GAS_CONSTANT_AIR)
        pressure = SEA_LEVEL_PRESSURE * (temperature / SEA_LEVEL_TEMPERATURE) ** exponent
    else:
        # Isothermal layer: the barometric formula, anchored at the tropopause.
        tropopause = atmosphere(TROPOPAUSE_ALTITUDE)
        temperature = tropopause.temperature
        pressure = tropopause.pressure * math.exp(
            -GRAVITY * (altitude - TROPOPAUSE_ALTITUDE)
            / (GAS_CONSTANT_AIR * temperature)
        )

    density = pressure / (GAS_CONSTANT_AIR * temperature)
    mu = _sutherland_viscosity(temperature)
    return Atmosphere(
        altitude=float(altitude),
        temperature=temperature,
        pressure=pressure,
        density=density,
        dynamic_viscosity=mu,
        kinematic_viscosity=mu / density,
        speed_of_sound=math.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT_AIR * temperature),
    )


# --------------------------------------------------------------------------- #
# Dimensionless groups
# --------------------------------------------------------------------------- #
def reynolds_number(speed: float, length: float,
                    kinematic_viscosity: float) -> float:
    """``Re = V L / nu``, from quantities in consistent units.

    Parameters
    ----------
    speed:
        Free-stream speed, m/s.
    length:
        Reference length, m -- the size the Reynolds number is *about*.  Which
        length that is, is a convention, not a fact: a cylinder uses its
        diameter, an aerofoil its chord, an aircraft its overall length.  Pick
        one deliberately and report it alongside the number.
    kinematic_viscosity:
        ``nu`` in m^2/s; :func:`atmosphere` supplies it for air.
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if length <= 0:
        raise ValueError(f"reference length must be positive, got {length}")
    if kinematic_viscosity <= 0:
        raise ValueError(
            f"kinematic viscosity must be positive, got {kinematic_viscosity}"
        )
    return speed * length / kinematic_viscosity


def mach_number(speed: float, speed_of_sound: float) -> float:
    """``M = V / a``, the ratio this solver is not allowed to forget.

    An incompressible formulation has no density equation at all, so a run above
    :data:`INCOMPRESSIBLE_MACH_LIMIT` is not a slightly worse answer -- it is a
    different set of equations from the ones that govern the flow.
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if speed_of_sound <= 0:
        raise ValueError(f"speed of sound must be positive, got {speed_of_sound}")
    return speed / speed_of_sound


# --------------------------------------------------------------------------- #
# Solver units <-> physical units
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scaling:
    """The exchange rate between solver units and SI.

    A run is set up in solver units and read back in real ones.  Two numbers fix
    the whole conversion: the physical speed that one unit of solver velocity
    stands for, and the physical length that one unit of solver length stands
    for.  Everything else follows from those two and the density.

    Note that ``length`` is the size of the solver's *length unit*, not the
    reference length of the Reynolds number.  For the cylinder benchmark the two
    coincide (the diameter is 1.0 and the domain is measured in diameters); for
    a body loaded from a geometry file they usually do not, since the file
    carries its own units and the reference length is some span within it.

    Parameters
    ----------
    speed:
        Metres per second represented by one solver velocity unit -- the real
        free-stream speed, when the inlet runs at ``u_ref = 1``.
    length:
        Metres represented by one solver length unit.
    density:
        Air density, kg/m^3.  Defaults to ISA sea level.
    kinematic_viscosity, speed_of_sound:
        The remaining fluid properties, defaulted to ISA sea level so a quick
        conversion needs only ``speed`` and ``length``.
    """

    speed: float
    length: float = 1.0
    density: float = 1.225
    kinematic_viscosity: float = 1.4607e-5
    speed_of_sound: float = 340.294

    # ------------------------------------------------------------------ #
    @classmethod
    def at_altitude(cls, speed: float, length: float = 1.0,
                    altitude: float = 0.0) -> "Scaling":
        """Build a scaling whose fluid properties come from the ISA."""
        air = atmosphere(altitude)
        return cls(
            speed=speed, length=length, density=air.density,
            kinematic_viscosity=air.kinematic_viscosity,
            speed_of_sound=air.speed_of_sound,
        )

    def __post_init__(self) -> None:
        for name in ("speed", "length", "density", "kinematic_viscosity",
                     "speed_of_sound"):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"Scaling.{name} must be positive, got {value}")

    # ------------------------------------------------------------------ #
    @property
    def time_scale(self) -> float:
        """Seconds represented by one solver time unit, ``L / V``."""
        return self.length / self.speed

    @property
    def pressure_scale(self) -> float:
        """Pascals represented by one solver pressure unit, ``rho V^2``.

        The solver integrates the kinematic pressure ``p / rho`` in units of
        ``V^2``, so this single factor covers both steps at once.
        """
        return self.density * self.speed ** 2

    @property
    def dynamic_pressure(self) -> float:
        """Free-stream dynamic pressure ``0.5 rho V^2``, Pa."""
        return 0.5 * self.density * self.speed ** 2

    @property
    def mach(self) -> float:
        """Free-stream Mach number."""
        return mach_number(self.speed, self.speed_of_sound)

    @property
    def compressible(self) -> bool:
        """True when the incompressible assumption no longer holds."""
        return self.mach > INCOMPRESSIBLE_MACH_LIMIT

    def reynolds(self, reference_length: float) -> float:
        """Reynolds number formed with ``reference_length`` in metres."""
        return reynolds_number(self.speed, reference_length,
                               self.kinematic_viscosity)

    # -- solver -> physical --------------------------------------------- #
    def to_speed(self, u: float) -> float:
        """Solver velocity -> m/s."""
        return u * self.speed

    def to_length(self, x: float) -> float:
        """Solver length -> m."""
        return x * self.length

    def to_seconds(self, t: float) -> float:
        """Solver time -> s."""
        return t * self.time_scale

    def to_pascals(self, p: float) -> float:
        """Solver (kinematic, ``V^2``-scaled) pressure -> Pa."""
        return p * self.pressure_scale

    def to_vorticity(self, w: float) -> float:
        """Solver vorticity -> 1/s.

        Vorticity is a velocity gradient, so it carries the reciprocal of the
        time scale rather than either primary scale on its own.
        """
        return w / self.time_scale

    # -- physical -> solver --------------------------------------------- #
    def from_speed(self, u: float) -> float:
        """m/s -> solver velocity."""
        return u / self.speed

    def from_length(self, x: float) -> float:
        """m -> solver length."""
        return x / self.length

    def from_seconds(self, t: float) -> float:
        """s -> solver time."""
        return t / self.time_scale

    def from_pascals(self, p: float) -> float:
        """Pa -> solver pressure."""
        return p / self.pressure_scale

    def from_vorticity(self, w: float) -> float:
        """1/s -> solver vorticity."""
        return w * self.time_scale

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        """Human-readable block describing what the solver's units mean."""
        return (
            f"1 solver velocity = {self.speed:g} m/s   "
            f"1 solver length = {self.length:g} m   "
            f"1 solver time = {self.time_scale:.4g} s\n"
            f"rho = {self.density:.4f} kg/m^3   "
            f"q_inf = {self.dynamic_pressure:.4g} Pa   "
            f"M = {self.mach:.3f}"
        )
