"""Dimensional bookkeeping: the ISA table, the two dimensionless groups, and
the guard that keeps ``u_ref`` meaning what the solver thinks it means.

The atmosphere assertions are against the *published* ISA table rather than
against whatever this implementation happens to return, so a rearranged formula
that no longer reproduces standard air is a failure rather than a re-recording.
"""

import math

import pytest

from pycfd.config import BCKind, BCSpec, SimulationConfig
from pycfd.units import (
    INCOMPRESSIBLE_MACH_LIMIT,
    MAX_ALTITUDE,
    TROPOPAUSE_ALTITUDE,
    Scaling,
    atmosphere,
    mach_number,
    reynolds_number,
)

from .conftest import walls


# --------------------------------------------------------------------------- #
# The standard atmosphere
# --------------------------------------------------------------------------- #
#: Published ISA values: altitude m -> (T K, p Pa, rho kg/m^3, a m/s).
ISA_TABLE = {
    0: (288.15, 101325.0, 1.2250, 340.29),
    5000: (255.65, 54019.9, 0.73612, 320.53),
    11000: (216.65, 22632.0, 0.36392, 295.07),
    15000: (216.65, 12044.6, 0.19367, 295.07),
}


@pytest.mark.parametrize("altitude", sorted(ISA_TABLE))
def test_atmosphere_reproduces_the_published_isa(altitude):
    temperature, pressure, density, sound = ISA_TABLE[altitude]
    air = atmosphere(altitude)
    assert air.temperature == pytest.approx(temperature, rel=1e-5)
    assert air.pressure == pytest.approx(pressure, rel=1e-4)
    assert air.density == pytest.approx(density, rel=1e-4)
    assert air.speed_of_sound == pytest.approx(sound, rel=1e-4)


def test_sea_level_viscosity_matches_the_standard_value():
    """The number the Reynolds number is actually formed from."""
    air = atmosphere()
    assert air.dynamic_viscosity == pytest.approx(1.7894e-5, rel=1e-3)
    assert air.kinematic_viscosity == pytest.approx(1.4607e-5, rel=1e-4)


def test_the_two_isa_layers_meet_at_the_tropopause():
    """A discontinuity here would be a sign-or-exponent slip in one branch."""
    below = atmosphere(TROPOPAUSE_ALTITUDE - 1e-6)
    above = atmosphere(TROPOPAUSE_ALTITUDE + 1e-6)
    assert below.pressure == pytest.approx(above.pressure, rel=1e-9)
    assert below.temperature == pytest.approx(above.temperature, rel=1e-9)


def test_air_thins_and_slows_sound_with_altitude():
    ground, high = atmosphere(0.0), atmosphere(10000.0)
    assert high.density < ground.density
    assert high.speed_of_sound < ground.speed_of_sound
    # Thinner air is also stickier per unit mass, which is why a high-altitude
    # Reynolds number is lower than the same speed and size at sea level.
    assert high.kinematic_viscosity > ground.kinematic_viscosity


@pytest.mark.parametrize("altitude", [-6000.0, MAX_ALTITUDE + 1.0])
def test_atmosphere_refuses_altitudes_it_does_not_model(altitude):
    with pytest.raises(ValueError, match="outside the modelled band"):
        atmosphere(altitude)


def test_density_is_consistent_with_the_gas_law():
    air = atmosphere(7000.0)
    from pycfd.units import GAS_CONSTANT_AIR

    assert air.pressure == pytest.approx(
        air.density * GAS_CONSTANT_AIR * air.temperature, rel=1e-12)


def test_kinematic_viscosity_is_the_dynamic_one_over_density():
    air = atmosphere(2500.0)
    assert air.kinematic_viscosity == pytest.approx(
        air.dynamic_viscosity / air.density, rel=1e-12)


# --------------------------------------------------------------------------- #
# Dimensionless groups
# --------------------------------------------------------------------------- #
def test_reynolds_number_is_the_textbook_ratio():
    assert reynolds_number(10.0, 2.0, 1e-5) == pytest.approx(2.0e6)


def test_reynolds_number_at_sea_level_matches_a_hand_calculation():
    """70 m/s over a 2.6 m body, the case that motivated the flag."""
    nu = atmosphere().kinematic_viscosity
    assert reynolds_number(70.0, 2.6, nu) == pytest.approx(70.0 * 2.6 / nu)


@pytest.mark.parametrize("args", [
    (0.0, 1.0, 1e-5),        # no speed
    (1.0, 0.0, 1e-5),        # no length
    (1.0, 1.0, 0.0),         # inviscid: Re is undefined, not infinite
])
def test_reynolds_number_refuses_a_degenerate_input(args):
    with pytest.raises(ValueError):
        reynolds_number(*args)


def test_mach_number_is_the_speed_ratio():
    assert mach_number(170.147, 340.294) == pytest.approx(0.5)


def test_mach_number_refuses_a_degenerate_input():
    with pytest.raises(ValueError):
        mach_number(1.0, 0.0)


# --------------------------------------------------------------------------- #
# Scaling: solver units <-> SI
# --------------------------------------------------------------------------- #
def test_scaling_derives_the_secondary_scales():
    s = Scaling(speed=50.0, length=2.0, density=1.2)
    assert s.time_scale == pytest.approx(2.0 / 50.0)
    assert s.pressure_scale == pytest.approx(1.2 * 50.0 ** 2)
    assert s.dynamic_pressure == pytest.approx(0.5 * 1.2 * 50.0 ** 2)


@pytest.mark.parametrize("to,back", [
    ("to_speed", "from_speed"),
    ("to_length", "from_length"),
    ("to_seconds", "from_seconds"),
    ("to_pascals", "from_pascals"),
])
def test_every_conversion_round_trips(to, back):
    s = Scaling.at_altitude(70.0, length=2.6, altitude=3000.0)
    assert getattr(s, back)(getattr(s, to)(3.25)) == pytest.approx(3.25, rel=1e-12)


def test_solver_units_are_the_identity_at_unit_scales():
    """With speed = length = 1 the bridge does nothing, as the docs claim."""
    s = Scaling(speed=1.0, length=1.0, density=1.0)
    assert s.to_speed(0.4) == pytest.approx(0.4)
    assert s.to_seconds(0.4) == pytest.approx(0.4)
    assert s.to_pascals(0.4) == pytest.approx(0.4)


def test_at_altitude_takes_its_fluid_properties_from_the_isa():
    air = atmosphere(8000.0)
    s = Scaling.at_altitude(120.0, length=1.0, altitude=8000.0)
    assert s.density == pytest.approx(air.density)
    assert s.kinematic_viscosity == pytest.approx(air.kinematic_viscosity)
    assert s.speed_of_sound == pytest.approx(air.speed_of_sound)
    assert s.mach == pytest.approx(120.0 / air.speed_of_sound)


def test_scaling_reynolds_uses_the_length_it_is_handed():
    """The reference length is an argument, not the solver's length unit."""
    s = Scaling.at_altitude(70.0, length=1.0)
    assert s.reynolds(2.6) == pytest.approx(2.6 * s.reynolds(1.0))


def test_compressible_flags_exactly_the_engineering_limit():
    a = atmosphere().speed_of_sound
    assert not Scaling(speed=0.9 * INCOMPRESSIBLE_MACH_LIMIT * a).compressible
    assert Scaling(speed=1.1 * INCOMPRESSIBLE_MACH_LIMIT * a).compressible


@pytest.mark.parametrize("field", ["speed", "length", "density",
                                  "kinematic_viscosity", "speed_of_sound"])
def test_scaling_refuses_a_non_positive_scale(field):
    with pytest.raises(ValueError, match=f"Scaling.{field} must be positive"):
        Scaling(**{"speed": 1.0, field: 0.0})


def test_summaries_mention_the_quantities_they_report():
    assert "m^2/s" in atmosphere(1000.0).summary()
    text = Scaling.at_altitude(70.0, altitude=3000.0).summary()
    assert "m/s" in text and "M =" in text


# --------------------------------------------------------------------------- #
# The guard: u_ref is the scale the fields are already in
# --------------------------------------------------------------------------- #
# Raising u_ref to a flight speed while the inlet still runs at 1.0 is the exact
# mistake this module exists to make unnecessary.  It is silent -- the run looks
# healthy and reports a drag coefficient near zero -- so the configuration
# refuses it outright rather than leaving it to be noticed downstream.
def inlet_config(velocity: float, u_ref: float) -> SimulationConfig:
    return SimulationConfig(
        nx=16, ny=16, re=100.0, u_ref=u_ref, name="test",
        boundary_config=walls(left=BCSpec(BCKind.INLET, velocity=velocity),
                              right=BCKind.OUTLET),
    )


def test_an_inlet_that_disagrees_with_u_ref_is_refused():
    with pytest.raises(ValueError, match="drives the flow at 1 but u_ref is 70"):
        inlet_config(velocity=1.0, u_ref=70.0)


def test_the_error_points_at_the_supported_route():
    with pytest.raises(ValueError, match="wind-speed"):
        inlet_config(velocity=1.0, u_ref=70.0)


@pytest.mark.parametrize("speed", [1.0, 0.5, 70.0])
def test_an_inlet_matching_u_ref_is_accepted(speed):
    assert inlet_config(velocity=speed, u_ref=speed).u_ref == speed


def test_an_inlet_blowing_the_other_way_is_compared_on_magnitude():
    """``u_ref`` is a speed; the sign only says which way the flow enters."""
    assert inlet_config(velocity=-2.0, u_ref=2.0).u_ref == 2.0
    with pytest.raises(ValueError, match="drives the flow at -2"):
        inlet_config(velocity=-2.0, u_ref=70.0)


def test_a_quiescent_inlet_does_not_constrain_u_ref():
    """Zero inflow sets no velocity scale, so there is nothing to disagree with."""
    assert inlet_config(velocity=0.0, u_ref=2.0).u_ref == 2.0


def test_the_benchmark_cases_all_satisfy_the_guard():
    """Every shipped case must survive its own validation."""
    from pycfd.cases import available_cases, load_case

    for name in available_cases():
        cfg = load_case(name).build().config
        cfg.validate()          # would already have raised in build; be explicit
        assert cfg.u_ref > 0


def test_a_flight_speed_belongs_in_the_reynolds_number_not_in_u_ref():
    """The positive half of the lesson, stated as an executable example."""
    nu = atmosphere(3000.0).kinematic_viscosity
    re = reynolds_number(70.0, 2.6, nu)
    cfg = inlet_config(velocity=1.0, u_ref=1.0).replace(re=re)
    # The solver stays non-dimensional...
    assert cfg.u_ref == 1.0
    # ...and the real speed survives in the only place it belongs.
    assert cfg.re == pytest.approx(re)
    # Reading the answer back out is then a conversion, not a re-run.
    s = Scaling.at_altitude(70.0, length=1.0, altitude=3000.0)
    assert s.to_speed(1.0) == pytest.approx(70.0)
    assert math.isclose(s.reynolds(2.6), cfg.re, rel_tol=1e-12)
