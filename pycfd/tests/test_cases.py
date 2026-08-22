"""Case construction: overrides reach the configuration they are meant to reach.

``build()`` on every case pins its own physics defaults (``use_les``, a name,
a steady-state tolerance tuned to how fast that flow settles) and then forwards
``**overrides`` into :class:`~pycfd.config.SimulationConfig`. A default that is
passed as an explicit keyword *and* left in ``overrides`` is a ``TypeError``
waiting for the first caller who supplies it -- which is exactly what happened
to ``steady_tol`` on the cavity and the channel: both passed it positionally
alongside ``**overrides`` instead of using ``overrides.setdefault``, so a
caller-supplied ``steady_tol`` collided with the case's own.
"""

import pycfd.cases.channel_flow as channel_flow
import pycfd.cases.lid_driven_cavity as lid_driven_cavity


def test_a_caller_supplied_steady_tol_reaches_the_cavity_configuration():
    sim = lid_driven_cavity.build(nx=8, ny=8, steady_tol=1e-3)
    assert sim.config.steady_tol == 1e-3


def test_the_cavity_falls_back_to_its_own_steady_tolerance():
    sim = lid_driven_cavity.build(nx=8, ny=8)
    assert sim.config.steady_tol == lid_driven_cavity.STEADY_TOLERANCE


def test_a_caller_supplied_steady_tol_reaches_the_channel_configuration():
    sim = channel_flow.build(nx=16, ny=32, steady_tol=1e-11)
    assert sim.config.steady_tol == 1e-11


def test_the_channel_falls_back_to_its_own_steady_tolerance():
    sim = channel_flow.build(nx=16, ny=32)
    assert sim.config.steady_tol == channel_flow.STEADY_TOLERANCE
