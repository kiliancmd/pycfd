"""Benchmark regression: the recorded numbers must not move silently.

The existing benchmark tests assert *tolerances* -- convergence order above 1.8,
cavity profiles within 0.02 of Ghia.  Those catch a break, but not a drift: a
change that quietly moved second-order convergence from 1.9997 to 1.85 would
still pass every one of them.

These tests pin the measured values themselves against
:mod:`baselines.json <pycfd.tests>`.  A failure here is not automatically a bug
-- it means a number moved, and someone has to decide whether that was a
regression or an improvement.  Re-recording is therefore a deliberate act with
its own script (``tools/record_baselines.py``), never an automatic refresh.

Two tiers, because fidelity costs time:

*Fast* (~10 s total) runs every case at a reduced resolution and is part of the
default suite, so CI catches drift on every push.

*Full* reruns the cases at the resolutions the README quotes -- minutes of work,
gated behind ``--runslow``.  This is what makes the documented headline numbers
executable rather than prose.
"""

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

from pycfd.cases import load_case  # noqa: E402

BASELINES = json.loads((Path(__file__).parent / "baselines.json").read_text())

#: Relative agreement required of a genuine measurement.  Tight enough that a
#: changed stencil or default shows up immediately, loose enough to survive the
#: float variation between platforms and BLAS builds.
DEFAULT_RTOL = 1.0e-4

#: A baseline smaller than this was never a measurement -- it is a quantity that
#: is structurally zero (a divergence the projection drives out, the lift on a
#: symmetric steady wake).  Comparing those relatively is meaningless; they are
#: checked against an absolute bound instead.
NOISE_FLOOR = 1.0e-9

#: What each structurally-zero quantity is allowed to be.
STRUCTURAL_BOUND = {"max_divergence": 1.0e-10, "cl_rms": 1.0e-8}
DEFAULT_STRUCTURAL_BOUND = 1.0e-8


def check(case: str, metric: str, measured: float, expected: float) -> None:
    """Compare one metric against its baseline, relatively or absolutely."""
    if abs(expected) < NOISE_FLOOR:
        bound = STRUCTURAL_BOUND.get(metric, DEFAULT_STRUCTURAL_BOUND)
        assert abs(measured) < bound, (
            f"{case}.{metric} is structurally zero in the baseline "
            f"({expected:.3e}) but measured {measured:.3e}, above the {bound:.0e} "
            "bound -- something stopped being exactly conserved."
        )
        return
    assert measured == pytest.approx(expected, rel=DEFAULT_RTOL), (
        f"{case}.{metric} drifted: baseline {expected!r}, measured {measured!r} "
        f"(relative change {abs(measured - expected) / abs(expected):.2e}, "
        f"tolerance {DEFAULT_RTOL:.0e}).\n"
        "If this change is intended, re-record with tools/record_baselines.py "
        "and commit the new baselines.json alongside the change that caused it."
    )


def compare_all(case: str, entry: dict, measured: dict) -> None:
    """Check every metric a baseline entry records."""
    for metric, expected in entry["metrics"].items():
        assert metric in measured, f"{case}: metric {metric!r} is no longer reported"
        check(case, metric, measured[metric], expected)


# --------------------------------------------------------------------------- #
# The baseline file itself
# --------------------------------------------------------------------------- #
def test_baseline_file_is_self_describing():
    """A baseline is only trustworthy if you can tell what recorded it."""
    meta = BASELINES["_meta"]
    for key in ("recorded_utc", "pycfd_version", "platform", "python", "purpose"):
        assert meta.get(key), f"baselines.json is missing {key!r}"
    assert BASELINES["fast"], "no fast baselines recorded"


def test_every_baseline_records_the_parameters_that_produced_it():
    for case, entry in BASELINES["fast"].items():
        assert entry.get("params"), f"{case} records no parameters"
        assert entry.get("metrics"), f"{case} records no metrics"


# --------------------------------------------------------------------------- #
# Fast tier -- runs on every push
# --------------------------------------------------------------------------- #
def test_taylor_green_convergence_matches_baseline():
    """Second-order convergence, pinned to the digit rather than a floor."""
    entry = BASELINES["fast"]["taylor_green_convergence"]
    tg = load_case("taylor_green")
    l2, linf, _ = tg.convergence_study(
        tuple(entry["params"]["resolutions"]), progress=False)
    compare_all("taylor_green_convergence", entry, {
        "observed_order_L2": l2.observed_order,
        "observed_order_Linf": linf.observed_order,
        "finest_L2": l2.errors[-1],
        "finest_Linf": linf.errors[-1],
    })


@pytest.mark.parametrize("case", ["channel_periodic", "channel_developing"])
def test_channel_matches_baseline(case):
    entry = BASELINES["fast"][case]
    p = entry["params"]
    result = load_case("channel").run(
        re=p["re"], nx=p["nx"], ny=p["ny"], mode=p["mode"], make_plots=False)
    compare_all(case, entry, result.metrics)


def test_cavity_matches_baseline():
    entry = BASELINES["fast"]["cavity_re100"]
    p = entry["params"]
    result = load_case("cavity").run(
        re=p["re"], nx=p["nx"], ny=p["ny"], make_plots=False)
    compare_all("cavity_re100", entry, result.metrics)


def test_cylinder_matches_baseline():
    entry = BASELINES["fast"]["cylinder_re20"]
    p = entry["params"]
    result = load_case("cylinder").run(
        re=p["re"], nx=p["nx"], ny=p["ny"], t_end=p["t_end"], make_plots=False)
    compare_all("cylinder_re20", entry, result.metrics)


# --------------------------------------------------------------------------- #
# Full tier -- the numbers the README quotes; needs --runslow
# --------------------------------------------------------------------------- #
def full(case: str) -> dict:
    """Fetch a full-fidelity baseline, skipping if it has not been recorded."""
    entry = BASELINES.get("full", {}).get(case)
    if entry is None:
        pytest.skip(f"no full-fidelity baseline recorded for {case}")
    return entry


@pytest.mark.slow
@pytest.mark.parametrize("case,mode", [
    ("channel_periodic_full", "periodic"),
    ("channel_developing_full", "developing"),
])
def test_channel_full_fidelity(case, mode):
    entry = full(case)
    p = entry["params"]
    result = load_case("channel").run(
        re=p["re"], nx=p["nx"], ny=p["ny"], mode=mode, make_plots=False)
    compare_all(case, entry, result.metrics)


@pytest.mark.slow
@pytest.mark.parametrize("re", [100, 400, 1000])
def test_cavity_full_fidelity_against_ghia(re):
    """The published cavity agreement, at the resolution the README reports."""
    case = f"cavity_re{re}_full"
    entry = full(case)
    p = entry["params"]
    result = load_case("cavity").run(
        re=p["re"], nx=p["nx"], ny=p["ny"], make_plots=False)
    compare_all(case, entry, result.metrics)


@pytest.mark.slow
def test_cylinder_full_fidelity():
    entry = full("cylinder_re20_full")
    p = entry["params"]
    result = load_case("cylinder").run(
        re=p["re"], nx=p["nx"], ny=p["ny"], make_plots=False)
    compare_all("cylinder_re20_full", entry, result.metrics)
