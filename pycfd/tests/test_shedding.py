"""Telling a vortex street from everything else that has a strongest frequency.

Every signal has a largest spectral bin, so a peak-picker never says "no". The
tests that matter here are therefore the negative ones: noise, a drift, a decay
and a record too short to have measured anything must all be *refused*, and
each is one that `strouhal_number` answers with a confident number.
"""

import math

import numpy as np
import pytest

from pycfd.analysis.postprocess import strouhal_number
from pycfd.analysis.shedding import (
    MIN_CONCENTRATION,
    MIN_PERIODICITY,
    MIN_PERIODS,
    MIN_SAMPLES,
    detect_shedding,
)

FREQUENCY = 0.2


def record(signal_of, n: int = 2_000, t_end: float = 100.0):
    t = np.linspace(0.0, t_end, n)
    return t, signal_of(t)


def sine(amplitude: float = 1.0, frequency: float = FREQUENCY, noise: float = 0.0,
         seed: int = 0):
    rng = np.random.default_rng(seed)
    def build(t):
        x = amplitude * np.sin(2 * np.pi * frequency * t)
        return x + noise * rng.standard_normal(t.size) if noise else x
    return build


def detect(t, x, **kw):
    """Detect over the whole record unless a test says otherwise."""
    kw.setdefault("transient_fraction", 0.0)
    return detect_shedding(t, x, 1.0, 1.0, **kw)


# --------------------------------------------------------------------------- #
# What shedding looks like
# --------------------------------------------------------------------------- #
def test_a_clean_oscillation_is_detected_at_its_own_frequency():
    result = detect(*record(sine()))

    assert result.detected
    assert result.strouhal == pytest.approx(FREQUENCY, rel=1e-3)
    assert result.periodicity == pytest.approx(1.0, abs=0.02)
    assert result.concentration > 0.9
    assert result.periods_observed == pytest.approx(20.0, rel=1e-2)


def test_the_strouhal_number_uses_the_scales_it_is_given():
    t, x = record(sine())
    doubled = detect_shedding(t, x, l_ref=2.0, u_ref=1.0, transient_fraction=0.0)
    halved = detect_shedding(t, x, l_ref=1.0, u_ref=2.0, transient_fraction=0.0)

    assert doubled.strouhal == pytest.approx(2.0 * FREQUENCY, rel=1e-3)
    assert halved.strouhal == pytest.approx(0.5 * FREQUENCY, rel=1e-3)


def test_moderate_noise_does_not_hide_a_real_oscillation():
    result = detect(*record(sine(noise=0.2)))
    assert result.detected
    assert result.strouhal == pytest.approx(FREQUENCY, rel=1e-2)


def test_amplitude_does_not_decide_detection():
    """A quiet clean tone is shedding; a loud mess is not."""
    quiet = detect(*record(sine(amplitude=1e-3)))
    assert quiet.detected


# --------------------------------------------------------------------------- #
# What it must refuse — each one a case a peak-picker answers confidently
# --------------------------------------------------------------------------- #
def test_white_noise_is_not_shedding():
    rng = np.random.default_rng(1)
    t, x = record(lambda tt: rng.standard_normal(tt.size))
    result = detect(t, x)

    assert not result.detected
    assert result.periodicity < MIN_PERIODICITY
    # The point of the exercise: the naive route answers anyway.
    assert math.isfinite(strouhal_number(t, x, 1.0, 1.0))


def test_a_linear_drift_is_not_shedding():
    t, x = record(lambda tt: 0.01 * tt)
    result = detect(t, x)

    assert not result.detected
    assert math.isfinite(strouhal_number(t, x, 1.0, 1.0))


def test_a_decaying_transient_is_not_shedding():
    t, x = record(lambda tt: 2.0 * np.exp(-tt / 20.0))
    assert not detect(t, x).detected


def test_a_steady_wake_is_not_shedding():
    t, x = record(lambda tt: np.full(tt.size, 0.7))
    result = detect(t, x)

    assert not result.detected
    assert "steady" in result.reason


def test_a_signal_buried_in_noise_is_refused():
    """Visible as a peak, but it does not repeat."""
    result = detect(*record(sine(amplitude=1.0, noise=2.0)))
    assert not result.detected
    assert result.periodicity < MIN_PERIODICITY


def test_too_few_periods_is_refused_however_clean():
    """A frequency inferred from two cycles is not a measurement."""
    result = detect(*record(sine(frequency=0.02)))       # 2 periods in 100 s

    assert not result.detected
    assert result.periods_observed < MIN_PERIODS
    assert result.periodicity == pytest.approx(1.0, abs=0.05)   # clean, but short
    assert "run longer" in result.reason


def test_too_few_samples_is_refused():
    t, x = record(sine(), n=MIN_SAMPLES - 1)
    result = detect(t, x)

    assert not result.detected
    assert "samples" in result.reason


def test_an_amplitude_floor_dismisses_a_negligible_wobble():
    t, x = record(sine(amplitude=1e-4))
    assert detect(t, x).detected                       # clean, so detected...
    result = detect(t, x, min_amplitude=0.02)          # ...but too small to care
    assert not result.detected
    assert "peak-to-peak" in result.reason


def test_an_empty_record_is_refused_rather_than_raising():
    result = detect_shedding([], [])
    assert not result.detected
    assert "empty" in result.reason


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError, match="same length"):
        detect_shedding([1.0, 2.0], [1.0])


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_a_detection_reports_its_evidence():
    text = detect(*record(sine())).report()
    assert "St =" in text and "periods observed" in text
    assert "periodicity" in text and "concentration" in text


def test_a_refusal_reports_what_failed():
    rng = np.random.default_rng(2)
    text = detect(*record(lambda tt: rng.standard_normal(tt.size))).report()
    assert "no periodic shedding detected" in text
    assert "does not repeat" in text


def test_the_transient_window_is_honoured():
    """Shedding that only starts halfway through is found by discarding the rest."""
    t = np.linspace(0.0, 200.0, 4_000)
    x = np.where(t < 100.0, 0.0, np.sin(2 * np.pi * FREQUENCY * t))
    # Over the whole record the flat first half dilutes everything.
    assert detect_shedding(t, x, 1.0, 1.0, transient_fraction=0.5).detected


# --------------------------------------------------------------------------- #
# Through the cylinder case
# --------------------------------------------------------------------------- #
def test_a_short_run_refuses_to_report_a_strouhal_number():
    """The failure the F-22 run hit: a confident frequency from no shedding."""
    from pycfd.cases import load_case

    result = load_case("cylinder").run(re=100, nx=64, ny=32, t_end=6.0,
                                       make_plots=False)
    verdicts = {label: ok for label, ok, _ in result.checks}
    assert verdicts["vortex shedding developed"] is False
    assert result.metrics["shedding_detected"] == 0.0


def test_the_case_reports_the_evidence_alongside_the_verdict():
    from pycfd.cases import load_case

    result = load_case("cylinder").run(re=100, nx=64, ny=32, t_end=6.0,
                                       make_plots=False)
    for key in ("shedding_detected", "shedding_periodicity",
                "shedding_concentration", "shedding_periods"):
        assert key in result.metrics


def test_the_thresholds_are_ordered_sensibly():
    """Guard rails on the constants themselves."""
    assert 0.0 < MIN_CONCENTRATION < 1.0
    assert 0.0 < MIN_PERIODICITY < 1.0
    assert MIN_PERIODS >= 2.0
    assert MIN_SAMPLES >= 16
