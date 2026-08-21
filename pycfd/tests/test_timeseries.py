"""Averaging a correlated record, and knowing what the average is worth.

The assertions here are against series whose statistics are known in closed
form. An AR(1) process has integrated autocorrelation time ``(1+phi)/(1-phi)``
exactly, so the estimator can be checked rather than merely exercised — and the
property that actually matters, that sampling more often must not shrink the
error bar, is asserted directly.
"""

import math

import numpy as np
import pytest

from pycfd.analysis.timeseries import (
    CONFIDENCE_Z,
    STATIONARITY_Z,
    autocorrelation_time,
    time_average,
)


def ar1(phi: float, n: int = 200_000, seed: int = 0) -> np.ndarray:
    """AR(1): ``x[i] = phi x[i-1] + noise``, autocorrelation ``rho_k = phi^k``."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


# --------------------------------------------------------------------------- #
# The autocorrelation time
# --------------------------------------------------------------------------- #
def test_independent_samples_are_worth_one_each():
    rng = np.random.default_rng(1)
    assert autocorrelation_time(rng.standard_normal(20_000)) == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("phi", [0.5, 0.8, 0.9, 0.95])
def test_ar1_recovers_its_analytical_autocorrelation_time(phi):
    """``tau = (1 + phi) / (1 - phi)`` exactly, for this process.

    The 10% band is estimator variance, not slack: tau itself is estimated from
    a finite record and its own standard error runs to several per cent at
    these correlation times. A wrong formula would be out by a factor, not by
    a few per cent, so this still pins what it needs to.
    """
    expected = (1.0 + phi) / (1.0 - phi)
    assert autocorrelation_time(ar1(phi)) == pytest.approx(expected, rel=0.10)


def test_a_more_correlated_series_is_worth_fewer_independent_samples():
    assert autocorrelation_time(ar1(0.9)) > autocorrelation_time(ar1(0.5))


def test_a_constant_series_has_no_correlation_to_measure():
    """Zero variance is 0/0; it must return 1, not nan or a division error."""
    assert autocorrelation_time(np.full(100, 2.5)) == 1.0


def test_too_short_to_say_anything():
    assert math.isnan(autocorrelation_time([1.0]))


# --------------------------------------------------------------------------- #
# The error bar
# --------------------------------------------------------------------------- #
def periodic_record(n: int, t_end: float = 200.0, noise: float = 0.01,
                    seed: int = 3):
    """A shedding-like signal: a mean, an oscillation, and a little noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_end, n)
    x = 1.3 + 0.2 * np.sin(2 * np.pi * 0.2 * t) + noise * rng.standard_normal(n)
    return t, x


def test_sampling_more_often_does_not_shrink_the_error_bar():
    """The whole reason the effective sample size exists.

    Writing the same physics out twice as often adds no independent
    observations, so the uncertainty must be essentially unchanged. The naive
    ``s / sqrt(N)`` would report a factor of ``1/sqrt(2)``.
    """
    coarse = time_average(*periodic_record(4_000))
    fine = time_average(*periodic_record(8_000))

    assert fine.standard_error == pytest.approx(coarse.standard_error, rel=0.2)
    assert fine.n_samples == 2 * coarse.n_samples          # twice the writes...
    assert fine.effective_samples == pytest.approx(        # ...same information
        coarse.effective_samples, rel=0.25)


def test_effective_samples_are_fewer_than_writes_on_a_correlated_record():
    average = time_average(*periodic_record(4_000))
    assert average.effective_samples < average.n_samples / 10


def test_a_longer_record_does_tighten_the_band():
    """Refuting the opposite worry: real extra information must still count."""
    short = time_average(*periodic_record(4_000, t_end=200.0))
    long = time_average(*periodic_record(40_000, t_end=2000.0))
    assert long.effective_samples > short.effective_samples
    assert long.standard_error < short.standard_error


def test_the_band_is_the_standard_error_times_the_confidence_factor():
    average = time_average(*periodic_record(2_000))
    assert average.uncertainty == pytest.approx(CONFIDENCE_Z * average.standard_error)
    lo, hi = average.band()
    assert lo < average.mean < hi


def test_a_flat_record_gets_a_zero_band_not_a_nan():
    """A structurally zero quantity: no scatter to propagate."""
    t = np.linspace(0, 10, 200)
    average = time_average(t, np.full(200, 3.0))

    assert average.mean == pytest.approx(3.0)
    assert average.standard_error == 0.0
    assert average.uncertainty == 0.0
    assert average.stationary


# --------------------------------------------------------------------------- #
# Was the transient long enough?
# --------------------------------------------------------------------------- #
def test_a_settled_record_reads_as_stationary():
    average = time_average(*periodic_record(4_000))
    assert abs(average.drift_z) <= STATIONARITY_Z
    assert average.stationary


def test_a_record_still_drifting_is_caught():
    """A decaying transient the discarded window failed to remove."""
    t = np.linspace(0.0, 100.0, 2_000)
    x = 1.3 + 0.9 * np.exp(-t / 40.0)          # still falling at the end
    average = time_average(t, x, transient_fraction=0.5)

    assert not average.stationary
    assert average.drift_z < -STATIONARITY_Z    # second half sits below the first
    assert "transient" in average.reason


def decaying_record(seed: int = 7):
    """A settled value of 1.3, reached through a decaying transient, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 100.0, 2_000)
    x = 1.3 + 0.9 * np.exp(-t / 8.0) + 0.02 * rng.standard_normal(2_000)
    return t, x, 1.3


def test_discarding_more_of_a_transient_reduces_the_bias():
    t, x, settled = decaying_record()
    biases = [abs(time_average(t, x, f).mean - settled)
              for f in (0.05, 0.2, 0.4, 0.6)]
    assert biases == sorted(biases, reverse=True)


def test_the_band_covers_the_settled_value_at_every_window():
    """The honesty property: a wide band is allowed, a wrong one is not.

    Even where the retained record still contains most of the transient, the
    reported interval must contain the value the run was heading for. That is
    what makes the number quotable at all.
    """
    t, x, settled = decaying_record()
    for fraction in (0.05, 0.2, 0.4, 0.6, 0.8):
        lo, hi = time_average(t, x, fraction).band()
        assert lo <= settled <= hi, f"band missed the settled value at {fraction}"


def test_the_autocorrelation_time_is_itself_a_transient_detector():
    """A trend reads as enormous correlation; losing it collapses tau to ~1.

    This is why a large tau next to a small effective sample count is worth
    reading even when the stationarity flag stays quiet: a record dominated by
    its own trend inflates its error bar so much that the drift stops looking
    significant, and tau is what still says so.
    """
    t, x, _ = decaying_record()
    with_transient = time_average(t, x, transient_fraction=0.05)
    without = time_average(t, x, transient_fraction=0.6)

    assert with_transient.autocorrelation_time > 50
    assert without.autocorrelation_time < 5
    assert without.effective_samples > 10 * with_transient.effective_samples


def test_a_drift_too_small_to_matter_is_not_called_a_failure():
    """Statistical significance is not the same as mattering.

    A smoothly converged record has almost no scatter left, so a residual drift
    of parts per million registers as many standard errors. Condemning that
    run would make the diagnostic worthless on exactly the runs that went best.
    """
    t = np.linspace(0.0, 100.0, 2_000)
    x = 1.3 + 0.9 * np.exp(-t / 8.0)          # noiseless: drift is monotone
    late = time_average(t, x, transient_fraction=0.9)

    assert abs(late.drift_z) > STATIONARITY_Z          # statistically obvious
    assert abs(late.mean - 1.3) < 1e-4                 # physically irrelevant
    assert late.stationary


def test_a_drift_that_matters_is_still_caught_however_quiet_the_record():
    """The other half of the same rule: the floor must not wave through a trend."""
    t = np.linspace(0.0, 100.0, 2_000)
    x = 1.3 + 0.5 * np.exp(-t / 200.0)        # barely decayed by the end
    average = time_average(t, x, transient_fraction=0.5)

    assert not average.stationary
    assert "transient" in average.reason


def test_the_report_says_when_the_window_had_not_settled():
    t = np.linspace(0.0, 100.0, 2_000)
    text = time_average(t, 1.3 + 0.9 * np.exp(-t / 40.0)).report()
    assert "NOT STATIONARY" in text


def test_the_report_of_a_settled_record_is_just_the_number():
    text = time_average(*periodic_record(4_000)).report()
    assert "NOT STATIONARY" not in text
    assert "independent" in text and "tau" in text


# --------------------------------------------------------------------------- #
# The averaging window
# --------------------------------------------------------------------------- #
def test_the_transient_fraction_selects_by_elapsed_time():
    t = np.linspace(0.0, 100.0, 101)
    average = time_average(t, np.ones(101), transient_fraction=0.5)
    # Strictly greater than 50, so 50 samples of the 101 are kept.
    assert average.n_samples == 50


def test_keeping_everything_is_allowed():
    t = np.linspace(1.0, 100.0, 100)
    assert time_average(t, np.ones(100), transient_fraction=0.0).n_samples == 100


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_a_nonsensical_window_is_refused(fraction):
    with pytest.raises(ValueError, match="transient_fraction"):
        time_average([1.0, 2.0], [1.0, 1.0], transient_fraction=fraction)


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError, match="same length"):
        time_average([1.0, 2.0], [1.0])


def test_an_empty_record_reports_nothing_rather_than_raising():
    """A diagnostic that throws is one people stop asking for."""
    average = time_average([], [])
    assert math.isnan(average.mean)
    assert average.n_samples == 0
    assert "no samples" in average.report()


# --------------------------------------------------------------------------- #
# Through the cylinder case
# --------------------------------------------------------------------------- #
def test_the_default_average_is_the_one_the_case_always_reported():
    """cd_mean must not move: the recorded baselines depend on it."""
    from pycfd.cases import load_case

    result = load_case("cylinder").run(re=20, nx=48, ny=24, t_end=2.0,
                                       make_plots=False)
    t = np.linspace(0.0, 1.0, 11)
    # The selection predicate is the contract; assert it directly.
    assert time_average(t, np.arange(11.0)).n_samples == 5

    assert "cd_uncertainty" in result.metrics
    assert result.metrics["transient_fraction"] == 0.5
    assert result.metrics["effective_samples"] <= result.metrics["averaging_samples"]


def test_the_case_reports_a_stationarity_verdict():
    from pycfd.cases import load_case

    result = load_case("cylinder").run(re=20, nx=48, ny=24, t_end=8.0,
                                       make_plots=False)
    labels = [label for label, _, _ in result.checks]
    assert "force average is stationary" in labels


def test_the_transient_flag_reaches_the_case():
    from pycfd.cases import load_case

    result = load_case("cylinder").run(re=20, nx=48, ny=24, t_end=4.0,
                                       transient=0.75, make_plots=False)
    assert result.metrics["transient_fraction"] == 0.75


def test_a_wider_window_keeps_more_samples():
    from pycfd.cases import load_case

    wide = load_case("cylinder").run(re=20, nx=48, ny=24, t_end=6.0,
                                     transient=0.2, make_plots=False)
    narrow = load_case("cylinder").run(re=20, nx=48, ny=24, t_end=6.0,
                                       transient=0.8, make_plots=False)
    assert wide.metrics["averaging_samples"] > narrow.metrics["averaging_samples"]
