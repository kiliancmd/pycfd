"""Richardson extrapolation, the GCI, and the refusal that matters most.

Every assertion here is against a sequence whose answer is known by
construction: build ``f(h) = f_exact + C h^p``, hand the helper three grids,
and require it to recover ``p`` and ``f_exact``. A helper that merely returns
plausible-looking floats would pass a test written the other way round.

The non-uniform-refinement cases are the load-bearing ones. With doubling grids
the order equation collapses to a single logarithm and a swapped refinement
ratio cancels itself out; only an uneven ladder tells the two conventions apart.
"""

import math

import pytest

from pycfd.analysis.richardson import (
    MAX_PLAUSIBLE_ORDER,
    MIN_PLAUSIBLE_ORDER,
    richardson,
)


def sequence(resolutions, exact, coefficient, order):
    """Values of ``exact + coefficient * h**order`` on the given grids."""
    return [exact + coefficient * (1.0 / n) ** order for n in resolutions]


# --------------------------------------------------------------------------- #
# Recovering a known answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("resolutions", [
    (32, 64, 128),          # doubling
    (40, 64, 128),          # uneven, then doubling
    (30, 50, 100),          # uneven throughout
    (16, 48, 96),           # a big first jump
])
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_a_known_sequence_is_recovered_exactly(resolutions, order):
    exact, coefficient = 1.37, 42.0
    estimate = richardson(resolutions, sequence(resolutions, exact, coefficient, order))

    assert estimate.observed_order == pytest.approx(order, rel=1e-6)
    assert estimate.extrapolated == pytest.approx(exact, rel=1e-6)
    assert estimate.regime == "monotonic"
    assert estimate.trustworthy


def test_a_negative_limit_is_recovered_too():
    resolutions = (30, 50, 100)
    estimate = richardson(resolutions, sequence(resolutions, -2.0, 7.0, 1.0))
    assert estimate.extrapolated == pytest.approx(-2.0, rel=1e-6)


def test_the_finest_three_grids_are_the_ones_used():
    """More grids are welcome; the coarsest are furthest from the limit."""
    resolutions = (8, 16, 32, 64, 128)
    estimate = richardson(resolutions, sequence(resolutions, 5.0, 100.0, 2.0))
    assert estimate.resolutions == (32, 64, 128)


def test_the_extrapolated_value_beats_the_finest_grid():
    resolutions = (32, 64, 128)
    values = sequence(resolutions, 5.0, 100.0, 2.0)
    estimate = richardson(resolutions, values)
    assert abs(estimate.extrapolated - 5.0) < abs(values[-1] - 5.0)


# --------------------------------------------------------------------------- #
# The refusal
# --------------------------------------------------------------------------- #
def test_a_diverging_sequence_is_refused_rather_than_extrapolated():
    """The whole reason this module exists."""
    estimate = richardson([32, 64, 128], [1.0, 1.5, 2.6])

    assert estimate.regime == "diverging"
    assert not estimate.trustworthy
    assert math.isnan(estimate.extrapolated)
    assert math.isnan(estimate.gci)
    assert "not converging" in estimate.reason


def test_a_refusal_still_reports_without_raising():
    """A refusal is a result: printing it must not be a special case."""
    text = richardson([32, 64, 128], [1.0, 1.5, 2.6]).report()
    assert "NOT EXTRAPOLATED" in text
    assert "diverging" in text


def test_an_oscillating_sequence_converges_but_is_flagged():
    """It approaches a limit from alternating sides -- usable, with a caveat."""
    estimate = richardson([32, 64, 128], [1.0, 1.4, 1.2])

    assert estimate.regime == "oscillatory"
    assert estimate.convergence_ratio < 0
    assert math.isfinite(estimate.extrapolated)
    assert "oscillates" in estimate.reason


def test_an_implausible_order_is_not_believed():
    """Order 7 is not a triumph; it is two differences made mostly of noise."""
    resolutions = (32, 64, 128)
    estimate = richardson(resolutions, sequence(resolutions, 1.0, 1e6, 8.0))

    assert estimate.observed_order > MAX_PLAUSIBLE_ORDER
    assert not estimate.trustworthy
    assert "asymptotic regime" in estimate.reason


def test_a_suspiciously_low_order_is_not_believed_either():
    resolutions = (32, 64, 128)
    estimate = richardson(resolutions, sequence(resolutions, 1.0, 0.5, 0.2))

    assert estimate.observed_order < MIN_PLAUSIBLE_ORDER
    assert not estimate.trustworthy


def test_grids_that_already_agree_are_converged_not_broken():
    """Dividing by a difference of zero must not be how this ends."""
    estimate = richardson([32, 64, 128], [2.5, 2.5, 2.5])

    assert estimate.regime == "converged"
    assert estimate.trustworthy
    assert estimate.extrapolated == pytest.approx(2.5)
    assert estimate.gci == 0.0


@pytest.mark.parametrize("resolutions,values,message", [
    ([32, 64], [1.0, 2.0], "needs three grids"),
    ([32, 64, 128], [1.0, 2.0], "same length"),
    ([128, 64, 32], [1.0, 2.0, 3.0], "must increase"),
])
def test_an_unusable_call_is_an_error_not_a_nan(resolutions, values, message):
    with pytest.raises(ValueError, match=message):
        richardson(resolutions, values)


# --------------------------------------------------------------------------- #
# The uncertainty band
# --------------------------------------------------------------------------- #
def test_the_gci_band_brackets_the_extrapolated_value():
    resolutions = (32, 64, 128)
    estimate = richardson(resolutions, sequence(resolutions, 5.0, 100.0, 2.0))
    lo, hi = estimate.band()

    assert lo <= estimate.extrapolated <= hi
    assert lo < estimate.values[-1] < hi


def test_a_finer_sequence_earns_a_tighter_band():
    """Refinement has to buy something, and this is what it buys."""
    coarse = richardson((16, 32, 64), sequence((16, 32, 64), 5.0, 100.0, 2.0))
    fine = richardson((64, 128, 256), sequence((64, 128, 256), 5.0, 100.0, 2.0))
    assert fine.gci < coarse.gci


def test_the_safety_factor_scales_the_band():
    resolutions = (32, 64, 128)
    values = sequence(resolutions, 5.0, 100.0, 2.0)
    assert richardson(resolutions, values, safety_factor=3.0).gci == pytest.approx(
        richardson(resolutions, values, safety_factor=1.5).gci * 2.0)


def test_relative_error_is_the_distance_from_the_limit():
    resolutions = (32, 64, 128)
    values = sequence(resolutions, 5.0, 100.0, 2.0)
    estimate = richardson(resolutions, values)
    assert estimate.relative_error == pytest.approx(
        abs(values[-1] - estimate.extrapolated) / abs(estimate.extrapolated))


# --------------------------------------------------------------------------- #
# Through a grid study
# --------------------------------------------------------------------------- #
def test_a_grid_study_extrapolates_each_metric():
    from pycfd.cases import GridStudy

    resolutions = [32, 64, 128]
    study = GridStudy(
        "test", resolutions,
        {"cd": sequence(resolutions, 1.32, 20.0, 2.0)},
        {"cd": "quantity"},
    )
    estimate = study.richardson("cd")
    assert estimate.extrapolated == pytest.approx(1.32, rel=1e-6)
    assert study.extrapolated() == {"cd": pytest.approx(1.32, rel=1e-6)}


def test_two_grids_cannot_be_extrapolated_from():
    from pycfd.cases import GridStudy

    study = GridStudy("test", [64, 128], {"cd": [1.0, 1.1]}, {"cd": "quantity"})
    assert study.richardson("cd") is None
    assert study.extrapolated() == {}
    assert "going anywhere" in study.report()


def test_an_untrustworthy_estimate_is_left_out_of_the_summary():
    """extrapolated() is for numbers worth quoting, so a refusal is excluded."""
    from pycfd.cases import GridStudy

    study = GridStudy("test", [32, 64, 128], {"cd": [1.0, 1.5, 2.6]},
                      {"cd": "quantity"})
    assert not study.richardson("cd").trustworthy
    assert study.extrapolated() == {}


def test_the_study_table_carries_the_richardson_verdict():
    from pycfd.cases import GridStudy

    resolutions = [32, 64, 128]
    study = GridStudy("test", resolutions,
                      {"cd": sequence(resolutions, 1.32, 20.0, 2.0)},
                      {"cd": "quantity"})
    text = study.table()
    assert "order from successive differences" in text
    assert "GCI" in text
