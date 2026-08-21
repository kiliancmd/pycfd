"""Averaging a time series that is still deciding what it is.

A force coefficient measured on a shedding wake is not a number, it is a
distribution sampled over time, and reducing it to a mean raises two questions
the mean itself cannot answer: has the transient actually finished, and how
much is the answer allowed to move if the run continued?

Why the naive error bar is wrong
--------------------------------
The textbook standard error ``s / sqrt(N)`` assumes independent samples.  A
solver writing out ``Cd`` every ten steps produces nothing of the kind: at a
Strouhal number near 0.2 a shedding period spans hundreds of samples, and
consecutive ones are almost the same measurement.  Using ``sqrt(N)`` there
reports an uncertainty that shrinks with the *sampling rate* rather than with
the amount of independent physics observed -- sample twice as often and the
error bar halves, which is obviously false.

The fix is the integrated autocorrelation time::

    tau = 1 + 2 * sum_k rho_k          rho_k = autocorrelation at lag k
    N_eff = N / tau
    standard error = s / sqrt(N_eff)

``tau`` is roughly how many samples a shedding cycle occupies, so ``N_eff``
counts independent *periods* rather than writes.  The sum is truncated at the
first non-positive ``rho_k`` (Geyer's initial-positive-sequence rule): past
that point the estimates are dominated by their own noise, and summing further
adds variance rather than information.

Has the transient finished?
---------------------------
Separately, the retained record is split in half and the two means are compared
against their own combined uncertainty.  A record still drifting down from its
start-up transient has a first half measurably above its second, and that shows
up as a large z-score even when the series looks settled by eye.  It is a
cheaper cousin of the Geweke diagnostic and answers the question that matters:
was the discarded window long enough?

Where this is blunt
-------------------
A record *dominated* by its own trend defeats the z-score, because the trend is
itself read as correlation: ``tau`` inflates, the error bar widens with it, and
the drift stops looking significant against a band it helped create.  The
reported interval stays honest -- it still covers the value the run was heading
for -- but the stationarity flag goes quiet exactly when the transient is worst.
What gives it away instead is ``tau``: a record whose transient is still in it
reports an autocorrelation time in the hundreds and a handful of effective
samples, where a settled one reports ``tau`` near 1.  Read the two together.
Detrending before estimating ``tau`` would sharpen this, and is a larger piece
of machinery than the question has so far justified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Fraction of the record discarded as start-up transient unless asked otherwise.
DEFAULT_TRANSIENT_FRACTION = 0.5

#: |z| above which the two halves of the retained record are called
#: inconsistent.  Two standard errors is the usual two-sided 95% convention.
STATIONARITY_Z = 2.0

#: Relative drift below which a record counts as settled whatever its z-score.
#:
#: Statistical significance is not the same as mattering.  A smoothly
#: converging steady wake has almost no scatter left, so its standard error
#: collapses and *any* residual monotone drift -- a decaying exponential's tail,
#: parts per million of the mean -- registers as many standard errors.  Without
#: this floor the diagnostic cries wolf on exactly the runs that converged best,
#: which is how a diagnostic gets ignored.  One part in a thousand is below the
#: precision at which a drag coefficient is ever quoted.
PRACTICAL_DRIFT_RTOL = 1.0e-3

#: Multiplier turning a standard error into a reported band.  1.96 is the
#: two-sided 95% normal quantile; the effective-sample-size machinery above is
#: what makes a normal quantile defensible on a correlated series at all.
CONFIDENCE_Z = 1.96

#: A series whose spread is this small relative to its mean is constant to
#: round-off -- a structurally zero quantity, not a fluctuating one.  Its
#: autocorrelation is 0/0 and must not be computed.
FLAT_SERIES_RTOL = 1.0e-12


def autocorrelation_time(values) -> float:
    """Integrated autocorrelation time ``tau`` of a sampled series.

    Returns the number of samples that one *independent* observation is worth.
    ``1.0`` means the samples already are independent; a shedding wake sampled
    every ten steps typically returns tens to hundreds.

    The lag sum stops at the first non-positive autocorrelation, which is the
    standard truncation: beyond it the individual estimates are mostly noise
    and including them inflates ``tau`` without adding information.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 2:
        return float("nan")

    centred = x - x.mean()
    variance = float(np.mean(centred ** 2))
    if variance <= 0:
        # A constant series: every sample is the same measurement, but it is
        # also a measurement with no scatter, so treating them as independent
        # is harmless and keeps the error bar at zero rather than undefined.
        return 1.0

    tau = 1.0
    # Half the record is the usual cap: lags beyond it average over so few
    # pairs that the estimate is meaningless.
    for lag in range(1, max(1, n // 2)):
        rho = float(np.mean(centred[:-lag] * centred[lag:])) / variance
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return tau


@dataclass(frozen=True)
class TimeAverage:
    """A mean, what it is worth, and whether it was taken too early.

    Attributes
    ----------
    mean, std:
        Sample mean and standard deviation over the retained record.
    n_samples:
        How many samples were retained after discarding the transient.
    autocorrelation_time:
        ``tau`` -- samples per independent observation.
    effective_samples:
        ``N / tau``.  The number the error bar is actually built from, and
        usually far smaller than ``n_samples``.
    standard_error:
        ``std / sqrt(effective_samples)``.
    drift_z:
        Difference between the first and second half of the retained record,
        in units of its own standard error.
    stationary:
        Whether ``|drift_z|`` is within :data:`STATIONARITY_Z`.
    """

    mean: float
    std: float
    n_samples: int
    autocorrelation_time: float
    effective_samples: float
    standard_error: float
    drift_z: float
    stationary: bool
    reason: str = ""

    @property
    def uncertainty(self) -> float:
        """Half-width of the reported confidence band."""
        return CONFIDENCE_Z * self.standard_error

    def band(self) -> tuple[float, float]:
        """The interval the converged mean is expected to lie in."""
        return (self.mean - self.uncertainty, self.mean + self.uncertainty)

    @property
    def relative_uncertainty(self) -> float:
        """Band half-width as a fraction of the mean."""
        return abs(self.uncertainty / self.mean) if self.mean else float("nan")

    def report(self) -> str:
        """One block, safe to print whatever the record turned out to be."""
        if not math.isfinite(self.mean):
            return "    no samples retained"
        lines = [
            f"    mean {self.mean:.6g} ± {self.uncertainty:.3g}  "
            f"(95%, {self.n_samples} samples -> {self.effective_samples:.1f} "
            f"independent, tau = {self.autocorrelation_time:.1f})"
        ]
        if not self.stationary:
            lines.append(
                f"    NOT STATIONARY: the two halves of the averaging window "
                f"disagree by {self.drift_z:.1f} standard errors — {self.reason}"
            )
        return "\n".join(lines)


def _half_split(values: np.ndarray) -> tuple[float, float]:
    """``(z, drift)`` between the two halves of a record.

    ``drift`` is the raw difference of the means; ``z`` expresses it in units of
    its own combined standard error.  Both are needed: one says whether the
    difference is real, the other whether it is worth anything.
    """
    n = values.size
    if n < 4:
        return float("nan"), float("nan")

    first, second = values[: n // 2], values[n // 2:]
    errors = []
    for half in (first, second):
        tau = autocorrelation_time(half)
        n_eff = max(1.0, half.size / tau) if math.isfinite(tau) and tau > 0 else 1.0
        errors.append(float(half.std()) / math.sqrt(n_eff))

    combined = math.hypot(*errors)
    drift = float(second.mean() - first.mean())
    if combined <= 0:
        # Both halves are constant. Identical means agree perfectly; different
        # ones disagree infinitely, and neither is a division to attempt.
        return (0.0 if drift == 0 else float("inf")), drift
    return drift / combined, drift


def time_average(times, values,
                 transient_fraction: float = DEFAULT_TRANSIENT_FRACTION) -> TimeAverage:
    """Average ``values`` over the settled part of the record.

    Parameters
    ----------
    times, values:
        Equal-length sample times and measurements.
    transient_fraction:
        Fraction of the *elapsed time* discarded before averaging.  ``0.5``
        keeps the second half, which is the long-standing default; raise it
        when a run started far from its eventual state.

    Returns
    -------
    TimeAverage
        Always returned.  A record too short to say anything comes back with
        ``nan`` fields rather than raising, since a diagnostic that throws is
        one people stop asking for.
    """
    if not 0.0 <= transient_fraction < 1.0:
        raise ValueError(
            f"transient_fraction must lie in [0, 1), got {transient_fraction}"
        )
    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    if t.size != x.size:
        raise ValueError(
            f"times and values must have the same length, got {t.size} and {x.size}"
        )

    empty = TimeAverage(float("nan"), float("nan"), 0, float("nan"),
                        float("nan"), float("nan"), float("nan"), False,
                        "no samples retained")
    if t.size == 0:
        return empty

    # Kept identical to the predicate this replaced, so a default run reports
    # exactly the mean it always did.
    retained = x[t > transient_fraction * t[-1]]
    if retained.size == 0:
        return empty

    mean = float(retained.mean())
    std = float(retained.std())
    tau = autocorrelation_time(retained)
    n_eff = retained.size / tau if math.isfinite(tau) and tau > 0 else float("nan")

    # A series flat to round-off has no scatter to propagate: the mean is what
    # it is, and an autocorrelation-based error bar on it would be noise about
    # noise. Report zero uncertainty and call it settled.
    scale = max(abs(mean), 1.0)
    if std <= FLAT_SERIES_RTOL * scale:
        return TimeAverage(mean, std, retained.size, 1.0, float(retained.size),
                           0.0, 0.0, True, "constant to round-off")

    standard_error = std / math.sqrt(n_eff) if n_eff and n_eff > 0 else float("nan")
    drift_z, drift = _half_split(retained)

    # Non-stationary means the drift is both real and large enough to matter.
    # Either test alone gives the wrong answer: the z-score alone condemns a
    # beautifully converged run whose remaining drift is parts per million,
    # and the relative size alone would wave through a genuine trend hiding in
    # a noisy record.
    significant = math.isfinite(drift_z) and abs(drift_z) > STATIONARITY_Z
    material = abs(drift) > PRACTICAL_DRIFT_RTOL * scale
    stationary = not (significant and material)
    reason = "" if stationary else "the discarded transient was probably too short"
    if math.isnan(drift_z):
        stationary, reason = True, "too few samples to split"
    return TimeAverage(
        mean=mean, std=std, n_samples=int(retained.size),
        autocorrelation_time=tau, effective_samples=float(n_eff),
        standard_error=standard_error, drift_z=drift_z,
        stationary=stationary, reason=reason,
    )
