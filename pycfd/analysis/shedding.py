"""Deciding whether a wake is actually shedding.

:func:`~pycfd.analysis.postprocess.strouhal_number` answers "where is the
largest peak in the spectrum?", and it answers it for any signal at all --
white noise, a linear drift, a steady wake wobbling at round-off.  The number
it returns looks identical in every case.  Deciding whether that frequency
*means* anything was, until this module, a judgement made by eye or by a
throwaway probe script written once per case.

What separates shedding from everything else
--------------------------------------------
Three questions, none of which the peak frequency alone can answer:

*Is the signal periodic at all?*  A genuinely periodic signal repeats: its
autocorrelation at a lag of exactly one period comes back near ``+1``.  Noise
returns ~0, and a monotone drift returns a value that decays smoothly with lag
rather than recovering at the period.  This is the discriminator that a
peak-picker lacks, and it is independent of the spectrum that proposed the
period in the first place.

*Is the power concentrated?*  A narrowband oscillation puts most of its energy
in a few spectral bins.  Broadband noise spreads it across all of them, and
still has a largest bin.

*Has enough of it been seen?*  A frequency inferred from one and a half cycles
is not a measurement.  The record has to span several periods before the
question is even well posed -- which is precisely the trap the F-22 run fell
into, where a wake that had not yet begun shedding still produced a confident
Strouhal number.

All three have to hold.  Any one of them alone is easy to fool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .timeseries import DEFAULT_TRANSIENT_FRACTION

#: Fewest samples that can support a spectrum worth reading.
MIN_SAMPLES = 32

#: Periods the record must span before a frequency counts as measured.  Below
#: about four cycles the FFT's own resolution is comparable to the frequency
#: being claimed.
MIN_PERIODS = 4.0

#: Autocorrelation at one-period lag below which the signal is not periodic.
#: A pure sinusoid gives 1.0; noise gives ~0. Half is a generous floor that
#: still admits a signal with substantial broadband content on top.
MIN_PERIODICITY = 0.5

#: Fraction of spectral power that must sit in the peak's own band.  White
#: noise over N bins scores ~5/N, which for any usable record is far below this.
MIN_CONCENTRATION = 0.15

#: Bins either side of the peak counted as belonging to it.  A Hann window
#: spreads a pure tone over about three bins; two either side covers it with
#: margin without absorbing an unrelated neighbour.
PEAK_HALF_WIDTH = 2

#: A signal whose scatter is this small relative to its own scale is flat --
#: a steady wake fluctuating at round-off, not a quiet oscillation.
FLAT_RTOL = 1.0e-12


@dataclass(frozen=True)
class Shedding:
    """Whether a signal is periodic, at what frequency, and how convincingly.

    Attributes
    ----------
    detected:
        True only when all three criteria hold.  This is the flag worth acting
        on; the individual measures below say which one failed.
    frequency:
        Dominant frequency in cycles per unit solver time, or ``nan``.
    strouhal:
        ``f L / U``.
    periods_observed:
        How many cycles the retained record spans.
    periodicity:
        Autocorrelation at a lag of one period, in ``[-1, 1]``.  Near 1 for a
        clean oscillation.
    concentration:
        Fraction of spectral power inside the peak's band.
    amplitude:
        Peak-to-peak swing of the retained signal.
    """

    detected: bool
    frequency: float
    strouhal: float
    periods_observed: float
    periodicity: float
    concentration: float
    amplitude: float
    n_samples: int
    reason: str = ""

    def report(self) -> str:
        """One block, printable whatever the verdict turned out to be."""
        if not self.detected:
            return f"    no periodic shedding detected: {self.reason}"
        return (
            f"    shedding at St = {self.strouhal:.4f} "
            f"(f = {self.frequency:.4g}, {self.periods_observed:.1f} periods "
            f"observed)\n"
            f"    periodicity {self.periodicity:.2f}, "
            f"spectral concentration {self.concentration:.2f}, "
            f"peak-to-peak {self.amplitude:.4g}"
        )


def _autocorrelation_at(signal: np.ndarray, lag: float) -> float:
    """Autocorrelation at a possibly fractional ``lag``, in samples.

    The period almost never lands on a whole number of samples, and rounding it
    to one costs exactly the sharpness this measure exists to provide -- so the
    two bracketing integer lags are interpolated between.
    """
    n = signal.size
    if lag < 1 or lag >= n - 1:
        return float("nan")

    centred = signal - signal.mean()
    variance = float(np.mean(centred ** 2))
    if variance <= 0:
        return float("nan")

    def at(k: int) -> float:
        return float(np.mean(centred[:-k] * centred[k:])) / variance

    low = int(math.floor(lag))
    weight = lag - low
    if low + 1 >= n - 1:
        return at(low)
    return (1.0 - weight) * at(low) + weight * at(low + 1)


def detect_shedding(times, signal, l_ref: float = 1.0, u_ref: float = 1.0,
                    transient_fraction: float = DEFAULT_TRANSIENT_FRACTION,
                    min_amplitude: float = 0.0) -> Shedding:
    """Decide whether ``signal`` is periodically shedding, and report why.

    Parameters
    ----------
    times, signal:
        Sample times and the oscillating quantity -- a transverse velocity
        probe in the wake, or a lift coefficient history.
    l_ref, u_ref:
        Scales for the Strouhal number.
    transient_fraction:
        Fraction of the record discarded before looking, matching
        :func:`pycfd.analysis.timeseries.time_average`.
    min_amplitude:
        Peak-to-peak swing below which the oscillation is dismissed as
        physically negligible however clean it looks.  Zero imposes no such
        floor; that judgement usually belongs to the case, not here.

    Returns
    -------
    Shedding
        Always returned.  ``detected`` carries the verdict and ``reason`` says
        what failed, since "no" is as useful an answer as "yes" here.
    """
    t = np.asarray(times, dtype=float)
    x = np.asarray(signal, dtype=float)
    if t.size != x.size:
        raise ValueError(
            f"times and signal must have the same length, got {t.size} and {x.size}"
        )

    def nothing(reason: str, **fields) -> Shedding:
        base = dict(detected=False, frequency=float("nan"), strouhal=float("nan"),
                    periods_observed=float("nan"), periodicity=float("nan"),
                    concentration=float("nan"), amplitude=float("nan"),
                    n_samples=0, reason=reason)
        base.update(fields)
        return Shedding(**base)

    if t.size == 0:
        return nothing("the record is empty")

    keep = t > transient_fraction * t[-1]
    t, x = t[keep], x[keep]
    if t.size < MIN_SAMPLES:
        return nothing(
            f"only {t.size} samples after discarding the transient, "
            f"below the {MIN_SAMPLES} a spectrum needs",
            n_samples=int(t.size),
        )

    amplitude = float(x.max() - x.min())
    scale = max(abs(float(x.mean())), amplitude, 1.0)
    if float(x.std()) <= FLAT_RTOL * scale:
        return nothing("the signal is flat — the wake is steady",
                       amplitude=amplitude, n_samples=int(t.size))
    if amplitude < min_amplitude:
        return nothing(
            f"peak-to-peak {amplitude:.3g} is below the {min_amplitude:.3g} "
            "worth calling an oscillation",
            amplitude=amplitude, n_samples=int(t.size),
        )

    # The solver takes adaptive steps, so the record has to be put on a uniform
    # time base before it can be transformed.
    uniform = np.linspace(t[0], t[-1], t.size)
    resampled = np.interp(uniform, t, x)
    dt = float(uniform[1] - uniform[0])
    duration = float(uniform[-1] - uniform[0])
    if dt <= 0 or duration <= 0:
        return nothing("the record covers no time", amplitude=amplitude,
                       n_samples=int(t.size))

    centred = resampled - resampled.mean()
    power = np.abs(np.fft.rfft(centred * np.hanning(centred.size))) ** 2
    freqs = np.fft.rfftfreq(centred.size, dt)
    power[0] = 0.0                       # the residual mean is not a frequency
    total = float(power.sum())
    if total <= 0:
        return nothing("the spectrum is empty", amplitude=amplitude,
                       n_samples=int(t.size))

    peak = int(np.argmax(power))
    frequency = float(freqs[peak])
    if frequency <= 0:
        return nothing("no frequency above the mean", amplitude=amplitude,
                       n_samples=int(t.size))

    lo = max(1, peak - PEAK_HALF_WIDTH)
    hi = min(power.size, peak + PEAK_HALF_WIDTH + 1)
    concentration = float(power[lo:hi].sum()) / total

    periods = duration * frequency
    periodicity = _autocorrelation_at(centred, 1.0 / (frequency * dt))
    strouhal = frequency * l_ref / u_ref

    measured = dict(frequency=frequency, strouhal=strouhal,
                    periods_observed=periods, periodicity=periodicity,
                    concentration=concentration, amplitude=amplitude,
                    n_samples=int(t.size))

    # Each failure names the measure that produced it, so the report says what
    # to change -- run longer, or stop calling this a wake.
    if periods < MIN_PERIODS:
        return nothing(
            f"only {periods:.1f} periods observed, below the {MIN_PERIODS:g} "
            "a frequency needs to be measured over; run longer",
            **measured,
        )
    if not math.isfinite(periodicity) or periodicity < MIN_PERIODICITY:
        return nothing(
            f"the signal does not repeat: autocorrelation at one period is "
            f"{periodicity:.2f}, below {MIN_PERIODICITY:g}. A drift or broadband "
            "wobble has a strongest frequency too",
            **measured,
        )
    if concentration < MIN_CONCENTRATION:
        return nothing(
            f"only {concentration * 100:.0f}% of the spectral power sits in the "
            f"peak, below {MIN_CONCENTRATION * 100:.0f}% — the signal is "
            "broadband, not a tone",
            **measured,
        )
    return Shedding(detected=True, reason="", **measured)
