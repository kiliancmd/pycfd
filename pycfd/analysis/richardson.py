"""Richardson extrapolation and the Grid Convergence Index.

Three grids and a measured quantity are enough to ask three questions at once:
what order the scheme is *actually* achieving, what the answer would be on an
infinitely fine grid, and how much of the finest grid's value is still
discretisation error.  The classical answers are Richardson extrapolation and
Roache's Grid Convergence Index, in the form standardised by ASME V&V 20.

The point of this module is the question that comes *first*, though.

Why the regime check comes first
--------------------------------
Richardson extrapolation assumes the sequence is in its asymptotic regime --
that the error is already dominated by a single power of the cell size.  Handed
a sequence that is not, it returns a number anyway, and that number is
worthless in a way nothing about it advertises: a diverging triplet and a
beautifully converging one both produce a float.

So the convergence ratio is computed before anything else::

    R = (f_fine - f_mid) / (f_mid - f_coarse)

    0 < R < 1   monotonic convergence      -- extrapolate
    -1 < R < 0  oscillatory convergence    -- extrapolate, with a wider band
    |R| >= 1    divergence                 -- refuse to extrapolate

This is the check the F-22 resolution sweep needed and did not have.  Its drag
coefficient moved materially between 5, 10 and 20 cells across the body, and
the question "is this sequence converging at all, or just moving?" was answered
by eye.

Refinement ratios
-----------------
Grids do not have to double.  When the two ratios differ the order appears on
both sides of its own equation, and ASME V&V 20 resolves it by fixed-point
iteration; with equal ratios that reduces to the textbook logarithm.  Both are
handled here, because a sweep constrained by what fits in memory rarely doubles
cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Roache's factor of safety for a three-grid study.  1.25 is the value ASME
#: V&V 20 recommends when an observed order has actually been computed from
#: three grids, rather than assumed from the scheme's formal order.
DEFAULT_SAFETY_FACTOR = 1.25

#: Orders outside this band are not believed.  A second-order scheme that
#: appears to be converging at order 7 is not doing better than expected -- it
#: is reporting the ratio of two differences that are mostly noise, or a
#: sequence that has not reached its asymptotic regime.
MIN_PLAUSIBLE_ORDER = 0.5
MAX_PLAUSIBLE_ORDER = 4.0

#: Differences below this are treated as exactly converged rather than divided
#: by: two grids agreeing to round-off carry no information about the order.
NEGLIGIBLE_DIFFERENCE = 1.0e-12

#: Iteration budget for the unequal-ratio order equation.
_MAX_ITERATIONS = 100
_ITERATION_TOL = 1.0e-12


@dataclass(frozen=True)
class RichardsonEstimate:
    """What three grids say about a quantity, and whether to believe it.

    Attributes
    ----------
    resolutions, values:
        The triplet used, coarsest first.
    convergence_ratio:
        ``R`` above.  The single number that decides whether the rest means
        anything.
    regime:
        ``"monotonic"``, ``"oscillatory"``, ``"diverging"`` or ``"converged"``
        (the last when the grids already agree to round-off).
    observed_order:
        The order the sequence actually exhibits, not the scheme's formal one.
    extrapolated:
        The zero-cell-size limit, or ``nan`` when the regime forbids it.
    gci:
        Fine-grid Grid Convergence Index, as a *fraction* -- the band within
        which the converged answer is expected to lie, at roughly 95%
        confidence.  ``nan`` when extrapolation was refused.
    trustworthy:
        True only when the sequence converges *and* the order it converges at
        is plausible.  This is the flag the whole module exists to raise.
    """

    resolutions: tuple[int, ...]
    values: tuple[float, ...]
    convergence_ratio: float
    regime: str
    observed_order: float
    extrapolated: float
    gci: float
    trustworthy: bool
    reason: str = ""

    @property
    def relative_error(self) -> float:
        """How far the finest grid sits from the extrapolated limit."""
        if not math.isfinite(self.extrapolated) or self.extrapolated == 0:
            return float("nan")
        return abs(self.values[-1] - self.extrapolated) / abs(self.extrapolated)

    def band(self) -> tuple[float, float]:
        """The interval the converged value is expected to lie in."""
        if not math.isfinite(self.gci):
            return (float("nan"), float("nan"))
        margin = abs(self.values[-1]) * self.gci
        return (self.values[-1] - margin, self.values[-1] + margin)

    def report(self) -> str:
        """Formatted verdict, safe to print whatever the regime turned out to be."""
        grids = " -> ".join(f"{n}" for n in self.resolutions)
        lines = [
            f"    grids {grids}   R = {self.convergence_ratio:+.4f}  "
            f"({self.regime})",
        ]
        if not math.isfinite(self.extrapolated):
            lines.append(f"    NOT EXTRAPOLATED: {self.reason}")
            return "\n".join(lines)

        lo, hi = self.band()
        lines.append(
            f"    order from successive differences {self.observed_order:.3f}   "
            f"extrapolated {self.extrapolated:.6g}   "
            f"finest is {self.relative_error * 100:.2f}% from it"
        )
        lines.append(
            f"    GCI {self.gci * 100:.2f}%  ->  {lo:.6g} .. {hi:.6g}"
        )
        if not self.trustworthy:
            lines.append(f"    TREAT WITH CAUTION: {self.reason}")
        return "\n".join(lines)


def _observed_order(e_coarse: float, e_fine: float,
                    r_coarse: float, r_fine: float) -> float:
    """Order of accuracy from two successive differences and their grid ratios.

    ``r_fine`` is the refinement ratio of the *finest* pair and ``r_coarse``
    that of the coarser pair -- the order is anchored on the finest pair, which
    is the one nearest the asymptotic regime.  (ASME V&V 20 numbers its grids
    from the finest, so its ``r21`` is this ``r_fine``; the sequences here run
    coarsest-first, and getting that backwards silently biases the order on any
    study whose grids do not double uniformly.)

    With equal ratios this is the textbook ``ln|e_coarse/e_fine| / ln(r)``.
    Otherwise the order appears inside its own right-hand side, and the ASME
    fixed-point iteration is used with the equal-ratio answer as its seed.
    """
    ratio = e_coarse / e_fine                 # > 1 in magnitude when converging
    s = 1.0 if ratio >= 0 else -1.0
    seed = abs(math.log(abs(ratio))) / math.log(r_fine)
    if abs(r_coarse - r_fine) < 1.0e-12:
        return seed                           # q(p) vanishes identically

    p = seed
    for _ in range(_MAX_ITERATIONS):
        try:
            q = math.log((r_fine ** p - s) / (r_coarse ** p - s))
        except (ValueError, ZeroDivisionError):
            return float("nan")
        nxt = abs(math.log(abs(ratio)) + q) / math.log(r_fine)
        if abs(nxt - p) < _ITERATION_TOL:
            return nxt
        p = nxt
    return float("nan")


def richardson(resolutions, values,
               safety_factor: float = DEFAULT_SAFETY_FACTOR) -> RichardsonEstimate:
    """Extrapolate a three-grid sequence, or refuse to and say why.

    Parameters
    ----------
    resolutions:
        Cell counts, coarsest first.  More than three may be given; the finest
        three are used, since those are the ones nearest the asymptotic regime.
    values:
        The quantity measured on each grid.
    safety_factor:
        Roache's ``Fs`` for the Grid Convergence Index.

    Returns
    -------
    RichardsonEstimate
        Always returned -- a refusal is a result, not an exception.  Check
        :attr:`~RichardsonEstimate.trustworthy` before using the number.
    """
    resolutions = [int(n) for n in resolutions]
    values = [float(v) for v in values]
    if len(resolutions) != len(values):
        raise ValueError("resolutions and values must have the same length")
    if len(resolutions) < 3:
        raise ValueError(
            f"Richardson extrapolation needs three grids to estimate an order "
            f"and a limit, got {len(resolutions)}"
        )
    if any(b <= a for a, b in zip(resolutions, resolutions[1:])):
        raise ValueError(f"resolutions must increase, got {resolutions}")

    n1, n2, n3 = resolutions[-3:]
    f1, f2, f3 = values[-3:]
    r21, r32 = n2 / n1, n3 / n2

    e_coarse, e_fine = f2 - f1, f3 - f2
    refused = dict(observed_order=float("nan"), extrapolated=float("nan"),
                   gci=float("nan"), trustworthy=False)

    scale = max(abs(f1), abs(f2), abs(f3), 1.0)
    if abs(e_coarse) < NEGLIGIBLE_DIFFERENCE * scale:
        # The two coarser grids already agree; there is no error sequence left
        # to fit, which is a good outcome rather than a failed one.
        return RichardsonEstimate(
            (n1, n2, n3), (f1, f2, f3), 0.0, "converged",
            observed_order=float("nan"), extrapolated=f3, gci=0.0,
            trustworthy=True,
            reason="the grids already agree to round-off",
        )

    ratio = e_fine / e_coarse
    if abs(ratio) >= 1.0:
        return RichardsonEstimate(
            (n1, n2, n3), (f1, f2, f3), ratio, "diverging", reason=(
                f"|R| = {abs(ratio):.3f} >= 1: refining changed the answer by "
                f"as much as the previous refinement did, so the sequence is "
                f"not converging and an extrapolated limit would be fiction"
            ), **refused,
        )

    regime = "monotonic" if ratio > 0 else "oscillatory"
    p = _observed_order(e_coarse, e_fine, r_coarse=r21, r_fine=r32)
    if not math.isfinite(p):
        return RichardsonEstimate(
            (n1, n2, n3), (f1, f2, f3), ratio, regime,
            reason="the order equation did not converge", **refused,
        )

    denominator = r32 ** p - 1.0
    if abs(denominator) < NEGLIGIBLE_DIFFERENCE:
        return RichardsonEstimate(
            (n1, n2, n3), (f1, f2, f3), ratio, regime,
            reason="the refinement ratio is too close to 1 to extrapolate from",
            **refused,
        )

    extrapolated = f3 + e_fine / denominator
    relative = abs(e_fine / f3) if f3 else float("nan")
    gci = safety_factor * relative / abs(denominator)

    trustworthy = MIN_PLAUSIBLE_ORDER <= p <= MAX_PLAUSIBLE_ORDER
    reason = ""
    if not trustworthy:
        reason = (
            f"observed order {p:.2f} is outside the plausible band "
            f"[{MIN_PLAUSIBLE_ORDER}, {MAX_PLAUSIBLE_ORDER}]: the sequence has "
            f"probably not reached its asymptotic regime, so the extrapolation "
            f"is an extrapolation of noise"
        )
    elif regime == "oscillatory":
        reason = ("the sequence oscillates rather than approaching from one "
                  "side; the limit is still meaningful but the band is looser "
                  "than it looks")

    return RichardsonEstimate(
        (n1, n2, n3), (f1, f2, f3), ratio, regime,
        observed_order=p, extrapolated=extrapolated, gci=gci,
        trustworthy=trustworthy, reason=reason,
    )
