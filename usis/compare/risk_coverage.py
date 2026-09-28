"""The risk-coverage curve - the project's headline progress curve, built so it cannot flatter.

`CLAUDE.md`: *near-zero risk at a stated coverage, and push coverage up while holding risk at the
floor. The risk-coverage frontier is the project's headline progress curve and appears in every
milestone report.* This module computes it, and four of its choices exist to stop the curve from
being the wrong curve.

**The denominator is the whole corpus, never the answered part.** Coverage is accepted items over
*items presented*. A method that refuses two thirds of the corpus and is perfect on the rest has
coverage 0.33, not 1.0. If the denominator were the answered set instead, the way to win would be
to answer less, which inverts the entire point of measuring coverage.

**Thresholds land on values, not on ranks.** Accepting "the top k by confidence" is not a
realisable operating point when several items share a confidence: no threshold separates them. So
a point is emitted only after a whole tie group is accepted, and the curve is the set of operating
points a deployed threshold could actually reach.

**AURC states the range it integrated over.** A method that can only reach coverage 0.4 has its
area computed over [0, 0.4], and comparing that number with another method's area over [0, 1.0] is
comparing two different integrals. :attr:`RiskCoverageCurve.max_coverage` travels with the number
and :func:`comparable_aurc` is the guard a leaderboard calls before ranking on it.

**The interval comes from resampling the items, not from a formula.** Risk at a coverage is a
ratio of counts over a subset chosen by the statistic itself; there is no closed form worth
trusting here, and the project forbids bare point estimates anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from usis.compare.scoring import ScoringError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from usis.compare.scoring import ItemScore

__all__ = [
    "CurveEstimate",
    "RiskCoverageCurve",
    "RiskCoveragePoint",
    "bootstrap_curve_statistic",
    "comparable_aurc",
    "risk_coverage_curve",
]


@dataclass(frozen=True)
class RiskCoveragePoint:
    """One realisable operating point: accept everything at or above ``threshold``."""

    threshold: float
    coverage: float
    risk: float
    n_accepted: int
    n_errors: int

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable point."""
        return {
            "threshold": self.threshold,
            "coverage": self.coverage,
            "risk": self.risk,
            "n_accepted": self.n_accepted,
            "n_errors": self.n_errors,
        }


@dataclass(frozen=True)
class RiskCoverageCurve:
    """The frontier, plus the two facts needed to read it honestly.

    ``n_presented`` is the coverage denominator and ``n_answerable`` the number of items that
    reached an answer at all. The gap between them is what abstention, refusal and crashes cost,
    and it is the difference between a curve that stops at 0.3 and one that reaches 1.0.
    """

    points: tuple[RiskCoveragePoint, ...]
    n_presented: int
    n_answerable: int

    @property
    def max_coverage(self) -> float:
        """The highest coverage this method can reach at any threshold."""
        return max((p.coverage for p in self.points), default=0.0)

    @property
    def aurc(self) -> float:
        """Area under the curve over ``[0, max_coverage]``. Lower is better.

        Integrated from the origin: the segment from coverage 0 to the first emitted point uses
        that point's risk, which is the conservative reading - it does not credit a method for an
        unmeasured stretch of very high confidence.
        """
        if not self.points:
            return 0.0
        coverages = np.array([0.0] + [p.coverage for p in self.points])
        risks = np.array([self.points[0].risk] + [p.risk for p in self.points])
        return float(np.trapezoid(risks, coverages))

    def risk_at_coverage(self, target: float) -> float | None:
        """Risk at the first operating point reaching ``target`` coverage.

        ``None`` when the method cannot reach that coverage at all. Returning the risk at its own
        maximum instead would report a method at an operating point it does not have, which is the
        specific way a selective-prediction table usually misleads.
        """
        if not 0.0 < target <= 1.0:
            raise ScoringError(f"target coverage must be in (0, 1], got {target}")
        reachable = [p for p in self.points if p.coverage >= target]
        return min(p.risk for p in reachable) if reachable else None

    def coverage_at_risk(self, max_risk: float) -> float:
        """The most coverage obtainable while holding risk at or below ``max_risk``.

        This is the number `CLAUDE.md` names as the goal: *push coverage up while holding risk at
        the floor.* Zero is a real answer - it means no threshold makes this method safe enough.
        """
        if not 0.0 <= max_risk <= 1.0:
            raise ScoringError(f"max_risk must be in [0, 1], got {max_risk}")
        return max((p.coverage for p in self.points if p.risk <= max_risk), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view carrying the curve and the range it spans."""
        return {
            "n_presented": self.n_presented,
            "n_answerable": self.n_answerable,
            "max_coverage": self.max_coverage,
            "aurc": self.aurc,
            "aurc_integrated_over": [0.0, self.max_coverage],
            "points": [p.to_dict() for p in self.points],
        }


def risk_coverage_curve(scores: Sequence[ItemScore]) -> RiskCoverageCurve:
    """Build the frontier from one method's per-item scores.

    ``UNSCORABLE`` items are dropped before anything is counted - they are not part of the corpus
    this method was measured on. Everything else is in the denominator, including the items the
    method refused or crashed on.
    """
    counted = [s for s in scores if s.outcome.is_counted]
    if not counted:
        raise ScoringError(
            "no scorable items: every item was UNSCORABLE, so there is no corpus to measure "
            "against. A curve over zero items would be a picture of nothing."
        )
    answered = [s for s in counted if s.outcome.is_answered]
    if not answered:
        return RiskCoverageCurve(points=(), n_presented=len(counted), n_answerable=0)

    confidences = np.array([s.confidence for s in answered], dtype=np.float64)
    errors = np.array([s.is_error for s in answered], dtype=np.float64)
    order = np.argsort(-confidences, kind="stable")
    sorted_conf = confidences[order]
    cumulative_errors = np.cumsum(errors[order])

    # A point is emitted only where the next item has a strictly lower confidence, so every point
    # is reachable by an actual threshold rather than by a rank cut inside a tie.
    boundary = np.append(sorted_conf[1:] != sorted_conf[:-1], True)
    indices = np.flatnonzero(boundary)

    n_presented = len(counted)
    points = tuple(
        RiskCoveragePoint(
            threshold=float(sorted_conf[i]),
            coverage=(i + 1) / n_presented,
            risk=float(cumulative_errors[i]) / (i + 1),
            n_accepted=int(i + 1),
            n_errors=int(cumulative_errors[i]),
        )
        for i in indices
    )
    return RiskCoverageCurve(points=points, n_presented=n_presented, n_answerable=len(answered))


@dataclass(frozen=True)
class CurveEstimate:
    """A curve statistic with a resampled interval, and the flag for when it collapsed.

    Deliberately not a :class:`~usis.eval.metrics.statistics.MetricResult`: that type is built from
    per-item scores and carries an IQM, and an interquartile mean of an area under a curve is not a
    quantity. The shared requirement - never a bare point estimate - is met by construction here
    too, since there is no way to build one without an interval.
    """

    name: str
    value: float
    ci_low: float
    ci_high: float
    n_items: int
    n_groups: int | None = None
    confidence: float = 0.95

    @property
    def is_degenerate(self) -> bool:
        """Whether the interval collapsed to a point, which is not an interval."""
        return self.ci_low == self.ci_high

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view including the degeneracy flag."""
        return {
            "name": self.name,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "n_items": self.n_items,
            "n_groups": self.n_groups,
            "degenerate": self.is_degenerate,
        }

    def render(self) -> str:
        """One line for a leaderboard or a result card."""
        degenerate = " (DEGENERATE: not an interval)" if self.is_degenerate else ""
        return (
            f"{self.name}: {self.value:.4g} [{self.ci_low:.4g}, {self.ci_high:.4g}] "
            f"at {self.confidence:.0%}, n={self.n_items}{degenerate}"
        )


def bootstrap_curve_statistic(
    scores: Sequence[ItemScore],
    statistic: Callable[[RiskCoverageCurve], float],
    *,
    name: str,
    groups: Sequence[object] | None = None,
    confidence: float = 0.95,
    n_samples: int = 2000,
    seed: int = 0,
) -> CurveEstimate:
    """Resample the items, rebuild the curve on each resample, and summarise the statistic.

    The curve is recomputed from scratch on every resample rather than the point estimate being
    perturbed, because the statistic depends on which items the threshold happens to accept - and
    that changes with the resample. ``groups`` resamples over groups, as replicate spectra of one
    sample are not independent; pass the same grouping the split used.
    """
    counted = [s for s in scores if s.outcome.is_counted]
    if not counted:
        raise ScoringError(f"{name}: no scorable items to resample")
    if not 0.0 < confidence < 1.0:
        raise ScoringError(f"confidence must be in (0, 1), got {confidence}")
    if n_samples < 100:
        raise ScoringError(f"n_samples must be at least 100 for a usable interval, got {n_samples}")

    point = statistic(risk_coverage_curve(counted))
    rng = np.random.default_rng(seed)

    if groups is None:
        buckets = [[s] for s in counted]
        n_groups = None
    else:
        keys = [str(g) for g in groups]
        if len(keys) != len(scores):
            raise ScoringError(f"{len(keys)} group labels for {len(scores)} scores")
        by_key: dict[str, list[ItemScore]] = {}
        for score, key in zip(scores, keys, strict=True):
            if score.outcome.is_counted:
                by_key.setdefault(key, []).append(score)
        buckets = [by_key[k] for k in sorted(by_key)]
        n_groups = len(buckets)
        if n_groups < 2:
            raise ScoringError(
                f"{name}: a group bootstrap over a single group resamples the same data every "
                "time, so the interval collapses to a point wearing a confidence interval. "
                "Report the point with n=1 group and say so, or group less coarsely."
            )

    draws = rng.integers(0, len(buckets), size=(n_samples, len(buckets)))
    values = []
    for row in draws:
        resampled = [s for i in row for s in buckets[i]]
        try:
            values.append(statistic(risk_coverage_curve(resampled)))
        except ScoringError:  # pragma: no cover - only if a resample holds no counted item
            continue
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [tail, 1.0 - tail])
    return CurveEstimate(
        name=name,
        value=point,
        ci_low=float(low),
        ci_high=float(high),
        n_items=len(counted),
        n_groups=n_groups,
        confidence=confidence,
    )


def comparable_aurc(curves: Sequence[RiskCoverageCurve], *, tolerance: float = 1e-9) -> bool:
    """Whether these curves' AURCs are integrated over the same coverage range.

    Called before ranking a leaderboard on AURC. When it is false the areas are over different
    intervals and the ordering is not a comparison - rank on risk at a fixed coverage, or on
    coverage at a fixed risk, both of which name their operating point.
    """
    maxima = [c.max_coverage for c in curves]
    return bool(maxima) and max(maxima) - min(maxima) <= tolerance
