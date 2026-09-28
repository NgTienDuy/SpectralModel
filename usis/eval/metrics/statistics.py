"""Interval estimates, because the brief forbids bare point estimates.

§11.3: *tail statistics — stratified bootstrap CIs and IQM, never bare point estimates.* The
enforcement is in the type: :class:`MetricResult` can only be built from per-item scores, so there
is no constructor that accepts a single number and calls it a result.

Two estimators, and they answer different questions:

* the **mean with a bootstrap interval** answers "what would this number be on another sample of
  the same size?";
* the **interquartile mean** answers "what is typical, ignoring the tails?" — useful precisely
  where a few catastrophic failures would otherwise dominate, which in this domain is common.

Reporting both is deliberate. A large gap between them *is* a finding: it says the mean is being
carried by a minority of cases, and that a headline average is hiding a bimodal outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from usis.core.errors import UsisError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "MetricResult",
    "bootstrap_ci",
    "interquartile_mean",
]

#: Enough resamples that the interval is stable to about the third decimal for a typical metric.
DEFAULT_BOOTSTRAP_SAMPLES: int = 10_000


def _as_scores(scores: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Validate and coerce per-item scores."""
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1:
        raise UsisError(f"{name}: scores must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise UsisError(f"{name}: cannot summarise an empty score vector")
    if not np.all(np.isfinite(array)):
        raise UsisError(
            f"{name}: scores must all be finite; a NaN here usually means a failed item was "
            "silently included rather than counted in the failure histogram"
        )
    return array


def bootstrap_ci(
    scores: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
    groups: Sequence[object] | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``scores``.

    When ``groups`` is given the resampling is **over groups, not items**: replicate spectra of one
    sample are not independent, so resampling them individually would produce an interval that is
    too narrow by roughly the square root of the replicate count. Passing the same grouping the
    split used is almost always the right thing.
    """
    array = _as_scores(scores, "bootstrap_ci")
    if not 0.0 < confidence < 1.0:
        raise UsisError(f"confidence must be in (0, 1), got {confidence}")
    if n_samples < 100:
        raise UsisError(f"n_samples must be at least 100 for a usable interval, got {n_samples}")

    rng = np.random.default_rng(seed)
    if groups is None:
        draws = rng.integers(0, array.size, size=(n_samples, array.size))
        means = array[draws].mean(axis=1)
    else:
        keys = np.asarray([str(g) for g in groups])
        if keys.size != array.size:
            raise UsisError(f"{keys.size} group labels for {array.size} scores")
        unique = np.unique(keys)
        if unique.size < 2:
            raise UsisError(
                "a group bootstrap over a single group cannot produce an interval: every resample "
                "is the same data, so the result collapses to a point estimate wearing a "
                "confidence interval. Report the point with n=1 group and say so, or group less "
                "coarsely."
            )
        buckets = [array[keys == key] for key in unique]
        picks = rng.integers(0, len(buckets), size=(n_samples, len(buckets)))
        means = np.array([np.concatenate([buckets[i] for i in row]).mean() for row in picks])

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def interquartile_mean(scores: Sequence[float] | np.ndarray) -> float:
    """Mean of the values between the 25th and 75th percentiles.

    Falls back to the plain mean when fewer than four values are present, since an interquartile
    range over three points is not a robust statistic — it is arithmetic with extra steps.
    """
    array = _as_scores(scores, "interquartile_mean")
    if array.size < 4:
        return float(array.mean())
    low, high = np.quantile(array, [0.25, 0.75])
    middle = array[(array >= low) & (array <= high)]
    return float(middle.mean() if middle.size else array.mean())


@dataclass(frozen=True)
class MetricResult:
    """One metric, reported as an interval rather than a number.

    There is no constructor taking a bare value: build it with :meth:`from_scores` and the interval
    comes with it.
    """

    name: str
    mean: float
    ci_low: float
    ci_high: float
    iqm: float
    n_items: int
    n_groups: int | None = None
    confidence: float = 0.95

    def __post_init__(self) -> None:
        """Validate the interval brackets the point estimate."""
        if self.ci_low > self.ci_high:
            raise UsisError(f"{self.name}: interval is inverted ({self.ci_low}, {self.ci_high})")
        if self.n_items < 1:
            raise UsisError(f"{self.name}: n_items must be positive")

    @classmethod
    def from_scores(
        cls,
        name: str,
        scores: Sequence[float] | np.ndarray,
        *,
        confidence: float = 0.95,
        n_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
        seed: int = 0,
        groups: Sequence[object] | None = None,
    ) -> MetricResult:
        """Summarise per-item scores into a reportable interval."""
        array = _as_scores(scores, name)
        low, high = bootstrap_ci(
            array, confidence=confidence, n_samples=n_samples, seed=seed, groups=groups
        )
        return cls(
            name=name,
            mean=float(array.mean()),
            ci_low=low,
            ci_high=high,
            iqm=interquartile_mean(array),
            n_items=int(array.size),
            n_groups=None if groups is None else len({str(g) for g in groups}),
            confidence=confidence,
        )

    @property
    def is_degenerate(self) -> bool:
        """Whether the interval has collapsed to a point, which is not an interval.

        Happens with one item, or with identical scores. The brief forbids bare point estimates, so
        a collapsed interval has to be visible rather than rendered as though it meant something.
        """
        return self.ci_low == self.ci_high

    @property
    def mean_iqm_gap(self) -> float:
        """Absolute difference between the mean and the IQM.

        A large gap says the mean is carried by the tails, which is a finding rather than a
        nuisance: it usually means a minority of catastrophic failures, and a headline average is
        hiding them.
        """
        return abs(self.mean - self.iqm)

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable view, including the degeneracy flag rather than only the numbers."""
        return {
            "name": self.name,
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "iqm": self.iqm,
            "n_items": self.n_items,
            "n_groups": self.n_groups,
            "degenerate": self.is_degenerate,
        }

    def render(self) -> str:
        """One line for a result card."""
        groups = "" if self.n_groups is None else f", {self.n_groups} groups"
        degenerate = " (DEGENERATE: not an interval)" if self.is_degenerate else ""
        return (
            f"{self.name}: {self.mean:.4g} "
            f"[{self.ci_low:.4g}, {self.ci_high:.4g}] at {self.confidence:.0%}, "
            f"IQM {self.iqm:.4g}, n={self.n_items}{groups}{degenerate}"
        )
