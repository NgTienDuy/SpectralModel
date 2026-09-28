"""Interval statistics. The brief forbids bare point estimates; these are the checks that hold."""

from __future__ import annotations

import numpy as np
import pytest

from usis.core.errors import UsisError
from usis.eval.metrics.statistics import (
    MetricResult,
    bootstrap_ci,
    interquartile_mean,
)

RNG = np.random.default_rng(0)


def test_a_bootstrap_interval_brackets_the_mean():
    scores = RNG.normal(0.8, 0.1, 500)
    low, high = bootstrap_ci(scores, seed=0)
    assert low < scores.mean() < high


def test_the_interval_narrows_as_the_sample_grows():
    def width(n: int) -> float:
        low, high = bootstrap_ci(RNG.normal(0.8, 0.1, n), seed=0)
        return high - low

    # Roughly 1/sqrt(n): a hundredfold sample should narrow it by about ten.
    assert width(50) / width(5000) == pytest.approx(10.0, rel=0.5)


def test_a_wider_confidence_level_gives_a_wider_interval():
    scores = RNG.normal(0.8, 0.1, 300)
    narrow = bootstrap_ci(scores, confidence=0.5, seed=0)
    wide = bootstrap_ci(scores, confidence=0.99, seed=0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_grouping_widens_the_interval_because_replicates_are_not_independent():
    # 20 samples with 10 replicates each. Resampling items pretends there are 200 independent
    # observations; resampling groups knows there are 20, and the honest interval is wider.
    group_means = RNG.normal(0.8, 0.1, 20)
    scores = np.repeat(group_means, 10) + RNG.normal(0, 0.005, 200)
    groups = np.repeat([f"g{i}" for i in range(20)], 10)
    by_item = bootstrap_ci(scores, seed=0)
    by_group = bootstrap_ci(scores, groups=groups, seed=0)
    assert (by_group[1] - by_group[0]) > 2 * (by_item[1] - by_item[0])


def test_a_single_group_bootstrap_is_refused_rather_than_collapsing():
    with pytest.raises(UsisError, match="single group"):
        bootstrap_ci([0.9, 0.95, 0.8], groups=["a", "a", "a"])


def test_bad_inputs_are_refused():
    with pytest.raises(UsisError, match="empty"):
        bootstrap_ci([])
    with pytest.raises(UsisError, match="finite"):
        bootstrap_ci([0.5, np.nan])
    with pytest.raises(UsisError, match="confidence"):
        bootstrap_ci([0.1, 0.2], confidence=1.5)
    with pytest.raises(UsisError, match="at least 100"):
        bootstrap_ci([0.1, 0.2], n_samples=10)
    with pytest.raises(UsisError, match="group labels"):
        bootstrap_ci([0.1, 0.2], groups=["a"])


def test_the_iqm_ignores_the_tails():
    scores = np.array([0.0, 0.8, 0.81, 0.82, 0.83, 100.0])
    assert interquartile_mean(scores) == pytest.approx(0.815, abs=0.01)
    assert scores.mean() > 15  # the mean is destroyed by one outlier


def test_the_iqm_falls_back_to_the_mean_for_tiny_samples():
    assert interquartile_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_a_metric_result_cannot_be_built_from_a_bare_number():
    # There is no constructor taking a point estimate; from_scores is the only way in.
    with pytest.raises(TypeError):
        MetricResult.from_scores("acc")  # type: ignore[call-arg]


def test_from_scores_reports_mean_interval_and_iqm_together():
    scores = RNG.beta(9, 1, 200)
    result = MetricResult.from_scores("recall@1", scores, seed=0)
    assert result.n_items == 200
    assert result.ci_low < result.mean < result.ci_high
    assert result.name in result.render()
    assert "IQM" in result.render()


def test_a_collapsed_interval_is_flagged_as_degenerate():
    # Identical scores give ci_low == ci_high. That is not an interval, and rendering it as one
    # would be exactly the bare point estimate the brief forbids.
    result = MetricResult.from_scores("acc", [0.5] * 8, seed=0)
    assert result.is_degenerate is True
    assert "DEGENERATE" in result.render()
    healthy = MetricResult.from_scores("acc", RNG.beta(9, 1, 100), seed=0)
    assert healthy.is_degenerate is False


def test_the_mean_iqm_gap_exposes_a_tail_carried_average():
    bimodal = np.concatenate([np.full(90, 0.95), np.zeros(10)])
    result = MetricResult.from_scores("acc", bimodal, seed=0)
    assert result.iqm == pytest.approx(0.95)
    assert result.mean < 0.9
    assert result.mean_iqm_gap > 0.05


def test_group_count_is_recorded():
    scores = RNG.beta(9, 1, 60)
    groups = [f"g{i % 6}" for i in range(60)]
    result = MetricResult.from_scores("acc", scores, groups=groups, seed=0)
    assert result.n_groups == 6
    assert "6 groups" in result.render()


def test_an_inverted_interval_is_refused():
    with pytest.raises(UsisError, match="inverted"):
        MetricResult(name="x", mean=0.5, ci_low=0.7, ci_high=0.3, iqm=0.5, n_items=10)
