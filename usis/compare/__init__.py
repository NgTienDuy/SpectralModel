"""The comparison harness: one task, every method registered for it, one machine-readable record.

The core of the product, not a test utility. Adding a method must never require touching it - and
neither must adding a benchmark, so the harness depends on two protocols and no concrete type:
:class:`~usis.methods.base.Method` on one side, :class:`~usis.compare.task.ComparisonTask` on the
other.

Milestone: M2.
"""

from __future__ import annotations

from usis.compare.harness import run_comparison, run_method
from usis.compare.leaderboard import rank_rows, render_markdown, render_table
from usis.compare.record import ApplicabilityTally, ComparisonRecord, MethodRow, RecordError
from usis.compare.risk_coverage import (
    CurveEstimate,
    RiskCoverageCurve,
    RiskCoveragePoint,
    comparable_aurc,
    risk_coverage_curve,
)
from usis.compare.scoring import ItemScore, Outcome, ScoringError, outcome_counts
from usis.compare.task import ComparisonTask, TaskError, TaskItem

__all__ = [
    "ApplicabilityTally",
    "ComparisonRecord",
    "ComparisonTask",
    "CurveEstimate",
    "ItemScore",
    "MethodRow",
    "Outcome",
    "RecordError",
    "RiskCoverageCurve",
    "RiskCoveragePoint",
    "ScoringError",
    "TaskError",
    "TaskItem",
    "comparable_aurc",
    "outcome_counts",
    "rank_rows",
    "render_markdown",
    "render_table",
    "risk_coverage_curve",
    "run_comparison",
    "run_method",
]
