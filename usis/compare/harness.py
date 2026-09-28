"""``run_comparison`` - one task, every method registered for it, one record.

This is the core of the product rather than a test utility, and the shape of the function says so:
it takes a :class:`~usis.compare.task.ComparisonTask` and a maturity filter, and it never takes a
list of methods to run. The set of competitors comes from the registry, which is what makes a
leaderboard that omits the baseline that would have beaten it impossible to produce by omission.

Four behaviours here are decisions, not implementation details.

**A refusal is recorded, not skipped.** When the applicability audit stops a method, the item still
counts in the denominator and the row's coverage falls. Skipping refused items instead would make
declaring a narrow validity domain a way to win the leaderboard, and would turn N7's contribution
into a scoring loophole.

**A crash is caught and named.** A defect in one method must not take the comparison down with it,
because the comparison is how defects are found. The exception type and message land in
``row.errors`` and the outcome is ``ERRORED``, which is loud in every rendering - a caught
exception here is a bug report, not a shrug.

**A shadow method is run and scored, and its row is marked.** `CLAUDE.md`: hypothesised methods are
*computed and scored on every run, never surfaced in a conclusion*. Both halves are here: the row
exists, and :attr:`~usis.compare.record.ComparisonRecord.surfaceable_rows` will not return it.
Scoring a shadow method against this task's independent references is legitimate precisely because
they are independent - ADR-0001 §S2 forbids scoring it against the portfolio's own answer, which is
a different thing and one the task layer refuses to supply.

**An interval resampled over items rather than groups says so.** When a task's items come from
fewer than two groups the group bootstrap is impossible, and the fallback narrows the interval by
roughly the square root of the replicate count. Rather than fail the whole comparison or quietly
substitute the narrower interval, the row carries the metric and a note naming the substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from usis.compare.record import ApplicabilityTally, ComparisonRecord, MethodRow
from usis.compare.risk_coverage import (
    RiskCoverageCurve,
    bootstrap_curve_statistic,
    risk_coverage_curve,
)
from usis.compare.scoring import ItemScore, Outcome
from usis.core.errors import AssumptionViolationError
from usis.eval.metrics.statistics import MetricResult
from usis.methods.applicability import Verdict
from usis.methods.base import execute
from usis.methods.registry import select_methods

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from usis.compare.task import ComparisonTask, TaskItem
    from usis.core.provenance import RunContext
    from usis.core.runlog import RunLog
    from usis.core.types import FailureCode, Maturity
    from usis.methods.applicability import Applicability
    from usis.methods.base import MethodBase
    from usis.methods.registry import MethodEntry

__all__ = ["MethodRun", "run_comparison", "run_method"]

#: Default number of bootstrap resamples for a curve statistic. Lower than the 10,000 used for a
#: simple mean because each resample rebuilds a whole curve; 2,000 holds the interval stable to
#: about the third decimal and keeps a six-method leaderboard interactive.
DEFAULT_CURVE_SAMPLES: int = 2000


@dataclass(frozen=True)
class MethodRun:
    """Everything one method produced over one task, before it is summarised into a row."""

    scores: list[ItemScore]
    audits: list[Applicability]
    errors: list[str]
    cost: dict[str, float]
    failures_declared: dict[FailureCode, int]
    failures_scored: dict[FailureCode, int]


def _histogram(per_item: Iterable[Sequence[FailureCode]]) -> dict[FailureCode, int]:
    """Count failure codes over items, at most once per item."""
    counts: dict[FailureCode, int] = {}
    for codes in per_item:
        for code in dict.fromkeys(codes):
            counts[code] = counts.get(code, 0) + 1
    return counts


def _tally(applicabilities: Sequence[Applicability]) -> ApplicabilityTally:
    """Summarise one method's applicability audits over a whole task."""
    violated: dict[str, int] = {}
    undecided: dict[str, int] = {}
    n_clean = n_degraded = n_refused = n_violated = n_undecided = 0
    for audit in applicabilities:
        if audit.verdict is Verdict.RUNNABLE:
            n_clean += 1
        elif audit.verdict is Verdict.RUNNABLE_DEGRADED:
            n_degraded += 1
        else:
            n_refused += 1
        if audit.violations:
            n_violated += 1
            for check in audit.violations:
                violated[check.assumption] = violated.get(check.assumption, 0) + 1
        if audit.undecided:
            n_undecided += 1
            for check in audit.undecided:
                label = f"{check.assumption} ({check.status})"
                undecided[label] = undecided.get(label, 0) + 1
    return ApplicabilityTally(
        n_runs=len(applicabilities),
        n_clean=n_clean,
        n_degraded=n_degraded,
        n_refused=n_refused,
        n_violated=n_violated,
        n_undecided=n_undecided,
        violated_assumptions=violated,
        undecided_assumptions=undecided,
    )


def _aurc_of(curve: RiskCoverageCurve) -> float:
    """Area under the risk-coverage curve, as a named function the bootstrap can resample."""
    return curve.aurc


def _coverage_at_risk_of(curve: RiskCoverageCurve, *, max_risk: float) -> float:
    """Coverage held at a risk ceiling - the project's headline statistic."""
    return curve.coverage_at_risk(max_risk)


def _refusal_codes(audit: Applicability) -> tuple[FailureCode, ...]:
    """The failure codes a refusal implicates, so a refused item still lands in the histogram."""
    return audit.failure_codes


def _indicator_metric(
    name: str,
    flags: Sequence[bool],
    groups: Sequence[str],
    *,
    confidence: float,
    n_samples: int,
    seed: int,
    notes: list[str],
) -> MetricResult | None:
    """A 0/1 metric with a group bootstrap, degrading to an item bootstrap *out loud*."""
    if not flags:
        return None
    scores = [1.0 if flag else 0.0 for flag in flags]
    if len({*groups}) >= 2:
        return MetricResult.from_scores(
            name, scores, confidence=confidence, n_samples=n_samples, seed=seed, groups=groups
        )
    notes.append(
        f"{name}: the interval was resampled over items, not groups - only "
        f"{len({*groups})} group(s) are present on this task, so a group bootstrap is impossible. "
        "Replicate correlation is therefore not reflected and the interval is narrower than the "
        "truth by roughly the square root of the replicate count."
    )
    return MetricResult.from_scores(
        name, scores, confidence=confidence, n_samples=n_samples, seed=seed
    )


def run_method(
    method: MethodBase,
    task: ComparisonTask,
    items: Sequence[TaskItem],
    ctx: RunContext,
    *,
    log: RunLog | None = None,
) -> MethodRun:
    """Run one method over every item, returning scores, audits, errors and accumulated cost.

    Separated from :func:`run_comparison` so a single method can be driven over a task without
    building a whole record - which is what a debugging session and a unit test both want.
    """
    scores: list[ItemScore] = []
    audits: list[Applicability] = []
    errors: list[str] = []
    cost: dict[str, float] = {}
    declared: dict[str, tuple[FailureCode, ...]] = {}
    scored: dict[str, tuple[FailureCode, ...]] = {}

    for item in items:
        if not item.is_scorable:
            scores.append(
                ItemScore(
                    item_id=item.item_id,
                    outcome=Outcome.UNSCORABLE,
                    detail=(
                        f"answer_source {item.answer_source} is not independent of the system's "
                        "own output, so this item was not run and is in no denominator "
                        "(ADR-0001 §S2)"
                    ),
                )
            )
            continue
        try:
            result = execute(method, item.inputs, ctx, log=log)
        except AssumptionViolationError as exc:
            audit = method.applicable(item.inputs)
            audits.append(audit)
            declared[item.item_id] = _refusal_codes(audit)
            scores.append(
                ItemScore(
                    item_id=item.item_id,
                    outcome=Outcome.REFUSED,
                    failure_codes=_refusal_codes(audit),
                    reference=item.reference,
                    detail=str(exc),
                )
            )
            continue
        # A defect in one method must not end the comparison: the comparison is how
        # defects are found. The type and message are kept and the outcome is loud.
        except Exception as exc:
            errors.append(f"{item.item_id}: {type(exc).__name__}: {exc}")
            scores.append(
                ItemScore(
                    item_id=item.item_id,
                    outcome=Outcome.ERRORED,
                    reference=item.reference,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        audits.append(result.applicability)
        for key, value in result.cost_incurred.items():
            cost[key] = cost.get(key, 0.0) + float(value)
        score = task.score(item, result)
        declared[item.item_id] = result.all_failure_codes
        scored[item.item_id] = score.failure_codes
        implied = tuple(dict.fromkeys((*score.failure_codes, *result.all_failure_codes)))
        scores.append(
            ItemScore(
                item_id=score.item_id,
                outcome=score.outcome,
                confidence=score.confidence,
                value=score.value,
                failure_codes=implied,
                reference=score.reference,
                predicted=score.predicted,
                detail=score.detail,
            )
        )
    return MethodRun(
        scores=scores,
        audits=audits,
        errors=errors,
        cost=cost,
        failures_declared=_histogram(declared.values()),
        failures_scored=_histogram(scored.values()),
    )


def run_comparison(
    task: ComparisonTask,
    ctx: RunContext,
    *,
    maturities: Sequence[Maturity] | None = None,
    entries: Sequence[MethodEntry] | None = None,
    risk_target: float = 0.01,
    coverage_target: float = 0.5,
    bootstrap_samples: int = DEFAULT_CURVE_SAMPLES,
    confidence: float = 0.95,
    shows: Sequence[str] = (),
    does_not_show: Sequence[str] = (),
    notes: Sequence[str] = (),
    config_hash: str = "",
    log: RunLog | None = None,
) -> ComparisonRecord:
    """Run every registered method for ``task.task_kind`` and build the comparison record.

    Args:
        task: the benchmark. Supplies the items, the grouping, the references and the scoring.
        ctx: the run context every result is attributed to.
        maturities: restrict the field. ``None`` runs every level, refuted entries included -
            re-running a refuted method to confirm it is still refuted is a legitimate query.
        entries: run exactly these registry entries instead of selecting by task. For tests and
            for a deliberate head-to-head; the CLI does not expose it, because a leaderboard whose
            field was hand-picked is not a leaderboard.
        risk_target: the risk ceiling for the headline ``coverage_at_risk`` statistic.
        coverage_target: retained on the record so a reader knows which operating point the
            accompanying prose was written about.
        bootstrap_samples: resamples behind every interval on the leaderboard.
        confidence: interval level, applied to every metric and curve estimate alike.
        shows: what this comparison establishes, in plain language, for the record.
        does_not_show: what it does not establish. Defaults to the task's own notes, and the
            record refuses to be built if both are empty.
        notes: anything else that must travel with the numbers.
        config_hash: the committed config this run resolved from, recorded for attribution.
        log: optional run log; every applicability audit and result is appended to it.

    Raises:
        RecordError: if no registered method declares this task family.
    """
    selected = (
        tuple(entries)
        if entries is not None
        else select_methods(task=task.task_kind, maturities=maturities)
    )
    items = tuple(task.items())
    groups = [item.group for item in items]
    n_unscorable = sum(1 for item in items if not item.is_scorable)

    rows: list[MethodRow] = []
    for entry in selected:
        run = run_method(entry.method, task, items, ctx, log=log)
        scores, audits, errors, cost = run.scores, run.audits, run.errors, run.cost
        counted = [s for s in scores if s.outcome.is_counted]
        counted_groups = [g for s, g in zip(scores, groups, strict=True) if s.outcome.is_counted]
        row_notes: list[str] = []
        metrics: list[MetricResult] = []

        coverage_metric = _indicator_metric(
            "coverage",
            [s.outcome.is_answered for s in counted],
            counted_groups,
            confidence=confidence,
            n_samples=bootstrap_samples,
            seed=ctx.seed,
            notes=row_notes,
        )
        if coverage_metric is not None:
            metrics.append(coverage_metric)

        answered = [(s, g) for s, g in zip(scores, groups, strict=True) if s.outcome.is_answered]
        accuracy_metric = _indicator_metric(
            "selective_accuracy",
            [s.outcome is Outcome.CORRECT for s, _ in answered],
            [g for _, g in answered],
            confidence=confidence,
            n_samples=bootstrap_samples,
            seed=ctx.seed,
            notes=row_notes,
        )
        if accuracy_metric is not None:
            metrics.append(accuracy_metric)

        valued = [(s, g) for s, g in zip(scores, groups, strict=True) if s.value is not None]
        if valued and task.value_metric_name:
            value_groups = [g for _, g in valued]
            values = [float(s.value) for s, _ in valued if s.value is not None]
            if len({*value_groups}) >= 2:
                metrics.append(
                    MetricResult.from_scores(
                        task.value_metric_name,
                        values,
                        confidence=confidence,
                        n_samples=bootstrap_samples,
                        seed=ctx.seed,
                        groups=value_groups,
                    )
                )
            else:
                row_notes.append(
                    f"{task.value_metric_name}: interval resampled over items, not groups - only "
                    f"{len({*value_groups})} group(s) present."
                )
                metrics.append(
                    MetricResult.from_scores(
                        task.value_metric_name,
                        values,
                        confidence=confidence,
                        n_samples=bootstrap_samples,
                        seed=ctx.seed,
                    )
                )

        if counted:
            curve = risk_coverage_curve(counted)
            estimates = [
                bootstrap_curve_statistic(
                    counted,
                    _aurc_of,
                    name="aurc",
                    groups=counted_groups if len({*counted_groups}) >= 2 else None,
                    confidence=confidence,
                    n_samples=bootstrap_samples,
                    seed=ctx.seed,
                ),
                bootstrap_curve_statistic(
                    counted,
                    partial(_coverage_at_risk_of, max_risk=risk_target),
                    name=f"coverage_at_risk<={risk_target:g}",
                    groups=counted_groups if len({*counted_groups}) >= 2 else None,
                    confidence=confidence,
                    n_samples=bootstrap_samples,
                    seed=ctx.seed,
                ),
            ]
        else:  # pragma: no cover - only when every item was unscorable
            curve = RiskCoverageCurve(points=(), n_presented=0, n_answerable=0)
            estimates = []

        failures: dict[FailureCode, int] = {}
        for score in scores:
            for code in score.failure_codes:
                failures[code] = failures.get(code, 0) + 1

        rows.append(
            MethodRow(
                method_key=entry.key,
                task=entry.method.task,
                maturity=entry.maturity,
                surfaceable=entry.surfaceable,
                scores=tuple(scores),
                curve=curve,
                applicability=_tally(audits),
                metrics=tuple(metrics),
                estimates=tuple(estimates),
                failures=failures,
                failures_declared=run.failures_declared,
                failures_scored=run.failures_scored,
                cost=cost,
                errors=tuple(errors),
                notes=tuple(row_notes),
            )
        )

    return ComparisonRecord(
        task_id=task.task_id,
        task_kind=task.task_kind,
        run_context=ctx,
        rows=tuple(rows),
        dataset_ids=task.dataset_ids,
        split=task.split,
        n_unscorable=n_unscorable,
        n_items_presented=len(items),
        risk_target=risk_target,
        coverage_target=coverage_target,
        bootstrap_samples=bootstrap_samples,
        config_hash=config_hash,
        shows=tuple(shows),
        does_not_show=tuple(does_not_show) or task.notes,
        notes=tuple(notes),
    )
