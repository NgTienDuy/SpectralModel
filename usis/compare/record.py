"""The comparison record: what ``usis compare`` emits, and what a reader is owed.

One JSON file per comparison, one row per method. The row carries five things, and the reason each
is mandatory is that a leaderboard missing it has been misread before:

**Metrics with intervals**, never bare points - the project convention, and here it also stops two
methods separated by less than their own resampling noise from being called an ordering.

**The risk-coverage frontier**, because selective accuracy alone hides what it cost. A method at
99% accuracy on the third of the corpus it will touch is a different product from one at 92% on
all of it, and only the pair (coverage, risk) distinguishes them.

**The failure-code histogram**, so "it was wrong" becomes "it was wrong *this way*". Two methods at
the same accuracy failing under ``F06`` and ``F12`` respectively have nothing in common.

**The assumption audit rate.** `PROJECT_NOVELTY.md#N7` claims that machine-checkable validity
domains are a research artifact rather than hygiene; this is where the claim is cashed. The rate at
which a method ran outside its own declared domain is reported next to its score, because a good
score obtained mostly in ``RUNNABLE_DEGRADED`` is a different claim from the same score obtained
clean.

**Cost**, in the currencies the agent layer will later trade off, so that the expensive answer does
not look free.

Two structural refusals live in this module. A record cannot be built without stating what the
comparison does not establish, for the reason :class:`~usis.eval.mrep.ResultCard` gives. And a row
for a ``HYPOTHESIZED`` method is written, scored and kept - shadow mode means computed and never
surfaced, not computed and discarded - but :attr:`ComparisonRecord.surfaceable_rows` will not
return it, so a report drawing on the record cannot reach it by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from usis.compare.scoring import Outcome, outcome_counts
from usis.core.errors import UsisError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from usis.compare.risk_coverage import CurveEstimate, RiskCoverageCurve
    from usis.compare.scoring import ItemScore
    from usis.core.provenance import RunContext
    from usis.core.types import FailureCode, Maturity, TaskKind
    from usis.data.splits import Split
    from usis.eval.metrics.statistics import MetricResult

__all__ = ["ApplicabilityTally", "ComparisonRecord", "MethodRow", "RecordError"]


class RecordError(UsisError):
    """A comparison record is missing something a reader needs to judge it."""


@dataclass(frozen=True)
class ApplicabilityTally:
    """How often a method ran inside its own declared domain - N7's number.

    ``n_undecided`` counts runs where an assumption could be neither confirmed nor denied, and it
    is kept apart from ``n_violated`` on purpose: one is a statement about the method meeting the
    data, the other is a statement about the metadata we hold. Merging them would let a corpus with
    poor provenance read as a corpus full of violations.
    """

    n_runs: int = 0
    n_clean: int = 0
    n_degraded: int = 0
    n_refused: int = 0
    n_violated: int = 0
    n_undecided: int = 0
    violated_assumptions: Mapping[str, int] = field(default_factory=dict)
    undecided_assumptions: Mapping[str, int] = field(default_factory=dict)

    @property
    def violation_rate(self) -> float:
        """Share of runs in which a declared assumption was checked and found false."""
        return self.n_violated / self.n_runs if self.n_runs else 0.0

    @property
    def undecided_rate(self) -> float:
        """Share of runs in which a declared assumption could not be decided."""
        return self.n_undecided / self.n_runs if self.n_runs else 0.0

    @property
    def clean_rate(self) -> float:
        """Share of runs that satisfied every declared assumption."""
        return self.n_clean / self.n_runs if self.n_runs else 0.0

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view."""
        return {
            "n_runs": self.n_runs,
            "n_clean": self.n_clean,
            "n_degraded": self.n_degraded,
            "n_refused": self.n_refused,
            "n_violated": self.n_violated,
            "n_undecided": self.n_undecided,
            "violation_rate": self.violation_rate,
            "undecided_rate": self.undecided_rate,
            "clean_rate": self.clean_rate,
            "violated_assumptions": dict(self.violated_assumptions),
            "undecided_assumptions": dict(self.undecided_assumptions),
        }


@dataclass(frozen=True)
class MethodRow:
    """One method's line on the leaderboard, with everything needed to read the number.

    The failure histogram is kept three ways, because one code can arrive from two places and mean
    two things. ``F12_retrieval_miss`` is the clear case: a matcher *declares* it when it believes
    the substance is absent from the library, and a benchmark *scores* it when a substance that
    was present went unretrieved. A prediction and an observation summed into one bar is a bar
    nobody can read, so ``failures_declared`` and ``failures_scored`` are kept apart and
    ``failures`` is their per-item union - the histogram the project convention asks every
    evaluation run to emit.
    """

    method_key: str
    task: TaskKind
    maturity: Maturity
    surfaceable: bool
    scores: tuple[ItemScore, ...]
    curve: RiskCoverageCurve
    applicability: ApplicabilityTally
    metrics: tuple[MetricResult, ...] = ()
    estimates: tuple[CurveEstimate, ...] = ()
    failures: Mapping[FailureCode, int] = field(default_factory=dict)
    failures_declared: Mapping[FailureCode, int] = field(default_factory=dict)
    failures_scored: Mapping[FailureCode, int] = field(default_factory=dict)
    cost: Mapping[str, float] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def outcomes(self) -> dict[Outcome, int]:
        """The outcome histogram, with every member present even at zero."""
        return outcome_counts(self.scores)

    @property
    def n_counted(self) -> int:
        """Items this method was measured on - the coverage denominator."""
        return sum(1 for s in self.scores if s.outcome.is_counted)

    @property
    def n_answered(self) -> int:
        """Items the method committed to an answer on."""
        return sum(1 for s in self.scores if s.outcome.is_answered)

    @property
    def coverage(self) -> float:
        """Share of the corpus this method answered at all, before any threshold is applied."""
        return self.n_answered / self.n_counted if self.n_counted else 0.0

    @property
    def selective_accuracy(self) -> float:
        """Accuracy over the items it answered. Meaningless without :attr:`coverage` beside it."""
        answered = [s for s in self.scores if s.outcome.is_answered]
        if not answered:
            return 0.0
        return sum(1 for s in answered if s.outcome is Outcome.CORRECT) / len(answered)

    def metric(self, name: str) -> MetricResult | None:
        """One named metric, or ``None`` when this row does not carry it."""
        return next((m for m in self.metrics if m.name == name), None)

    def estimate(self, name: str) -> CurveEstimate | None:
        """One named curve estimate, or ``None`` when this row does not carry it."""
        return next((e for e in self.estimates if e.name == name), None)

    def to_dict(self, *, include_items: bool = True) -> dict[str, Any]:
        """A JSON-serialisable row. ``include_items`` keeps the per-item detail."""
        payload: dict[str, Any] = {
            "method_key": self.method_key,
            "task": str(self.task),
            "maturity": str(self.maturity),
            "surfaceable": self.surfaceable,
            "n_counted": self.n_counted,
            "n_answered": self.n_answered,
            "coverage": self.coverage,
            "selective_accuracy": self.selective_accuracy,
            "outcomes": {str(k): v for k, v in self.outcomes.items()},
            "metrics": [m.to_dict() for m in self.metrics],
            "estimates": [e.to_dict() for e in self.estimates],
            "risk_coverage": self.curve.to_dict(),
            "failure_codes": {str(k): v for k, v in self.failures.items()},
            "failure_codes_by_source": {
                "declared_by_method": {str(k): v for k, v in self.failures_declared.items()},
                "scored_by_benchmark": {str(k): v for k, v in self.failures_scored.items()},
            },
            "applicability": self.applicability.to_dict(),
            "cost": dict(self.cost),
            "errors": list(self.errors),
            "notes": list(self.notes),
        }
        if include_items:
            payload["items"] = [s.to_dict() for s in self.scores]
        return payload


@dataclass(frozen=True)
class ComparisonRecord:
    """One ``usis compare`` run: the task, the split, every method's row, and the caveats."""

    task_id: str
    task_kind: TaskKind
    run_context: RunContext
    rows: tuple[MethodRow, ...]
    dataset_ids: tuple[str, ...] = ()
    split: Split | None = None
    n_unscorable: int = 0
    n_items_presented: int = 0
    risk_target: float = 0.01
    coverage_target: float = 0.5
    bootstrap_samples: int = 2000
    config_hash: str = ""
    shows: tuple[str, ...] = ()
    does_not_show: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce the parts a leaderboard is usually published without."""
        if not self.rows:
            raise RecordError(
                f"{self.task_id}: no method produced a row. Either no registered method declares "
                f"task {self.task_kind}, or the selection filtered them all out - and a "
                "comparison of nothing should not be written to disk as though it were a result."
            )
        if not self.does_not_show:
            raise RecordError(
                f"{self.task_id}: a comparison record must state what it does NOT establish. A "
                "leaderboard is read as a ranking of methods in general, and it never is one."
            )

    @property
    def is_attributable(self) -> bool:
        """Whether the numbers can be traced to committed code."""
        return self.run_context.git_dirty == "false"

    @property
    def surfaceable_rows(self) -> tuple[MethodRow, ...]:
        """The rows a report conclusion may draw on. Shadow rows are not among them."""
        return tuple(r for r in self.rows if r.surfaceable)

    @property
    def shadow_rows(self) -> tuple[MethodRow, ...]:
        """The rows computed and scored but never surfaced. Kept, not discarded."""
        return tuple(r for r in self.rows if not r.surfaceable)

    def row(self, method_key: str) -> MethodRow | None:
        """One method's row by registry key."""
        return next((r for r in self.rows if r.method_key == method_key), None)

    def to_dict(self, *, include_items: bool = True) -> dict[str, Any]:
        """The machine-readable record. This is the deliverable, not the printed table."""
        split = self.split
        return {
            "task_id": self.task_id,
            "task_kind": str(self.task_kind),
            "dataset_ids": list(self.dataset_ids),
            "attributable": self.is_attributable,
            "run_context": self.run_context.to_dict(),
            "config_hash": self.config_hash,
            "n_items_presented": self.n_items_presented,
            "n_unscorable": self.n_unscorable,
            "risk_target": self.risk_target,
            "coverage_target": self.coverage_target,
            "bootstrap_samples": self.bootstrap_samples,
            "split": (
                None
                if split is None
                else {
                    "kind": str(split.kind),
                    "group_key": split.group_key,
                    "n_train": split.n_train,
                    "n_test": split.n_test,
                    "test_fraction": split.test_fraction,
                    "seed": split.seed,
                    "notes": list(split.notes),
                }
            ),
            "shows": list(self.shows),
            "does_not_show": list(self.does_not_show),
            "notes": list(self.notes),
            "rows": [r.to_dict(include_items=include_items) for r in self.rows],
        }

    def write(self, directory: Path, filename: str = "comparison.json") -> Path:
        """Write the record and the run context into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        self.run_context.write(directory)
        path = directory / filename
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return path
