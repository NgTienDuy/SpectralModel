"""Shadow mode, with the ADR-0001 §S2 fix built into the type system.

Brief §6.1 says hypothesised methods *"run on every analysis, are logged and scored against the
accepted answer"*. ADR-0001 §S2 rates that high severity, because in routine use there is no ground
truth: the accepted answer **is** the output of the established portfolio. A shadow method scored on
agreement with the incumbent is scored on imitating it, so a shadow method that is *better* scores
*worse* - and the promotion gate inverts exactly when it matters. It is the "gold standard is the
model" trap, and it fails silently: the system keeps working and keeps promoting the wrong things.

The fix, as adopted:

1. shadow scoring counts **only** runs with an independent answer - labelled benchmark data, a
   re-measurement that resolved the case, or an orthogonal modality that settled it;
2. on unlabelled runs, disagreement with the portfolio is recorded as a **flag for review** and
   never as a score;
3. the promotion gate reads only the independently-answered subset **and states its size**.

Here, (1) is a property of :class:`AnswerSource`: :meth:`ShadowRecord.is_scorable` is false for
consensus and for absent answers, and :meth:`ShadowLedger.score` never looks at a record that is
not scorable. (2) is :meth:`ShadowLedger.flags`, which returns records and no numbers. (3) lives in
:mod:`usis.methods.promotion`, which reads a :class:`ShadowScore` that carries ``n`` and cannot be
constructed without it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from usis.core.errors import UsisError
from usis.data.splits import SplitKind

__all__ = [
    "INDEPENDENT_SOURCES",
    "AnswerSource",
    "ShadowError",
    "ShadowLedger",
    "ShadowRecord",
    "ShadowScore",
]


class ShadowError(UsisError):
    """A shadow-mode operation was asked for something its evidence cannot support."""


class AnswerSource(StrEnum):
    """Where the answer a shadow run is compared against came from.

    The distinction between the first three members and the last two is the whole of the §S2 fix.
    The first three are independent of the system's own output; the last two are not, and no
    quantity of them adds up to evidence about correctness.
    """

    LABELLED_BENCHMARK = "labelled_benchmark"
    RESOLVED_REMEASUREMENT = "resolved_remeasurement"
    ORTHOGONAL_MODALITY = "orthogonal_modality"
    PORTFOLIO_CONSENSUS = "portfolio_consensus"
    NONE = "none"

    @property
    def is_independent(self) -> bool:
        """Whether this answer was established without using the system's own conclusion."""
        return self in INDEPENDENT_SOURCES


#: The only answer sources a shadow score may be computed from.
INDEPENDENT_SOURCES: frozenset[AnswerSource] = frozenset(
    {
        AnswerSource.LABELLED_BENCHMARK,
        AnswerSource.RESOLVED_REMEASUREMENT,
        AnswerSource.ORTHOGONAL_MODALITY,
    }
)


@dataclass(frozen=True)
class ShadowRecord:
    """One shadow run: what the shadow said, what the incumbent said, and who else knew.

    ``reference_answer`` is the independent answer when there is one and ``None`` otherwise. It is
    a separate field from ``incumbent_answer`` on purpose: collapsing them into one "accepted
    answer" field is the data-model shape that makes §S2 easy to commit, and keeping them apart
    makes it impossible to commit by accident.
    """

    method_key: str
    incumbent_key: str
    spectrum_id: str
    run_id: str
    answer_source: AnswerSource
    shadow_answer: str | None = None
    incumbent_answer: str | None = None
    reference_answer: str | None = None
    split_kind: SplitKind | None = None
    dataset_id: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        """Reject the two record shapes that would make the ledger lie."""
        if self.answer_source.is_independent and self.reference_answer is None:
            raise ShadowError(
                f"{self.method_key} on {self.spectrum_id}: answer_source "
                f"{self.answer_source} claims an independent answer but reference_answer is None. "
                "Record AnswerSource.NONE instead - a claimed-but-absent reference is worse than "
                "no reference, because it counts."
            )
        if not self.answer_source.is_independent and self.reference_answer is not None:
            raise ShadowError(
                f"{self.method_key} on {self.spectrum_id}: a reference_answer was supplied with "
                f"answer_source {self.answer_source}. If the answer really is independent, name "
                "which of the three independent sources it came from; if it is the portfolio's "
                "own conclusion, it belongs in incumbent_answer (ADR-0001 §S2)."
            )

    @property
    def is_scorable(self) -> bool:
        """Whether this record may contribute to a score at all."""
        return self.answer_source.is_independent and self.reference_answer is not None

    @property
    def shadow_correct(self) -> bool | None:
        """Whether the shadow matched the independent answer, or ``None`` if unscorable."""
        if not self.is_scorable:
            return None
        return self.shadow_answer == self.reference_answer

    @property
    def incumbent_correct(self) -> bool | None:
        """Whether the incumbent matched the independent answer, or ``None`` if unscorable."""
        if not self.is_scorable:
            return None
        return self.incumbent_answer == self.reference_answer

    @property
    def disagrees_with_incumbent(self) -> bool:
        """Whether the shadow and the incumbent gave different answers."""
        return self.shadow_answer != self.incumbent_answer

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable row for the shadow log."""
        return {
            "method_key": self.method_key,
            "incumbent_key": self.incumbent_key,
            "spectrum_id": self.spectrum_id,
            "run_id": self.run_id,
            "answer_source": str(self.answer_source),
            "shadow_answer": self.shadow_answer,
            "incumbent_answer": self.incumbent_answer,
            "reference_answer": self.reference_answer,
            "split_kind": str(self.split_kind) if self.split_kind is not None else None,
            "dataset_id": self.dataset_id,
            "scorable": self.is_scorable,
            "note": self.note,
        }


@dataclass(frozen=True)
class ShadowScore:
    """A shadow method's performance on the independently-answered subset, and that subset's size.

    ``n`` is not a diagnostic field, it is part of the result. ADR-0001 §S2 item 3: *a promotion on
    n = 30 labelled cases is not a promotion*. Reporting accuracy without the count is how it
    becomes one.
    """

    method_key: str
    incumbent_key: str
    n: int
    shadow_correct: int
    incumbent_correct: int
    n_flagged_unscorable: int
    split_kinds: tuple[SplitKind, ...]
    datasets: tuple[str, ...]

    @property
    def shadow_accuracy(self) -> float:
        """Accuracy on the independently-answered subset."""
        return self.shadow_correct / self.n

    @property
    def incumbent_accuracy(self) -> float:
        """The incumbent's accuracy on the same subset - the only fair comparison."""
        return self.incumbent_correct / self.n

    @property
    def margin(self) -> float:
        """Shadow accuracy minus incumbent accuracy on the shared, independently-answered subset."""
        return self.shadow_accuracy - self.incumbent_accuracy

    @property
    def evidence_ratio(self) -> float:
        """Scorable runs as a fraction of all runs, scorable and flagged.

        Worth reading alongside the accuracy: a ratio near zero means the method has been running
        for a long time and learning almost nothing that counts, which is a fact about the
        deployment rather than about the method.
        """
        total = self.n + self.n_flagged_unscorable
        return self.n / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view."""
        return {
            "method_key": self.method_key,
            "incumbent_key": self.incumbent_key,
            "n_independent": self.n,
            "shadow_accuracy": self.shadow_accuracy,
            "incumbent_accuracy": self.incumbent_accuracy,
            "margin": self.margin,
            "n_flagged_unscorable": self.n_flagged_unscorable,
            "evidence_ratio": self.evidence_ratio,
            "split_kinds": [str(kind) for kind in self.split_kinds],
            "datasets": list(self.datasets),
        }


@dataclass
class ShadowLedger:
    """Every shadow run ever recorded, and the two very different things one can ask of them."""

    records: list[ShadowRecord] = field(default_factory=list)

    def add(self, record: ShadowRecord) -> ShadowRecord:
        """Append one record."""
        self.records.append(record)
        return record

    def extend(self, records: Iterable[ShadowRecord]) -> None:
        """Append many records."""
        self.records.extend(records)

    def __len__(self) -> int:
        """Total number of recorded shadow runs, scorable or not."""
        return len(self.records)

    def __iter__(self) -> Iterator[ShadowRecord]:
        """Iterate over every record in insertion order."""
        return iter(self.records)

    def for_method(self, method_key: str) -> tuple[ShadowRecord, ...]:
        """Every record for one shadow method."""
        return tuple(r for r in self.records if r.method_key == method_key)

    def scorable(self, method_key: str) -> tuple[ShadowRecord, ...]:
        """The records for one method that may contribute to a score."""
        return tuple(r for r in self.for_method(method_key) if r.is_scorable)

    def flags(self, method_key: str) -> tuple[ShadowRecord, ...]:
        """Unscorable runs where the shadow disagreed with the portfolio.

        These are review items, deliberately returned as records rather than as a number. A count
        of disagreements looks like a score, gets compared across methods, and is then quoted as
        evidence - which is the §S2 failure arriving by a slightly longer route.
        """
        return tuple(
            r
            for r in self.for_method(method_key)
            if not r.is_scorable and r.disagrees_with_incumbent
        )

    def score(self, method_key: str, incumbent_key: str | None = None) -> ShadowScore:
        """Score one shadow method on its independently-answered runs only.

        Args:
            method_key: the shadow method.
            incumbent_key: restrict to runs against this incumbent. Left ``None`` when a single
                incumbent produced every record; required when several did, since accuracies
                measured against different portfolios are not comparable.

        Raises:
            ShadowError: if there are no independently-answered runs, or if the records span
                several incumbents and none was named.
        """
        scorable = self.scorable(method_key)
        if incumbent_key is not None:
            scorable = tuple(r for r in scorable if r.incumbent_key == incumbent_key)
        flagged = len(self.for_method(method_key)) - len(self.scorable(method_key))
        if not scorable:
            raise ShadowError(
                f"{method_key} has 0 independently-answered runs out of "
                f"{len(self.for_method(method_key))} recorded. Agreement with the portfolio is "
                "not evidence of correctness (ADR-0001 §S2): the portfolio's answer is what the "
                "shadow method is a candidate to replace. Score it on labelled benchmark data, a "
                "re-measurement that resolved the case, or an orthogonal modality - or do not "
                "score it."
            )
        incumbents = {r.incumbent_key for r in scorable}
        if incumbent_key is None and len(incumbents) > 1:
            raise ShadowError(
                f"{method_key} was shadowing several incumbents ({sorted(incumbents)}); pass "
                "incumbent_key. A margin computed against a mixture of portfolios is not a margin "
                "over any of them."
            )
        split_kinds = tuple(sorted({r.split_kind for r in scorable if r.split_kind is not None}))
        datasets = tuple(sorted({r.dataset_id for r in scorable if r.dataset_id}))
        return ShadowScore(
            method_key=method_key,
            incumbent_key=incumbent_key or next(iter(incumbents)),
            n=len(scorable),
            shadow_correct=sum(1 for r in scorable if r.shadow_correct),
            incumbent_correct=sum(1 for r in scorable if r.incumbent_correct),
            n_flagged_unscorable=flagged,
            split_kinds=split_kinds,
            datasets=datasets,
        )

    def method_keys(self) -> tuple[str, ...]:
        """Every shadow method that appears in the ledger, in key order."""
        return tuple(sorted({r.method_key for r in self.records}))

    def to_rows(self) -> list[dict[str, Any]]:
        """Every record as a JSON-serialisable row, for writing the shadow log to disk."""
        return [r.to_dict() for r in self.records]

    @classmethod
    def of(cls, records: Sequence[ShadowRecord]) -> ShadowLedger:
        """A ledger holding ``records``."""
        return cls(records=list(records))
