"""Per-item outcomes: the unit the comparison harness counts, and what it refuses to count.

One spectrum, one method, one outcome. Everything the leaderboard reports - accuracy, the
risk-coverage curve, the failure histogram - is an aggregate over these, so the distinctions drawn
here are the distinctions the whole comparison can express.

Four of them are load-bearing:

**Abstention is not a wrong answer.** :attr:`Outcome.ABSTAINED` is its own member. A method that
declines is not accepted at any threshold, so it never contributes an error - but it stays in the
coverage denominator, so declining is not free either. That is the project's stated position
("abstention is a first-class output") made arithmetic.

**A refusal is not an abstention.** :attr:`Outcome.REFUSED` means the applicability audit stopped
the run before it started: the method declared a domain and these inputs are outside it. Counted
separately because the two say different things to a reader - one is the method's judgement about
this spectrum, the other is its declared domain meeting the corpus.

**A crash is not a refusal.** :attr:`Outcome.ERRORED` is a defect, and folding it into either of
the above would let a broken method look merely cautious.

**An item with no independent answer is not scored at all.** :attr:`Outcome.UNSCORABLE` leaves the
item out of every numerator *and* every denominator, and the record states how many there were.
This is ADR-0001 §S2 applied to the harness rather than only to shadow mode: scoring against the
portfolio's own conclusion measures imitation, and no quantity of it adds up to evidence about
correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from usis.core.errors import UsisError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from usis.core.types import FailureCode

__all__ = ["ItemScore", "Outcome", "ScoringError", "outcome_counts"]


class ScoringError(UsisError):
    """An item score is internally inconsistent, or was asked for something it cannot support."""


class Outcome(StrEnum):
    """What happened when one method met one item."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    ABSTAINED = "abstained"
    REFUSED = "refused"
    ERRORED = "errored"
    UNSCORABLE = "unscorable"

    @property
    def is_answered(self) -> bool:
        """Whether the method committed to an answer that could be right or wrong.

        The only two outcomes eligible to enter the accepted set of a risk-coverage curve. A
        method cannot be charged with an error for an answer it did not give.
        """
        return self in (Outcome.CORRECT, Outcome.INCORRECT)

    @property
    def is_counted(self) -> bool:
        """Whether the item is part of the corpus this method was measured on.

        Everything except ``UNSCORABLE``. A refusal and a crash both stay in the denominator: a
        method that refuses half the corpus has coverage at most one half, and that is the number
        a reader needs. Refusing must never be a way to win.
        """
        return self is not Outcome.UNSCORABLE


@dataclass(frozen=True)
class ItemScore:
    """One method's outcome on one item, with the confidence the selective curve ranks on.

    ``confidence`` is what the method itself claims, on a scale where larger means *more willing
    to be held to this answer*. It is required for an answered item and forbidden to be invented
    for one that was not answered: a curve built over fabricated confidences would rank items by
    an artefact of the harness rather than by anything the method said.

    ``value`` carries the task's own continuous metric when it has one - an absolute error, a
    similarity, an R^2 contribution. It is separate from correctness because a quantification task
    has no notion of "correct" and a retrieval task has no notion of "residual", and collapsing
    them into one float is how a leaderboard ends up comparing two different things in one column.
    """

    item_id: str
    outcome: Outcome
    confidence: float | None = None
    value: float | None = None
    failure_codes: tuple[FailureCode, ...] = ()
    reference: str | None = None
    predicted: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        """Reject the two score shapes that would make an aggregate lie."""
        if self.outcome.is_answered and self.confidence is None:
            raise ScoringError(
                f"{self.item_id}: outcome {self.outcome} was answered but carries no confidence. "
                "The risk-coverage curve ranks accepted items by confidence; an answered item "
                "without one cannot be placed on it, and defaulting it would rank the item by a "
                "number the method never claimed."
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ScoringError(
                f"{self.item_id}: confidence must be in [0, 1], got {self.confidence}"
            )

    @property
    def is_error(self) -> bool:
        """Whether an accepted answer here would be a mistake."""
        return self.outcome is Outcome.INCORRECT

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable row, so a record keeps its per-item detail, not only the summary."""
        return {
            "item_id": self.item_id,
            "outcome": str(self.outcome),
            "confidence": self.confidence,
            "value": self.value,
            "failure_codes": [str(code) for code in self.failure_codes],
            "reference": self.reference,
            "predicted": self.predicted,
            "detail": self.detail,
        }


def outcome_counts(scores: Sequence[ItemScore]) -> dict[Outcome, int]:
    """Count outcomes, with every member present so a zero is visible rather than absent.

    An absent key and a zero read the same way in a JSON file only until someone filters on it.
    Here ``errored: 0`` is a statement that nothing crashed, which is worth making.
    """
    counts = dict.fromkeys(Outcome, 0)
    for score in scores:
        counts[score.outcome] += 1
    return counts
