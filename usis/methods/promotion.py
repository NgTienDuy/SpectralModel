"""The promotion gate - which reads evidence and does not promote anything.

There is deliberately **no function in this module that changes a maturity level.** That is the
strongest available form of ADR-0001 §S2 item 3 and of the brief's own rule that *promotion is an
ADR*. :func:`evaluate` returns a :class:`PromotionDecision` saying whether a method is *eligible*;
a human then writes the ADR and registers a new entry at the new level. A code path that flipped
the level would, sooner or later, be called in a loop.

The criteria have to be written down **before** the numbers are looked at, so
:class:`PromotionCriteria` refuses to exist without a pre-registration reference. This is the same
constraint the result-card writer already enforces one level up, for the same reason: criteria
chosen after seeing the result are not a gate, they are a description.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from usis.core.errors import UsisError
from usis.data.splits import SplitKind
from usis.methods.shadow import ShadowError, ShadowLedger, ShadowScore

__all__ = [
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionError",
    "evaluate",
    "evaluate_all",
]


class PromotionError(UsisError):
    """A promotion gate was asked to evaluate without the pre-conditions a gate needs."""


@dataclass(frozen=True)
class PromotionCriteria:
    """Pre-registered thresholds a shadow method must clear to become *eligible* for promotion.

    ``required_split_kinds`` exists because accuracy on a random split is not evidence of anything
    the project cares about; the default is empty only so that a caller must state the splits
    deliberately, and :func:`evaluate` reports an empty requirement as a weakness of the gate
    rather than silently accepting it.
    """

    min_independent_n: int
    min_margin: float
    preregistration_ref: str
    required_split_kinds: frozenset[SplitKind] = frozenset()
    min_datasets: int = 1
    note: str = ""

    def __post_init__(self) -> None:
        """Refuse a gate with no pre-registration, or with thresholds that cannot bind."""
        if not self.preregistration_ref.strip():
            raise PromotionError(
                "PromotionCriteria requires preregistration_ref: the path or commit of the "
                "document that fixed these thresholds before the numbers were seen. Criteria "
                "written afterwards describe a result; they do not gate it."
            )
        if self.min_independent_n < 1:
            raise PromotionError(
                "min_independent_n must be at least 1, and ADR-0001 §S2 is explicit that a "
                "promotion on a few dozen labelled cases is not a promotion - set it to the "
                "number the pre-registration argued for."
            )


@dataclass(frozen=True)
class PromotionDecision:
    """Whether a shadow method is eligible, and every reason it is or is not.

    ``eligible`` never means promoted. ``requires_adr`` is always ``True`` and is in the record so
    that a downstream reader cannot mistake eligibility for a state change.
    """

    method_key: str
    eligible: bool
    score: ShadowScore | None
    criteria: PromotionCriteria
    reasons: tuple[str, ...]
    requires_adr: bool = True

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view for the promotion record."""
        return {
            "method_key": self.method_key,
            "eligible": self.eligible,
            "requires_adr": self.requires_adr,
            "preregistration_ref": self.criteria.preregistration_ref,
            "criteria": {
                "min_independent_n": self.criteria.min_independent_n,
                "min_margin": self.criteria.min_margin,
                "required_split_kinds": [
                    str(k) for k in sorted(self.criteria.required_split_kinds)
                ],
                "min_datasets": self.criteria.min_datasets,
            },
            "score": self.score.to_dict() if self.score is not None else None,
            "reasons": list(self.reasons),
        }


def evaluate(
    ledger: ShadowLedger,
    method_key: str,
    criteria: PromotionCriteria,
    incumbent_key: str | None = None,
) -> PromotionDecision:
    """Read the independently-answered subset and decide eligibility. Never promotes.

    A method with no independently-answered runs is not an error here - it is the expected state
    for a method that has only ever run in production - so the missing evidence is reported as a
    reason rather than raised.
    """
    try:
        score = ledger.score(method_key, incumbent_key)
    except ShadowError as exc:
        return PromotionDecision(
            method_key=method_key,
            eligible=False,
            score=None,
            criteria=criteria,
            reasons=(str(exc),),
        )
    reasons: list[str] = []
    if score.n < criteria.min_independent_n:
        reasons.append(
            f"n = {score.n} independently-answered runs, gate requires "
            f"{criteria.min_independent_n} (ADR-0001 §S2 item 3)"
        )
    if score.margin < criteria.min_margin:
        reasons.append(
            f"margin over {score.incumbent_key} is {score.margin:+.4f} "
            f"({score.shadow_accuracy:.4f} vs {score.incumbent_accuracy:.4f}), gate requires "
            f"{criteria.min_margin:+.4f}"
        )
    missing_splits = sorted(criteria.required_split_kinds - set(score.split_kinds))
    if missing_splits:
        reasons.append(
            f"no independently-answered runs on {[str(k) for k in missing_splits]}; the evidence "
            f"covers {[str(k) for k in score.split_kinds]}"
        )
    if len(score.datasets) < criteria.min_datasets:
        reasons.append(
            f"evidence spans {len(score.datasets)} dataset(s) {list(score.datasets)}, gate "
            f"requires {criteria.min_datasets}"
        )
    if not criteria.required_split_kinds:
        reasons.append(
            "WEAK GATE: no split kinds were required, so this evidence may be entirely from "
            "random splits - which the project treats as a smoke test, not a result"
        )
    eligible = not reasons
    if eligible:
        reasons.append(
            f"every pre-registered threshold met on n = {score.n} independently-answered runs; "
            "eligible for promotion by ADR, which must still be written"
        )
    return PromotionDecision(
        method_key=method_key,
        eligible=eligible,
        score=score,
        criteria=criteria,
        reasons=tuple(reasons),
    )


def evaluate_all(
    ledger: ShadowLedger,
    criteria: PromotionCriteria,
    method_keys: Sequence[str] | None = None,
) -> tuple[PromotionDecision, ...]:
    """Evaluate every shadow method in the ledger against one set of criteria."""
    keys = tuple(method_keys) if method_keys is not None else ledger.method_keys()
    return tuple(evaluate(ledger, key, criteria) for key in keys)
