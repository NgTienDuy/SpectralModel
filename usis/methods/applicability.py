"""Turning a set of assumption checks into a verdict, and the policy behind it.

The policy is three lines long and one of them is a real decision, recorded in
`PROJECT_DECISIONS.md` D13:

1. a **hard** assumption that was checked and **violated** refuses the run;
2. a **soft** violation, or any assumption that could not be decided, runs *and is declared*;
3. everything satisfied runs clean.

Line 2 is the decision. A hard assumption whose status is ``UNKNOWN`` could equally be made to
refuse, and that reading is defensible - but it would refuse almost every real spectrum we hold,
because ``resolution_fwhm`` is absent from most records and because a silent processing chain is
the norm rather than the exception. The brief's input contract already answers the question the
other way: missing metadata must *widen the posterior*, not crash the pipeline. So an undecidable
precondition runs in ``RUNNABLE_DEGRADED``, carries the thing that would decide it, and forbids the
result being presented as clean.

What this module deliberately does **not** do is score. The violation *rate* over a benchmark is
N7's actual contribution and belongs to the comparison harness; here we only make each individual
verdict a typed, serialisable object that the harness can count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from usis.core.types import FailureCode
from usis.methods.assumptions import Assumption, AssumptionCheck, AssumptionStatus, check_all
from usis.methods.inputs import MethodInputs

__all__ = ["Applicability", "Verdict", "assess"]


class Verdict(StrEnum):
    """Whether a method may run on these inputs, and on what terms."""

    RUNNABLE = "runnable"
    RUNNABLE_DEGRADED = "runnable_degraded"
    REFUSED = "refused"

    @property
    def may_run(self) -> bool:
        """Whether execution is permitted at all."""
        return self is not Verdict.REFUSED


@dataclass(frozen=True)
class Applicability:
    """The full assumption audit for one method on one set of inputs.

    Every check is retained, satisfied ones included. A report that lists only the problems cannot
    be distinguished from a report produced by a method that declared no assumptions at all, and
    that difference is the entire point of N7.
    """

    verdict: Verdict
    checks: tuple[AssumptionCheck, ...]
    reason: str

    @property
    def violations(self) -> tuple[AssumptionCheck, ...]:
        """Checks that were decided and failed."""
        return tuple(c for c in self.checks if c.status is AssumptionStatus.VIOLATED)

    @property
    def undecided(self) -> tuple[AssumptionCheck, ...]:
        """Checks that could not be decided, whether for want of metadata or of a procedure."""
        return tuple(
            c
            for c in self.checks
            if c.status in (AssumptionStatus.UNKNOWN, AssumptionStatus.NOT_CHECKABLE)
        )

    @property
    def failure_codes(self) -> tuple[FailureCode, ...]:
        """Distinct failure codes implicated by violations, for the run's histogram.

        Violations only. An undecided check is not evidence that the failure occurred, and
        counting it as one would inflate every histogram with the state of our metadata.
        """
        seen: dict[FailureCode, None] = {}
        for check in self.violations:
            seen.setdefault(check.failure_code, None)
        return tuple(seen)

    @property
    def evidence_needed(self) -> tuple[str, ...]:
        """What would settle each undecided check - section 9 of the Sample Report, in miniature."""
        return tuple(c.evidence_needed for c in self.undecided if c.evidence_needed)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view for the comparison record."""
        return {
            "verdict": str(self.verdict),
            "reason": self.reason,
            "checks": [c.to_dict() for c in self.checks],
            "failure_codes": [str(code) for code in self.failure_codes],
        }


def assess(assumptions: Sequence[Assumption], inputs: MethodInputs) -> Applicability:
    """Check every declared assumption and apply the D13 policy to the result."""
    checks = check_all(assumptions, inputs)
    hard_violations = [c for c in checks if c.status is AssumptionStatus.VIOLATED and c.hard]
    soft_violations = [c for c in checks if c.status is AssumptionStatus.VIOLATED and not c.hard]
    undecided = [
        c for c in checks if c.status in (AssumptionStatus.UNKNOWN, AssumptionStatus.NOT_CHECKABLE)
    ]
    if hard_violations:
        return Applicability(
            verdict=Verdict.REFUSED,
            checks=checks,
            reason="refused: " + "; ".join(f"{c.assumption} - {c.detail}" for c in hard_violations),
        )
    if soft_violations or undecided:
        parts = [f"{c.assumption} violated ({c.detail})" for c in soft_violations]
        parts += [f"{c.assumption} {c.status} ({c.detail})" for c in undecided]
        return Applicability(
            verdict=Verdict.RUNNABLE_DEGRADED,
            checks=checks,
            reason="runnable, declared: " + "; ".join(parts),
        )
    if not checks:
        return Applicability(
            verdict=Verdict.RUNNABLE,
            checks=(),
            reason="no assumptions declared - which is itself a claim, and a strong one",
        )
    return Applicability(
        verdict=Verdict.RUNNABLE,
        checks=checks,
        reason=f"all {len(checks)} declared assumptions satisfied",
    )
