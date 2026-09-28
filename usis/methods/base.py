"""The ``Method`` protocol - the central abstraction of the repository.

Brief §6.1: *everything that turns a spectrum into a claim implements one interface, declares
machine-readable assumptions, and is registered with a maturity level.* Three properties of this
module are enforced rather than documented, because each of them is a way results go wrong quietly:

**A method cannot run outside its declared domain.** :func:`execute` is the only sanctioned entry
point. It computes applicability first and raises
:class:`~usis.core.errors.AssumptionViolationError` on a refusal, so a caller cannot reach ``run``
by forgetting to check. Hard constraints enforced at inference rather than learned, as the project
conventions require.

**A method cannot report its own audit.** ``run`` returns a result; ``execute`` overwrites its
``applicability`` field with the audit it computed itself. A method that returned a flattering
audit for its own inputs would be the same failure as a model that reports its own accuracy.

**A method cannot make a causal claim.** ``ClaimType.CAUSAL_CLAIM`` on a result raises with
``F18``. The type exists so that an attempt is a typed violation instead of an untracked sentence,
and this is where the type earns that.

The protocol here is structural, matching the brief's shape. Real methods subclass
:class:`MethodBase`, which implements ``applicable`` once, in terms of the declared assumptions, so
that no method can implement its own applicability policy.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from usis.core.errors import AssumptionViolationError, UsisError
from usis.core.provenance import RunContext
from usis.core.runlog import RunLog
from usis.core.types import (
    BeliefState,
    ClaimType,
    Evidence,
    FailureCode,
    IntensityKind,
    Maturity,
    Modality,
    TaskKind,
)
from usis.methods.applicability import Applicability, Verdict, assess
from usis.methods.assumptions import (
    Assumption,
    RequiresIntensityKind,
    RequiresLibraryCoverage,
    RequiresMetadata,
    RequiresModality,
)
from usis.methods.inputs import MethodInputs

__all__ = [
    "SURFACEABLE_MATURITIES",
    "Citation",
    "CostModel",
    "Method",
    "MethodBase",
    "MethodError",
    "MethodResult",
    "Requirements",
    "execute",
    "surfaceable_results",
]

#: Maturity levels whose results may appear in a report conclusion. ``HYPOTHESIZED`` is absent by
#: construction - that is what shadow mode means - and so are ``REFUTED`` and ``DEPRECATED``, whose
#: entries are retained for the record and not for use.
SURFACEABLE_MATURITIES: frozenset[Maturity] = frozenset({Maturity.ESTABLISHED, Maturity.EMERGING})


class MethodError(UsisError):
    """A method or its declaration is structurally invalid."""


@dataclass(frozen=True)
class Citation:
    """Where a method comes from.

    ``internal_ref`` is for our own conjectures - ``"PROJECT_NOVELTY.md#N11"`` - and is not a
    lesser citation: it is what makes a hypothesised method traceable to the place its claim is
    written down and can be argued with. At least one field must be set, checked by the registry,
    because "no provenance" is how a conjecture becomes folklore.
    """

    title: str
    doi: str | None = None
    url: str | None = None
    internal_ref: str | None = None
    note: str = ""

    @property
    def is_attributed(self) -> bool:
        """Whether this citation points anywhere at all."""
        return bool(self.doi or self.url or self.internal_ref)

    def __str__(self) -> str:
        """A one-line rendering for a report or a leaderboard row."""
        where = self.doi or self.url or self.internal_ref or "unattributed"
        return f"{self.title} ({where})"


@dataclass(frozen=True)
class CostModel:
    """Declared cost, in the three currencies the agent layer will later have to trade off.

    ``instrument_time_s`` is not padding for M8: an action that asks for a longer acquisition
    spends a real resource, and a comparison that reports only compute makes the expensive answer
    look free.
    """

    compute_seconds_per_spectrum: float = 0.0
    peak_memory_mb: float = 0.0
    instrument_time_s: float = 0.0
    needs_gpu: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view."""
        return {
            "compute_seconds_per_spectrum": self.compute_seconds_per_spectrum,
            "peak_memory_mb": self.peak_memory_mb,
            "instrument_time_s": self.instrument_time_s,
            "needs_gpu": self.needs_gpu,
            "note": self.note,
        }


@dataclass(frozen=True)
class Requirements:
    """Declarative shorthand that **expands into assumptions** rather than being checked separately.

    Brief §6.1 lists ``requires: Requirements`` and ``assumptions: list[Assumption]`` as separate
    fields. Two mechanisms enforcing the same class of precondition is how one of them ends up
    unchecked, so this one is sugar: :meth:`as_assumptions` turns it into ordinary assumptions and
    there is exactly one checking path. The field survives because it makes the common declarations
    short, and short declarations are the ones people actually write.
    """

    modalities: frozenset[Modality] = frozenset()
    intensity_kinds: frozenset[IntensityKind] = frozenset()
    metadata_fields: tuple[str, ...] = ()
    reference_library: bool = False

    def as_assumptions(self) -> tuple[Assumption, ...]:
        """Expand into the assumption objects the runtime checks."""
        expanded: list[Assumption] = []
        if self.modalities:
            expanded.append(RequiresModality(self.modalities))
        if self.intensity_kinds:
            expanded.append(RequiresIntensityKind(self.intensity_kinds))
        if self.metadata_fields:
            expanded.append(RequiresMetadata(self.metadata_fields))
        if self.reference_library:
            expanded.append(RequiresLibraryCoverage())
        return tuple(expanded)


@dataclass(frozen=True)
class MethodResult:
    """What a method returns: the answer, its audit, and what it cost to get.

    ``belief`` is the identity/structure answer and is a :class:`~usis.core.types.BeliefState`, so
    the "no bare answers" convention is carried by the type rather than by review. ``values`` is
    for the numeric outputs that are not identities - a verdict score, a concentration, a detected
    band count - and every one of them is named.
    """

    method_id: str
    version: str
    task: TaskKind
    maturity: Maturity
    applicability: Applicability
    belief: BeliefState | None = None
    values: Mapping[str, float] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()
    failure_codes: tuple[FailureCode, ...] = ()
    cost_incurred: Mapping[str, float] = field(default_factory=dict)
    claim_type: ClaimType = ClaimType.CORRELATION
    notes: str = ""

    @property
    def method_key(self) -> str:
        """The registry key this result is attributable to."""
        return f"{self.method_id}@{self.version}"

    @property
    def surfaceable(self) -> bool:
        """Whether this result may appear in a report conclusion.

        Computed, never assigned. Two conditions, and both are structural: the maturity level must
        be one that may surface at all, and the run must not have been degraded by an assumption
        the method declared and could not satisfy.
        """
        return (
            self.maturity in SURFACEABLE_MATURITIES
            and self.applicability.verdict is Verdict.RUNNABLE
        )

    @property
    def all_failure_codes(self) -> tuple[FailureCode, ...]:
        """The method's own codes plus those implied by its assumption violations."""
        seen: dict[FailureCode, None] = {}
        for code in (*self.failure_codes, *self.applicability.failure_codes):
            seen.setdefault(code, None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view for the comparison record."""
        return {
            "method_key": self.method_key,
            "task": str(self.task),
            "maturity": str(self.maturity),
            "surfaceable": self.surfaceable,
            "claim_type": str(self.claim_type),
            "values": dict(self.values),
            "labels": dict(self.labels),
            "failure_codes": [str(code) for code in self.all_failure_codes],
            "cost_incurred": dict(self.cost_incurred),
            "applicability": self.applicability.to_dict(),
            "notes": self.notes,
        }


@runtime_checkable
class Method(Protocol):
    """The structural interface of brief §6.1. Implement it by subclassing :class:`MethodBase`."""

    method_id: str
    version: str
    task: TaskKind
    maturity: Maturity
    provenance: Citation | None
    assumptions: tuple[Assumption, ...]
    requires: Requirements
    cost: CostModel

    def applicable(self, inputs: MethodInputs) -> Applicability:
        """Can this run here, should it, and which assumptions are violated?"""
        ...

    def run(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Result, uncertainty, evidence chain, assumption audit and cost actually incurred."""
        ...


class MethodBase(ABC):
    """Base class implementing everything a method should not be free to reimplement.

    A subclass sets the declaration fields as class attributes and implements
    :meth:`compute`. It does not implement ``applicable``, does not decide what a violation means,
    and does not fill in its own audit - all three are the runtime's job, and a method that could
    override them could exempt itself from the checks that make the comparison meaningful.
    """

    method_id: str
    version: str
    task: TaskKind
    maturity: Maturity
    provenance: Citation | None = None
    assumptions: tuple[Assumption, ...] = ()
    requires: Requirements = Requirements()
    cost: CostModel = CostModel()

    @property
    def method_key(self) -> str:
        """``id@version``: the registry key, and what every result is attributed to."""
        return f"{self.method_id}@{self.version}"

    @property
    def all_assumptions(self) -> tuple[Assumption, ...]:
        """The expanded requirements followed by the explicitly declared assumptions."""
        return (*self.requires.as_assumptions(), *self.assumptions)

    def applicable(self, inputs: MethodInputs) -> Applicability:
        """Apply the one applicability policy to this method's declared domain."""
        return assess(self.all_assumptions, inputs)

    def run(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Check applicability, compute, and attach the audit. Prefer :func:`execute`."""
        return execute(self, inputs, ctx)

    @abstractmethod
    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Do the actual work. Called only after applicability permitted the run."""

    def _result(self, **kwargs: Any) -> MethodResult:
        """Build a result carrying this method's identity, for use inside :meth:`compute`.

        ``applicability`` is deliberately absent from the signature: :func:`execute` attaches the
        audit it computed, so a method has no way to describe its own domain compliance.
        """
        placeholder = Applicability(
            verdict=Verdict.RUNNABLE,
            checks=(),
            reason="placeholder, replaced by execute() with the audit it computed",
        )
        return MethodResult(
            method_id=self.method_id,
            version=self.version,
            task=self.task,
            maturity=self.maturity,
            applicability=placeholder,
            **kwargs,
        )


def execute(
    method: MethodBase,
    inputs: MethodInputs,
    ctx: RunContext,
    log: RunLog | None = None,
) -> MethodResult:
    """Run ``method`` on ``inputs`` under the applicability policy. The only sanctioned entry point.

    Raises:
        AssumptionViolationError: if applicability refused the run. The message names every hard
            violation with its observed value, because "not applicable" without the number is not
            actionable.
        MethodError: if the method returned a result attributed to a different method, or a result
            carrying a causal claim.
    """
    applicability = method.applicable(inputs)
    if log is not None:
        log.event(
            "method_applicability",
            method_key=method.method_key,
            spectrum_ids=list(inputs.spectrum_ids),
            **applicability.to_dict(),
        )
    if not applicability.verdict.may_run:
        raise AssumptionViolationError(
            f"{method.method_key} is not applicable to "
            f"{list(inputs.spectrum_ids)}: {applicability.reason}"
        )
    started = time.perf_counter()
    result = method.compute(inputs, ctx)
    elapsed = time.perf_counter() - started
    if result.method_id != method.method_id or result.version != method.version:
        raise MethodError(
            f"{method.method_key} returned a result attributed to "
            f"{result.method_id}@{result.version}; a result must name the implementation that "
            "produced it or the comparison record is wrong about who did what"
        )
    if result.claim_type is ClaimType.CAUSAL_CLAIM:
        raise MethodError(
            f"{method.method_key} emitted ClaimType.CAUSAL_CLAIM "
            f"({FailureCode.F18_UNWARRANTED_CAUSAL_CLAIM}): a static spectrum carries no "
            "interventional evidence, so no method may return one"
        )
    cost = dict(result.cost_incurred)
    cost.setdefault("wall_clock_s", elapsed)
    cost.setdefault("spectra", float(len(inputs.spectra)))
    final = replace(result, applicability=applicability, cost_incurred=cost)
    if log is not None:
        log.event("method_result", run_id=ctx.run_id, **final.to_dict())
    return final


def surfaceable_results(results: Sequence[MethodResult]) -> tuple[MethodResult, ...]:
    """The subset of ``results`` a report conclusion may draw on.

    Used by the reporting layer instead of a maturity comparison written out at each call site.
    A hypothesised method's result is computed, logged, scored where it can be scored, and dropped
    here - which is the whole of shadow mode from the report's point of view.
    """
    return tuple(r for r in results if r.surfaceable)
