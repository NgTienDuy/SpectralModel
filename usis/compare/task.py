"""What a comparison is run *on*: the benchmark side of the harness.

`CLAUDE.md` requires that *adding a method must never require touching the harness*. The mirror of
that requirement is this module: the harness must not know what any particular benchmark is either,
or every new dataset would edit the same file the methods are trying not to edit. So the harness
depends on two protocols and nothing else - :class:`~usis.methods.base.Method` on one side and
:class:`ComparisonTask` on the other - and a benchmark is a new file that implements one of them.

A task owns three things the harness deliberately cannot decide for it:

**Which items exist, and how they are grouped.** ``group`` is the resampling unit for every
interval on the leaderboard. Replicate scans of one vial are not independent evidence, and a
bootstrap that resamples them individually returns an interval narrower than the truth by roughly
the square root of the replicate count.

**Where the answer came from.** :class:`~usis.methods.shadow.AnswerSource` is reused rather than
re-invented, because the rule it encodes is the same one: an answer produced by the system's own
portfolio is not evidence about the system. A :class:`TaskItem` whose source is not independent
carries no reference, and the harness scores it ``UNSCORABLE``.

**What counts as right.** Correctness in retrieval is not correctness in quantification, and a
harness that owned the comparison would have to grow a branch per task family. :meth:`score` keeps
that knowledge in the benchmark, where the domain argument for it lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from usis.core.errors import UsisError
from usis.methods.shadow import AnswerSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from usis.compare.scoring import ItemScore
    from usis.core.types import TaskKind
    from usis.data.splits import Split
    from usis.methods.base import MethodResult
    from usis.methods.inputs import MethodInputs

__all__ = ["ComparisonTask", "TaskError", "TaskItem"]


class TaskError(UsisError):
    """A benchmark task is structurally unable to support the comparison asked of it."""


@dataclass(frozen=True)
class TaskItem:
    """One unit of the benchmark: the inputs a method is given, and the answer it is held to.

    The reference/source invariant is the same one :class:`~usis.methods.shadow.ShadowRecord`
    enforces, for the same reason and with the same wording: a claimed-but-absent reference is
    worse than no reference, because it counts.
    """

    item_id: str
    inputs: MethodInputs
    group: str
    reference: str | None = None
    answer_source: AnswerSource = AnswerSource.NONE
    reference_value: float | None = None
    context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Reject an item that claims an independent answer it does not carry, and the converse."""
        has_answer = self.reference is not None or self.reference_value is not None
        if self.answer_source.is_independent and not has_answer:
            raise TaskError(
                f"{self.item_id}: answer_source {self.answer_source} claims an independent answer "
                "but neither reference nor reference_value is set. Record AnswerSource.NONE "
                "instead - a claimed-but-absent reference is worse than no reference, because it "
                "counts."
            )
        if not self.answer_source.is_independent and has_answer:
            raise TaskError(
                f"{self.item_id}: an answer was supplied with answer_source "
                f"{self.answer_source}, which is not independent of the system's own output. Name "
                "which of the three independent sources it came from, or drop it (ADR-0001 §S2)."
            )
        if not self.group:
            raise TaskError(
                f"{self.item_id}: every item needs a group key. It is the resampling unit for "
                "every interval on the leaderboard, and 'one item per group' has to be stated "
                "rather than assumed - replicates of one sample are not independent evidence."
            )

    @property
    def is_scorable(self) -> bool:
        """Whether this item can contribute to a score at all."""
        return self.answer_source.is_independent


@runtime_checkable
class ComparisonTask(Protocol):
    """A benchmark the harness can run any registered method against.

    ``task_kind`` selects the methods: the harness runs every registered method declaring this
    task family and no others, so a benchmark never names the methods it will be compared with.
    That is what keeps a leaderboard from silently omitting the baseline that would have beaten it.
    """

    @property
    def task_id(self) -> str:
        """Names this benchmark on every record and result card it appears in."""
        ...

    @property
    def task_kind(self) -> TaskKind:
        """The task family. Selects the field: every registered method declaring it, no other."""
        ...

    @property
    def value_metric_name(self) -> str:
        """Name of this task's continuous metric, or ``""`` when it has none.

        Empty is the honest default: a retrieval task has no residual and a quantification task has
        no notion of "correct", and one leaderboard column that silently means something different
        per row is worse than a missing column.
        """
        ...

    def items(self) -> Sequence[TaskItem]:
        """Every item in the evaluation set, in a stable order."""
        ...

    def score(self, item: TaskItem, result: MethodResult) -> ItemScore:
        """Judge one method's result on one item. Never called for an unscorable item."""
        ...

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """The datasets this benchmark draws on, for the record's provenance section."""
        ...

    @property
    def split(self) -> Split | None:
        """The split this benchmark evaluates on, whose caveats travel onto every number."""
        ...

    @property
    def notes(self) -> tuple[str, ...]:
        """What this benchmark does not establish. Written onto the record, not optional."""
        ...
