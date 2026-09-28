"""Task family G - compound retrieval, evaluated open-set, which is the only honest way to do it.

A closed-set retrieval benchmark asks "which library entry is this?" and guarantees that one of
them is right. Nothing in the field works that way: the substance in front of the instrument is
routinely absent from whatever library is at hand, and a matcher that must name an entry will name
one. So this task presents two kinds of query and scores the same decision rule on both:

* **positive queries** - a spectrum of an identity that *is* in the library, taken from a different
  record than the library entry itself;
* **negative queries** - every spectrum of an identity held out of the library entirely, for which
  the correct answer is "none of these".

**"None of these" is scored as a claim here, not as a refusal.** This is the one place the
project's own convention needs stating carefully. Abstention is a first-class output, and on an
unlabelled sample "not enough evidence" is a success. But on a *labelled* open-set benchmark,
"absent from this library" has a truth value, and a method that says it about a present substance
is wrong. So every query is an answered item and the outcome is correctness of the decision; the
selective behaviour a reader also wants is then measured on top, by the risk-coverage curve, which
thresholds the confidence the method put on whichever decision it took.

**The library holds one spectrum per identity.** The alternative - every training spectrum as its
own entry - measures something different and easier: with eight records of calcite in the library,
a query need only resemble the most similar one. One entry per identity is the harder and more
realistic setting, and it makes the library size equal to the number of identities, which is the
number a reader assumes it is.

**The decision rule is the method's own.** The task does not threshold: it reads
``p_none > top.posterior`` - the same comparison
:func:`~usis.chemometrics.matching.belief_from_matches` uses to raise ``LIBRARY_GAP`` - so the
task's verdict and the method's own flag can never disagree. That the underlying ``p_none`` is an
uncalibrated convention is `PROJECT_NOVELTY.md#N10`'s subject and travels onto every number here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from usis.compare.scoring import ItemScore, Outcome
from usis.compare.task import TaskError, TaskItem
from usis.core.types import FailureCode, TaskKind
from usis.data.splits import SplitKind, make_split
from usis.methods.inputs import LibraryHandle, MethodInputs
from usis.methods.shadow import AnswerSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from usis.core.types import Spectrum
    from usis.data.splits import Split
    from usis.methods.base import MethodResult
    from usis.preprocess.pipeline import Pipeline

__all__ = ["NONE_KEY", "OpenSetRetrievalTask", "identity_key"]

#: The reference written on a query whose identity was held out of the library. Not a compound.
NONE_KEY = "__none_of_these__"


def identity_key(spectrum: Spectrum) -> str:
    """The identity a retrieval answer is judged against.

    The same expression :mod:`usis.data.splits` groups ``molecule_disjoint`` and ``open_set`` on,
    so the library partition and the scoring key cannot drift apart. An unlabelled spectrum has no
    identity and raises rather than being given an empty key that would then match other unlabelled
    spectra.
    """
    label = spectrum.label
    key = None if label is None else (label.inchikey or label.smiles or label.name)
    if not key:
        raise TaskError(
            f"{spectrum.spectrum_id!r} carries no identity, so a retrieval answer about it cannot "
            "be right or wrong. Filter unlabelled spectra out of the corpus deliberately - "
            "dropping them inside the scorer would silently change the benchmark."
        )
    return key


@dataclass(frozen=True)
class OpenSetRetrievalTask:
    """An open-set compound-retrieval benchmark over a corpus already on a common axis.

    Built by :meth:`from_spectra`, which does the partitioning. The constructor is not the entry
    point because the split, the library and the query list have to be derived together or they
    can contradict each other.
    """

    task_id: str
    library_keys: tuple[str, ...]
    library_matrix: np.ndarray
    library_spectra: tuple[Spectrum, ...]
    queries: tuple[TaskItem, ...]
    dataset_ids: tuple[str, ...]
    split: Split | None
    notes: tuple[str, ...]
    pipeline_label: str = ""
    task_kind: TaskKind = TaskKind.COMPOUND_RETRIEVAL
    value_metric_name: str = "reciprocal_rank"
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spectra(
        cls,
        spectra: Sequence[Spectrum],
        *,
        task_id: str,
        dataset_ids: Sequence[str],
        seed: int,
        held_out_fraction: float = 0.2,
        pipeline: Pipeline | None = None,
        extra_notes: Sequence[str] = (),
    ) -> OpenSetRetrievalTask:
        """Partition ``spectra`` into a one-per-identity library and a query set.

        Args:
            spectra: the corpus, already preprocessed onto one common axis. Preprocessing is a
                recorded step and belongs to the caller, so that the chain travels on the record
                rather than being applied invisibly here.
            task_id: names this benchmark on every record and result card it appears in.
            dataset_ids: the datasets the corpus came from, for the record's provenance section.
            seed: fixes which identities are held out. Recorded on the split.
            held_out_fraction: target share of *spectra* whose identity is kept out of the library.
            pipeline: the preprocessing chain the caller applied, recorded by label so a reader
                can tell which chain the numbers belong to.
            extra_notes: further caveats to carry onto every record built from this task.

        Raises:
            TaskError: if the corpus cannot support the benchmark - no held-out identity, or no
                identity with a second record to query with.
        """
        if not spectra:
            raise TaskError(f"{task_id}: cannot build a retrieval benchmark from an empty corpus")
        widths = {int(np.asarray(s.axis.values).size) for s in spectra}
        if len(widths) > 1:
            raise TaskError(
                f"{task_id}: the corpus is on {sorted(widths)} channels. A retrieval benchmark "
                "compares channel k with channel k, so the corpus must be resampled onto one grid "
                "first - as a recorded preprocessing step, not inside the benchmark."
            )
        split = make_split(spectra, SplitKind.OPEN_SET, seed=seed, test_fraction=held_out_fraction)
        known = split.train_groups

        by_identity: dict[str, list[Spectrum]] = {}
        for spectrum in spectra:
            by_identity.setdefault(identity_key(spectrum), []).append(spectrum)

        library_spectra: list[Spectrum] = []
        positives: list[tuple[Spectrum, str]] = []
        for key in sorted(known):
            records = sorted(by_identity[key], key=lambda s: s.spectrum_id)
            library_spectra.append(records[0])
            positives.extend((record, key) for record in records[1:])
        if not library_spectra:
            raise TaskError(f"{task_id}: the split left no identity in the library")
        if not positives:
            raise TaskError(
                f"{task_id}: every library identity has exactly one record, so there is no query "
                "of a present substance. A benchmark of negatives alone measures only the "
                "abstention rate and cannot detect a matcher that abstains on everything."
            )

        negatives = [
            (spectrum, NONE_KEY)
            for key in sorted(split.test_groups)
            for spectrum in sorted(by_identity[key], key=lambda s: s.spectrum_id)
        ]
        if not negatives:
            raise TaskError(f"{task_id}: the split held out no identity, so the task is closed-set")

        matrix = np.asarray([s.intensity for s in library_spectra], dtype=np.float64)
        keys = tuple(identity_key(s) for s in library_spectra)
        handle = LibraryHandle(
            library_id=f"{task_id}:library",
            n_entries=len(keys),
            modality=library_spectra[0].modality,
            inchikeys=None,
            note=(
                "One spectrum per identity, taken from the lowest record id of that identity. "
                "Membership is left unenumerated because ROD identifies minerals by name, not by "
                "InChIKey, so a key-based coverage check would be answering a question the "
                "corpus cannot support."
            ),
        )

        items: list[TaskItem] = []
        for spectrum, reference in (*positives, *negatives):
            items.append(
                TaskItem(
                    item_id=spectrum.spectrum_id,
                    inputs=MethodInputs(
                        spectra=(spectrum,),
                        library=handle,
                        calibration=tuple(library_spectra),
                        aux={
                            "library_spectra": matrix,
                            "library_keys": list(keys),
                        },
                    ),
                    group=identity_key(spectrum),
                    reference=reference,
                    answer_source=AnswerSource.LABELLED_BENCHMARK,
                )
            )

        notes = (
            f"Library holds {len(keys)} identities, one spectrum each. Queries: {len(positives)} "
            f"of a present identity, {len(negatives)} of an identity held out of the library.",
            "'None of these' is scored as a claim with a truth value, not as a refusal to answer. "
            "Selective behaviour is measured separately, by the risk-coverage curve over the "
            "confidence each method put on the decision it took.",
            "p_none underlying every decision is an UNCALIBRATED convention (1 - best "
            "similarity), so the position of a method on the risk-coverage curve reflects that "
            "convention as much as the method. Calibrating it is PROJECT_NOVELTY.md#N10, at M6.",
            "reciprocal_rank is truncated at the candidate list each method returns. Methods "
            "configured to return different numbers of candidates are NOT comparable on that "
            "column; they remain comparable on accuracy and on the curve.",
            *extra_notes,
        )
        return cls(
            task_id=task_id,
            library_keys=keys,
            library_matrix=matrix,
            library_spectra=tuple(library_spectra),
            queries=tuple(items),
            dataset_ids=tuple(dataset_ids),
            split=split,
            notes=notes,
            pipeline_label="" if pipeline is None else pipeline.label,
            context={
                "n_library_identities": len(keys),
                "n_positive_queries": len(positives),
                "n_negative_queries": len(negatives),
                "n_channels": int(matrix.shape[1]),
            },
        )

    def items(self) -> tuple[TaskItem, ...]:
        """Every query, positives first then negatives, in a stable order."""
        return self.queries

    def score(self, item: TaskItem, result: MethodResult) -> ItemScore:
        """Judge one decision: the named identity, or "none of these", against the reference."""
        belief = result.belief
        if belief is None:
            raise TaskError(
                f"{result.method_key} returned no BeliefState for {item.item_id}. A retrieval "
                "answer is an identity claim, and this project has no way to express one without "
                "a posterior and an open-set residual."
            )
        top = belief.top
        if top is None or belief.p_none > top.posterior:
            decision, confidence = NONE_KEY, belief.p_none
        else:
            decision, confidence = top.name or "", float(top.posterior)
        chose_none = decision == NONE_KEY
        correct = decision == item.reference
        was_present = item.reference != NONE_KEY

        codes: tuple[FailureCode, ...] = ()
        if not correct:
            codes = (
                (FailureCode.F12_RETRIEVAL_MISS,)
                if was_present
                else (FailureCode.F10_HALLUCINATED_ASSIGNMENT,)
            )

        rank = None
        if was_present:
            ordered = sorted(belief.candidates, key=lambda c: -c.posterior)
            positions = [i for i, c in enumerate(ordered, 1) if c.name == item.reference]
            rank = 1.0 / positions[0] if positions else 0.0

        return ItemScore(
            item_id=item.item_id,
            outcome=Outcome.CORRECT if correct else Outcome.INCORRECT,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            value=rank,
            failure_codes=codes,
            reference=item.reference,
            predicted=decision,
            detail=(
                f"decided {'none of these' if chose_none else decision!r} at confidence "
                f"{confidence:.3f} (p_none {belief.p_none:.3f}); reference "
                f"{'absent from library' if not was_present else repr(item.reference)}"
            ),
        )
