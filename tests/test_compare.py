"""The comparison harness: what it counts, what it refuses to count, and what it will not hide."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from usis.compare.harness import run_comparison, run_method
from usis.compare.leaderboard import rank_rows, render_markdown, render_table
from usis.compare.record import ComparisonRecord, RecordError
from usis.compare.risk_coverage import (
    ScoringError,
    bootstrap_curve_statistic,
    comparable_aurc,
    risk_coverage_curve,
)
from usis.compare.scoring import ItemScore, Outcome, outcome_counts
from usis.compare.task import TaskError, TaskItem
from usis.core.provenance import RunContext
from usis.core.types import (
    AxisKind,
    AxisSpec,
    BeliefState,
    Candidate,
    ClaimType,
    FailureCode,
    IntensityKind,
    Maturity,
    Modality,
    Origin,
    SourceRecord,
    Spectrum,
    TaskKind,
)
from usis.methods.assumptions import MinSnr
from usis.methods.base import Citation, MethodBase, MethodResult
from usis.methods.inputs import MethodInputs
from usis.methods.registry import MethodEntry
from usis.methods.shadow import AnswerSource

CTX = RunContext(
    run_id="test", started_at="2026-09-11T00:00:00+00:00", seed=7, git_sha="0" * 40, git_dirty="no"
)
AXIS = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, np.linspace(300.0, 1500.0, 64))
SOURCE = SourceRecord(dataset_id="fixture", licence="CC0-1.0", retrieved_at="2026-09-11T00:00:00Z")


# ----------------------------------------------------------------------------------------------
# A minimal task and three minimal methods. Deliberately trivial: what is under test here is the
# harness's bookkeeping, and a realistic benchmark would let a scoring bug hide behind a real one.
# ----------------------------------------------------------------------------------------------
def _spectrum(index: int, *, level: float = 1.0) -> Spectrum:
    rng = np.random.default_rng(index)
    return Spectrum(
        spectrum_id=f"s{index:03d}",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=AXIS,
        intensity=level * (10.0 + rng.normal(0.0, 0.1, AXIS.n_points)),
        intensity_kind=IntensityKind.COUNTS,
        sample_id=f"sample-{index % 5}",
        source=SOURCE,
    )


class ToyTask:
    """A yes/no task where the reference alternates and the group repeats every five items."""

    task_id = "toy"
    task_kind = TaskKind.CLASSIFICATION
    value_metric_name = "margin"

    def __init__(self, n: int = 40, n_unscorable: int = 0, level: float = 1.0) -> None:
        self._items: list[TaskItem] = []
        for i in range(n):
            scorable = i >= n_unscorable
            self._items.append(
                TaskItem(
                    item_id=f"s{i:03d}",
                    inputs=MethodInputs.of(_spectrum(i, level=level)),
                    group=f"g{i % 5}",
                    reference=("yes" if i % 2 == 0 else "no") if scorable else None,
                    answer_source=(
                        AnswerSource.LABELLED_BENCHMARK if scorable else AnswerSource.NONE
                    ),
                )
            )

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return ("fixture",)

    @property
    def split(self) -> None:
        return None

    @property
    def notes(self) -> tuple[str, ...]:
        return ("a toy task, which shows nothing about spectroscopy",)

    def items(self) -> tuple[TaskItem, ...]:
        return tuple(self._items)

    def score(self, item: TaskItem, result: MethodResult) -> ItemScore:
        belief = result.belief
        assert belief is not None
        top = belief.top
        assert top is not None
        correct = top.name == item.reference
        return ItemScore(
            item_id=item.item_id,
            outcome=Outcome.CORRECT if correct else Outcome.INCORRECT,
            confidence=float(top.posterior),
            value=float(top.posterior) - belief.p_none,
            failure_codes=() if correct else (FailureCode.F14_SHORTCUT_LEARNING,),
            reference=item.reference,
            predicted=top.name,
        )


def _belief(answer: str, confidence: float) -> BeliefState:
    return BeliefState(
        candidates=(Candidate(posterior=confidence, name=answer),), p_none=1.0 - confidence
    )


class _Toy(MethodBase):
    """Answers 'yes' with a confidence that rises with the item index."""

    method_id = "toy.always_yes"
    version = "1.0.0"
    task = TaskKind.CLASSIFICATION
    maturity = Maturity.ESTABLISHED
    provenance = Citation(title="toy", internal_ref="tests/test_compare.py")

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        index = int(inputs.primary.spectrum_id[1:])
        return self._result(belief=_belief("yes", 0.05 + 0.9 * index / 64.0))


class _Oracle(MethodBase):
    """Always right, and says so at full confidence."""

    method_id = "toy.oracle"
    version = "1.0.0"
    task = TaskKind.CLASSIFICATION
    maturity = Maturity.ESTABLISHED
    provenance = Citation(title="toy", internal_ref="tests/test_compare.py")

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        index = int(inputs.primary.spectrum_id[1:])
        return self._result(belief=_belief("yes" if index % 2 == 0 else "no", 0.99))


class _Crasher(MethodBase):
    """Raises on every other item, to prove one defect does not end a comparison."""

    method_id = "toy.crasher"
    version = "1.0.0"
    task = TaskKind.CLASSIFICATION
    maturity = Maturity.ESTABLISHED
    provenance = Citation(title="toy", internal_ref="tests/test_compare.py")

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        index = int(inputs.primary.spectrum_id[1:])
        if index % 2:
            raise ZeroDivisionError("a deliberate defect")
        return self._result(belief=_belief("yes", 0.5))


class _Picky(_Toy):
    """Declares an SNR floor no fixture spectrum clears, so every run is refused."""

    method_id = "toy.picky"
    assumptions = (MinSnr(1e9),)


class _Shadow(_Oracle):
    """A hypothesised method: computed, scored, never surfaced."""

    method_id = "toy.shadow"
    maturity = Maturity.HYPOTHESIZED


def _entry(method: MethodBase, **kwargs: Any) -> MethodEntry:
    return MethodEntry(method=method, evidence=("tests/test_compare.py",), **kwargs)


def _record(*methods: MethodBase, task: ToyTask | None = None, **kwargs: Any) -> ComparisonRecord:
    return run_comparison(
        task or ToyTask(),
        CTX,
        entries=[_entry(m) for m in methods],
        bootstrap_samples=200,
        shows=("bookkeeping",),
        **kwargs,
    )


# ----------------------------------------------------------------------------------------------
# Outcomes and scores
# ----------------------------------------------------------------------------------------------
def test_only_answered_outcomes_are_eligible_and_only_unscorable_leaves_the_denominator() -> None:
    answered = {o for o in Outcome if o.is_answered}
    assert answered == {Outcome.CORRECT, Outcome.INCORRECT}
    assert {o for o in Outcome if not o.is_counted} == {Outcome.UNSCORABLE}


def test_an_answered_item_without_a_confidence_is_refused() -> None:
    with pytest.raises(ScoringError, match="carries no confidence"):
        ItemScore(item_id="x", outcome=Outcome.CORRECT)


def test_outcome_counts_show_a_zero_rather_than_omitting_the_key() -> None:
    counts = outcome_counts([ItemScore(item_id="x", outcome=Outcome.ABSTAINED)])
    assert counts[Outcome.ERRORED] == 0
    assert set(counts) == set(Outcome)


# ----------------------------------------------------------------------------------------------
# The risk-coverage curve
# ----------------------------------------------------------------------------------------------
def _scores(pattern: str, confidences: list[float] | None = None) -> list[ItemScore]:
    """``pattern`` uses c=correct, i=incorrect, a=abstained, r=refused, u=unscorable."""
    mapping = {
        "c": Outcome.CORRECT,
        "i": Outcome.INCORRECT,
        "a": Outcome.ABSTAINED,
        "r": Outcome.REFUSED,
        "u": Outcome.UNSCORABLE,
    }
    out = []
    for index, symbol in enumerate(pattern):
        outcome = mapping[symbol]
        confidence = None
        if outcome.is_answered:
            confidence = 0.9 - 0.01 * index if confidences is None else confidences[index]
        out.append(ItemScore(item_id=f"i{index}", outcome=outcome, confidence=confidence))
    return out


def test_coverage_denominator_is_the_whole_corpus_so_refusing_cannot_be_a_win() -> None:
    """A method perfect on the third it answers reaches coverage 1/3, not 1.0."""
    curve = risk_coverage_curve(_scores("ccrrrrrrr"))
    assert curve.n_presented == 9
    assert curve.n_answerable == 2
    assert curve.max_coverage == pytest.approx(2 / 9)
    assert curve.risk_at_coverage(0.5) is None
    assert curve.coverage_at_risk(0.0) == pytest.approx(2 / 9)


def test_unscorable_items_leave_the_denominator_entirely() -> None:
    assert risk_coverage_curve(_scores("ccuuu")).n_presented == 2


def test_a_point_is_emitted_per_threshold_not_per_rank() -> None:
    """Three items sharing a confidence are one operating point: no threshold separates them."""
    curve = risk_coverage_curve(_scores("cic", [0.5, 0.5, 0.5]))
    assert len(curve.points) == 1
    assert curve.points[0].n_accepted == 3
    assert curve.points[0].risk == pytest.approx(1 / 3)


def test_aurc_is_lower_when_the_confidence_ranks_the_errors_last() -> None:
    ordered = risk_coverage_curve(_scores("cccciiii", [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]))
    inverted = risk_coverage_curve(_scores("cccciiii", [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]))
    assert ordered.aurc < inverted.aurc
    assert ordered.coverage_at_risk(0.0) == pytest.approx(0.5)
    assert inverted.coverage_at_risk(0.0) == 0.0


def test_coverage_at_risk_is_zero_when_no_threshold_is_safe_enough() -> None:
    assert risk_coverage_curve(_scores("ii")).coverage_at_risk(0.1) == 0.0


def test_aurc_is_not_comparable_across_methods_that_reach_different_coverage() -> None:
    full = risk_coverage_curve(_scores("cccc"))
    partial = risk_coverage_curve(_scores("ccrr"))
    assert comparable_aurc([full, full])
    assert not comparable_aurc([full, partial])


def test_a_curve_over_zero_scorable_items_refuses_rather_than_drawing_nothing() -> None:
    with pytest.raises(ScoringError, match="no scorable items"):
        risk_coverage_curve(_scores("uuu"))


def test_a_curve_statistic_comes_with_a_resampled_interval() -> None:
    estimate = bootstrap_curve_statistic(
        _scores("ccccciiiii"), lambda c: c.aurc, name="aurc", n_samples=400, seed=3
    )
    assert estimate.ci_low <= estimate.value <= estimate.ci_high
    assert not estimate.is_degenerate
    assert "aurc" in estimate.render()


def test_a_group_bootstrap_over_one_group_refuses_instead_of_collapsing() -> None:
    scores = _scores("cici")
    with pytest.raises(ScoringError, match="single group"):
        bootstrap_curve_statistic(
            scores, lambda c: c.aurc, name="aurc", groups=["g"] * 4, n_samples=200
        )


# ----------------------------------------------------------------------------------------------
# Task items
# ----------------------------------------------------------------------------------------------
def test_a_claimed_but_absent_reference_is_refused() -> None:
    with pytest.raises(TaskError, match="claims an independent answer"):
        TaskItem(
            item_id="x",
            inputs=MethodInputs.of(_spectrum(0)),
            group="g",
            answer_source=AnswerSource.LABELLED_BENCHMARK,
        )


def test_an_answer_from_the_portfolio_itself_is_refused() -> None:
    with pytest.raises(TaskError, match="not independent"):
        TaskItem(
            item_id="x",
            inputs=MethodInputs.of(_spectrum(0)),
            group="g",
            reference="yes",
            answer_source=AnswerSource.PORTFOLIO_CONSENSUS,
        )


def test_an_item_without_a_group_is_refused_because_the_interval_needs_one() -> None:
    with pytest.raises(TaskError, match="group key"):
        TaskItem(item_id="x", inputs=MethodInputs.of(_spectrum(0)), group="")


# ----------------------------------------------------------------------------------------------
# The harness end to end
# ----------------------------------------------------------------------------------------------
def test_the_oracle_leads_the_leaderboard_and_every_number_is_an_interval() -> None:
    record = _record(_Toy(), _Oracle())
    ranked = rank_rows(record.surfaceable_rows, record.risk_target)
    assert ranked[0].method_key == "toy.oracle@1.0.0"
    oracle = record.row("toy.oracle@1.0.0")
    assert oracle is not None
    assert oracle.selective_accuracy == pytest.approx(1.0)
    assert oracle.coverage == pytest.approx(1.0)
    for metric in oracle.metrics:
        assert metric.ci_low <= metric.mean <= metric.ci_high
    assert oracle.metric("margin") is not None


def test_a_refused_run_costs_coverage_rather_than_being_skipped() -> None:
    record = _record(_Picky())
    row = record.row("toy.picky@1.0.0")
    assert row is not None
    assert row.outcomes[Outcome.REFUSED] == 40
    assert row.n_counted == 40
    assert row.coverage == 0.0
    assert row.applicability.n_refused == 40
    assert row.applicability.violation_rate == pytest.approx(1.0)
    assert FailureCode.F08_NON_IDENTIFIABILITY in row.failures


def test_one_crashing_method_does_not_end_the_comparison() -> None:
    record = _record(_Crasher(), _Oracle())
    crasher = record.row("toy.crasher@1.0.0")
    oracle = record.row("toy.oracle@1.0.0")
    assert crasher is not None and oracle is not None
    assert crasher.outcomes[Outcome.ERRORED] == 20
    assert any("ZeroDivisionError" in message for message in crasher.errors)
    assert oracle.selective_accuracy == pytest.approx(1.0)
    assert "toy.crasher@1.0.0" in render_table(record)


def test_an_unscorable_item_is_in_no_denominator_and_is_counted_on_the_record() -> None:
    record = _record(_Oracle(), task=ToyTask(n=40, n_unscorable=8))
    row = record.row("toy.oracle@1.0.0")
    assert row is not None
    assert record.n_items_presented == 40
    assert record.n_unscorable == 8
    assert row.n_counted == 32
    assert row.outcomes[Outcome.UNSCORABLE] == 8
    assert row.curve.n_presented == 32


def test_a_shadow_method_is_scored_and_kept_out_of_the_surfaceable_rows() -> None:
    record = _record(_Toy(), _Shadow())
    assert [r.method_key for r in record.surfaceable_rows] == ["toy.always_yes@1.0.0"]
    shadow = record.row("toy.shadow@1.0.0")
    assert shadow is not None
    assert not shadow.surfaceable
    assert shadow.selective_accuracy == pytest.approx(1.0)
    rendered = render_table(record)
    assert "NOT A CONCLUSION" in rendered
    assert "toy.shadow@1.0.0" in rendered


def test_declared_and_scored_failure_codes_are_never_summed_into_one_bar() -> None:
    record = _record(_Toy())
    row = record.row("toy.always_yes@1.0.0")
    assert row is not None
    assert row.failures_scored[FailureCode.F14_SHORTCUT_LEARNING] == 20
    assert not row.failures_declared
    assert "declared by the method" in render_markdown(record)


def test_the_record_refuses_to_exist_without_stating_what_it_does_not_show() -> None:
    class Silent(ToyTask):
        @property
        def notes(self) -> tuple[str, ...]:
            return ()

    with pytest.raises(RecordError, match="does NOT establish"):
        run_comparison(
            Silent(), CTX, entries=[_entry(_Oracle())], bootstrap_samples=200, shows=("x",)
        )


def test_a_comparison_with_no_entrant_refuses_rather_than_writing_an_empty_leaderboard() -> None:
    with pytest.raises(RecordError, match="no method produced a row"):
        run_comparison(ToyTask(), CTX, entries=[], shows=("x",), does_not_show=("x",))


def test_the_field_comes_from_the_registry_not_from_a_list_the_caller_passes() -> None:
    """The C6 property: a benchmark cannot choose who it is compared against."""
    from usis.methods.registry import METHODS, load_builtin_methods, register_method

    load_builtin_methods()
    key = "toy.registered@1.0.0"
    if key not in list(METHODS):

        class Registered(_Oracle):
            method_id = "toy.registered"

        register_method(Registered(), evidence=("tests/test_compare.py",))
    record = run_comparison(
        ToyTask(), CTX, bootstrap_samples=200, shows=("x",), does_not_show=("x",)
    )
    keys = {row.method_key for row in record.rows}
    assert key in keys
    assert "baseline.pca_knn@1.0.0" in keys, "every registered classifier must enter"
    assert all(row.task is TaskKind.CLASSIFICATION for row in record.rows)


def test_the_record_round_trips_through_json_with_its_per_item_detail() -> None:
    record = _record(_Oracle())
    payload = json.loads(json.dumps(record.to_dict(), default=str))
    row = payload["rows"][0]
    assert len(row["items"]) == 40
    assert row["risk_coverage"]["aurc_integrated_over"] == [0.0, 1.0]
    assert row["applicability"]["n_runs"] == 40
    assert payload["n_unscorable"] == 0


def test_a_method_run_records_its_own_cost_rather_than_the_harness_guessing() -> None:
    run = run_method(_Oracle(), ToyTask(n=4), ToyTask(n=4).items(), CTX)
    assert run.cost["spectra"] == 4.0
    assert run.cost["wall_clock_s"] > 0.0


def test_a_causal_claim_from_a_method_is_a_typed_violation_not_a_sentence() -> None:
    class Causal(_Oracle):
        method_id = "toy.causal"

        def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
            return self._result(belief=_belief("yes", 0.9), claim_type=ClaimType.CAUSAL_CLAIM)

    record = _record(Causal())
    row = record.row("toy.causal@1.0.0")
    assert row is not None
    assert row.outcomes[Outcome.ERRORED] == 40
    assert any("CAUSAL_CLAIM" in message for message in row.errors)


def test_a_refusal_is_not_an_abstention_and_neither_is_a_crash() -> None:
    record = _record(_Picky(), _Crasher())
    picky = record.row("toy.picky@1.0.0")
    crasher = record.row("toy.crasher@1.0.0")
    assert picky is not None and crasher is not None
    assert picky.outcomes[Outcome.ABSTAINED] == 0
    assert crasher.outcomes[Outcome.ABSTAINED] == 0
    assert picky.outcomes[Outcome.REFUSED] == 40
    assert crasher.outcomes[Outcome.ERRORED] == 20


def test_an_interval_resampled_over_items_says_so_instead_of_looking_narrower() -> None:
    class OneGroup(ToyTask):
        def items(self) -> tuple[TaskItem, ...]:
            return tuple(
                TaskItem(
                    item_id=item.item_id,
                    inputs=item.inputs,
                    group="only",
                    reference=item.reference,
                    answer_source=item.answer_source,
                )
                for item in super().items()
            )

    record = _record(_Toy(), task=OneGroup())
    row = record.row("toy.always_yes@1.0.0")
    assert row is not None
    assert any("resampled over items, not groups" in note for note in row.notes)
    assert row.metric("selective_accuracy") is not None
