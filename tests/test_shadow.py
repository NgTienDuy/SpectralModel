"""Shadow mode and the promotion gate: what ADR-0001 §S2 makes impossible."""

from __future__ import annotations

import pytest

from usis.data.splits import SplitKind
from usis.methods.promotion import (
    PromotionCriteria,
    PromotionError,
    evaluate,
    evaluate_all,
)
from usis.methods.shadow import (
    AnswerSource,
    ShadowError,
    ShadowLedger,
    ShadowRecord,
)

SHADOW = "shape.broadening_origin@0.1.0"
INCUMBENT = "baseline.nearest_neighbour@1.0.0"


def _consensus(index: int, *, agrees: bool = True) -> ShadowRecord:
    """A production run: no independent answer, only the portfolio's own conclusion."""
    return ShadowRecord(
        method_key=SHADOW,
        incumbent_key=INCUMBENT,
        spectrum_id=f"prod-{index:04d}",
        run_id="prod",
        answer_source=AnswerSource.PORTFOLIO_CONSENSUS,
        shadow_answer="A" if agrees else "B",
        incumbent_answer="A",
        dataset_id="production",
    )


def _labelled(
    index: int,
    *,
    shadow: str,
    incumbent: str,
    truth: str = "A",
    split_kind: SplitKind = SplitKind.INSTRUMENT_DISJOINT,
    dataset_id: str = "rod",
) -> ShadowRecord:
    return ShadowRecord(
        method_key=SHADOW,
        incumbent_key=INCUMBENT,
        spectrum_id=f"bench-{index:04d}",
        run_id="bench",
        answer_source=AnswerSource.LABELLED_BENCHMARK,
        shadow_answer=shadow,
        incumbent_answer=incumbent,
        reference_answer=truth,
        split_kind=split_kind,
        dataset_id=dataset_id,
    )


CRITERIA = PromotionCriteria(
    min_independent_n=200,
    min_margin=0.02,
    preregistration_ref="experiments/E-promotion/preregistration.md@abc1234",
    required_split_kinds=frozenset({SplitKind.INSTRUMENT_DISJOINT}),
)


# ------------------------------------------------------------------------------------------------
# The record type refuses the shapes that would let §S2 back in
# ------------------------------------------------------------------------------------------------
def test_an_independent_source_without_a_reference_answer_is_rejected() -> None:
    with pytest.raises(ShadowError, match="claims an independent answer"):
        ShadowRecord(
            method_key=SHADOW,
            incumbent_key=INCUMBENT,
            spectrum_id="x",
            run_id="r",
            answer_source=AnswerSource.LABELLED_BENCHMARK,
            shadow_answer="A",
            incumbent_answer="A",
        )


def test_a_reference_answer_from_the_portfolio_is_rejected() -> None:
    """The exact §S2 move: passing the incumbent's own conclusion off as ground truth."""
    with pytest.raises(ShadowError, match="belongs in incumbent_answer"):
        ShadowRecord(
            method_key=SHADOW,
            incumbent_key=INCUMBENT,
            spectrum_id="x",
            run_id="r",
            answer_source=AnswerSource.PORTFOLIO_CONSENSUS,
            shadow_answer="A",
            incumbent_answer="A",
            reference_answer="A",
        )


def test_only_the_three_independent_sources_are_scorable() -> None:
    assert AnswerSource.LABELLED_BENCHMARK.is_independent
    assert AnswerSource.RESOLVED_REMEASUREMENT.is_independent
    assert AnswerSource.ORTHOGONAL_MODALITY.is_independent
    assert not AnswerSource.PORTFOLIO_CONSENSUS.is_independent
    assert not AnswerSource.NONE.is_independent


# ------------------------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------------------------
def test_perfect_agreement_with_the_incumbent_scores_nothing() -> None:
    """M2 acceptance 2.3: a shadow method cannot be promoted on agreement with the incumbent."""
    ledger = ShadowLedger.of([_consensus(i) for i in range(500)])
    assert len(ledger) == 500
    with pytest.raises(ShadowError, match="0 independently-answered runs"):
        ledger.score(SHADOW)
    decision = evaluate(ledger, SHADOW, CRITERIA)
    assert not decision.eligible
    assert any("0 independently-answered runs" in reason for reason in decision.reasons)
    assert decision.score is None


def test_disagreement_on_a_production_run_is_a_flag_and_not_a_number() -> None:
    ledger = ShadowLedger.of(
        [_consensus(i, agrees=i % 2 == 0) for i in range(10)],
    )
    flags = ledger.flags(SHADOW)
    assert len(flags) == 5
    assert all(isinstance(flag, ShadowRecord) for flag in flags)
    with pytest.raises(ShadowError):
        ledger.score(SHADOW)


def test_a_score_reports_its_own_n_and_the_evidence_ratio() -> None:
    ledger = ShadowLedger.of(
        [_consensus(i) for i in range(90)]
        + [_labelled(i, shadow="A", incumbent="A") for i in range(10)]
    )
    score = ledger.score(SHADOW)
    assert score.n == 10
    assert score.n_flagged_unscorable == 90
    assert score.shadow_accuracy == 1.0
    assert score.margin == 0.0
    assert score.evidence_ratio == pytest.approx(0.1)


def test_the_margin_is_measured_on_the_same_subset() -> None:
    records = [_labelled(i, shadow="A", incumbent="B") for i in range(60)]
    records += [_labelled(60 + i, shadow="B", incumbent="B") for i in range(40)]
    score = ShadowLedger.of(records).score(SHADOW)
    assert score.n == 100
    assert score.shadow_accuracy == pytest.approx(0.6)
    assert score.incumbent_accuracy == pytest.approx(0.0)
    assert score.margin == pytest.approx(0.6)


def test_a_ledger_spanning_several_incumbents_refuses_to_pool_them() -> None:
    other = ShadowRecord(
        method_key=SHADOW,
        incumbent_key="baseline.pls@1.0.0",
        spectrum_id="bench-9999",
        run_id="bench",
        answer_source=AnswerSource.LABELLED_BENCHMARK,
        shadow_answer="A",
        incumbent_answer="A",
        reference_answer="A",
        split_kind=SplitKind.INSTRUMENT_DISJOINT,
        dataset_id="rod",
    )
    ledger = ShadowLedger.of([_labelled(0, shadow="A", incumbent="A"), other])
    with pytest.raises(ShadowError, match="several incumbents"):
        ledger.score(SHADOW)
    assert ledger.score(SHADOW, INCUMBENT).n == 1


def test_the_ledger_serialises_every_record_with_its_scorability() -> None:
    ledger = ShadowLedger()
    ledger.add(_consensus(0))
    ledger.add(_labelled(1, shadow="A", incumbent="A"))
    rows = ledger.to_rows()
    assert [row["scorable"] for row in rows] == [False, True]
    assert rows[0]["reference_answer"] is None


# ------------------------------------------------------------------------------------------------
# The promotion gate
# ------------------------------------------------------------------------------------------------
def test_criteria_without_a_preregistration_reference_cannot_exist() -> None:
    with pytest.raises(PromotionError, match="preregistration_ref"):
        PromotionCriteria(min_independent_n=200, min_margin=0.02, preregistration_ref="  ")


def test_a_gate_with_a_trivial_n_is_refused() -> None:
    with pytest.raises(PromotionError, match="min_independent_n"):
        PromotionCriteria(min_independent_n=0, min_margin=0.02, preregistration_ref="pre.md")


def test_a_small_labelled_subset_is_not_a_promotion() -> None:
    ledger = ShadowLedger.of([_labelled(i, shadow="A", incumbent="B") for i in range(30)])
    decision = evaluate(ledger, SHADOW, CRITERIA)
    assert not decision.eligible
    assert any("n = 30" in reason for reason in decision.reasons)


def test_evidence_from_the_wrong_split_does_not_count() -> None:
    ledger = ShadowLedger.of(
        [_labelled(i, shadow="A", incumbent="B", split_kind=SplitKind.RANDOM) for i in range(400)]
    )
    decision = evaluate(ledger, SHADOW, CRITERIA)
    assert not decision.eligible
    assert any("instrument_disjoint" in reason for reason in decision.reasons)


def test_a_gate_that_requires_no_split_kind_declares_itself_weak() -> None:
    weak = PromotionCriteria(min_independent_n=1, min_margin=0.0, preregistration_ref="pre.md@sha")
    ledger = ShadowLedger.of([_labelled(i, shadow="A", incumbent="B") for i in range(5)])
    decision = evaluate(ledger, SHADOW, weak)
    assert not decision.eligible
    assert any("WEAK GATE" in reason for reason in decision.reasons)


def test_eligibility_still_requires_an_adr() -> None:
    ledger = ShadowLedger.of(
        [_labelled(i, shadow="A", incumbent="B") for i in range(150)]
        + [
            _labelled(150 + i, shadow="A", incumbent="B", dataset_id="scidata-api")
            for i in range(150)
        ]
    )
    criteria = PromotionCriteria(
        min_independent_n=200,
        min_margin=0.02,
        preregistration_ref="experiments/E-promotion/preregistration.md@abc1234",
        required_split_kinds=frozenset({SplitKind.INSTRUMENT_DISJOINT}),
        min_datasets=2,
    )
    decision = evaluate(ledger, SHADOW, criteria)
    assert decision.eligible
    assert decision.requires_adr
    assert decision.score is not None and decision.score.n == 300
    assert "must still be written" in decision.reasons[0]
    assert decision.to_dict()["preregistration_ref"].endswith("@abc1234")


def test_evaluate_all_covers_every_method_in_the_ledger() -> None:
    ledger = ShadowLedger.of([_consensus(0)])
    decisions = evaluate_all(ledger, CRITERIA)
    assert len(decisions) == 1
    assert decisions[0].method_key == SHADOW
