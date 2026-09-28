"""The leakage checker, including against a deliberately planted leak.

M1 acceptance requires that the checker catch a planted leak in tests; that is
``test_a_planted_replicate_leak_is_caught`` below.
"""

from __future__ import annotations

import numpy as np
import pytest

from usis.core.errors import UsisError
from usis.core.types import (
    AxisKind,
    AxisSpec,
    GroundTruth,
    InstrumentSpec,
    IntensityKind,
    Modality,
    Origin,
    SourceRecord,
    Spectrum,
)
from usis.data.leakage import (
    NEAR_DUPLICATE_CORRELATION,
    assert_no_leakage,
    check_leakage,
)
from usis.data.registry import DATASETS, DatasetSpec, ReplicateStructure, register_dataset
from usis.data.splits import Split, SplitKind, make_split

AXIS = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, np.linspace(400.0, 1800.0, 48))
DATASET = "leakage-test"


def _register() -> str:
    if DATASET not in DATASETS:
        register_dataset(
            DatasetSpec(
                dataset_id=DATASET,
                name=DATASET,
                modality=Modality.RAMAN,
                origin=Origin.EXPERIMENTAL,
                licence="CC0-1.0",
                replicate_structure=ReplicateStructure.KNOWN,
            )
        )
    return DATASET


def _spectrum(
    index: int,
    sample: str,
    *,
    inchikey: str | None = None,
    values=None,
    instrument: str = "inst",
) -> Spectrum:
    rng = np.random.default_rng(index)
    intensity = rng.normal(100.0, 5.0, AXIS.n_points) if values is None else values
    return Spectrum(
        spectrum_id=f"leak-{index}",
        modality=Modality.RAMAN,
        origin=Origin.EXPERIMENTAL,
        axis=AXIS,
        intensity=intensity,
        intensity_kind=IntensityKind.ARBITRARY,
        sample_id=sample,
        source=SourceRecord(dataset_id=_register(), licence="CC0-1.0", retrieved_at="t"),
        instrument=InstrumentSpec(instrument_id=instrument, lab_id="lab", excitation_nm=785.0),
        label=GroundTruth(name=sample, inchikey=inchikey),
    )


@pytest.fixture
def corpus() -> list[Spectrum]:
    return [_spectrum(i, f"sample-{i // 3}", inchikey=f"KEY{i // 3}") for i in range(12)]


def test_a_clean_sample_disjoint_split_has_no_findings(corpus):
    split = make_split(corpus, SplitKind.SAMPLE_DISJOINT, seed=0)
    assert check_leakage(corpus, split) == []
    assert_no_leakage(corpus, split)


def test_a_planted_replicate_leak_is_caught(corpus):
    # The M1 acceptance criterion. Index 0 and 1 are replicates of sample-0; putting one on each
    # side is exactly the failure the checker exists for.
    leaking = Split(
        kind=SplitKind.SAMPLE_DISJOINT,
        train=(0, 3, 4, 5),
        test=(1, 6, 7, 8),
        group_key="sample_id",
        # The split's own bookkeeping is made to *look* clean, which is the point: the checker
        # recomputes from the spectra rather than trusting it.
        train_groups=frozenset({"sample-0", "sample-1"}),
        test_groups=frozenset({"sample-2"}),
        seed=0,
    )
    findings = check_leakage(corpus, leaking)
    kinds = {f.kind for f in findings}
    assert "sample_id" in kinds
    assert any("sample-0" in f.detail for f in findings)
    with pytest.raises(UsisError, match="leaks"):
        assert_no_leakage(corpus, leaking)


def test_the_checker_recomputes_the_group_key_instead_of_trusting_the_split():
    # A split that claims disjoint instruments but whose indices straddle them must be caught.
    corpus = [_spectrum(i, f"sample-{i}", instrument=f"inst-{i % 2}") for i in range(4)]
    lying = Split(
        kind=SplitKind.INSTRUMENT_DISJOINT,
        train=(0, 1),  # inst-0 and inst-1
        test=(2, 3),  # inst-0 and inst-1 again
        group_key="instrument_id",
        train_groups=frozenset({"claimed-a"}),
        test_groups=frozenset({"claimed-b"}),
        seed=0,
    )
    findings = check_leakage(corpus, lying)
    assert any(f.kind == "instrument_id" for f in findings)


def test_a_group_key_that_cannot_be_recomputed_is_itself_a_finding(corpus):
    # The fixture corpus has one instrument, so an instrument-disjoint split over it is not
    # constructible at all. The checker says so rather than passing silently.
    lying = Split(
        kind=SplitKind.INSTRUMENT_DISJOINT,
        train=(0, 1),
        test=(2, 3),
        group_key="instrument_id",
        train_groups=frozenset({"a"}),
        test_groups=frozenset({"b"}),
        seed=0,
    )
    findings = check_leakage(corpus, lying)
    assert any(f.kind == "group_key" for f in findings)
    assert any("at least two instruments" in f.detail for f in findings)


def test_random_splits_are_checked_too_and_do_leak(corpus):
    # The checker exists to demonstrate that a random split leaks, not to exempt it.
    split = make_split(corpus, SplitKind.RANDOM, seed=0, test_fraction=0.5)
    findings = check_leakage(corpus, split)
    assert any(f.kind == "sample_id" for f in findings)


def test_inchikey_overlap_is_reported_for_identity_splits(corpus):
    lying = Split(
        kind=SplitKind.MOLECULE_DISJOINT,
        train=(0,),
        test=(1,),
        group_key="molecule identity",
        train_groups=frozenset({"a"}),
        test_groups=frozenset({"b"}),
        seed=0,
    )
    findings = check_leakage(corpus, lying)
    assert any(f.kind == "inchikey" for f in findings)
    assert any("KEY0" in f.detail for f in findings)


def test_a_near_duplicate_across_the_boundary_is_caught():
    base = np.random.default_rng(0).normal(100.0, 5.0, AXIS.n_points)
    corpus = [
        _spectrum(0, "sample-a", values=base),
        # A different physical sample by label, but a copy of the signal: exactly the duplicate a
        # group-based check cannot see.
        _spectrum(1, "sample-b", values=base * 1.01 + 0.5),
        _spectrum(2, "sample-c"),
        _spectrum(3, "sample-d"),
    ]
    split = Split(
        kind=SplitKind.SAMPLE_DISJOINT,
        train=(0, 2),
        test=(1, 3),
        group_key="sample_id",
        train_groups=frozenset({"sample-a", "sample-c"}),
        test_groups=frozenset({"sample-b", "sample-d"}),
        seed=0,
    )
    findings = check_leakage(corpus, split)
    duplicates = [f for f in findings if f.kind == "near_duplicate"]
    assert len(duplicates) == 1
    assert duplicates[0].train_index == 0
    assert duplicates[0].test_index == 1


def test_a_scaled_copy_is_caught_because_correlation_ignores_scale():
    # Rescaling and offsetting is the cheapest way to hide a duplicate from an equality check.
    base = np.random.default_rng(1).normal(50.0, 3.0, AXIS.n_points)
    corpus = [_spectrum(0, "a", values=base), _spectrum(1, "b", values=7.0 * base - 20.0)]
    split = Split(
        kind=SplitKind.SAMPLE_DISJOINT,
        train=(0,),
        test=(1,),
        group_key="sample_id",
        train_groups=frozenset({"a"}),
        test_groups=frozenset({"b"}),
        seed=0,
    )
    assert any(f.kind == "near_duplicate" for f in check_leakage(corpus, split))


def test_unrelated_spectra_are_not_flagged(corpus):
    split = make_split(corpus, SplitKind.SAMPLE_DISJOINT, seed=1)
    assert not [f for f in check_leakage(corpus, split) if f.kind == "near_duplicate"]


def test_the_threshold_is_high_enough_not_to_flag_ordinary_replicates():
    # Replicates of one sample correlate highly; a threshold that flagged them would train people
    # to ignore the check.
    rng = np.random.default_rng(2)
    base = rng.normal(100.0, 10.0, AXIS.n_points)
    a, b = base + rng.normal(0, 1.0, AXIS.n_points), base + rng.normal(0, 1.0, AXIS.n_points)
    corr = float(np.corrcoef(a, b)[0, 1])
    assert 0.9 < corr < NEAR_DUPLICATE_CORRELATION


def test_near_duplicate_search_is_skipped_loudly_when_too_large(corpus):
    split = make_split(corpus, SplitKind.SAMPLE_DISJOINT, seed=0)
    findings = check_leakage(corpus, split, max_pairs=1)
    assert any("skipped" in f.detail for f in findings)


def test_findings_render_for_a_ci_log(corpus):
    split = make_split(corpus, SplitKind.RANDOM, seed=0, test_fraction=0.5)
    text = str(check_leakage(corpus, split)[0])
    assert "sample_id" in text
