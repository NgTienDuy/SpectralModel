"""Split construction, and the refusals that are the point of it."""

from __future__ import annotations

import numpy as np
import pytest

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
from usis.data.registry import DATASETS, DatasetSpec, ReplicateStructure, register_dataset
from usis.data.splits import Split, SplitError, SplitKind, group_keys, make_split

AXIS = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, np.linspace(400.0, 1800.0, 32))


def _dataset(dataset_id: str, replicates: str) -> str:
    if dataset_id not in DATASETS:
        register_dataset(
            DatasetSpec(
                dataset_id=dataset_id,
                name=dataset_id,
                modality=Modality.RAMAN,
                origin=Origin.EXPERIMENTAL,
                licence="CC0-1.0",
                replicate_structure=replicates,
            )
        )
    return dataset_id


def _spectrum(
    index: int,
    *,
    dataset: str,
    sample: str,
    instrument: str | None = "inst-a",
    lab: str | None = "lab-a",
    excitation: float | None = 532.0,
    resolution: float | None = 4.0,
    name: str | None = None,
    inchikey: str | None = None,
    origin: Origin = Origin.EXPERIMENTAL,
    seed: int | None = None,
) -> Spectrum:
    rng = np.random.default_rng(index if seed is None else seed)
    axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, AXIS.values, resolution_fwhm=resolution)
    spec = None
    if instrument is not None:
        spec = InstrumentSpec(instrument_id=instrument, lab_id=lab, excitation_nm=excitation)
    return Spectrum(
        spectrum_id=f"s{index}",
        modality=Modality.RAMAN,
        origin=origin,
        axis=axis,
        intensity=rng.normal(100.0, 5.0, axis.n_points),
        intensity_kind=IntensityKind.ARBITRARY,
        sample_id=sample,
        source=SourceRecord(dataset_id=dataset, licence="CC0-1.0", retrieved_at="t"),
        instrument=spec,
        label=(
            None
            if (name is None and inchikey is None)
            else GroundTruth(name=name, inchikey=inchikey)
        ),
    )


@pytest.fixture
def corpus() -> list[Spectrum]:
    ds = _dataset("split-test-known", ReplicateStructure.KNOWN)
    out = []
    for i in range(12):
        sample = f"sample-{i // 3}"  # 4 samples, 3 replicates each
        out.append(
            _spectrum(
                i,
                dataset=ds,
                sample=sample,
                instrument=f"inst-{i // 6}",  # 2 instruments
                lab=f"lab-{i // 6}",
                excitation=532.0 if i % 2 == 0 else 785.0,
                resolution=4.0 if i < 6 else 8.0,
                name=f"molecule-{i // 3}",
            )
        )
    return out


# --- the required-argument rule ------------------------------------------------------------------
def test_split_kind_has_no_default(corpus):
    with pytest.raises(TypeError):
        make_split(corpus, seed=0)  # type: ignore[call-arg]


def test_seed_is_required_and_recorded(corpus):
    split = make_split(corpus, SplitKind.SAMPLE_DISJOINT, seed=7)
    assert split.seed == 7
    with pytest.raises(TypeError):
        make_split(corpus, SplitKind.SAMPLE_DISJOINT)  # type: ignore[call-arg]


# --- group-preserving behaviour --------------------------------------------------------------
def test_sample_disjoint_keeps_replicates_together(corpus):
    split = make_split(corpus, SplitKind.SAMPLE_DISJOINT, seed=0)
    train_samples = {corpus[i].sample_id for i in split.train}
    test_samples = {corpus[i].sample_id for i in split.test}
    assert not (train_samples & test_samples)
    assert split.n_train + split.n_test == len(corpus)


def test_a_split_is_reproducible_from_its_seed(corpus):
    a = make_split(corpus, SplitKind.SAMPLE_DISJOINT, seed=3)
    b = make_split(corpus, SplitKind.SAMPLE_DISJOINT, seed=3)
    assert a.train == b.train and a.test == b.test and a.test_groups == b.test_groups


def test_the_seed_actually_changes_the_partition(corpus):
    # With four groups and a 20% target only one group lands in test, so two given seeds can
    # coincide; what must hold is that the seed reaches more than one partition.
    partitions = {
        make_split(corpus, SplitKind.SAMPLE_DISJOINT, seed=s).test_groups for s in range(8)
    }
    assert len(partitions) > 1


def test_instrument_and_lab_splits_group_correctly(corpus):
    for kind, key in (
        (SplitKind.INSTRUMENT_DISJOINT, lambda s: s.instrument.instrument_id),
        (SplitKind.LAB_DISJOINT, lambda s: s.instrument.lab_id),
    ):
        split = make_split(corpus, kind, seed=1, test_fraction=0.5)
        assert not ({key(corpus[i]) for i in split.train} & {key(corpus[i]) for i in split.test})


def test_excitation_split_is_named_separately_from_instrument(corpus):
    split = make_split(corpus, SplitKind.EXCITATION_DISJOINT, seed=1, test_fraction=0.5)
    assert not (
        {corpus[i].excitation_nm for i in split.train}
        & {corpus[i].excitation_nm for i in split.test}
    )
    assert any("not instrument shift" in note for note in split.notes)


def test_resolution_shift_groups_by_recorded_resolution(corpus):
    split = make_split(corpus, SplitKind.RESOLUTION_SHIFT, seed=1, test_fraction=0.5)
    assert not (
        {corpus[i].axis.resolution_fwhm for i in split.train}
        & {corpus[i].axis.resolution_fwhm for i in split.test}
    )


def test_sim2real_trains_on_simulated_and_tests_on_experimental():
    ds = _dataset("split-test-none", ReplicateStructure.NONE)
    corpus = [
        _spectrum(i, dataset=ds, sample=f"m{i}", origin=Origin.SIMULATED, instrument=None)
        for i in range(4)
    ] + [_spectrum(10 + i, dataset=ds, sample=f"m{i}") for i in range(4)]
    split = make_split(corpus, SplitKind.SIM2REAL, seed=0)
    assert all(corpus[i].origin is Origin.SIMULATED for i in split.train)
    assert all(corpus[i].origin is Origin.EXPERIMENTAL for i in split.test)
    assert any("line broadening" in note for note in split.notes)


def test_open_set_holds_out_whole_identities_and_says_so(corpus):
    split = make_split(corpus, SplitKind.OPEN_SET, seed=0)
    assert not (
        {corpus[i].label.name for i in split.train} & {corpus[i].label.name for i in split.test}
    )
    assert any("Score p_none" in note for note in split.notes)


def test_random_carries_a_warning_that_it_may_not_be_reported(corpus):
    split = make_split(corpus, SplitKind.RANDOM, seed=0)
    assert any("smoke test only" in note for note in split.notes)
    assert any("No reported result may use it" in note for note in split.notes)


# --- the refusals ---------------------------------------------------------------------------------
def test_sample_disjoint_refuses_when_replicate_structure_is_unknown():
    ds = _dataset("split-test-unknown", ReplicateStructure.UNKNOWN)
    corpus = [_spectrum(i, dataset=ds, sample=f"row-{i}") for i in range(6)]
    with pytest.raises(SplitError, match="do not record it"):
        make_split(corpus, SplitKind.SAMPLE_DISJOINT, seed=0)
    # The coarser split the message recommends does work.
    corpus = [_spectrum(i, dataset=ds, sample=f"row-{i}", name=f"mol-{i // 2}") for i in range(6)]
    assert make_split(corpus, SplitKind.MOLECULE_DISJOINT, seed=0).n_test > 0


def test_instrument_disjoint_refuses_with_a_single_instrument():
    ds = _dataset("split-test-known", ReplicateStructure.KNOWN)
    corpus = [_spectrum(i, dataset=ds, sample=f"s{i}", instrument="only-one") for i in range(6)]
    with pytest.raises(SplitError, match="at least two instruments"):
        make_split(corpus, SplitKind.INSTRUMENT_DISJOINT, seed=0)


def test_instrument_disjoint_refuses_on_simulated_spectra():
    ds = _dataset("split-test-none", ReplicateStructure.NONE)
    corpus = [
        _spectrum(i, dataset=ds, sample=f"s{i}", origin=Origin.SIMULATED, instrument=None)
        for i in range(4)
    ]
    with pytest.raises(SplitError, match="belong in a sim2real split"):
        make_split(corpus, SplitKind.INSTRUMENT_DISJOINT, seed=0)


def test_molecule_disjoint_refuses_unlabelled_spectra():
    ds = _dataset("split-test-known", ReplicateStructure.KNOWN)
    corpus = [_spectrum(i, dataset=ds, sample=f"s{i}") for i in range(4)]
    with pytest.raises(SplitError, match="needs an identity per spectrum"):
        make_split(corpus, SplitKind.MOLECULE_DISJOINT, seed=0)


def test_lab_disjoint_refuses_without_labs():
    ds = _dataset("split-test-known", ReplicateStructure.KNOWN)
    corpus = [_spectrum(i, dataset=ds, sample=f"s{i}", lab=None) for i in range(4)]
    with pytest.raises(SplitError, match="does not record one"):
        make_split(corpus, SplitKind.LAB_DISJOINT, seed=0)


def test_resolution_shift_refuses_when_resolution_is_unknown():
    ds = _dataset("split-test-known", ReplicateStructure.KNOWN)
    corpus = [_spectrum(i, dataset=ds, sample=f"s{i}", resolution=None) for i in range(4)]
    with pytest.raises(SplitError, match="needs a resolution per spectrum"):
        make_split(corpus, SplitKind.RESOLUTION_SHIFT, seed=0)


def test_excitation_disjoint_refuses_with_one_wavelength():
    ds = _dataset("split-test-known", ReplicateStructure.KNOWN)
    corpus = [_spectrum(i, dataset=ds, sample=f"s{i}", excitation=785.0) for i in range(4)]
    with pytest.raises(SplitError, match="at least two excitation wavelengths"):
        make_split(corpus, SplitKind.EXCITATION_DISJOINT, seed=0)


def test_sim2real_refuses_without_both_sides():
    ds = _dataset("split-test-known", ReplicateStructure.KNOWN)
    corpus = [_spectrum(i, dataset=ds, sample=f"s{i}") for i in range(4)]
    with pytest.raises(SplitError, match="needs both simulated and experimental"):
        make_split(corpus, SplitKind.SIM2REAL, seed=0)


def test_scaffold_disjoint_refuses_and_names_what_is_missing(corpus):
    with pytest.raises(SplitError, match="RDKit"):
        make_split(corpus, SplitKind.SCAFFOLD_DISJOINT, seed=0)


def test_an_empty_corpus_and_a_silly_fraction_are_refused(corpus):
    with pytest.raises(SplitError, match="empty corpus"):
        make_split([], SplitKind.RANDOM, seed=0)
    with pytest.raises(SplitError, match="test_fraction must be"):
        make_split(corpus, SplitKind.RANDOM, seed=0, test_fraction=1.5)


def test_a_split_object_rejects_straddling_groups():
    with pytest.raises(SplitError, match="appear on both sides"):
        Split(
            kind=SplitKind.SAMPLE_DISJOINT,
            train=(0,),
            test=(1,),
            group_key="sample_id",
            train_groups=frozenset({"a"}),
            test_groups=frozenset({"a"}),
            seed=0,
        )


def test_group_keys_can_be_asked_without_building_a_split(corpus):
    assert len(group_keys(corpus, SplitKind.SAMPLE_DISJOINT)) == len(corpus)
    assert len(set(group_keys(corpus, SplitKind.INSTRUMENT_DISJOINT))) == 2
