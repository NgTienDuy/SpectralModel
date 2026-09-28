"""Splits over the real committed fixtures.

The unit tests use synthetic spectra to exercise every branch. This file checks the behaviour that
actually matters: what the split machinery does with the four published datasets we load, where
the answers are decided by the data rather than by a fixture we wrote to suit ourselves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from usis.data.leakage import check_leakage
from usis.data.sources.qm9s import Qm9sSource
from usis.data.sources.rod import RodSource
from usis.data.sources.scidata_api import SciDataApiSource
from usis.data.splits import SplitError, SplitKind, make_split

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def rod() -> list:
    return [RodSource().parse_one(p) for p in sorted((FIXTURES / "rod").glob("*.rod"))]


@pytest.fixture(scope="module")
def api() -> list:
    return list(
        SciDataApiSource().parse_file(FIXTURES / "scidata_api" / "api_compounds_subset.csv.gz")
    )


@pytest.fixture(scope="module")
def qm9s() -> list:
    return list(Qm9sSource("raman").parse_file(FIXTURES / "qm9s" / "qm9s_raman_subset.csv.gz"))


def test_rod_supports_both_an_instrument_and_an_excitation_split(rod):
    # The two committed ROD records are deliberately from different instruments, labs and
    # excitations, so both splits are constructible over them.
    instrument = make_split(rod, SplitKind.INSTRUMENT_DISJOINT, seed=0, test_fraction=0.5)
    excitation = make_split(rod, SplitKind.EXCITATION_DISJOINT, seed=0, test_fraction=0.5)
    assert instrument.n_train == instrument.n_test == 1
    assert excitation.n_train == excitation.n_test == 1
    assert check_leakage(rod, instrument) == []
    # And the excitation split says out loud that it is not instrument shift.
    assert any("not instrument shift" in note for note in excitation.notes)


def test_rod_sample_ids_come_from_the_upstream_specimen(rod):
    split = make_split(rod, SplitKind.SAMPLE_DISJOINT, seed=0, test_fraction=0.5)
    assert split.test_groups <= {"rod:R060425", "rod:YE0002"}


def test_the_sci_data_set_refuses_an_instrument_disjoint_split(api):
    # One instrument, stated only in the paper. Holding it out leaves nothing to train on, so the
    # split refuses instead of returning an empty test side.
    with pytest.raises(SplitError, match="at least two instruments"):
        make_split(api, SplitKind.INSTRUMENT_DISJOINT, seed=0)


def test_a_random_split_over_the_sci_data_set_leaks_and_the_checker_proves_it(api):
    # The headline demonstration. One 4 mL vial per compound means a random split puts scans of the
    # same physical sample on both sides, on real published data, with nothing in the file to warn
    # anybody. This is the failure the whole splits module exists to prevent.
    split = make_split(api, SplitKind.RANDOM, seed=0, test_fraction=0.5)
    findings = check_leakage(api, split)
    assert any(f.kind == "sample_id" for f in findings)
    assert any("scidata-api:" in f.detail for f in findings)


def test_the_sample_disjoint_split_over_the_same_data_is_clean(api):
    split = make_split(api, SplitKind.SAMPLE_DISJOINT, seed=0, test_fraction=0.34)
    assert check_leakage(api, split) == []
    assert split.n_train + split.n_test == len(api)


def test_sample_and_molecule_splits_coincide_for_the_sci_data_set(api):
    # One vial per compound, so grouping by sample and grouping by molecule are the same partition.
    # Worth asserting because it means the set cannot distinguish the two kinds of generalisation.
    by_sample = make_split(api, SplitKind.SAMPLE_DISJOINT, seed=0, test_fraction=0.34)
    by_molecule = make_split(api, SplitKind.MOLECULE_DISJOINT, seed=0, test_fraction=0.34)
    assert set(by_sample.test) == set(by_molecule.test)


def test_sim2real_can_be_built_across_the_simulated_and_experimental_sources(qm9s, api):
    corpus = qm9s + api
    split = make_split(corpus, SplitKind.SIM2REAL, seed=0)
    assert split.n_train == len(qm9s)
    assert split.n_test == len(api)
    assert any("line broadening" in note for note in split.notes)


def test_qm9s_alone_cannot_produce_an_instrument_split(qm9s):
    with pytest.raises(SplitError, match="belong in a sim2real split"):
        make_split(qm9s, SplitKind.INSTRUMENT_DISJOINT, seed=0)


def test_resolution_shift_across_simulated_and_experimental(qm9s, rod):
    # QM9S Raman carries the measured 10 cm^-1 broadening; ROD records 4 cm^-1. Different numbers,
    # and the note records that they mean different things.
    corpus = qm9s + rod
    split = make_split(corpus, SplitKind.RESOLUTION_SHIFT, seed=0, test_fraction=0.4)
    assert not (
        {corpus[i].axis.resolution_fwhm for i in split.train}
        & {corpus[i].axis.resolution_fwhm for i in split.test}
    )
    assert any("applied line broadening" in note for note in split.notes)
