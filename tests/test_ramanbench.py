"""The RamanBench catalogue, its licence gate, and the array-to-Spectrum conversion.

Fully offline: the conversion is duck-typed on arrays, so nothing here needs the optional
``raman-data`` extra installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from usis.core.errors import SpectrumValidationError, UsisError
from usis.core.types import AxisKind, IntensityKind, Modality, Origin
from usis.data.licences import lookup
from usis.data.registry import DATASETS, load_sources
from usis.data.sources.ramanbench import (
    BIOPROCESS_ANALYTE_DATASETS,
    CATALOGUE,
    LICENCE_MAP,
    RamanBenchSource,
    dataset_spec_id,
    map_licence,
    unresolved_licences,
)

CLASSIFICATION_KEY = "illicit_adulterants_sers"
# kaiser has is_grouped=False (no replicate structure); tornado has True. The pair is what makes
# the "absent group ids" distinction testable.
REGRESSION_KEY = "bioprocess_analytes_kaiser"
GROUPED_KEY = "bioprocess_analytes_tornado"


# --- the catalogue snapshot ---------------------------------------------------------------------
def test_the_snapshot_records_where_it_came_from():
    snapshot = CATALOGUE["_snapshot"]
    assert snapshot["package"] == "raman-data"
    assert snapshot["version"]
    assert snapshot["captured_at"] == "2026-09-03"
    assert snapshot["n_datasets"] == len(CATALOGUE["datasets"]) == 93


def test_the_catalogue_is_keyed_by_name_because_source_ids_collide():
    # Four adenine datasets share Zenodo record 3572359. Keying on the upstream id would have
    # merged six datasets into two, silently.
    ids = [e["source_id"] for e in CATALOGUE["datasets"].values()]
    assert len(set(ids)) < len(ids)
    adenine = [k for k in CATALOGUE["datasets"] if k.startswith("adenine_")]
    assert len(adenine) == 4
    assert len({CATALOGUE["datasets"][k]["source_id"] for k in adenine}) == 1


# --- the licence gate --------------------------------------------------------------------------
def test_unambiguous_licences_map_and_ambiguous_ones_do_not():
    assert map_licence("CC BY 4.0") == "CC-BY-4.0"
    assert map_licence("CC0 1.0") == "CC0-1.0"
    assert map_licence("research use only") == "ACADEMIC-USE-ONLY"
    for ambiguous in ("Free access", "BY-NC", "unknown", "Requested from authors", "Optica OAPA"):
        assert map_licence(ambiguous) == "UNVERIFIED"


def test_every_mapped_licence_is_in_our_vocabulary():
    for key in LICENCE_MAP.values():
        assert lookup(key).key == key


def test_unmappable_licences_block_vendoring_and_are_listed_as_a_worklist():
    blocked = unresolved_licences()
    assert blocked
    total = sum(len(v) for v in blocked.values())
    assert total == 20
    assert "Optica OAPA" in blocked
    for datasets in blocked.values():
        for key in datasets:
            spec = DATASETS.get(dataset_spec_id(key))
            assert spec.licence == "UNVERIFIED"
            assert spec.vendorable is False


def test_the_original_licence_string_is_preserved_even_when_unmappable():
    load_sources()
    spec = DATASETS.get(dataset_spec_id(next(iter(unresolved_licences()["Optica OAPA"]))))
    assert "Optica OAPA" in spec.notes
    assert "blocked from vendoring" in spec.notes


# --- the eight-instrument collection -------------------------------------------------------------
def test_the_bioprocess_collection_is_eight_datasets_one_per_spectrometer():
    assert len(BIOPROCESS_ANALYTE_DATASETS) == 8
    assert set(BIOPROCESS_ANALYTE_DATASETS) == {
        "bioprocess_analytes_anton_532",
        "bioprocess_analytes_anton_785",
        "bioprocess_analytes_kaiser",
        "bioprocess_analytes_metrohm",
        "bioprocess_analytes_mettler_toledo",
        "bioprocess_analytes_tec5",
        "bioprocess_analytes_timegate",
        "bioprocess_analytes_tornado",
    }


def test_the_whole_collection_is_vendorable():
    # All eight are CC BY 4.0, which is what makes the instrument-disjoint experiment usable.
    for key in BIOPROCESS_ANALYTE_DATASETS:
        spec = DATASETS.get(dataset_spec_id(key))
        assert spec.licence == "CC-BY-4.0"
        assert spec.vendorable is True


def test_the_eight_datasets_give_eight_distinct_instrument_keys():
    # This is the point of the collection, and the reason a dataset is used as the instrument key.
    keys = {RamanBenchSource(k).instrument().instrument_id for k in BIOPROCESS_ANALYTE_DATASETS}
    assert len(keys) == 8


def test_excitation_is_filled_in_only_where_the_catalogue_states_it():
    assert RamanBenchSource("bioprocess_analytes_anton_532").instrument().excitation_nm == 532.0
    assert RamanBenchSource("bioprocess_analytes_anton_785").instrument().excitation_nm == 785.0
    # Vendor known, wavelength not stated: left None rather than inferred from a typical setup.
    kaiser = RamanBenchSource("bioprocess_analytes_kaiser").instrument()
    assert kaiser.manufacturer == "Kaiser Optical Systems"
    assert kaiser.excitation_nm is None


# --- conversion --------------------------------------------------------------------------------
def _arrays(n=4, m=5):
    rng = np.random.default_rng(0)
    return rng.random((n, m)), np.linspace(400.0, 1800.0, m)


def test_conversion_yields_one_spectrum_per_row():
    source = RamanBenchSource(REGRESSION_KEY)
    spectra, shifts = _arrays()
    out = list(source.spectra_from_arrays(spectra, shifts))
    assert len(out) == 4
    assert out[0].axis.kind is AxisKind.RAMAN_SHIFT_CM1
    assert out[0].origin is Origin.EXPERIMENTAL
    assert out[0].intensity_kind is IntensityKind.ARBITRARY
    assert out[0].instrument.instrument_id == f"ramanbench:{REGRESSION_KEY}"


def test_regression_targets_go_to_targets_not_components():
    # Concentrations in g/L do not sum to 1; forcing them into `components` would trip the closure
    # check, which exists for mixture fractions.
    source = RamanBenchSource(REGRESSION_KEY)
    spectra, shifts = _arrays(n=2)
    targets = np.array([[10.0, 2.0, 0.5], [12.0, 1.0, 0.25]])
    out = list(
        source.spectra_from_arrays(
            spectra, shifts, targets, ["glucose", "acetate", "magnesium_sulfate"]
        )
    )
    assert out[0].label.targets == {"glucose": 10.0, "acetate": 2.0, "magnesium_sulfate": 0.5}
    assert out[0].label.components is None
    assert sum(out[0].label.targets.values()) != pytest.approx(1.0)


def test_missing_regression_targets_are_dropped_and_stated():
    source = RamanBenchSource(REGRESSION_KEY)
    spectra, shifts = _arrays(n=1)
    out = list(source.spectra_from_arrays(spectra, shifts, np.array([[1.0, np.nan]]), ["a", "b"]))
    assert out[0].label.targets == {"a": 1.0}
    assert "missing (NaN)" in out[0].label.annotation_note


def test_classification_targets_become_a_name():
    source = RamanBenchSource(CLASSIFICATION_KEY)
    spectra, shifts = _arrays(n=2)
    out = list(
        source.spectra_from_arrays(spectra, shifts, np.array([1, 0]), ["sibutramine", "sildenafil"])
    )
    assert out[0].label.name == "sildenafil"
    assert out[1].label.name == "sibutramine"
    assert out[0].label.targets is None


def test_group_ids_become_the_sample_id():
    source = RamanBenchSource(REGRESSION_KEY)
    spectra, shifts = _arrays(n=4)
    out = list(source.spectra_from_arrays(spectra, shifts, group_ids=np.array([7, 7, 8, 8])))
    assert out[0].sample_id == out[1].sample_id
    assert out[2].sample_id == out[3].sample_id
    assert out[0].sample_id != out[2].sample_id


def test_absent_group_ids_are_flagged_when_replicates_are_known_or_unknown():
    # is_grouped is True or None for 72 of the 93 datasets. Without group ids we cannot
    # reconstruct which spectra are replicates, and saying so is what lets a split refuse instead
    # of leaking.
    assert CATALOGUE["datasets"][GROUPED_KEY]["is_grouped"] is True
    source = RamanBenchSource(GROUPED_KEY)
    spectra, shifts = _arrays(n=2)
    out = list(source.spectra_from_arrays(spectra, shifts))
    assert out[0].sample_id != out[1].sample_id
    assert "replicate structure not reconstructible" in out[0].label.annotation_note


def test_no_warning_when_the_upstream_says_there_is_no_replicate_structure():
    # is_grouped=False is a checked finding, not a default, so row-unique sample ids are correct
    # here and the note would be noise.
    assert CATALOGUE["datasets"][REGRESSION_KEY]["is_grouped"] is False
    source = RamanBenchSource(REGRESSION_KEY)
    spectra, shifts = _arrays(n=2)
    out = list(source.spectra_from_arrays(spectra, shifts))
    assert out[0].label.annotation_note is None


def test_sers_datasets_get_the_sers_modality():
    assert RamanBenchSource(CLASSIFICATION_KEY).spec.modality is Modality.RAMAN_SERS
    assert RamanBenchSource(REGRESSION_KEY).spec.modality is Modality.RAMAN


# --- refusals -------------------------------------------------------------------------------------
def test_an_unknown_dataset_key_is_refused():
    with pytest.raises(UsisError, match="unknown RamanBench dataset"):
        RamanBenchSource("not_a_dataset")


def test_mismatched_shapes_are_refused():
    source = RamanBenchSource(REGRESSION_KEY)
    spectra, shifts = _arrays(n=3, m=5)
    with pytest.raises(SpectrumValidationError, match="raman_shifts has"):
        list(source.spectra_from_arrays(spectra, shifts[:3]))
    with pytest.raises(SpectrumValidationError, match="targets for"):
        list(source.spectra_from_arrays(spectra, shifts, np.zeros(2)))
    with pytest.raises(SpectrumValidationError, match="group ids for"):
        list(source.spectra_from_arrays(spectra, shifts, group_ids=np.zeros(2)))
    with pytest.raises(SpectrumValidationError, match=r"must be 2-D"):
        list(source.spectra_from_arrays(np.zeros(5), shifts))


def test_text_parse_and_record_fetch_both_refuse_with_a_pointer():
    source = RamanBenchSource(REGRESSION_KEY)
    with pytest.raises(NotImplementedError, match="spectra_from_arrays"):
        list(source.parse("anything"))
    with pytest.raises(NotImplementedError, match="raman-data"):
        source.fetch("x", __import__("pathlib").Path("unused"))


# --- manifest integration -----------------------------------------------------------------
def test_the_catalogue_ships_inside_the_package():
    # It is package data, not a repo file: an installed wheel must carry it or the manifest breaks
    # on any machine that is not this checkout.
    from usis.data.sources.ramanbench import CATALOGUE_PATH

    assert CATALOGUE_PATH.is_file()
    assert CATALOGUE_PATH.parent.name == "data"
    assert CATALOGUE_PATH.parent.parent.name == "sources"


def test_blocked_datasets_include_mapped_but_restrictive_licences():
    # 20 datasets are blocked because their licence string could not be mapped at all; three more
    # are blocked by licences we *did* map and that still forbid redistribution here
    # (ACADEMIC-USE-ONLY, and CC BY-NC-ND whose no-derivatives clause we treat as blocking).
    manifest = load_sources()
    blocked = [k for k, v in manifest.items() if not v.vendorable]
    unmappable = sum(len(v) for v in unresolved_licences().values())
    assert unmappable == 20
    assert len(blocked) == 23
    restrictive = {manifest[k].licence for k in blocked} - {"UNVERIFIED"}
    assert restrictive == {"ACADEMIC-USE-ONLY", "CC-BY-NC-ND-4.0"}


def test_all_93_datasets_are_registered_and_verification_is_partial():
    manifest = load_sources()
    registered = [k for k in manifest if k.startswith("ramanbench:")]
    assert len(registered) == 93
    # PARTIAL, not VERIFIED: we read the catalogue, never the spectra.
    assert all(manifest[k].verification == "PARTIAL" for k in registered)
