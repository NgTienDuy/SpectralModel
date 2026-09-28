"""The ROD loader, tested against two real records committed verbatim.

No test here touches the network. The fixtures' checksums are asserted so that an accidental edit
fails loudly rather than silently changing what the parser is validated against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from usis.core.errors import SpectrumValidationError
from usis.core.provenance import content_hash
from usis.core.types import (
    AxisKind,
    IntensityKind,
    Modality,
    Origin,
)
from usis.data.sources.rod import UNMAPPED_TAGS, RodSource, rod_record_url

FIXTURES = Path(__file__).parent / "fixtures" / "rod"
CHAMBERSITE = FIXTURES / "8100020.rod"
HEMATITE = FIXTURES / "1000001.rod"

# Retrieved 2026-09-03 from solsa.crystallography.net and committed unmodified.
EXPECTED_SHA256 = {
    "8100020.rod": "a9dd4f94c6162dab6099a4ab5bf59996d7c21f75056cf95af696aaf9e26bd32a",
    "1000001.rod": "26d332854140a91f1886c5ca274e4d88ae9e7e9a3499c3bc76b8d932b9b4e5e1",
}


def _only(spectra) -> object:
    """A ROD record holds exactly one spectrum; unwrap the iterator for readability."""
    items = list(spectra)
    assert len(items) == 1
    return items[0]


@pytest.fixture
def source() -> RodSource:
    return RodSource()


def test_fixtures_are_byte_identical_to_what_was_retrieved():
    for name, digest in EXPECTED_SHA256.items():
        assert content_hash(FIXTURES / name) == digest, f"{name} was modified"


def test_parses_the_chambersite_record(source):
    spectrum = source.parse_one(CHAMBERSITE)
    assert spectrum.spectrum_id == "rod:8100020"
    assert spectrum.modality is Modality.RAMAN
    assert spectrum.origin is Origin.EXPERIMENTAL
    assert spectrum.axis.kind is AxisKind.RAMAN_SHIFT_CM1
    assert spectrum.intensity_kind is IntensityKind.ARBITRARY
    assert spectrum.axis.n_points == 2297
    assert spectrum.intensity.size == 2297
    assert spectrum.axis.span == pytest.approx((195.6546, 1302.596), abs=1e-3)
    assert spectrum.axis.resolution_fwhm == pytest.approx(4.0)


def test_instrument_and_lab_are_split_keys(source):
    chambersite = source.parse_one(CHAMBERSITE).instrument
    hematite = source.parse_one(HEMATITE).instrument
    assert chambersite is not None and hematite is not None

    assert chambersite.manufacturer == "Thermo Scientific"
    assert chambersite.model == "Nicolet Almega XR"
    assert chambersite.excitation_nm == pytest.approx(532.0)
    assert chambersite.lab_id == "arizona-university"

    assert hematite.manufacturer == "HORIBA Jobin Yvon"
    assert hematite.model == "T64000"
    assert hematite.excitation_nm == pytest.approx(514.0)
    assert hematite.lab_id == "immm-maine-university"

    # The two records must land on different instrument keys, or an instrument-disjoint split
    # built on this loader silently degenerates into a random one.
    assert chambersite.instrument_id != hematite.instrument_id
    assert chambersite.lab_id != hematite.lab_id


def test_instrument_id_includes_the_location_not_only_the_model(source):
    # Two labs running the same model are not the same instrument for a transfer experiment.
    instrument = source.parse_one(CHAMBERSITE).instrument
    assert instrument is not None
    assert "nicolet-almega-xr" in instrument.instrument_id
    assert "arizona-university" in instrument.instrument_id


def test_sample_id_groups_on_the_upstream_record_not_the_rod_code(source):
    # 8100020 derives from RRUFF R060425; 1000001 from the contributor key YE0002. Grouping on the
    # ROD code would split the same specimen's 532 nm and 780 nm spectra across a random split.
    assert source.parse_one(CHAMBERSITE).sample_id == "rod:R060425"
    assert source.parse_one(HEMATITE).sample_id == "rod:YE0002"


def test_acquisition_parameters_are_mapped(source):
    chambersite = source.parse_one(CHAMBERSITE).acquisition
    hematite = source.parse_one(HEMATITE).acquisition
    assert chambersite is not None and hematite is not None
    assert chambersite.exposure_s == pytest.approx(10.0)
    assert chambersite.laser_power_mw == pytest.approx(15.0)
    # ROD reports kelvin: the JCAMP twin of this record gives 26.85 degC, which is 300.00 K.
    assert chambersite.temperature_k == pytest.approx(300.0)
    assert hematite.exposure_s == pytest.approx(600.0)
    assert hematite.laser_power_mw == pytest.approx(2.0)


def test_declared_preprocessing_becomes_the_processing_chain(source):
    spectrum = source.parse_one(CHAMBERSITE)
    names = [step.name for step in spectrum.processing]
    # This record declares a neon-lamp calibration and explicitly no background subtraction and no
    # baseline correction, so exactly one step is recorded.
    assert names == ["calibration.wavenumber"]
    step = spectrum.processing[0]
    assert step.params["standard"] == "neon_lamp"
    assert step.version == "source-declared"
    # An unknown time is recorded as unknown; a plausible wrong timestamp is worse than none.
    assert step.timestamp == "unknown"


def test_a_record_declaring_no_preprocessing_records_none(source):
    # The hematite record declares no calibration standard and no corrections applied.
    assert source.parse_one(HEMATITE).processing == ()


def test_provenance_carries_the_licence_url_and_checksum(source):
    spectrum = source.parse_one(CHAMBERSITE)
    assert spectrum.source.dataset_id == "rod"
    assert spectrum.source.licence == "CC0-1.0"
    assert spectrum.source.sha256 == EXPECTED_SHA256["8100020.rod"]
    assert spectrum.source.url == rod_record_url("8100020")


def test_labels_are_carried_with_the_formula_as_a_note(source):
    chambersite = source.parse_one(CHAMBERSITE).label
    hematite = source.parse_one(HEMATITE).label
    assert chambersite is not None and hematite is not None
    assert chambersite.name == "Chambersite"
    assert "B7 Cl Mn3 O13" in (chambersite.annotation_note or "")
    assert hematite.name == "Hematite"
    assert "Fe2 O3" in (hematite.annotation_note or "")
    # A mineral name is not a structure: no SMILES is invented for it.
    assert chambersite.smiles is None
    assert chambersite.inchikey is None


def test_axis_is_strictly_monotonic_and_read_only(source):
    spectrum = source.parse_one(CHAMBERSITE)
    assert np.all(np.diff(spectrum.axis.values) > 0)
    with pytest.raises(ValueError, match="read-only"):
        spectrum.intensity[0] = 1.0


def test_parse_is_pure_and_deterministic(source):
    text = CHAMBERSITE.read_text(encoding="utf-8")
    first = _only(source.parse(text, retrieved_at="2026-09-03T00:00:00+00:00"))
    second = _only(source.parse(text, retrieved_at="2026-09-03T00:00:00+00:00"))
    assert first.spectrum_id == second.spectrum_id
    assert np.array_equal(first.intensity, second.intensity)
    assert first.source.retrieved_at == second.source.retrieved_at


def test_a_record_without_a_spectrum_loop_is_refused(source):
    with pytest.raises(SpectrumValidationError, match="no _raman_spectrum loop"):
        _only(source.parse("data_empty\n_rod_database.code 1\n"))


def test_a_non_monotonic_axis_is_refused(source):
    payload = (
        "data_x\n_rod_database.code 1\n_raman_determination_method experimental\n"
        "_raman_measurement_device.model M\n_raman_measurement_device.location L\n"
        "loop_\n_raman_spectrum.raman_shift\n_raman_spectrum.intensity\n"
        "100 1\n102 2\n101 3\n"
    )
    with pytest.raises(SpectrumValidationError, match="not strictly monotonic"):
        _only(source.parse(payload))


def test_a_declared_range_that_contradicts_the_data_is_refused(source):
    payload = (
        "data_x\n_rod_database.code 1\n_raman_determination_method experimental\n"
        "_raman_measurement_device.model M\n_raman_measurement_device.location L\n"
        "_raman_measurement.range_min 900\n_raman_measurement.range_max 1000\n"
        "loop_\n_raman_spectrum.raman_shift\n_raman_spectrum.intensity\n"
        "100 1\n101 2\n102 3\n"
    )
    with pytest.raises(SpectrumValidationError, match="declares"):
        _only(source.parse(payload))


def test_an_experimental_record_without_a_device_is_refused(source):
    payload = (
        "data_x\n_rod_database.code 1\n_raman_determination_method experimental\n"
        "loop_\n_raman_spectrum.raman_shift\n_raman_spectrum.intensity\n"
        "100 1\n101 2\n102 3\n"
    )
    with pytest.raises(SpectrumValidationError, match="names no measurement device"):
        _only(source.parse(payload))


def test_a_predicted_record_is_marked_simulated_and_may_omit_the_device(source):
    payload = (
        "data_x\n_rod_database.code 1\n_raman_determination_method predicted\n"
        "loop_\n_raman_spectrum.raman_shift\n_raman_spectrum.intensity\n"
        "100 1\n101 2\n102 3\n"
    )
    spectrum = _only(source.parse(payload))
    assert spectrum.origin is Origin.SIMULATED
    assert spectrum.instrument is None


def test_unmapped_tags_are_declared_rather_than_dropped_silently():
    # The point of the constant is that the omission is inspectable. If a tag gains a home in the
    # core types, it must leave this dict in the same change.
    assert "_raman_measurement_device.direction_polarization" in UNMAPPED_TAGS
    assert all(reason for reason in UNMAPPED_TAGS.values())


def test_the_source_exposes_its_dataset_spec(source):
    assert source.spec.dataset_id == "rod"
    assert source.spec.vendorable is True
