"""The Sci Data API-compounds loader, tested against six real rows of the published CSV.

Offline. The fixture is a row subset of the 193 MB original: the header and six rows are the
publisher's bytes unchanged, and no value was altered.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pytest

from usis.core.errors import SpectrumValidationError
from usis.core.types import AxisKind, IntensityKind, Modality, Origin
from usis.data.sources.scidata_api import (
    AXIS_MAX_CM1,
    AXIS_MIN_CM1,
    AXIS_N_POINTS,
    COMPOUNDS,
    INSTRUMENT,
    SciDataApiSource,
    normalise_label,
)

FIXTURE = Path(__file__).parent / "fixtures" / "scidata_api" / "api_compounds_subset.csv.gz"


@pytest.fixture
def source() -> SciDataApiSource:
    return SciDataApiSource()


@pytest.fixture
def spectra(source: SciDataApiSource) -> list:
    return list(source.parse_file(FIXTURE))


def _payload() -> str:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        return handle.read()


# --- the published table ----------------------------------------------------------------------
def test_the_compound_table_matches_the_papers_totals():
    # 32 compounds and 3,510 spectra are the figures the paper reports; the table was transcribed
    # programmatically from the published xlsx, and this is the independent check that it is whole.
    assert len(COMPOUNDS) == 32
    assert sum(c.n_samples for c in COMPOUNDS.values()) == 3510


def test_a_few_acquisition_values_match_the_papers_table_1():
    assert COMPOUNDS["acetone"].exposure_s == pytest.approx(7.0)
    assert COMPOUNDS["acetone"].n_samples == 120
    assert COMPOUNDS["cyclohexane"].n_samples == 145
    assert COMPOUNDS["n-heptane"].exposure_s == pytest.approx(45.0)
    assert COMPOUNDS["toluene"].n_samples == 127


def test_label_normalisation_reconciles_the_csv_and_the_table():
    # The CSV and the product table disagree on spacing; every form must join.
    assert normalise_label("2 - Propanol") == "2-propanol"
    assert normalise_label('"2, 2 - Dimethoxy Propane"') == "2,2-dimethoxypropane"
    assert normalise_label('"1,3-Dimethyl-2-imidazolidinone"') == "1,3-dimethyl-2-imidazolidinone"
    for key in ("2-propanol", "2,2-dimethoxypropane", "1,3-dimethyl-2-imidazolidinone"):
        assert key in COMPOUNDS


# --- parsing ------------------------------------------------------------------------------------
def test_yields_one_spectrum_per_row(spectra):
    assert len(spectra) == 6
    assert [s.spectrum_id for s in spectra] == [f"scidata-api:{i:05d}" for i in range(6)]


def test_axis_is_the_published_grid(spectra):
    axis = spectra[0].axis
    assert axis.kind is AxisKind.RAMAN_SHIFT_CM1
    assert axis.n_points == AXIS_N_POINTS
    assert axis.span == (AXIS_MIN_CM1, AXIS_MAX_CM1)
    assert axis.uniform is True
    assert axis.mean_step == pytest.approx(1.0)


def test_resolution_is_left_unset_because_the_paper_reports_a_sample_spacing(spectra):
    # The paper calls 1 cm^-1 the "average spectral resolution", but 3276 points across
    # 150-3425 cm^-1 is a 1 cm^-1 grid step. Recording it as an instrument FWHM would assert a
    # resolution nobody measured and would corrupt any resolution-shift experiment (F07).
    assert spectra[0].axis.resolution_fwhm is None


def test_every_spectrum_shares_the_one_instrument(spectra):
    assert {s.instrument for s in spectra} == {INSTRUMENT}
    assert INSTRUMENT.excitation_nm == pytest.approx(785.0)
    # A dataset with one instrument cannot support an instrument-disjoint split, and the loader
    # must make that impossible to miss rather than fake diversity.
    assert len({s.instrument.instrument_id for s in spectra}) == 1


def test_sample_id_groups_scans_of_one_vial(spectra):
    # One 4 mL vial per compound; the repeated scans are replicates of one physical sample. Two
    # rows of a compound must therefore share a sample_id, or a random split leaks.
    expected = "scidata-api:1,3-dimethyl-2-imidazolidinone"
    assert spectra[0].sample_id == spectra[1].sample_id == expected
    assert spectra[2].sample_id == spectra[3].sample_id == "scidata-api:2-propanol"
    assert len({s.sample_id for s in spectra}) == 3


def test_declared_instrument_pretreatment_is_recorded_not_assumed_away(spectra):
    # The paper states dark noise subtraction, cosmic ray filtering and intensity correction were
    # applied. An empty chain would assert that nothing was, which is false.
    names = [step.name for step in spectra[0].processing]
    assert names == [
        "dark_noise.subtraction",
        "despike.cosmic_ray_filter",
        "response.intensity_correction",
    ]
    assert all(step.version == "source-declared" for step in spectra[0].processing)
    assert all(step.params["software"] == "iC Raman 4.1" for step in spectra[0].processing)


def test_labels_join_to_the_published_names(spectra):
    names = [s.label.name for s in spectra]
    assert names[0] == "1,3-Dimethyl-2-imidazolidinone"
    assert names[2] == "2-Propanol"
    assert names[4] == "2,2-Dimethoxypropane"
    assert "formula=C3H8O" in spectra[2].label.annotation_note


def test_acquisition_exposure_comes_from_the_published_table(spectra):
    assert spectra[0].acquisition.exposure_s == pytest.approx(3.0)
    assert spectra[2].acquisition.exposure_s == pytest.approx(5.0)


def test_intensity_is_arbitrary_not_counts(spectra):
    # Intensity correction was applied but no absolute radiometric calibration, so calling these
    # counts would imply a Poisson noise model we have no basis for.
    assert spectra[0].intensity_kind is IntensityKind.ARBITRARY


def test_basic_spectrum_facts(spectra):
    first = spectra[0]
    assert first.modality is Modality.RAMAN
    assert first.origin is Origin.EXPERIMENTAL
    assert first.intensity.size == AXIS_N_POINTS
    assert np.all(np.isfinite(first.intensity))
    assert first.source.dataset_id == "scidata-api-raman"
    assert first.source.licence == "CC-BY-4.0"
    assert first.source.doi == "10.1038/s41597-025-04848-6"


def test_parse_from_a_string_matches_parse_from_the_file(source, spectra):
    from_string = list(source.parse(_payload()))
    assert len(from_string) == len(spectra)
    assert np.array_equal(from_string[0].intensity, spectra[0].intensity)


# --- refusals -----------------------------------------------------------------------------------
def test_an_empty_csv_is_refused(source):
    with pytest.raises(SpectrumValidationError, match="empty"):
        list(source.parse(""))


def test_a_header_without_a_label_column_is_refused(source):
    with pytest.raises(SpectrumValidationError, match="must be 'label'"):
        list(source.parse("150.0,151.0,something\n"))


def test_a_header_with_the_wrong_number_of_columns_is_refused(source):
    with pytest.raises(SpectrumValidationError, match="expected 3276 wavenumber columns"):
        list(source.parse("150.0,151.0,label\n"))


def test_a_header_with_a_shifted_axis_is_refused(source):
    header = ",".join(str(float(v)) for v in range(1, AXIS_N_POINTS + 1)) + ",label\n"
    with pytest.raises(SpectrumValidationError, match="published range"):
        list(source.parse(header))


def _real_header() -> str:
    """The fixture's real header row, reused to build deliberately malformed rows."""
    return _payload().split("\n", 1)[0]


def _row(n_values: int, label: str) -> str:
    """A synthetic data row with ``n_values`` intensities and the given raw label field."""
    return ",".join("1.0" for _ in range(n_values)) + "," + label


def test_a_short_row_is_refused_rather_than_padded(source):
    payload = _real_header() + "\n" + _row(AXIS_N_POINTS - 1, "Toluene") + "\n"
    with pytest.raises(SpectrumValidationError, match="fields but the header declares"):
        list(source.parse(payload))


def test_a_long_row_is_refused_too(source):
    payload = _real_header() + "\n" + _row(AXIS_N_POINTS + 1, "Toluene") + "\n"
    with pytest.raises(SpectrumValidationError, match="fields but the header declares"):
        list(source.parse(payload))


def test_an_empty_label_is_refused(source):
    payload = _real_header() + "\n" + _row(AXIS_N_POINTS, "") + "\n"
    with pytest.raises(SpectrumValidationError, match="empty label"):
        list(source.parse(payload))


def test_an_unjoined_label_still_loads_with_the_gap_visible(source):
    payload = _real_header() + "\n" + _row(AXIS_N_POINTS, "Unobtainium") + "\n"
    spectrum = next(iter(source.parse(payload)))
    assert spectrum.label.name == "Unobtainium"
    assert "not matched" in spectrum.label.annotation_note
    assert spectrum.acquisition is None


def test_fetch_refuses_because_the_dataset_is_one_bulk_file(source, tmp_path):
    with pytest.raises(NotImplementedError, match="single 193 MB CSV"):
        source.fetch("anything", tmp_path / "x.csv")


def test_gz_and_plain_csv_parse_identically(source, tmp_path):
    plain = tmp_path / "subset.csv"
    plain.write_text(_payload(), encoding="utf-8")
    from_plain = list(source.parse_file(plain))
    from_gz = list(source.parse_file(FIXTURE))
    assert len(from_plain) == len(from_gz) == 6
    assert np.array_equal(from_plain[3].intensity, from_gz[3].intensity)
    # The checksums differ because the bytes differ; that is provenance working, not a bug.
    assert from_plain[0].source.sha256 != from_gz[0].source.sha256


def test_the_source_exposes_its_dataset_spec(source):
    assert source.spec.dataset_id == "scidata-api-raman"
    assert source.spec.vendorable is True
    assert io  # keep the import meaningful for the streaming path
