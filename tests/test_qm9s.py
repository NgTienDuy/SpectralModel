"""The QM9S loader: the first simulated source, tested against real published rows. Offline."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from usis.core.errors import SpectrumValidationError
from usis.core.types import AxisKind, IntensityKind, Modality, Origin
from usis.data.sources.qm9s import (
    AXIS_MAX_CM1,
    AXIS_MIN_CM1,
    AXIS_N_POINTS,
    BROADENING_FWHM_CM1,
    LEVEL_OF_THEORY,
    Qm9sSource,
)

FIXTURES = Path(__file__).parent / "fixtures" / "qm9s"


def _fixture(kind: str) -> Path:
    return FIXTURES / f"qm9s_{kind}_subset.csv.gz"


def _payload(kind: str) -> str:
    with gzip.open(_fixture(kind), "rt", encoding="utf-8") as handle:
        return handle.read()


def _spectra(kind: str) -> list:
    return list(Qm9sSource(kind).parse_file(_fixture(kind)))


@pytest.fixture
def ir() -> list:
    return _spectra("ir")


@pytest.fixture
def raman() -> list:
    return _spectra("raman")


# --- simulation is not measurement --------------------------------------------------------------
@pytest.mark.parametrize("kind", ["ir", "raman"])
def test_spectra_are_simulated_and_name_no_instrument(kind):
    for spectrum in _spectra(kind):
        assert spectrum.origin is Origin.SIMULATED
        assert spectrum.instrument is None
        assert spectrum.acquisition is None
        assert spectrum.excitation_nm is None


def test_the_level_of_theory_is_recorded_as_provenance(ir):
    step = ir[0].processing[0]
    assert step.name == "simulate.dft_frequency"
    assert step.params["functional"] == "B3LYP"
    assert step.params["basis_set"] == "def-TZVP"
    assert step.params["package"] == LEVEL_OF_THEORY["package"]


def test_the_broadening_is_recorded_as_measured_by_us_not_published(ir, raman):
    for spectra, kind in ((ir, "ir"), (raman, "raman")):
        step = spectra[0].processing[1]
        assert step.name == "simulate.broadening"
        assert step.params["lineshape"] == "lorentzian"
        assert step.params["fwhm_cm1"] == BROADENING_FWHM_CM1[kind]
        # The distinction matters: the readme states the grid but never the width, so this number
        # is ours and must not be mistaken for a published figure.
        assert step.version == "measured"
        assert step.params["source_states_width"] is False


# --- the axis distinction -----------------------------------------------------------------------
def test_ir_and_raman_use_different_axis_kinds_for_the_same_numbers(ir, raman):
    # A mode at 1650 cm^-1 is at Raman shift 1650 and at absolute wavenumber 1650. Same value,
    # different quantity; conflating them is F01, which is why the axis kind is part of the type.
    assert ir[0].axis.kind is AxisKind.ABSOLUTE_WAVENUMBER_CM1
    assert raman[0].axis.kind is AxisKind.RAMAN_SHIFT_CM1
    assert ir[0].axis.span == raman[0].axis.span


@pytest.mark.parametrize("kind", ["ir", "raman"])
def test_axis_is_the_published_grid(kind):
    axis = _spectra(kind)[0].axis
    assert axis.n_points == AXIS_N_POINTS
    assert axis.span == (AXIS_MIN_CM1, AXIS_MAX_CM1)
    assert axis.uniform is True
    assert axis.mean_step == pytest.approx(1.0)


def test_resolution_records_the_broadening_not_the_grid_step(ir, raman):
    # Contrast with the Sci Data loader, which leaves resolution_fwhm unset because the only number
    # on offer there was the sampling interval. Here the number is a real linewidth we measured, so
    # recording it is what lets a sim-to-real step convolve on top of it correctly.
    assert ir[0].axis.resolution_fwhm == pytest.approx(15.0)
    assert raman[0].axis.resolution_fwhm == pytest.approx(10.0)
    assert ir[0].axis.mean_step == pytest.approx(1.0)


def test_the_measured_broadening_is_reproducible_from_the_fixture(ir):
    # Re-derive the FWHM from the committed bytes, so the constant cannot drift away from the data
    # it was measured on. Isolated peak, half-maximum crossings interpolated between grid points.
    wavenumbers = ir[0].axis.values
    intensity = ir[0].intensity
    peak = int(np.argmax(intensity))
    half = intensity[peak] / 2.0
    left = peak
    while intensity[left] > half:
        left -= 1
    right = peak
    while intensity[right] > half:
        right += 1
    lo = np.interp(half, intensity[left : left + 2], wavenumbers[left : left + 2])
    hi = np.interp(
        half, intensity[right - 1 : right + 1][::-1], wavenumbers[right - 1 : right + 1][::-1]
    )
    assert hi - lo == pytest.approx(BROADENING_FWHM_CM1["ir"], abs=1.5)


# --- identity and grouping ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["ir", "raman"])
def test_rows_map_to_qm9s_molecule_indices(kind):
    spectra = _spectra(kind)
    assert [s.spectrum_id for s in spectra] == [f"qm9s-{kind}:{i}" for i in range(3)]
    assert [s.sample_id for s in spectra] == [f"qm9s:{i}" for i in range(3)]


def test_ir_and_raman_of_one_molecule_share_a_sample_id(ir, raman):
    # The pairing is what makes a joint Raman+IR conditioning experiment possible at all, and what
    # a split must not break.
    assert {s.sample_id for s in ir} == {s.sample_id for s in raman}


def test_the_missing_structure_is_stated_rather_than_invented(ir):
    label = ir[0].label
    assert label.smiles is None
    assert "qm9s_csv.zip" in label.annotation_note


def test_supplying_a_smiles_map_upgrades_the_label():
    source = Qm9sSource("ir", smiles_by_index={0: "C", 1: "N"})
    spectra = list(source.parse_file(_fixture("ir")))
    assert spectra[0].label.smiles == "C"
    assert spectra[1].label.smiles == "N"
    assert spectra[2].label.smiles is None


# --- basic facts ---------------------------------------------------------------------------------
def test_modalities_and_intensity_kind(ir, raman):
    assert ir[0].modality is Modality.IR_TRANSMISSION
    assert raman[0].modality is Modality.RAMAN
    # Unnormalised km/mol and A^4/amu are neither absorbance nor counts.
    assert ir[0].intensity_kind is IntensityKind.ARBITRARY
    assert raman[0].intensity_kind is IntensityKind.ARBITRARY


def test_provenance_cites_the_article_doi_the_publisher_asked_for(ir):
    # The readme asks explicitly that the article DOI be cited rather than the figshare one.
    assert ir[0].source.doi == "10.1038/s43588-023-00550-y"
    assert ir[0].source.licence == "CC-BY-4.0"
    assert ir[0].source.dataset_id == "qm9s-ir"


def test_intensities_are_finite_and_non_negative(ir, raman):
    for spectra in (ir, raman):
        for spectrum in spectra:
            assert spectrum.intensity.size == AXIS_N_POINTS
            assert np.all(np.isfinite(spectrum.intensity))
            assert np.all(spectrum.intensity >= 0.0)


def test_parse_from_string_matches_parse_from_file(ir):
    from_string = list(Qm9sSource("ir").parse(_payload("ir")))
    assert len(from_string) == len(ir)
    assert np.array_equal(from_string[0].intensity, ir[0].intensity)


# --- refusals -------------------------------------------------------------------------------------
def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="must be 'ir' or 'raman'"):
        Qm9sSource("uv")


def test_an_empty_csv_is_refused():
    with pytest.raises(SpectrumValidationError, match="empty"):
        list(Qm9sSource("ir").parse(""))


def test_a_header_without_the_index_column_is_refused():
    with pytest.raises(SpectrumValidationError, match="empty index column"):
        list(Qm9sSource("ir").parse("500.0,501.0\n"))


def test_a_header_with_the_wrong_column_count_is_refused():
    with pytest.raises(SpectrumValidationError, match="expected 3501 wavenumber columns"):
        list(Qm9sSource("ir").parse(",500.0,501.0\n"))


def test_a_header_with_a_shifted_axis_is_refused():
    header = "," + ",".join(str(float(v)) for v in range(1, AXIS_N_POINTS + 1)) + "\n"
    with pytest.raises(SpectrumValidationError, match="published range"):
        list(Qm9sSource("ir").parse(header))


def test_a_short_row_is_refused_rather_than_padded():
    header = _payload("ir").split("\n", 1)[0]
    row = "0," + ",".join("1.0" for _ in range(AXIS_N_POINTS - 1))
    with pytest.raises(SpectrumValidationError, match="fields but the header declares"):
        list(Qm9sSource("ir").parse(header + "\n" + row + "\n"))


def test_a_non_numeric_molecule_index_is_refused():
    header = _payload("ir").split("\n", 1)[0]
    row = "molecule," + ",".join("1.0" for _ in range(AXIS_N_POINTS))
    with pytest.raises(SpectrumValidationError, match="not a molecule index"):
        list(Qm9sSource("ir").parse(header + "\n" + row + "\n"))


def test_fetch_refuses_because_each_file_is_one_bulk_download():
    with pytest.raises(NotImplementedError, match=r"8.7 GB"):
        Qm9sSource("raman").fetch("0", Path("unused"))


def test_both_datasets_are_registered_and_vendorable():
    assert Qm9sSource("ir").spec.dataset_id == "qm9s-ir"
    assert Qm9sSource("raman").spec.dataset_id == "qm9s-raman"
    assert Qm9sSource("ir").spec.vendorable is True
