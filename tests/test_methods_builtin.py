"""The two built-in methods: one trivial and established, one deliberately hypothesised."""

from __future__ import annotations

import numpy as np
import pytest

from usis.core.errors import AssumptionViolationError
from usis.core.provenance import RunContext
from usis.core.types import (
    AxisKind,
    AxisSpec,
    ClaimType,
    FailureCode,
    IntensityKind,
    Maturity,
    MeasurementVerdict,
    Modality,
    Origin,
    SourceRecord,
    Spectrum,
    TaskKind,
)
from usis.measurement.artifacts import Saturation
from usis.measurement.diagnostics import (
    autocorrelation_length,
    band_statistics,
    median_band_spacing,
)
from usis.methods.applicability import Verdict
from usis.methods.base import execute
from usis.methods.builtin.broadening_origin import (
    E2_CORPUS_RATIOS,
    BroadeningOrigin,
    CalibrationError,
    RatioCalibration,
)
from usis.methods.builtin.saturation_audit import SaturationAudit
from usis.methods.inputs import MethodInputs
from usis.methods.registry import METHODS, load_builtin_methods, select_methods

CTX = RunContext(
    run_id="test", started_at="2026-09-03T00:00:00+00:00", seed=0, git_sha="0" * 40, git_dirty="no"
)


def _banded(
    source: SourceRecord,
    *,
    fwhm_cm1: float,
    shape: str,
    n_bands: int = 12,
    noise: float = 0.5,
    resolution_fwhm: float | None = 4.0,
    spectrum_id: str = "band-0001",
) -> Spectrum:
    """A synthetic spectrum of identical bands with a controllable line shape.

    Gaussian and Lorentzian bands of the *same* FWHM differ only in their tails, which is exactly
    the discrimination BroadeningOrigin conjectures is readable from one ratio.
    """
    axis_values = np.linspace(200.0, 1800.0, 3201)
    axis = AxisSpec.from_values(
        AxisKind.RAMAN_SHIFT_CM1, axis_values, resolution_fwhm=resolution_fwhm
    )
    centres = np.linspace(300.0, 1700.0, n_bands)
    intensity = np.full_like(axis_values, 10.0)
    half = fwhm_cm1 / 2.0
    for centre in centres:
        offset = axis_values - centre
        if shape == "gaussian":
            sigma = fwhm_cm1 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            intensity += 500.0 * np.exp(-0.5 * (offset / sigma) ** 2)
        else:
            intensity += 500.0 / (1.0 + (offset / half) ** 2)
    intensity += np.random.default_rng(3).normal(0.0, noise, axis_values.size)
    return Spectrum(
        spectrum_id=spectrum_id,
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=axis,
        intensity=intensity,
        intensity_kind=IntensityKind.COUNTS,
        sample_id=f"sample-{spectrum_id}",
        source=source,
    )


# ------------------------------------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------------------------------------
def test_the_builtin_package_is_discovered_not_listed() -> None:
    loaded = load_builtin_methods()
    assert "saturation_audit" in loaded
    assert "broadening_origin" in loaded
    assert "audit.saturation@1.0.0" in METHODS
    assert "shape.broadening_origin@0.2.0" in METHODS


def test_the_hypothesized_method_is_excluded_from_a_surfaceable_selection() -> None:
    load_builtin_methods()
    surfaceable = select_methods(maturities=[Maturity.ESTABLISHED, Maturity.EMERGING])
    keys = [entry.key for entry in surfaceable]
    assert "audit.saturation@1.0.0" in keys
    assert "shape.broadening_origin@0.2.0" not in keys


def test_the_established_entry_states_what_its_evidence_actually_is() -> None:
    load_builtin_methods()
    entry = METHODS.get("audit.saturation@1.0.0")
    assert entry.maturity is Maturity.ESTABLISHED
    assert entry.evidence
    assert "D14" in entry.notes
    assert "not a third-party validation" in entry.notes


# ------------------------------------------------------------------------------------------------
# SaturationAudit
# ------------------------------------------------------------------------------------------------
def test_saturation_audit_recovers_injected_clipping(source: SourceRecord) -> None:
    """The reproduction the ESTABLISHED registration cites: inject a known ceiling, recover it."""
    clean = _banded(source, fwhm_cm1=12.0, shape="gaussian")
    clipped = Saturation(level_relative=0.5).apply(clean, seed=1)

    clean_result = execute(SaturationAudit(), MethodInputs.of(clean), CTX)
    assert clean_result.labels["verdict"] == str(MeasurementVerdict.USABLE)
    assert clean_result.values["saturated_fraction"] == 0.0
    assert clean_result.all_failure_codes == ()

    clipped_result = execute(SaturationAudit(), MethodInputs.of(clipped), CTX)
    ceiling = 0.5 * float(np.max(np.asarray(clean.intensity, dtype=np.float64)))
    expected = float(np.sum(np.asarray(clipped.intensity, dtype=np.float64) >= ceiling))
    assert clipped_result.values["clipped_samples"] == pytest.approx(expected, abs=2.0)
    assert clipped_result.labels["verdict"] == str(MeasurementVerdict.REJECT)
    assert FailureCode.F05_SATURATION in clipped_result.all_failure_codes
    assert "not recoverable" in clipped_result.evidence[0].statement


def test_a_single_sample_at_the_maximum_is_not_saturation(source: SourceRecord) -> None:
    result = execute(
        SaturationAudit(), MethodInputs.of(_banded(source, fwhm_cm1=12.0, shape="lorentzian")), CTX
    )
    assert result.values["clipped_samples"] == 0.0
    assert "single sample" in result.evidence[0].statement


def test_saturation_audit_refuses_a_transformed_intensity_axis(source: SourceRecord) -> None:
    """Clipping at the ADC ceiling is not visible once the intensity has been transformed."""
    clean = _banded(source, fwhm_cm1=12.0, shape="gaussian")
    absorbance = Spectrum(
        spectrum_id="abs-0001",
        modality=clean.modality,
        origin=clean.origin,
        axis=clean.axis,
        intensity=clean.intensity,
        intensity_kind=IntensityKind.ABSORBANCE,
        sample_id=clean.sample_id,
        source=clean.source,
    )
    with pytest.raises(AssumptionViolationError, match="requires_intensity_kind"):
        execute(SaturationAudit(), MethodInputs.of(absorbance), CTX)


def test_saturation_audit_is_surfaceable_on_a_clean_run(source: SourceRecord) -> None:
    result = execute(
        SaturationAudit(), MethodInputs.of(_banded(source, fwhm_cm1=12.0, shape="gaussian")), CTX
    )
    assert result.applicability.verdict is Verdict.RUNNABLE
    assert result.surfaceable
    assert result.task is TaskKind.PREPROCESSING_QUALITY


# ------------------------------------------------------------------------------------------------
# BroadeningOrigin (HYPOTHESIZED)
# ------------------------------------------------------------------------------------------------
def test_the_e2_corpus_ratios_are_the_measured_numbers() -> None:
    """Both constants must stay traceable to experiments/E2-intrinsic-linewidth/comparison.json."""
    assert E2_CORPUS_RATIOS["real_rod"] == pytest.approx(21.494872296601443 / 23.141609037306605)
    assert E2_CORPUS_RATIOS["lorentzian_simulation"] == pytest.approx(73.0 / 23.0)


def test_the_registered_method_carries_no_calibration_and_says_so(source: SourceRecord) -> None:
    """A threshold with no measurement behind it is not a threshold."""
    result = execute(
        BroadeningOrigin(),
        MethodInputs.of(_banded(source, fwhm_cm1=20.0, shape="lorentzian", n_bands=6)),
        CTX,
    )
    assert result.labels["dominant_mechanism"] == "undecided"
    assert "no calibration" in result.evidence[0].statement
    assert "variable-temperature series" in result.evidence[0].statement
    assert result.values["width_normalised_correlation_length"] > 0.0
    assert result.values["median_band_spacing_cm1"] > 0.0


def test_a_calibration_must_state_where_it_was_measured() -> None:
    with pytest.raises(CalibrationError, match="measured_on"):
        RatioCalibration(light_tail_anchor=0.7, heavy_tail_anchor=1.0, measured_on="  ")


def test_a_calibration_whose_band_swallows_its_own_anchors_is_refused() -> None:
    """A classifier whose undecided band contains both anchors only looks like a classifier."""
    with pytest.raises(CalibrationError, match="decorative"):
        RatioCalibration(
            light_tail_anchor=0.9,
            heavy_tail_anchor=1.0,
            measured_on="tests",
            abstain_factor=1.3,
        )


def test_misordered_anchors_are_refused() -> None:
    with pytest.raises(CalibrationError, match="ordered light < heavy"):
        RatioCalibration(light_tail_anchor=3.0, heavy_tail_anchor=1.0, measured_on="tests")


# -- the control that weakened the conjecture ---------------------------------------------------
def _statistic(source: SourceRecord, **kwargs: object) -> tuple[float, float]:
    """The raw ratio and the spacing-to-width ratio, computed without the method's domain gate.

    Deliberately not via ``execute``: the control is a measurement of the *statistic*, including in
    the crowded regime the method itself now refuses, and refusing to measure there is how a
    confound stays invisible.
    """
    spectrum = _banded(source, **kwargs)
    _, median_fwhm, _ = band_statistics(spectrum)
    return (
        autocorrelation_length(spectrum) / median_fwhm,
        median_band_spacing(spectrum) / median_fwhm,
    )


def test_line_shape_moves_the_ratio_in_the_conjectured_direction(source: SourceRecord) -> None:
    """Control 1: identical widths and spacings, only the tails differ.

    The direction holds everywhere - Gaussian below Lorentzian - but the size of the effect is
    1.17x to 1.41x, and it *shrinks as the bands crowd*: 1.41 at 33 widths apart, 1.17 at 6.
    A single global threshold therefore cannot exist even in principle.
    """
    factors = {}
    for n_bands in (3, 6, 12):
        gaussian, spacing = _statistic(
            source, fwhm_cm1=20.0, shape="gaussian", n_bands=n_bands, spectrum_id="g"
        )
        lorentzian, _ = _statistic(
            source, fwhm_cm1=20.0, shape="lorentzian", n_bands=n_bands, spectrum_id="l"
        )
        assert gaussian < lorentzian
        factors[round(spacing)] = lorentzian / gaussian
    assert min(factors.values()) > 1.1 and max(factors.values()) < 1.5, factors
    widest, narrowest = max(factors), min(factors)
    assert factors[widest] > factors[narrowest], factors


def test_band_spacing_moves_the_ratio_as_much_as_the_line_shape_does(
    source: SourceRecord,
) -> None:
    """Control 2: identical width and shape, only the packing differs.

    Within the resolved regime the spacing effect is 1.34x for Lorentzian bands, against a shape
    effect of 1.36x at comparable spacing - the same size. Allowing the bands to crowd to 3.5
    widths apart takes the spacing effect to 1.7x. Neither is anywhere near the 3.4x gap E2
    measured between real and Lorentzian-broadened simulated corpora, so that gap cannot be
    attributed to tail shape on this evidence. Hence: no calibration by default, and
    RequiresBandSeparation declared.
    """
    sparse, sparse_spacing = _statistic(
        source, fwhm_cm1=20.0, shape="lorentzian", n_bands=3, spectrum_id="s"
    )
    resolved, resolved_spacing = _statistic(
        source, fwhm_cm1=20.0, shape="lorentzian", n_bands=12, spectrum_id="r"
    )
    crowded, crowded_spacing = _statistic(
        source, fwhm_cm1=20.0, shape="lorentzian", n_bands=24, spectrum_id="c"
    )
    assert sparse_spacing > 30.0 and resolved_spacing > 6.0 and crowded_spacing < 4.0
    within_regime = sparse / resolved
    full_range = sparse / crowded
    shape_effect = (
        _statistic(source, fwhm_cm1=20.0, shape="lorentzian", n_bands=6, spectrum_id="l")[0]
        / _statistic(source, fwhm_cm1=20.0, shape="gaussian", n_bands=6, spectrum_id="g")[0]
    )
    assert within_regime == pytest.approx(shape_effect, rel=0.25), (within_regime, shape_effect)
    assert full_range > 1.6
    assert full_range < 0.6 * (
        E2_CORPUS_RATIOS["lorentzian_simulation"] / E2_CORPUS_RATIOS["real_rod"]
    )


def test_neither_synthetic_shape_comes_near_the_e2_simulated_anchor(
    source: SourceRecord,
) -> None:
    """The E2 simulated corpus sat at 3.17; a pure Lorentzian band set does not get close."""
    lorentzian, _ = _statistic(
        source, fwhm_cm1=20.0, shape="lorentzian", n_bands=3, spectrum_id="l"
    )
    assert lorentzian < 0.5 * E2_CORPUS_RATIOS["lorentzian_simulation"]


# -- classification, exercised through a calibration measured by the control above ---------------
CONTROL_CALIBRATION = RatioCalibration(
    light_tail_anchor=0.738,
    heavy_tail_anchor=1.000,
    measured_on=(
        "tests/test_methods_builtin.py synthetic band sets, FWHM 20 cm-1, 6 bands, "
        "spacing 280 cm-1 - synthetic and spacing-specific, not a validated calibration"
    ),
    abstain_factor=1.05,
)


def test_a_calibrated_instance_classifies_both_shapes(source: SourceRecord) -> None:
    method = BroadeningOrigin(CONTROL_CALIBRATION)
    gaussian = execute(
        method,
        MethodInputs.of(_banded(source, fwhm_cm1=20.0, shape="gaussian", n_bands=6)),
        CTX,
    )
    lorentzian = execute(
        method,
        MethodInputs.of(_banded(source, fwhm_cm1=20.0, shape="lorentzian", n_bands=6)),
        CTX,
    )
    assert gaussian.labels["dominant_mechanism"] == "inhomogeneous_dominated"
    assert lorentzian.labels["dominant_mechanism"] == "lifetime_dominated"
    assert "confounded with band spacing" in str(lorentzian.evidence[0].conditions)


def test_a_calibrated_result_still_never_surfaces(source: SourceRecord) -> None:
    result = execute(
        BroadeningOrigin(CONTROL_CALIBRATION),
        MethodInputs.of(_banded(source, fwhm_cm1=20.0, shape="lorentzian", n_bands=6)),
        CTX,
    )
    assert result.maturity is Maturity.HYPOTHESIZED
    assert not result.surfaceable
    assert result.evidence[0].claim_type is ClaimType.MECHANISTIC_HYPOTHESIS
    assert "HYPOTHESIZED" in result.notes


def test_it_abstains_when_there_is_no_band_to_measure(source: SourceRecord) -> None:
    axis = AxisSpec.from_values(
        AxisKind.RAMAN_SHIFT_CM1, np.linspace(200.0, 1800.0, 3201), resolution_fwhm=4.0
    )
    flat = Spectrum(
        spectrum_id="flat",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=axis,
        intensity=np.full(3201, 100.0),
        intensity_kind=IntensityKind.COUNTS,
        sample_id="flat",
        source=source,
    )
    result = execute(BroadeningOrigin(CONTROL_CALIBRATION), MethodInputs.of(flat), CTX)
    assert result.labels["dominant_mechanism"] == "undecided"
    assert "no band with a measurable width" in result.evidence[0].statement


def test_overlapping_bands_are_refused_not_scored(source: SourceRecord) -> None:
    """The confound, enforced: 24 bands 61 cm-1 apart at 20 cm-1 FWHM overlap."""
    crowded = _banded(source, fwhm_cm1=20.0, shape="lorentzian", n_bands=24)
    with pytest.raises(AssumptionViolationError, match="requires_band_separation"):
        execute(BroadeningOrigin(CONTROL_CALIBRATION), MethodInputs.of(crowded), CTX)


def test_it_refuses_a_non_uniform_axis_because_the_statistic_changes_meaning(
    source: SourceRecord,
) -> None:
    wavelength_nm = np.linspace(800.0, 900.0, 3201)
    axis = AxisSpec.from_values(
        AxisKind.ABSOLUTE_WAVENUMBER_CM1, np.sort(1e7 / wavelength_nm), resolution_fwhm=4.0
    )
    reference = _banded(source, fwhm_cm1=20.0, shape="lorentzian", n_bands=6)
    dispersed = Spectrum(
        spectrum_id="dispersed",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=axis,
        intensity=reference.intensity,
        intensity_kind=IntensityKind.COUNTS,
        sample_id="dispersed",
        source=source,
    )
    with pytest.raises(AssumptionViolationError, match="requires_uniform_axis"):
        execute(BroadeningOrigin(), MethodInputs.of(dispersed), CTX)


def test_it_refuses_when_the_instrument_response_dominates(source: SourceRecord) -> None:
    coarse = _banded(source, fwhm_cm1=20.0, shape="lorentzian", n_bands=6, resolution_fwhm=25.0)
    with pytest.raises(AssumptionViolationError, match="valid_resolution_below"):
        execute(BroadeningOrigin(), MethodInputs.of(coarse), CTX)


def test_an_undeclared_resolution_degrades_instead_of_refusing(source: SourceRecord) -> None:
    silent = _banded(source, fwhm_cm1=20.0, shape="lorentzian", n_bands=6, resolution_fwhm=None)
    result = execute(BroadeningOrigin(), MethodInputs.of(silent), CTX)
    assert result.applicability.verdict is Verdict.RUNNABLE_DEGRADED
    assert result.applicability.evidence_needed
