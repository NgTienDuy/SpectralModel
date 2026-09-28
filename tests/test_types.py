"""The repo's first invariants: normalisation, monotonicity, append-only provenance, licences."""

from __future__ import annotations

import numpy as np
import pytest

from usis.core.errors import (
    BeliefStateError,
    ProvenanceError,
    SpectrumValidationError,
)
from usis.core.types import (
    AxisKind,
    AxisSpec,
    BeliefState,
    Candidate,
    ClaimType,
    ComponentHypothesis,
    Evidence,
    GroundTruth,
    InstrumentSpec,
    IntensityKind,
    MixtureHypothesis,
    Modality,
    Origin,
    ProcessingStep,
    SourceRecord,
    Spectrum,
)

STEP = ProcessingStep(name="despike", params={"threshold": 5.0}, version="1", timestamp="t0")


# --- axis -------------------------------------------------------------------------------------
def test_axis_must_be_strictly_monotonic():
    with pytest.raises(SpectrumValidationError, match="monotonic"):
        AxisSpec(AxisKind.RAMAN_SHIFT_CM1, np.array([1.0, 2.0, 2.0, 3.0]), uniform=False)


def test_descending_axis_is_allowed():
    axis = AxisSpec.from_values(AxisKind.ABSOLUTE_WAVENUMBER_CM1, np.array([400.0, 300.0, 200.0]))
    assert axis.ascending is False
    assert axis.span == (200.0, 400.0)


def test_axis_rejects_non_finite_values():
    with pytest.raises(SpectrumValidationError, match="finite"):
        AxisSpec.from_values(AxisKind.WAVELENGTH_NM, np.array([1.0, np.nan, 3.0]))


def test_uniformity_claim_is_checked():
    with pytest.raises(SpectrumValidationError, match="uniform"):
        AxisSpec(AxisKind.WAVELENGTH_NM, np.array([1.0, 2.0, 4.0]), uniform=True)
    assert AxisSpec.from_values(AxisKind.WAVELENGTH_NM, np.array([1.0, 2.0, 4.0])).uniform is False
    assert AxisSpec.from_values(AxisKind.WAVELENGTH_NM, np.array([1.0, 2.0, 3.0])).uniform is True


def test_axis_values_are_read_only():
    axis = AxisSpec.from_values(AxisKind.WAVELENGTH_NM, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="read-only"):
        axis.values[0] = 99.0


# --- spectrum ---------------------------------------------------------------------------------
def test_spectrum_intensity_must_match_the_axis(raman_axis, source, instrument):
    with pytest.raises(SpectrumValidationError, match="points"):
        Spectrum(
            spectrum_id="s",
            modality=Modality.RAMAN,
            origin=Origin.EXPERIMENTAL,
            axis=raman_axis,
            intensity=np.zeros(raman_axis.n_points - 1),
            intensity_kind=IntensityKind.COUNTS,
            sample_id="a",
            source=source,
            instrument=instrument,
        )


def test_experimental_spectrum_must_name_its_instrument(raman_axis, source):
    with pytest.raises(SpectrumValidationError, match="instrument"):
        Spectrum(
            spectrum_id="s",
            modality=Modality.RAMAN,
            origin=Origin.EXPERIMENTAL,
            axis=raman_axis,
            intensity=np.zeros(raman_axis.n_points),
            intensity_kind=IntensityKind.COUNTS,
            sample_id="a",
            source=source,
        )


def test_augmented_spectrum_may_omit_the_instrument_too(raman_axis, source):
    # An augmentation of a simulated spectrum has no instrument either; requiring one would force
    # a fabricated InstrumentSpec into the provenance.
    spectrum = Spectrum(
        spectrum_id="s",
        modality=Modality.RAMAN,
        origin=Origin.AUGMENTED,
        axis=raman_axis,
        intensity=np.zeros(raman_axis.n_points),
        intensity_kind=IntensityKind.ARBITRARY,
        sample_id="a",
        source=source,
    )
    assert spectrum.instrument is None


def test_simulated_spectrum_may_omit_the_instrument(raman_axis, source):
    spectrum = Spectrum(
        spectrum_id="s",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=raman_axis,
        intensity=np.zeros(raman_axis.n_points),
        intensity_kind=IntensityKind.ARBITRARY,
        sample_id="a",
        source=source,
    )
    assert spectrum.instrument is None
    assert spectrum.excitation_nm is None


def test_sigma_must_be_non_negative_and_shaped_like_the_intensity(spectrum):
    with pytest.raises(SpectrumValidationError, match="sigma"):
        Spectrum(
            spectrum_id="s",
            modality=spectrum.modality,
            origin=spectrum.origin,
            axis=spectrum.axis,
            intensity=spectrum.intensity,
            intensity_kind=spectrum.intensity_kind,
            sample_id=spectrum.sample_id,
            source=spectrum.source,
            instrument=spectrum.instrument,
            sigma=-np.ones(spectrum.axis.n_points),
        )


def test_processing_chain_is_append_only(spectrum):
    assert spectrum.is_processed is False
    once = spectrum.with_processing(STEP)
    twice = once.with_processing(
        ProcessingStep(name="baseline", params={"lam": 1e5}, version="1", timestamp="t1")
    )
    assert [s.name for s in twice.processing] == ["despike", "baseline"]
    # The original is untouched: provenance is immutable, not merely conventionally respected.
    assert spectrum.processing == ()
    assert once.processing == (STEP,)


def test_with_processing_refuses_to_rewrite_the_chain(spectrum):
    with pytest.raises(ProvenanceError, match=r"append-only|processing chain"):
        spectrum.with_processing(STEP, processing=())


def test_processing_step_requires_a_version():
    with pytest.raises(ProvenanceError, match="version"):
        ProcessingStep(name="x", params={}, version="", timestamp="t")


def test_every_spectrum_carries_a_licence(spectrum):
    assert spectrum.source.licence
    with pytest.raises(ProvenanceError, match="licence"):
        SourceRecord(dataset_id="d", licence="", retrieved_at="t")


def test_source_record_checksum_length_is_validated():
    with pytest.raises(ProvenanceError, match="sha256"):
        SourceRecord(dataset_id="d", licence="CC0-1.0", retrieved_at="t", sha256="deadbeef")


def test_instrument_requires_a_stable_id():
    with pytest.raises(SpectrumValidationError, match="instrument_id"):
        InstrumentSpec(instrument_id="")


# --- beliefs ----------------------------------------------------------------------------------
def _candidate(posterior: float, name: str = "c") -> Candidate:
    return Candidate(posterior=posterior, name=name)


def test_posterior_plus_p_none_must_equal_one():
    BeliefState(candidates=(_candidate(0.6), _candidate(0.3, "d")), p_none=0.1)
    with pytest.raises(BeliefStateError, match="p_none"):
        BeliefState(candidates=(_candidate(0.6), _candidate(0.3, "d")), p_none=0.2)


def test_an_empty_candidate_list_needs_all_the_mass_in_p_none():
    belief = BeliefState(candidates=(), p_none=1.0)
    assert belief.top is None
    assert belief.abstains_at(0.5) is True


def test_open_set_mass_raises_the_entropy():
    confident = BeliefState(candidates=(_candidate(0.99),), p_none=0.01)
    unsure = BeliefState(candidates=(_candidate(0.5),), p_none=0.5)
    assert unsure.entropy > confident.entropy


def test_a_candidate_needs_an_identifier():
    with pytest.raises(BeliefStateError, match="identifier"):
        Candidate(posterior=0.5)


def test_closure_is_enforced_on_mixtures():
    a = ComponentHypothesis(candidate=_candidate(0.5, "a"), fraction=0.7)
    b = ComponentHypothesis(candidate=_candidate(0.5, "b"), fraction=0.3)
    MixtureHypothesis(components=(a, b))
    bad = ComponentHypothesis(candidate=_candidate(0.5, "b"), fraction=0.5)
    with pytest.raises(BeliefStateError, match="closure"):
        MixtureHypothesis(components=(a, bad))


def test_negative_fractions_are_rejected():
    with pytest.raises(BeliefStateError, match="non-negative"):
        ComponentHypothesis(candidate=_candidate(0.5), fraction=-0.1)


def test_ground_truth_mixture_fractions_obey_closure():
    GroundTruth(components={"water": 0.5, "ethanol": 0.5})
    with pytest.raises(SpectrumValidationError, match="closure"):
        GroundTruth(components={"water": 0.5, "ethanol": 0.4})


def test_a_causal_claim_requires_interventional_evidence():
    Evidence(kind="band", statement="peak at 1650", claim_type=ClaimType.CORRELATION)
    with pytest.raises(SpectrumValidationError, match="interventional"):
        Evidence(kind="effect", statement="causes X", claim_type=ClaimType.CAUSAL_CLAIM)
