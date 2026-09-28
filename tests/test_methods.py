"""The method layer: what it refuses, and what it will not let a method say about itself."""

from __future__ import annotations

import numpy as np
import pytest

from usis.core.errors import AssumptionViolationError
from usis.core.provenance import RunContext
from usis.core.registry import RegistryError
from usis.core.types import (
    AxisKind,
    AxisSpec,
    ClaimType,
    GroundTruth,
    IntensityKind,
    Maturity,
    Modality,
    Origin,
    ProcessingStep,
    SourceRecord,
    Spectrum,
    TaskKind,
)
from usis.methods.applicability import Verdict, assess
from usis.methods.assumptions import (
    AssumesLinearMixing,
    AssumesNoResonanceEnhancement,
    AssumptionStatus,
    MaxResolutionFwhm,
    MaxSaturatedFraction,
    MinSnr,
    RequiresAxisCoverage,
    RequiresBaselineCorrected,
    RequiresIntensityKind,
    RequiresLibraryCoverage,
    RequiresMetadata,
    RequiresModality,
    RequiresUniformAxis,
    estimate_snr,
)
from usis.methods.base import (
    Citation,
    CostModel,
    Method,
    MethodBase,
    MethodError,
    MethodResult,
    Requirements,
    execute,
    surfaceable_results,
)
from usis.methods.inputs import LibraryHandle, MethodInputError, MethodInputs
from usis.methods.registry import (
    METHODS,
    MethodRegistrationError,
    load_methods_from,
    register_method,
    select_methods,
)

CTX = RunContext(
    run_id="test", started_at="2026-09-03T00:00:00+00:00", seed=0, git_sha="0" * 40, git_dirty="no"
)


def _spectrum(
    source: SourceRecord,
    *,
    spectrum_id: str = "t-0001",
    noise: float = 1.0,
    peak: float = 500.0,
    axis: AxisSpec | None = None,
    modality: Modality = Modality.RAMAN,
    intensity_kind: IntensityKind = IntensityKind.COUNTS,
    processing: tuple[ProcessingStep, ...] = (),
    label: GroundTruth | None = None,
    origin: Origin = Origin.SIMULATED,
) -> Spectrum:
    """A synthetic spectrum with one Gaussian band, so SNR is controllable."""
    axis = axis or AxisSpec.from_values(
        AxisKind.RAMAN_SHIFT_CM1, np.linspace(200.0, 1800.0, 401), resolution_fwhm=4.0
    )
    values = np.asarray(axis.values, dtype=np.float64)
    rng = np.random.default_rng(7)
    band = peak * np.exp(-0.5 * ((values - 1000.0) / 8.0) ** 2)
    intensity = 10.0 + band + rng.normal(0.0, noise, values.size)
    return Spectrum(
        spectrum_id=spectrum_id,
        modality=modality,
        origin=origin,
        axis=axis,
        intensity=intensity,
        intensity_kind=intensity_kind,
        sample_id=f"sample-{spectrum_id}",
        source=source,
        processing=processing,
        label=label,
    )


class _Trivial(MethodBase):
    """A method that reports the number of samples it was given."""

    method_id = "test.trivial"
    version = "1.0.0"
    task = TaskKind.PREPROCESSING_QUALITY
    maturity = Maturity.EMERGING
    provenance = Citation(title="test", internal_ref="tests/test_methods.py")
    cost = CostModel()

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        return self._result(values={"n_points": float(inputs.primary.intensity.size)})


class _NeedsSnr(_Trivial):
    method_id = "test.needs_snr"
    assumptions = (MinSnr(50.0),)


# ------------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------------
def test_empty_inputs_are_rejected() -> None:
    with pytest.raises(MethodInputError, match="no subject"):
        MethodInputs(spectra=())


def test_primary_refuses_to_pick_the_first_of_many(source: SourceRecord) -> None:
    inputs = MethodInputs.of(_spectrum(source, spectrum_id="a"), _spectrum(source, spectrum_id="b"))
    with pytest.raises(MethodInputError, match="2 spectra"):
        _ = inputs.primary


def test_library_membership_unknown_is_not_empty() -> None:
    unknown = LibraryHandle(library_id="lib", n_entries=10, modality=Modality.RAMAN)
    known = LibraryHandle(
        library_id="lib", n_entries=1, modality=Modality.RAMAN, inchikeys=frozenset({"AAA"})
    )
    assert unknown.covers("AAA") is None
    assert known.covers("AAA") is True
    assert known.covers("BBB") is False


# ------------------------------------------------------------------------------------------------
# Assumptions: the four-valued status
# ------------------------------------------------------------------------------------------------
def test_snr_is_estimated_peak_to_floor_not_median_relative(source: SourceRecord) -> None:
    """The E2 lesson, as a test: a near-zero floor must not inflate the estimate."""
    high = _spectrum(source, noise=1.0, peak=500.0)
    low = _spectrum(source, noise=100.0, peak=500.0)
    assert estimate_snr(high) > 100.0
    assert estimate_snr(low) < 20.0


def test_min_snr_violation_names_the_numbers(source: SourceRecord) -> None:
    check = MinSnr(50.0).check(MethodInputs.of(_spectrum(source, noise=100.0)))
    assert check.status is AssumptionStatus.VIOLATED
    assert check.assumption == "valid_snr_above=50"
    assert "estimated SNR" in check.detail and "required > 50" in check.detail


def test_missing_resolution_is_unknown_not_satisfied(source: SourceRecord) -> None:
    """A hard assumption with no metadata to decide it must not read as a pass."""
    axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, np.linspace(200.0, 1800.0, 401))
    check = MaxResolutionFwhm(10.0).check(MethodInputs.of(_spectrum(source, axis=axis)))
    assert check.status is AssumptionStatus.UNKNOWN
    assert "lamp calibration" in check.evidence_needed


def test_declared_resolution_decides_both_ways(source: SourceRecord) -> None:
    fine = AxisSpec.from_values(
        AxisKind.RAMAN_SHIFT_CM1, np.linspace(200.0, 1800.0, 401), resolution_fwhm=4.0
    )
    coarse = AxisSpec.from_values(
        AxisKind.RAMAN_SHIFT_CM1, np.linspace(200.0, 1800.0, 401), resolution_fwhm=16.0
    )
    assert (
        MaxResolutionFwhm(10.0).check(MethodInputs.of(_spectrum(source, axis=fine))).status
        is AssumptionStatus.SATISFIED
    )
    assert (
        MaxResolutionFwhm(10.0).check(MethodInputs.of(_spectrum(source, axis=coarse))).status
        is AssumptionStatus.VIOLATED
    )


def test_non_uniform_axis_is_measured_not_read_from_the_flag(source: SourceRecord) -> None:
    """A grating disperses linearly in wavelength, so a cm-1 axis is not uniform."""
    wavelength_nm = np.linspace(800.0, 900.0, 401)
    shifted = 1e7 / wavelength_nm
    axis = AxisSpec.from_values(AxisKind.ABSOLUTE_WAVENUMBER_CM1, np.sort(shifted))
    check = RequiresUniformAxis(0.01).check(MethodInputs.of(_spectrum(source, axis=axis)))
    assert check.status is AssumptionStatus.VIOLATED
    assert "resample" in check.detail


def test_axis_coverage_compares_the_recorded_span(source: SourceRecord) -> None:
    inputs = MethodInputs.of(_spectrum(source))
    assert RequiresAxisCoverage(400.0, 1600.0).check(inputs).status is AssumptionStatus.SATISFIED
    outside = RequiresAxisCoverage(400.0, 3200.0).check(inputs)
    assert outside.status is AssumptionStatus.VIOLATED
    assert "3200" in outside.detail


def test_modality_and_intensity_kind_are_always_decidable(source: SourceRecord) -> None:
    inputs = MethodInputs.of(_spectrum(source, intensity_kind=IntensityKind.TRANSMITTANCE))
    assert (
        RequiresModality(frozenset({Modality.RAMAN})).check(inputs).status
        is AssumptionStatus.SATISFIED
    )
    assert (
        RequiresIntensityKind(frozenset({IntensityKind.ABSORBANCE})).check(inputs).status
        is AssumptionStatus.VIOLATED
    )


def test_requires_metadata_walks_dotted_paths(source: SourceRecord, spectrum: Spectrum) -> None:
    present = RequiresMetadata(("instrument.excitation_nm",)).check(MethodInputs.of(spectrum))
    assert present.status is AssumptionStatus.SATISFIED
    absent = RequiresMetadata(("instrument.excitation_nm",)).check(
        MethodInputs.of(_spectrum(source))
    )
    assert absent.status is AssumptionStatus.UNKNOWN
    assert "instrument.excitation_nm" in absent.detail


def test_baseline_state_has_three_outcomes(source: SourceRecord) -> None:
    """Satisfied, violated, and - the one that matters - unknown for a silent chain."""
    silent = RequiresBaselineCorrected().check(MethodInputs.of(_spectrum(source)))
    assert silent.status is AssumptionStatus.UNKNOWN
    assert "silence, not evidence" in silent.evidence_needed

    corrected = ProcessingStep(
        name="baseline_airpls", params={"lam": 1e5}, version="1.0", timestamp="unknown"
    )
    assert (
        RequiresBaselineCorrected()
        .check(MethodInputs.of(_spectrum(source, processing=(corrected,))))
        .status
        is AssumptionStatus.SATISFIED
    )

    declared = ProcessingStep(
        name="source_declared_preprocessing",
        params={"background_subtraction": "no_baseline_correction"},
        version="source-declared",
        timestamp="unknown",
    )
    violated = RequiresBaselineCorrected().check(
        MethodInputs.of(_spectrum(source, processing=(declared,)))
    )
    assert violated.status is AssumptionStatus.VIOLATED


def test_sers_violates_no_resonance_and_plain_raman_is_unknown(source: SourceRecord) -> None:
    sers = AssumesNoResonanceEnhancement().check(
        MethodInputs.of(_spectrum(source, modality=Modality.RAMAN_SERS))
    )
    assert sers.status is AssumptionStatus.VIOLATED
    plain = AssumesNoResonanceEnhancement().check(MethodInputs.of(_spectrum(source)))
    assert plain.status is AssumptionStatus.UNKNOWN
    assert "excitation-wavelength series" in plain.evidence_needed


def test_linear_mixing_declines_and_names_the_experiment(source: SourceRecord) -> None:
    check = AssumesLinearMixing().check(MethodInputs.of(_spectrum(source)))
    assert check.status is AssumptionStatus.NOT_CHECKABLE
    assert "concentration series" in check.evidence_needed


def test_library_coverage_distinguishes_absent_unenumerated_and_missing(
    source: SourceRecord,
) -> None:
    no_library = RequiresLibraryCoverage().check(MethodInputs.of(_spectrum(source)))
    assert no_library.status is AssumptionStatus.VIOLATED

    unenumerated = RequiresLibraryCoverage().check(
        MethodInputs.of(
            _spectrum(source),
            library=LibraryHandle(library_id="lib", n_entries=99, modality=Modality.RAMAN),
        )
    )
    assert unenumerated.status is AssumptionStatus.UNKNOWN

    labelled = _spectrum(source, label=GroundTruth(inchikey="AAAAAAAAAAAAAA-AAAAAAAAAA-A"))
    missing = RequiresLibraryCoverage().check(
        MethodInputs.of(
            labelled,
            library=LibraryHandle(
                library_id="lib",
                n_entries=1,
                modality=Modality.RAMAN,
                inchikeys=frozenset({"BBBBBBBBBBBBBB-BBBBBBBBBB-B"}),
            ),
        )
    )
    assert missing.status is AssumptionStatus.VIOLATED
    assert "outside the candidate set" in missing.detail


def test_saturation_check_finds_a_plateau_and_ignores_a_lone_apex(source: SourceRecord) -> None:
    clean = _spectrum(source)
    assert MaxSaturatedFraction().check(MethodInputs.of(clean)).status is AssumptionStatus.SATISFIED
    clipped_values = np.asarray(clean.intensity, dtype=np.float64).copy()
    clipped_values[clipped_values > 300.0] = 300.0
    clipped = Spectrum(
        spectrum_id="clipped",
        modality=clean.modality,
        origin=clean.origin,
        axis=clean.axis,
        intensity=clipped_values,
        intensity_kind=clean.intensity_kind,
        sample_id=clean.sample_id,
        source=clean.source,
    )
    assert (
        MaxSaturatedFraction().check(MethodInputs.of(clipped)).status is AssumptionStatus.VIOLATED
    )


def test_a_batch_reports_the_worst_case_and_counts_the_offenders(source: SourceRecord) -> None:
    good = _spectrum(source, spectrum_id="good", noise=1.0)
    bad = _spectrum(source, spectrum_id="bad", noise=200.0)
    check = MinSnr(50.0).check(MethodInputs.of(good, bad))
    assert check.status is AssumptionStatus.VIOLATED
    assert "1/2 spectra" in check.detail and "bad" in check.detail


# ------------------------------------------------------------------------------------------------
# Applicability policy (PROJECT_DECISIONS.md D13)
# ------------------------------------------------------------------------------------------------
def test_hard_violation_refuses_and_soft_violation_degrades(source: SourceRecord) -> None:
    inputs = MethodInputs.of(_spectrum(source, noise=200.0))
    assert assess([MinSnr(50.0)], inputs).verdict is Verdict.REFUSED
    soft = assess([MinSnr(50.0, hard=False)], inputs)
    assert soft.verdict is Verdict.RUNNABLE_DEGRADED
    assert soft.failure_codes


def test_undecidable_hard_assumption_degrades_rather_than_refusing(source: SourceRecord) -> None:
    axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, np.linspace(200.0, 1800.0, 401))
    applicability = assess([MaxResolutionFwhm(10.0)], MethodInputs.of(_spectrum(source, axis=axis)))
    assert applicability.verdict is Verdict.RUNNABLE_DEGRADED
    assert applicability.undecided and not applicability.violations
    assert applicability.evidence_needed


def test_undecided_checks_do_not_enter_the_failure_histogram(source: SourceRecord) -> None:
    """An unknown is not evidence the failure happened; counting it would inflate every run."""
    applicability = assess([AssumesLinearMixing()], MethodInputs.of(_spectrum(source)))
    assert applicability.undecided
    assert applicability.failure_codes == ()


def test_declaring_no_assumptions_is_recorded_as_a_claim(source: SourceRecord) -> None:
    applicability = assess([], MethodInputs.of(_spectrum(source)))
    assert applicability.verdict is Verdict.RUNNABLE
    assert "itself a claim" in applicability.reason


def test_satisfied_checks_are_retained(source: SourceRecord) -> None:
    applicability = assess([MinSnr(5.0)], MethodInputs.of(_spectrum(source)))
    assert len(applicability.checks) == 1
    assert applicability.verdict is Verdict.RUNNABLE


# ------------------------------------------------------------------------------------------------
# execute()
# ------------------------------------------------------------------------------------------------
def test_execute_refuses_a_low_snr_spectrum_and_names_the_violation(source: SourceRecord) -> None:
    """M2 acceptance 2.1, verbatim: refused, with the violation named."""
    with pytest.raises(AssumptionViolationError) as excinfo:
        execute(_NeedsSnr(), MethodInputs.of(_spectrum(source, noise=200.0)), CTX)
    message = str(excinfo.value)
    assert "test.needs_snr@1.0.0" in message
    assert "valid_snr_above=50" in message
    assert "estimated SNR" in message


def test_execute_attaches_the_audit_it_computed_itself(source: SourceRecord) -> None:
    result = execute(_NeedsSnr(), MethodInputs.of(_spectrum(source)), CTX)
    assert result.applicability.checks[0].assumption == "valid_snr_above=50"
    assert "placeholder" not in result.applicability.reason
    assert result.cost_incurred["wall_clock_s"] >= 0.0
    assert result.cost_incurred["spectra"] == 1.0


def test_execute_rejects_a_result_attributed_to_another_method(source: SourceRecord) -> None:
    class _Liar(_Trivial):
        method_id = "test.liar"

        def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
            result = super().compute(inputs, ctx)
            object.__setattr__(result, "method_id", "test.someone_else")
            return result

    with pytest.raises(MethodError, match="attributed to"):
        execute(_Liar(), MethodInputs.of(_spectrum(source)), CTX)


def test_no_method_may_emit_a_causal_claim(source: SourceRecord) -> None:
    class _Causal(_Trivial):
        method_id = "test.causal"

        def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
            return self._result(claim_type=ClaimType.CAUSAL_CLAIM)

    with pytest.raises(MethodError, match="F18"):
        execute(_Causal(), MethodInputs.of(_spectrum(source)), CTX)


def test_requirements_expand_into_assumptions_so_there_is_one_checking_path() -> None:
    requirements = Requirements(
        modalities=frozenset({Modality.RAMAN}),
        intensity_kinds=frozenset({IntensityKind.COUNTS}),
        metadata_fields=("instrument.excitation_nm",),
        reference_library=True,
    )
    described = [a.describe() for a in requirements.as_assumptions()]
    assert len(described) == 4
    assert any("requires_library_coverage" in d for d in described)


def test_method_base_satisfies_the_protocol() -> None:
    assert isinstance(_Trivial(), Method)


# ------------------------------------------------------------------------------------------------
# Surfacing and maturity
# ------------------------------------------------------------------------------------------------
def test_a_hypothesized_result_is_never_surfaceable(source: SourceRecord) -> None:
    class _Shadow(_Trivial):
        method_id = "test.shadow"
        maturity = Maturity.HYPOTHESIZED

    established = execute(_Trivial(), MethodInputs.of(_spectrum(source)), CTX)
    shadow = execute(_Shadow(), MethodInputs.of(_spectrum(source)), CTX)
    assert established.surfaceable and not shadow.surfaceable
    assert surfaceable_results([established, shadow]) == (established,)


def test_a_degraded_run_is_not_surfaceable_even_at_established_maturity(
    source: SourceRecord,
) -> None:
    class _Degrades(_Trivial):
        method_id = "test.degrades"
        maturity = Maturity.ESTABLISHED
        assumptions = (AssumesLinearMixing(),)

    result = execute(_Degrades(), MethodInputs.of(_spectrum(source)), CTX)
    assert result.applicability.verdict is Verdict.RUNNABLE_DEGRADED
    assert not result.surfaceable


# ------------------------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------------------------
def test_established_needs_provenance_and_a_reproduction_reference() -> None:
    class _Bare(_Trivial):
        method_id = "test.registry.bare_established"
        maturity = Maturity.ESTABLISHED
        provenance = None

    with pytest.raises(MethodRegistrationError, match="no attributed provenance"):
        register_method(_Bare())

    class _Unreproduced(_Trivial):
        method_id = "test.registry.unreproduced"
        maturity = Maturity.ESTABLISHED
        provenance = Citation(title="paper", doi="10.0000/fake-for-test")

    with pytest.raises(MethodRegistrationError, match="reproduction"):
        register_method(_Unreproduced())


def test_a_conjecture_must_point_at_where_it_is_written_down() -> None:
    class _Folklore(_Trivial):
        method_id = "test.registry.folklore"
        maturity = Maturity.HYPOTHESIZED
        provenance = Citation(title="a good idea")

    with pytest.raises(MethodRegistrationError, match="nowhere to check it"):
        register_method(_Folklore())


def test_refuted_and_deprecated_entries_must_carry_their_evidence() -> None:
    class _Refuted(_Trivial):
        method_id = "test.registry.refuted"
        maturity = Maturity.REFUTED

    with pytest.raises(MethodRegistrationError, match="refuted_by"):
        register_method(_Refuted())

    class _Deprecated(_Trivial):
        method_id = "test.registry.deprecated"
        maturity = Maturity.DEPRECATED

    with pytest.raises(MethodRegistrationError, match="superseded_by"):
        register_method(_Deprecated())


def test_a_refuted_entry_is_retained_and_selectable() -> None:
    class _Dead(_Trivial):
        method_id = "test.registry.dead_idea"
        maturity = Maturity.REFUTED

    register_method(_Dead(), refuted_by="experiments/E1-sim2real-comparability/result_card.md")
    entry = METHODS.get("test.registry.dead_idea@1.0.0")
    assert entry.refuted_by.endswith("result_card.md")
    assert not entry.surfaceable
    assert entry in select_methods(maturities=[Maturity.REFUTED])
    assert entry not in select_methods(maturities=[Maturity.ESTABLISHED, Maturity.EMERGING])


def test_a_version_bump_is_a_new_entry_not_an_overwrite() -> None:
    class _V1(_Trivial):
        method_id = "test.registry.versioned"
        version = "1.0.0"

    class _V2(_Trivial):
        method_id = "test.registry.versioned"
        version = "2.0.0"

    register_method(_V1())
    register_method(_V2())
    assert "test.registry.versioned@1.0.0" in METHODS
    assert "test.registry.versioned@2.0.0" in METHODS
    with pytest.raises(RegistryError, match="already registered"):
        register_method(_V1())


def test_a_registry_row_is_publishable_on_its_own() -> None:
    class _Documented(_Trivial):
        method_id = "test.registry.documented"

    register_method(_Documented(), tags=("test",), notes="a note")
    row = METHODS.get("test.registry.documented@1.0.0").to_dict()
    assert set(row) >= {"key", "task", "maturity", "provenance", "assumptions", "cost", "evidence"}


def test_adding_a_method_touches_only_its_own_file(tmp_path, monkeypatch) -> None:
    """M2 acceptance 2.2: discovery, not a list, is what makes this true."""
    package = tmp_path / "extra_methods"
    package.mkdir()
    (package / "__init__.py").write_text('"""A method package outside the repository."""\n')
    (package / "an_added_method.py").write_text(
        '"""One added method, in one added file."""\n'
        "from usis.core.types import Maturity, TaskKind\n"
        "from usis.methods.base import Citation, MethodBase, MethodResult\n"
        "from usis.methods.registry import register_method\n"
        "\n"
        "class Added(MethodBase):\n"
        '    """Added without touching any existing file."""\n'
        '    method_id = "test.discovered.added"\n'
        '    version = "1.0.0"\n'
        "    task = TaskKind.PREPROCESSING_QUALITY\n"
        "    maturity = Maturity.HYPOTHESIZED\n"
        '    provenance = Citation(title="added", internal_ref="tests")\n'
        "\n"
        "    def compute(self, inputs, ctx) -> MethodResult:\n"
        '        return self._result(values={"answer": 42.0})\n'
        "\n"
        "register_method(Added())\n"
    )
    (package / "_private.py").write_text('"""Skipped: leading underscore."""\nraise SystemExit\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    loaded = load_methods_from("extra_methods")
    assert loaded == ("an_added_method",)
    assert "test.discovered.added@1.0.0" in METHODS
    assert load_methods_from("extra_methods") == ("an_added_method",)


def test_loading_a_module_that_is_not_a_package_is_refused() -> None:
    with pytest.raises(MethodRegistrationError, match="not a package"):
        load_methods_from("usis.methods.base")
