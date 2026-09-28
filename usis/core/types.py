"""The data contracts every other module in USIS speaks.

These types encode the brief's non-negotiable invariants directly:

* axis and intensity *semantics* are part of the type, not a convention in a docstring;
* a spectrum carries its provenance and its ordered, append-only processing chain;
* an identity answer is a distribution plus an explicit open-set residual ``p_none``, never a
  bare molecule name.

Frozen dataclasses are used rather than pydantic models: the payloads are numpy arrays, the
validation we need is structural rather than coercive, and this keeps ``mypy --strict`` clean
without a runtime schema dependency. See ``PROJECT_DECISIONS.md`` D2.

Types holding numpy arrays are declared ``eq=False``: element-wise ``__eq__`` on an array field
would return an array, and a dataclass-generated ``__eq__`` would then raise on truth-testing it.
Identity comparison is the honest default; compare spectra with an explicit metric instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from usis.core.errors import (
    BeliefStateError,
    ProvenanceError,
    SpectrumValidationError,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
else:  # pragma: no cover - runtime alias only
    FloatArray = np.ndarray

__all__ = [
    "AcquisitionParams",
    "Action",
    "ActionKind",
    "AxisKind",
    "AxisSpec",
    "BeliefState",
    "Candidate",
    "ClaimType",
    "ComponentHypothesis",
    "Evidence",
    "FailureCode",
    "GroundTruth",
    "InstrumentSpec",
    "IntensityKind",
    "Maturity",
    "MeasurementVerdict",
    "MixtureHypothesis",
    "Modality",
    "NonIdentifiability",
    "Origin",
    "ProcessingStep",
    "RamanBranch",
    "SourceRecord",
    "Spectrum",
    "SpectrumId",
    "TaskKind",
]

SpectrumId = str

# Tolerance for the ``sum(posterior) + p_none == 1`` invariant. Chosen to absorb float32
# accumulation over a few thousand candidates without admitting a genuine normalisation bug.
POSTERIOR_SUM_TOL = 1e-6

# Relative tolerance for deciding whether an axis grid is uniform.
UNIFORM_AXIS_RTOL = 1e-6


# --------------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------------
class Modality(StrEnum):
    """Measurement modality.

    SERS is a distinct member rather than a flag on ``RAMAN``: its enhancement factors vary by
    orders of magnitude and are not transferable, so pooling it with spontaneous Raman is a
    domain error, not a convenience.

    ``NMR_1H`` and ``MS`` are placeholders for later phases and carry a design constraint. Mass
    spectrometry is **not** "another spectroscopy": it has a different physical observable and its
    own mature structure-elucidation literature, so it belongs behind its own adapter and must not
    be flattened into the vibrational pipeline. Read ``docs/knowledge/KEYWORD_MAP.md`` §C7 before
    writing a loader or a method for either. See ``PROJECT_DECISIONS.md`` D6.
    """

    RAMAN = "raman"
    RAMAN_SERS = "raman_sers"
    IR_ATR = "ir_atr"
    IR_TRANSMISSION = "ir_transmission"
    IR_DRIFT = "ir_drift"
    NIR = "nir"
    UV_VIS = "uv_vis"
    FLUORESCENCE = "fluorescence"
    NMR_1H = "nmr_1h"
    MS = "ms"


class AxisKind(StrEnum):
    """Semantics of the spectral axis.

    ``RAMAN_SHIFT_CM1`` and ``ABSOLUTE_WAVENUMBER_CM1`` share a unit and mean different things;
    conflating them is failure code ``F01``. ``ENERGY_EV`` extends the brief's three-member list
    because §3 invariant 5 names eV explicitly.
    """

    RAMAN_SHIFT_CM1 = "raman_shift_cm-1"
    ABSOLUTE_WAVENUMBER_CM1 = "wavenumber_cm-1"
    WAVELENGTH_NM = "wavelength_nm"
    ENERGY_EV = "energy_eV"


class IntensityKind(StrEnum):
    """Semantics of the intensity axis."""

    COUNTS = "counts"
    ABSORBANCE = "absorbance"
    TRANSMITTANCE = "transmittance"
    REFLECTANCE = "reflectance"
    KUBELKA_MUNK = "kubelka_munk"
    ARBITRARY = "arbitrary"


class RamanBranch(StrEnum):
    """Which side of the Rayleigh line a Raman shift refers to."""

    STOKES = "stokes"
    ANTI_STOKES = "anti_stokes"


class Origin(StrEnum):
    """Where a spectrum came from. Simulated is never silently treated as experimental."""

    EXPERIMENTAL = "experimental"
    SIMULATED = "simulated"
    AUGMENTED = "augmented"


class ClaimType(StrEnum):
    """The epistemic status of an emitted statement.

    ``CAUSAL_CLAIM`` requires interventional evidence and is normally forbidden by the refusal
    contract; it exists so that an attempt to make one is a *typed* violation (``F18``) rather
    than an untracked sentence in a report.
    """

    CORRELATION = "correlation"
    MECHANISTIC_HYPOTHESIS = "mechanistic_hypothesis"
    CAUSAL_CLAIM = "causal_claim"


class Maturity(StrEnum):
    """Evidence status of a registered method. Governs whether it may surface in a conclusion."""

    ESTABLISHED = "established"
    EMERGING = "emerging"
    HYPOTHESIZED = "hypothesized"
    REFUTED = "refuted"
    DEPRECATED = "deprecated"


class TaskKind(StrEnum):
    """The evaluation task families of brief §11.1."""

    PREPROCESSING_QUALITY = "A_preprocessing_quality"
    DENOISING = "B_denoising"
    PEAK_DETECTION = "C_peak_detection"
    CLASSIFICATION = "D_classification"
    QUANTIFICATION = "E_quantification"
    MIXTURE_DECOMPOSITION = "F_mixture_decomposition"
    COMPOUND_RETRIEVAL = "G_compound_retrieval"
    FUNCTIONAL_GROUP_ID = "H_functional_group_id"
    SPECTRUM_TO_STRUCTURE = "I_spectrum_to_structure"
    STRUCTURE_TO_SPECTRUM = "J_structure_to_spectrum"
    CROSS_INSTRUMENT_TRANSFER = "K_cross_instrument_transfer"
    CROSS_DOMAIN_GENERALISATION = "L_cross_domain_generalisation"
    UNCERTAINTY_ABSTENTION = "M_uncertainty_abstention"
    EXPLANATION_FAITHFULNESS = "N_explanation_faithfulness"
    PHYSICS_CONSISTENCY = "O_physics_consistency"
    MEASUREMENT_PLANNING = "P_measurement_planning"


class FailureCode(StrEnum):
    """Brief §11.4. Every evaluation run emits a histogram over these."""

    F01_UNIT_AXIS = "F01_unit_axis"
    F02_PREPROCESSING_ARTIFACT = "F02_preprocessing_artifact"
    F03_SPIKE_LEAKAGE = "F03_spike_leakage"
    F04_FLUORESCENCE = "F04_fluorescence"
    F05_SATURATION = "F05_saturation"
    F06_DOMAIN_SHIFT = "F06_domain_shift"
    F07_RESOLUTION_MISMATCH = "F07_resolution_mismatch"
    F08_NON_IDENTIFIABILITY = "F08_non_identifiability"
    F09_MIXTURE_AMBIGUITY = "F09_mixture_ambiguity"
    F10_HALLUCINATED_ASSIGNMENT = "F10_hallucinated_assignment"
    F11_OVERCONFIDENCE_UNDER_SHIFT = "F11_overconfidence_under_shift"
    F12_RETRIEVAL_MISS = "F12_retrieval_miss"
    F13_SIMULATION_BIAS = "F13_simulation_bias"
    F14_SHORTCUT_LEARNING = "F14_shortcut_learning"
    F15_LEAKAGE = "F15_leakage"
    F16_LABEL_NOISE = "F16_label_noise"
    F17_INVALID_MOLECULE = "F17_invalid_molecule"
    F18_UNWARRANTED_CAUSAL_CLAIM = "F18_unwarranted_causal_claim"


class NonIdentifiability(StrEnum):
    """Named classes for section 8 of the Sample Report: *what could not be determined*."""

    ISOMER_DEGENERATE = "isomer_degenerate"
    POLYMORPH_DEGENERATE = "polymorph_degenerate"
    LIBRARY_GAP = "library_gap"
    RESOLUTION_LIMITED = "resolution_limited"
    FLUORESCENCE_DOMINATED = "fluorescence_dominated"
    MIXTURE_UNDERDETERMINED = "mixture_underdetermined"
    SNR_LIMITED = "snr_limited"
    RANGE_LIMITED = "range_limited"


class MeasurementVerdict(StrEnum):
    """Section 1 of the Sample Report. ``REJECT`` forbids any identification downstream."""

    USABLE = "usable"
    DEGRADED = "degraded"
    REJECT = "reject"


class ActionKind(StrEnum):
    """The agent action space of brief §4.1."""

    ACQUIRE = "acquire"
    PREPROCESS = "preprocess"
    QUERY_LIBRARY = "query_library"
    SIMULATE = "simulate"
    RUN_CHEMOMETRICS = "run_chemometrics"
    CHECK_CALIBRATION = "check_calibration"
    REQUEST_MODALITY = "request_modality"
    REJECT_HYPOTHESIS = "reject_hypothesis"
    ABSTAIN = "abstain"
    COMMIT = "commit"


# --------------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessingStep:
    """One applied processing operation. The chain is ordered and append-only."""

    name: str
    params: Mapping[str, Any]
    version: str
    timestamp: str

    def __post_init__(self) -> None:
        """Validate that the step is identifiable."""
        if not self.name:
            raise ProvenanceError("ProcessingStep.name must be non-empty")
        if not self.version:
            raise ProvenanceError(f"ProcessingStep {self.name!r} must declare a version")


@dataclass(frozen=True)
class SourceRecord:
    """Where a spectrum came from, with the licence it came under.

    ``licence`` is a key into :mod:`usis.data.licences`. ``"UNVERIFIED"`` is a legal value and a
    deliberately loud one: it is non-redistributable, so the licence gate blocks it in CI.
    """

    dataset_id: str
    licence: str
    retrieved_at: str
    sha256: str = ""
    url: str | None = None
    doi: str | None = None

    def __post_init__(self) -> None:
        """Validate the identifying fields."""
        if not self.dataset_id:
            raise ProvenanceError("SourceRecord.dataset_id must be non-empty")
        if not self.licence:
            raise ProvenanceError(
                f"SourceRecord {self.dataset_id!r} must declare a licence; "
                "use 'UNVERIFIED' if it is genuinely unknown"
            )
        if self.sha256 and len(self.sha256) != 64:
            raise ProvenanceError(
                f"SourceRecord {self.dataset_id!r}: sha256 must be 64 hex chars, "
                f"got {len(self.sha256)}"
            )


# --------------------------------------------------------------------------------------------
# Measurement description
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class InstrumentSpec:
    """The instrument a spectrum was measured on.

    ``instrument_id`` is the key for the instrument-disjoint split and ``lab_id`` for the
    lab-disjoint split, so both are stable strings rather than free text.
    """

    instrument_id: str
    manufacturer: str | None = None
    model: str | None = None
    excitation_nm: float | None = None
    grating: str | None = None
    objective: str | None = None
    detector: str | None = None
    lab_id: str | None = None

    def __post_init__(self) -> None:
        """Validate identifiers and the excitation wavelength."""
        if not self.instrument_id:
            raise SpectrumValidationError("InstrumentSpec.instrument_id must be non-empty")
        if self.excitation_nm is not None and self.excitation_nm <= 0:
            raise SpectrumValidationError(
                f"excitation_nm must be positive, got {self.excitation_nm}"
            )


@dataclass(frozen=True)
class AcquisitionParams:
    """Acquisition settings. Every field is optional and every absence is priced downstream."""

    exposure_s: float | None = None
    accumulations: int | None = None
    laser_power_mw: float | None = None
    aperture: str | None = None
    temperature_k: float | None = None

    def __post_init__(self) -> None:
        """Validate that the physical quantities are non-negative where present."""
        for name in ("exposure_s", "accumulations", "laser_power_mw", "temperature_k"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise SpectrumValidationError(f"AcquisitionParams.{name} must be >= 0, got {value}")


@dataclass(frozen=True, eq=False)
class AxisSpec:
    """A spectral axis: its semantics, its sample points, and its resolution.

    ``values`` must be strictly monotonic and finite. ``resolution_fwhm`` is in axis units and is
    optional but required for any cross-instrument comparison: without it a resolution-shift
    split cannot be constructed and a convolution cannot be inverted or matched.
    """

    kind: AxisKind
    values: FloatArray
    uniform: bool
    resolution_fwhm: float | None = None

    def __post_init__(self) -> None:
        """Validate monotonicity, finiteness and the uniformity claim, then freeze the array."""
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise SpectrumValidationError(f"axis values must be 1-D, got shape {values.shape}")
        if values.size < 2:
            raise SpectrumValidationError("axis must have at least 2 points")
        if not np.all(np.isfinite(values)):
            raise SpectrumValidationError("axis values must all be finite")
        diffs = np.diff(values)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            raise SpectrumValidationError("axis values must be strictly monotonic")
        if self.uniform and not _is_uniform(diffs):
            raise SpectrumValidationError(
                "axis declared uniform but spacing varies by more than the tolerance"
            )
        if self.resolution_fwhm is not None and self.resolution_fwhm <= 0:
            raise SpectrumValidationError(
                f"resolution_fwhm must be positive, got {self.resolution_fwhm}"
            )
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_values(
        cls,
        kind: AxisKind,
        values: Sequence[float] | FloatArray,
        resolution_fwhm: float | None = None,
    ) -> AxisSpec:
        """Build an axis, inferring the uniformity flag from the sample spacing."""
        array = np.asarray(values, dtype=np.float64)
        uniform = bool(array.size >= 2 and _is_uniform(np.diff(array)))
        return cls(kind=kind, values=array, uniform=uniform, resolution_fwhm=resolution_fwhm)

    @property
    def n_points(self) -> int:
        """Number of sample points."""
        return int(self.values.size)

    @property
    def span(self) -> tuple[float, float]:
        """The (low, high) extent of the axis, irrespective of storage direction."""
        return (float(self.values.min()), float(self.values.max()))

    @property
    def ascending(self) -> bool:
        """Whether the stored values increase with index."""
        return bool(self.values[-1] > self.values[0])

    @property
    def mean_step(self) -> float:
        """Mean absolute spacing between adjacent sample points."""
        return float(np.abs(np.diff(self.values)).mean())


def _is_uniform(diffs: FloatArray) -> bool:
    """Return whether the spacings are constant to within :data:`UNIFORM_AXIS_RTOL`."""
    if diffs.size == 0:
        return True
    return bool(np.allclose(diffs, diffs[0], rtol=UNIFORM_AXIS_RTOL, atol=0.0))


# --------------------------------------------------------------------------------------------
# Labels and evidence
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GroundTruth:
    """A dataset's stated answer for a sample.

    A label is an *assertion by a source*, not truth: ``label_confidence`` and ``annotation_note``
    exist so that label noise (``F16``) can be modelled rather than assumed away.

    ``components`` and ``targets`` are different things and the distinction is load-bearing.
    ``components`` are mixture fractions and are checked for closure; ``targets`` are named
    quantitative measurements in whatever unit the source used — concentrations in g/L, an octane
    number, a temperature — which do not sum to anything. Regression datasets need the second and
    would fail the closure check if forced into the first.
    """

    name: str | None = None
    smiles: str | None = None
    inchikey: str | None = None
    cas: str | None = None
    components: Mapping[str, float] | None = None
    targets: Mapping[str, float] | None = None
    chemical_state: Mapping[str, Any] | None = None
    label_confidence: float | None = None
    annotation_note: str | None = None

    def __post_init__(self) -> None:
        """Validate mixture fractions and the confidence range."""
        if self.label_confidence is not None and not 0.0 <= self.label_confidence <= 1.0:
            raise SpectrumValidationError(
                f"label_confidence must be in [0, 1], got {self.label_confidence}"
            )
        if self.components is not None:
            if any(v < 0 for v in self.components.values()):
                raise SpectrumValidationError("component fractions must be non-negative")
            total = sum(self.components.values())
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise SpectrumValidationError(
                    f"component fractions must satisfy closure (sum to 1), got {total}"
                )


@dataclass(frozen=True)
class Evidence:
    """One piece of support for or against a claim, with its epistemic status."""

    kind: str
    statement: str
    claim_type: ClaimType = ClaimType.CORRELATION
    weight: float | None = None
    source_ref: str | None = None
    conditions: str | None = None

    def __post_init__(self) -> None:
        """Validate the evidence is attributable and not an unwarranted causal claim."""
        if not self.kind or not self.statement:
            raise SpectrumValidationError("Evidence needs both a kind and a statement")
        if self.claim_type is ClaimType.CAUSAL_CLAIM and not self.source_ref:
            raise SpectrumValidationError(
                "a CAUSAL_CLAIM requires a source_ref pointing at interventional evidence "
                f"(failure code {FailureCode.F18_UNWARRANTED_CAUSAL_CLAIM})"
            )


# --------------------------------------------------------------------------------------------
# The spectrum
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Spectrum:
    """A physical measurement of matter, with everything needed to interpret it.

    Invariants enforced here: the intensity vector matches the axis length; ``sigma``, when
    present, matches too and is non-negative; an experimental spectrum names its instrument; the
    processing chain is a tuple, extended only through :meth:`with_processing`.

    Only ``EXPERIMENTAL`` requires an instrument. ``SIMULATED`` never had one, and ``AUGMENTED``
    inherits whatever its source had — degrading a simulated spectrum through the forward model
    produces something that *models* an instrument without having been measured on one, and
    demanding an instrument there would force us to invent a fake.
    """

    spectrum_id: SpectrumId
    modality: Modality
    origin: Origin
    axis: AxisSpec
    intensity: FloatArray
    intensity_kind: IntensityKind
    sample_id: str
    source: SourceRecord
    instrument: InstrumentSpec | None = None
    acquisition: AcquisitionParams | None = None
    processing: tuple[ProcessingStep, ...] = ()
    label: GroundTruth | None = None
    sigma: FloatArray | None = None

    def __post_init__(self) -> None:
        """Validate shapes, origin/instrument consistency, and freeze the arrays."""
        if not self.spectrum_id:
            raise SpectrumValidationError("Spectrum.spectrum_id must be non-empty")
        if not self.sample_id:
            raise SpectrumValidationError(
                f"Spectrum {self.spectrum_id!r}: sample_id must be non-empty; it is what the "
                "sample-disjoint split groups on"
            )
        intensity = np.asarray(self.intensity, dtype=np.float64)
        if intensity.ndim != 1:
            raise SpectrumValidationError(f"intensity must be 1-D, got shape {intensity.shape}")
        if intensity.size != self.axis.n_points:
            raise SpectrumValidationError(
                f"intensity has {intensity.size} points but the axis has {self.axis.n_points}"
            )
        intensity.setflags(write=False)
        object.__setattr__(self, "intensity", intensity)

        if self.sigma is not None:
            sigma = np.asarray(self.sigma, dtype=np.float64)
            if sigma.shape != intensity.shape:
                raise SpectrumValidationError(
                    f"sigma shape {sigma.shape} does not match intensity shape {intensity.shape}"
                )
            if np.any(sigma < 0):
                raise SpectrumValidationError("sigma must be non-negative")
            sigma.setflags(write=False)
            object.__setattr__(self, "sigma", sigma)

        if self.origin is Origin.EXPERIMENTAL and self.instrument is None:
            raise SpectrumValidationError(
                f"Spectrum {self.spectrum_id!r}: an experimental spectrum must name the "
                "instrument it was measured on"
            )
        if not isinstance(self.processing, tuple):
            raise ProvenanceError("Spectrum.processing must be a tuple (ordered, append-only)")

    def with_processing(self, step: ProcessingStep, **changes: Any) -> Spectrum:
        """Return a new spectrum with ``step`` appended to the processing chain.

        This is the only sanctioned way to record processing. It cannot shorten or reorder the
        existing chain, which is what makes the record append-only in practice and not only in
        principle.
        """
        if "processing" in changes:
            raise ProvenanceError(
                "with_processing() controls the processing chain; pass the step, not a new chain"
            )
        return replace(self, processing=(*self.processing, step), **changes)

    @property
    def is_processed(self) -> bool:
        """Whether any processing step has been recorded."""
        return len(self.processing) > 0

    @property
    def excitation_nm(self) -> float | None:
        """Excitation wavelength, if the instrument declared one."""
        return None if self.instrument is None else self.instrument.excitation_nm


# --------------------------------------------------------------------------------------------
# Beliefs
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One structural hypothesis with its posterior mass and its two-sided evidence."""

    posterior: float
    smiles: str | None = None
    inchikey: str | None = None
    name: str | None = None
    supporting_evidence: tuple[Evidence, ...] = ()
    conflicting_evidence: tuple[Evidence, ...] = ()
    simulated_spectrum_id: SpectrumId | None = None
    cycle_consistency: float | None = None

    def __post_init__(self) -> None:
        """Validate the posterior range and that the candidate is identifiable at all."""
        if not 0.0 <= self.posterior <= 1.0:
            raise BeliefStateError(f"posterior must be in [0, 1], got {self.posterior}")
        if self.smiles is None and self.inchikey is None and self.name is None:
            raise BeliefStateError("a Candidate needs at least one identifier")


@dataclass(frozen=True)
class ComponentHypothesis:
    """One component of a mixture, with its fraction and a confidence interval."""

    candidate: Candidate
    fraction: float
    fraction_ci: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate non-negativity and interval ordering."""
        if self.fraction < 0.0:
            raise BeliefStateError(
                f"mixture fractions are non-negative by hard constraint, got {self.fraction}"
            )
        if self.fraction_ci is not None:
            low, high = self.fraction_ci
            if low > high:
                raise BeliefStateError(f"fraction_ci is inverted: {self.fraction_ci}")


@dataclass(frozen=True)
class MixtureHypothesis:
    """A composition hypothesis: how many components, which, in what proportion.

    Closure (fractions summing to 1) is enforced here rather than learned, per invariant 7.
    """

    components: tuple[ComponentHypothesis, ...]
    n_components_evidence: Mapping[str, float] | None = None
    residual_variance: float | None = None

    def __post_init__(self) -> None:
        """Enforce the closure constraint over the component fractions."""
        if not self.components:
            raise BeliefStateError("a MixtureHypothesis needs at least one component")
        total = sum(c.fraction for c in self.components)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise BeliefStateError(f"closure violated: fractions sum to {total}, expected 1.0")

    @property
    def n_components(self) -> int:
        """Number of hypothesised components."""
        return len(self.components)


@dataclass(frozen=True)
class Action:
    """A proposed next step, priced by expected information gain against expected cost."""

    kind: ActionKind
    params: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_information_gain: float | None = None
    expected_cost: float | None = None


@dataclass(frozen=True)
class BeliefState:
    """The system's answer: a posterior over candidates plus explicit open-set mass.

    ``sum(posterior) + p_none == 1`` is asserted at construction. There is no way to express a
    bare answer in this type, which is the point.
    """

    candidates: tuple[Candidate, ...]
    p_none: float
    epistemic: float | None = None
    aleatoric: float | None = None
    mixture: MixtureHypothesis | None = None
    evidence_log: tuple[Evidence, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    non_identifiability: tuple[NonIdentifiability, ...] = ()
    recommended_next_action: Action | None = None

    def __post_init__(self) -> None:
        """Assert the normalisation invariant, including the open-set residual."""
        if not 0.0 <= self.p_none <= 1.0:
            raise BeliefStateError(f"p_none must be in [0, 1], got {self.p_none}")
        total = sum(c.posterior for c in self.candidates) + self.p_none
        if not math.isclose(total, 1.0, abs_tol=POSTERIOR_SUM_TOL):
            raise BeliefStateError(
                f"sum(posterior) + p_none must equal 1, got {total:.12g}. This invariant is the "
                "difference between an open-set posterior and a softmax over a closed library."
            )

    @property
    def top(self) -> Candidate | None:
        """The highest-posterior candidate, or ``None`` when the belief is empty."""
        return max(self.candidates, key=lambda c: c.posterior, default=None)

    @property
    def entropy(self) -> float:
        """Shannon entropy in nats over the candidates plus the open-set mass.

        The open-set mass participates: a belief that is confident among its candidates but
        holds large ``p_none`` is *not* a confident belief.
        """
        masses = [c.posterior for c in self.candidates] + [self.p_none]
        return -sum(p * math.log(p) for p in masses if p > 0.0)

    def abstains_at(self, min_posterior: float) -> bool:
        """Whether the belief should abstain given a decision threshold on the top candidate."""
        top = self.top
        return top is None or top.posterior < min_posterior
