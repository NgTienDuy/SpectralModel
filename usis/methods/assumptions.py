"""Machine-readable preconditions, and the four answers a check can give.

Brief §6.1 asserts that *silently applying a method outside its validity domain is the single most
common way published spectroscopy results fail to replicate*. Novelty candidate N7 takes that
assertion and makes it measurable: if every method declares its domain in a form the runtime can
check, the violation rate becomes a number a comparison can report.

**The four-valued status is the whole design.** A boolean check has to answer "is this satisfied?"
for a spectrum whose metadata does not say - and both answers are wrong. ``True`` is the silent
failure ADR-0001 §S2 warns about in a different guise; ``False`` is a false alarm that will train
the reader to ignore the audit. So a check returns one of:

- ``SATISFIED`` - the measurement was examined and meets the declared domain;
- ``VIOLATED`` - the measurement was examined and does not;
- ``UNKNOWN`` - the metadata that would decide it is absent. Fixable by recording it;
- ``NOT_CHECKABLE`` - no procedure in this repository decides it from these inputs. Fixable only
  by a different measurement, which the check names in ``evidence_needed``.

Only ``SATISFIED`` clears. The other three all reach the report, each with the thing that would
resolve it - the same discipline the Sample Report applies one level up in section 9.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np

from usis.core.types import FailureCode, IntensityKind, Modality, Spectrum
from usis.measurement.diagnostics import band_statistics, median_band_spacing, noise_level
from usis.measurement.forward_model import ScaleMode, signal_scale
from usis.methods.inputs import MethodInputs

__all__ = [
    "AssumesLinearMixing",
    "AssumesNoResonanceEnhancement",
    "Assumption",
    "AssumptionCheck",
    "AssumptionStatus",
    "MaxResolutionFwhm",
    "MaxSaturatedFraction",
    "MinSnr",
    "RequiresAxisCoverage",
    "RequiresBandSeparation",
    "RequiresBaselineCorrected",
    "RequiresCommonAxis",
    "RequiresIntensityKind",
    "RequiresLibraryCoverage",
    "RequiresMetadata",
    "RequiresModality",
    "RequiresNonNegativeIntensity",
    "RequiresUniformAxis",
    "check_all",
    "estimate_snr",
    "saturated_fraction",
]


class AssumptionStatus(StrEnum):
    """The four answers a precondition check can honestly give."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    NOT_CHECKABLE = "not_checkable"

    @property
    def clears(self) -> bool:
        """Whether this status lets a method run with nothing to declare."""
        return self is AssumptionStatus.SATISFIED

    @property
    def severity(self) -> int:
        """Rank used when folding one assumption's checks over a batch of spectra.

        ``VIOLATED`` outranks the two undecided statuses because it is positive evidence of a
        problem; ``UNKNOWN`` outranks ``NOT_CHECKABLE`` because it is the one the operator can
        act on.
        """
        return {
            AssumptionStatus.SATISFIED: 0,
            AssumptionStatus.NOT_CHECKABLE: 1,
            AssumptionStatus.UNKNOWN: 2,
            AssumptionStatus.VIOLATED: 3,
        }[self]


@dataclass(frozen=True)
class AssumptionCheck:
    """The outcome of checking one declared assumption against one method invocation."""

    assumption: str
    status: AssumptionStatus
    detail: str
    hard: bool
    failure_code: FailureCode
    evidence_needed: str = ""

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view, written into every comparison record."""
        return asdict(self)


def estimate_snr(spectrum: Spectrum) -> float:
    """Signal-to-noise as peak-to-floor amplitude over the noise sigma.

    The scale is :attr:`~usis.measurement.forward_model.ScaleMode.AMPLITUDE`, not the median, and
    that is not a detail. E2 measured the same spectra under both conventions and the estimates
    differed by two orders of magnitude, because for a simulated spectrum the median *is* the
    near-zero floor between bands while for a real one it *is* the baseline. Amplitude over the
    floor is the quantity a spectroscopist means by SNR in both cases.
    """
    sigma = noise_level(spectrum)
    if sigma <= 0.0:
        return float("inf")
    amplitude = signal_scale(np.asarray(spectrum.intensity, dtype=np.float64), ScaleMode.AMPLITUDE)
    return float(amplitude / sigma)


@dataclass(frozen=True)
class Assumption(ABC):
    """A precondition a method declares and the runtime checks.

    ``hard`` says what a violation means: a hard violation refuses the run, a soft one runs and is
    reported. It is keyword-only so that a declaration reads
    ``MinSnr(3.0, hard=False)`` rather than ``MinSnr(3.0, False)``.

    Subclasses that answer per-spectrum implement :meth:`check_one` and inherit the fold over the
    batch. Subclasses that answer about the invocation as a whole - a library, a calibration
    series - override :meth:`check` directly.
    """

    hard: bool = field(default=True, kw_only=True)

    #: Stable identifier, in the brief's own vocabulary where it named one.
    name: ClassVar[str]
    #: Why the method needs it. Reaches the reader of the audit, so write it for them.
    rationale: ClassVar[str]
    #: The failure code a violation of this assumption manifests as.
    failure_code: ClassVar[FailureCode]

    def describe(self) -> str:
        """The assumption with the parameter it was declared with, e.g. ``valid_snr_above=3.0``."""
        return self.name

    def check(self, inputs: MethodInputs) -> AssumptionCheck:
        """Check every input spectrum and report the worst outcome, naming the offender."""
        outcomes = [(s, self.check_one(s, inputs)) for s in inputs.spectra]
        worst_spectrum, worst = max(outcomes, key=lambda pair: pair[1].status.severity)
        if worst.status.clears:
            return worst
        offenders = sum(1 for _, outcome in outcomes if outcome.status is worst.status)
        if len(outcomes) == 1:
            return worst
        return dataclass_replace_detail(
            worst,
            f"{offenders}/{len(outcomes)} spectra, worst {worst_spectrum.spectrum_id}: "
            f"{worst.detail}",
        )

    @abstractmethod
    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Check one spectrum. Override :meth:`check` instead for whole-invocation assumptions."""

    # -- helpers used by subclasses ----------------------------------------------------------
    def _outcome(
        self, status: AssumptionStatus, detail: str, evidence_needed: str = ""
    ) -> AssumptionCheck:
        return AssumptionCheck(
            assumption=self.describe(),
            status=status,
            detail=detail,
            hard=self.hard,
            failure_code=self.failure_code,
            evidence_needed=evidence_needed,
        )

    def satisfied(self, detail: str) -> AssumptionCheck:
        """The measurement was examined and meets the declared domain."""
        return self._outcome(AssumptionStatus.SATISFIED, detail)

    def violated(self, detail: str) -> AssumptionCheck:
        """The measurement was examined and does not meet the declared domain."""
        return self._outcome(AssumptionStatus.VIOLATED, detail)

    def unknown(self, detail: str, evidence_needed: str) -> AssumptionCheck:
        """The metadata that would decide this is absent; ``evidence_needed`` names it."""
        return self._outcome(AssumptionStatus.UNKNOWN, detail, evidence_needed)

    def not_checkable(self, detail: str, evidence_needed: str) -> AssumptionCheck:
        """Nothing here decides this; ``evidence_needed`` names the measurement that would."""
        return self._outcome(AssumptionStatus.NOT_CHECKABLE, detail, evidence_needed)


def dataclass_replace_detail(check: AssumptionCheck, detail: str) -> AssumptionCheck:
    """A copy of ``check`` with a rewritten ``detail``, used when folding over a batch."""
    return AssumptionCheck(
        assumption=check.assumption,
        status=check.status,
        detail=detail,
        hard=check.hard,
        failure_code=check.failure_code,
        evidence_needed=check.evidence_needed,
    )


# ------------------------------------------------------------------------------------------------
# Signal-quality domain
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MinSnr(Assumption):
    """``valid_snr_above=X`` from brief §6.1."""

    threshold: float

    name: ClassVar[str] = "valid_snr_above"
    rationale: ClassVar[str] = (
        "Below its SNR floor a method's output is dominated by noise realisation rather than by "
        "the sample, and the resulting posterior is confident about the wrong thing."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F08_NON_IDENTIFIABILITY

    def describe(self) -> str:
        """Name and threshold."""
        return f"{self.name}={self.threshold:g}"

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Estimate SNR from the spectrum itself; nothing here needs metadata."""
        snr = estimate_snr(spectrum)
        detail = f"estimated SNR {snr:.2f}, required > {self.threshold:g}"
        if snr > self.threshold:
            return self.satisfied(detail)
        return self.violated(detail)


@dataclass(frozen=True)
class MaxSaturatedFraction(Assumption):
    """No more than ``fraction`` of samples may sit on a clipped plateau.

    Detected as runs of two or more consecutive samples at the observed maximum: a single sample
    at the maximum is just the tallest band, whereas a flat top is the detector ceiling. Brief
    §11.4 ``F05``; ``docs/knowledge/measurement.md`` §7 states the consequence - a saturated
    spectrum cannot support a quantitative claim at all.
    """

    fraction: float = 0.0

    name: ClassVar[str] = "max_saturated_fraction"
    rationale: ClassVar[str] = (
        "Clipping is a hard nonlinearity: the intensity that was actually incident on the "
        "detector is not recoverable, so any quantitative use of the clipped region is invented."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F05_SATURATION

    def describe(self) -> str:
        """Name and threshold."""
        return f"{self.name}={self.fraction:g}"

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Count samples on flat tops at the observed maximum."""
        observed = saturated_fraction(spectrum)
        detail = f"{observed:.4f} of samples on a clipped plateau, allowed {self.fraction:g}"
        if observed <= self.fraction:
            return self.satisfied(detail)
        return self.violated(detail)


def saturated_fraction(spectrum: Spectrum) -> float:
    """Fraction of samples lying on a plateau at the observed maximum intensity."""
    intensity = np.asarray(spectrum.intensity, dtype=np.float64)
    if intensity.size == 0:
        return 0.0
    ceiling = float(intensity.max())
    at_ceiling = np.isclose(intensity, ceiling, rtol=0.0, atol=abs(ceiling) * 1e-12)
    if not at_ceiling.any():
        return 0.0
    neighbours = np.zeros_like(at_ceiling)
    neighbours[:-1] |= at_ceiling[:-1] & at_ceiling[1:]
    neighbours[1:] |= at_ceiling[1:] & at_ceiling[:-1]
    return float(neighbours.sum() / intensity.size)


# ------------------------------------------------------------------------------------------------
# Axis domain
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MaxResolutionFwhm(Assumption):
    """``valid_resolution_below=Y`` from brief §6.1, in axis units."""

    threshold_cm1: float

    name: ClassVar[str] = "valid_resolution_below"
    rationale: ClassVar[str] = (
        "A method that resolves close bands cannot do so once the instrument response is wider "
        "than their separation; the bands are gone from the measurement, not merely hard to see."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F07_RESOLUTION_MISMATCH

    def describe(self) -> str:
        """Name and threshold."""
        return f"{self.name}={self.threshold_cm1:g}"

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Read the declared instrument response; refuse to infer it from the data."""
        declared = spectrum.axis.resolution_fwhm
        if declared is None:
            return self.unknown(
                f"axis declares no resolution_fwhm; required < {self.threshold_cm1:g} cm-1",
                "the instrument's spectral resolution, from its specification or a lamp "
                "calibration measurement. Estimating it from band widths would confuse the "
                "instrument response with the sample's intrinsic linewidth - E2 measured those "
                "to be 4 and ~12 cm-1 respectively on the same records.",
            )
        detail = f"declared resolution {declared:g} cm-1, required < {self.threshold_cm1:g} cm-1"
        if declared < self.threshold_cm1:
            return self.satisfied(detail)
        return self.violated(detail)


@dataclass(frozen=True)
class RequiresUniformAxis(Assumption):
    """The axis step must be constant to within ``tolerance`` (relative).

    Not a formality. M1 measured two real ROD records whose wavenumber step varies by 0.23% and
    **25.5%** across the axis, because a grating disperses linearly in wavelength and ``1/lambda``
    is not linear. Any method whose kernel is expressed in cm-1 - convolution, a fixed-width
    Savitzky-Golay window, a peak-width statistic - has no fixed width in samples on such an axis.
    """

    tolerance: float = 0.01

    name: ClassVar[str] = "requires_uniform_axis"
    rationale: ClassVar[str] = (
        "A kernel specified in cm-1 spans a different number of samples at each end of a "
        "non-uniform axis, so the operation applied is not the operation declared."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F02_PREPROCESSING_ARTIFACT

    def describe(self) -> str:
        """Name and tolerance."""
        return f"{self.name}(tolerance={self.tolerance:g})"

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Measure the spread of the axis step rather than trusting the ``uniform`` flag."""
        steps = np.abs(np.diff(np.asarray(spectrum.axis.values, dtype=np.float64)))
        if steps.size == 0:
            return self.unknown(
                "axis has fewer than two samples", "a spectrum with a measurable axis step"
            )
        spread = float((steps.max() - steps.min()) / steps.mean())
        detail = f"axis step varies by {spread * 100:.2f}%, tolerance {self.tolerance * 100:.2f}%"
        if spread <= self.tolerance:
            return self.satisfied(detail)
        return self.violated(detail + "; resample onto a uniform axis first, as a recorded step")


@dataclass(frozen=True)
class RequiresAxisCoverage(Assumption):
    """The spectrum must span ``[low_cm1, high_cm1]``, the window the method reads."""

    low_cm1: float
    high_cm1: float

    name: ClassVar[str] = "requires_axis_coverage"
    rationale: ClassVar[str] = (
        "A method that reads a diagnostic window cannot report on a region the measurement never "
        "covered; absence of a band outside the recorded range is not evidence of its absence."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F08_NON_IDENTIFIABILITY

    def describe(self) -> str:
        """Name and window."""
        return f"{self.name}=[{self.low_cm1:g}, {self.high_cm1:g}]"

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Compare the recorded span against the required window."""
        values = np.asarray(spectrum.axis.values, dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        detail = (
            f"spectrum spans [{low:g}, {high:g}], method reads "
            f"[{self.low_cm1:g}, {self.high_cm1:g}]"
        )
        if low <= self.low_cm1 and high >= self.high_cm1:
            return self.satisfied(detail)
        return self.violated(detail)


@dataclass(frozen=True)
class RequiresBandSeparation(Assumption):
    """Bands must be separated by at least ``min_ratio`` times their own width.

    Declared by any method whose statistic is computed over a whole spectrum but attributed to
    single-band physics. It exists because the confound was *measured*: varying only band spacing,
    at fixed line shape and width, moved ``shape.broadening_origin``'s ratio by a factor of 1.6,
    while changing the line shape itself moved it by 1.25. Bands packed closer than a few widths
    apart overlap, and an autocorrelation over overlapping bands describes the packing.

    This is the assumption layer doing the job it was built for: a confound found during
    development becomes a declared, machine-checked precondition rather than a caveat in a
    docstring nobody reads at the call site.
    """

    min_ratio: float = 6.0

    name: ClassVar[str] = "requires_band_separation"
    rationale: ClassVar[str] = (
        "A whole-spectrum statistic attributed to single-band physics measures band packing "
        "instead when the bands overlap."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F08_NON_IDENTIFIABILITY

    def describe(self) -> str:
        """Name and the required spacing-to-width ratio."""
        return f"{self.name}={self.min_ratio:g}"

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Compare median band spacing against median band width."""
        _, median_fwhm, _ = band_statistics(spectrum)
        spacing = median_band_spacing(spectrum)
        if not np.isfinite(spacing) or not np.isfinite(median_fwhm) or median_fwhm <= 0.0:
            return self.unknown(
                "fewer than two bands with measurable widths were detected, so no "
                "spacing-to-width ratio exists",
                "a spectrum with at least two resolved bands, or a method that does not depend "
                "on their separation",
            )
        ratio = spacing / median_fwhm
        detail = (
            f"median band spacing {spacing:.1f} cm-1 over median FWHM {median_fwhm:.1f} cm-1 "
            f"= {ratio:.2f}, required >= {self.min_ratio:g}"
        )
        if ratio >= self.min_ratio:
            return self.satisfied(detail)
        return self.violated(detail + "; the bands overlap and the statistic describes the packing")


# ------------------------------------------------------------------------------------------------
# Metadata and modality domain
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RequiresModality(Assumption):
    """The method is defined only for the listed modalities."""

    modalities: frozenset[Modality]

    name: ClassVar[str] = "requires_modality"
    rationale: ClassVar[str] = (
        "Selection rules, intensity conventions and band positions differ between modalities; a "
        "method calibrated on one is not silently valid on another."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F06_DOMAIN_SHIFT

    def describe(self) -> str:
        """Name and the permitted modalities."""
        return f"{self.name}={{{', '.join(sorted(self.modalities))}}}"

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """A declared field, so this one is always decidable."""
        detail = f"spectrum modality {spectrum.modality}"
        if spectrum.modality in self.modalities:
            return self.satisfied(detail)
        return self.violated(detail + f", method accepts {sorted(self.modalities)}")


@dataclass(frozen=True)
class RequiresIntensityKind(Assumption):
    """The method's arithmetic is valid only for the listed intensity conventions."""

    kinds: frozenset[IntensityKind]

    name: ClassVar[str] = "requires_intensity_kind"
    rationale: ClassVar[str] = (
        "Beer-Lambert additivity holds in absorbance and not in transmittance or reflectance, so "
        "a linear-mixing method applied to the wrong convention is solving a different equation."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F01_UNIT_AXIS

    def describe(self) -> str:
        """Name and the permitted conventions."""
        return f"{self.name}={{{', '.join(sorted(self.kinds))}}}"

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """A declared field, so this one is always decidable."""
        detail = f"spectrum intensity_kind {spectrum.intensity_kind}"
        if spectrum.intensity_kind in self.kinds:
            return self.satisfied(detail)
        return self.violated(detail + f", method accepts {sorted(self.kinds)}")


@dataclass(frozen=True)
class RequiresMetadata(Assumption):
    """Named metadata fields must be present, addressed as dotted paths on the spectrum."""

    fields: tuple[str, ...]

    name: ClassVar[str] = "requires_metadata"
    rationale: ClassVar[str] = (
        "A method that needs the excitation wavelength or the exposure time cannot substitute a "
        "typical value: the brief's input contract says missing metadata widens the posterior, "
        "which requires knowing that it is missing."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F01_UNIT_AXIS

    def describe(self) -> str:
        """Name and the required paths."""
        return f"{self.name}={list(self.fields)}"

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Resolve each dotted path; a ``None`` anywhere along it counts as absent."""
        missing = [path for path in self.fields if _resolve(spectrum, path) is None]
        if not missing:
            return self.satisfied(f"all of {list(self.fields)} present")
        return self.unknown(
            f"missing {missing}",
            f"the source record or instrument log carrying {missing}",
        )


def _resolve(spectrum: Spectrum, path: str) -> Any:
    """Walk a dotted attribute path, returning ``None`` at the first absent segment."""
    current: Any = spectrum
    for segment in path.split("."):
        current = getattr(current, segment, None)
        if current is None:
            return None
    return current


# ------------------------------------------------------------------------------------------------
# Preprocessing-state domain
# ------------------------------------------------------------------------------------------------
#: Substrings identifying a baseline or background removal step in a processing chain. Matched
#: against ``ProcessingStep.name``, which is why every operator in usis.measurement names itself.
_BASELINE_STEP_MARKERS: tuple[str, ...] = (
    "baseline",
    "background_subtract",
    "airpls",
    "snip",
    "rubberband",
    "detrend",
)
#: Source-declared statements that a chain step explicitly asserts *no* correction was applied.
_NO_BASELINE_MARKERS: tuple[str, ...] = ("no_baseline", "no_background", "not_baseline_corrected")


@dataclass(frozen=True)
class RequiresBaselineCorrected(Assumption):
    """``requires_baseline_corrected`` from brief §6.1, decided from the processing chain.

    Three outcomes, and the third is the one that matters. A chain naming a baseline step is
    ``SATISFIED``; a chain in which the *source itself* declared that no background subtraction
    was performed - which is exactly what the ROD loader records, verbatim, with
    ``version="source-declared"`` - is ``VIOLATED``; an empty or silent chain is ``UNKNOWN``,
    because "nothing recorded" and "nothing applied" are different claims and the loaders were
    written specifically so the difference survives.
    """

    name: ClassVar[str] = "requires_baseline_corrected"
    rationale: ClassVar[str] = (
        "Methods that treat intensity as proportional to concentration, or that integrate band "
        "areas, attribute the background to the sample when it has not been removed."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F02_PREPROCESSING_ARTIFACT

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Read the processing chain; never infer correction from the shape of the data."""
        names = [step.name.lower() for step in spectrum.processing]
        params = " ".join(
            f"{key}={value}".lower()
            for step in spectrum.processing
            for key, value in step.params.items()
        )
        haystack = " ".join(names) + " " + params
        if any(marker in haystack for marker in _NO_BASELINE_MARKERS):
            return self.violated(
                "the source declares that no background subtraction or baseline correction was "
                "applied; the recorded chain is "
                f"{names or '[]'}"
            )
        if any(marker in name for name in names for marker in _BASELINE_STEP_MARKERS):
            return self.satisfied(f"processing chain contains a baseline step: {names}")
        return self.unknown(
            f"processing chain {names or '[]'} names no baseline step and does not declare its "
            "absence",
            "the acquisition or vendor-software record stating whether a background was "
            "subtracted. An empty chain is silence, not evidence of a raw spectrum.",
        )


@dataclass(frozen=True)
class AssumesNoResonanceEnhancement(Assumption):
    """``assumes_no_resonance_enhancement`` from brief §6.1.

    Partly decidable, and the partition is the useful part. A SERS measurement violates it
    outright - plasmonic enhancement is the point of the technique. Ordinary Raman is ``UNKNOWN``:
    ruling out electronic resonance needs an excitation series, which is a measurement, not a
    field. Reporting plain Raman as satisfied would be exactly the silent pass this module exists
    to prevent.
    """

    name: ClassVar[str] = "assumes_no_resonance_enhancement"
    rationale: ClassVar[str] = (
        "Resonance and surface enhancement rescale band intensities by orders of magnitude and "
        "non-uniformly, so relative intensities stop being a property of the molecule alone."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F10_HALLUCINATED_ASSIGNMENT

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Decide from the modality; refuse to guess for ordinary Raman."""
        if spectrum.modality is Modality.RAMAN_SERS:
            return self.violated(
                "modality is raman_sers: enhancement is intrinsic to the measurement, and "
                "relative band intensities cannot be read as molecular"
            )
        return self.unknown(
            f"modality {spectrum.modality} does not by itself exclude electronic resonance",
            "an excitation-wavelength series on the same sample; a band whose relative intensity "
            "tracks the excitation is resonance-enhanced. EXCITATION_DISJOINT exists as a split "
            "kind for precisely this reason.",
        )


@dataclass(frozen=True)
class AssumesLinearMixing(Assumption):
    """``assumes_linear_mixing`` from brief §6.1 - and it is ``NOT_CHECKABLE`` from one spectrum.

    This is the class of assumption the four-valued status was added for. Additivity of component
    spectra is a statement about a *series* of samples, so no examination of a single measurement
    can support or refute it. The check therefore declines, and names the experiment that decides:
    a concentration series with independently determined concentrations. RamanBench's Bioprocess
    Analytes set is such a series and is CC BY 4.0, so this is a gap with a route out of it, not a
    permanent unknown.
    """

    name: ClassVar[str] = "assumes_linear_mixing"
    rationale: ClassVar[str] = (
        "Unmixing solves S = C A for C; if the true mapping is nonlinear - saturation, "
        "self-absorption, particle-size or scattering effects - the residual is absorbed into "
        "fitted concentrations rather than reported."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F09_MIXTURE_AMBIGUITY

    def check(self, inputs: MethodInputs) -> AssumptionCheck:
        """Whole-invocation, not per-spectrum: linearity is a property of a series."""
        n_calibration = len(inputs.calibration)
        return self.not_checkable(
            f"additivity is a property of a concentration series; {len(inputs.spectra)} "
            f"spectra and {n_calibration} calibration spectra were supplied, and no "
            "independently determined concentrations accompany them",
            "a graded concentration series with assay-confirmed concentrations - e.g. "
            "RamanBench's Bioprocess Analytes (three analytes, enzymatic assay, CC BY 4.0) - "
            "scored for residual nonlinearity. Until then this assumption is declared and "
            "unverified, and any mixture claim inherits that.",
        )

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Unreachable: :meth:`check` answers for the whole invocation."""
        raise NotImplementedError("AssumesLinearMixing answers per invocation, not per spectrum")


@dataclass(frozen=True)
class RequiresLibraryCoverage(Assumption):
    """``requires_library_coverage`` from brief §6.1.

    A retrieval method with no library supplied is ``VIOLATED`` - it has nothing to retrieve from.
    With a library whose membership is not enumerated it is ``UNKNOWN``. With an enumerated
    library and a labelled spectrum whose key is absent it is ``VIOLATED`` with ``F12``, which is
    the honest reading: the correct answer is not among the options, so the method's top candidate
    is wrong by construction and its posterior mass belongs to ``p_none``.
    """

    name: ClassVar[str] = "requires_library_coverage"
    rationale: ClassVar[str] = (
        "A closed-set matcher can only return a library entry, so when the true substance is "
        "absent every candidate it offers is wrong and its confidence is meaningless."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F12_RETRIEVAL_MISS

    def check(self, inputs: MethodInputs) -> AssumptionCheck:
        """Whole-invocation: the library is a property of the invocation, not of a spectrum."""
        library = inputs.library
        if library is None:
            return self.violated(
                "no reference library was supplied, and this method can only return entries "
                "from one"
            )
        if library.inchikeys is None:
            return self.unknown(
                f"library {library.library_id} declares {library.n_entries} entries but its "
                "membership is not enumerated here",
                "the library's key list, so that coverage of the queried substance can be "
                "decided rather than assumed",
            )
        labelled = [s for s in inputs.spectra if s.label is not None and s.label.inchikey]
        if not labelled:
            return self.unknown(
                f"library {library.library_id} is enumerated ({library.n_entries} entries) but no "
                "input spectrum carries an InChIKey to check coverage against",
                "a labelled evaluation spectrum, or an orthogonal identification, giving the key "
                "whose presence in the library decides this",
            )
        missing = [
            s.spectrum_id
            for s in labelled
            if s.label is not None
            and s.label.inchikey is not None
            and not library.covers(s.label.inchikey)
        ]
        if missing:
            return self.violated(
                f"{len(missing)}/{len(labelled)} labelled spectra have no library entry "
                f"(e.g. {missing[0]}): the correct answer is outside the candidate set"
            )
        return self.satisfied(
            f"all {len(labelled)} labelled spectra are covered by library {library.library_id}"
        )

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Unreachable: :meth:`check` answers for the whole invocation."""
        raise NotImplementedError("RequiresLibraryCoverage answers per invocation")


@dataclass(frozen=True)
class RequiresNonNegativeIntensity(Assumption):
    """Intensity must be non-negative, as a physical count or absorbance is.

    Declared by every method whose model is non-negative - NNMF, MCR-ALS, anything solving for
    concentrations. It exists at *this* layer rather than as an exception inside the algorithm
    because of what it catches: a baseline correction that over-subtracted, or an SNV
    normalisation, both routinely push a real preprocessed spectrum slightly negative, and both are
    recorded in the processing chain. Raised from inside a decomposition, that is a crash halfway
    through a comparison; declared here, it is a named violation in the audit pointing at the
    pipeline step responsible.
    """

    tolerance: float = 0.0

    name: ClassVar[str] = "requires_non_negative_intensity"
    rationale: ClassVar[str] = (
        "A model of the form D = C S with C, S >= 0 cannot represent negative data, so the "
        "negative part is absorbed into the fitted factors and reported as chemistry."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F02_PREPROCESSING_ARTIFACT

    def describe(self) -> str:
        """Name and tolerance."""
        return f"{self.name}(tolerance={self.tolerance:g})"

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Find the most negative sample and compare it against the tolerance."""
        values = np.asarray(spectrum.intensity, dtype=np.float64)
        minimum = float(values.min())
        scale = float(np.abs(values).max()) or 1.0
        detail = (
            f"minimum intensity {minimum:.4g} ({minimum / scale:.2%} of the peak), "
            f"tolerance {-self.tolerance:g}"
        )
        if minimum >= -self.tolerance:
            return self.satisfied(detail)
        chain = [step.name for step in spectrum.processing]
        return self.violated(
            detail + f"; the recorded processing chain is {chain or '[]'}, which is where the "
            "negative part came from"
        )


@dataclass(frozen=True)
class RequiresCommonAxis(Assumption):
    """Every input spectrum must lie on the same axis, to within ``tolerance_cm1``.

    Every matrix method - PCA, PLS, MCR-ALS, NNMF, library matching - treats channel *k* of one
    spectrum as commensurable with channel *k* of another. On spectra from different instruments
    that is false before resampling, and the failure is silent: the arithmetic succeeds and the
    resolved components are mixtures of different spectral positions.
    """

    tolerance_cm1: float = 1e-6

    name: ClassVar[str] = "requires_common_axis"
    rationale: ClassVar[str] = (
        "A bilinear model indexes channels by position, so spectra on different axes are being "
        "added together at different wavenumbers without anything raising."
    )
    failure_code: ClassVar[FailureCode] = FailureCode.F01_UNIT_AXIS

    def describe(self) -> str:
        """Name and tolerance."""
        return f"{self.name}(tolerance_cm1={self.tolerance_cm1:g})"

    def check(self, inputs: MethodInputs) -> AssumptionCheck:
        """Whole-invocation: a common axis is a property of the set, not of one spectrum."""
        population = (*inputs.spectra, *inputs.calibration)
        reference = np.asarray(population[0].axis.values, dtype=np.float64)
        lengths = {int(np.asarray(s.axis.values).size) for s in population}
        if len(lengths) > 1:
            return self.violated(
                f"input spectra have {sorted(lengths)} channels respectively: they are not on a "
                "common axis, and resampling is a recorded preprocessing step, not something this "
                "method should do silently"
            )
        kinds = {str(s.axis.kind) for s in population}
        if len(kinds) > 1:
            return self.violated(
                f"input spectra mix axis kinds {sorted(kinds)}: a Raman shift and an absolute "
                "wavenumber share a unit and mean different things (F01)"
            )
        worst = 0.0
        offender = population[0].spectrum_id
        for spectrum in population[1:]:
            deviation = float(
                np.abs(np.asarray(spectrum.axis.values, dtype=np.float64) - reference).max()
            )
            if deviation > worst:
                worst, offender = deviation, spectrum.spectrum_id
        detail = (
            f"{len(population)} spectra on {lengths.pop()} channels, worst axis deviation "
            f"{worst:.3g} cm-1 ({offender}), tolerance {self.tolerance_cm1:g}"
        )
        if worst <= self.tolerance_cm1:
            return self.satisfied(detail)
        return self.violated(detail + "; resample onto a common grid first")

    def check_one(self, spectrum: Spectrum, inputs: MethodInputs) -> AssumptionCheck:
        """Unreachable: :meth:`check` answers for the whole invocation."""
        raise NotImplementedError("RequiresCommonAxis answers per invocation")


def check_all(
    assumptions: Sequence[Assumption], inputs: MethodInputs
) -> tuple[AssumptionCheck, ...]:
    """Check every assumption against ``inputs``, preserving declaration order."""
    return tuple(assumption.check(inputs) for assumption in assumptions)
