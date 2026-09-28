"""HYPOTHESIZED: which broadening mechanism dominates a band shape - and why it will not say yet.

The deliberately-hypothesised half of the pair that proves a method can be added as a single file,
and a real conjecture rather than a placeholder. Registered ``HYPOTHESIZED``, so it is computed and
logged on every run, scored only where an independent answer exists, and **never** reaches a report
conclusion. Ledger entry: `PROJECT_NOVELTY.md#N11`.

**Where the conjecture came from.** E2 broadened QM9S spectra with a Lorentzian intrinsic linewidth
until their median band width matched real ROD spectra - and matching the width made the comparison
*worse* on correlation length: 73 cm-1 simulated against 21.49 cm-1 real, at 23.0 and 23.14 cm-1
median band FWHM respectively. At equal width the real bands decorrelate about three times faster,
which reads as tails lighter than Lorentzian: inhomogeneous broadening dominating over lifetime
broadening. So the conjecture was that ``autocorrelation_length / median_band_fwhm`` is a cheap,
width-normalised, per-spectrum estimator of *which mechanism dominates*.

**What the first control did to it.** Synthetic band sets with identical widths and spacings, and
only the line shape changed, do separate in the conjectured direction - Gaussian below Lorentzian,
at every spacing tried - but by a factor of 1.17 to 1.41, nowhere near the 3.42 E2 measured. Two
things then went wrong for the conjecture. Varying only the *band spacing*, at fixed shape and
width, moves the same ratio by 1.34 within the resolved regime and 1.71 across the full range - as
much as the shape does. And the size of the shape effect itself *depends* on the spacing, falling
from 1.41 at 33 band widths apart to 1.17 at 6, so no single global threshold exists even in
principle. Both measurements are tests, not prose: ``tests/test_methods_builtin.py``.

**What that changes here.** Two things, and neither is a caveat in a comment.

The confound became a declared assumption -
:class:`~usis.methods.assumptions.RequiresBandSeparation` - so a spectrum whose bands overlap is
refused instead of scored. That is the assumption layer doing the job it exists for.

And the decision threshold became a **required argument with no default**. ``calibration=None``, the
form this method is registered in, computes the ratio, reports it with the band spacing beside it,
and abstains - naming the corpus that would calibrate it. A threshold with no measurement behind it
is not a threshold, and hard-coding the E2 anchors as one would have shipped exactly the
unsupported claim the control just weakened. This follows the precedent of ``split_kind`` and of
``PromotionCriteria.preregistration_ref``: the parameter that decides an outcome must be supplied
deliberately, by someone who can say where it came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from usis.core.errors import UsisError
from usis.core.provenance import RunContext
from usis.core.types import ClaimType, Evidence, Maturity, TaskKind
from usis.measurement.diagnostics import (
    autocorrelation_length,
    band_statistics,
    median_band_spacing,
)
from usis.methods.assumptions import (
    MaxResolutionFwhm,
    MinSnr,
    RequiresBandSeparation,
    RequiresUniformAxis,
)
from usis.methods.base import Citation, CostModel, MethodBase, MethodResult
from usis.methods.inputs import MethodInputs
from usis.methods.registry import register_method

__all__ = ["E2_CORPUS_RATIOS", "BroadeningOrigin", "RatioCalibration"]

#: The two E2 corpus medians the conjecture started from, kept as documentation and deliberately
#: **not** wired up as a calibration. Real ROD 21.494872296601443 / 23.141609037306605 = 0.929;
#: QM9S under fitted Lorentzian broadening 73.0 / 23.0 = 3.174. Source:
#: experiments/E2-intrinsic-linewidth/comparison.json. They are corpus medians over two datasets,
#: one simulated by us, on spectra of uncontrolled band spacing - which the control below showed is
#: the wrong quantity to hold fixed.
E2_CORPUS_RATIOS: dict[str, float] = {
    "real_rod": 21.494872296601443 / 23.141609037306605,
    "lorentzian_simulation": 73.0 / 23.0,
}


class CalibrationError(UsisError):
    """A ratio calibration cannot separate the classes it claims to separate."""


@dataclass(frozen=True)
class RatioCalibration:
    """Anchor ratios for the two mechanisms, and where they were measured.

    ``measured_on`` is required. An anchor without a stated provenance is a number someone once
    saw, and the whole reason this method carries no default calibration is that the first
    plausible-looking pair of anchors turned out to be measured on the wrong thing.

    ``abstain_factor`` sets the multiplicative half-width of the undecided band around the
    geometric midpoint. Construction fails when the band would swallow both anchors, because a
    classifier whose decision band contains its own reference points does not classify - it just
    looks like it does.
    """

    light_tail_anchor: float
    heavy_tail_anchor: float
    measured_on: str
    abstain_factor: float = 1.1

    def __post_init__(self) -> None:
        """Refuse a calibration that is unattributed, misordered, or cannot separate."""
        if not self.measured_on.strip():
            raise CalibrationError(
                "RatioCalibration requires measured_on: the dataset, experiment or test that "
                "produced these anchors. Anchors without provenance are how an unsupported "
                "threshold becomes a default."
            )
        if not (0.0 < self.light_tail_anchor < self.heavy_tail_anchor):
            raise CalibrationError(
                f"anchors must be positive and ordered light < heavy, got "
                f"{self.light_tail_anchor} and {self.heavy_tail_anchor}. Lighter tails "
                "decorrelate faster, so the inhomogeneous anchor is the smaller ratio."
            )
        if self.abstain_factor < 1.0:
            raise CalibrationError("abstain_factor must be at least 1.0")
        separation = self.heavy_tail_anchor / self.light_tail_anchor
        if separation <= self.abstain_factor**2:
            raise CalibrationError(
                f"anchors separated by a factor of {separation:.3f} cannot support an abstention "
                f"band of +/-{self.abstain_factor:.3f}: the band [{self.lower:.3f}, "
                f"{self.upper:.3f}] contains both anchors, so every spectrum would be undecided "
                "and the classification would be decorative. Widen the anchors or narrow the band "
                "- with a measurement, not a preference."
            )

    @property
    def midpoint(self) -> float:
        """The log-scale midpoint between the anchors."""
        return math.sqrt(self.light_tail_anchor * self.heavy_tail_anchor)

    @property
    def lower(self) -> float:
        """Below this ratio the method calls the light-tailed (inhomogeneous) mechanism."""
        return self.midpoint / self.abstain_factor

    @property
    def upper(self) -> float:
        """Above this ratio the method calls the heavy-tailed (lifetime) mechanism."""
        return self.midpoint * self.abstain_factor

    def describe(self) -> str:
        """A one-line rendering carried into every result."""
        return (
            f"anchors {self.light_tail_anchor:.3f} / {self.heavy_tail_anchor:.3f}, undecided band "
            f"[{self.lower:.3f}, {self.upper:.3f}], measured on {self.measured_on}"
        )


class BroadeningOrigin(MethodBase):
    """Reports a width-normalised correlation length, and classifies only if calibrated."""

    method_id = "shape.broadening_origin"
    version = "0.2.0"
    task = TaskKind.PHYSICS_CONSISTENCY
    maturity = Maturity.HYPOTHESIZED
    provenance = Citation(
        title="Width-normalised correlation length as a broadening-mechanism estimator",
        internal_ref="PROJECT_NOVELTY.md#N11",
        note=(
            "Conjecture arising from experiments/E2-intrinsic-linewidth; weakened by its own "
            "first control, which found the statistic confounded with band spacing."
        ),
    )
    #: ``MinSnr`` is hard because both statistics are computed from detected bands, and noise
    #: maxima are detected as bands. ``RequiresUniformAxis`` is hard because
    #: ``autocorrelation_length`` converts lag to cm-1 with the *mean* axis step, and M1 measured a
    #: real ROD record whose step varies by 25.5%. ``MaxResolutionFwhm`` is hard because once the
    #: instrument response is comparable to the band width the ratio describes the spectrometer.
    #: ``RequiresBandSeparation`` is hard because the control measured that confound directly.
    assumptions = (
        MinSnr(5.0),
        RequiresUniformAxis(0.01),
        MaxResolutionFwhm(10.0),
        RequiresBandSeparation(6.0),
    )
    cost = CostModel(
        compute_seconds_per_spectrum=2e-3,
        note="one peak detection plus one autocorrelation over the detrended spectrum",
    )

    def __init__(self, calibration: RatioCalibration | None = None) -> None:
        self.calibration = calibration

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Compute the ratio, report the confounder beside it, and classify only if calibrated."""
        spectrum = inputs.primary
        n_bands, median_fwhm, fwhm_iqr = band_statistics(spectrum)
        correlation_length = autocorrelation_length(spectrum)
        spacing = median_band_spacing(spectrum)
        values = {
            "n_bands": float(n_bands),
            "median_band_fwhm_cm1": median_fwhm,
            "band_fwhm_iqr_cm1": fwhm_iqr,
            "autocorrelation_length_cm1": correlation_length,
            "median_band_spacing_cm1": spacing,
        }
        if n_bands == 0 or not math.isfinite(median_fwhm) or median_fwhm <= 0.0:
            return self._abstain(
                spectrum.spectrum_id,
                values,
                "no band with a measurable width was detected, so the ratio has no denominator",
            )
        if not math.isfinite(correlation_length):
            return self._abstain(
                spectrum.spectrum_id,
                values,
                "the autocorrelation never falls below 1/e, so no correlation length exists on "
                "this axis span",
            )
        ratio = correlation_length / median_fwhm
        values["width_normalised_correlation_length"] = ratio
        if math.isfinite(spacing):
            values["spacing_to_fwhm"] = spacing / median_fwhm
        if self.calibration is None:
            return self._abstain(
                spectrum.spectrum_id,
                values,
                f"ratio {ratio:.3f} computed, but this method carries no calibration. Its first "
                "control found the statistic as sensitive to band spacing (1.34x within the "
                "resolved regime, 1.71x across the full range) as to the line shape it was meant "
                "to read (1.17-1.41x, itself spacing-dependent), so the E2 corpus anchors "
                f"({E2_CORPUS_RATIOS['real_rod']:.3f} real, "
                f"{E2_CORPUS_RATIOS['lorentzian_simulation']:.3f} simulated) are not usable as a "
                "threshold. Supply RatioCalibration measured on spectra of known dominant "
                "mechanism at controlled band spacing - a variable-temperature series, where "
                "lifetime broadening scales with temperature and static disorder does not.",
            )
        calibration = self.calibration
        values["log_distance_to_midpoint"] = math.log(ratio / calibration.midpoint)
        anchors = (
            f"ratio {ratio:.3f} (correlation length {correlation_length:.2f} cm-1 / median band "
            f"FWHM {median_fwhm:.2f} cm-1, median band spacing {spacing:.1f} cm-1); "
            f"{calibration.describe()}"
        )
        if ratio < calibration.lower:
            label, statement = (
                "inhomogeneous_dominated",
                "Bands decorrelate faster than their width alone implies, i.e. lighter tails than "
                "a Lorentzian. Consistent with static disorder, site inhomogeneity or packing "
                f"variation dominating over excited-state lifetime. {anchors}",
            )
        elif ratio > calibration.upper:
            label, statement = (
                "lifetime_dominated",
                "Bands decorrelate slowly for their width, i.e. heavy tails. Consistent with "
                f"lifetime (Lorentzian) broadening dominating. {anchors}",
            )
        else:
            return self._abstain(
                spectrum.spectrum_id,
                values,
                f"the ratio falls inside the calibrated undecided band. {anchors}",
            )
        return self._result(
            values=values,
            labels={"dominant_mechanism": label, "spectrum_id": spectrum.spectrum_id},
            evidence=(
                Evidence(
                    kind="band_shape",
                    statement=statement,
                    claim_type=ClaimType.MECHANISTIC_HYPOTHESIS,
                    source_ref="experiments/E2-intrinsic-linewidth/result_card.md",
                    conditions=(
                        "the statistic is confounded with band spacing; the reported "
                        "median_band_spacing_cm1 must be read alongside the ratio, and the "
                        f"calibration is only as good as {calibration.measured_on}"
                    ),
                ),
            ),
            notes="HYPOTHESIZED: shadow output, not a conclusion.",
        )

    def _abstain(self, spectrum_id: str, values: dict[str, float], why: str) -> MethodResult:
        """Return the undecided answer, with the reason and the statistics that led to it."""
        return self._result(
            values=values,
            labels={"dominant_mechanism": "undecided", "spectrum_id": spectrum_id},
            evidence=(
                Evidence(
                    kind="band_shape",
                    statement=f"No mechanism claimed: {why}",
                    claim_type=ClaimType.MECHANISTIC_HYPOTHESIS,
                    source_ref="experiments/E2-intrinsic-linewidth/result_card.md",
                ),
            ),
            notes="HYPOTHESIZED: shadow output, not a conclusion.",
        )


register_method(
    BroadeningOrigin(),
    tags=("shape", "physics", "shadow"),
    notes=(
        "Runs in shadow mode with no calibration, so it reports the statistic and abstains. "
        "shadow_of is deliberately empty: no ESTABLISHED method answers PHYSICS_CONSISTENCY yet, "
        "so there is no incumbent to measure a margin against and the promotion gate cannot be "
        "met on any amount of data - which is the correct state, not a gap to route around. "
        "Promotion requires the variable-temperature corpus named in PROJECT_NOVELTY.md#N11, and "
        "a spacing-controlled repeat of the control in tests/test_methods_builtin.py."
    ),
)
