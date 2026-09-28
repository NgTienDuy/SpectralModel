"""Intrinsic linewidth: the width a band has before any instrument sees it.

This module exists because experiment E1 measured its absence. Real ROD bands are 16.5 cm^-1 wide
[12.1, 23.1] on an instrument that declares 4 cm^-1, so roughly **12.5 cm^-1 of that width is the
sample's**, not the spectrometer's. A harmonic DFT calculation produces zero-width sticks, and the
publisher of a simulated corpus convolves them with one constant kernel. Two consequences, and the
second is the damaging one:

* matching the instrument resolution does **not** match the bands, because the residual is physics
  rather than instrumentation;
* the simulated widths have **no spread at all** - QM9S band FWHM has interquartile range
  ``[11.00, 11.00]``, every band of every molecule identical. That is a perfect shortcut feature
  (``F14``): band width alone separates simulated from real without any chemistry, so a domain
  adaptation that ignores linewidth is adapting the wrong variable.

So the operator here does two things a plain convolution cannot. It is applied **before** the
instrument response in the canonical order, because the sample is broad before the light is
dispersed. And its width is **drawn per band from a distribution**, because mode-to-mode variation
is the property that is missing, not the mean.

**What is modelled and what is not.** The physical sources of Raman linewidth are excited-state
lifetime (anharmonic decay, Lorentzian), inhomogeneous broadening from site disorder and strain
(closer to Gaussian), and thermal population effects. Which one dominates is a per-material
question. This module lets you choose the shape and the spread; it does **not** model a systematic
dependence of width on wavenumber, on mode symmetry, or on temperature, because we have no measured
value for any of those and inventing one would be a fabricated physics constant. Widths are drawn
independently per band from a lognormal, and that independence is itself an assumption - a stated
one, recorded in the processing chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np

from usis.core.types import AxisSpec
from usis.measurement.diagnostics import detect_peaks, observed_band_widths
from usis.measurement.forward_model import (
    Degradation,
    DegradationError,
    positive,
    require_uniform,
)
from usis.measurement.response import composed_fwhm, gaussian_kernel, lorentzian_kernel

if TYPE_CHECKING:
    from numpy.random import Generator

    from usis.core.types import AxisKind, FloatArray, Spectrum

__all__ = ["IntrinsicBroadening", "deconvolved_fwhm", "measured_intrinsic_widths"]

#: Shapes a band profile may take here. Lorentzian is lifetime broadening; Gaussian is the
#: inhomogeneous limit. Anything else is a Voigt, which is what composing the two already gives.
_SHAPES = ("lorentzian", "gaussian")


def deconvolved_fwhm(
    observed_fwhm: float,
    known_fwhm: float,
    observed_shape: str = "lorentzian",
    known_shape: str = "gaussian",
) -> float:
    """The width of the *unknown* component of an observed band, given the known one.

    The inverse of :func:`~usis.measurement.response.composed_fwhm`: given a measured band width
    and the instrument response that contributed to it, return the width the sample itself carried.
    That is the number this module's ``median_fwhm_cm1`` should be set from, and computing it by
    subtraction would be wrong for every shape combination except two pure Lorentzians.

    Solved by bisection rather than algebra. Composition is strictly increasing in each argument
    for every shape pair, so bisection is exact to tolerance and - unlike a closed form - does not
    have to be re-derived when a shape is added.

    Refuses when the observed width is not wider than the known component: that is a request to
    deconvolve past zero, and the honest answers are "the instrument resolution is misreported" or
    "the band is not resolved", neither of which a number would convey.
    """
    positive(observed_fwhm, "observed_fwhm")
    positive(known_fwhm, "known_fwhm")
    for name, value in (("observed_shape", observed_shape), ("known_shape", known_shape)):
        if value not in _SHAPES:
            raise DegradationError(f"{name} must be one of {_SHAPES}, got {value!r}")
    if observed_fwhm <= known_fwhm:
        raise DegradationError(
            f"observed width {observed_fwhm:g} cm^-1 is not wider than the known component "
            f"{known_fwhm:g} cm^-1, so there is no width left to attribute to the sample. Either "
            "the declared resolution is wrong or the band is unresolved; both need a decision, "
            "not a number."
        )
    low, high = 1e-9, observed_fwhm
    for _ in range(200):
        middle = 0.5 * (low + high)
        if composed_fwhm(middle, known_fwhm, observed_shape, known_shape) < observed_fwhm:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass(frozen=True)
class IntrinsicBroadening(Degradation):
    """Broaden each band by its own width, drawn from a lognormal distribution.

    Args:
        median_fwhm_cm1: the median of the drawn widths. Set it from real data with
            :func:`deconvolved_fwhm`, not from a guess.
        geometric_sd: the multiplicative spread. ``1.0`` means every band gets exactly the median
            width, which reproduces the degenerate behaviour E1 found in the published simulation
            and is available **only** so that it can be run as an explicit control arm. ``1.5``
            means the middle two-thirds of bands span roughly ``median / 1.5`` to
            ``median * 1.5``.
        shape: ``"lorentzian"`` (lifetime broadening) or ``"gaussian"`` (inhomogeneous).

    **The width varies across the spectrum, so this is not a convolution.** A convolution has one
    kernel; here every input sample is spread with the width of the band it belongs to, and the
    contributions are summed. Band membership is by nearest detected peak, which is crude between
    bands and does not matter there: the intensity being spread is the floor, and the floor is
    flat. Each kernel is normalised, so total intensity is conserved except for the tails that fall
    off the ends of the measured window - which is what a real spectrometer does too.

    Because the tails leave at the edges, this must run **before** any baseline is added, or a
    constant offset would be given a roll-off it never had. The canonical order in
    :class:`~usis.measurement.forward_model.ForwardModel` enforces exactly that.

    **What the recorded resolution means afterwards.** ``AxisSpec.resolution_fwhm`` is one number,
    and after this step the narrowest surviving feature width is a *distribution*. The axis is
    updated with the composition of the input width and the distribution's **median**, which is the
    honest single-number summary; the parameters and the seed are in the processing chain, so the
    full set of drawn widths is recoverable by replay rather than by being written down.
    """

    median_fwhm_cm1: float
    geometric_sd: float
    shape: str = "lorentzian"
    prominence_fraction: float = 0.05

    def __post_init__(self) -> None:
        """Validate the distribution and the shape."""
        positive(self.median_fwhm_cm1, "median_fwhm_cm1")
        if self.geometric_sd < 1.0 or not np.isfinite(self.geometric_sd):
            raise DegradationError(
                f"geometric_sd must be finite and >= 1.0 (1.0 meaning no spread at all), got "
                f"{self.geometric_sd}"
            )
        if self.shape not in _SHAPES:
            raise DegradationError(f"shape must be one of {_SHAPES}, got {self.shape!r}")
        positive(self.prominence_fraction, "prominence_fraction")

    def draw_widths(self, n_bands: int, rng: Generator) -> FloatArray:
        """Draw one FWHM per band: lognormal about ``median_fwhm_cm1``.

        Lognormal because a linewidth is strictly positive and its variation is multiplicative -
        a mode is "half as broad again", never "10 cm^-1 narrower than zero". The median of a
        lognormal is ``exp(mu)``, so ``mu = ln(median)`` sets it exactly, with no bias correction
        needed.
        """
        if n_bands < 1:
            raise DegradationError("draw_widths needs at least one band")
        sigma = math.log(self.geometric_sd)
        if sigma == 0.0:
            return np.full(n_bands, float(self.median_fwhm_cm1), dtype=np.float64)
        drawn: FloatArray = self.median_fwhm_cm1 * np.exp(rng.normal(0.0, sigma, n_bands))
        return drawn

    def resulting_fwhm(self, current_fwhm: float | None) -> float | None:
        """The output's single-number width: the input composed with the *median* drawn width."""
        if current_fwhm is None:
            return None
        return composed_fwhm(current_fwhm, self.median_fwhm_cm1, self.shape, self.shape)

    def _kernel(self, fwhm: float, step: float) -> FloatArray:
        return (
            lorentzian_kernel(fwhm, step)
            if self.shape == "lorentzian"
            else gaussian_kernel(fwhm, step)
        )

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        step = require_uniform(axis, "intrinsic broadening")
        # Work in an ascending frame: nearest-band assignment is a searchsorted, which needs one.
        # Everything is flipped back at the end, so a descending axis is unchanged as an axis.
        values = np.asarray(axis.values, dtype=np.float64)
        signal = np.asarray(intensity, dtype=np.float64)
        if not axis.ascending:
            values, signal = values[::-1], signal[::-1]

        centres = self._band_centres(axis.kind, values, signal)
        widths = self.draw_widths(centres.size, rng)
        owner = self._nearest_band(values, centres)

        out = np.zeros_like(signal)
        for band, width in enumerate(widths):
            segment = np.where(owner == band, signal, 0.0)
            if not segment.any():
                continue
            kernel = self._kernel(float(width), step)
            if kernel.size > signal.size:
                raise DegradationError(
                    f"a {self.shape} kernel of {width:.4g} cm^-1 spans {kernel.size} points but "
                    f"the spectrum has only {signal.size}: the drawn width is wider than the "
                    "measured range. Lower median_fwhm_cm1 or geometric_sd, or use a wider window."
                )
            pad = kernel.size // 2
            # Zero padding, unlike the instrument convolution which reflects: what leaves the ends
            # here is band intensity scattered outside the measured window, and it is genuinely
            # gone. Reflecting it back would manufacture signal at the edges.
            padded = np.pad(segment, pad, mode="constant")
            out += np.convolve(padded, kernel, mode="valid")

        if not axis.ascending:
            out = out[::-1]
        return replace(axis, resolution_fwhm=self.resulting_fwhm(axis.resolution_fwhm)), out

    def _band_centres(self, kind: AxisKind, values: FloatArray, signal: FloatArray) -> FloatArray:
        """Axis positions of the detected bands, or a refusal explaining why there are none."""
        # detect_peaks takes a Spectrum, but the chain hands us bare arrays; a light shim keeps one
        # peak-detection implementation rather than a second one that could drift from it.
        probe = _PeakProbe(AxisSpec.from_values(kind, values), signal)
        indices, _ = detect_peaks(cast("Spectrum", probe), self.prominence_fraction)
        if indices.size == 0:
            raise DegradationError(
                "no bands were detected, so there is nothing whose linewidth this could be. "
                "Intrinsic broadening is a property of the vibrational modes, not of the window: "
                "broadening a featureless spectrum would invent structure rather than model it."
            )
        centres: FloatArray = np.sort(values[indices])
        return centres

    @staticmethod
    def _nearest_band(values: FloatArray, centres: FloatArray) -> FloatArray:
        """For each sample, the index of the nearest band centre (both ascending)."""
        right = np.searchsorted(centres, values)
        left = np.clip(right - 1, 0, centres.size - 1)
        right = np.clip(right, 0, centres.size - 1)
        take_right = np.abs(centres[right] - values) < np.abs(values - centres[left])
        owner: FloatArray = np.where(take_right, right, left)
        return owner


@dataclass(frozen=True)
class _PeakProbe:
    """The two attributes :func:`detect_peaks` reads, so it can run on a bare axis and array.

    Not a :class:`~usis.core.types.Spectrum`: constructing one needs a source record and a licence,
    and inventing those to run a peak finder would put a fabricated provenance into the repo.
    """

    axis: AxisSpec
    intensity: FloatArray


def measured_intrinsic_widths(
    spectrum: Spectrum, instrument_fwhm: float, instrument_shape: str = "gaussian"
) -> FloatArray:
    """Per-band intrinsic widths implied by one measured spectrum.

    Every detected band's observed FWHM, deconvolved against the stated instrument response. Bands
    narrower than the instrument are dropped rather than clamped to zero: a band that measures
    narrower than the response function is evidence about the *metadata*, not about the sample, and
    averaging a clamped zero into the distribution would drag the estimate toward the instrument.
    The count of dropped bands is what a caller should report, so it is recoverable as
    ``n_detected - result.size``.
    """
    widths = observed_band_widths(spectrum)
    kept = [
        deconvolved_fwhm(float(w), instrument_fwhm, "lorentzian", instrument_shape)
        for w in widths
        if w > instrument_fwhm
    ]
    return np.asarray(kept, dtype=np.float64)
