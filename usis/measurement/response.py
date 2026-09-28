"""Instrument response: resolution convolution and wavelength-dependent efficiency.

The two are separated because they are separable failures. Resolution loss destroys information
irreversibly; a response curve distorts intensities but is invertible given a calibration standard.
A model that lumps them together cannot tell you which one your method is failing on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from usis.measurement.forward_model import (
    Degradation,
    DegradationError,
    as_severity,
    normalised_positions,
    positive,
    require_uniform,
)

if TYPE_CHECKING:
    from numpy.random import Generator

    from usis.core.types import AxisSpec, FloatArray

__all__ = [
    "IntensityResponse",
    "ResolutionConvolution",
    "composed_fwhm",
    "gaussian_kernel",
    "lorentzian_kernel",
]

#: FWHM = 2*sqrt(2*ln 2) * sigma for a Gaussian.
_FWHM_PER_SIGMA: float = 2.0 * math.sqrt(2.0 * math.log(2.0))


def gaussian_kernel(fwhm: float, step: float) -> FloatArray:
    """A normalised Gaussian kernel sampled on a grid of spacing ``step``."""
    sigma = fwhm / _FWHM_PER_SIGMA
    half = max(1, math.ceil(4.0 * sigma / step))
    offsets = np.arange(-half, half + 1, dtype=np.float64) * step
    kernel: FloatArray = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def lorentzian_kernel(fwhm: float, step: float) -> FloatArray:
    """A normalised Lorentzian kernel.

    Truncated at 20 half-widths rather than the 4 sigma a Gaussian needs: Lorentzian tails decay as
    ``1/x^2``, so a short kernel would quietly remove the very tails that distinguish the shape.
    """
    gamma = fwhm / 2.0
    half = max(1, math.ceil(20.0 * gamma / step))
    offsets = np.arange(-half, half + 1, dtype=np.float64) * step
    kernel = (gamma**2) / (offsets**2 + gamma**2)
    return kernel / kernel.sum()


def composed_fwhm(
    first_fwhm: float, second_fwhm: float, first_shape: str, second_shape: str
) -> float:
    """FWHM of the convolution of two profiles.

    Like shapes have exact rules: Gaussians add in quadrature, Lorentzians add linearly. **Mixed
    shapes give a Voigt profile, whose width has no closed form**, so the Olivero-Longbothum
    approximation is used::

        f_V ~= 0.5346 * f_L + sqrt(0.2166 * f_L^2 + f_G^2)

    That distinction is not pedantry. Applying the quadrature rule to a 10 cm^-1 Lorentzian
    convolved with a 20 cm^-1 Gaussian predicts 22.4 cm^-1; the width measured back off the result
    is 26.0. The approximation above predicts 25.9. A 13% error in a stated resolution is enough to
    invalidate a resolution-matching experiment.
    """
    positive(first_fwhm, "first_fwhm")
    positive(second_fwhm, "second_fwhm")
    if first_shape == second_shape == "gaussian":
        return math.sqrt(first_fwhm**2 + second_fwhm**2)
    if first_shape == second_shape == "lorentzian":
        return first_fwhm + second_fwhm
    lorentz = first_fwhm if first_shape == "lorentzian" else second_fwhm
    gauss = second_fwhm if first_shape == "lorentzian" else first_fwhm
    return 0.5346 * lorentz + math.sqrt(0.2166 * lorentz**2 + gauss**2)


@dataclass(frozen=True)
class ResolutionConvolution(Degradation):
    """Convolve with an instrument response function of a given FWHM.

    ``kernel_fwhm_cm1`` is the width of the *added* broadening, not the target width, because that
    is the only form that is unambiguous when the input's own resolution is unknown. Use
    :meth:`to_target` when you know both and want the model to compute the kernel.

    The result's ``resolution_fwhm`` is composed by :func:`composed_fwhm`, which needs to know the
    *input's* lineshape as well as the kernel's. ``AxisSpec`` records a width but not a shape, so
    ``input_shape`` states it; leaving it ``None`` means "assume the same shape as the kernel",
    which is exact for the like-shape case and is recorded in the provenance as an assumption.
    """

    kernel_fwhm_cm1: float
    shape: str = "gaussian"
    input_shape: str | None = None

    def __post_init__(self) -> None:
        """Validate the width and the shape names."""
        positive(self.kernel_fwhm_cm1, "kernel_fwhm_cm1")
        for name, value in (("shape", self.shape), ("input_shape", self.input_shape)):
            if value is not None and value not in {"gaussian", "lorentzian"}:
                raise DegradationError(f"{name} must be 'gaussian' or 'lorentzian', got {value!r}")

    @classmethod
    def to_target(
        cls,
        current_fwhm: float,
        target_fwhm: float,
        shape: str = "gaussian",
        input_shape: str | None = None,
    ) -> ResolutionConvolution:
        """Build the convolution that takes ``current_fwhm`` to ``target_fwhm``.

        Refuses to sharpen: convolution only ever broadens, and asking for a narrower target is a
        request for deconvolution, which is a different and much harder operation.
        """
        positive(current_fwhm, "current_fwhm")
        positive(target_fwhm, "target_fwhm")
        if target_fwhm <= current_fwhm:
            raise DegradationError(
                f"convolution cannot sharpen: target {target_fwhm} is not wider than the current "
                f"{current_fwhm}. Sharpening is deconvolution, not a forward-model step."
            )
        if (input_shape or shape) != shape:
            raise DegradationError(
                "to_target only inverts the like-shape composition rules; for a mixed Voigt "
                "composition, choose the kernel width yourself and check it with composed_fwhm()"
            )
        if shape == "gaussian":
            kernel = math.sqrt(target_fwhm**2 - current_fwhm**2)
        else:
            kernel = target_fwhm - current_fwhm
        return cls(kernel_fwhm_cm1=kernel, shape=shape, input_shape=input_shape)

    def resulting_fwhm(self, current_fwhm: float | None) -> float | None:
        """The output resolution, or ``None`` when the input's own resolution is unknown."""
        if current_fwhm is None:
            return None
        return composed_fwhm(
            current_fwhm, self.kernel_fwhm_cm1, self.input_shape or self.shape, self.shape
        )

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        step = require_uniform(axis, "resolution convolution")
        kernel = (
            gaussian_kernel(self.kernel_fwhm_cm1, step)
            if self.shape == "gaussian"
            else lorentzian_kernel(self.kernel_fwhm_cm1, step)
        )
        if kernel.size > intensity.size:
            raise DegradationError(
                f"the {self.shape} kernel spans {kernel.size} points but the spectrum has only "
                f"{intensity.size}: the broadening is wider than the measured range"
            )
        # 'reflect' at the edges: zero-padding would pull the first and last points toward zero and
        # manufacture a baseline roll-off that looks like a real instrument artifact.
        pad = kernel.size // 2
        padded = np.pad(intensity, pad, mode="reflect")
        convolved = np.convolve(padded, kernel, mode="valid")
        return replace(axis, resolution_fwhm=self.resulting_fwhm(axis.resolution_fwhm)), convolved

    def params(self) -> dict[str, object]:
        """Record the assumed input shape explicitly, since it changes the reported resolution."""
        return {**super().params(), "assumed_input_shape": self.input_shape or self.shape}


@dataclass(frozen=True)
class IntensityResponse(Degradation):
    """A smooth multiplicative efficiency curve across the spectral range.

    Real optics and detectors are more efficient in the middle of their range than at the edges.
    The curve is a broad Gaussian hump in normalised axis position, so it is smooth, strictly
    positive, and has no preferred wavenumber built in — a fixed cm^-1 centre would be an
    instrument fingerprint we invented.

    ``amplitude`` is the fractional depth of the roll-off at the edges: 0.0 is a flat response, 0.5
    means the edges transmit half of what the centre does.
    """

    amplitude: float = 0.3
    centre: float = 0.5
    width: float = 0.4

    def __post_init__(self) -> None:
        """Validate the shape parameters."""
        as_severity(self.amplitude, "amplitude")
        as_severity(self.centre, "centre")
        positive(self.width, "width")

    def curve(self, axis: AxisSpec) -> FloatArray:
        """The multiplicative response, normalised to 1.0 at the centre."""
        positions = normalised_positions(axis)
        hump = np.exp(-0.5 * ((positions - self.centre) / self.width) ** 2)
        return (1.0 - self.amplitude) + self.amplitude * hump

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        return axis, intensity * self.curve(axis)
