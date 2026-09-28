"""Smoothing and derivatives, and the axis assumption both of them hide.

Every kernel here is specified in **samples**, not in cm-1, and that is deliberate: on a real
dispersive axis the two are not interchangeable. M1 measured a ROD record whose wavenumber step
varies by 25.5% across the range, so a window of "9 points" spans 25% more cm-1 at one end than the
other. :func:`window_cm1` reports what a window actually spans so the choice can be made in the
units a spectroscopist thinks in, and :class:`~usis.preprocess.pipeline.Pipeline` can be asked to
require a uniform axis before either of these runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from usis.core.types import AxisSpec, FloatArray
from usis.preprocess.base import PreprocessError, PreprocessStep, register_step

__all__ = ["MovingAverage", "SavitzkyGolay", "window_cm1"]


def window_cm1(axis: AxisSpec, window: int) -> tuple[float, float]:
    """The narrowest and widest cm-1 span a ``window``-sample window covers on this axis.

    Equal on a uniform axis. On a dispersive one they differ, and the difference is the amount by
    which a single stated kernel width is wrong somewhere on the spectrum.
    """
    steps = np.abs(np.diff(np.asarray(axis.values, dtype=np.float64)))
    if steps.size == 0:
        return (float("nan"), float("nan"))
    spans = np.convolve(steps, np.ones(min(window - 1, steps.size)), mode="valid")
    return (float(spans.min()), float(spans.max()))


@register_step
@dataclass(frozen=True)
class SavitzkyGolay(PreprocessStep):
    """Local polynomial fit over a sliding window; optionally its ``derivative``-th derivative.

    Implemented directly from the Vandermonde least-squares solution rather than through scipy, so
    the repository keeps its "tests never need the network and the core needs no optional
    dependency" property. The coefficients are the exact ones scipy would produce: the smoothing
    filter is row ``derivative`` of the pseudo-inverse of the local design matrix, scaled by the
    factorial and the step.

    A derivative is a *baseline-insensitive* representation - a first derivative removes an additive
    constant, a second removes a linear ramp - which is why chemometrics reaches for it instead of
    fighting the baseline. The cost is noise amplification, and it is not small: differentiating
    multiplies high-frequency content by frequency, so a second derivative on a noisy spectrum can
    be dominated by noise structure that looks exactly like narrow bands (``F02``).

    ``derivative`` is in units of intensity per (axis unit)^``derivative``, using the mean axis
    step - so on a non-uniform axis the scale is wrong by the step variation, which is what
    :func:`window_cm1` is for.
    """

    window: int = 9
    order: int = 3
    derivative: int = 0

    def __post_init__(self) -> None:
        """Validate the window, order and derivative order."""
        if self.window < 3 or self.window % 2 == 0:
            raise PreprocessError(f"window must be odd and at least 3, got {self.window}")
        if not 0 <= self.order < self.window:
            raise PreprocessError(
                f"order must satisfy 0 <= order < window, got {self.order} with "
                f"window {self.window}"
            )
        if not 0 <= self.derivative <= self.order:
            raise PreprocessError(
                f"derivative must satisfy 0 <= derivative <= order, got {self.derivative} with "
                f"order {self.order}: a polynomial of order {self.order} has no non-zero "
                f"{self.derivative}-th derivative"
            )

    def coefficients(self, step: float) -> FloatArray:
        """The convolution kernel for this window, order, derivative and axis step."""
        half = self.window // 2
        offsets = np.arange(-half, half + 1, dtype=np.float64)
        design = np.vander(offsets, self.order + 1, increasing=True)
        pseudo_inverse = np.linalg.pinv(design)
        scale = float(math.factorial(self.derivative)) / (step**self.derivative)
        return np.asarray(pseudo_inverse[self.derivative] * scale, dtype=np.float64)

    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Convolve with the local-polynomial kernel, holding the edges by reflection."""
        if intensity.size < self.window:
            raise PreprocessError(
                f"a {self.window}-sample window does not fit a {intensity.size}-sample spectrum"
            )
        steps = np.abs(np.diff(np.asarray(axis.values, dtype=np.float64)))
        step = float(steps.mean()) if steps.size else 1.0
        kernel = self.coefficients(step)
        half = self.window // 2
        # Reflect rather than pad with a constant: a constant edge invents a plateau, and a
        # derivative of that plateau is a band at the edge of every spectrum.
        padded = np.pad(intensity, half, mode="reflect")
        # Correlation, not convolution: the kernel is indexed by signed offset from the centre.
        smoothed = np.convolve(padded, kernel[::-1], mode="valid")
        return axis, np.asarray(smoothed, dtype=np.float64)


@register_step
@dataclass(frozen=True)
class MovingAverage(PreprocessStep):
    """The boxcar, kept as the honest lower bound on smoothing.

    It is here to be beaten. A boxcar is a Savitzky-Golay filter of order zero, so any comparison
    in which a higher-order local fit does *not* beat it is telling you the differences between
    smoothers do not matter for that task - which is a finding worth having before tuning one.
    """

    window: int = 5

    def __post_init__(self) -> None:
        """Validate the window."""
        if self.window < 3 or self.window % 2 == 0:
            raise PreprocessError(f"window must be odd and at least 3, got {self.window}")

    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Convolve with a normalised boxcar, holding the edges by reflection."""
        if intensity.size < self.window:
            raise PreprocessError(
                f"a {self.window}-sample window does not fit a {intensity.size}-sample spectrum"
            )
        half = self.window // 2
        padded = np.pad(intensity, half, mode="reflect")
        kernel = np.full(self.window, 1.0 / self.window)
        return axis, np.asarray(np.convolve(padded, kernel, mode="valid"), dtype=np.float64)
