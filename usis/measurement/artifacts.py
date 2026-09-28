"""Backgrounds and detector artifacts: baseline drift, fluorescence, cosmic rays, saturation.

These are the effects that make a spectrum *look* like chemistry when it is not. Each corresponds
to a failure code the evaluation layer counts: ``F02`` for a mishandled baseline, ``F04`` for
fluorescence dominance, ``F03`` for spikes, ``F05`` for saturation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from usis.measurement.forward_model import (
    Degradation,
    DegradationError,
    ScaleMode,
    as_severity,
    non_negative,
    normalised_positions,
    positive,
    signal_scale,
)

if TYPE_CHECKING:
    from numpy.random import Generator

    from usis.core.types import AxisSpec, FloatArray

__all__ = ["BaselineDrift", "CosmicRaySpikes", "FluorescenceBackground", "Saturation"]


@dataclass(frozen=True)
class BaselineDrift(Degradation):
    """A low-order polynomial added across the spectrum.

    Coefficients are in units of the signal magnitude named by ``scale_mode`` and act on the
    normalised axis position, so ``(0.1, -0.2, 0.15)`` means the same shape whatever the spectral
    range. **The scale mode is a required argument.** E1 measured what choosing it implicitly
    costs: expressed against the median, a baseline on a simulated spectrum is a ratio taken
    against the near-zero floor between bands, and "1.14 x median" adds almost nothing.
    """

    coefficients: tuple[float, ...] = (0.05, 0.1)
    scale_mode: ScaleMode = field(kw_only=True)

    def __post_init__(self) -> None:
        """Validate the polynomial."""
        if not self.coefficients:
            raise DegradationError("BaselineDrift needs at least one coefficient")
        if len(self.coefficients) > 6:
            raise DegradationError(
                "a baseline above order 5 is no longer a baseline; it will absorb real bands"
            )
        if not all(np.isfinite(self.coefficients)):
            raise DegradationError("baseline coefficients must be finite")

    def baseline(self, axis: AxisSpec, intensity: FloatArray) -> FloatArray:
        """The additive baseline in intensity units."""
        positions = normalised_positions(axis)
        scale = signal_scale(intensity, self.scale_mode)
        total = np.zeros_like(positions)
        for power, coefficient in enumerate(self.coefficients):
            total = total + coefficient * positions**power
        return total * scale

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        return axis, intensity + self.baseline(axis, intensity)


@dataclass(frozen=True)
class FluorescenceBackground(Degradation):
    """A broad, intense, smoothly decaying background.

    Modelled as an exponential in normalised axis position, which is the shape a fluorescence tail
    usually takes across a Raman window. ``amplitude`` is relative to the signal magnitude named by
    ``scale_mode`` and is deliberately allowed to exceed 1: fluorescence routinely buries the Raman
    signal entirely, and a forward model that cannot express that cannot generate an ``F04`` case.

    ``decay`` sets how fast it falls across the range; positive values decay from the low-wavenumber
    end, negative from the high end.
    """

    amplitude: float = 1.0
    decay: float = 3.0
    scale_mode: ScaleMode = field(kw_only=True)

    def __post_init__(self) -> None:
        """Validate the amplitude and decay."""
        non_negative(self.amplitude, "amplitude")
        if not np.isfinite(self.decay):
            raise DegradationError("decay must be finite")

    def background(self, axis: AxisSpec, intensity: FloatArray) -> FloatArray:
        """The additive fluorescence background in intensity units."""
        positions = normalised_positions(axis)
        shape = np.exp(-self.decay * positions)
        return self.amplitude * signal_scale(intensity, self.scale_mode) * shape

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        return axis, intensity + self.background(axis, intensity)


@dataclass(frozen=True)
class CosmicRaySpikes(Degradation):
    """Sparse, narrow, intense spikes, as a cosmic ray leaves on a CCD.

    Width is in *samples*, not cm^-1, because a cosmic ray hits pixels: its apparent spectral width
    is an artifact of the dispersion, which is exactly why despiking works and why a spike that
    survives into training data is a grid-dependent shortcut (``F03``).

    ``amplitude_relative`` is measured against
    :attr:`ScaleMode.PEAK <usis.measurement.forward_model.ScaleMode>` - the 99th percentile of the
    signal - and unlike the backgrounds and noise in this package that is *not* a choice the caller
    makes: a spike buried under the bands is not a despiking problem at all, so there is only one
    magnitude the parameter can mean.
    """

    count: int = 3
    amplitude_relative: float = 5.0
    width_samples: int = 1

    def __post_init__(self) -> None:
        """Validate the spike parameters."""
        if self.count < 1:
            raise DegradationError("CosmicRaySpikes needs count >= 1")
        positive(self.amplitude_relative, "amplitude_relative")
        if not 1 <= self.width_samples <= 5:
            raise DegradationError(
                f"width_samples must be 1-5; {self.width_samples} is a band, not a cosmic ray"
            )

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        if self.count > intensity.size:
            raise DegradationError(
                f"cannot place {self.count} spikes in a spectrum of {intensity.size} points"
            )
        out = intensity.copy()
        peak = self.amplitude_relative * signal_scale(intensity, ScaleMode.PEAK)
        positions = rng.choice(intensity.size, size=self.count, replace=False)
        for centre in positions:
            low = max(0, int(centre) - self.width_samples // 2)
            high = min(intensity.size, low + self.width_samples)
            out[low:high] += peak
        return axis, out


@dataclass(frozen=True)
class Saturation(Degradation):
    """Hard clipping at a detector ceiling.

    ``level_relative`` is the ceiling as a fraction of the spectrum's maximum, so 1.0 clips nothing
    and 0.5 clips everything above half the peak. Saturation destroys quantitative information
    without leaving an obvious mark, which is why the measurement audit counts clipped samples
    rather than trusting the shape (``F05``).
    """

    level_relative: float = 0.8

    def __post_init__(self) -> None:
        """Validate the clipping level."""
        as_severity(self.level_relative, "level_relative")
        if self.level_relative == 0.0:
            raise DegradationError("level_relative of 0 would clip the entire spectrum to zero")

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        ceiling = self.level_relative * float(np.max(intensity))
        return axis, np.minimum(intensity, ceiling)
