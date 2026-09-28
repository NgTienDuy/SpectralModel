"""Detector noise: dark current, photon shot noise, readout noise.

The three are separate operators because they scale differently and a method that copes with one
need not cope with another. Poisson noise grows as the square root of the signal, so the
signal-to-noise ratio *varies across the spectrum*; readout noise is constant; dark current grows
with exposure. Denoisers that assume homoscedastic Gaussian noise are misspecified in exactly the
photon-limited regime where denoising is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from usis.measurement.forward_model import (
    Degradation,
    DegradationError,
    ScaleMode,
    non_negative,
    positive,
    signal_scale,
)

if TYPE_CHECKING:
    from numpy.random import Generator

    from usis.core.types import AxisSpec, FloatArray

__all__ = ["DarkCurrent", "PoissonNoise", "ReadoutNoise"]


@dataclass(frozen=True)
class PoissonNoise(Degradation):
    """Photon shot noise.

    ``gain`` is counts per unit of the spectrum's intensity and **must be supplied**: an
    arbitrary-unit spectrum carries no information about how many photons it represents, and
    guessing would silently set the noise level. A larger gain means more photons and so a higher
    signal-to-noise ratio.

    Negative inputs are clipped to zero before sampling, since a photon count cannot be negative;
    that is a real consequence of applying this after an over-subtracted baseline, and it is left
    visible rather than smoothed over.
    """

    gain: float

    def __post_init__(self) -> None:
        """Validate the gain."""
        positive(self.gain, "gain")

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        counts = np.clip(intensity * self.gain, 0.0, None)
        if not np.all(np.isfinite(counts)):
            raise DegradationError("intensity must be finite to sample Poisson noise")
        return axis, rng.poisson(counts).astype(np.float64) / self.gain


@dataclass(frozen=True)
class ReadoutNoise(Degradation):
    """Additive Gaussian noise, constant across the spectrum.

    ``sigma_relative`` is expressed against a magnitude of the spectrum rather than an absolute
    count, so the same parameter means the same thing on an arbitrary-unit spectrum as on a
    calibrated one. **Which magnitude is a required argument**: see
    :class:`~usis.measurement.forward_model.ScaleMode`, and E1's dynamic-range finding for what
    happens when it is chosen implicitly.
    """

    sigma_relative: float = 0.01
    scale_mode: ScaleMode = field(kw_only=True)

    def __post_init__(self) -> None:
        """Validate the noise level."""
        non_negative(self.sigma_relative, "sigma_relative")

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        sigma = self.sigma_relative * signal_scale(intensity, self.scale_mode)
        return axis, intensity + rng.normal(0.0, sigma, intensity.size)


@dataclass(frozen=True)
class DarkCurrent(Degradation):
    """A constant additive offset with its own shot noise, growing with exposure.

    ``rate_relative`` is the dark level per second relative to the signal magnitude named by
    ``scale_mode``; ``exposure_s`` multiplies it. Both appear because the pair is what distinguishes
    dark current from a plain offset: doubling the exposure doubles the level but only raises its
    noise by ``sqrt(2)``.
    """

    rate_relative: float = 0.01
    exposure_s: float = 1.0
    scale_mode: ScaleMode = field(kw_only=True)

    def __post_init__(self) -> None:
        """Validate the rate and exposure."""
        non_negative(self.rate_relative, "rate_relative")
        positive(self.exposure_s, "exposure_s")

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        level = self.rate_relative * self.exposure_s * signal_scale(intensity, self.scale_mode)
        if level == 0.0:
            return axis, intensity
        # Poisson on the dark level itself, so the noise grows as its square root.
        return axis, intensity + rng.poisson(level, intensity.size).astype(np.float64)
