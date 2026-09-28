"""The instrument forward model: resolution, response, calibration, noise, artifacts.

The highest-leverage component in the repo: it powers augmentation, sim-to-real transfer,
domain-shift stress tests and simulation-based inference from one implementation.

Milestone: M1.
"""

from __future__ import annotations

from usis.measurement.artifacts import (
    BaselineDrift,
    CosmicRaySpikes,
    FluorescenceBackground,
    Saturation,
)
from usis.measurement.broadening import (
    IntrinsicBroadening,
    deconvolved_fwhm,
    measured_intrinsic_widths,
)
from usis.measurement.calibration import AxisMiscalibration, Resampling
from usis.measurement.diagnostics import SpectralDiagnostics, observed_band_widths
from usis.measurement.forward_model import (
    Degradation,
    DegradationError,
    ForwardModel,
    ScaleMode,
    signal_scale,
)
from usis.measurement.noise import DarkCurrent, PoissonNoise, ReadoutNoise
from usis.measurement.response import IntensityResponse, ResolutionConvolution

__all__ = [
    "AxisMiscalibration",
    "BaselineDrift",
    "CosmicRaySpikes",
    "DarkCurrent",
    "Degradation",
    "DegradationError",
    "FluorescenceBackground",
    "ForwardModel",
    "IntensityResponse",
    "IntrinsicBroadening",
    "PoissonNoise",
    "ReadoutNoise",
    "Resampling",
    "ResolutionConvolution",
    "Saturation",
    "ScaleMode",
    "SpectralDiagnostics",
    "deconvolved_fwhm",
    "measured_intrinsic_widths",
    "observed_band_widths",
    "signal_scale",
]
