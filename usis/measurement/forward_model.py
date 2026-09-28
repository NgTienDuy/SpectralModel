"""The instrument forward model: ``f(clean_spectrum, instrument_state) -> observed_spectrum``.

Brief §6: *the highest-leverage component in the repo*, because one implementation powers
augmentation, sim-to-real transfer, domain-shift stress tests and simulation-based inference. Two
design rules follow from what it has to serve, and both are visible in this file:

**Each degradation is independent and separately parameterised.** E0 probe T4 sweeps one
degradation at a time and plots metric against severity. A forward model that only applies a
plausible *combined* degradation cannot produce that curve, so there is no "realistic preset"
here — there is a list of operators and an explicit order.

**Every degradation records itself.** Applying one returns a new :class:`~usis.core.types.Spectrum`
whose origin is ``AUGMENTED`` and whose processing chain has gained a step naming the operator and
its parameters. A degraded spectrum that cannot say what was done to it is indistinguishable from a
measurement, which is failure code ``F13`` waiting to happen.

The canonical order, which :class:`ForwardModel` enforces, follows the physics of the measurement
chain rather than convenience:

1. intrinsic broadening — the sample's own linewidth, before any light reaches the spectrometer
2. resolution convolution — the instrument response function
3. axis miscalibration — the reported wavenumber is not the true one
4. intensity response — wavelength-dependent efficiency of optics and detector
5. additive backgrounds — baseline drift, then fluorescence
6. noise — dark current, then Poisson, then readout
7. artifacts — cosmic-ray spikes, then saturation

Saturation is last because it is a hard nonlinearity applied by the detector to everything that
reached it, and spikes come before it because a cosmic ray can itself saturate a pixel.

Intrinsic broadening is first because it is not an instrument effect at all: a vibrational band has
a width before it is measured, and a spectrometer convolves *on top of* it. Experiment E1 found
that omitting it is the single largest sim-to-real gap we have measured — see
:mod:`usis.measurement.broadening`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from usis.core.errors import UsisError
from usis.core.provenance import utc_now
from usis.core.types import Origin, ProcessingStep, Spectrum

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.random import Generator

    from usis.core.types import AxisSpec, FloatArray

__all__ = ["Degradation", "DegradationError", "ForwardModel", "ScaleMode", "signal_scale"]


class DegradationError(UsisError):
    """A degradation was asked for something physically impossible or undefined."""


@dataclass(frozen=True)
class Degradation(ABC):
    """One operator in the measurement chain.

    Subclasses implement :meth:`_degrade` and declare their parameters as dataclass fields; the
    base handles provenance and the origin change so no subclass can forget them.
    """

    @property
    def step_name(self) -> str:
        """The name recorded in the processing chain, e.g. ``measure.noise.poisson``."""
        return f"measure.{type(self).__name__}"

    def params(self) -> dict[str, Any]:
        """The parameters recorded alongside the step. Defaults to every dataclass field."""
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @abstractmethod
    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        """Return the degraded axis and intensity. Called once per application."""

    def apply(self, spectrum: Spectrum, *, seed: int) -> Spectrum:
        """Apply this degradation, recording it and marking the result as augmented.

        The seed is required rather than optional: an unseeded degradation produces a spectrum
        nobody can reproduce, and the whole point of the module is that its outputs are replayable.
        """
        rng = np.random.default_rng(seed)
        axis, intensity = self._degrade(spectrum.axis, spectrum.intensity, rng)
        step = ProcessingStep(
            name=self.step_name,
            params={**self.params(), "seed": seed},
            version="1",
            timestamp=utc_now(),
        )
        # sigma described the input's uncertainty and no longer describes the output. Dropping it
        # is honest; carrying it forward would understate the noise we just added.
        return replace(
            spectrum,
            axis=axis,
            intensity=intensity,
            origin=Origin.AUGMENTED,
            processing=(*spectrum.processing, step),
            sigma=None,
        )


@dataclass(frozen=True)
class ForwardModel:
    """A composed measurement chain applied in the canonical order.

    The order is a property of the physics, not of the list you pass in, so the model sorts by it.
    Two degradations of the same kind are applied in the order given.
    """

    degradations: tuple[Degradation, ...]

    #: Canonical ordering, by class name. Anything unlisted runs last, in the order supplied.
    ORDER: tuple[str, ...] = (
        # Resampling comes first: a kernel width in cm^-1 is undefined on a non-uniform grid.
        "Resampling",
        # Then the sample's own linewidth, which exists before the spectrometer sees the light.
        "IntrinsicBroadening",
        "ResolutionConvolution",
        "AxisMiscalibration",
        "IntensityResponse",
        "BaselineDrift",
        "FluorescenceBackground",
        "DarkCurrent",
        "PoissonNoise",
        "ReadoutNoise",
        "CosmicRaySpikes",
        "Saturation",
    )

    def __post_init__(self) -> None:
        """Reject an empty chain, which would silently be a no-op."""
        if not self.degradations:
            raise DegradationError("a ForwardModel needs at least one degradation")

    @classmethod
    def of(cls, *degradations: Degradation) -> ForwardModel:
        """Build a model from positional degradations."""
        return cls(tuple(degradations))

    def ordered(self) -> tuple[Degradation, ...]:
        """The degradations in the order they will actually be applied."""
        rank = {name: i for i, name in enumerate(self.ORDER)}
        return tuple(sorted(self.degradations, key=lambda d: rank.get(type(d).__name__, len(rank))))

    def apply(self, spectrum: Spectrum, *, seed: int) -> Spectrum:
        """Apply the whole chain.

        Each degradation gets its own derived seed, so adding a step to the chain does not silently
        change the random draw of every step after it.
        """
        result = spectrum
        for index, degradation in enumerate(self.ordered()):
            result = degradation.apply(result, seed=seed + 1000 * (index + 1))
        return result

    def apply_each(self, spectrum: Spectrum, *, seed: int) -> list[Spectrum]:
        """Apply each degradation **separately** to the original spectrum.

        This is what E0 probe T4 needs: one degradation at a time, so a metric can be plotted
        against the severity of that one effect rather than against an entangled combination.
        """
        return [
            degradation.apply(spectrum, seed=seed + 1000 * (index + 1))
            for index, degradation in enumerate(self.ordered())
        ]


def require_uniform(axis: AxisSpec, operation: str) -> float:
    """Return the grid step, or explain why ``operation`` cannot run on a non-uniform axis."""
    if not axis.uniform:
        raise DegradationError(
            f"{operation} needs a uniform axis: a convolution kernel in cm^-1 has no fixed width "
            "in samples otherwise. Resample the spectrum first, and record that as its own step."
        )
    return axis.mean_step


def as_severity(value: float, name: str) -> float:
    """Validate a severity-like parameter in ``[0, 1]``."""
    if not 0.0 <= value <= 1.0:
        raise DegradationError(f"{name} must be in [0, 1], got {value}")
    return value


def positive(value: float, name: str) -> float:
    """Validate a strictly positive parameter."""
    if value <= 0.0 or not np.isfinite(value):
        raise DegradationError(f"{name} must be finite and positive, got {value}")
    return value


def non_negative(value: float, name: str) -> float:
    """Validate a non-negative parameter."""
    if value < 0.0 or not np.isfinite(value):
        raise DegradationError(f"{name} must be finite and non-negative, got {value}")
    return value


def normalised_positions(axis: AxisSpec) -> FloatArray:
    """The axis mapped to ``[0, 1]``, so shape parameters do not depend on the spectral range."""
    values = np.asarray(axis.values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low)


class ScaleMode(StrEnum):
    """Which magnitude an amplitude is expressed against.

    Every additive term in this package - baselines, fluorescence, readout noise, dark current -
    takes its amplitude *relative* to the spectrum, so that one parameter means the same thing on
    an arbitrary-unit spectrum as on a calibrated one. Which magnitude to divide by is not a
    detail: experiment E1 measured a simulated-to-real dynamic-range gap of **+18.4 IQR**, and part
    of it was this choice made implicitly.

    On a measured spectrum the median *is* the baseline, so :attr:`MEDIAN` is the level a
    background sits on. On a simulated stick-and-broadening spectrum the median is the near-zero
    floor between bands, so "1.14 x median" adds almost nothing and the same parameter means two
    different physical things. :attr:`AMPLITUDE` is the peak-to-floor signal size, which is
    well-defined for both.

    There is no default anywhere this is used. A default here is exactly the kind of implicit
    choice that produced the E1 gap.
    """

    #: Median absolute intensity - the *baseline level*. Right when the median is the baseline.
    MEDIAN = "median"
    #: 99.9th percentile minus the median - the *peak-to-floor amplitude*. Right when the median is
    #: a floor rather than a baseline, and well-defined in both cases.
    AMPLITUDE = "amplitude"
    #: 99th percentile of the absolute intensity - the top of the dynamic range. What a cosmic ray
    #: has to stand above to be a despiking problem at all.
    PEAK = "peak"


def signal_scale(intensity: FloatArray, mode: ScaleMode) -> float:
    """A robust magnitude for the signal, so amplitudes can be relative rather than absolute.

    Percentiles rather than the maximum throughout: one cosmic ray already in the input would
    otherwise set the scale for everything applied afterwards.

    Degenerate spectra fall back rather than divide by zero - to the maximum absolute intensity,
    and finally to 1.0 for an all-zero spectrum. The fallback is deliberate and narrow: it keeps a
    flat input from raising, and any spectrum it fires on has no signal to scale against anyway.
    """
    values = np.asarray(intensity, dtype=np.float64)
    if mode is ScaleMode.MEDIAN:
        scale = float(np.median(np.abs(values)))
    elif mode is ScaleMode.AMPLITUDE:
        scale = float(np.quantile(values, 0.999) - np.median(values))
    elif mode is ScaleMode.PEAK:
        scale = float(np.quantile(np.abs(values), 0.99))
    else:  # pragma: no cover - Enum membership is exhaustive
        raise DegradationError(f"unknown scale mode {mode!r}")
    if scale <= 0.0:
        scale = float(np.max(np.abs(values)))
    return scale if scale > 0.0 else 1.0


def sequence_of(values: Sequence[float]) -> FloatArray:
    """Coerce a sequence of floats to an array, for parameters given as tuples."""
    return np.asarray(values, dtype=np.float64)
