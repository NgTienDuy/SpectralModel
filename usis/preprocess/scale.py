"""Cropping, resampling and normalisation - the steps that change what a number means.

Normalisation is where a quantitative claim quietly becomes a relative one. Each mode below states
what it destroys, because "normalised" on its own does not say whether the result can still support
a concentration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from usis.core.types import AxisSpec, FloatArray, IntensityKind, Spectrum
from usis.preprocess.base import PreprocessError, PreprocessStep, register_step

__all__ = ["CropRange", "Normalisation", "NormaliseMode", "ResampleToGrid", "ResampleUniform"]


class NormaliseMode(StrEnum):
    """How to rescale intensity, and therefore what the result may still be used for.

    ``VECTOR``, ``AREA`` and ``MAX`` are single positive scalings: they preserve non-negativity and
    band ratios, and they destroy absolute scale, so a *relative* concentration survives and an
    absolute one does not. ``MINMAX`` and ``SNV`` additionally subtract an offset, which destroys
    non-negativity and with it Beer-Lambert additivity - after either of those, a linear-mixing
    method is solving a different equation, and
    :class:`~usis.methods.assumptions.AssumesLinearMixing` is the declaration that ought to catch
    it.
    """

    VECTOR = "vector"
    AREA = "area"
    MAX = "max"
    MINMAX = "minmax"
    SNV = "snv"

    @property
    def preserves_non_negativity(self) -> bool:
        """Whether a non-negative input stays non-negative."""
        return self in (NormaliseMode.VECTOR, NormaliseMode.AREA, NormaliseMode.MAX)


@register_step
@dataclass(frozen=True)
class Normalisation(PreprocessStep):
    """Rescale intensity by one of the :class:`NormaliseMode` conventions."""

    mode: NormaliseMode = NormaliseMode.VECTOR

    def __post_init__(self) -> None:
        """Reject an unknown mode rather than falling back to a default one."""
        NormaliseMode(self.mode)

    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Apply the chosen scaling, refusing when its divisor is degenerate."""
        mode = NormaliseMode(self.mode)
        values = np.asarray(intensity, dtype=np.float64)
        if mode is NormaliseMode.VECTOR:
            divisor = float(np.sqrt((values**2).sum()))
            return axis, _divide(values, divisor, mode)
        if mode is NormaliseMode.AREA:
            axis_values = np.asarray(axis.values, dtype=np.float64)
            divisor = float(abs(np.trapezoid(values, axis_values)))
            return axis, _divide(values, divisor, mode)
        if mode is NormaliseMode.MAX:
            divisor = float(np.abs(values).max())
            return axis, _divide(values, divisor, mode)
        if mode is NormaliseMode.MINMAX:
            low, high = float(values.min()), float(values.max())
            return axis, _divide(values - low, high - low, mode)
        standard_deviation = float(values.std())
        return axis, _divide(values - float(values.mean()), standard_deviation, mode)

    def describes_relative_intensity_only(self) -> bool:
        """Whether an absolute intensity claim survives this step. It never does."""
        return True


def _divide(values: FloatArray, divisor: float, mode: NormaliseMode) -> FloatArray:
    """Divide, refusing a degenerate divisor instead of returning zeros or infinities."""
    if not np.isfinite(divisor) or divisor == 0.0:
        raise PreprocessError(
            f"{mode} normalisation has divisor {divisor}: the spectrum is flat or empty on this "
            "measure, so the result would be all zeros or all infinities. A flat spectrum is a "
            "finding about the measurement, not something to normalise past."
        )
    return np.asarray(values / divisor, dtype=np.float64)


@register_step
@dataclass(frozen=True)
class CropRange(PreprocessStep):
    """Restrict the spectrum to ``[low_cm1, high_cm1]``.

    Refuses when the spectrum does not cover the requested window. Returning the intersection
    instead - the convenient behaviour - would make every downstream statistic silently conditional
    on a range that varies per spectrum, which is one of the ways a "cross-instrument" comparison
    ends up comparing spectral ranges.
    """

    low_cm1: float
    high_cm1: float

    def __post_init__(self) -> None:
        """Validate the window."""
        if not self.low_cm1 < self.high_cm1:
            raise PreprocessError(f"need low_cm1 < high_cm1, got [{self.low_cm1}, {self.high_cm1}]")

    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Select the samples inside the window."""
        values = np.asarray(axis.values, dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        if low > self.low_cm1 or high < self.high_cm1:
            raise PreprocessError(
                f"spectrum spans [{low:g}, {high:g}] and cannot be cropped to "
                f"[{self.low_cm1:g}, {self.high_cm1:g}]: the requested window is not covered by "
                "the measurement"
            )
        keep = (values >= self.low_cm1) & (values <= self.high_cm1)
        if int(keep.sum()) < 2:
            raise PreprocessError(
                f"cropping to [{self.low_cm1:g}, {self.high_cm1:g}] leaves {int(keep.sum())} "
                "samples"
            )
        return (
            AxisSpec.from_values(axis.kind, values[keep], resolution_fwhm=axis.resolution_fwhm),
            np.asarray(intensity, dtype=np.float64)[keep],
        )


@register_step
@dataclass(frozen=True)
class ResampleUniform(PreprocessStep):
    """Linear interpolation onto a uniform grid over the same span.

    Deliberately a separate step from :class:`usis.measurement.calibration.Resampling`, which does
    the same arithmetic and marks its output ``AUGMENTED``. The distinction is not bookkeeping:
    the measurement operator *simulates* an instrument that sampled differently, so its output is
    partly invented, while this one rearranges a measurement that already exists. Merging them
    would make every preprocessed real spectrum read as augmented.

    Interpolation adds no information. A finer grid gives more points carrying the same content; a
    coarser one loses content irreversibly, which is why ``n_points`` below the input count is
    allowed but recorded.
    """

    step_cm1: float | None = None
    n_points: int | None = None

    def __post_init__(self) -> None:
        """Require exactly one target specification."""
        if (self.step_cm1 is None) == (self.n_points is None):
            raise PreprocessError("give exactly one of step_cm1 or n_points")
        if self.step_cm1 is not None and self.step_cm1 <= 0:
            raise PreprocessError(f"step_cm1 must be positive, got {self.step_cm1}")
        if self.n_points is not None and self.n_points < 2:
            raise PreprocessError(f"n_points must be at least 2, got {self.n_points}")

    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Interpolate onto the requested uniform grid."""
        values = np.asarray(axis.values, dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        if self.n_points is not None:
            grid = np.linspace(low, high, self.n_points)
        else:
            step = float(self.step_cm1 or 0.0)
            count = int(np.floor((high - low) / step)) + 1
            if count < 2:
                raise PreprocessError(
                    f"a step of {step} cm-1 spans [{low:g}, {high:g}] in fewer than two points"
                )
            grid = low + np.arange(count, dtype=np.float64) * step
        order = np.argsort(values)
        resampled = np.interp(grid, values[order], np.asarray(intensity, dtype=np.float64)[order])
        return (
            AxisSpec.from_values(axis.kind, grid, resolution_fwhm=axis.resolution_fwhm),
            np.asarray(resampled, dtype=np.float64),
        )


@register_step
@dataclass(frozen=True)
class ResampleToGrid(PreprocessStep):
    """Linear interpolation onto a grid stated in absolute wavenumbers, shared by every spectrum.

    :class:`ResampleUniform` regularises a spectrum over *its own* span, so two spectra that began
    on different ranges end on different grids and remain incommensurable. That is the right
    behaviour when the question is about one measurement, and the wrong one for every comparison
    that indexes channels by position - library matching, PCA, MCR-ALS, any cross-instrument
    benchmark. This step is the other half: the grid is a parameter, so the output axis is
    identical for every spectrum the step is applied to, by construction rather than by luck.

    **It refuses rather than extrapolates.** A spectrum that does not cover the requested window
    raises, exactly as :class:`CropRange` does. Padding the ends - the convenient behaviour, and
    what ``numpy.interp`` does by default - would invent intensity outside the measurement and then
    hand it to a method as though it had been observed. The cost is that the corpus must be
    filtered to the window first, deliberately and with the count recorded, which is the intended
    cost: the subset a window selects is never random.
    """

    low_cm1: float
    high_cm1: float
    step_cm1: float

    def __post_init__(self) -> None:
        """Validate the target grid."""
        if not self.low_cm1 < self.high_cm1:
            raise PreprocessError(f"need low_cm1 < high_cm1, got [{self.low_cm1}, {self.high_cm1}]")
        if self.step_cm1 <= 0:
            raise PreprocessError(f"step_cm1 must be positive, got {self.step_cm1}")
        if self.grid.size < 2:
            raise PreprocessError(
                f"a step of {self.step_cm1:g} cm-1 spans "
                f"[{self.low_cm1:g}, {self.high_cm1:g}] in fewer than two points"
            )

    @property
    def grid(self) -> FloatArray:
        """The target axis. Derived from the recorded parameters, so a replay reproduces it."""
        count = int(np.floor((self.high_cm1 - self.low_cm1) / self.step_cm1)) + 1
        return self.low_cm1 + np.arange(max(count, 0), dtype=np.float64) * self.step_cm1

    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Interpolate onto the declared grid, refusing when the measurement does not cover it."""
        values = np.asarray(axis.values, dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        grid = self.grid
        if low > grid[0] or high < grid[-1]:
            raise PreprocessError(
                f"spectrum spans [{low:g}, {high:g}] and cannot be resampled onto "
                f"[{grid[0]:g}, {grid[-1]:g}]: the requested grid is not covered by the "
                "measurement, and interpolating past the ends would invent intensity that was "
                "never observed"
            )
        order = np.argsort(values)
        resampled = np.interp(grid, values[order], np.asarray(intensity, dtype=np.float64)[order])
        return (
            AxisSpec.from_values(axis.kind, grid, resolution_fwhm=axis.resolution_fwhm),
            np.asarray(resampled, dtype=np.float64),
        )


def relative_intensity_after(spectrum: Spectrum) -> bool:
    """Whether ``spectrum``'s recorded chain contains a normalisation.

    Used by the audit: an absolute-intensity claim about a normalised spectrum is unsupportable,
    and the chain is where that is knowable. ``IntensityKind`` cannot express it - a normalised
    count is still ``COUNTS`` - so the chain has to answer.
    """
    normalisation = f"preprocess.{Normalisation.__name__}"
    return any(step.name == normalisation for step in spectrum.processing)


def is_absorbance_like(kind: IntensityKind) -> bool:
    """Whether Beer-Lambert additivity is expected to hold for this intensity convention."""
    return kind in (IntensityKind.ABSORBANCE, IntensityKind.KUBELKA_MUNK)
