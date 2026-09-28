"""Axis miscalibration: the wavenumber the instrument reports is not the one it measured.

This is the cheapest way to make two instruments disagree and the hardest to notice by eye, because
a shifted spectrum still looks like a spectrum. It is failure code ``F01`` in its measurement form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from usis.core.types import AxisSpec
from usis.measurement.forward_model import (
    Degradation,
    DegradationError,
    normalised_positions,
)

if TYPE_CHECKING:
    from numpy.random import Generator

    from usis.core.types import FloatArray

__all__ = ["AxisMiscalibration", "Resampling"]


@dataclass(frozen=True)
class AxisMiscalibration(Degradation):
    """Distort the wavenumber axis by an offset, a gain error and a quadratic term.

    The observed spectrum is the true one *read at the wrong positions*: the instrument reports
    position ``p`` for what is actually at ``p + offset + gain*p' + curvature*p'^2``, where ``p'``
    is the normalised position in the measured range. So the operation is an interpolation of the
    true intensity onto the distorted positions, not a shift of the axis values — the axis the
    instrument reports is unchanged, which is precisely why the error is invisible.

    Args:
        offset_cm1: constant shift, the dominant term in practice.
        gain_cm1: additional linear stretch across the full range.
        curvature_cm1: additional quadratic term across the full range.
    """

    offset_cm1: float = 0.0
    gain_cm1: float = 0.0
    curvature_cm1: float = 0.0

    def __post_init__(self) -> None:
        """Reject a no-op, which would record a step that did nothing."""
        if self.offset_cm1 == self.gain_cm1 == self.curvature_cm1 == 0.0:
            raise DegradationError(
                "AxisMiscalibration with all-zero terms is a no-op; omit it rather than recording "
                "a step that changed nothing"
            )
        for name in ("offset_cm1", "gain_cm1", "curvature_cm1"):
            if not np.isfinite(getattr(self, name)):
                raise DegradationError(f"{name} must be finite")

    def displacement(self, axis: AxisSpec) -> FloatArray:
        """The per-point displacement in cm^-1 between reported and true position."""
        positions = normalised_positions(axis)
        return self.offset_cm1 + self.gain_cm1 * positions + self.curvature_cm1 * positions**2

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        values = np.asarray(axis.values, dtype=np.float64)
        sample_at = values + self.displacement(axis)
        # np.interp needs an increasing grid; a descending axis is legal in this repo.
        if axis.ascending:
            observed = np.interp(sample_at, values, intensity)
        else:
            observed = np.interp(sample_at[::-1], values[::-1], intensity[::-1])[::-1]
        return axis, observed


@dataclass(frozen=True)
class Resampling(Degradation):
    """Interpolate onto a uniform grid over the same spectral span.

    Real dispersive spectrometers do **not** sample uniformly in wavenumber. A grating disperses
    linearly in wavelength across the detector, and ``1/lambda`` is non-linear, so the wavenumber
    step varies across the range. It is not noise and it is not a defect: the two committed ROD
    records vary by 0.2% and 25% respectively.

    That matters because convolution with a kernel specified in cm^-1 has no fixed width in samples
    on such a grid, so the resolution model refuses until the spectrum is resampled. This is that
    step, made explicit and recorded rather than folded silently into a loader.

    Interpolation cannot add information: resampling to a finer grid produces more points carrying
    the same content, and to a coarser one loses content irreversibly. Give exactly one of
    ``step_cm1`` or ``n_points``.
    """

    step_cm1: float | None = None
    n_points: int | None = None

    def __post_init__(self) -> None:
        """Require exactly one target specification."""
        if (self.step_cm1 is None) == (self.n_points is None):
            raise DegradationError("give exactly one of step_cm1 or n_points")
        if self.step_cm1 is not None and self.step_cm1 <= 0:
            raise DegradationError(f"step_cm1 must be positive, got {self.step_cm1}")
        if self.n_points is not None and self.n_points < 2:
            raise DegradationError(f"n_points must be at least 2, got {self.n_points}")

    def _degrade(
        self, axis: AxisSpec, intensity: FloatArray, rng: Generator
    ) -> tuple[AxisSpec, FloatArray]:
        low, high = axis.span
        if self.n_points is not None:
            grid = np.linspace(low, high, self.n_points)
        else:
            step = float(self.step_cm1 or 0.0)
            count = int(np.floor((high - low) / step)) + 1
            if count < 2:
                raise DegradationError(
                    f"a step of {step} cm^-1 spans the range [{low}, {high}] in fewer than two "
                    "points"
                )
            grid = low + np.arange(count, dtype=np.float64) * step
        values = np.asarray(axis.values, dtype=np.float64)
        if axis.ascending:
            resampled = np.interp(grid, values, intensity)
        else:
            resampled = np.interp(grid, values[::-1], intensity[::-1])
        new_axis = AxisSpec.from_values(axis.kind, grid, resolution_fwhm=axis.resolution_fwhm)
        return new_axis, resampled
