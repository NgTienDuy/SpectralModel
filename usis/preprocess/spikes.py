"""Cosmic-ray spike removal, and why it is the most dangerous step in the pipeline.

A cosmic ray is a genuine artifact: a single sample, one or two wide, tens of sigma above its
neighbours, from an ionising particle in the detector rather than from the sample. Removing it is
correct. But the operation is *also* a perfect way to produce failure code ``F03``, spike leakage,
in either direction:

- **Under-removal** leaves a feature that peak detection reports as an extremely narrow band, and
  that a downstream matcher may treat as diagnostic.
- **Over-removal** deletes real narrow bands. Sharp lattice modes in a crystalline sample can be
  only a few samples wide on a coarse grid, and a threshold set by "how many sigma above the
  neighbours" cannot distinguish them from a spike on shape alone.

The consequence is recorded rather than resolved: this step reports how many samples it replaced,
so the audit can say so, and a spectrum with many replacements is a measurement to re-take rather
than a spectrum to analyse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from usis.core.types import AxisSpec, FloatArray, Spectrum
from usis.preprocess.base import PreprocessError, PreprocessStep, register_step

__all__ = ["Despike", "spike_indices"]


@register_step
@dataclass(frozen=True)
class Despike(PreprocessStep):
    """Replace samples whose deviation from a running median exceeds ``threshold_sigma``.

    The detector is a median filter of width ``window`` and the scale is the median absolute
    deviation of the residual, both robust, because the thing being detected is itself an extreme
    value: a mean and a standard deviation are pulled up by the spike and hide it.

    ``max_width`` is the honest part. A spike is narrow by physics, so a run of flagged samples
    longer than ``max_width`` is not a spike - it is a band the threshold cannot tell from one, or
    a saturated plateau. Those runs are left alone and counted, so that over-removal becomes a
    reported number instead of a silent deletion.
    """

    window: int = 5
    threshold_sigma: float = 8.0
    max_width: int = 3

    def __post_init__(self) -> None:
        """Validate the window, threshold and maximum spike width."""
        if self.window < 3 or self.window % 2 == 0:
            raise PreprocessError(f"window must be odd and at least 3, got {self.window}")
        if self.threshold_sigma <= 0:
            raise PreprocessError(f"threshold_sigma must be positive, got {self.threshold_sigma}")
        if self.max_width < 1:
            raise PreprocessError(f"max_width must be at least 1, got {self.max_width}")

    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Replace flagged samples by the running median, leaving over-wide runs in place."""
        values = np.asarray(intensity, dtype=np.float64)
        smoothed = _running_median(values, self.window)
        flagged, _ = _flags(values, smoothed, self.threshold_sigma, self.max_width)
        cleaned = values.copy()
        cleaned[flagged] = smoothed[flagged]
        return axis, cleaned

    def report(self, spectrum: Spectrum) -> dict[str, float]:
        """What this step would do to ``spectrum``, without doing it.

        Separated from :meth:`_transform` so the measurement audit can report spike counts on a
        spectrum it is not about to modify - section 1 of the Sample Report describes the
        measurement, and describing it should not change it.
        """
        values = np.asarray(spectrum.intensity, dtype=np.float64)
        smoothed = _running_median(values, self.window)
        flagged, too_wide = _flags(values, smoothed, self.threshold_sigma, self.max_width)
        return {
            "spikes_replaced": float(flagged.sum()),
            "runs_left_as_too_wide": float(too_wide),
            "fraction_replaced": float(flagged.sum() / values.size) if values.size else 0.0,
        }


def spike_indices(
    intensity: FloatArray, window: int = 5, threshold_sigma: float = 8.0, max_width: int = 3
) -> tuple[int, ...]:
    """Indices flagged as spikes, for callers that want the positions rather than a new spectrum."""
    values = np.asarray(intensity, dtype=np.float64)
    smoothed = _running_median(values, window)
    flagged, _ = _flags(values, smoothed, threshold_sigma, max_width)
    return tuple(int(index) for index in np.flatnonzero(flagged))


def _running_median(values: FloatArray, window: int) -> FloatArray:
    """Median over a sliding window, edges held by reflection."""
    half = window // 2
    padded = np.pad(values, half, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.asarray(np.median(windows, axis=-1), dtype=np.float64)


def _flags(
    values: FloatArray, smoothed: FloatArray, threshold_sigma: float, max_width: int
) -> tuple[np.ndarray, int]:
    """Boolean spike mask and the number of flagged runs rejected for being too wide."""
    residual = values - smoothed
    mad = float(np.median(np.abs(residual - np.median(residual))))
    # 1.4826 makes the MAD a consistent estimator of sigma for a normal distribution; the same
    # constant is used in usis.measurement.diagnostics, deliberately, so the two agree on what
    # "sigma" means.
    sigma = mad * 1.4826
    if sigma <= 0.0:
        return np.zeros_like(values, dtype=bool), 0
    candidate = np.abs(residual) > threshold_sigma * sigma
    flagged = np.zeros_like(candidate)
    too_wide = 0
    for start, stop in _runs(candidate):
        if stop - start <= max_width:
            flagged[start:stop] = True
        else:
            too_wide += 1
    return flagged, too_wide


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges of consecutive ``True`` values."""
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2], strict=True)]
