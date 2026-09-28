"""Baseline estimation, three ways, none of them the right one everywhere.

Baseline correction is the step most likely to *create* the feature a downstream method then
identifies (``F02``), so three properties matter more than sophistication.

**All three are deterministic and pure numpy.** No optional dependency, no random state, so a
recorded chain replays bit-for-bit - which is what the reproduction criterion needs.

**None of them is a default.** :class:`~usis.preprocess.pipeline.Pipeline` takes the steps it is
given; there is no "standard preprocessing". The three exist to be *compared*, which is novelty
candidate N8: report how much a conclusion depends on which of them was chosen.

**Each one states what it does to a peak**, because that is the failure mode. A polynomial fitted
to the whole spectrum is pulled up by the bands and subtracts part of them; SNIP is peak-preserving
by construction but its iteration count sets a width above which a broad band is treated as
background; a rubberband hull is exact under a convex background and cuts into the first and last
bands when the background is not convex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from usis.core.types import AxisSpec, FloatArray
from usis.preprocess.base import PreprocessError, PreprocessStep, register_step

__all__ = ["PolynomialBaseline", "RubberbandBaseline", "SnipBaseline"]


@register_step
@dataclass(frozen=True)
class PolynomialBaseline(PreprocessStep):
    """Iteratively-reweighted polynomial baseline: fit, keep only what lies below the fit, repeat.

    A plain least-squares polynomial is pulled upward by every band, so subtracting it removes part
    of the signal. The iteration is the standard fix: after each fit, replace any sample above the
    fit by the fit itself, so bands stop voting on the next one. It converges to the lower envelope
    in a few passes.

    ``order`` is the real choice, and it is not free: too low leaves curvature, too high starts
    following the bands however many iterations are run. Compare orders rather than picking one.
    """

    order: int = 3
    iterations: int = 10

    def __post_init__(self) -> None:
        """Validate the polynomial order and iteration count."""
        if self.order < 0:
            raise PreprocessError(f"order must be non-negative, got {self.order}")
        if self.iterations < 1:
            raise PreprocessError(f"iterations must be at least 1, got {self.iterations}")

    def estimate(self, axis: AxisSpec, intensity: FloatArray) -> FloatArray:
        """The estimated baseline, exposed so the audit can report it, not only its effect."""
        values = np.asarray(axis.values, dtype=np.float64)
        # Fit on a normalised abscissa: a Vandermonde matrix in raw cm^-1 is badly conditioned at
        # order 3 and above, and the resulting baseline depends on where the axis starts.
        span = float(values.max() - values.min()) or 1.0
        positions = (values - values.min()) / span
        working = np.array(intensity, dtype=np.float64)
        fitted = working
        for _ in range(self.iterations):
            coefficients = np.polyfit(positions, working, self.order)
            fitted = np.asarray(np.polyval(coefficients, positions), dtype=np.float64)
            working = np.minimum(working, fitted)
        return fitted

    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Subtract the estimated baseline."""
        return axis, intensity - self.estimate(axis, intensity)


@register_step
@dataclass(frozen=True)
class SnipBaseline(PreprocessStep):
    """Statistics-sensitive non-linear iterative peak clipping, on a growing window.

    At window ``w`` each sample is replaced by the smaller of itself and the mean of the samples
    ``w`` away on either side. Run for ``w = 1 .. iterations``, a feature narrower than ``2w`` is
    clipped away and the background is left. It is peak-preserving in the sense that matters: it
    never *adds* to a sample, so it cannot create a band.

    The parameter to think about is therefore not a smoothness but a **width**: ``iterations`` in
    samples times the axis step is the width above which a band is treated as background. On a
    1 cm-1 grid, 40 iterations means bands wider than about 80 cm-1 are removed as background -
    which for a broad amorphous feature is a substantive claim about the sample, not a setting.
    """

    iterations: int = 40
    decreasing: bool = True

    def __post_init__(self) -> None:
        """Validate the iteration count."""
        if self.iterations < 1:
            raise PreprocessError(f"iterations must be at least 1, got {self.iterations}")

    def estimate(self, axis: AxisSpec, intensity: FloatArray) -> FloatArray:
        """The estimated background."""
        working = np.array(intensity, dtype=np.float64)
        n = working.size
        windows = (
            range(self.iterations, 0, -1) if self.decreasing else range(1, self.iterations + 1)
        )
        for window in windows:
            if 2 * window >= n:
                continue
            padded = np.pad(working, window, mode="edge")
            neighbour_mean = 0.5 * (padded[:n] + padded[2 * window :])
            working = np.minimum(working, neighbour_mean)
        return working

    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Subtract the estimated background."""
        return axis, intensity - self.estimate(axis, intensity)


@register_step
@dataclass(frozen=True)
class RubberbandBaseline(PreprocessStep):
    """The lower convex hull of the spectrum, interpolated between its vertices.

    Exact when the background really is convex, and its failure mode is specific rather than
    diffuse: a hull touches the spectrum at only a few points, so where the true background is
    concave the hull runs below it and leaves a residual hump, and at the ends it cuts into the
    first and last bands. Reported here rather than smoothed over because a downstream method
    cannot tell a residual hump from a broad band.
    """

    def estimate(self, axis: AxisSpec, intensity: FloatArray) -> FloatArray:
        """The lower hull, evaluated at every axis point."""
        values = np.asarray(axis.values, dtype=np.float64)
        order = np.argsort(values)
        x, y = values[order], np.asarray(intensity, dtype=np.float64)[order]
        hull = _lower_hull(x, y)
        estimate = np.interp(x, x[hull], y[hull])
        restored = np.empty_like(estimate)
        restored[order] = estimate
        return restored

    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Subtract the lower hull."""
        return axis, intensity - self.estimate(axis, intensity)


def _lower_hull(x: FloatArray, y: FloatArray) -> list[int]:
    """Indices of the lower convex hull of ``(x, y)``, x ascending. Monotone chain, O(n)."""
    hull: list[int] = []
    for index in range(x.size):
        while len(hull) >= 2 and _turns_upward(x, y, hull[-2], hull[-1], index):
            hull.pop()
        hull.append(index)
    return hull


def _turns_upward(x: FloatArray, y: FloatArray, a: int, b: int, c: int) -> bool:
    """Whether ``b`` lies on or above the segment ``a``-``c``, i.e. is not a hull vertex."""
    cross = (x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a])
    return bool(cross <= 0.0)
