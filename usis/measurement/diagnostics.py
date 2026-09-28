"""Summary statistics that describe a spectrum as a *measurement*.

These are the numbers that answer "is this signal chemistry, or is it the instrument?" — section 1
of the Sample Report — and they are also what a sim-to-real comparison has to compare. Building
them once serves both.

Every estimator here is robust by construction, because the things they measure are exactly the
things outliers ruin: a single cosmic ray sets the maximum, a fluorescence tail sets the mean, and
a saturated region flattens the variance. Where a robust and a naive estimator would disagree, the
robust one is used and the docstring says why.

Nothing here is fitted to data or tuned against an expected answer. They are descriptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from usis.core.errors import UsisError
from usis.measurement.forward_model import ScaleMode, signal_scale

if TYPE_CHECKING:
    from usis.core.types import FloatArray, Spectrum

__all__ = [
    "SpectralDiagnostics",
    "autocorrelation_length",
    "band_statistics",
    "baseline_curvature",
    "detect_peaks",
    "dynamic_range",
    "median_band_spacing",
    "noise_level",
    "noise_signal_dependence",
    "observed_band_widths",
]

#: sqrt(6): the standard deviation of the second difference of white noise is sqrt(6)*sigma.
_SECOND_DIFFERENCE_FACTOR: float = float(np.sqrt(6.0))
#: Scale factor making the median absolute deviation a consistent estimator of sigma for a normal.
_MAD_TO_SIGMA: float = 1.4826


def _values(spectrum: Spectrum) -> tuple[FloatArray, FloatArray]:
    """Axis and intensity as plain arrays, sorted ascending by axis."""
    axis = np.asarray(spectrum.axis.values, dtype=np.float64)
    intensity = np.asarray(spectrum.intensity, dtype=np.float64)
    if not spectrum.axis.ascending:
        axis, intensity = axis[::-1], intensity[::-1]
    return axis, intensity


def noise_level(spectrum: Spectrum) -> float:
    """Robust noise standard deviation, from the second difference of the signal.

    A spectrum is smooth on the scale of its bands and rough on the scale of its noise, so the
    second difference is dominated by noise. For white noise its standard deviation is ``sqrt(6)``
    times the noise sigma, and the median absolute deviation of that difference gives an estimate
    that a handful of sharp bands or a cosmic ray cannot inflate.

    It is an *upper* bound where bands are only a few samples wide, because a genuinely sharp band
    also contributes to the second difference. That bias is stated rather than corrected: the
    correction would need a band model, and the estimator would stop being assumption-free.
    """
    _, intensity = _values(spectrum)
    if intensity.size < 3:
        raise UsisError("noise estimation needs at least three points")
    second = np.diff(intensity, n=2)
    mad = float(np.median(np.abs(second - np.median(second))))
    return mad * _MAD_TO_SIGMA / _SECOND_DIFFERENCE_FACTOR


def noise_signal_dependence(spectrum: Spectrum, n_bins: int = 8) -> tuple[float, float]:
    """How the noise variance scales with the signal level.

    Returns ``(slope, r_squared)`` of a least-squares fit of local noise **variance** against local
    signal level. The distinction it draws is the one that matters for denoising:

    * ``slope ~ 0`` — additive, signal-independent noise (readout-dominated);
    * ``slope > 0`` — photon-limited, where variance grows with signal, and a denoiser assuming
      homoscedastic Gaussian noise is misspecified in exactly the low-signal regime it is for.

    Bins are by signal level with equal counts, so a spectrum spending most of its range near the
    baseline does not put every bin there.
    """
    _, intensity = _values(spectrum)
    if intensity.size < 4 * n_bins:
        raise UsisError(
            f"noise_signal_dependence with {n_bins} bins needs at least {4 * n_bins} points"
        )
    second = np.diff(intensity, n=2)
    local_noise = np.abs(second) * _MAD_TO_SIGMA / _SECOND_DIFFERENCE_FACTOR
    level = intensity[1:-1]

    order = np.argsort(level)
    level, local_noise = level[order], local_noise[order]
    splits = np.array_split(np.arange(level.size), n_bins)
    x = np.array([level[idx].mean() for idx in splits])
    # Median of |second difference| per bin, squared to give a variance-like quantity.
    y = np.array([np.median(local_noise[idx]) ** 2 for idx in splits])

    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(((y - predicted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    return float(slope), float(1.0 - residual / total) if total > 0 else 0.0


def baseline_curvature(
    spectrum: Spectrum,
    window_fraction: float = 0.05,
    degree: int = 3,
    scale_mode: ScaleMode = ScaleMode.MEDIAN,
) -> float:
    """Peak-to-peak swing of the fitted lower envelope, relative to the signal's robust scale.

    A rolling minimum tracks the baseline under the bands; a low-order polynomial through it is the
    smooth background. The returned number is how much that background moves across the spectrum,
    in units of the magnitude named by ``scale_mode`` — so 0 is flat and 5 means the background
    swings by five times that magnitude.

    ``scale_mode`` defaults to ``MEDIAN`` so that the number keeps meaning what it meant in E1.
    Where this figure is used to *set* a degradation amplitude it must be measured against the same
    mode the degradation will use, or the parameter is transferred between two different units.
    """
    axis, intensity = _values(spectrum)
    if intensity.size < 4 * (degree + 1):
        raise UsisError("baseline estimation needs more points than the polynomial degree")
    window = max(3, int(window_fraction * intensity.size))
    pad = window // 2
    padded = np.pad(intensity, pad, mode="edge")
    envelope = np.array(
        [padded[i : i + window].min() for i in range(intensity.size)], dtype=np.float64
    )
    positions = (axis - axis.min()) / (axis.max() - axis.min())
    fitted = np.polyval(np.polyfit(positions, envelope, degree), positions)
    return float((fitted.max() - fitted.min()) / signal_scale(intensity, scale_mode))


def detect_peaks(
    spectrum: Spectrum, prominence_fraction: float = 0.05, min_snr: float = 2.5
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima that stand above both a fraction of the range **and** the noise floor.

    A deliberately simple prominence estimate: for each local maximum, the smaller of the two drops
    to the nearest higher-or-equal neighbour on each side. That is the textbook definition and
    needs no dependency; the full peak-fitting stack arrives with the M2 preprocessing layer.

    The noise term is not decoration, and it is not a plain multiple of sigma. A threshold
    expressed only as a fraction of the intensity range reports about 120 bands on a spectrum that
    has two, because it does not know where the noise floor is. But a fixed ``k * sigma`` is also
    wrong: **the largest noise excursion grows with the number of points**, which is the
    multiple-comparisons problem in disguise.

    Measured on pure Gaussian noise (20 spectra per length), the largest noise-only prominence is:

    ===========  ==========  =========  ================
    points       mean/sigma  max/sigma  2*sqrt(2 ln n)
    ===========  ==========  =========  ================
    500          5.7         7.1        7.05
    2,000        6.4         7.5        7.80
    8,000        7.2         8.0        8.48
    ===========  ==========  =========  ================

    So the threshold is ``min_snr * sigma * sqrt(2 ln n)``: the extreme-value scale, with
    ``min_snr`` as the margin above it. The default of 2.5 puts the cut at roughly 10 sigma for a
    2,000-point spectrum, comfortably above the ~7.5 sigma noise reaches there.

    Returns ``(indices, prominences)``.
    """
    _, intensity = _values(spectrum)
    if intensity.size < 3:
        return np.array([], dtype=int), np.array([], dtype=np.float64)
    low, high = np.quantile(intensity, [0.05, 0.999])
    threshold = prominence_fraction * float(high - low)
    if intensity.size >= 3:
        extreme_value_scale = float(np.sqrt(2.0 * np.log(max(intensity.size, 3))))
        threshold = max(threshold, min_snr * noise_level(spectrum) * extreme_value_scale)
    if threshold <= 0:
        return np.array([], dtype=int), np.array([], dtype=np.float64)

    interior = np.arange(1, intensity.size - 1)
    is_max = (intensity[interior] > intensity[interior - 1]) & (
        intensity[interior] >= intensity[interior + 1]
    )
    candidates = interior[is_max]

    keep: list[int] = []
    prominences: list[float] = []
    for index in candidates:
        height = intensity[index]
        left = index
        while left > 0 and intensity[left - 1] < height:
            left -= 1
        right = index
        while right < intensity.size - 1 and intensity[right + 1] < height:
            right += 1
        base = max(intensity[left : index + 1].min(), intensity[index : right + 1].min())
        prominence = height - base
        if prominence >= threshold:
            keep.append(int(index))
            prominences.append(float(prominence))
    return np.array(keep, dtype=int), np.array(prominences, dtype=np.float64)


def observed_band_widths(
    spectrum: Spectrum, prominence_fraction: float = 0.05, min_snr: float = 2.5
) -> FloatArray:
    """FWHM of every detected band, in axis units.

    Width is measured at half the band's *prominence* above its local base, not half its absolute
    height, so a band sitting on a fluorescence background is not reported as absurdly wide.

    Returns the whole set rather than a summary, because what a sim-to-real comparison has to
    compare is a *distribution*: E1's finding was not that the simulated median width was wrong but
    that its interquartile range was ``[11.00, 11.00]``.
    """
    axis, intensity = _values(spectrum)
    indices, prominences = detect_peaks(spectrum, prominence_fraction, min_snr)
    widths: list[float] = []
    for index, prominence in zip(indices, prominences, strict=True):
        half = intensity[index] - prominence / 2.0
        left = index
        while left > 0 and intensity[left] > half:
            left -= 1
        right = index
        while right < intensity.size - 1 and intensity[right] > half:
            right += 1
        if right > left:
            widths.append(float(axis[right] - axis[left]))
    return np.asarray(widths, dtype=np.float64)


def band_statistics(
    spectrum: Spectrum, prominence_fraction: float = 0.05, min_snr: float = 2.5
) -> tuple[int, float, float]:
    """Number of detected bands, their median FWHM, and the interquartile range of those widths.

    The third number is the one E1 needed and did not have. A corpus can match on median width and
    still be trivially separable from a real one if every band in it is the same width.
    """
    indices, _ = detect_peaks(spectrum, prominence_fraction, min_snr)
    widths = observed_band_widths(spectrum, prominence_fraction, min_snr)
    if widths.size == 0:
        return len(indices), float("nan"), float("nan")
    q25, q75 = np.quantile(widths, [0.25, 0.75])
    return len(indices), float(np.median(widths)), float(q75 - q25)


def median_band_spacing(
    spectrum: Spectrum, prominence_fraction: float = 0.05, min_snr: float = 2.5
) -> float:
    """Median distance between adjacent detected bands, in axis units; ``nan`` under two bands.

    Added at M2 because it turned out to be a confounder rather than a curiosity. The
    width-normalised correlation length that ``shape.broadening_origin`` conjectures reads band
    *shape* moves by a factor of 1.6 when band spacing alone is varied at fixed shape and width -
    more than the factor of 1.25 the shape change itself produces. A statistic computed from an
    autocorrelation over a whole spectrum cannot be attributed to single-band physics without
    reporting this number alongside it.
    """
    indices, _ = detect_peaks(spectrum, prominence_fraction, min_snr)
    if indices.size < 2:
        return float("nan")
    axis, _ = _values(spectrum)
    positions = axis[indices]
    return float(np.median(np.abs(np.diff(positions))))


def dynamic_range(spectrum: Spectrum) -> float:
    """Ratio of the 99.9th percentile to the median absolute intensity.

    A percentile rather than the maximum, so one cosmic ray does not define the range of the
    instrument.
    """
    _, intensity = _values(spectrum)
    median = float(np.median(np.abs(intensity)))
    if median == 0.0:
        return float("inf")
    return float(np.quantile(intensity, 0.999) / median)


def autocorrelation_length(spectrum: Spectrum) -> float:
    """Lag, in axis units, at which the autocorrelation of the detrended signal falls below ``1/e``.

    A compact summary of how wide the structure in a spectrum is: broad, heavily-broadened spectra
    decorrelate slowly, sharp ones quickly. Detrended first so that a fluorescence background does
    not dominate the correlation and report the whole spectrum as one feature.
    """
    axis, intensity = _values(spectrum)
    positions = (axis - axis.min()) / (axis.max() - axis.min())
    detrended = intensity - np.polyval(np.polyfit(positions, intensity, 3), positions)
    centred = detrended - detrended.mean()
    variance = float((centred**2).sum())
    if variance == 0.0:
        return float("nan")
    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1 :] / variance
    below = np.flatnonzero(correlation < 1.0 / np.e)
    if below.size == 0:
        return float("nan")
    step = float(np.abs(np.diff(axis)).mean())
    return float(below[0] * step)


@dataclass(frozen=True)
class SpectralDiagnostics:
    """Every statistic in this module, computed once for one spectrum."""

    spectrum_id: str
    n_points: int
    span_cm1: tuple[float, float]
    mean_step_cm1: float
    noise_sigma: float
    noise_relative: float
    noise_variance_slope: float
    noise_slope_r2: float
    baseline_swing: float
    n_bands: int
    median_band_fwhm_cm1: float
    band_fwhm_iqr_cm1: float
    dynamic_range: float
    autocorrelation_length_cm1: float

    @classmethod
    def of(cls, spectrum: Spectrum, prominence_fraction: float = 0.05) -> SpectralDiagnostics:
        """Compute the full set for one spectrum."""
        axis, intensity = _values(spectrum)
        sigma = noise_level(spectrum)
        scale = float(np.median(np.abs(intensity))) or 1.0
        slope, r2 = noise_signal_dependence(spectrum)
        n_bands, fwhm, fwhm_iqr = band_statistics(spectrum, prominence_fraction)
        return cls(
            spectrum_id=spectrum.spectrum_id,
            n_points=int(intensity.size),
            span_cm1=(float(axis.min()), float(axis.max())),
            mean_step_cm1=float(np.abs(np.diff(axis)).mean()),
            noise_sigma=sigma,
            noise_relative=sigma / scale,
            noise_variance_slope=slope,
            noise_slope_r2=r2,
            baseline_swing=baseline_curvature(spectrum),
            n_bands=n_bands,
            median_band_fwhm_cm1=fwhm,
            band_fwhm_iqr_cm1=fwhm_iqr,
            dynamic_range=dynamic_range(spectrum),
            autocorrelation_length_cm1=autocorrelation_length(spectrum),
        )

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable view."""
        return asdict(self)
