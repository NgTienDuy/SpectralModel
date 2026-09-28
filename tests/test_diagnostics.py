"""Spectral diagnostics, checked against synthetic signals whose answers are known in advance."""

from __future__ import annotations

import numpy as np
import pytest

from usis.core.errors import UsisError
from usis.core.types import (
    AxisKind,
    AxisSpec,
    IntensityKind,
    Modality,
    Origin,
    SourceRecord,
    Spectrum,
)
from usis.measurement.diagnostics import (
    SpectralDiagnostics,
    autocorrelation_length,
    band_statistics,
    baseline_curvature,
    detect_peaks,
    dynamic_range,
    noise_level,
    noise_signal_dependence,
)


def _spectrum(intensity: np.ndarray, low: float = 400.0, high: float = 1800.0) -> Spectrum:
    axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, np.linspace(low, high, intensity.size))
    return Spectrum(
        spectrum_id="diag",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=axis,
        intensity=intensity,
        intensity_kind=IntensityKind.ARBITRARY,
        sample_id="diag",
        source=SourceRecord(dataset_id="rod", licence="CC0-1.0", retrieved_at="t"),
    )


def _bands(n: int = 2000, centres=(800.0, 1200.0), fwhm: float = 20.0, height: float = 50.0):
    axis = np.linspace(400.0, 1800.0, n)
    signal = np.full(n, 10.0)
    gamma = fwhm / 2.0
    for centre in centres:
        signal = signal + height * gamma**2 / ((axis - centre) ** 2 + gamma**2)
    return axis, signal


# --- noise ------------------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.5, 2.0, 8.0])
def test_noise_level_recovers_a_known_sigma(sigma):
    _, signal = _bands()
    noisy = signal + np.random.default_rng(0).normal(0.0, sigma, signal.size)
    assert noise_level(_spectrum(noisy)) == pytest.approx(sigma, rel=0.1)


def test_noise_level_is_not_inflated_by_a_cosmic_ray():
    _, signal = _bands()
    noisy = signal + np.random.default_rng(0).normal(0.0, 1.0, signal.size)
    spiked = noisy.copy()
    spiked[500] += 10_000.0
    clean_estimate = noise_level(_spectrum(noisy))
    spiked_estimate = noise_level(_spectrum(spiked))
    # A naive standard deviation would be destroyed here; the robust estimate barely moves.
    assert spiked_estimate == pytest.approx(clean_estimate, rel=0.05)
    assert spiked.std() > 10 * noisy.std()


def test_additive_noise_has_no_signal_dependence():
    _, signal = _bands()
    noisy = signal + np.random.default_rng(1).normal(0.0, 2.0, signal.size)
    slope, _ = noise_signal_dependence(_spectrum(noisy))
    assert abs(slope) < 0.2


def test_poisson_noise_has_a_positive_signal_dependence():
    # Variance equals the mean for a Poisson process, so the fitted slope should be near 1.
    _, signal = _bands(height=400.0)
    counts = np.random.default_rng(2).poisson(signal).astype(float)
    slope, r2 = noise_signal_dependence(_spectrum(counts))
    assert slope > 0.4
    assert r2 > 0.5


def test_noise_estimation_refuses_a_spectrum_too_short_to_estimate_from():
    with pytest.raises(UsisError, match="at least three points"):
        noise_level(_spectrum(np.array([1.0, 2.0])))
    with pytest.raises(UsisError, match="at least"):
        noise_signal_dependence(_spectrum(np.ones(10)))


# --- baseline ---------------------------------------------------------------------------------
def test_a_flat_spectrum_has_only_a_small_baseline_swing():
    # Not zero: Lorentzian tails genuinely lift the lower envelope near the bands, and an estimator
    # that reported zero here would be hiding a real property of the line shape.
    _, signal = _bands()
    assert baseline_curvature(_spectrum(signal)) < 0.2


def test_a_fluorescence_background_shows_up_as_a_large_swing():
    axis, signal = _bands()
    positions = (axis - axis.min()) / (axis.max() - axis.min())
    buried = signal + 200.0 * np.exp(-3.0 * positions)
    swing = baseline_curvature(_spectrum(buried))
    assert swing > 2.0


# --- bands ------------------------------------------------------------------------------------
def test_peak_detection_finds_the_planted_bands():
    _, signal = _bands(centres=(700.0, 1000.0, 1500.0))
    indices, prominences = detect_peaks(_spectrum(signal))
    assert len(indices) == 3
    assert np.all(prominences > 0)


def test_peak_detection_does_not_count_noise_as_bands():
    # Without a noise-floor term this reports around 120 bands on a spectrum that has two.
    _, signal = _bands()
    noisy = signal + np.random.default_rng(3).normal(0.0, 1.0, signal.size)
    assert len(detect_peaks(_spectrum(noisy))[0]) == 2
    # Lowering the margin reproduces the failure, which is the point of the parameter.
    assert len(detect_peaks(_spectrum(noisy), prominence_fraction=1e-9, min_snr=1e-9)[0]) > 50


def test_the_noise_threshold_scales_with_spectrum_length():
    # The largest noise excursion grows with the number of points; a fixed k*sigma would let long
    # spectra through. Measured: pure noise reaches ~7.1 sigma at n=500 and ~8.0 at n=8000.
    for n in (500, 2000, 8000):
        pure_noise = np.random.default_rng(7).normal(100.0, 1.0, n)
        assert len(detect_peaks(_spectrum(pure_noise))[0]) == 0


def test_band_width_recovers_the_planted_fwhm():
    for fwhm in (10.0, 30.0):
        _, signal = _bands(fwhm=fwhm)
        count, measured, _ = band_statistics(_spectrum(signal))
        assert count == 2
        assert measured == pytest.approx(fwhm, rel=0.25)


def test_band_width_is_measured_from_prominence_not_absolute_height():
    # A band on a large background would look absurdly wide if measured at half its absolute
    # height; measuring at half its prominence keeps it honest.
    axis, signal = _bands(fwhm=20.0)
    positions = (axis - axis.min()) / (axis.max() - axis.min())
    on_background = signal + 500.0 * np.exp(-2.0 * positions)
    _, measured, _ = band_statistics(_spectrum(on_background))
    assert measured == pytest.approx(20.0, rel=0.4)


def test_a_featureless_spectrum_reports_no_bands():
    count, width, _ = band_statistics(_spectrum(np.ones(500)))
    assert count == 0
    assert np.isnan(width)


# --- range and correlation ----------------------------------------------------------------------
def test_dynamic_range_is_not_set_by_one_spike():
    _, signal = _bands()
    spiked = signal.copy()
    spiked[100] = 1e6
    assert dynamic_range(_spectrum(spiked)) == pytest.approx(
        dynamic_range(_spectrum(signal)), rel=0.1
    )


def test_broader_bands_decorrelate_more_slowly():
    _, narrow = _bands(fwhm=5.0)
    _, wide = _bands(fwhm=50.0)
    assert autocorrelation_length(_spectrum(wide)) > autocorrelation_length(_spectrum(narrow))


# --- the bundle ---------------------------------------------------------------------------------
def test_the_diagnostics_bundle_computes_everything():
    _, signal = _bands()
    noisy = signal + np.random.default_rng(3).normal(0.0, 1.0, signal.size)
    diagnostics = SpectralDiagnostics.of(_spectrum(noisy))
    assert diagnostics.n_points == noisy.size
    assert diagnostics.n_bands == 2
    assert diagnostics.noise_sigma == pytest.approx(1.0, rel=0.15)
    assert diagnostics.span_cm1 == (400.0, 1800.0)
    assert set(diagnostics.to_dict()) >= {"noise_sigma", "baseline_swing", "n_bands"}


def test_diagnostics_are_orientation_independent():
    _, signal = _bands()
    ascending = SpectralDiagnostics.of(_spectrum(signal))
    axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, np.linspace(1800.0, 400.0, signal.size))
    descending = SpectralDiagnostics.of(
        Spectrum(
            spectrum_id="d",
            modality=Modality.RAMAN,
            origin=Origin.SIMULATED,
            axis=axis,
            intensity=signal[::-1].copy(),
            intensity_kind=IntensityKind.ARBITRARY,
            sample_id="d",
            source=SourceRecord(dataset_id="rod", licence="CC0-1.0", retrieved_at="t"),
        )
    )
    assert descending.n_bands == ascending.n_bands
    assert descending.noise_sigma == pytest.approx(ascending.noise_sigma, rel=1e-9)


def test_the_band_width_iqr_separates_a_constant_broadening_from_a_varied_one():
    """The statistic E1 needed and did not have.

    A corpus can match on median band width and still be trivially separable from a real one if
    every band in it has the same width. QM9S measured ``[11.00, 11.00]``; that is what this
    number exists to catch.
    """
    axis = np.linspace(400.0, 1800.0, 1401)
    constant = np.zeros_like(axis)
    varied = np.zeros_like(axis)
    rng = np.random.default_rng(0)
    for centre in np.arange(500.0, 1700.0, 100.0):
        gamma_same = 6.0
        gamma_other = float(rng.uniform(2.0, 15.0))
        constant = constant + 100.0 * gamma_same**2 / ((axis - centre) ** 2 + gamma_same**2)
        varied = varied + 100.0 * gamma_other**2 / ((axis - centre) ** 2 + gamma_other**2)

    _, _, iqr_constant = band_statistics(_spectrum(constant))
    _, _, iqr_varied = band_statistics(_spectrum(varied))
    assert iqr_constant < 2.0, iqr_constant
    assert iqr_varied > 3.0 * max(iqr_constant, 1.0), (iqr_constant, iqr_varied)


def test_the_iqr_is_not_a_number_when_there_are_no_bands():
    """No bands means no width distribution. Reporting 0.0 would read as 'perfectly uniform'."""
    count, width, iqr = band_statistics(_spectrum(np.ones(500)))
    assert count == 0
    assert np.isnan(width) and np.isnan(iqr)
