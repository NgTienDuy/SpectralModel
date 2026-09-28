"""The instrument forward model.

Physics first: each degradation is checked against a property that follows from what it claims to
model, not merely against its own output. The resolution rules are additionally validated by
measuring the width back off a *real* QM9S spectrum, whose input broadening we measured
independently in `tests/test_qm9s.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from usis.core.types import (
    AxisKind,
    AxisSpec,
    IntensityKind,
    Modality,
    Origin,
    SourceRecord,
    Spectrum,
)
from usis.data.sources.qm9s import Qm9sSource
from usis.measurement import (
    AxisMiscalibration,
    BaselineDrift,
    CosmicRaySpikes,
    DarkCurrent,
    DegradationError,
    FluorescenceBackground,
    ForwardModel,
    IntensityResponse,
    PoissonNoise,
    ReadoutNoise,
    Resampling,
    ResolutionConvolution,
    Saturation,
    ScaleMode,
)
from usis.measurement.response import composed_fwhm

QM9S_FIXTURE = Path(__file__).parent / "fixtures" / "qm9s" / "qm9s_raman_subset.csv.gz"


def _measured_fwhm(wavenumbers, intensity) -> float:
    """FWHM of the tallest peak, with half-maximum crossings interpolated between grid points."""
    peak = int(np.argmax(intensity))
    half = intensity[peak] / 2.0
    left = peak
    while intensity[left] > half:
        left -= 1
    right = peak
    while intensity[right] > half:
        right += 1
    low = np.interp(half, intensity[left : left + 2], wavenumbers[left : left + 2])
    high = np.interp(
        half, intensity[right - 1 : right + 1][::-1], wavenumbers[right - 1 : right + 1][::-1]
    )
    return float(high - low)


def _synthetic(n=512, peaks=((800.0, 100.0), (1400.0, 60.0)), fwhm=6.0) -> Spectrum:
    axis = AxisSpec.from_values(
        AxisKind.RAMAN_SHIFT_CM1, np.linspace(400.0, 1800.0, n), resolution_fwhm=fwhm
    )
    values = np.asarray(axis.values)
    intensity = np.full(n, 1.0)
    for centre, height in peaks:
        gamma = fwhm / 2.0
        intensity = intensity + height * gamma**2 / ((values - centre) ** 2 + gamma**2)
    return Spectrum(
        spectrum_id="synthetic",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=axis,
        intensity=intensity,
        intensity_kind=IntensityKind.ARBITRARY,
        sample_id="synthetic",
        source=SourceRecord(dataset_id="rod", licence="CC0-1.0", retrieved_at="t"),
    )


@pytest.fixture
def clean() -> Spectrum:
    return _synthetic()


@pytest.fixture(scope="module")
def qm9s() -> Spectrum:
    return next(iter(Qm9sSource("raman").parse_file(QM9S_FIXTURE)))


# --- provenance -------------------------------------------------------------------------------
def test_a_degradation_records_itself_and_marks_the_result_augmented(clean):
    out = ReadoutNoise(sigma_relative=0.05, scale_mode=ScaleMode.MEDIAN).apply(clean, seed=1)
    assert out.origin is Origin.AUGMENTED
    step = out.processing[-1]
    assert step.name == "measure.ReadoutNoise"
    assert step.params["sigma_relative"] == 0.05
    assert step.params["seed"] == 1
    # sigma described the input's uncertainty and no longer describes the output.
    assert out.sigma is None


def test_degradations_are_deterministic_given_a_seed(clean):
    a = ReadoutNoise(scale_mode=ScaleMode.MEDIAN).apply(clean, seed=3)
    b = ReadoutNoise(scale_mode=ScaleMode.MEDIAN).apply(clean, seed=3)
    c = ReadoutNoise(scale_mode=ScaleMode.MEDIAN).apply(clean, seed=4)
    assert np.array_equal(a.intensity, b.intensity)
    assert not np.array_equal(a.intensity, c.intensity)


def test_the_input_spectrum_is_never_mutated(clean):
    before = clean.intensity.copy()
    CosmicRaySpikes().apply(clean, seed=0)
    assert np.array_equal(clean.intensity, before)
    assert clean.origin is Origin.SIMULATED


# --- resolution -------------------------------------------------------------------------------
def test_like_shape_composition_rules():
    assert composed_fwhm(3.0, 4.0, "gaussian", "gaussian") == pytest.approx(5.0)
    assert composed_fwhm(3.0, 4.0, "lorentzian", "lorentzian") == pytest.approx(7.0)


def test_mixed_shapes_use_the_voigt_approximation_not_quadrature():
    # Quadrature would give 22.36 here, and the width measured off a real spectrum is 26.0.
    voigt = composed_fwhm(10.0, 20.0, "lorentzian", "gaussian")
    assert voigt == pytest.approx(25.88, abs=0.05)
    assert voigt > np.hypot(10.0, 20.0)


@pytest.mark.parametrize("target", [15.0, 20.0, 30.0])
def test_lorentzian_broadening_of_real_qm9s_data_hits_its_target(qm9s, target):
    # QM9S Raman carries a measured 10 cm^-1 Lorentzian. Broadening it to a stated target and
    # measuring the width back checks the composition rule against data, not against itself.
    conv = ResolutionConvolution.to_target(10.0, target, shape="lorentzian")
    out = conv.apply(qm9s, seed=0)
    measured = _measured_fwhm(out.axis.values, out.intensity)
    assert out.axis.resolution_fwhm == pytest.approx(target)
    assert measured == pytest.approx(target, rel=0.03)


def test_mixed_shape_broadening_of_real_data_matches_the_declared_width(qm9s):
    conv = ResolutionConvolution(kernel_fwhm_cm1=20.0, shape="gaussian", input_shape="lorentzian")
    out = conv.apply(qm9s, seed=0)
    measured = _measured_fwhm(out.axis.values, out.intensity)
    assert out.axis.resolution_fwhm == pytest.approx(measured, rel=0.02)
    assert out.processing[-1].params["assumed_input_shape"] == "lorentzian"


def test_convolution_broadens_and_never_sharpens(clean):
    out = ResolutionConvolution(kernel_fwhm_cm1=12.0).apply(clean, seed=0)
    assert _measured_fwhm(out.axis.values, out.intensity) > _measured_fwhm(
        clean.axis.values, clean.intensity
    )
    assert out.axis.resolution_fwhm > clean.axis.resolution_fwhm


def test_convolution_conserves_area(clean):
    out = ResolutionConvolution(kernel_fwhm_cm1=10.0).apply(clean, seed=0)
    assert out.intensity.sum() == pytest.approx(clean.intensity.sum(), rel=1e-3)


def test_to_target_refuses_to_sharpen():
    with pytest.raises(DegradationError, match="cannot sharpen"):
        ResolutionConvolution.to_target(10.0, 5.0)
    with pytest.raises(DegradationError, match="cannot sharpen"):
        ResolutionConvolution.to_target(10.0, 10.0)


def test_to_target_refuses_a_mixed_composition_it_cannot_invert():
    with pytest.raises(DegradationError, match="composed_fwhm"):
        ResolutionConvolution.to_target(10.0, 20.0, shape="gaussian", input_shape="lorentzian")


def test_convolution_refuses_a_non_uniform_axis():
    axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, np.array([100.0, 101.0, 103.0, 110.0]))
    spectrum = Spectrum(
        spectrum_id="x",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=axis,
        intensity=np.ones(4),
        intensity_kind=IntensityKind.ARBITRARY,
        sample_id="x",
        source=SourceRecord(dataset_id="rod", licence="CC0-1.0", retrieved_at="t"),
    )
    with pytest.raises(DegradationError, match="uniform axis"):
        ResolutionConvolution(kernel_fwhm_cm1=5.0).apply(spectrum, seed=0)


def test_a_kernel_wider_than_the_spectrum_is_refused(clean):
    with pytest.raises(DegradationError, match="wider than the measured range"):
        ResolutionConvolution(kernel_fwhm_cm1=5000.0, shape="lorentzian").apply(clean, seed=0)


# --- axis miscalibration ----------------------------------------------------------------------
def test_an_axis_offset_moves_the_peak_but_not_the_axis(clean):
    out = AxisMiscalibration(offset_cm1=10.0).apply(clean, seed=0)
    assert np.array_equal(out.axis.values, clean.axis.values)
    before = clean.axis.values[int(np.argmax(clean.intensity))]
    after = out.axis.values[int(np.argmax(out.intensity))]
    # The instrument reports position p for what is really at p + offset, so the band appears to
    # move down by the offset.
    assert after == pytest.approx(before - 10.0, abs=clean.axis.mean_step * 1.5)


def test_an_all_zero_miscalibration_is_refused_as_a_no_op():
    with pytest.raises(DegradationError, match="no-op"):
        AxisMiscalibration()


def test_miscalibration_works_on_a_descending_axis():
    ascending = _synthetic()
    axis = AxisSpec.from_values(
        AxisKind.RAMAN_SHIFT_CM1, ascending.axis.values[::-1], resolution_fwhm=6.0
    )
    descending = Spectrum(
        spectrum_id="d",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=axis,
        intensity=ascending.intensity[::-1].copy(),
        intensity_kind=IntensityKind.ARBITRARY,
        sample_id="d",
        source=SourceRecord(dataset_id="rod", licence="CC0-1.0", retrieved_at="t"),
    )
    out = AxisMiscalibration(offset_cm1=10.0).apply(descending, seed=0)
    forward = AxisMiscalibration(offset_cm1=10.0).apply(ascending, seed=0)
    assert np.allclose(out.intensity[::-1], forward.intensity, atol=1e-9)


# --- response, backgrounds --------------------------------------------------------------------
def test_intensity_response_is_positive_and_attenuates_the_edges(clean):
    response = IntensityResponse(amplitude=0.5)
    curve = response.curve(clean.axis)
    assert np.all(curve > 0.0)
    assert curve[0] < curve[len(curve) // 2]
    assert curve[-1] < curve[len(curve) // 2]
    out = response.apply(clean, seed=0)
    assert np.all(out.intensity <= clean.intensity + 1e-9)


def test_baseline_and_fluorescence_only_add(clean):
    for degradation in (
        BaselineDrift(scale_mode=ScaleMode.MEDIAN),
        FluorescenceBackground(amplitude=2.0, scale_mode=ScaleMode.MEDIAN),
    ):
        out = degradation.apply(clean, seed=0)
        assert np.all(out.intensity >= clean.intensity - 1e-9)


def test_fluorescence_can_bury_the_signal_because_real_fluorescence_does(clean):
    out = FluorescenceBackground(amplitude=50.0, scale_mode=ScaleMode.MEDIAN).apply(clean, seed=0)
    contrast_before = clean.intensity.max() / np.median(clean.intensity)
    contrast_after = out.intensity.max() / np.median(out.intensity)
    assert contrast_after < contrast_before / 5


def test_an_absurd_baseline_order_is_refused():
    with pytest.raises(DegradationError, match="no longer a baseline"):
        BaselineDrift(coefficients=(1.0,) * 7, scale_mode=ScaleMode.MEDIAN)


# --- noise ------------------------------------------------------------------------------------
def test_poisson_noise_scales_as_the_square_root_of_the_signal(clean):
    # The defining property: SNR grows as sqrt(gain), so a hundredfold gain halves relative noise
    # by ten, not by a hundred.
    def relative_noise(gain: float) -> float:
        out = PoissonNoise(gain=gain).apply(clean, seed=0)
        residual = out.intensity - clean.intensity
        return float(residual.std() / clean.intensity.mean())

    low, high = relative_noise(1.0), relative_noise(100.0)
    assert low / high == pytest.approx(10.0, rel=0.25)


def test_poisson_noise_requires_a_gain_and_rejects_a_bad_one():
    with pytest.raises(TypeError):
        PoissonNoise()  # type: ignore[call-arg]
    with pytest.raises(DegradationError, match="gain"):
        PoissonNoise(gain=0.0)


def test_readout_noise_is_constant_across_the_spectrum(clean):
    out = ReadoutNoise(sigma_relative=0.1, scale_mode=ScaleMode.MEDIAN).apply(clean, seed=0)
    residual = out.intensity - clean.intensity
    first, second = residual[: len(residual) // 2], residual[len(residual) // 2 :]
    assert first.std() == pytest.approx(second.std(), rel=0.35)


def test_dark_current_grows_with_exposure_faster_than_its_noise(clean):
    short = DarkCurrent(rate_relative=0.5, exposure_s=1.0, scale_mode=ScaleMode.MEDIAN).apply(
        clean, seed=0
    )
    long = DarkCurrent(rate_relative=0.5, exposure_s=4.0, scale_mode=ScaleMode.MEDIAN).apply(
        clean, seed=0
    )
    level_short = (short.intensity - clean.intensity).mean()
    level_long = (long.intensity - clean.intensity).mean()
    assert level_long == pytest.approx(4.0 * level_short, rel=0.2)
    noise_ratio = (long.intensity - clean.intensity).std() / (
        short.intensity - clean.intensity
    ).std()
    assert noise_ratio == pytest.approx(2.0, rel=0.35)


# --- artifacts --------------------------------------------------------------------------------
def test_spikes_are_sparse_narrow_and_tall(clean):
    spikes = CosmicRaySpikes(count=3, amplitude_relative=20.0, width_samples=1)
    out = spikes.apply(clean, seed=0)
    changed = np.flatnonzero(np.abs(out.intensity - clean.intensity) > 1e-9)
    assert len(changed) == 3
    assert out.intensity.max() > clean.intensity.max() * 2


def test_a_wide_spike_is_refused_because_it_is_a_band():
    with pytest.raises(DegradationError, match="not a cosmic ray"):
        CosmicRaySpikes(width_samples=9)


def test_saturation_clips_and_destroys_the_peak_ratio(clean):
    out = Saturation(level_relative=0.3).apply(clean, seed=0)
    ceiling = 0.3 * clean.intensity.max()
    assert out.intensity.max() == pytest.approx(ceiling)
    assert np.all(out.intensity <= ceiling + 1e-9)
    assert (out.intensity >= ceiling - 1e-9).sum() > 1


# --- composition ------------------------------------------------------------------------------
def test_the_model_applies_degradations_in_the_canonical_order(clean):
    model = ForwardModel.of(
        Saturation(level_relative=0.9),
        ReadoutNoise(scale_mode=ScaleMode.MEDIAN),
        ResolutionConvolution(kernel_fwhm_cm1=4.0),
    )
    assert [type(d).__name__ for d in model.ordered()] == [
        "ResolutionConvolution",
        "ReadoutNoise",
        "Saturation",
    ]
    out = model.apply(clean, seed=0)
    assert [s.name for s in out.processing[-3:]] == [
        "measure.ResolutionConvolution",
        "measure.ReadoutNoise",
        "measure.Saturation",
    ]


def test_apply_each_degrades_independently_for_the_t4_sweep(clean):
    # E0 probe T4 sweeps one effect at a time; a model that could only apply a combined
    # degradation could not produce a metric-versus-severity curve.
    model = ForwardModel.of(
        ReadoutNoise(scale_mode=ScaleMode.MEDIAN),
        CosmicRaySpikes(),
        FluorescenceBackground(scale_mode=ScaleMode.MEDIAN),
    )
    separate = model.apply_each(clean, seed=0)
    assert len(separate) == 3
    for spectrum in separate:
        assert len(spectrum.processing) == len(clean.processing) + 1


def test_severity_is_controllable_and_monotone(clean):
    # The property T4 actually relies on: turning one knob up makes the effect bigger.
    deviations = [
        float(
            np.abs(
                ReadoutNoise(sigma_relative=s, scale_mode=ScaleMode.MEDIAN)
                .apply(clean, seed=0)
                .intensity
                - clean.intensity
            ).mean()
        )
        for s in (0.01, 0.05, 0.2)
    ]
    assert deviations == sorted(deviations)


def test_an_empty_model_is_refused():
    with pytest.raises(DegradationError, match="at least one degradation"):
        ForwardModel(())


def test_each_step_in_a_chain_gets_its_own_seed(clean):
    model = ForwardModel.of(ReadoutNoise(scale_mode=ScaleMode.MEDIAN), CosmicRaySpikes())
    out = model.apply(clean, seed=0)
    seeds = [s.params["seed"] for s in out.processing[-2:]]
    assert len(set(seeds)) == 2


# --- resampling and the sim-to-real direction --------------------------------------------
def test_real_rod_axes_are_not_uniform_and_that_is_physics():
    # A grating disperses linearly in wavelength; 1/lambda is not linear, so the wavenumber step
    # varies. The refusal in the convolution is responding to a real property of real data.
    from usis.data.sources.rod import RodSource

    rod = RodSource().parse_one(Path(__file__).parent / "fixtures" / "rod" / "8100020.rod")
    steps = np.diff(rod.axis.values)
    assert rod.axis.uniform is False
    spread = (steps.max() - steps.min()) / steps.mean()
    assert 0.001 < spread < 0.01  # ~0.2%: small, and still not uniform


def test_resampling_makes_a_real_spectrum_convolvable():
    from usis.data.sources.rod import RodSource

    rod = RodSource().parse_one(Path(__file__).parent / "fixtures" / "rod" / "8100020.rod")
    with pytest.raises(DegradationError, match="uniform axis"):
        ResolutionConvolution(kernel_fwhm_cm1=5.0).apply(rod, seed=0)
    resampled = Resampling(step_cm1=0.5).apply(rod, seed=0)
    assert resampled.axis.uniform is True
    assert resampled.axis.resolution_fwhm == rod.axis.resolution_fwhm
    ResolutionConvolution(kernel_fwhm_cm1=5.0).apply(resampled, seed=0)


def test_resampling_needs_exactly_one_target():
    with pytest.raises(DegradationError, match="exactly one"):
        Resampling()
    with pytest.raises(DegradationError, match="exactly one"):
        Resampling(step_cm1=1.0, n_points=100)


def test_the_simulation_cannot_be_sharpened_to_match_the_experiment():
    # QM9S carries 10 cm^-1 of applied broadening; ROD's instrument resolution is 4 cm^-1. So the
    # sim-to-real comparison cannot be made by degrading the simulation, which is the direction
    # everyone reaches for first. Recorded in docs/knowledge/datasets.md.
    with pytest.raises(DegradationError, match="cannot sharpen"):
        ResolutionConvolution.to_target(10.0, 4.0, shape="lorentzian")


def test_the_workable_direction_matches_the_experiment_to_the_simulation():
    from usis.data.sources.qm9s import Qm9sSource
    from usis.data.sources.rod import RodSource

    fixtures = Path(__file__).parent / "fixtures"
    qm9s = next(
        iter(Qm9sSource("raman").parse_file(fixtures / "qm9s" / "qm9s_raman_subset.csv.gz"))
    )
    rod = RodSource().parse_one(fixtures / "rod" / "8100020.rod")

    model = ForwardModel.of(
        ResolutionConvolution.to_target(4.0, 10.0, shape="lorentzian"),
        Resampling(step_cm1=0.5),
    )
    # The model applies resampling first regardless of the order given, because the kernel width
    # is undefined until the grid is uniform.
    assert [type(d).__name__ for d in model.ordered()] == ["Resampling", "ResolutionConvolution"]
    out = model.apply(rod, seed=0)
    assert out.axis.uniform is True
    assert out.axis.resolution_fwhm == pytest.approx(qm9s.axis.resolution_fwhm)
    # ROD's own source-declared calibration survives at the head of the chain.
    assert out.processing[0].name == "calibration.wavenumber"
    assert [s.name for s in out.processing[1:]] == [
        "measure.Resampling",
        "measure.ResolutionConvolution",
    ]
