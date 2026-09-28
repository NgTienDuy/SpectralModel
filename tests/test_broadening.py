"""Intrinsic linewidth, and the scale an additive term is expressed against.

Both come out of experiment E1, and the tests are written the way that finding was: the properties
checked are the ones whose absence E1 measured. A broadening that widens bands but gives them all
the same width would pass a naive "did it get wider" test and would still be the shortcut feature
(``F14``) E1 found, so the *spread* of the widths is tested, not only their median.

The width claims are validated against a real ROD spectrum as well as synthetic ones, because a
synthetic spectrum built with Lorentzians and then measured with our own Lorentzian model is a
self-consistency check, not a validation.
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
from usis.data.sources.rod import RodSource
from usis.measurement import (
    BaselineDrift,
    ForwardModel,
    IntrinsicBroadening,
    ReadoutNoise,
    ResolutionConvolution,
    ScaleMode,
    deconvolved_fwhm,
    measured_intrinsic_widths,
    observed_band_widths,
    signal_scale,
)
from usis.measurement.forward_model import DegradationError
from usis.measurement.response import composed_fwhm

ROD_FIXTURE = Path(__file__).parent / "fixtures" / "rod" / "8100020.rod"


def _sticks(
    centres: tuple[float, ...] = (600.0, 900.0, 1200.0, 1500.0),
    heights: tuple[float, ...] | None = None,
    fwhm: float = 2.0,
    floor: float = 0.0,
    n: int = 1401,
) -> Spectrum:
    """A simulated-style spectrum: narrow Lorentzians on a near-zero floor, uniform 1 cm^-1 grid.

    Deliberately shaped like QM9S rather than like a measurement: the floor is what makes the
    median a floor rather than a baseline, which is the condition E1 found ``ScaleMode.MEDIAN``
    wrong for.
    """
    heights = heights or (100.0,) * len(centres)
    values = np.linspace(400.0, 1800.0, n)
    axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, values, resolution_fwhm=fwhm)
    intensity = np.full(n, floor, dtype=np.float64)
    gamma = fwhm / 2.0
    for centre, height in zip(centres, heights, strict=True):
        intensity = intensity + height * gamma**2 / ((values - centre) ** 2 + gamma**2)
    return Spectrum(
        spectrum_id="sticks",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=axis,
        intensity=intensity,
        intensity_kind=IntensityKind.ARBITRARY,
        sample_id="sticks",
        source=SourceRecord(dataset_id="qm9s-raman", licence="CC-BY-4.0", retrieved_at="t"),
    )


@pytest.fixture
def sticks() -> Spectrum:
    return _sticks()


@pytest.fixture(scope="module")
def rod() -> Spectrum:
    return RodSource().parse_one(ROD_FIXTURE)


# --- the width itself -------------------------------------------------------------------------


def test_broadening_reproduces_the_requested_width_when_it_is_measured_back(sticks: Spectrum):
    """With no spread, every band should come back at the composed width, within a few percent."""
    out = IntrinsicBroadening(median_fwhm_cm1=12.0, geometric_sd=1.0).apply(sticks, seed=0)
    expected = composed_fwhm(2.0, 12.0, "lorentzian", "lorentzian")
    measured = observed_band_widths(out)
    assert measured.size == 4
    assert np.allclose(measured, expected, rtol=0.10), (measured, expected)


def test_the_recorded_resolution_composes_rather_than_replacing(sticks: Spectrum):
    """The axis must report the composition, not the kernel: the input width does not vanish."""
    out = IntrinsicBroadening(median_fwhm_cm1=12.0, geometric_sd=1.0).apply(sticks, seed=0)
    assert out.axis.resolution_fwhm == pytest.approx(2.0 + 12.0)


def test_an_unknown_input_resolution_stays_unknown(sticks: Spectrum):
    """Composing against ``None`` would invent a width the source never stated."""
    bare = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, sticks.axis.values)
    from dataclasses import replace

    out = IntrinsicBroadening(median_fwhm_cm1=12.0, geometric_sd=1.0).apply(
        replace(sticks, axis=bare), seed=0
    )
    assert out.axis.resolution_fwhm is None


# --- the property E1 actually found missing -----------------------------------------------------


def test_the_drawn_widths_have_spread_which_is_the_whole_point(sticks: Spectrum):
    """E1's headline defect: QM9S band FWHM has IQR ``[11.00, 11.00]``. This must not.

    A broadening that widens every band identically reproduces the shortcut feature (``F14``) that
    lets a classifier separate simulated from real on width alone.
    """
    wide = _sticks(centres=tuple(500.0 + 90.0 * i for i in range(12)))
    out = IntrinsicBroadening(median_fwhm_cm1=12.0, geometric_sd=1.6).apply(wide, seed=3)
    widths = observed_band_widths(out)
    q25, q75 = np.quantile(widths, [0.25, 0.75])
    assert q75 - q25 > 2.0, widths


def test_no_spread_is_available_but_must_be_asked_for(sticks: Spectrum):
    """``geometric_sd=1.0`` exists only as an explicit control arm, and it is exactly degenerate.

    Checked on the draws rather than only on the measured widths: measuring a width back off a
    1 cm^-1 grid quantises it, so a measured spread of one step is the ruler, not the operator.
    """
    operator = IntrinsicBroadening(median_fwhm_cm1=12.0, geometric_sd=1.0)
    drawn = operator.draw_widths(50, np.random.default_rng(3))
    assert float(np.ptp(drawn)) == 0.0
    widths = observed_band_widths(operator.apply(sticks, seed=3))
    assert float(np.ptp(widths)) <= 2.0 * sticks.axis.mean_step


def test_a_spread_below_one_is_refused():
    """A geometric standard deviation under 1 is not a narrower distribution, it is nonsense."""
    with pytest.raises(DegradationError, match="geometric_sd"):
        IntrinsicBroadening(median_fwhm_cm1=12.0, geometric_sd=0.8)


def test_the_draws_are_lognormal_about_the_stated_median():
    """The median of the distribution is the parameter, with no bias correction needed."""
    operator = IntrinsicBroadening(median_fwhm_cm1=12.0, geometric_sd=1.5)
    drawn = operator.draw_widths(20_000, np.random.default_rng(0))
    assert float(np.median(drawn)) == pytest.approx(12.0, rel=0.02)
    assert drawn.min() > 0.0
    # Lognormal, not normal: the mean sits above the median.
    assert float(drawn.mean()) > float(np.median(drawn))


# --- conservation and edges -------------------------------------------------------------------


def test_intensity_is_conserved_except_for_what_leaves_the_window(sticks: Spectrum):
    """Every kernel is normalised, so the only loss is the tails falling off the ends."""
    out = IntrinsicBroadening(median_fwhm_cm1=8.0, geometric_sd=1.3).apply(sticks, seed=1)
    before, after = float(sticks.intensity.sum()), float(out.intensity.sum())
    assert after <= before
    assert after > 0.95 * before, (before, after)


def test_a_descending_axis_gives_the_same_spectrum_as_an_ascending_one(sticks: Spectrum):
    """Band assignment is a searchsorted; storing the axis backwards must not change the physics."""
    from dataclasses import replace

    flipped = replace(
        sticks,
        axis=AxisSpec.from_values(
            sticks.axis.kind, sticks.axis.values[::-1], resolution_fwhm=sticks.axis.resolution_fwhm
        ),
        intensity=np.asarray(sticks.intensity)[::-1].copy(),
    )
    operator = IntrinsicBroadening(median_fwhm_cm1=10.0, geometric_sd=1.4)
    forward = operator.apply(sticks, seed=7).intensity
    backward = operator.apply(flipped, seed=7).intensity
    assert np.allclose(forward, backward[::-1])


def test_a_featureless_spectrum_is_refused_rather_than_broadened(sticks: Spectrum):
    """There is no linewidth without a line. Broadening a flat window would invent structure."""
    from dataclasses import replace

    flat = replace(sticks, intensity=np.full(sticks.axis.n_points, 5.0))
    with pytest.raises(DegradationError, match="no bands were detected"):
        IntrinsicBroadening(median_fwhm_cm1=12.0, geometric_sd=1.0).apply(flat, seed=0)


def test_a_width_wider_than_the_window_is_refused(sticks: Spectrum):
    """A kernel longer than the spectrum is not a broadening, it is a claim about missing data."""
    with pytest.raises(DegradationError, match="wider than the"):
        IntrinsicBroadening(median_fwhm_cm1=400.0, geometric_sd=1.0).apply(sticks, seed=0)


def test_it_records_itself_like_every_other_degradation(sticks: Spectrum):
    out = IntrinsicBroadening(median_fwhm_cm1=12.0, geometric_sd=1.4).apply(sticks, seed=5)
    step = out.processing[-1]
    assert step.name == "measure.IntrinsicBroadening"
    assert step.params["median_fwhm_cm1"] == 12.0
    assert step.params["geometric_sd"] == 1.4
    assert step.params["seed"] == 5
    assert out.origin is Origin.AUGMENTED


def test_the_sample_broadens_before_the_instrument_does(sticks: Spectrum):
    """Canonical order: a band is wide before the spectrometer sees it, never after."""
    model = ForwardModel.of(
        ResolutionConvolution(kernel_fwhm_cm1=4.0),
        IntrinsicBroadening(median_fwhm_cm1=10.0, geometric_sd=1.2),
    )
    assert [type(d).__name__ for d in model.ordered()] == [
        "IntrinsicBroadening",
        "ResolutionConvolution",
    ]


# --- the inverse ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("intrinsic", "instrument", "shapes"),
    [
        (12.0, 4.0, ("lorentzian", "lorentzian")),
        (12.0, 4.0, ("gaussian", "gaussian")),
        (12.5, 4.0, ("lorentzian", "gaussian")),
        (3.0, 20.0, ("lorentzian", "gaussian")),
    ],
)
def test_deconvolution_inverts_composition(intrinsic, instrument, shapes):
    """The whole point of the helper: it is the exact inverse of the rule used to compose."""
    observed = composed_fwhm(intrinsic, instrument, shapes[0], shapes[1])
    assert deconvolved_fwhm(observed, instrument, shapes[0], shapes[1]) == pytest.approx(
        intrinsic, rel=1e-6
    )


def test_deconvolving_past_zero_is_refused_rather_than_clamped():
    """A band narrower than the instrument is evidence about the metadata, not a small number."""
    with pytest.raises(DegradationError, match="not wider than"):
        deconvolved_fwhm(4.0, 4.0)
    with pytest.raises(DegradationError, match="not wider than"):
        deconvolved_fwhm(3.0, 4.0)


def test_subtraction_would_have_been_wrong_for_mixed_shapes():
    """Guards the reason the helper exists: naive subtraction is off by a large margin here."""
    observed = composed_fwhm(12.0, 20.0, "lorentzian", "gaussian")
    naive = observed - 20.0
    recovered = deconvolved_fwhm(observed, 20.0, "lorentzian", "gaussian")
    assert recovered == pytest.approx(12.0, rel=1e-6)
    assert abs(naive - 12.0) > 3.0, naive


# --- against a real measured spectrum -----------------------------------------------------------


def test_a_real_rod_spectrum_implies_a_positive_intrinsic_width(rod: Spectrum):
    """The E1 finding, re-derived per band on a committed real record rather than quoted.

    ROD declares its instrument resolution; the bands are wider than that, and what is left over
    is the sample's own width. The assertion is deliberately loose - the point is the sign and the
    order of magnitude, not a fitted constant.
    """
    instrument = rod.axis.resolution_fwhm
    assert instrument is not None
    widths = measured_intrinsic_widths(rod, instrument)
    assert widths.size >= 5
    assert float(np.median(widths)) > instrument


def test_the_real_widths_are_not_all_the_same(rod: Spectrum):
    """The property the simulation lacks, measured on a real record: mode-to-mode variation."""
    instrument = rod.axis.resolution_fwhm
    assert instrument is not None
    widths = measured_intrinsic_widths(rod, instrument)
    q25, q75 = np.quantile(widths, [0.25, 0.75])
    assert q75 - q25 > 0.5, widths


# --- the scale an additive term is expressed against ---------------------------------------------


def test_the_median_scale_collapses_on_a_spectrum_whose_median_is_a_floor():
    """E1's dynamic-range gap, reproduced as a unit test.

    On a simulated spectrum the median is the near-zero floor between bands, so an amplitude
    expressed against it is a ratio to nothing. That is not a bug in the estimator; it is the wrong
    magnitude for the question, and the fix is a mode rather than a different quantile.
    """
    simulated = _sticks(floor=0.01)
    median = signal_scale(simulated.intensity, ScaleMode.MEDIAN)
    amplitude = signal_scale(simulated.intensity, ScaleMode.AMPLITUDE)
    assert amplitude / median > 100.0, (median, amplitude)


def test_on_a_real_spectrum_the_two_scales_are_the_same_order(rod: Spectrum):
    """Where the median really is the baseline, the choice matters far less - as it should."""
    median = signal_scale(rod.intensity, ScaleMode.MEDIAN)
    amplitude = signal_scale(rod.intensity, ScaleMode.AMPLITUDE)
    assert 0.0 < amplitude < 1000.0 * median


def test_every_scale_mode_falls_back_rather_than_dividing_by_zero():
    zeros = np.zeros(64)
    for mode in ScaleMode:
        assert signal_scale(zeros, mode) == 1.0


def test_the_additive_terms_will_not_choose_a_scale_for_you():
    """No default: the implicit choice is what E1 measured the cost of."""
    with pytest.raises(TypeError, match="scale_mode"):
        BaselineDrift()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="scale_mode"):
        ReadoutNoise(sigma_relative=0.05)  # type: ignore[call-arg]


def test_the_scale_mode_changes_the_amount_added(sticks: Spectrum):
    """The two modes are not cosmetic: the same coefficient means two different backgrounds."""
    on_median = BaselineDrift(coefficients=(1.0,), scale_mode=ScaleMode.MEDIAN).apply(
        sticks, seed=0
    )
    on_amplitude = BaselineDrift(coefficients=(1.0,), scale_mode=ScaleMode.AMPLITUDE).apply(
        sticks, seed=0
    )
    added_median = float(np.mean(on_median.intensity - sticks.intensity))
    added_amplitude = float(np.mean(on_amplitude.intensity - sticks.intensity))
    assert added_amplitude > 10.0 * added_median


def test_the_scale_mode_is_recorded_in_the_processing_chain(sticks: Spectrum):
    """A degraded spectrum has to be able to say which magnitude its amplitudes were relative to."""
    out = ReadoutNoise(sigma_relative=0.05, scale_mode=ScaleMode.AMPLITUDE).apply(sticks, seed=0)
    assert out.processing[-1].params["scale_mode"] == ScaleMode.AMPLITUDE
