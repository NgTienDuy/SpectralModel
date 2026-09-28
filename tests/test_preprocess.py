"""Preprocessing: the numerics, the refusals, and the chain that must reproduce the spectrum."""

from __future__ import annotations

import json

import numpy as np
import pytest

from usis.core.types import (
    AxisKind,
    AxisSpec,
    InstrumentSpec,
    IntensityKind,
    Modality,
    Origin,
    ProcessingStep,
    SourceRecord,
    Spectrum,
)
from usis.measurement.artifacts import CosmicRaySpikes
from usis.measurement.diagnostics import noise_level
from usis.preprocess.base import (
    PREPROCESS_STEPS,
    PreprocessError,
    added_steps,
    from_recorded,
)
from usis.preprocess.baseline import PolynomialBaseline, RubberbandBaseline, SnipBaseline
from usis.preprocess.pipeline import (
    Pipeline,
    require_reproduction,
    verify_reproduction,
)
from usis.preprocess.scale import (
    CropRange,
    Normalisation,
    NormaliseMode,
    ResampleUniform,
    relative_intensity_after,
)
from usis.preprocess.search import (
    SearchError,
    defensible_pipelines,
    identity,
    search,
    sensitivity,
)
from usis.preprocess.smoothing import MovingAverage, SavitzkyGolay, window_cm1
from usis.preprocess.spikes import Despike, spike_indices

AXIS_VALUES = np.linspace(200.0, 1800.0, 1601)


def _spectrum(
    source: SourceRecord,
    *,
    spectrum_id: str = "p-0001",
    baseline: str = "none",
    noise: float = 1.0,
    axis_values: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    processing: tuple[ProcessingStep, ...] = (),
    origin: Origin = Origin.SIMULATED,
    instrument: InstrumentSpec | None = None,
) -> Spectrum:
    """A synthetic spectrum with four Gaussian bands and an optional background.

    Origin defaults to SIMULATED so no instrument declaration is required; the tests that care
    about origin pass one explicitly.
    """
    values = AXIS_VALUES if axis_values is None else axis_values
    axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, values, resolution_fwhm=4.0)
    if intensity is None:
        signal = np.zeros_like(values)
        for centre, height in ((420.0, 300.0), (800.0, 500.0), (1100.0, 200.0), (1580.0, 400.0)):
            signal += height * np.exp(-0.5 * ((values - centre) / 8.0) ** 2)
        positions = (values - values.min()) / (values.max() - values.min())
        if baseline == "linear":
            signal += 100.0 + 400.0 * positions
        elif baseline == "curved":
            signal += 100.0 + 900.0 * np.exp(-3.0 * positions)
        else:
            signal += 50.0
        signal += np.random.default_rng(11).normal(0.0, noise, values.size)
        intensity = signal
    return Spectrum(
        spectrum_id=spectrum_id,
        modality=Modality.RAMAN,
        origin=origin,
        axis=axis,
        intensity=np.asarray(intensity, dtype=np.float64),
        intensity_kind=IntensityKind.COUNTS,
        sample_id=f"sample-{spectrum_id}",
        source=source,
        instrument=instrument,
        processing=processing,
    )


# ------------------------------------------------------------------------------------------------
# The criterion: the recorded chain reproduces the spectrum
# ------------------------------------------------------------------------------------------------
def test_the_recorded_chain_reproduces_the_spectrum_exactly(source: SourceRecord) -> None:
    """M2 acceptance C4, verbatim - and the comparison demands bit equality, not closeness."""
    original = _spectrum(source, baseline="curved")
    pipeline = Pipeline.of(
        Despike(),
        SnipBaseline(iterations=40),
        SavitzkyGolay(window=9, order=3),
        Normalisation(NormaliseMode.VECTOR),
    )
    processed = pipeline.apply(original)
    outcome = verify_reproduction(original, processed)
    assert outcome.reproduced
    assert outcome.n_steps == 4
    assert outcome.max_abs_deviation == 0.0
    assert outcome.axis_max_abs_deviation == 0.0


def test_the_chain_replays_after_a_round_trip_through_json(source: SourceRecord) -> None:
    """A record that only replays in the process that wrote it is not provenance."""
    original = _spectrum(source, baseline="linear")
    processed = Pipeline.of(
        CropRange(300.0, 1700.0),
        PolynomialBaseline(order=3),
        MovingAverage(window=5),
    ).apply(original)
    chain = added_steps(original, processed)
    serialised = json.dumps(
        [{"name": s.name, "params": dict(s.params), "version": s.version} for s in chain]
    )
    rebuilt = Pipeline.from_records(
        tuple(
            ProcessingStep(
                name=entry["name"],
                params=entry["params"],
                version=entry["version"],
                timestamp="replayed",
            )
            for entry in json.loads(serialised)
        )
    )
    replayed = rebuilt.apply(original)
    assert np.array_equal(replayed.intensity, processed.intensity)
    assert np.array_equal(replayed.axis.values, processed.axis.values)


def test_reproduction_is_verified_from_the_record_not_the_object(source: SourceRecord) -> None:
    """Tampering with the recorded parameters must break the check."""
    original = _spectrum(source, baseline="curved")
    processed = Pipeline.of(SnipBaseline(iterations=40)).apply(original)
    tampered = Spectrum(
        spectrum_id=processed.spectrum_id,
        modality=processed.modality,
        origin=processed.origin,
        axis=processed.axis,
        intensity=processed.intensity,
        intensity_kind=processed.intensity_kind,
        sample_id=processed.sample_id,
        source=processed.source,
        processing=(
            *original.processing,
            ProcessingStep(
                name="preprocess.SnipBaseline",
                params={"iterations": 5, "decreasing": True},
                version="1",
                timestamp="tampered",
            ),
        ),
    )
    outcome = verify_reproduction(original, tampered)
    assert not outcome.reproduced
    assert outcome.max_abs_deviation > 0.0
    assert "record is incomplete" in outcome.detail
    with pytest.raises(PreprocessError):
        require_reproduction(original, tampered)


def test_a_chain_holding_a_foreign_step_refuses_to_replay(source: SourceRecord) -> None:
    """A measurement degradation is deliberately not reconstructible as preprocessing."""
    original = _spectrum(source)
    degraded = CosmicRaySpikes(count=5).apply(original, seed=1)
    processed = Pipeline.of(Despike()).apply(degraded)
    with pytest.raises(PreprocessError, match="not a registered preprocessing step"):
        verify_reproduction(original, processed)
    assert verify_reproduction(degraded, processed).reproduced


def test_a_version_mismatch_refuses_rather_than_replaying_different_numerics() -> None:
    record = ProcessingStep(
        name="preprocess.SnipBaseline",
        params={"iterations": 40, "decreasing": True},
        version="99",
        timestamp="x",
    )
    with pytest.raises(PreprocessError, match="version"):
        from_recorded(record)


def test_an_unknown_parameter_refuses_rather_than_being_ignored() -> None:
    record = ProcessingStep(
        name="preprocess.SnipBaseline",
        params={"iterations": 40, "decreasing": True, "smoothness": 3},
        version="1",
        timestamp="x",
    )
    with pytest.raises(PreprocessError, match="no parameter"):
        from_recorded(record)


def test_a_chain_that_is_not_an_extension_is_refused(source: SourceRecord) -> None:
    a = Pipeline.of(SnipBaseline()).apply(_spectrum(source, spectrum_id="a"))
    b = Pipeline.of(MovingAverage()).apply(_spectrum(source, spectrum_id="b"))
    with pytest.raises(PreprocessError, match="not an extension"):
        added_steps(a, b)


def test_every_step_class_is_registered_under_its_recorded_name() -> None:
    for name in PREPROCESS_STEPS:
        cls = PREPROCESS_STEPS.get(name)
        assert name == f"preprocess.{cls.__name__}"
    assert len(PREPROCESS_STEPS) >= 8


def test_a_parameter_that_would_not_survive_json_is_refused(source: SourceRecord) -> None:
    class BadStep(SnipBaseline):
        def params(self) -> dict[str, object]:
            return {"kernel": np.ones(3)}

    with pytest.raises(PreprocessError, match="does not survive a JSON round trip"):
        BadStep().apply(_spectrum(source))


# ------------------------------------------------------------------------------------------------
# Numerics
# ------------------------------------------------------------------------------------------------
def test_savitzky_golay_coefficients_are_the_textbook_ones() -> None:
    """window 5, order 2: (-3, 12, 17, 12, -3)/35 for smoothing, (-2, -1, 0, 1, 2)/10 for d/dx."""
    smoothing = SavitzkyGolay(window=5, order=2).coefficients(1.0)
    assert smoothing == pytest.approx(np.array([-3.0, 12.0, 17.0, 12.0, -3.0]) / 35.0)
    derivative = SavitzkyGolay(window=5, order=2, derivative=1).coefficients(1.0)
    assert derivative == pytest.approx(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / 10.0)


def test_savitzky_golay_reproduces_a_polynomial_it_can_represent(source: SourceRecord) -> None:
    """A cubic filter must leave a cubic untouched, which is the property the fit guarantees."""
    values = AXIS_VALUES
    positions = (values - values.min()) / (values.max() - values.min())
    cubic = 3.0 + 2.0 * positions - 5.0 * positions**2 + 4.0 * positions**3
    spectrum = _spectrum(source, intensity=cubic)
    smoothed = SavitzkyGolay(window=11, order=3).apply(spectrum)
    interior = slice(20, -20)
    assert np.asarray(smoothed.intensity)[interior] == pytest.approx(cubic[interior], abs=1e-9)


def test_savitzky_golay_smoothing_actually_reduces_the_noise(source: SourceRecord) -> None:
    noisy = _spectrum(source, noise=10.0)
    smoothed = SavitzkyGolay(window=11, order=3).apply(noisy)
    assert noise_level(smoothed) < 0.5 * noise_level(noisy)


def test_the_first_derivative_removes_an_additive_constant(source: SourceRecord) -> None:
    """The reason chemometrics reaches for derivatives instead of fighting the baseline."""
    base = _spectrum(source, baseline="none", noise=0.0)
    shifted = _spectrum(
        source, intensity=np.asarray(base.intensity, dtype=np.float64) + 250.0, noise=0.0
    )
    step = SavitzkyGolay(window=9, order=3, derivative=1)
    assert np.asarray(step.apply(base).intensity) == pytest.approx(
        np.asarray(step.apply(shifted).intensity), abs=1e-9
    )


def test_a_boxcar_is_a_zero_order_savitzky_golay(source: SourceRecord) -> None:
    spectrum = _spectrum(source, noise=5.0)
    boxcar = MovingAverage(window=7).apply(spectrum)
    order_zero = SavitzkyGolay(window=7, order=0).apply(spectrum)
    assert np.asarray(boxcar.intensity) == pytest.approx(np.asarray(order_zero.intensity))


def test_window_cm1_exposes_the_dispersion_a_sample_window_hides() -> None:
    """On a real dispersive axis one stated kernel width is wrong somewhere on the spectrum."""
    uniform = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, AXIS_VALUES)
    narrow, wide = window_cm1(uniform, 9)
    assert narrow == pytest.approx(wide)
    dispersive = AxisSpec.from_values(
        AxisKind.ABSOLUTE_WAVENUMBER_CM1, np.sort(1e7 / np.linspace(800.0, 900.0, 1601))
    )
    narrow, wide = window_cm1(dispersive, 9)
    assert wide > 1.2 * narrow


def test_snip_never_adds_to_a_sample_so_it_cannot_create_a_band(source: SourceRecord) -> None:
    spectrum = _spectrum(source, baseline="curved")
    background = SnipBaseline(iterations=40).estimate(spectrum.axis, spectrum.intensity)
    assert np.all(background <= np.asarray(spectrum.intensity, dtype=np.float64) + 1e-12)


def test_all_three_baselines_remove_a_curved_background(source: SourceRecord) -> None:
    """The measure is the residual at band-free positions, not eyeball agreement."""
    spectrum = _spectrum(source, baseline="curved", noise=0.0)
    values = np.asarray(spectrum.axis.values, dtype=np.float64)
    band_free = (values > 600.0) & (values < 750.0)
    before = float(np.abs(np.asarray(spectrum.intensity)[band_free]).mean())
    for step in (SnipBaseline(iterations=60), PolynomialBaseline(order=4), RubberbandBaseline()):
        corrected = np.asarray(step.apply(spectrum).intensity)[band_free]
        assert float(np.abs(corrected).mean()) < 0.1 * before, type(step).__name__


def test_the_three_baselines_disagree_with_each_other(source: SourceRecord) -> None:
    """Which is why N8 exists: the choice between them is not free."""
    spectrum = _spectrum(source, baseline="curved", noise=0.0)
    estimates = [
        step.estimate(spectrum.axis, spectrum.intensity)
        for step in (SnipBaseline(iterations=40), PolynomialBaseline(order=3), RubberbandBaseline())
    ]
    scale = float(np.abs(np.asarray(spectrum.intensity)).max())
    spreads = [
        float(np.abs(a - b).max()) / scale
        for a, b in ((estimates[0], estimates[1]), (estimates[1], estimates[2]))
    ]
    assert min(spreads) > 0.01, spreads


def test_a_polynomial_baseline_is_pulled_up_by_bands_without_iteration(
    source: SourceRecord,
) -> None:
    """One iteration is a plain least-squares fit, which subtracts part of every band."""
    spectrum = _spectrum(source, baseline="linear", noise=0.0)
    values = np.asarray(spectrum.axis.values, dtype=np.float64)
    band = np.abs(values - 800.0) < 20.0
    naive = np.asarray(PolynomialBaseline(order=3, iterations=1).apply(spectrum).intensity)
    iterated = np.asarray(PolynomialBaseline(order=3, iterations=10).apply(spectrum).intensity)
    assert naive[band].max() < iterated[band].max()


# ------------------------------------------------------------------------------------------------
# Scaling, cropping, resampling
# ------------------------------------------------------------------------------------------------
def test_cropping_refuses_a_window_the_measurement_does_not_cover(source: SourceRecord) -> None:
    spectrum = _spectrum(source)
    assert CropRange(300.0, 1700.0).apply(spectrum).axis.values.min() >= 300.0
    with pytest.raises(PreprocessError, match="not covered by the measurement"):
        CropRange(100.0, 1700.0).apply(spectrum)


def test_normalisation_modes_differ_in_what_they_destroy(source: SourceRecord) -> None:
    spectrum = _spectrum(source, baseline="none", noise=0.0)
    vector = np.asarray(Normalisation(NormaliseMode.VECTOR).apply(spectrum).intensity)
    snv = np.asarray(Normalisation(NormaliseMode.SNV).apply(spectrum).intensity)
    assert np.all(vector >= 0.0)
    assert snv.min() < 0.0
    assert NormaliseMode.VECTOR.preserves_non_negativity
    assert not NormaliseMode.SNV.preserves_non_negativity
    assert float(np.sqrt((vector**2).sum())) == pytest.approx(1.0)
    assert float(snv.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(snv.std()) == pytest.approx(1.0)


def test_normalisation_refuses_a_flat_spectrum_instead_of_dividing_by_zero(
    source: SourceRecord,
) -> None:
    flat = _spectrum(source, intensity=np.full(AXIS_VALUES.size, 7.0))
    with pytest.raises(PreprocessError, match="flat or empty"):
        Normalisation(NormaliseMode.MINMAX).apply(flat)


def test_an_unknown_normalisation_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="not a valid NormaliseMode"):
        Normalisation("standardise")  # type: ignore[arg-type]


def test_the_chain_is_what_records_that_intensity_became_relative(source: SourceRecord) -> None:
    """IntensityKind cannot express it: a normalised count is still COUNTS."""
    spectrum = _spectrum(source)
    normalised = Normalisation(NormaliseMode.VECTOR).apply(spectrum)
    assert not relative_intensity_after(spectrum)
    assert relative_intensity_after(normalised)
    assert normalised.intensity_kind is IntensityKind.COUNTS


def test_resampling_makes_a_dispersive_axis_uniform(source: SourceRecord) -> None:
    dispersive = np.sort(1e7 / np.linspace(800.0, 900.0, 1601))
    spectrum = _spectrum(source, axis_values=dispersive)
    steps = np.diff(np.asarray(spectrum.axis.values, dtype=np.float64))
    assert (steps.max() - steps.min()) / steps.mean() > 0.1
    resampled = ResampleUniform(n_points=1601).apply(spectrum)
    new_steps = np.diff(np.asarray(resampled.axis.values, dtype=np.float64))
    assert (new_steps.max() - new_steps.min()) / new_steps.mean() < 1e-9


def test_resampling_requires_exactly_one_target() -> None:
    with pytest.raises(PreprocessError, match="exactly one"):
        ResampleUniform()
    with pytest.raises(PreprocessError, match="exactly one"):
        ResampleUniform(step_cm1=1.0, n_points=100)


def test_a_step_does_not_change_the_origin(
    source: SourceRecord, instrument: InstrumentSpec
) -> None:
    """Preprocessing rearranges a measurement; the forward model invents part of one."""
    spectrum = _spectrum(source, origin=Origin.EXPERIMENTAL, instrument=instrument)
    assert spectrum.origin is Origin.EXPERIMENTAL
    assert SnipBaseline().apply(spectrum).origin is Origin.EXPERIMENTAL
    assert CosmicRaySpikes(count=1).apply(spectrum, seed=0).origin is Origin.AUGMENTED


def test_a_step_drops_a_stale_per_sample_sigma(source: SourceRecord) -> None:
    values = AXIS_VALUES
    spectrum = Spectrum(
        spectrum_id="with-sigma",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, values),
        intensity=np.full(values.size, 100.0),
        intensity_kind=IntensityKind.COUNTS,
        sample_id="s",
        source=source,
        sigma=np.full(values.size, 3.0),
    )
    assert MovingAverage(window=5).apply(spectrum).sigma is None


# ------------------------------------------------------------------------------------------------
# Despiking
# ------------------------------------------------------------------------------------------------
def test_despiking_removes_injected_cosmic_rays(source: SourceRecord) -> None:
    """Against ground truth from our own forward model, not against a hand-placed spike."""
    clean = _spectrum(source, noise=1.0)
    spiked = CosmicRaySpikes(count=8, amplitude_relative=8.0).apply(clean, seed=5)
    difference = np.abs(
        np.asarray(spiked.intensity, dtype=np.float64)
        - np.asarray(clean.intensity, dtype=np.float64)
    )
    injected = set(np.flatnonzero(difference > 1e-9).tolist())
    assert injected
    found = set(spike_indices(np.asarray(spiked.intensity, dtype=np.float64)))
    assert len(found & injected) >= 0.8 * len(injected), (len(found & injected), len(injected))
    cleaned = Despike().apply(spiked)
    residual = np.abs(
        np.asarray(cleaned.intensity, dtype=np.float64)
        - np.asarray(clean.intensity, dtype=np.float64)
    )
    assert residual.max() < 0.5 * difference.max()


def test_despiking_leaves_a_run_too_wide_to_be_a_spike_and_counts_it(
    source: SourceRecord,
) -> None:
    """Over-removal becomes a reported number rather than a silent deletion (F03)."""
    values = AXIS_VALUES
    intensity = np.full(values.size, 100.0) + np.random.default_rng(2).normal(0.0, 1.0, values.size)
    # Four consecutive samples: wider than a cosmic ray, which hits one or two pixels, and
    # exactly the case where a narrow crystalline band must not be deleted.
    intensity[700:704] += 500.0
    spectrum = _spectrum(source, intensity=intensity)
    step = Despike(window=11, max_width=3)
    report = step.report(spectrum)
    assert report["runs_left_as_too_wide"] >= 1.0
    assert report["spikes_replaced"] == 0.0
    cleaned = np.asarray(step.apply(spectrum).intensity)
    assert cleaned[701] == pytest.approx(intensity[701])


def test_the_audit_can_report_spikes_without_modifying_the_spectrum(
    source: SourceRecord,
) -> None:
    clean = _spectrum(source, noise=1.0)
    spiked = CosmicRaySpikes(count=6).apply(clean, seed=5)
    before = np.array(spiked.intensity, dtype=np.float64)
    report = Despike().report(spiked)
    assert report["spikes_replaced"] > 0.0
    assert np.array_equal(np.asarray(spiked.intensity, dtype=np.float64), before)


# ------------------------------------------------------------------------------------------------
# Pipeline behaviour and the search
# ------------------------------------------------------------------------------------------------
def test_a_pipeline_does_not_reorder_its_steps(source: SourceRecord) -> None:
    """Smooth-then-baseline and baseline-then-smooth are two pipelines, not one."""
    spectrum = _spectrum(source, baseline="curved", noise=5.0)
    smooth, baseline = SavitzkyGolay(window=21, order=3), SnipBaseline(iterations=40)
    forward = np.asarray(Pipeline.of(smooth, baseline).apply(spectrum).intensity)
    reverse = np.asarray(Pipeline.of(baseline, smooth).apply(spectrum).intensity)
    assert not np.allclose(forward, reverse)


def test_the_identity_pipeline_is_a_real_arm(source: SourceRecord) -> None:
    spectrum = _spectrum(source)
    unchanged = identity().apply(spectrum)
    assert np.array_equal(unchanged.intensity, spectrum.intensity)
    assert unchanged.processing == spectrum.processing
    assert identity() in defensible_pipelines()


def test_the_pipeline_label_is_stable_and_names_its_parameters() -> None:
    pipeline = Pipeline.of(SnipBaseline(iterations=40), MovingAverage(window=5))
    assert pipeline.label == Pipeline.of(SnipBaseline(iterations=40), MovingAverage(window=5)).label
    assert pipeline.label == (
        "SnipBaseline(decreasing=True,iterations=40) > MovingAverage(window=5)"
    )
    assert json.dumps(pipeline.to_dict())


def test_the_search_ranks_pipelines_with_intervals_not_point_estimates(
    source: SourceRecord,
) -> None:
    spectra = [
        _spectrum(source, spectrum_id=f"s{i}", baseline="curved", noise=3.0 + i) for i in range(6)
    ]
    results = search(
        spectra,
        score=lambda s: -float(np.abs(np.asarray(s.intensity, dtype=np.float64)).mean()),
        metric_name="negative_mean_abs",
        seed=1,
    )
    assert len(results) == len(defensible_pipelines())
    assert results[0].metric.mean >= results[-1].metric.mean
    for result in results:
        assert result.metric.n_items >= 1
        assert result.metric.ci_low <= result.metric.mean <= result.metric.ci_high
    assert json.dumps(results[0].to_dict())


def test_a_pipeline_that_fails_on_some_spectra_is_scored_on_the_rest_and_the_count_reported(
    source: SourceRecord,
) -> None:
    """Silently scoring on the easy subset is how a fragile pipeline wins a leaderboard."""
    good = _spectrum(source, spectrum_id="good")
    flat = _spectrum(source, spectrum_id="flat", intensity=np.full(AXIS_VALUES.size, 7.0))
    results = search(
        [good, flat],
        score=lambda s: float(np.asarray(s.intensity, dtype=np.float64).max()),
        pipelines=[identity(), Pipeline.of(Normalisation(NormaliseMode.MINMAX), name="minmax")],
    )
    by_label = {r.label: r for r in results}
    assert by_label["identity"].n_failed == 0
    assert by_label["minmax"].n_failed == 1
    assert "flat" in by_label["minmax"].failures[0]


def test_a_pipeline_that_fails_on_everything_is_refused_not_ranked(source: SourceRecord) -> None:
    flat = _spectrum(source, intensity=np.full(AXIS_VALUES.size, 7.0))
    with pytest.raises(SearchError, match="scored nothing"):
        search(
            [flat],
            score=lambda s: 1.0,
            pipelines=[Pipeline.of(Normalisation(NormaliseMode.MINMAX), name="minmax")],
        )


def test_an_empty_corpus_is_refused() -> None:
    with pytest.raises(SearchError, match="at least one spectrum"):
        search([], score=lambda s: 1.0)


def test_sensitivity_measures_how_much_the_answer_depends_on_the_pipeline(
    source: SourceRecord,
) -> None:
    """N8's quantity, on a real readout: band count under nine defensible pipelines."""
    from usis.measurement.diagnostics import band_statistics

    spectrum = _spectrum(source, baseline="curved", noise=4.0)
    result = sensitivity(spectrum, readout=lambda s: float(band_statistics(s)[0]))
    assert len(result.values) >= 2
    assert result.spread >= 0.0
    assert result.relative_spread >= 0.0
    assert json.dumps(result.to_dict())


def test_sensitivity_refuses_when_almost_everything_failed(source: SourceRecord) -> None:
    """Reporting zero sensitivity because nothing worked would say the opposite of what happened."""
    spectrum = _spectrum(source)
    with pytest.raises(SearchError, match="no spread to report"):
        sensitivity(
            spectrum,
            readout=lambda s: float("nan"),
            pipelines=[identity(), Pipeline.of(SnipBaseline(), name="snip")],
        )
