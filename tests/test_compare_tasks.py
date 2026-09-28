"""The open-set retrieval benchmark, the grid resampler it needs, and the corpus loader."""

from __future__ import annotations

import numpy as np
import pytest

from usis.compare.build import BuildError, build_pipeline, build_task
from usis.compare.harness import run_comparison
from usis.compare.scoring import Outcome
from usis.compare.task import TaskError
from usis.core.config import ExperimentConfig
from usis.core.errors import UsisError
from usis.core.provenance import RunContext
from usis.core.types import (
    AxisKind,
    AxisSpec,
    FailureCode,
    GroundTruth,
    IntensityKind,
    Modality,
    NonIdentifiability,
    Origin,
    SourceRecord,
    Spectrum,
)
from usis.data.corpus import load_corpus
from usis.eval.tasks.retrieval import NONE_KEY, OpenSetRetrievalTask, identity_key
from usis.methods.base import execute
from usis.methods.builtin.baseline_nn_match import NearestNeighbourMatch
from usis.methods.builtin.baseline_nn_match_cosine import CosineLibraryMatch
from usis.methods.builtin.baseline_nn_match_sca import SpectralContrastAngleMatch
from usis.methods.inputs import LibraryHandle, MethodInputs
from usis.preprocess.base import PreprocessError
from usis.preprocess.baseline import SnipBaseline
from usis.preprocess.pipeline import Pipeline, verify_reproduction
from usis.preprocess.scale import ResampleToGrid

CTX = RunContext(
    run_id="test", started_at="2026-09-11T00:00:00+00:00", seed=5, git_sha="0" * 40, git_dirty="no"
)
SOURCE = SourceRecord(dataset_id="fixture", licence="CC0-1.0", retrieved_at="2026-09-11T00:00:00Z")
GRID = np.linspace(300.0, 1500.0, 121)
AXIS = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, GRID, resolution_fwhm=6.0)


def _band(axis: np.ndarray, centre: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((axis - centre) / width) ** 2)


def _pure(index: int, axis: np.ndarray = GRID) -> np.ndarray:
    """A distinct two-band pattern per identity, well separated so retrieval is decidable."""
    return _band(axis, 350.0 + 55.0 * index, 18.0, 1.0) + _band(
        axis, 1450.0 - 45.0 * index, 22.0, 0.55
    )


def _spectrum(
    intensity: np.ndarray,
    *,
    spectrum_id: str,
    name: str | None,
    axis: AxisSpec = AXIS,
) -> Spectrum:
    return Spectrum(
        spectrum_id=spectrum_id,
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=axis,
        intensity=np.asarray(intensity, dtype=np.float64),
        intensity_kind=IntensityKind.COUNTS,
        sample_id=f"sample-{spectrum_id}",
        source=SOURCE,
        label=None if name is None else GroundTruth(name=name),
    )


def _corpus(n_identities: int = 12, replicates: int = 3, seed: int = 0) -> list[Spectrum]:
    rng = np.random.default_rng(seed)
    out = []
    for identity in range(n_identities):
        for replicate in range(replicates):
            noise = rng.normal(0.0, 0.01, GRID.size)
            out.append(
                _spectrum(
                    _pure(identity) + noise,
                    spectrum_id=f"id{identity:02d}-r{replicate}",
                    name=f"compound-{identity:02d}",
                )
            )
    return out


# ----------------------------------------------------------------------------------------------
# The similarity metrics, as registered methods
# ----------------------------------------------------------------------------------------------
def test_every_matching_metric_retrieves_planted_entry() -> None:
    """Cited as the reproduction evidence for baseline.nn_match_cosine and _sca.

    The same planted-truth retrieval the correlation metric is held to, run under each registered
    metric: the library entry the query was built from must come back at rank 1 with little
    open-set mass left over.
    """
    entries = np.vstack([_pure(i) for i in range(10)])
    keys = [f"compound-{i:02d}" for i in range(10)]
    rng = np.random.default_rng(1)
    query = _spectrum(
        _pure(6) + rng.normal(0.0, 0.02, GRID.size), spectrum_id="q", name="compound-06"
    )
    inputs = MethodInputs.of(
        query,
        library=LibraryHandle(library_id="planted", n_entries=10, modality=Modality.RAMAN),
        aux={"library_spectra": entries, "library_keys": keys},
    )
    for method in (NearestNeighbourMatch(), CosineLibraryMatch(), SpectralContrastAngleMatch()):
        result = execute(method, inputs, CTX)
        assert result.belief is not None, method.method_key
        top = result.belief.top
        assert top is not None and top.name == "compound-06", method.method_key
        assert result.belief.p_none < 0.2, method.method_key
        assert FailureCode.F12_RETRIEVAL_MISS not in result.all_failure_codes, method.method_key


def test_every_matching_metric_puts_the_mass_on_p_none_for_an_absent_query() -> None:
    entries = np.vstack([_pure(i) for i in range(10)])
    keys = [f"compound-{i:02d}" for i in range(10)]
    absent = _spectrum(_band(GRID, 900.0, 6.0, 1.0), spectrum_id="q-out", name="absent")
    inputs = MethodInputs.of(
        absent,
        library=LibraryHandle(library_id="planted", n_entries=10, modality=Modality.RAMAN),
        aux={"library_spectra": entries, "library_keys": keys},
    )
    for method in (NearestNeighbourMatch(), CosineLibraryMatch(), SpectralContrastAngleMatch()):
        result = execute(method, inputs, CTX)
        assert result.belief is not None
        assert NonIdentifiability.LIBRARY_GAP in result.belief.non_identifiability, (
            method.method_key
        )


def test_the_contrast_angle_cannot_reorder_the_cosine_candidate_list() -> None:
    """A monotone rescaling changes confidence, never rank. The registration says so; check it."""
    entries = np.vstack([_pure(i) for i in range(10)])
    keys = [f"compound-{i:02d}" for i in range(10)]
    rng = np.random.default_rng(2)
    for trial in range(8):
        # A contested query: a mixture of two library patterns, so the ranking below rank 1
        # is genuinely decided by the metric rather than by one obvious winner.
        a, b = trial % 10, (trial + 3) % 10
        query = _spectrum(
            5.0 * (0.6 * _pure(a) + 0.4 * _pure(b)) + rng.normal(0.0, 0.15, GRID.size),
            spectrum_id=f"q{trial}",
            name="whatever",
        )
        inputs = MethodInputs.of(
            query,
            library=LibraryHandle(library_id="planted", n_entries=10, modality=Modality.RAMAN),
            aux={"library_spectra": entries, "library_keys": keys},
        )
        cosine = execute(CosineLibraryMatch(), inputs, CTX).belief
        angle = execute(SpectralContrastAngleMatch(), inputs, CTX).belief
        assert cosine is not None and angle is not None
        order_cosine = [c.name for c in sorted(cosine.candidates, key=lambda c: -c.posterior)]
        order_angle = [c.name for c in sorted(angle.candidates, key=lambda c: -c.posterior)]
        assert len(order_cosine) > 1, "a single candidate cannot demonstrate an ordering"
        assert order_cosine == order_angle, f"trial {trial}: the rescaling reordered the list"


# ----------------------------------------------------------------------------------------------
# The benchmark's construction
# ----------------------------------------------------------------------------------------------
def _task(**kwargs: object) -> OpenSetRetrievalTask:
    return OpenSetRetrievalTask.from_spectra(
        _corpus(), task_id="t", dataset_ids=("fixture",), seed=11, **kwargs
    )


def test_the_library_holds_one_spectrum_per_identity_and_never_the_query_itself() -> None:
    task = _task()
    assert len(task.library_keys) == len(set(task.library_keys))
    library_ids = {s.spectrum_id for s in task.library_spectra}
    assert not library_ids & {item.item_id for item in task.items()}


def test_held_out_identities_become_queries_whose_answer_is_none_of_these() -> None:
    task = _task()
    negatives = [i for i in task.items() if i.reference == NONE_KEY]
    positives = [i for i in task.items() if i.reference != NONE_KEY]
    assert negatives and positives
    for item in negatives:
        assert identity_key(item.inputs.primary) not in set(task.library_keys)
    for item in positives:
        assert item.reference in set(task.library_keys)


def test_every_item_carries_an_independent_reference_so_none_is_unscorable() -> None:
    assert all(item.is_scorable for item in _task().items())


def test_the_benchmark_refuses_a_corpus_on_mixed_axes() -> None:
    corpus = _corpus(n_identities=4)
    short_axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, GRID[:80])
    corpus.append(_spectrum(_pure(0, GRID[:80]), spectrum_id="odd", name="odd", axis=short_axis))
    with pytest.raises(TaskError, match="channels"):
        OpenSetRetrievalTask.from_spectra(corpus, task_id="t", dataset_ids=("fixture",), seed=11)


def test_the_benchmark_refuses_a_corpus_with_no_second_record_per_identity() -> None:
    singles = [
        _spectrum(_pure(i), spectrum_id=f"only{i}", name=f"compound-{i:02d}") for i in range(8)
    ]
    with pytest.raises(TaskError, match="no query of a present substance"):
        OpenSetRetrievalTask.from_spectra(singles, task_id="t", dataset_ids=("fixture",), seed=11)


def test_an_unlabelled_spectrum_has_no_identity_and_is_refused_rather_than_given_an_empty_key() -> (
    None
):
    with pytest.raises(TaskError, match="carries no identity"):
        identity_key(_spectrum(_pure(0), spectrum_id="x", name=None))


# ----------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------
def test_a_missed_present_substance_is_f12_and_a_named_absent_one_is_f10() -> None:
    task = _task()
    record = run_comparison(
        task,
        CTX,
        bootstrap_samples=200,
        shows=("planted retrieval",),
    )
    row = record.row("baseline.nn_match@1.0.0")
    assert row is not None
    by_id = {s.item_id: s for s in row.scores}
    for item in task.items():
        score = by_id[item.item_id]
        assert score.outcome in (Outcome.CORRECT, Outcome.INCORRECT)
        if score.outcome is Outcome.INCORRECT:
            expected = (
                FailureCode.F10_HALLUCINATED_ASSIGNMENT
                if item.reference == NONE_KEY
                else FailureCode.F12_RETRIEVAL_MISS
            )
            assert expected in score.failure_codes


def test_on_planted_data_the_matchers_retrieve_nearly_everything_present() -> None:
    """A sanity floor: if this drops, the benchmark is broken, not the methods."""
    record = run_comparison(_task(), CTX, bootstrap_samples=200, shows=("planted retrieval",))
    for row in record.rows:
        positives = [s for s in row.scores if s.reference not in (None, NONE_KEY)]
        hits = sum(1 for s in positives if s.outcome is Outcome.CORRECT)
        assert hits / len(positives) > 0.9, row.method_key


def test_the_split_caveat_travels_onto_the_record() -> None:
    record = run_comparison(_task(), CTX, bootstrap_samples=200, shows=("x",))
    assert record.split is not None
    assert any("open_set holds out whole identities" in note for note in record.split.notes)
    assert any("UNCALIBRATED" in note for note in record.does_not_show)


# ----------------------------------------------------------------------------------------------
# ResampleToGrid: the step that makes a common axis possible
# ----------------------------------------------------------------------------------------------
def _ragged(spectrum_id: str, low: float, high: float, n: int) -> Spectrum:
    axis_values = np.linspace(low, high, n)
    axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, axis_values)
    return _spectrum(_pure(0, axis_values), spectrum_id=spectrum_id, name="c", axis=axis)


def test_resample_to_grid_puts_every_spectrum_on_the_identical_axis() -> None:
    step = ResampleToGrid(low_cm1=400.0, high_cm1=1400.0, step_cm1=5.0)
    a = step.apply(_ragged("a", 310.0, 1490.0, 512))
    b = step.apply(_ragged("b", 290.7, 1502.3, 991))
    assert np.array_equal(a.axis.values, b.axis.values)
    assert a.axis.values[0] == 400.0
    assert a.axis.n_points == 201


def test_resample_to_grid_refuses_to_extrapolate_past_the_measurement() -> None:
    step = ResampleToGrid(low_cm1=200.0, high_cm1=1400.0, step_cm1=5.0)
    with pytest.raises(PreprocessError, match="never observed"):
        step.apply(_ragged("a", 310.0, 1490.0, 512))


def test_resample_to_grid_rejects_a_grid_of_fewer_than_two_points() -> None:
    with pytest.raises(PreprocessError, match="fewer than two points"):
        ResampleToGrid(low_cm1=400.0, high_cm1=401.0, step_cm1=5.0)


def test_the_grid_step_replays_exactly_from_its_own_record() -> None:
    """The C4 guarantee, extended to the step this benchmark depends on."""
    original = _ragged("a", 310.0, 1490.0, 512)
    pipeline = Pipeline.of(ResampleToGrid(400.0, 1400.0, 5.0), SnipBaseline())
    processed = pipeline.apply(original)
    reproduction = verify_reproduction(original, processed)
    assert reproduction.reproduced, reproduction.detail
    assert reproduction.max_abs_deviation == 0.0


# ----------------------------------------------------------------------------------------------
# The corpus loader and the config-driven builder
# ----------------------------------------------------------------------------------------------
def test_load_corpus_refuses_a_dataset_that_was_never_mirrored(tmp_path) -> None:
    with pytest.raises(UsisError, match="usis data pull"):
        load_corpus("rod", root=tmp_path)


def test_load_corpus_itemises_what_it_could_not_parse(tmp_path) -> None:
    directory = tmp_path / "rod"
    directory.mkdir(parents=True)
    (directory / "broken.rod").write_text("not a CIF at all", encoding="utf-8")
    load = load_corpus("rod", root=tmp_path)
    assert load.n_files == 1
    assert load.n_skipped == 1
    assert load.spectra == ()
    assert load.to_dict()["skip_reasons"]


def _config(**overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "name": "t",
        "seed": 3,
        "split_kind": "open_set",
        "task": {"kind": "open_set_retrieval", "id": "t"},
        "dataset": {"id": "rod"},
        "window": {"low_cm1": 300.0, "high_cm1": 1500.0, "step_cm1": 10.0},
        "preprocess": {"baseline": "snip", "normalise": "none"},
    }
    values.update(overrides)
    return ExperimentConfig(name="t", seed=3, split_kind="open_set", values=values)


def test_the_pipeline_a_config_declares_starts_with_the_absolute_grid() -> None:
    pipeline = build_pipeline(_config())
    assert isinstance(pipeline.steps[0], ResampleToGrid)
    assert "ResampleToGrid" in pipeline.label
    assert "SnipBaseline" in pipeline.label


def test_a_config_naming_an_unknown_baseline_is_refused() -> None:
    with pytest.raises(BuildError, match=r"preprocess\.baseline"):
        build_pipeline(_config(preprocess={"baseline": "wavelet"}))


def test_a_config_naming_an_unregistered_benchmark_lists_what_exists() -> None:
    with pytest.raises(BuildError, match="open_set_retrieval"):
        build_task(_config(task={"kind": "telepathy"}))
