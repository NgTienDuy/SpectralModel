"""The five mandatory classical baselines, verified against planted ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from usis.chemometrics.decomposition import (
    DecompositionError,
    mcr_als,
    nnls,
    nnmf,
    pca,
)
from usis.chemometrics.matching import (
    MatchingError,
    SimilarityMetric,
    belief_from_matches,
    match_library,
    similarity,
)
from usis.chemometrics.matrix import MatrixError, as_matrix, class_labels, target_values
from usis.chemometrics.regression import RegressionError, one_hot, pls, pls_da_predict
from usis.core.errors import AssumptionViolationError
from usis.core.provenance import RunContext
from usis.core.types import (
    AxisKind,
    AxisSpec,
    FailureCode,
    GroundTruth,
    IntensityKind,
    Maturity,
    Modality,
    NonIdentifiability,
    Origin,
    SourceRecord,
    Spectrum,
)
from usis.methods.applicability import Verdict
from usis.methods.base import execute
from usis.methods.builtin.baseline_mcr_als import McrAls
from usis.methods.builtin.baseline_nn_match import NearestNeighbourMatch
from usis.methods.builtin.baseline_nnmf import Nnmf
from usis.methods.builtin.baseline_pca_knn import PcaKnn
from usis.methods.builtin.baseline_pls_da import PlsDiscriminantAnalysis
from usis.methods.builtin.baseline_pls_r import PlsRegression
from usis.methods.inputs import LibraryHandle, MethodInputs
from usis.methods.registry import METHODS, load_builtin_methods, select_methods

CTX = RunContext(
    run_id="test", started_at="2026-09-03T00:00:00+00:00", seed=1, git_sha="0" * 40, git_dirty="no"
)
AXIS_VALUES = np.linspace(300.0, 1700.0, 140)
AXIS = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, AXIS_VALUES, resolution_fwhm=4.0)


def _band(centre: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((AXIS_VALUES - centre) / width) ** 2)


#: Three planted pure-component spectra. Overlapping, as real components are.
PURE = np.vstack(
    [
        _band(500.0, 20.0, 1.0) + _band(900.0, 25.0, 0.4),
        _band(1100.0, 18.0, 1.0) + _band(700.0, 30.0, 0.3),
        _band(1500.0, 22.0, 1.0) + _band(400.0, 25.0, 0.5),
    ]
)


def _spectrum(
    source: SourceRecord,
    intensity: np.ndarray,
    *,
    spectrum_id: str,
    intensity_kind: IntensityKind = IntensityKind.COUNTS,
    label: GroundTruth | None = None,
    axis: AxisSpec | None = None,
) -> Spectrum:
    return Spectrum(
        spectrum_id=spectrum_id,
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        axis=axis or AXIS,
        intensity=np.asarray(intensity, dtype=np.float64),
        intensity_kind=intensity_kind,
        sample_id=f"sample-{spectrum_id}",
        source=source,
        label=label,
    )


def _mixtures(n: int = 24, noise: float = 0.002, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative mixtures of the three planted components, with their true fractions."""
    rng = np.random.default_rng(seed)
    fractions = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    data = np.clip(fractions @ PURE + rng.normal(0.0, noise, (n, AXIS_VALUES.size)), 0.0, None)
    return data, fractions


def _worst_recovery(resolved: np.ndarray, truth: np.ndarray = PURE) -> float:
    """Correlation of the *worst-recovered* planted component with its best resolved match."""
    resolved_unit = resolved / np.linalg.norm(resolved, axis=1, keepdims=True)
    truth_unit = truth / np.linalg.norm(truth, axis=1, keepdims=True)
    return min(
        float(max(abs(truth_unit[i] @ resolved_unit[j]) for j in range(resolved_unit.shape[0])))
        for i in range(truth_unit.shape[0])
    )


# ------------------------------------------------------------------------------------------------
# NNLS - the primitive everything non-negative rests on
# ------------------------------------------------------------------------------------------------
def test_nnls_matches_an_exhaustive_search_over_active_sets() -> None:
    """The strongest available check: brute force every subset and compare the residual."""
    from itertools import combinations

    rng = np.random.default_rng(0)
    worst_gap = 0.0
    for _ in range(60):
        design = rng.normal(size=(8, 4))
        target = rng.normal(size=8)
        best = np.inf
        for size in range(5):
            for subset in combinations(range(4), size):
                candidate = np.zeros(4)
                if size:
                    solution, *_ = np.linalg.lstsq(design[:, subset], target, rcond=None)
                    if np.any(solution < -1e-12):
                        continue
                    candidate[list(subset)] = np.maximum(solution, 0.0)
                best = min(best, float(np.linalg.norm(design @ candidate - target)))
        ours = nnls(design, target)
        assert np.all(ours >= 0.0)
        worst_gap = max(worst_gap, abs(float(np.linalg.norm(design @ ours - target)) - best))
    assert worst_gap < 1e-10, worst_gap


def test_nnls_returns_the_boundary_solution_when_the_free_optimum_is_negative() -> None:
    design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = np.array([2.0, -1.0, 1.0])
    assert nnls(design, target) == pytest.approx(np.array([1.5, 0.0]))


def test_nnls_rejects_mismatched_shapes() -> None:
    with pytest.raises(DecompositionError, match="nnls needs"):
        nnls(np.zeros((3, 2)), np.zeros(4))


# ------------------------------------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------------------------------------
def test_pca_recovers_a_planted_subspace_one_rank_short_because_of_closure() -> None:
    """Three components, but the residual collapses at **two** - and that is closure, not a bug.

    The planted fractions are Dirichlet, so they sum to one, so the mixtures lie in a 2-dimensional
    affine subspace of the 3-component space. Mean-centring removes the offset and PCA needs one
    fewer component than there are substances. Anyone reading a scree plot to count components on a
    closed system is off by one, every time.
    """
    data, _ = _mixtures(30)
    residuals = {k: float(np.median(pca(data, k).residual_norm(data))) for k in (1, 2, 3)}
    assert residuals[2] < 0.1 * residuals[1], residuals
    assert residuals[3] > 0.5 * residuals[2], residuals
    assert float(pca(data, 2).explained_variance_ratio.sum()) > 0.99


def test_pca_refuses_more_components_than_the_data_can_support() -> None:
    with pytest.raises(DecompositionError, match="exceeds the rank"):
        pca(np.zeros((4, 10)), 5)


def test_pca_refuses_non_finite_input() -> None:
    data = np.zeros((4, 10))
    data[0, 0] = np.nan
    with pytest.raises(DecompositionError, match="non-finite"):
        pca(data, 2)


def test_pca_refuses_to_project_onto_a_different_channel_count() -> None:
    model = pca(_mixtures(10)[0], 2)
    with pytest.raises(DecompositionError, match="fitted on"):
        model.transform(np.zeros((3, 7)))


# ------------------------------------------------------------------------------------------------
# NNMF and MCR-ALS
# ------------------------------------------------------------------------------------------------
def test_nnmf_recovers_planted_components() -> None:
    """Cited as the reproduction evidence for baseline.nnmf."""
    data, _ = _mixtures(24)
    result = nnmf(data, 3, seed=1)
    assert _worst_recovery(result.spectra) > 0.99
    assert result.relative_residual < 0.05
    assert np.all(result.spectra >= 0.0) and np.all(result.concentrations >= 0.0)


def test_mcr_als_recovers_planted_components() -> None:
    """Cited as the reproduction evidence for baseline.mcr_als."""
    data, fractions = _mixtures(24)
    result = mcr_als(data, 3, seed=1)
    assert _worst_recovery(result.spectra) > 0.99
    assert result.relative_residual < 0.05
    assert np.all(result.spectra >= 0.0) and np.all(result.concentrations >= 0.0)
    # Concentrations recover the planted fractions up to per-component scale and permutation.
    scaled = result.concentrations / result.concentrations.sum(axis=1, keepdims=True)
    best = max(
        float(np.abs(np.corrcoef(fractions[:, i], scaled[:, j])[0, 1]))
        for i in range(3)
        for j in range(3)
    )
    assert best > 0.95


def test_mcr_als_closure_makes_the_concentrations_sum_to_one() -> None:
    data, _ = _mixtures(20)
    result = mcr_als(data, 2, seed=1, closure=True)
    assert result.concentrations.sum(axis=1) == pytest.approx(
        np.ones(result.concentrations.shape[0]), abs=1e-9
    )


def test_known_pure_spectra_are_held_fixed(source: SourceRecord) -> None:
    """Supplying a known component turns MCR-ALS toward classical least squares."""
    data, _ = _mixtures(20)
    result = mcr_als(data, 3, seed=1, known_spectra=PURE[:1])
    fixed = result.spectra[0]
    assert fixed == pytest.approx(PURE[0])


def test_the_ambiguity_bound_reports_the_number_of_comparable_starts() -> None:
    data, _ = _mixtures(20)
    single = mcr_als(data, 2, seed=1, n_starts=1)
    assert single.n_starts == 1
    assert np.isnan(single.ambiguity), "a bound from one start is not a bound"
    several = mcr_als(data, 2, seed=1, n_starts=3)
    assert several.n_starts >= 1
    assert np.isfinite(several.ambiguity) or several.n_starts == 1


def test_both_decompositions_refuse_negative_data() -> None:
    """The guard that a real over-subtracted baseline trips, and it names where to look."""
    data, _ = _mixtures(10)
    data[0, 0] = -1.0
    with pytest.raises(DecompositionError, match="processing chain"):
        nnmf(data, 2)
    with pytest.raises(DecompositionError, match="negative entries"):
        mcr_als(data, 2)


# ------------------------------------------------------------------------------------------------
# PLS
# ------------------------------------------------------------------------------------------------
def test_pls_recovers_a_planted_linear_relation() -> None:
    """Cited as the reproduction evidence for baseline.pls_r."""
    rng = np.random.default_rng(0)
    concentrations = rng.uniform(0.1, 1.0, (60, 2))
    data = concentrations @ PURE[:2] + rng.normal(0.0, 0.01, (60, AXIS_VALUES.size))
    response = concentrations[:, :1]
    model = pls(data, response, 3)
    predicted = model.predict(data)
    r2 = 1.0 - float(((response - predicted) ** 2).sum()) / float(
        ((response - response.mean()) ** 2).sum()
    )
    assert r2 > 0.999
    assert float(model.y_variance_explained[0]) > 0.9


def test_pls_da_separates_planted_classes() -> None:
    """Cited as the reproduction evidence for baseline.pls_da."""
    rng = np.random.default_rng(0)
    labels = ["alpha", "beta", "gamma"] * 20
    centres = {"alpha": 500.0, "beta": 900.0, "gamma": 1400.0}
    data = np.vstack([_band(centres[label], 25.0, 1.0) for label in labels]) + rng.normal(
        0.0, 0.05, (60, AXIS_VALUES.size)
    )
    indicator, classes = one_hot(labels)
    model = pls(data, indicator, 3)
    predicted, scores = pls_da_predict(model, data, classes)
    assert predicted == labels
    # The scores are not probabilities, and the test says so rather than the docstring alone.
    assert float(scores.max()) > 1.0 or float(scores.min()) < 0.0


def test_pls_names_the_latent_variables_that_buy_no_response_variance() -> None:
    """Asked for four latent variables on a two-direction problem, it names the two that are noise.

    Note what this test does *not* find, which is why the diagnostic was rewritten: PLS never
    leaves a large-X, zero-y component lying around, because NIPALS orders by covariance and
    absorbs the interferent into the early components. The detectable overfit is components that
    contribute nothing at all.
    """
    rng = np.random.default_rng(3)
    concentrations = rng.uniform(0.1, 1.0, (60, 1))
    interferent = rng.uniform(0.0, 6.0, (60, 1)) @ PURE[1:2]
    data = concentrations @ PURE[:1] + interferent + rng.normal(0.0, 0.01, (60, AXIS_VALUES.size))
    assert pls(data, concentrations, 2).capacity_warning == ""
    warning = pls(data, concentrations, 4).capacity_warning
    assert "2 of 4 latent variables" in warning
    assert "#3" in warning and "#4" in warning


def test_pls_refuses_mismatched_sample_counts() -> None:
    with pytest.raises(RegressionError, match="samples"):
        pls(np.zeros((10, 5)), np.zeros((9, 1)), 2)


def test_one_hot_refuses_an_undeclared_label() -> None:
    with pytest.raises(RegressionError, match="not among the declared classes"):
        one_hot(["a", "b"], classes=["a"])


# ------------------------------------------------------------------------------------------------
# Library matching
# ------------------------------------------------------------------------------------------------
def _library() -> tuple[np.ndarray, list[str]]:
    entries = np.vstack(
        [
            _band(400.0 + 80.0 * i, 20.0, 1.0) + _band(1200.0 - 50.0 * i, 25.0, 0.5)
            for i in range(12)
        ]
    )
    return entries, [f"entry-{i:02d}" for i in range(12)]


def test_nn_match_retrieves_the_planted_entry(source: SourceRecord) -> None:
    """Cited as the reproduction evidence for baseline.nn_match."""
    entries, keys = _library()
    rng = np.random.default_rng(0)
    query = _spectrum(
        source, entries[5] + rng.normal(0.0, 0.02, AXIS_VALUES.size), spectrum_id="q-in"
    )
    result = execute(
        NearestNeighbourMatch(),
        MethodInputs.of(
            query,
            library=LibraryHandle(library_id="planted", n_entries=12, modality=Modality.RAMAN),
            aux={"library_spectra": entries, "library_keys": keys},
        ),
        CTX,
    )
    assert result.belief is not None
    top = result.belief.top
    assert top is not None and top.name == "entry-05"
    assert result.belief.p_none < 0.05
    assert result.values["best_similarity"] > 0.99
    assert FailureCode.F12_RETRIEVAL_MISS not in result.all_failure_codes


def test_nn_match_puts_the_mass_on_p_none_for_an_absent_query(source: SourceRecord) -> None:
    """Cited as the reproduction evidence for baseline.nn_match: the open-set case."""
    entries, keys = _library()
    absent = _spectrum(
        source, _band(1650.0, 5.0, 1.0) + _band(1655.0, 5.0, 1.0), spectrum_id="q-out"
    )
    result = execute(
        NearestNeighbourMatch(),
        MethodInputs.of(
            absent,
            library=LibraryHandle(library_id="planted", n_entries=12, modality=Modality.RAMAN),
            aux={"library_spectra": entries, "library_keys": keys},
        ),
        CTX,
    )
    assert result.belief is not None
    assert result.belief.p_none > 0.5
    assert NonIdentifiability.LIBRARY_GAP in result.belief.non_identifiability
    assert FailureCode.F12_RETRIEVAL_MISS in result.all_failure_codes


def test_the_belief_invariant_holds_for_every_metric() -> None:
    entries, keys = _library()
    rng = np.random.default_rng(1)
    query = entries[3] + rng.normal(0.0, 0.05, AXIS_VALUES.size)
    for metric in SimilarityMetric:
        belief = belief_from_matches(match_library(query, entries, keys, metric=metric, top_k=4))
        total = sum(c.posterior for c in belief.candidates) + belief.p_none
        assert total == pytest.approx(1.0)


def test_every_belief_states_that_its_p_none_is_uncalibrated() -> None:
    entries, keys = _library()
    belief = belief_from_matches(match_library(entries[0], entries, keys, top_k=3))
    assert any("UNCALIBRATED" in question for question in belief.unresolved_questions)
    assert any("N10" in question for question in belief.unresolved_questions)


def test_correlation_is_insensitive_to_an_additive_offset() -> None:
    """Which is why it, not cosine, is the default: baseline treatment differs between labs."""
    entries, _ = _library()
    offset = entries[0] + 5.0
    assert similarity(entries[0], offset, SimilarityMetric.CORRELATION) == pytest.approx(1.0)
    assert similarity(entries[0], offset, SimilarityMetric.COSINE) < 0.9


def test_matching_refuses_a_flat_query_and_a_mismatched_axis() -> None:
    entries, keys = _library()
    with pytest.raises(MatchingError, match="zero norm"):
        similarity(np.ones(AXIS_VALUES.size), entries[0], SimilarityMetric.CORRELATION)
    with pytest.raises(MatchingError, match="common axis"):
        similarity(np.ones(7), entries[0])
    with pytest.raises(MatchingError, match="keys were supplied"):
        match_library(entries[0], entries, keys[:3])


# ------------------------------------------------------------------------------------------------
# The matrix helpers
# ------------------------------------------------------------------------------------------------
def test_stacking_refuses_spectra_on_different_axes(source: SourceRecord) -> None:
    shifted = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, AXIS_VALUES + 3.0)
    a = _spectrum(source, PURE[0], spectrum_id="a")
    b = _spectrum(source, PURE[1], spectrum_id="b", axis=shifted)
    assert as_matrix([a, a]).shape == (2, AXIS_VALUES.size)
    with pytest.raises(MatrixError, match="different axis"):
        as_matrix([a, b])


def test_labels_and_targets_refuse_a_missing_entry(source: SourceRecord) -> None:
    labelled = _spectrum(
        source,
        PURE[0],
        spectrum_id="labelled",
        label=GroundTruth(name="alpha", targets={"concentration": 0.5}),
    )
    unlabelled = _spectrum(source, PURE[1], spectrum_id="bare")
    assert class_labels([labelled]) == ["alpha"]
    assert target_values([labelled], "concentration") == pytest.approx(np.array([[0.5]]))
    with pytest.raises(MatrixError, match=r"no label\.name"):
        class_labels([labelled, unlabelled])
    with pytest.raises(MatrixError, match=r"label\.targets"):
        target_values([labelled], "purity")


# ------------------------------------------------------------------------------------------------
# The registered methods
# ------------------------------------------------------------------------------------------------
def _classification_set(source: SourceRecord) -> tuple[list[Spectrum], list[Spectrum], list[str]]:
    rng = np.random.default_rng(0)
    names = ["alpha", "beta", "gamma"]
    calibration, queries, truth = [], [], []
    for index in range(30):
        name = names[index % 3]
        centre = {"alpha": 500.0, "beta": 900.0, "gamma": 1400.0}[name]
        calibration.append(
            _spectrum(
                source,
                _band(centre, 25.0, 1.0) + rng.normal(0.0, 0.05, AXIS_VALUES.size),
                spectrum_id=f"cal-{index:02d}",
                label=GroundTruth(name=name),
            )
        )
    for index, name in enumerate(names):
        centre = {"alpha": 500.0, "beta": 900.0, "gamma": 1400.0}[name]
        queries.append(
            _spectrum(
                source,
                _band(centre, 25.0, 1.0) + rng.normal(0.0, 0.05, AXIS_VALUES.size),
                spectrum_id=f"query-{index}",
            )
        )
        truth.append(name)
    return calibration, queries, truth


def test_pca_knn_recovers_planted_classes(source: SourceRecord) -> None:
    """Cited as the reproduction evidence for baseline.pca_knn."""
    calibration, queries, truth = _classification_set(source)
    result = execute(
        PcaKnn(n_components=3, k=3),
        MethodInputs(spectra=tuple(queries), calibration=tuple(calibration)),
        CTX,
    )
    predicted = [result.labels[spectrum.spectrum_id] for spectrum in queries]
    assert predicted == truth
    assert result.values["n_classes"] == 3.0
    assert result.values["median_margin"] > 0.0


def test_pls_da_as_a_registered_method_classifies_and_reports_uncalibrated_scores(
    source: SourceRecord,
) -> None:
    calibration, queries, truth = _classification_set(source)
    result = execute(
        PlsDiscriminantAnalysis(n_components=3),
        MethodInputs(spectra=tuple(queries), calibration=tuple(calibration)),
        CTX,
    )
    assert [result.labels[s.spectrum_id] for s in queries] == truth
    assert "not a probability" in result.evidence[0].statement


def test_pls_r_is_always_degraded_because_its_central_assumption_is_uncheckable(
    source: SourceRecord,
) -> None:
    """The intended behaviour: an unverified Beer-Lambert claim cannot look clean."""
    rng = np.random.default_rng(0)
    calibration = [
        _spectrum(
            source,
            float(value) * PURE[0] + rng.normal(0.0, 0.01, AXIS_VALUES.size),
            spectrum_id=f"cal-{index}",
            intensity_kind=IntensityKind.ABSORBANCE,
            label=GroundTruth(targets={"concentration": float(value)}),
        )
        for index, value in enumerate(np.linspace(0.1, 1.0, 20))
    ]
    query = _spectrum(
        source,
        0.5 * PURE[0],
        spectrum_id="query",
        intensity_kind=IntensityKind.ABSORBANCE,
    )
    result = execute(
        PlsRegression(target="concentration", n_components=2),
        MethodInputs(spectra=(query,), calibration=tuple(calibration)),
        CTX,
    )
    assert result.applicability.verdict is Verdict.RUNNABLE_DEGRADED
    assert not result.surfaceable
    assert any("concentration series" in need for need in result.applicability.evidence_needed)
    assert result.values["calibration_r2"] > 0.99
    assert result.values["predicted_query"] == pytest.approx(0.5, abs=0.05)


def test_pls_r_refuses_transmittance_because_beer_lambert_is_not_additive_there(
    source: SourceRecord,
) -> None:
    query = _spectrum(source, PURE[0], spectrum_id="t", intensity_kind=IntensityKind.TRANSMITTANCE)
    with pytest.raises(AssumptionViolationError, match="requires_intensity_kind"):
        execute(PlsRegression(), MethodInputs.of(query), CTX)


def test_the_mixture_methods_report_an_ambiguity_bound_and_read_it_honestly(
    source: SourceRecord,
) -> None:
    data, _ = _mixtures(20)
    spectra = tuple(
        _spectrum(source, row, spectrum_id=f"mix-{index:02d}") for index, row in enumerate(data)
    )
    for method in (McrAls(n_components=3), Nnmf(n_components=3)):
        result = execute(method, MethodInputs(spectra=spectra), CTX)
        assert "ambiguity_lower_bound" in result.values
        statement = result.evidence[0].statement
        assert "ambiguity" in statement or "found no ambiguity" in statement
        assert result.values["relative_residual"] < 0.1


def test_a_negative_intensity_is_a_named_violation_not_a_crash(source: SourceRecord) -> None:
    """The assumption layer catches what would otherwise raise inside the decomposition."""
    data, _ = _mixtures(6)
    data[0, 0] = -0.5
    spectra = tuple(
        _spectrum(source, row, spectrum_id=f"neg-{index}") for index, row in enumerate(data)
    )
    with pytest.raises(AssumptionViolationError, match="requires_non_negative_intensity"):
        execute(McrAls(n_components=2), MethodInputs(spectra=spectra), CTX)


def test_mixed_axes_are_refused_by_the_assumption_not_by_the_algorithm(
    source: SourceRecord,
) -> None:
    shifted = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, AXIS_VALUES + 3.0)
    spectra = (
        _spectrum(source, PURE[0], spectrum_id="a"),
        _spectrum(source, PURE[1], spectrum_id="b", axis=shifted),
    )
    with pytest.raises(AssumptionViolationError, match="requires_common_axis"):
        execute(Nnmf(n_components=2), MethodInputs(spectra=spectra), CTX)


def test_the_five_mandatory_baselines_are_registered_across_the_task_families() -> None:
    load_builtin_methods()
    keys = {entry.key for entry in select_methods(maturities=[Maturity.ESTABLISHED])}
    assert {
        "baseline.pca_knn@1.0.0",
        "baseline.pls_r@1.0.0",
        "baseline.pls_da@1.0.0",
        "baseline.mcr_als@1.0.0",
        "baseline.nnmf@1.0.0",
        "baseline.nn_match@1.0.0",
    } <= keys
    tasks = {METHODS.get(key).method.task for key in keys if key.startswith("baseline.")}
    assert len(tasks) == 4, tasks


def test_every_cited_reproduction_reference_points_at_a_test_that_exists() -> None:
    """A maturity claim resting on a reference nobody can follow is not evidence."""
    import pathlib
    import re

    load_builtin_methods()
    for key in METHODS:
        entry = METHODS.get(key)
        for reference in entry.evidence:
            if "::" not in reference:
                continue
            path, node = reference.split("::", 1)
            source_file = pathlib.Path(path)
            assert source_file.exists(), f"{key} cites missing file {path}"
            text = source_file.read_text(encoding="utf-8")
            assert re.search(rf"^def {re.escape(node)}\(", text, re.MULTILINE), (
                f"{key} cites {reference}, which does not exist"
            )
