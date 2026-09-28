"""ESTABLISHED baseline: library matching on the spectral contrast angle.

The third of the three metrics :mod:`usis.chemometrics.matching` implements, registered as its own
method for the reason recorded in `PROJECT_DECISIONS.md` D17. Its distinguishing property is
resolution rather than accuracy: cosine similarities between real spectra crowd into the top few
percent of their range, where the differences that decide a retrieval are smaller than the width of
the number being compared. Rescaling the angle spreads that region out.

The rescaling is monotone in the cosine, so it cannot reorder a candidate list: the *ranking* of
library entries is identical to ``baseline.nn_match_cosine`` on every query, and a difference in
mean reciprocal rank between the two would be a bug rather than a result. Anything else that
differs between them therefore isolates one thing - what the open-set decision does with the
similarity, since ``p_none = 1 - best`` reads the number itself and not its rank.

That is not a null prediction, and the first run on ROD (``experiments/M2-rod-open-set-retrieval``)
made it concrete: identical ranking, identical reciprocal rank to six decimals, and a selective
accuracy 0.05 higher, bought by declaring a library gap on 324 queries instead of 142 - 78 fewer
hallucinated assignments for 39 more missed retrievals. The metric did not retrieve better. It
abstained better, on a threshold nobody set.
"""

from __future__ import annotations

from usis.chemometrics.matching import SimilarityMetric
from usis.methods.base import Citation
from usis.methods.builtin.baseline_nn_match import NearestNeighbourMatch
from usis.methods.registry import register_method

__all__ = ["SpectralContrastAngleMatch"]


class SpectralContrastAngleMatch(NearestNeighbourMatch):
    """Library matching scored by the spectral contrast angle rescaling of the cosine."""

    method_id = "baseline.nn_match_sca"
    version = "1.0.0"
    provenance = Citation(
        title="Library search by spectral contrast angle",
        internal_ref="docs/knowledge/chemometrics.md#the-methods-and-what-each-assumes",
        note=(
            "A monotone rescaling of the cosine that spreads the crowded region near one. The "
            "referenced reproduction is the shared planted-truth retrieval test under this metric."
        ),
    )

    def __init__(self, top_k: int = 5) -> None:
        super().__init__(metric=SimilarityMetric.SPECTRAL_CONTRAST_ANGLE, top_k=top_k)


register_method(
    SpectralContrastAngleMatch(),
    evidence=("tests/test_compare_tasks.py::test_every_matching_metric_retrieves_planted_entry",),
    tags=("baseline", "chemometrics", "retrieval"),
    notes=(
        "ESTABLISHED under PROJECT_DECISIONS.md D17. Monotone in the cosine, so it cannot reorder "
        "candidates: a difference in top-1 accuracy against baseline.nn_match_cosine would be a "
        "bug, not a result. What it can change is the calibration of the confidence, and with it "
        "the risk-coverage frontier."
    ),
)
