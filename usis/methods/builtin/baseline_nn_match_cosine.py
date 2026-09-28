"""ESTABLISHED baseline: library matching on the cosine (dot-product) similarity.

Not a hyperparameter of :class:`~usis.methods.builtin.baseline_nn_match.NearestNeighbourMatch`
wearing a second name. The similarity function *is* the method in library search: the cosine
between raw intensity vectors is what commercial library systems report as a hit-quality index, and
the mean-centred correlation used by ``baseline.nn_match`` is a different method that different
software ships. Which one is better under open-set conditions on real data is the sort of question
the field settles by convention, so registering them separately is what lets the harness settle it
by measurement instead.

The documented difference is a prediction this entry exists to test:
:mod:`usis.chemometrics.matching` states that the cosine is *sensitive to the baseline, because a
shared background inflates every similarity toward one*. If that is right, this method should sit
worse on the open-set frontier than the correlation - it should be the more willing of the two to
name an entry for a substance that is absent - and the gap should be the larger the less baseline
correction the pipeline applied.
"""

from __future__ import annotations

from usis.chemometrics.matching import SimilarityMetric
from usis.methods.base import Citation
from usis.methods.builtin.baseline_nn_match import NearestNeighbourMatch
from usis.methods.registry import register_method

__all__ = ["CosineLibraryMatch"]


class CosineLibraryMatch(NearestNeighbourMatch):
    """Nearest-neighbour library matching scored by the cosine between intensity vectors."""

    method_id = "baseline.nn_match_cosine"
    version = "1.0.0"
    provenance = Citation(
        title="Library search by cosine (dot-product) hit quality",
        internal_ref="docs/knowledge/chemometrics.md#the-methods-and-what-each-assumes",
        note=(
            "The default similarity of commercial spectral library search. The referenced "
            "reproduction is the shared planted-truth retrieval test, run under this metric."
        ),
    )

    def __init__(self, top_k: int = 5) -> None:
        super().__init__(metric=SimilarityMetric.COSINE, top_k=top_k)


register_method(
    CosineLibraryMatch(),
    evidence=("tests/test_compare_tasks.py::test_every_matching_metric_retrieves_planted_entry",),
    tags=("baseline", "chemometrics", "retrieval"),
    notes=(
        "ESTABLISHED under PROJECT_DECISIONS.md D17, which records why a similarity metric is a "
        "method here and not a hyperparameter. Shares every assumption, every caveat and the "
        "UNCALIBRATED p_none convention of baseline.nn_match; only the similarity differs."
    ),
)
