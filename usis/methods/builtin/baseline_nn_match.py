"""ESTABLISHED baseline: nearest-neighbour library matching, forced to state its open-set mass.

The oldest method in spectroscopy and still the hardest retrieval baseline to beat. Its
load-bearing assumption is the one the whole project is aimed at: *the answer is in the library* -
which an open-set setting denies. So this method's output is a
:class:`~usis.core.types.BeliefState`, and the type will not let it omit ``p_none``.

The ``p_none`` rule is an **uncalibrated convention** (one minus the best similarity), stated in
:mod:`usis.chemometrics.matching` and repeated in every result. Replacing it with a validated
open-set residual is `PROJECT_NOVELTY.md#N10`, at M6. Until then the number has the right shape and
the right monotonicity and is not a probability, and the method says so rather than letting the
normalisation invariant imply otherwise.

The library arrives through ``MethodInputs.library`` for the coverage *check* and through
``aux["library_spectra"]`` / ``aux["library_keys"]`` for the arrays. That split is deliberate:
:class:`~usis.methods.assumptions.RequiresLibraryCoverage` can then refuse before any arithmetic
runs, and the refusal names the missing library rather than surfacing as an index error.
"""

from __future__ import annotations

import numpy as np

from usis.chemometrics.matching import (
    MatchingError,
    SimilarityMetric,
    belief_from_matches,
    match_library,
)
from usis.core.provenance import RunContext
from usis.core.types import (
    ClaimType,
    Evidence,
    FailureCode,
    Maturity,
    NonIdentifiability,
    TaskKind,
)
from usis.methods.assumptions import MinSnr, RequiresCommonAxis, RequiresLibraryCoverage
from usis.methods.base import Citation, CostModel, MethodBase, MethodResult
from usis.methods.inputs import MethodInputs
from usis.methods.registry import register_method

__all__ = ["NearestNeighbourMatch"]


class NearestNeighbourMatch(MethodBase):
    """Rank library entries by spectral similarity and emit a belief state with open-set mass."""

    method_id = "baseline.nn_match"
    version = "1.0.0"
    task = TaskKind.COMPOUND_RETRIEVAL
    maturity = Maturity.ESTABLISHED
    provenance = Citation(
        title="Nearest-neighbour spectral library matching",
        internal_ref="docs/knowledge/chemometrics.md#the-methods-and-what-each-assumes",
        note=(
            "Textbook method; the referenced reproduction retrieves a planted library entry at "
            "rank 1 and, for a query absent from the library, puts the mass on p_none instead."
        ),
    )
    assumptions = (RequiresCommonAxis(), RequiresLibraryCoverage(hard=False), MinSnr(3.0))
    cost = CostModel(compute_seconds_per_spectrum=1e-4, note="one dot product per library entry")

    def __init__(
        self,
        metric: SimilarityMetric = SimilarityMetric.CORRELATION,
        top_k: int = 5,
    ) -> None:
        self.metric = metric
        self.top_k = top_k

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Match the single query spectrum against the supplied library arrays."""
        spectrum = inputs.primary
        entries = inputs.aux.get("library_spectra")
        keys = inputs.aux.get("library_keys")
        if entries is None or keys is None:
            raise MatchingError(
                f"{self.method_key} needs aux['library_spectra'] and aux['library_keys']. The "
                "LibraryHandle in MethodInputs.library declares what the library *is*, so that "
                "coverage can be checked before any arithmetic; the arrays are passed separately "
                "so a handle can be supplied without loading them."
            )
        matches = match_library(
            np.asarray(spectrum.intensity, dtype=np.float64),
            np.asarray(entries, dtype=np.float64),
            list(keys),
            metric=self.metric,
            top_k=self.top_k,
            names=inputs.aux.get("library_names"),
            inchikeys=inputs.aux.get("library_inchikeys"),
        )
        library = inputs.library
        belief = belief_from_matches(
            matches, library_size=library.n_entries if library is not None else None
        )
        top = belief.top
        gap = NonIdentifiability.LIBRARY_GAP in belief.non_identifiability
        return self._result(
            belief=belief,
            values={
                "best_similarity": float(matches[0].score),
                "runner_up_similarity": (
                    float(matches[1].score) if len(matches) > 1 else float("nan")
                ),
                "p_none": belief.p_none,
                "top_posterior": float(top.posterior) if top is not None else 0.0,
                "n_library_entries": float(len(keys)),
                "n_candidates": float(len(belief.candidates)),
            },
            labels={
                "spectrum_id": spectrum.spectrum_id,
                "metric": str(self.metric),
                "top_candidate": top.name or "" if top is not None else "",
            },
            evidence=(
                Evidence(
                    kind="library_match",
                    statement=(
                        f"Best {self.metric} similarity {matches[0].score:.4f} to "
                        f"{matches[0].key!r} out of {len(keys)} library entries; open-set mass "
                        f"{belief.p_none:.4f}."
                        + (
                            " The open-set mass exceeds the best candidate, so the honest reading "
                            "is that the substance is probably not in this library."
                            if gap
                            else ""
                        )
                    ),
                    claim_type=ClaimType.CORRELATION,
                    source_ref="usis/chemometrics/matching.py",
                    conditions=(
                        "p_none follows a stated UNCALIBRATED convention (1 - best similarity); "
                        "calibrating it is PROJECT_NOVELTY.md#N10 at M6"
                    ),
                ),
            ),
            failure_codes=(FailureCode.F12_RETRIEVAL_MISS,) if gap else (),
        )


register_method(
    NearestNeighbourMatch(),
    evidence=(
        "tests/test_chemometrics.py::test_nn_match_retrieves_the_planted_entry",
        "tests/test_chemometrics.py::test_nn_match_puts_the_mass_on_p_none_for_an_absent_query",
    ),
    tags=("baseline", "chemometrics", "retrieval"),
    notes=(
        "ESTABLISHED under PROJECT_DECISIONS.md D14; the reproduction is our own planted-truth "
        "retrieval. The p_none it reports is an UNCALIBRATED convention with the right shape, not "
        "a probability - N10 at M6 replaces it. RequiresLibraryCoverage is declared soft here on "
        "purpose: a retrieval run on a query outside the library is exactly the open-set case the "
        "project needs to measure, so it must run and report F12, not refuse."
    ),
)
