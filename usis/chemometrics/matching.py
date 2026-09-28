"""Nearest-neighbour library matching, and the honest form of "the answer might not be here".

``docs/knowledge/chemometrics.md``: nearest-neighbour matching assumes *the answer is in the
library* - **precisely what an open-set setting denies**. So this baseline cannot be allowed to
return a ranked list and stop. It returns a :class:`~usis.core.types.BeliefState`, whose
``sum(posterior) + p_none == 1`` invariant forces the open-set mass to be stated.

The rule that produces ``p_none`` is a **stated convention and is not calibrated**: the open-set
mass is one minus the best similarity, and the rest is distributed over candidates by a sharpened
similarity ratio. It is monotone in the right direction and it is honest about being a convention.
Novelty candidate N10 is precisely the work of replacing it with a ``p_none`` validated against
held-out open-set data with measured coverage - that is M6, and until then any number this module
emits for ``p_none`` must be read as a placeholder with the right shape rather than a probability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from usis.core.errors import UsisError
from usis.core.types import BeliefState, Candidate, FloatArray, NonIdentifiability

__all__ = [
    "MatchingError",
    "SimilarityMetric",
    "SpectralMatch",
    "belief_from_matches",
    "match_library",
    "similarity",
]


class MatchingError(UsisError):
    """A match was requested against a library that cannot support it."""


class SimilarityMetric(StrEnum):
    """How to score a query against a library entry.

    ``COSINE`` is the field's default and is sensitive to the baseline: a shared background inflates
    every similarity toward one. ``CORRELATION`` mean-centres first, which removes an additive
    offset and makes the comparison about shape - usually the better choice on spectra whose
    baseline treatment differs. ``SPECTRAL_CONTRAST_ANGLE`` rescales the cosine to spread the
    crowded region near one, where library matches actually live.
    """

    COSINE = "cosine"
    CORRELATION = "correlation"
    SPECTRAL_CONTRAST_ANGLE = "spectral_contrast_angle"


def similarity(
    query: FloatArray | np.ndarray,
    reference: FloatArray | np.ndarray,
    metric: SimilarityMetric = SimilarityMetric.CORRELATION,
) -> float:
    """Similarity of one query spectrum to one reference, in ``[-1, 1]``."""
    a = np.asarray(query, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    if a.size != b.size:
        raise MatchingError(
            f"query has {a.size} channels and reference has {b.size}: match on a common axis, "
            "which is what the resampling step in usis.preprocess is for"
        )
    if metric is SimilarityMetric.CORRELATION:
        a = a - a.mean()
        b = b - b.mean()
    norms = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if norms == 0.0:
        raise MatchingError(
            "a spectrum with zero norm cannot be matched: under the correlation metric this means "
            "a perfectly flat spectrum, which carries no shape to compare"
        )
    cosine = float(np.dot(a, b) / norms)
    if metric is SimilarityMetric.SPECTRAL_CONTRAST_ANGLE:
        return float(1.0 - 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)
    return cosine


@dataclass(frozen=True)
class SpectralMatch:
    """One library hit: which entry, how similar, and by which metric."""

    key: str
    score: float
    metric: SimilarityMetric
    name: str | None = None
    inchikey: str | None = None

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable row."""
        return {
            "key": self.key,
            "score": self.score,
            "metric": str(self.metric),
            "name": self.name,
            "inchikey": self.inchikey,
        }


def match_library(
    query: FloatArray | np.ndarray,
    library: FloatArray | np.ndarray,
    keys: Sequence[str],
    *,
    metric: SimilarityMetric = SimilarityMetric.CORRELATION,
    top_k: int = 5,
    names: Sequence[str | None] | None = None,
    inchikeys: Sequence[str | None] | None = None,
) -> tuple[SpectralMatch, ...]:
    """Rank library entries by similarity to ``query``, best first."""
    entries = np.asarray(library, dtype=np.float64)
    if entries.ndim != 2:
        raise MatchingError(f"library must be 2-D (entries x channels), got {entries.shape}")
    if entries.shape[0] != len(keys):
        raise MatchingError(
            f"library has {entries.shape[0]} entries and {len(keys)} keys were supplied"
        )
    if top_k < 1:
        raise MatchingError(f"top_k must be at least 1, got {top_k}")
    scores = [similarity(query, entry, metric) for entry in entries]
    order = np.argsort(scores)[::-1][:top_k]
    return tuple(
        SpectralMatch(
            key=keys[int(index)],
            score=float(scores[int(index)]),
            metric=metric,
            name=None if names is None else names[int(index)],
            inchikey=None if inchikeys is None else inchikeys[int(index)],
        )
        for index in order
    )


#: Sharpness of the posterior over candidates. Above one it concentrates mass on the best match.
#: A convention, like everything else in this rule, and the reason it is named rather than inlined.
POSTERIOR_SHARPNESS = 4.0


def belief_from_matches(
    matches: Sequence[SpectralMatch],
    *,
    sharpness: float = POSTERIOR_SHARPNESS,
    library_size: int | None = None,
) -> BeliefState:
    """Turn ranked matches into a belief state with explicit open-set mass.

    The convention, stated in full so it can be argued with rather than inferred from code:

    - similarities are clipped to ``[0, 1]``, since a negative correlation is not weak evidence for
      a candidate, it is evidence against it;
    - ``p_none = 1 - best`` - a perfect match leaves no open-set mass, a mediocre one leaves a lot;
    - the remaining ``best`` is split across candidates in proportion to ``score ** sharpness``.

    None of this is calibrated. ``NonIdentifiability.LIBRARY_GAP`` is attached whenever the
    open-set mass exceeds the best candidate's posterior, because at that point the honest reading
    of the answer is "probably not in this library" - which is a result, not a failure.
    """
    if not matches:
        raise MatchingError(
            "no matches to build a belief from. An empty candidate list with p_none = 1 would be "
            "a defensible answer, but it must be constructed deliberately by the caller, not "
            "produced by a function that was asked to rank an empty library."
        )
    scores = np.clip(np.array([m.score for m in matches], dtype=np.float64), 0.0, 1.0)
    best = float(scores.max())
    p_none = 1.0 - best
    weights = scores**sharpness
    total = float(weights.sum())
    if total <= 0.0:
        # Every similarity was non-positive: nothing in the library resembles the query at all.
        return BeliefState(
            candidates=(),
            p_none=1.0,
            non_identifiability=(NonIdentifiability.LIBRARY_GAP,),
            unresolved_questions=(
                "No library entry has a positive similarity to this spectrum, so the identity is "
                "not constrained by this library at all.",
            ),
        )
    shares = weights / total * best
    candidates = tuple(
        Candidate(
            posterior=float(share),
            name=match.name or match.key,
            inchikey=match.inchikey,
        )
        for match, share in zip(matches, shares, strict=True)
    )
    top = max(candidates, key=lambda c: c.posterior)
    questions = []
    non_identifiability: tuple[NonIdentifiability, ...] = ()
    if p_none > top.posterior:
        non_identifiability = (NonIdentifiability.LIBRARY_GAP,)
        questions.append(
            f"Open-set mass {p_none:.3f} exceeds the best candidate's {top.posterior:.3f}: the "
            "substance is more likely absent from this library"
            + (f" of {library_size} entries" if library_size else "")
            + " than to be any entry in it."
        )
    questions.append(
        "p_none here follows a stated, UNCALIBRATED convention (1 - best similarity). "
        "Calibrating it against held-out open-set data is PROJECT_NOVELTY.md#N10, at M6."
    )
    return BeliefState(
        candidates=candidates,
        p_none=float(p_none),
        unresolved_questions=tuple(questions),
        non_identifiability=non_identifiability,
    )
