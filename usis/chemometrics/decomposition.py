"""Bilinear decomposition: PCA, NNMF, MCR-ALS - and the ambiguity all three of them live inside.

``docs/knowledge/chemometrics.md`` states the fact that shapes this module: for any invertible
``T``, ``D = C S' = (C T)(T^-1 S')``, so a bilinear decomposition is **not unique**. Non-negativity
and closure shrink the feasible set; they generally do not collapse it to a point. Reporting one
member of that family as *the* answer is a mixture-ambiguity failure (``F09``), and the knowledge
note's requirement is explicit: *any mixture result must be able to report the feasible band, not
only the point solution.*

So :func:`mcr_als` and :func:`nnmf` return a **multi-start spread** alongside the solution. That
spread is a *lower bound* on the feasible band and nothing more - it measures the disagreement
between the optima this optimiser happened to reach, not the extent of the set of all solutions
compatible with the data. Saying so precisely is the point; a proper band needs a rotation search
(MCR-BANDS and its successors), which is not implemented here. The bound is still useful, because a
*large* lower bound is already proof of ambiguity.

Everything is pure numpy and seeded. Deep-model comparisons are only meaningful against baselines
that actually run, so these must never depend on an optional install.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from usis.core.errors import UsisError
from usis.core.types import FloatArray

__all__ = [
    "Decomposition",
    "DecompositionError",
    "PcaModel",
    "mcr_als",
    "nnls",
    "nnmf",
    "pca",
]


class DecompositionError(UsisError):
    """A decomposition was asked for something the data cannot support."""


def _matrix(data: FloatArray | np.ndarray, name: str) -> FloatArray:
    """Coerce to a 2-D float array of finite values, refusing anything else."""
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise DecompositionError(
            f"{name} must be 2-D (samples x channels), got shape {array.shape}"
        )
    if array.size == 0:
        raise DecompositionError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise DecompositionError(
            f"{name} contains non-finite values. A NaN here silently becomes a NaN component and "
            "then a NaN score, so it is refused at the door."
        )
    return array


def _check_rank(n_components: int, n_samples: int, n_channels: int) -> None:
    limit = min(n_samples, n_channels)
    if n_components < 1:
        raise DecompositionError(f"n_components must be at least 1, got {n_components}")
    if n_components > limit:
        raise DecompositionError(
            f"n_components={n_components} exceeds the rank available from {n_samples} samples x "
            f"{n_channels} channels (max {limit}). Extra components would fit noise and report it "
            "as structure."
        )


# ------------------------------------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PcaModel:
    """A fitted PCA, with what it explains and what it assumes.

    The load-bearing assumption is in the knowledge note: *variance is the interesting direction*,
    which in vibrational spectroscopy is *often false* - the dominant variance is routinely the
    baseline, the instrument or the operator. So ``explained_variance_ratio`` is not a quality
    score, and a first component explaining 95% of the variance is as likely to be a fluorescence
    background as a chemical axis.
    """

    mean: FloatArray
    loadings: FloatArray
    singular_values: FloatArray
    explained_variance_ratio: FloatArray
    n_samples: int

    @property
    def n_components(self) -> int:
        """Number of retained components."""
        return int(self.loadings.shape[0])

    def transform(self, data: FloatArray | np.ndarray) -> FloatArray:
        """Project spectra onto the retained components."""
        array = _matrix(data, "data")
        if array.shape[1] != self.mean.size:
            raise DecompositionError(
                f"model was fitted on {self.mean.size} channels, got {array.shape[1]}. Spectra "
                "must be on a common axis before projection - which is what the resampling step "
                "in usis.preprocess exists for."
            )
        return np.asarray((array - self.mean) @ self.loadings.T, dtype=np.float64)

    def reconstruct(self, scores: FloatArray | np.ndarray) -> FloatArray:
        """Map scores back to spectra, for the residual that says what the model missed."""
        return np.asarray(
            np.asarray(scores, dtype=np.float64) @ self.loadings + self.mean, dtype=np.float64
        )

    def residual_norm(self, data: FloatArray | np.ndarray) -> FloatArray:
        """Per-spectrum reconstruction error - the outlier statistic PCA is genuinely good at."""
        array = _matrix(data, "data")
        residual = array - self.reconstruct(self.transform(array))
        return np.asarray(np.sqrt((residual**2).sum(axis=1)), dtype=np.float64)


def pca(data: FloatArray | np.ndarray, n_components: int) -> PcaModel:
    """Principal components by SVD of the mean-centred data.

    Mean-centred and *not* scaled to unit variance. Autoscaling a spectrum divides every channel by
    its standard deviation, which promotes the quietest channels - usually the empty regions
    between bands, where the standard deviation is noise - to equal footing with the bands. It is
    standard for tabular data and wrong here by default.
    """
    array = _matrix(data, "data")
    n_samples, n_channels = array.shape
    _check_rank(n_components, n_samples, n_channels)
    mean = array.mean(axis=0)
    centred = array - mean
    _, singular_values, right = np.linalg.svd(centred, full_matrices=False)
    total = float((singular_values**2).sum())
    ratio = (singular_values**2) / total if total > 0 else np.zeros_like(singular_values)
    return PcaModel(
        mean=np.asarray(mean, dtype=np.float64),
        loadings=np.asarray(right[:n_components], dtype=np.float64),
        singular_values=np.asarray(singular_values[:n_components], dtype=np.float64),
        explained_variance_ratio=np.asarray(ratio[:n_components], dtype=np.float64),
        n_samples=n_samples,
    )


# ------------------------------------------------------------------------------------------------
# Non-negative least squares, the primitive MCR-ALS is built on
# ------------------------------------------------------------------------------------------------
def nnls(
    design: FloatArray | np.ndarray, target: FloatArray | np.ndarray, max_iter: int = 0
) -> FloatArray:
    """Solve ``min ||A x - b||`` subject to ``x >= 0`` by the Lawson-Hanson active-set method.

    Implemented rather than imported because scipy is an optional extra and the classical baselines
    must run in the base environment. Non-negativity here is a hard constraint enforced at
    inference, per the project conventions, not a penalty that can be traded away.

    Solved through the **normal equations** - ``A'A`` and ``A'b`` - rather than by a least-squares
    call per active set. That is not only speed, although it is a large amount of speed: MCR-ALS
    alternates one of these per sample and per channel, per iteration, per start, so the naive form
    made the baseline too slow to run at all, which for a *mandatory* baseline is the same as not
    having it. With ``n`` components the inner solves are ``n x n`` regardless of how many channels
    the spectrum has.
    """
    a = np.asarray(design, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64).ravel()
    if a.ndim != 2 or a.shape[0] != b.size:
        raise DecompositionError(
            f"nnls needs A of shape (m, n) and b of length m, got {a.shape} and {b.shape}"
        )
    return _nnls_normal(a.T @ a, a.T @ b, max_iter)


def _nnls_normal(gram: FloatArray, projection: FloatArray, max_iter: int = 0) -> FloatArray:
    """Non-negative least squares from ``A'A`` and ``A'b``. The active-set loop itself."""
    n = gram.shape[0]
    limit = max_iter if max_iter > 0 else 30 * n
    passive = np.zeros(n, dtype=bool)
    x = np.zeros(n, dtype=np.float64)
    tolerance = 10.0 * float(np.finfo(np.float64).eps) * max(float(np.abs(gram).max()), 1.0) * n
    gradient = projection - gram @ x
    for _ in range(limit):
        if passive.all() or not np.any(gradient[~passive] > tolerance):
            break
        passive[int(np.argmax(np.where(passive, -np.inf, gradient)))] = True
        for _ in range(limit):
            candidate = np.zeros(n, dtype=np.float64)
            candidate[passive] = _solve(gram, projection, passive)
            if np.all(candidate[passive] > 0.0):
                x = candidate
                break
            # A non-positive entry means the unconstrained solution left the feasible region:
            # step to the boundary and drop whichever variable reaches zero first.
            blocking = passive & (candidate <= 0.0)
            movement = x - candidate
            ratios = np.divide(
                x,
                movement,
                out=np.full(n, np.inf),
                where=blocking & (movement != 0.0),
            )
            alpha = float(ratios.min())
            if not np.isfinite(alpha):
                x = np.maximum(candidate, 0.0)
                passive &= x > tolerance
                break
            x = x + alpha * (candidate - x)
            passive &= x > tolerance
            if not passive.any():
                x = np.zeros(n, dtype=np.float64)
                break
        gradient = projection - gram @ x
    return np.asarray(np.maximum(x, 0.0), dtype=np.float64)


def _solve(gram: FloatArray, projection: FloatArray, passive: np.ndarray) -> FloatArray:
    """Least-squares solve of the passive-set subsystem, tolerant of a singular submatrix."""
    submatrix = gram[np.ix_(passive, passive)]
    subvector = projection[passive]
    try:
        return np.asarray(np.linalg.solve(submatrix, subvector), dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.lstsq(submatrix, subvector, rcond=None)[0], dtype=np.float64)


def _nnls_rows(design: FloatArray, targets: FloatArray) -> FloatArray:
    """Solve one non-negative least-squares problem per row of ``targets``, sharing the Gram matrix.

    Every row shares ``design``, so ``A'A`` is formed once and ``A'b`` for all rows in one product.
    This is the whole reason MCR-ALS is usable here.
    """
    gram = design.T @ design
    projections = targets @ design
    return np.asarray(
        np.vstack([_nnls_normal(gram, projection) for projection in projections]),
        dtype=np.float64,
    )


# ------------------------------------------------------------------------------------------------
# NNMF and MCR-ALS
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Decomposition:
    """A bilinear decomposition ``D ~ concentrations @ spectra``, with its ambiguity bound.

    ``ambiguity`` is the largest disagreement between the multi-start solutions **that fit the data
    equally well**, after matching their components to the best one and normalising each resolved
    spectrum - a *lower bound* on the rotational feasible band, in the units of the normalised
    spectra. ``n_starts`` is how many optima it was computed from, and is part of the reading: a
    bound from one start is ``nan``, and a bound from two is barely a bound.
    """

    concentrations: FloatArray
    spectra: FloatArray
    residual_norm: float
    relative_residual: float
    iterations: int
    converged: bool
    ambiguity: float
    n_starts: int
    method: str

    @property
    def n_components(self) -> int:
        """Number of resolved components."""
        return int(self.spectra.shape[0])

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable summary; the matrices themselves stay out of the record."""
        return {
            "method": self.method,
            "n_components": self.n_components,
            "residual_norm": self.residual_norm,
            "relative_residual": self.relative_residual,
            "iterations": self.iterations,
            "converged": self.converged,
            "ambiguity_lower_bound": self.ambiguity,
            "n_starts": self.n_starts,
        }


def nnmf(
    data: FloatArray | np.ndarray,
    n_components: int,
    *,
    iterations: int = 500,
    tolerance: float = 1e-7,
    seed: int = 0,
    n_starts: int = 3,
) -> Decomposition:
    """Non-negative matrix factorisation by multiplicative updates on the Frobenius loss.

    Fewer chemical constraints than MCR-ALS - non-negativity and nothing else - which is exactly
    why it is here: it is the baseline that shows how much of MCR-ALS's performance comes from the
    chemistry it is told and how much from non-negativity alone.
    """
    array = _matrix(data, "data")
    if np.any(array < 0.0):
        raise DecompositionError(
            "NNMF requires non-negative data, and this matrix has negative entries. That is "
            "usually a baseline correction that over-subtracted, or an SNV normalisation - both "
            "recorded in the processing chain. Fix the pipeline rather than clipping here."
        )
    _check_rank(n_components, *array.shape)
    solutions = []
    for start in range(max(n_starts, 1)):
        rng = np.random.default_rng(seed + start)
        scale = float(np.sqrt(array.mean() / n_components)) or 1.0
        concentrations = np.abs(rng.normal(scale, scale * 0.1, (array.shape[0], n_components)))
        spectra = np.abs(rng.normal(scale, scale * 0.1, (n_components, array.shape[1])))
        used = iterations
        previous = np.inf
        for step in range(iterations):
            spectra = (
                spectra
                * (concentrations.T @ array)
                / (concentrations.T @ concentrations @ spectra + 1e-12)
            )
            concentrations = (
                concentrations
                * (array @ spectra.T)
                / (concentrations @ spectra @ spectra.T + 1e-12)
            )
            residual = float(np.linalg.norm(array - concentrations @ spectra))
            if np.isfinite(previous) and abs(previous - residual) <= tolerance * max(previous, 1.0):
                used = step + 1
                break
            previous = residual
        solutions.append((concentrations, spectra, used))
    return _assemble(array, solutions, iterations, "nnmf")


def mcr_als(
    data: FloatArray | np.ndarray,
    n_components: int,
    *,
    iterations: int = 200,
    tolerance: float = 1e-5,
    seed: int = 0,
    n_starts: int = 3,
    closure: bool = False,
    known_spectra: FloatArray | np.ndarray | None = None,
) -> Decomposition:
    """Multivariate curve resolution by alternating non-negative least squares.

    Non-negativity on both factors is enforced exactly, by :func:`nnls`, rather than by clipping -
    a clipped negative concentration is a silently infeasible solution that still reports a
    residual. ``closure`` additionally constrains each sample's concentrations to sum to one, which
    is the mixture-fraction constraint :class:`~usis.core.types.MixtureHypothesis` asserts at
    construction. ``known_spectra`` fixes the leading rows of the resolved spectra, which is what
    turns MCR-ALS into classical least squares as more pure components become known.
    """
    array = _matrix(data, "data")
    if np.any(array < 0.0):
        raise DecompositionError(
            "MCR-ALS as set up here resolves non-negative components from non-negative data, and "
            "this matrix has negative entries; see the processing chain for the step that "
            "produced them"
        )
    _check_rank(n_components, *array.shape)
    fixed = None
    if known_spectra is not None:
        fixed = _matrix(known_spectra, "known_spectra")
        if fixed.shape[0] > n_components or fixed.shape[1] != array.shape[1]:
            raise DecompositionError(
                f"known_spectra has shape {fixed.shape}, incompatible with {n_components} "
                f"components on {array.shape[1]} channels"
            )
    solutions = []
    for start in range(max(n_starts, 1)):
        rng = np.random.default_rng(seed + start)
        # Initialise the spectra from actual samples rather than from noise: a random spectral
        # profile is not a plausible starting point and the alternation stalls on it.
        picks = rng.choice(array.shape[0], size=n_components, replace=array.shape[0] < n_components)
        spectra = np.abs(array[picks]) + 1e-9
        if fixed is not None:
            spectra[: fixed.shape[0]] = fixed
        concentrations = np.zeros((array.shape[0], n_components), dtype=np.float64)
        used = iterations
        previous = np.inf
        for step in range(iterations):
            concentrations = _nnls_rows(spectra.T, array)
            if closure:
                totals = concentrations.sum(axis=1, keepdims=True)
                concentrations = np.divide(
                    concentrations, totals, out=np.zeros_like(concentrations), where=totals > 0
                )
            spectra = _nnls_rows(concentrations, array.T).T
            if fixed is not None:
                spectra[: fixed.shape[0]] = fixed
            residual = float(np.linalg.norm(array - concentrations @ spectra))
            if np.isfinite(previous) and abs(previous - residual) <= tolerance * max(previous, 1.0):
                used = step + 1
                break
            previous = residual
        solutions.append((concentrations, spectra, used))
    return _assemble(array, solutions, iterations, "mcr_als")


def _assemble(
    array: FloatArray,
    solutions: list[tuple[FloatArray, FloatArray, int]],
    max_iterations: int,
    method: str,
) -> Decomposition:
    """Pick the best multi-start solution and measure how much the others disagree with it."""
    scored = [(float(np.linalg.norm(array - c @ s)), c, s, used) for c, s, used in solutions]
    scored.sort(key=lambda entry: entry[0])
    residual, concentrations, spectra, used = scored[0]
    total = float(np.linalg.norm(array)) or 1.0
    ambiguity = _multistart_spread([entry[2] for entry in scored])
    return Decomposition(
        concentrations=np.asarray(concentrations, dtype=np.float64),
        spectra=np.asarray(spectra, dtype=np.float64),
        residual_norm=residual,
        relative_residual=residual / total,
        iterations=used,
        converged=used < max_iterations,
        ambiguity=ambiguity,
        n_starts=len(solutions),
        method=method,
    )


def _multistart_spread(all_spectra: list[FloatArray]) -> float:
    """Largest normalised disagreement between multi-start solutions, components matched.

    A lower bound on rotational ambiguity, and honest about being one: two optimisers that happen
    to land in the same basin report zero, which means "this bound found nothing", not "the
    solution is unique".
    """
    if len(all_spectra) < 2:
        return float("nan")
    reference = _unit_rows(all_spectra[0])
    worst = 0.0
    for candidate in all_spectra[1:]:
        matched = _unit_rows(candidate)
        # Greedy component matching by correlation: the factorisation is defined up to a
        # permutation, so comparing rows in storage order would report permutation as ambiguity.
        available = list(range(matched.shape[0]))
        for row in reference:
            similarities = [float(np.dot(row, matched[index])) for index in available]
            best = available.pop(int(np.argmax(similarities)))
            worst = max(worst, float(np.abs(row - matched[best]).max()))
    return worst


def _unit_rows(matrix: FloatArray) -> FloatArray:
    """Rows scaled to unit L2 norm, so the comparison is of shape rather than of scale."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(
        np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0), dtype=np.float64
    )
