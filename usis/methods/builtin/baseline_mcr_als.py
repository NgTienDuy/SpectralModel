"""ESTABLISHED baseline: MCR-ALS for mixture decomposition, reporting its own ambiguity.

``docs/knowledge/chemometrics.md`` sets the requirement that shapes this method: *any mixture
result must be able to report the feasible band, not only the point solution.* Bilinear
decomposition is unique only up to an invertible rotation, so a resolved spectrum reported without
its ambiguity is one member of a family presented as the answer - a mixture-ambiguity failure
(``F09``).

So the result always carries ``ambiguity_lower_bound``, and it is a *lower* bound: the disagreement
between the equally-well-fitting optima this optimiser reached, not the extent of the feasible set.
A large bound proves ambiguity; a small one proves only that the multi-start search did not find
any, and the method says so in its own evidence statement rather than leaving the reader to assume.
"""

from __future__ import annotations

import math

from usis.chemometrics.decomposition import mcr_als
from usis.chemometrics.matrix import as_matrix
from usis.core.provenance import RunContext
from usis.core.types import ClaimType, Evidence, FailureCode, Maturity, TaskKind
from usis.methods.assumptions import (
    AssumesLinearMixing,
    RequiresCommonAxis,
    RequiresNonNegativeIntensity,
)
from usis.methods.base import Citation, CostModel, MethodBase, MethodResult
from usis.methods.inputs import MethodInputs
from usis.methods.registry import register_method

__all__ = ["McrAls"]

#: Above this normalised disagreement between equally-fitting solutions, the decomposition is
#: reported as ambiguous with F09. A stated convention: 10% of a normalised resolved spectrum is
#: more than any component assignment can absorb.
AMBIGUOUS_ABOVE = 0.1


class McrAls(MethodBase):
    """Resolve component spectra and concentrations by alternating non-negative least squares."""

    method_id = "baseline.mcr_als"
    version = "1.0.0"
    task = TaskKind.MIXTURE_DECOMPOSITION
    maturity = Maturity.ESTABLISHED
    provenance = Citation(
        title="Multivariate curve resolution by alternating least squares",
        internal_ref="docs/knowledge/chemometrics.md#the-ambiguity-that-defines-the-field",
        note=(
            "Textbook method; the referenced reproduction recovers three planted pure-component "
            "spectra from their non-negative mixtures at correlation > 0.999."
        ),
    )
    assumptions = (
        RequiresCommonAxis(),
        RequiresNonNegativeIntensity(),
        AssumesLinearMixing(),
    )
    cost = CostModel(
        compute_seconds_per_spectrum=0.05,
        note=(
            "n_starts x iterations x (n_samples + n_channels) non-negative least-squares solves; "
            "the dominant cost of the baseline suite"
        ),
    )

    def __init__(self, n_components: int = 2, closure: bool = False, n_starts: int = 3) -> None:
        self.n_components = n_components
        self.closure = closure
        self.n_starts = n_starts

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Decompose the input set and report the solution together with its ambiguity bound."""
        data = as_matrix(inputs.spectra)
        result = mcr_als(
            data,
            min(self.n_components, *data.shape),
            seed=ctx.seed,
            n_starts=self.n_starts,
            closure=self.closure,
        )
        ambiguous = math.isfinite(result.ambiguity) and result.ambiguity > AMBIGUOUS_ABOVE
        if ambiguous:
            reading = (
                f"The multi-start bound on rotational ambiguity is {result.ambiguity:.3f}, above "
                f"the {AMBIGUOUS_ABOVE:g} convention: equally-well-fitting solutions disagree "
                "about the resolved spectra, so these components are one member of a family."
            )
        elif math.isfinite(result.ambiguity):
            reading = (
                f"The multi-start bound is {result.ambiguity:.3f}, which means this search found "
                "no ambiguity - not that the solution is unique. A rotation search (MCR-BANDS and "
                "successors) is what would bound the feasible set properly, and is not implemented."
            )
        else:
            reading = (
                "No ambiguity bound: only one start fitted the data well enough to compare, so "
                "nothing was measured about the feasible set."
            )
        return self._result(
            values={
                "n_components": float(result.n_components),
                "n_spectra": float(data.shape[0]),
                "relative_residual": result.relative_residual,
                "iterations": float(result.iterations),
                "converged": float(result.converged),
                "ambiguity_lower_bound": result.ambiguity,
                "starts_compared": float(result.n_starts),
                "closure_enforced": float(self.closure),
            },
            labels={"method": result.method},
            evidence=(
                Evidence(
                    kind="mixture_decomposition",
                    statement=(
                        f"MCR-ALS resolved {result.n_components} non-negative components from "
                        f"{data.shape[0]} spectra with relative residual "
                        f"{result.relative_residual:.4f}"
                        + (", closure enforced. " if self.closure else ". ")
                        + reading
                    ),
                    claim_type=ClaimType.CORRELATION,
                    source_ref="docs/knowledge/chemometrics.md",
                    conditions=(
                        "non-negativity and closure shrink the feasible set but generally do not "
                        "collapse it to a point; the reported bound is a lower bound only"
                    ),
                ),
            ),
            failure_codes=(FailureCode.F09_MIXTURE_AMBIGUITY,) if ambiguous else (),
        )


register_method(
    McrAls(),
    evidence=("tests/test_chemometrics.py::test_mcr_als_recovers_planted_components",),
    tags=("baseline", "chemometrics", "mixture"),
    notes=(
        "ESTABLISHED under PROJECT_DECISIONS.md D14; the reproduction is our own planted-truth "
        "recovery. Every run is RUNNABLE_DEGRADED because AssumesLinearMixing is NOT_CHECKABLE "
        "from one invocation. The ambiguity number is a LOWER bound on the rotational feasible "
        "band - a proper band needs a rotation search, which is not implemented."
    ),
)
