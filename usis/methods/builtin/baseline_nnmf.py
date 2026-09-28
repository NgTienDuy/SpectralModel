"""ESTABLISHED baseline: NNMF, the control that isolates what non-negativity alone buys.

Registered alongside MCR-ALS on purpose. The two solve the same bilinear problem and differ in the
chemistry they are told: MCR-ALS can be given closure and known pure components, NNMF is given
non-negativity and nothing else. Comparing them measures **how much of MCR-ALS's advantage comes
from the constraints rather than from the factorisation** - which is a question about the field's
practice, and it is free once both are registered.
"""

from __future__ import annotations

import math

from usis.chemometrics.decomposition import nnmf
from usis.chemometrics.matrix import as_matrix
from usis.core.provenance import RunContext
from usis.core.types import ClaimType, Evidence, FailureCode, Maturity, TaskKind
from usis.methods.assumptions import RequiresCommonAxis, RequiresNonNegativeIntensity
from usis.methods.base import Citation, CostModel, MethodBase, MethodResult
from usis.methods.builtin.baseline_mcr_als import AMBIGUOUS_ABOVE
from usis.methods.inputs import MethodInputs
from usis.methods.registry import register_method

__all__ = ["Nnmf"]


class Nnmf(MethodBase):
    """Factorise the spectrum matrix into non-negative components by multiplicative updates."""

    method_id = "baseline.nnmf"
    version = "1.0.0"
    task = TaskKind.MIXTURE_DECOMPOSITION
    maturity = Maturity.ESTABLISHED
    provenance = Citation(
        title="Non-negative matrix factorisation by multiplicative updates",
        internal_ref="docs/knowledge/chemometrics.md#the-methods-and-what-each-assumes",
        note=(
            "Textbook method; the referenced reproduction recovers planted non-negative "
            "components. Deliberately *not* given AssumesLinearMixing: NNMF makes no "
            "Beer-Lambert claim, it factorises - and that difference from MCR-ALS is the "
            "comparison the pair exists to support."
        ),
    )
    assumptions = (RequiresCommonAxis(), RequiresNonNegativeIntensity())
    cost = CostModel(
        compute_seconds_per_spectrum=2e-3,
        note="multiplicative updates, far cheaper per iteration than MCR-ALS's exact NNLS",
    )

    def __init__(self, n_components: int = 2, n_starts: int = 3) -> None:
        self.n_components = n_components
        self.n_starts = n_starts

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Factorise the input set and report the solution with its ambiguity bound."""
        data = as_matrix(inputs.spectra)
        result = nnmf(
            data,
            min(self.n_components, *data.shape),
            seed=ctx.seed,
            n_starts=self.n_starts,
        )
        ambiguous = math.isfinite(result.ambiguity) and result.ambiguity > AMBIGUOUS_ABOVE
        return self._result(
            values={
                "n_components": float(result.n_components),
                "n_spectra": float(data.shape[0]),
                "relative_residual": result.relative_residual,
                "iterations": float(result.iterations),
                "converged": float(result.converged),
                "ambiguity_lower_bound": result.ambiguity,
                "starts_compared": float(result.n_starts),
            },
            labels={"method": result.method},
            evidence=(
                Evidence(
                    kind="mixture_decomposition",
                    statement=(
                        f"NNMF factorised {data.shape[0]} spectra into {result.n_components} "
                        f"non-negative components with relative residual "
                        f"{result.relative_residual:.4f} and a multi-start ambiguity bound of "
                        f"{result.ambiguity:.3f}. Non-negativity is the only constraint applied, "
                        "so the components carry no chemical interpretation on their own."
                    ),
                    claim_type=ClaimType.CORRELATION,
                    source_ref="docs/knowledge/chemometrics.md",
                    conditions=(
                        "a factorisation is not a resolution: without closure or known pure "
                        "components there is nothing tying a factor to a substance"
                    ),
                ),
            ),
            failure_codes=(FailureCode.F09_MIXTURE_AMBIGUITY,) if ambiguous else (),
        )


register_method(
    Nnmf(),
    evidence=("tests/test_chemometrics.py::test_nnmf_recovers_planted_components",),
    tags=("baseline", "chemometrics", "mixture"),
    notes=(
        "ESTABLISHED under PROJECT_DECISIONS.md D14; the reproduction is our own planted-truth "
        "recovery. Registered as the constraint-free control for baseline.mcr_als: the pair "
        "measures what the chemical constraints are worth."
    ),
)
