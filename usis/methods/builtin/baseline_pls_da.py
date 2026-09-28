"""ESTABLISHED baseline: PLS-DA for classification, with its scores kept out of probability space.

PLS-DA regresses spectra onto an indicator matrix and takes the argmax. The scores it produces are
**not probabilities** - they exceed one, they go negative, and two classes can both score high -
so this method reports them as ``values`` and refuses to dress them up. Turning them into a
posterior needs a calibration step; inventing one here would put an uncalibrated softmax inside a
:class:`~usis.core.types.BeliefState` and satisfy the normalisation invariant while meaning
nothing. The abstention machinery that does this properly is M6.
"""

from __future__ import annotations

import numpy as np

from usis.chemometrics.matrix import MatrixError, as_matrix, class_labels
from usis.chemometrics.regression import one_hot, pls, pls_da_predict
from usis.core.provenance import RunContext
from usis.core.types import ClaimType, Evidence, FailureCode, Maturity, TaskKind
from usis.methods.assumptions import MinSnr, RequiresCommonAxis
from usis.methods.base import Citation, CostModel, MethodBase, MethodResult
from usis.methods.inputs import MethodInputs
from usis.methods.registry import register_method

__all__ = ["PlsDiscriminantAnalysis"]


class PlsDiscriminantAnalysis(MethodBase):
    """Classify by PLS onto a class-indicator matrix, then argmax."""

    method_id = "baseline.pls_da"
    version = "1.0.0"
    task = TaskKind.CLASSIFICATION
    maturity = Maturity.ESTABLISHED
    provenance = Citation(
        title="PLS discriminant analysis",
        internal_ref="docs/knowledge/chemometrics.md#the-methods-and-what-each-assumes",
        note="Textbook method; the referenced reproduction separates planted classes exactly.",
    )
    assumptions = (RequiresCommonAxis(), MinSnr(3.0, hard=False))
    cost = CostModel(compute_seconds_per_spectrum=5e-4, note="one NIPALS fit per invocation")

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Fit on the calibration set and classify the queries."""
        if not inputs.calibration:
            raise MatrixError(f"{self.method_key} needs a labelled calibration set")
        calibration = as_matrix(inputs.calibration)
        labels = class_labels(inputs.calibration)
        indicator, classes = one_hot(labels)
        components = min(self.n_components, *calibration.shape)
        model = pls(calibration, indicator, components)
        predicted, scores = pls_da_predict(model, as_matrix(inputs.spectra), classes)
        fitted, _ = pls_da_predict(model, calibration, classes)
        accuracy = float(np.mean([a == b for a, b in zip(fitted, labels, strict=True)]))
        # The gap between the winning indicator score and the runner-up: an uncalibrated
        # confidence, named as such so nobody reads it as a probability.
        sorted_scores = np.sort(scores, axis=1)
        gaps = sorted_scores[:, -1] - sorted_scores[:, -2] if scores.shape[1] > 1 else np.zeros(1)
        return self._result(
            values={
                "n_calibration": float(calibration.shape[0]),
                "n_classes": float(len(classes)),
                "n_components": float(components),
                "calibration_accuracy": accuracy,
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "median_score_gap": float(np.median(gaps)),
            },
            labels={
                spectrum.spectrum_id: prediction
                for spectrum, prediction in zip(inputs.spectra, predicted, strict=True)
            },
            evidence=(
                Evidence(
                    kind="chemometric_classification",
                    statement=(
                        f"PLS-DA with {components} latent variables over {len(classes)} classes; "
                        f"indicator scores span [{scores.min():.3f}, {scores.max():.3f}], which "
                        "is outside [0, 1] and therefore not a probability."
                    ),
                    claim_type=ClaimType.CORRELATION,
                    source_ref="docs/knowledge/chemometrics.md",
                    conditions=(
                        "indicator scores are uncalibrated; calibration accuracy is in-sample and "
                        "is not a performance estimate"
                    ),
                ),
            ),
            failure_codes=(
                (FailureCode.F11_OVERCONFIDENCE_UNDER_SHIFT,) if float(scores.max()) > 1.5 else ()
            ),
        )


register_method(
    PlsDiscriminantAnalysis(),
    evidence=("tests/test_chemometrics.py::test_pls_da_separates_planted_classes",),
    tags=("baseline", "chemometrics", "classification"),
    notes=(
        "ESTABLISHED under PROJECT_DECISIONS.md D14; the reproduction is our own planted-truth "
        "recovery. Emits F11_OVERCONFIDENCE_UNDER_SHIFT when an indicator score exceeds 1.5, "
        "which is the model extrapolating past its own indicator coding."
    ),
)
