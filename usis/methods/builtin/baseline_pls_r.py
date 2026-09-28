"""ESTABLISHED baseline: PLS regression for quantification.

The workhorse of quantitative vibrational spectroscopy, and the baseline any deep quantification
model has to beat on the same split. Two declarations do the real work here.

``AssumesLinearMixing`` is declared even though it is ``NOT_CHECKABLE`` from the inputs, and that
is the point rather than a limitation: PLS regression rests on Beer-Lambert linearity, the check
cannot decide it from a single invocation, and so every PLS result runs in
``RUNNABLE_DEGRADED`` and carries the concentration series that would settle it. A method whose
central assumption is unverified should not be able to produce a clean-looking result.

``RequiresIntensityKind`` restricts it to absorbance-like conventions. Beer-Lambert additivity
holds in absorbance and not in transmittance, so the same arithmetic on the wrong convention is
solving a different equation - silently, and with a plausible R-squared.
"""

from __future__ import annotations

import numpy as np

from usis.chemometrics.matrix import MatrixError, as_matrix, target_values
from usis.chemometrics.regression import pls
from usis.core.provenance import RunContext
from usis.core.types import (
    ClaimType,
    Evidence,
    FailureCode,
    IntensityKind,
    Maturity,
    TaskKind,
)
from usis.methods.assumptions import AssumesLinearMixing, MinSnr, RequiresCommonAxis
from usis.methods.base import Citation, CostModel, MethodBase, MethodResult, Requirements
from usis.methods.inputs import MethodInputs
from usis.methods.registry import register_method

__all__ = ["PlsRegression"]


class PlsRegression(MethodBase):
    """Predict a named quantitative target by partial least squares."""

    method_id = "baseline.pls_r"
    version = "1.0.0"
    task = TaskKind.QUANTIFICATION
    maturity = Maturity.ESTABLISHED
    provenance = Citation(
        title="Partial least squares regression by NIPALS",
        internal_ref="docs/knowledge/chemometrics.md#the-methods-and-what-each-assumes",
        note=(
            "Textbook method; the referenced reproduction recovers a planted linear "
            "concentration-intensity relation to R^2 > 0.999."
        ),
    )
    requires = Requirements(
        intensity_kinds=frozenset({IntensityKind.ABSORBANCE, IntensityKind.KUBELKA_MUNK})
    )
    assumptions = (RequiresCommonAxis(), AssumesLinearMixing(), MinSnr(5.0, hard=False))
    cost = CostModel(
        compute_seconds_per_spectrum=5e-4, note="one NIPALS fit over the calibration set"
    )

    def __init__(self, target: str = "concentration", n_components: int = 3) -> None:
        self.target = target
        self.n_components = n_components

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Fit on the calibration set, predict the queries, and report where capacity went."""
        if not inputs.calibration:
            raise MatrixError(
                f"{self.method_key} needs a calibration set carrying label.targets[{self.target!r}]"
            )
        calibration = as_matrix(inputs.calibration)
        response = target_values(inputs.calibration, self.target)
        components = min(self.n_components, *calibration.shape)
        model = pls(calibration, response, components)
        fitted = model.predict(calibration)
        residual = float(((response - fitted) ** 2).sum())
        variance = float(((response - response.mean()) ** 2).sum()) or 1.0
        predictions = model.predict(as_matrix(inputs.spectra))
        warning = model.capacity_warning
        return self._result(
            values={
                "n_calibration": float(calibration.shape[0]),
                "n_components": float(components),
                "calibration_r2": 1.0 - residual / variance,
                "calibration_rmse": float(np.sqrt(residual / response.size)),
                "x_variance_explained": float(model.x_variance_explained.sum()),
                "y_variance_explained": float(model.y_variance_explained.sum()),
                **{
                    f"predicted_{spectrum.spectrum_id}": float(value[0])
                    for spectrum, value in zip(inputs.spectra, predictions, strict=True)
                },
            },
            labels={"target": self.target},
            evidence=(
                Evidence(
                    kind="chemometric_quantification",
                    statement=(
                        f"PLS with {components} latent variables on {calibration.shape[0]} "
                        f"calibration spectra explains "
                        f"{model.y_variance_explained.sum():.1%} of the variance in "
                        f"{self.target!r} and {model.x_variance_explained.sum():.1%} of the "
                        "spectral variance." + (f" Capacity warning: {warning}." if warning else "")
                    ),
                    claim_type=ClaimType.CORRELATION,
                    source_ref="docs/knowledge/chemometrics.md",
                    conditions=(
                        "calibration R^2 is an in-sample number and is not a performance estimate; "
                        "Beer-Lambert linearity is declared and unverified"
                    ),
                ),
            ),
            failure_codes=(FailureCode.F14_SHORTCUT_LEARNING,) if warning else (),
            notes=warning,
        )


register_method(
    PlsRegression(),
    evidence=("tests/test_chemometrics.py::test_pls_recovers_a_planted_linear_relation",),
    tags=("baseline", "chemometrics", "quantification"),
    notes=(
        "ESTABLISHED under PROJECT_DECISIONS.md D14; the reproduction is our own planted-truth "
        "recovery. Every run of this method is RUNNABLE_DEGRADED by construction, because "
        "AssumesLinearMixing is NOT_CHECKABLE from one invocation - which is the intended "
        "behaviour, not a defect: a quantification resting on unverified Beer-Lambert linearity "
        "must not be presentable as a clean result."
    ),
)
