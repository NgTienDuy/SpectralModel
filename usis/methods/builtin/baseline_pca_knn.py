"""ESTABLISHED baseline: PCA scores plus a k-nearest-neighbour vote.

The classification baseline the field actually uses, and it is here to be *hard to beat*. Brief
§3: classical baselines are mandatory before any deep-model claim, on the same split and the same
metric. RamanBench's finding that tabular models beat domain-specific spectral architectures makes
this baseline's score the number that matters, not a formality.

PCA's load-bearing assumption travels with it: variance is the interesting direction. In
vibrational spectroscopy that is often false, and this method reports
``leading_component_variance`` so a reader can see when the first component is carrying almost
everything - which on real spectra is usually the baseline or the instrument, not the chemistry.
"""

from __future__ import annotations

import numpy as np

from usis.chemometrics.decomposition import pca
from usis.chemometrics.matrix import MatrixError, as_matrix, class_labels
from usis.core.provenance import RunContext
from usis.core.types import ClaimType, Evidence, FailureCode, Maturity, TaskKind
from usis.methods.assumptions import MinSnr, RequiresCommonAxis
from usis.methods.base import Citation, CostModel, MethodBase, MethodResult
from usis.methods.inputs import MethodInputs
from usis.methods.registry import register_method

__all__ = ["PcaKnn"]


class PcaKnn(MethodBase):
    """Classify by majority vote among the ``k`` nearest calibration spectra in PCA score space."""

    method_id = "baseline.pca_knn"
    version = "1.0.0"
    task = TaskKind.CLASSIFICATION
    maturity = Maturity.ESTABLISHED
    provenance = Citation(
        title="PCA with nearest-neighbour classification, the standard chemometric baseline",
        internal_ref="docs/knowledge/chemometrics.md#the-methods-and-what-each-assumes",
        note=(
            "Textbook method with no single citable origin; the reproduction referenced in the "
            "registry entry recovers planted classes from planted spectra."
        ),
    )
    assumptions = (RequiresCommonAxis(), MinSnr(3.0, hard=False))
    cost = CostModel(
        compute_seconds_per_spectrum=1e-3,
        note="one SVD over the calibration set, then a distance to every calibration spectrum",
    )

    def __init__(self, n_components: int = 5, k: int = 1) -> None:
        self.n_components = n_components
        self.k = k

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Fit PCA on the calibration set, project the queries, vote among neighbours."""
        if not inputs.calibration:
            raise MatrixError(
                f"{self.method_key} needs a labelled calibration set in MethodInputs.calibration; "
                "a nearest-neighbour classifier with no neighbours has nothing to answer with"
            )
        calibration = as_matrix(inputs.calibration)
        labels = class_labels(inputs.calibration)
        queries = as_matrix(inputs.spectra)
        components = min(self.n_components, *calibration.shape)
        model = pca(calibration, components)
        reference = model.transform(calibration)
        projected = model.transform(queries)
        predictions, margins = [], []
        for row in projected:
            distances = np.linalg.norm(reference - row, axis=1)
            nearest = np.argsort(distances)[: self.k]
            votes: dict[str, int] = {}
            for index in nearest:
                votes[labels[int(index)]] = votes.get(labels[int(index)], 0) + 1
            winner = max(votes, key=lambda name: votes[name])
            predictions.append(winner)
            # Distance to the nearest spectrum of a *different* class, minus the nearest of the
            # predicted class: a per-spectrum margin, which is what an abstention rule needs.
            same = min(
                (d for d, label in zip(distances, labels, strict=True) if label == winner),
                default=float("nan"),
            )
            other = min(
                (d for d, label in zip(distances, labels, strict=True) if label != winner),
                default=float("nan"),
            )
            margins.append(float(other - same))
        return self._result(
            values={
                "n_calibration": float(calibration.shape[0]),
                "n_classes": float(len(set(labels))),
                "n_components": float(components),
                "leading_component_variance": float(model.explained_variance_ratio[0]),
                "median_margin": float(np.median(margins)),
                "min_margin": float(np.min(margins)),
            },
            labels={
                spectrum.spectrum_id: prediction
                for spectrum, prediction in zip(inputs.spectra, predictions, strict=True)
            },
            evidence=(
                Evidence(
                    kind="chemometric_classification",
                    statement=(
                        f"{self.k}-nearest-neighbour vote over {components} principal components "
                        f"of {calibration.shape[0]} calibration spectra in "
                        f"{len(set(labels))} classes. Leading component carries "
                        f"{model.explained_variance_ratio[0]:.1%} of the calibration variance."
                    ),
                    claim_type=ClaimType.CORRELATION,
                    source_ref="docs/knowledge/chemometrics.md",
                    conditions=(
                        "PCA assumes variance is the informative direction, which on real spectra "
                        "is often the baseline or the instrument rather than the chemistry"
                    ),
                ),
            ),
            failure_codes=(
                (FailureCode.F14_SHORTCUT_LEARNING,)
                if float(model.explained_variance_ratio[0]) > 0.95
                else ()
            ),
        )


register_method(
    PcaKnn(),
    evidence=("tests/test_chemometrics.py::test_pca_knn_recovers_planted_classes",),
    tags=("baseline", "chemometrics", "classification"),
    notes=(
        "ESTABLISHED under PROJECT_DECISIONS.md D14: textbook method, and the reproduction is our "
        "own planted-truth recovery rather than a third-party validation. Emits "
        "F14_SHORTCUT_LEARNING when the leading principal component carries over 95% of the "
        "calibration variance, because at that point the classifier is almost certainly keyed on "
        "one dominant direction that is unlikely to be chemical."
    ),
)
