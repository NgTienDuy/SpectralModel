"""Detector-saturation detection: the smallest honest method in the repository.

It exists for three reasons. It is section 1 of the Sample Report - the measurement audit `usis
audit` ships at M2 - in its simplest form. It is the trivial half of the pair that proves a method
can be added without touching anything else. And it is the entry that forced the maturity ladder to
be read carefully (`PROJECT_DECISIONS.md` D14): a flat top at the detector ceiling is not a
contested claim, but it also has no paper behind it, so ``ESTABLISHED`` here means *the physics is
definitional and we reproduced the detection against ground truth we injected ourselves* - which is
stated in the entry's note rather than implied by the level.

Two gaps in the core enums showed up while writing it and are deliberately not patched here, since
both are additive changes to `usis/core/types.py` that belong in one ADR:
:class:`~usis.core.types.TaskKind` has no measurement-audit family (``PREPROCESSING_QUALITY`` is
the nearest), and :class:`~usis.core.types.ClaimType` has no member for a direct observation, so a
clipped-sample count has to travel as ``CORRELATION``.
"""

from __future__ import annotations

import numpy as np

from usis.core.provenance import RunContext
from usis.core.types import (
    ClaimType,
    Evidence,
    FailureCode,
    IntensityKind,
    Maturity,
    MeasurementVerdict,
    TaskKind,
)
from usis.methods.assumptions import saturated_fraction
from usis.methods.base import Citation, CostModel, MethodBase, MethodResult, Requirements
from usis.methods.inputs import MethodInputs
from usis.methods.registry import register_method

__all__ = ["SaturationAudit"]

#: Above this fraction of clipped samples the spectrum is rejected outright rather than degraded.
#: Not a measured constant: 1% of a 2000-point spectrum is twenty samples of invented intensity,
#: which is more than any quantitative claim can absorb. Stated so it can be argued with.
REJECT_ABOVE_FRACTION = 0.01


class SaturationAudit(MethodBase):
    """Reports the fraction of samples sitting on a clipped plateau, and a measurement verdict.

    A single sample at the maximum is the tallest band; two or more consecutive samples at exactly
    the maximum is the detector ceiling, because a real band apex is a curve and lands on one
    sample. The method is therefore looking for flat tops, not for large values, which is why it
    needs no reference level and no exposure metadata.
    """

    method_id = "audit.saturation"
    version = "1.0.0"
    task = TaskKind.PREPROCESSING_QUALITY
    maturity = Maturity.ESTABLISHED
    provenance = Citation(
        title="Saturation as a hard nonlinearity in optical detection",
        internal_ref="docs/knowledge/measurement.md#7",
        note=(
            "Definitional rather than contested; the reproduction referenced in the registry "
            "entry injects known clipping with usis.measurement.artifacts.Saturation and recovers "
            "the injected fraction."
        ),
    )
    requires = Requirements(
        intensity_kinds=frozenset({IntensityKind.COUNTS, IntensityKind.ARBITRARY}),
    )
    cost = CostModel(
        compute_seconds_per_spectrum=1e-5,
        note="one pass over the intensity array",
    )

    def compute(self, inputs: MethodInputs, ctx: RunContext) -> MethodResult:
        """Count clipped plateaus in the single input spectrum."""
        spectrum = inputs.primary
        intensity = np.asarray(spectrum.intensity, dtype=np.float64)
        fraction = saturated_fraction(spectrum)
        clipped = round(fraction * intensity.size)
        if fraction > REJECT_ABOVE_FRACTION:
            verdict = MeasurementVerdict.REJECT
            statement = (
                f"{clipped} of {intensity.size} samples ({fraction:.2%}) lie on a plateau at the "
                f"observed maximum {intensity.max():.4g}. The incident intensity over those "
                "samples is not recoverable, so no quantitative claim may be made from this "
                "spectrum."
            )
        elif clipped > 0:
            verdict = MeasurementVerdict.DEGRADED
            statement = (
                f"{clipped} of {intensity.size} samples ({fraction:.2%}) lie on a plateau at the "
                f"observed maximum {intensity.max():.4g}. Band intensities in the affected region "
                "are lower bounds, not measurements."
            )
        else:
            verdict = MeasurementVerdict.USABLE
            statement = (
                f"no plateau at the observed maximum {intensity.max():.4g}: the apex lands on a "
                "single sample, as an unclipped band does."
            )
        return self._result(
            values={
                "saturated_fraction": fraction,
                "clipped_samples": float(clipped),
                "n_samples": float(intensity.size),
            },
            labels={"verdict": str(verdict), "spectrum_id": spectrum.spectrum_id},
            evidence=(
                Evidence(
                    kind="measurement_audit",
                    statement=statement,
                    claim_type=ClaimType.CORRELATION,
                    source_ref="docs/knowledge/measurement.md#7",
                    conditions="plateau detection requires an untransformed intensity axis",
                ),
            ),
            failure_codes=(FailureCode.F05_SATURATION,) if clipped else (),
        )


register_method(
    SaturationAudit(),
    evidence=("tests/test_methods_builtin.py::test_saturation_audit_recovers_injected_clipping",),
    tags=("audit", "measurement"),
    notes=(
        "ESTABLISHED under PROJECT_DECISIONS.md D14: the physics is definitional and the "
        "reproduction is our own injection test, not a third-party validation. The reject "
        "threshold of 1% clipped samples is a stated convention, not a measured constant."
    ),
)
