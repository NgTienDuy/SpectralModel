"""Run E1. See ``preregistration.md``, which was committed before this script produced a number.

Usage::

    uv run python experiments/E1-sim2real-comparability/run.py --corpus <dir> --out <dir>

The corpus directory holds ``rod/*.rod``, ``qm9s/raman.csv.gz`` and ``api/api.csv.gz`` as fetched
from the published sources. It is not committed: the data is large and the loaders reproduce it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from usis.core.config import ExperimentConfig
from usis.core.provenance import RunContext
from usis.core.types import Spectrum
from usis.data.sources.qm9s import Qm9sSource
from usis.data.sources.rod import RodSource
from usis.data.sources.scidata_api import SciDataApiSource
from usis.eval.metrics.statistics import MetricResult
from usis.eval.mrep import ResultCard
from usis.measurement import (
    BaselineDrift,
    ForwardModel,
    ReadoutNoise,
    Resampling,
    ResolutionConvolution,
    ScaleMode,
)
from usis.measurement.diagnostics import SpectralDiagnostics

#: Held out from parameter estimation; these are the test. Preregistration section "The statistics".
HELD_OUT = (
    "noise_variance_slope",
    "n_bands",
    "median_band_fwhm_cm1",
    "dynamic_range",
    "autocorrelation_length_cm1",
)
#: Used to set the degradation parameters, and therefore excluded from the comparison.
ESTIMATED = ("noise_relative", "baseline_swing")

#: The simulated corpus's own broadening, measured in a previous session and recorded on the loader.
SIMULATED_FWHM_CM1 = 10.0


@dataclass(frozen=True)
class Comparison:
    """One statistic, compared per the pre-registered rule."""

    statistic: str
    simulated_median: float
    real_median: float
    real_q25: float
    real_q75: float

    @property
    def real_iqr(self) -> float:
        """Interquartile range of the real corpus."""
        return self.real_q75 - self.real_q25

    @property
    def comparable(self) -> bool:
        """Pre-registered rule: the simulated median falls inside the real interquartile range."""
        return bool(self.real_q25 <= self.simulated_median <= self.real_q75)

    @property
    def standardised_gap(self) -> float:
        """``(median_sim - median_real) / IQR_real``; 0 is a perfect match."""
        if self.real_iqr == 0.0:
            return float("inf") if self.simulated_median != self.real_median else 0.0
        return (self.simulated_median - self.real_median) / self.real_iqr


def _load(corpus: Path) -> tuple[list[Spectrum], list[Spectrum], list[Spectrum]]:
    """Load the three corpora through the committed loaders."""
    rod = [RodSource().parse_one(p) for p in sorted((corpus / "rod").glob("*.rod"))]
    qm9s = list(Qm9sSource("raman").parse_file(corpus / "qm9s" / "raman.csv.gz"))
    api = list(SciDataApiSource().parse_file(corpus / "api" / "api.csv.gz"))
    return rod, qm9s, api


def _diagnostics(spectra: list[Spectrum]) -> list[SpectralDiagnostics]:
    """Diagnostics for every spectrum, skipping any the estimators cannot describe."""
    out: list[SpectralDiagnostics] = []
    for spectrum in spectra:
        try:
            out.append(SpectralDiagnostics.of(spectrum))
        except Exception as exc:  # a skipped spectrum is reported, never hidden
            print(f"  skipped {spectrum.spectrum_id}: {type(exc).__name__}: {exc}")
    return out


def _median(diagnostics: list[SpectralDiagnostics], field: str) -> float:
    values = np.array([getattr(d, field) for d in diagnostics], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _quantiles(diagnostics: list[SpectralDiagnostics], field: str) -> tuple[float, float, float]:
    values = np.array([getattr(d, field) for d in diagnostics], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"),) * 3
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return float(q25), float(q50), float(q75)


def main() -> int:
    """Run the protocol in the order the preregistration states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    context = RunContext.create("E1-sim2real-comparability", seed=args.seed)
    rod, qm9s, api = _load(args.corpus)
    print(f"loaded: ROD {len(rod)}, QM9S {len(qm9s)}, Sci Data {len(api)}")

    # Step 2 - parameters from the real corpora only.
    print("diagnostics: real (ROD)")
    rod_diag = _diagnostics(rod)
    print("diagnostics: real (Sci Data)")
    api_diag = _diagnostics(api)
    real_diag = rod_diag + api_diag

    noise_relative = _median(real_diag, "noise_relative")
    baseline_swing = _median(real_diag, "baseline_swing")
    print(
        f"estimated from real: noise_relative={noise_relative:.4g}, "
        f"baseline_swing={baseline_swing:.4g}"
    )

    # Step 3 - degrade the simulation with exactly those parameters.
    # ScaleMode.MEDIAN is stated explicitly here because it was implicit when this ran, and this
    # experiment is what measured its cost: on a simulated spectrum the median is the floor between
    # bands, so these amplitudes are ratios against nearly nothing. It is deliberately NOT changed -
    # the script has to keep reproducing the run written up in findings.md. The corrected mode is an
    # arm of the follow-up experiment, not an edit to this one.
    degrade = ForwardModel.of(
        ReadoutNoise(sigma_relative=noise_relative, scale_mode=ScaleMode.MEDIAN),
        BaselineDrift(coefficients=(0.0, baseline_swing), scale_mode=ScaleMode.MEDIAN),
    )
    degraded = [degrade.apply(s, seed=args.seed + i) for i, s in enumerate(qm9s)]

    # Bring the real corpus up to the simulation's resolution: the only direction physics allows.
    matched: list[Spectrum] = []
    skipped_no_resolution = 0
    for spectrum in rod:
        current = spectrum.axis.resolution_fwhm
        if current is None or current >= SIMULATED_FWHM_CM1:
            skipped_no_resolution += 1
            continue
        step = float(np.abs(np.diff(spectrum.axis.values)).mean())
        chain = ForwardModel.of(
            Resampling(step_cm1=step),
            ResolutionConvolution.to_target(current, SIMULATED_FWHM_CM1, shape="lorentzian"),
        )
        matched.append(chain.apply(spectrum, seed=args.seed))
    print(f"resolution-matched {len(matched)} ROD spectra, skipped {skipped_no_resolution}")

    print("diagnostics: degraded simulation")
    degraded_diag = _diagnostics(degraded)
    print("diagnostics: resolution-matched real")
    matched_diag = _diagnostics(matched)

    # Step 4 - compare the held-out statistics.
    comparisons = []
    for statistic in HELD_OUT:
        q25, q50, q75 = _quantiles(matched_diag, statistic)
        comparisons.append(
            Comparison(
                statistic=statistic,
                simulated_median=_median(degraded_diag, statistic),
                real_median=q50,
                real_q25=q25,
                real_q75=q75,
            )
        )

    for comparison in comparisons:
        verdict = "COMPARABLE" if comparison.comparable else "NOT comparable"
        print(
            f"  {comparison.statistic:32} sim {comparison.simulated_median:12.4g}  "
            f"real {comparison.real_median:12.4g} [{comparison.real_q25:.4g}, "
            f"{comparison.real_q75:.4g}]  gap {comparison.standardised_gap:+7.2f} IQR  {verdict}"
        )

    payload = {
        "n": {"rod": len(rod), "qm9s": len(qm9s), "api": len(api), "matched": len(matched)},
        "parameters_from_real": {
            "noise_relative": noise_relative,
            "baseline_swing": baseline_swing,
        },
        "estimated_statistics": list(ESTIMATED),
        "held_out": [
            {
                "statistic": c.statistic,
                "simulated_median": c.simulated_median,
                "real_median": c.real_median,
                "real_q25": c.real_q25,
                "real_q75": c.real_q75,
                "standardised_gap": c.standardised_gap,
                "comparable": c.comparable,
            }
            for c in comparisons
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _write_card(args, context, comparisons, degraded_diag, matched_diag, payload)
    return 0


def _write_card(
    args: argparse.Namespace,
    context: RunContext,
    comparisons: list[Comparison],
    degraded_diag: list[SpectralDiagnostics],
    matched_diag: list[SpectralDiagnostics],
    payload: dict[str, object],
) -> None:
    """Write the result card, with the verdict stated in the terms the preregistration fixed."""
    n_comparable = sum(c.comparable for c in comparisons)
    config = ExperimentConfig(
        name="E1-sim2real-comparability",
        seed=args.seed,
        split_kind="sim2real",
        values={"seed": args.seed, "corpus": str(args.corpus), **payload["parameters_from_real"]},
        source_path="experiments/E1-sim2real-comparability/preregistration.md",
    )
    metrics = tuple(
        MetricResult.from_scores(
            f"{statistic} (degraded simulation)",
            [getattr(d, statistic) for d in degraded_diag if np.isfinite(getattr(d, statistic))],
            seed=args.seed,
        )
        for statistic in HELD_OUT
    )
    shows = [
        f"{n_comparable} of {len(comparisons)} held-out statistics are comparable under the "
        "pre-registered rule (simulated median inside the real interquartile range).",
        "Parameters came from the real corpora and the held-out statistics were never used to set "
        "them, so the comparison is not circular.",
    ]
    does_not_show = [
        "Nothing about chemistry: band positions are not compared, only their count and width.",
        "Nothing about a trained model's transferability; no model is involved.",
        "Nothing about matched samples: the corpora differ in spectral range and in chemistry "
        "(minerals and solvents versus small organics), and that confound is not removed.",
        "Nothing about the Sci Data corpus's resolution, which is unrecorded, so it could not be "
        "resolution-matched and contributes only to the parameter estimate.",
    ]
    card = ResultCard(
        title="E1 — Are degraded simulated spectra statistically comparable to real ones?",
        run_context=context,
        config=config,
        metrics=metrics,
        split=None,
        datasets=("rod", "qm9s-raman", "scidata-api-raman"),
        shows=tuple(shows),
        does_not_show=tuple(does_not_show),
        notes=tuple(
            f"{c.statistic}: simulated {c.simulated_median:.4g} vs real {c.real_median:.4g} "
            f"[{c.real_q25:.4g}, {c.real_q75:.4g}], gap {c.standardised_gap:+.2f} IQR — "
            f"{'comparable' if c.comparable else 'NOT comparable'}"
            for c in comparisons
        ),
    )
    path = card.write(args.out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
