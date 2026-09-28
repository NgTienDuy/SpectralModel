"""Run E2. See ``preregistration.md``, committed before this script produced a number.

Usage::

    uv run usis data pull rod --all
    uv run usis data pull qm9s-raman --bulk --max-bytes 16000000
    uv run usis data pull scidata-api-raman --bulk --max-bytes 16000000
    uv run python experiments/E2-intrinsic-linewidth/run.py --corpus data/raw --out <dir>

The corpus directory is the one ``usis data pull`` writes: ``rod/*.rod``,
``qm9s-raman/*.csv.gz`` and ``scidata-api-raman/*.csv.gz``. It is not committed - the loaders and
the committed ROD index reproduce it, and the partial fetches record their own byte spans in
``FETCH.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from usis.core.config import ExperimentConfig
from usis.core.provenance import RunContext
from usis.core.types import Spectrum
from usis.data.sources.qm9s import Qm9sSource
from usis.data.sources.rod import RodSource
from usis.data.sources.scidata_api import SciDataApiSource
from usis.data.splits import SplitKind, make_split
from usis.eval.metrics.statistics import MetricResult
from usis.eval.mrep import ResultCard
from usis.measurement import (
    BaselineDrift,
    ForwardModel,
    IntrinsicBroadening,
    ReadoutNoise,
    Resampling,
    ResolutionConvolution,
    ScaleMode,
    measured_intrinsic_widths,
)
from usis.measurement.diagnostics import (
    SpectralDiagnostics,
    baseline_curvature,
    noise_level,
)
from usis.measurement.forward_model import signal_scale

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The test. No parameter is set from any of these. Preregistration, "What is estimated".
HELD_OUT = (
    "noise_variance_slope",
    "n_bands",
    "dynamic_range",
    "autocorrelation_length_cm1",
)

#: Fitted quantities, checked on a disjoint half. **Not evidence**, and reported separately so a
#: reader cannot mistake six numbers for six results.
MANIPULATION_CHECKS = ("median_band_fwhm_cm1", "band_fwhm_iqr_cm1")

#: The simulated corpus's own broadening, measured from the published bytes (see the QM9S loader).
SIMULATED_FWHM_CM1 = 10.0

#: Seed for the ROD fit/test split. Fixed in the preregistration, not chosen here.
SPLIT_SEED = 20260903

#: ROD's declared instrument response is a grating spectrometer's, taken as Gaussian for the
#: deconvolution. The choice matters: a Lorentzian assumption would subtract linearly and return a
#: different intrinsic width. It is stated rather than hidden because it is an assumption.
INSTRUMENT_SHAPE = "gaussian"


@dataclass(frozen=True)
class Arm:
    """One cell of the 2 x 2 design."""

    name: str
    broadening: bool
    scale_mode: ScaleMode

    @property
    def label(self) -> str:
        """How the arm is written in the tables."""
        return (
            f"{self.name} (broadening {'on ' if self.broadening else 'off'}, "
            f"scale {self.scale_mode.value})"
        )


ARMS = (
    Arm("A0B0", broadening=False, scale_mode=ScaleMode.MEDIAN),
    Arm("A1B0", broadening=True, scale_mode=ScaleMode.MEDIAN),
    Arm("A0B1", broadening=False, scale_mode=ScaleMode.AMPLITUDE),
    Arm("A1B1", broadening=True, scale_mode=ScaleMode.AMPLITUDE),
)


@dataclass(frozen=True)
class Comparison:
    """One statistic, compared under the rule E1 fixed and E2 keeps unchanged."""

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
        """The simulated median falls inside the real interquartile range."""
        return bool(self.real_q25 <= self.simulated_median <= self.real_q75)

    @property
    def standardised_gap(self) -> float:
        """``(median_sim - median_real) / IQR_real``; 0 is a perfect match."""
        if self.real_iqr == 0.0:
            return float("inf") if self.simulated_median != self.real_median else 0.0
        return (self.simulated_median - self.real_median) / self.real_iqr

    def as_dict(self) -> dict[str, object]:
        """JSON view."""
        return {
            "statistic": self.statistic,
            "simulated_median": self.simulated_median,
            "real_median": self.real_median,
            "real_q25": self.real_q25,
            "real_q75": self.real_q75,
            "standardised_gap": self.standardised_gap,
            "comparable": self.comparable,
        }


def _load(corpus: Path) -> tuple[list[Spectrum], list[Spectrum], list[Spectrum]]:
    """Load the three corpora through the committed loaders."""
    rod = []
    for path in sorted((corpus / "rod").glob("*.rod")):
        try:
            rod.append(RodSource().parse_one(path))
        except Exception as exc:  # a skipped record is reported, never hidden
            print(f"  skipped {path.name}: {type(exc).__name__}: {exc}")
    qm9s = list(Qm9sSource("raman").parse_file(_one(corpus / "qm9s-raman", "*.csv.gz")))
    api = list(SciDataApiSource().parse_file(_one(corpus / "scidata-api-raman", "*.csv.gz")))
    return rod, qm9s, api


def _one(directory: Path, pattern: str) -> Path:
    """The single file matching ``pattern``, or a refusal naming what was found instead."""
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"{directory}: expected exactly one {pattern}, found {len(matches)}")
    return matches[0]


def _diagnostics(spectra: Sequence[Spectrum], label: str) -> list[SpectralDiagnostics]:
    """Diagnostics for every spectrum, reporting rather than hiding the ones that fail."""
    out: list[SpectralDiagnostics] = []
    skipped = 0
    for spectrum in spectra:
        try:
            out.append(SpectralDiagnostics.of(spectrum))
        except Exception:
            skipped += 1
    print(f"  {label}: {len(out)} described, {skipped} skipped")
    return out


def _quantiles(
    diagnostics: Sequence[SpectralDiagnostics], field: str
) -> tuple[float, float, float]:
    values = np.array([getattr(d, field) for d in diagnostics], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"),) * 3
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return float(q25), float(q50), float(q75)


def _median(diagnostics: Sequence[SpectralDiagnostics], field: str) -> float:
    return _quantiles(diagnostics, field)[1]


def _resolution_match(spectra: Sequence[Spectrum], seed: int) -> tuple[list[Spectrum], int]:
    """Broaden real spectra up to the simulation's width. The direction physics allows."""
    matched: list[Spectrum] = []
    skipped = 0
    for spectrum in spectra:
        current = spectrum.axis.resolution_fwhm
        if current is None or current >= SIMULATED_FWHM_CM1:
            skipped += 1
            continue
        step = float(np.abs(np.diff(spectrum.axis.values)).mean())
        chain = ForwardModel.of(
            Resampling(step_cm1=step),
            ResolutionConvolution.to_target(current, SIMULATED_FWHM_CM1, shape="lorentzian"),
        )
        try:
            matched.append(chain.apply(spectrum, seed=seed))
        except Exception:
            skipped += 1
    return matched, skipped


def _fit_intrinsic_width(spectra: Sequence[Spectrum]) -> tuple[float, float, int, int]:
    """Median and geometric spread of the per-band intrinsic widths in a real corpus.

    Returns ``(median_fwhm, geometric_sd, n_widths, n_spectra_used)``.

    The spread is taken from the interquartile range of the log widths rather than their standard
    deviation: one mis-detected band merging two neighbours produces a width several times the
    median, and a standard deviation would carry that straight into the distribution the simulation
    is broadened with.
    """
    widths: list[float] = []
    used = 0
    for spectrum in spectra:
        resolution = spectrum.axis.resolution_fwhm
        if resolution is None:
            continue
        try:
            per_band = measured_intrinsic_widths(spectrum, resolution, INSTRUMENT_SHAPE)
        except Exception:
            continue
        if per_band.size:
            widths.extend(float(w) for w in per_band)
            used += 1
    if len(widths) < 20:
        raise SystemExit(f"only {len(widths)} intrinsic widths measured; refusing to fit on that")
    array = np.asarray(widths, dtype=np.float64)
    logs = np.log(array)
    q25, q75 = np.quantile(logs, [0.25, 0.75])
    # 1.349 sigma is the interquartile range of a normal, so this is a robust sigma of log-width.
    sigma_log = float((q75 - q25) / 1.349)
    return float(np.median(array)), float(np.exp(sigma_log)), array.size, used


def _fit_additive(spectra: Sequence[Spectrum], mode: ScaleMode) -> tuple[float, float]:
    """``(noise_relative, baseline_swing)`` measured against ``mode``.

    Measured against the mode the degradation will use, not against the median and then reused:
    transferring a ratio between two different denominators is how the E1 dynamic-range gap
    happened in the first place.
    """
    noise: list[float] = []
    swing: list[float] = []
    for spectrum in spectra:
        try:
            scale = signal_scale(spectrum.intensity, mode)
            noise.append(noise_level(spectrum) / scale)
            swing.append(baseline_curvature(spectrum, scale_mode=mode))
        except Exception:
            continue
    return float(np.median(noise)), float(np.median(swing))


def _degrade(
    spectra: Sequence[Spectrum],
    arm: Arm,
    intrinsic: tuple[float, float],
    additive: tuple[float, float],
    seed: int,
) -> tuple[list[Spectrum], int]:
    """Apply one arm's chain to the simulated corpus."""
    noise_relative, baseline_swing = additive
    steps: list[object] = [
        ReadoutNoise(sigma_relative=noise_relative, scale_mode=arm.scale_mode),
        BaselineDrift(coefficients=(0.0, baseline_swing), scale_mode=arm.scale_mode),
    ]
    if arm.broadening:
        steps.insert(
            0,
            IntrinsicBroadening(
                median_fwhm_cm1=intrinsic[0], geometric_sd=intrinsic[1], shape="lorentzian"
            ),
        )
    model = ForwardModel.of(*steps)  # type: ignore[arg-type]
    out: list[Spectrum] = []
    skipped = 0
    for index, spectrum in enumerate(spectra):
        try:
            out.append(model.apply(spectrum, seed=seed + index))
        except Exception:
            skipped += 1
    return out, skipped


def main() -> int:
    """Run the protocol in the order the preregistration states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    context = RunContext.create("E2-intrinsic-linewidth", seed=args.seed)

    # Step 1 - load.
    rod, qm9s, api = _load(args.corpus)
    print(f"loaded: ROD {len(rod)}, QM9S {len(qm9s)}, Sci Data {len(api)}")

    # Step 2 - split ROD sample-disjoint, and never let the test half near a parameter.
    split = make_split(rod, SplitKind.SAMPLE_DISJOINT, seed=SPLIT_SEED, test_fraction=0.5)
    fit = [rod[i] for i in split.train]
    test = [rod[i] for i in split.test]
    print(
        f"ROD split {split.kind}: fit {len(fit)}, test {len(test)} "
        f"({len(split.train_groups)}/{len(split.test_groups)} specimens), seed {SPLIT_SEED}"
    )

    # Step 3 - fit the intrinsic width distribution, on the fit half only.
    median_fwhm, geometric_sd, n_widths, n_used = _fit_intrinsic_width(fit)
    print(
        f"intrinsic width fitted on {n_used} ROD fit-half spectra ({n_widths} bands): "
        f"median {median_fwhm:.2f} cm^-1, geometric sd {geometric_sd:.3f}"
    )

    # Step 4 - resolution-match the real test half up to the simulation.
    matched, skipped = _resolution_match(test, args.seed)
    print(f"resolution-matched {len(matched)} ROD test-half spectra, skipped {skipped}")
    print("diagnostics:")
    real_diag = _diagnostics(matched, "real (ROD test half, matched)")

    estimation_corpus = [*fit, *api]
    results: dict[str, object] = {}
    arm_rows: list[dict[str, object]] = []

    for arm in ARMS:
        noise_relative, baseline_swing = _fit_additive(estimation_corpus, arm.scale_mode)
        degraded, dropped = _degrade(
            qm9s, arm, (median_fwhm, geometric_sd), (noise_relative, baseline_swing), args.seed
        )
        print(
            f"\n{arm.label}: noise_relative={noise_relative:.4g}, "
            f"baseline_swing={baseline_swing:.4g}, degraded {len(degraded)}, dropped {dropped}"
        )
        sim_diag = _diagnostics(degraded, f"simulated ({arm.name})")

        held_out = [
            Comparison(statistic, _median(sim_diag, statistic), *_reorder(real_diag, statistic))
            for statistic in HELD_OUT
        ]
        checks = [
            Comparison(statistic, _median(sim_diag, statistic), *_reorder(real_diag, statistic))
            for statistic in MANIPULATION_CHECKS
        ]
        for comparison in [*held_out, *checks]:
            role = "TEST " if comparison.statistic in HELD_OUT else "check"
            verdict = "COMPARABLE" if comparison.comparable else "not comparable"
            print(
                f"  {role} {comparison.statistic:28} sim {comparison.simulated_median:11.4g}  "
                f"real {comparison.real_median:11.4g} "
                f"[{comparison.real_q25:.4g}, {comparison.real_q75:.4g}]  "
                f"gap {comparison.standardised_gap:+7.2f} IQR  {verdict}"
            )
        n_comparable = sum(c.comparable for c in held_out)
        total_gap = float(sum(abs(c.standardised_gap) for c in held_out))
        print(f"  => {n_comparable}/{len(held_out)} comparable, total |gap| {total_gap:.2f} IQR")
        arm_rows.append(
            {
                "arm": arm.name,
                "broadening": arm.broadening,
                "scale_mode": arm.scale_mode.value,
                "noise_relative": noise_relative,
                "baseline_swing": baseline_swing,
                "n_degraded": len(degraded),
                "n_dropped": dropped,
                "n_comparable": n_comparable,
                "total_absolute_gap": total_gap,
                "held_out": [c.as_dict() for c in held_out],
                "manipulation_checks": [c.as_dict() for c in checks],
            }
        )

    results = {
        "n": {
            "rod_total": len(rod),
            "rod_fit": len(fit),
            "rod_test": len(test),
            "rod_test_matched": len(matched),
            "qm9s": len(qm9s),
            "scidata": len(api),
        },
        "split": {"kind": str(split.kind), "seed": SPLIT_SEED, "test_fraction": 0.5},
        "intrinsic_width_fit": {
            "median_fwhm_cm1": median_fwhm,
            "geometric_sd": geometric_sd,
            "n_bands": n_widths,
            "n_spectra": n_used,
            "instrument_shape_assumed": INSTRUMENT_SHAPE,
        },
        "held_out_statistics": list(HELD_OUT),
        "manipulation_checks": list(MANIPULATION_CHECKS),
        "arms": arm_rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_card(args, context, results)
    return 0


def _reorder(
    diagnostics: Sequence[SpectralDiagnostics], statistic: str
) -> tuple[float, float, float]:
    """``(median, q25, q75)`` in the order :class:`Comparison` takes them."""
    q25, q50, q75 = _quantiles(diagnostics, statistic)
    return q50, q25, q75


def _write_card(args: argparse.Namespace, context: RunContext, results: dict[str, object]) -> None:
    """Write the result card, stating the verdict in the terms the preregistration fixed."""
    arms = list(results["arms"])  # type: ignore[call-overload]
    control = next(a for a in arms if a["arm"] == "A0B0")
    best = min(arms, key=lambda a: (-a["n_comparable"], a["total_absolute_gap"]))
    fit = results["intrinsic_width_fit"]

    config = ExperimentConfig(
        name="E2-intrinsic-linewidth",
        seed=args.seed,
        split_kind="sim2real",
        values={
            "seed": args.seed,
            "corpus": str(args.corpus),
            "split_seed": SPLIT_SEED,
            **{f"fit_{k}": v for k, v in fit.items()},  # type: ignore[union-attr]
        },
        source_path="experiments/E2-intrinsic-linewidth/preregistration.md",
    )
    metrics = tuple(
        MetricResult.from_scores(
            f"{arm['arm']} total |gap| over the held-out statistics (IQR units)",
            [abs(c["standardised_gap"]) for c in arm["held_out"]],
            seed=args.seed,
        )
        for arm in arms
    )
    shows = [
        f"Control arm A0B0 (E1's configuration on this corpus): {control['n_comparable']} of "
        f"{len(HELD_OUT)} held-out statistics comparable, total |gap| "
        f"{control['total_absolute_gap']:.2f} IQR.",
        f"Best arm {best['arm']}: {best['n_comparable']} of {len(HELD_OUT)} comparable, total "
        f"|gap| {best['total_absolute_gap']:.2f} IQR.",
        "Parameters were estimated on the ROD fit half plus the whole Sci Data corpus, both "
        "disjoint from the ROD test half every comparison is made against.",
    ]
    does_not_show = [
        "The two manipulation checks (median_band_fwhm_cm1, band_fwhm_iqr_cm1) are FITTED "
        "quantities and are not evidence. They verify the operator does what it was set to do.",
        "Nothing about chemistry: band positions are not compared, only their count and width.",
        "Nothing about a trained model's transferability; no model is involved.",
        "Nothing about matched samples: real is minerals and solvents, simulated is small "
        "organics, and the spectral ranges differ. That confound is not removed.",
        "Nothing about whether the fitted intrinsic width generalises past ROD's instrument "
        "family: it is estimated from one corpus and applied to a different chemistry.",
        "Nothing that rests on QM9S's 10 cm^-1 being physical. It is a publisher's arbitrary "
        "choice, and the resolution-matching step treats it as an instrument response, which is "
        "an assumption inherited from E1 rather than tested here.",
        "The QM9S and Sci Data corpora are partial fetches taken at fixed byte offsets, not "
        "random samples; see FETCH.json in each dataset directory.",
    ]
    notes = [
        f"intrinsic width fitted: median {fit['median_fwhm_cm1']:.2f} cm^-1, "  # type: ignore[index]
        f"geometric sd {fit['geometric_sd']:.3f}, from {fit['n_bands']} bands "  # type: ignore[index]
        f"over {fit['n_spectra']} ROD fit-half spectra",  # type: ignore[index]
    ]
    for arm in arms:
        notes.append(
            f"{arm['arm']} (broadening {'on' if arm['broadening'] else 'off'}, scale "
            f"{arm['scale_mode']}): {arm['n_comparable']}/{len(HELD_OUT)} comparable, total |gap| "
            f"{arm['total_absolute_gap']:.2f} IQR"
        )
        for comparison in arm["held_out"]:
            notes.append(
                f"    TEST  {comparison['statistic']}: sim {comparison['simulated_median']:.4g} "
                f"vs real {comparison['real_median']:.4g} "
                f"[{comparison['real_q25']:.4g}, {comparison['real_q75']:.4g}], gap "
                f"{comparison['standardised_gap']:+.2f} IQR - "
                f"{'comparable' if comparison['comparable'] else 'NOT comparable'}"
            )
        for comparison in arm["manipulation_checks"]:
            notes.append(
                f"    check {comparison['statistic']}: sim {comparison['simulated_median']:.4g} "
                f"vs real {comparison['real_median']:.4g}, gap "
                f"{comparison['standardised_gap']:+.2f} IQR (fitted, not evidence)"
            )

    title = "E2 — Does an intrinsic linewidth drawn from a distribution close the sim-to-real gap?"
    card = ResultCard(
        title=title,
        run_context=context,
        config=config,
        metrics=metrics,
        split=None,
        datasets=("rod", "qm9s-raman", "scidata-api-raman"),
        shows=tuple(shows),
        does_not_show=tuple(does_not_show),
        notes=tuple(notes),
    )
    path = card.write(args.out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
