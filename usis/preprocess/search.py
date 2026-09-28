"""The pipeline search, and the sensitivity number that comes out of it.

Two things live here, and they answer different questions.

**The search** ranks a set of pipelines on a task, with a bootstrap interval per pipeline, so
"which preprocessing" becomes an empirical question with an error bar instead of a lab convention.
The set always includes an :func:`identity` arm - *no preprocessing at all* - because a search over
five ways of correcting a baseline cannot discover that correcting it was unnecessary.

**The sensitivity** is novelty candidate N8: for one spectrum, how much does the answer move when
only the pipeline changes? Reported per spectrum, it is a candidate abstention signal that costs
nothing but compute; reported per corpus, it says how much of a published number is a property of
the sample and how much is a property of the analyst's habits. This module computes it for any
scalar readout, so it is available now, on diagnostics, before the identity task exists.

The pipelines in :func:`defensible_pipelines` are *defensible*, not recommended. Each is something a
competent spectroscopist would do and defend; the point of enumerating them is that they disagree,
and the disagreement is data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from usis.core.errors import UsisError
from usis.core.types import Spectrum
from usis.eval.metrics.statistics import MetricResult
from usis.preprocess.baseline import PolynomialBaseline, RubberbandBaseline, SnipBaseline
from usis.preprocess.pipeline import Pipeline
from usis.preprocess.scale import Normalisation, NormaliseMode
from usis.preprocess.smoothing import MovingAverage, SavitzkyGolay
from usis.preprocess.spikes import Despike

__all__ = [
    "PipelineScore",
    "SearchError",
    "Sensitivity",
    "defensible_pipelines",
    "identity",
    "search",
    "sensitivity",
]


class SearchError(UsisError):
    """A pipeline search or sensitivity measurement was asked for something it cannot report."""


def identity() -> Pipeline:
    """The control arm: no preprocessing at all.

    Its presence is not a formality. RamanBench found tabular models beating domain-specific
    architectures; the analogous outcome here - raw spectra beating every correction - is a real
    possibility, and a search that cannot express it would never report it.
    """
    return Pipeline(steps=(), name="identity")


def defensible_pipelines() -> tuple[Pipeline, ...]:
    """A small, deliberately opinionated set of pipelines that disagree with each other.

    Chosen so that the axes of disagreement are separable rather than tangled: three baseline
    estimators at one smoothing setting, two smoothing settings at one baseline, one normalisation
    contrast, one ordering contrast (smooth-then-baseline against baseline-then-smooth), and the
    identity control. Eight arms, each differing from a neighbour in one respect.
    """
    smooth = SavitzkyGolay(window=9, order=3)
    vector = Normalisation(NormaliseMode.VECTOR)
    return (
        identity(),
        Pipeline.of(SnipBaseline(iterations=40), smooth, vector, name="snip+sg+vector"),
        Pipeline.of(PolynomialBaseline(order=3), smooth, vector, name="poly3+sg+vector"),
        Pipeline.of(RubberbandBaseline(), smooth, vector, name="rubberband+sg+vector"),
        Pipeline.of(
            SnipBaseline(iterations=40),
            MovingAverage(window=5),
            vector,
            name="snip+boxcar+vector",
        ),
        Pipeline.of(
            SnipBaseline(iterations=40),
            SavitzkyGolay(window=21, order=3),
            vector,
            name="snip+sg21+vector",
        ),
        Pipeline.of(
            SnipBaseline(iterations=40),
            smooth,
            Normalisation(NormaliseMode.SNV),
            name="snip+sg+snv",
        ),
        Pipeline.of(smooth, SnipBaseline(iterations=40), vector, name="sg+snip+vector"),
        Pipeline.of(
            Despike(), SnipBaseline(iterations=40), smooth, vector, name="despike+snip+sg+vector"
        ),
    )


@dataclass(frozen=True)
class PipelineScore:
    """One pipeline's score on a task, with its interval and its failures."""

    pipeline: Pipeline
    metric: MetricResult
    n_failed: int
    failures: tuple[str, ...]

    @property
    def label(self) -> str:
        """The pipeline's name if it has one, else its structural label."""
        return self.pipeline.name or self.pipeline.label

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable row for the comparison record."""
        return {
            "pipeline": self.pipeline.to_dict(),
            "label": self.label,
            "metric": self.metric.to_dict(),
            "n_failed": self.n_failed,
            "failures": list(self.failures),
        }


def search(
    spectra: Sequence[Spectrum],
    score: Callable[[Spectrum], float],
    pipelines: Sequence[Pipeline] | None = None,
    *,
    metric_name: str = "score",
    groups: Sequence[object] | None = None,
    seed: int = 0,
) -> tuple[PipelineScore, ...]:
    """Score every pipeline on every spectrum, ranked best first, each with a bootstrap interval.

    ``score`` maps a preprocessed spectrum to a number where higher is better. It is supplied by
    the caller rather than chosen here because the right readout is a property of the task, and a
    search that picked its own objective would optimise the wrong thing convincingly.

    A pipeline that raises on some spectra is **not** dropped: its failures are counted and its
    score is computed on the rest, with the count reported. Silently scoring a pipeline on the
    easy subset it happened to survive is how a fragile pipeline wins a leaderboard.

    Args:
        spectra: the corpus.
        score: per-spectrum readout, higher is better.
        pipelines: the arms. Defaults to :func:`defensible_pipelines`.
        metric_name: name recorded on the metric.
        groups: group ids for the bootstrap, when spectra are replicates of shared samples.
            Passing them is not optional in spirit: ignoring replicate structure narrows the
            interval by roughly the square root of the replicate count.
        seed: bootstrap seed.

    Raises:
        SearchError: if no spectra were supplied, or if some pipeline scored nothing at all.
    """
    if not spectra:
        raise SearchError("a pipeline search needs at least one spectrum")
    arms = tuple(pipelines) if pipelines is not None else defensible_pipelines()
    results = []
    for pipeline in arms:
        scores: list[float] = []
        kept_groups: list[object] = []
        failures: list[str] = []
        for index, spectrum in enumerate(spectra):
            try:
                scores.append(float(score(pipeline.apply(spectrum))))
            except UsisError as exc:
                failures.append(f"{spectrum.spectrum_id}: {type(exc).__name__}: {exc}")
                continue
            if groups is not None:
                kept_groups.append(groups[index])
        if not scores:
            raise SearchError(
                f"pipeline {pipeline.name or pipeline.label!r} scored nothing: every spectrum "
                f"failed. First failure: {failures[0] if failures else 'unknown'}"
            )
        metric = MetricResult.from_scores(
            metric_name,
            scores,
            seed=seed,
            groups=kept_groups if groups is not None else None,
        )
        results.append(
            PipelineScore(
                pipeline=pipeline,
                metric=metric,
                n_failed=len(failures),
                failures=tuple(failures),
            )
        )
    return tuple(sorted(results, key=lambda r: r.metric.mean, reverse=True))


@dataclass(frozen=True)
class Sensitivity:
    """How much one readout moves when only the pipeline changes - N8's quantity.

    ``spread`` is the peak-to-peak range across pipelines and ``relative_spread`` divides it by the
    absolute median, so it can be compared across readouts with different units. ``n_failed`` is
    part of the answer: a spectrum on which three of eight pipelines raise is a spectrum whose
    analysis is pipeline-dependent in the strongest possible sense.
    """

    spectrum_id: str
    values: dict[str, float]
    median: float
    spread: float
    relative_spread: float
    n_failed: int

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view."""
        return {
            "spectrum_id": self.spectrum_id,
            "values": dict(self.values),
            "median": self.median,
            "spread": self.spread,
            "relative_spread": self.relative_spread,
            "n_failed": self.n_failed,
        }


def sensitivity(
    spectrum: Spectrum,
    readout: Callable[[Spectrum], float],
    pipelines: Sequence[Pipeline] | None = None,
) -> Sensitivity:
    """Measure how much ``readout`` depends on the pipeline, for one spectrum.

    Raises:
        SearchError: if fewer than two pipelines produced a finite readout, since a spread over one
            value is not a spread. That refusal is the point: reporting 0.0 sensitivity because
            everything else failed would invert the meaning of the number.
    """
    arms = tuple(pipelines) if pipelines is not None else defensible_pipelines()
    values: dict[str, float] = {}
    failed = 0
    for pipeline in arms:
        label = pipeline.name or pipeline.label
        try:
            value = float(readout(pipeline.apply(spectrum)))
        except UsisError:
            failed += 1
            continue
        if not np.isfinite(value):
            failed += 1
            continue
        values[label] = value
    if len(values) < 2:
        raise SearchError(
            f"{spectrum.spectrum_id}: only {len(values)} of {len(arms)} pipelines produced a "
            "finite readout, so there is no spread to report. Reporting zero sensitivity here "
            "would say the opposite of what happened."
        )
    array = np.array(list(values.values()), dtype=np.float64)
    median = float(np.median(array))
    spread = float(array.max() - array.min())
    scale = abs(median) if median != 0.0 else float(np.abs(array).max())
    return Sensitivity(
        spectrum_id=spectrum.spectrum_id,
        values=values,
        median=median,
        spread=spread,
        relative_spread=spread / scale if scale else float("inf"),
        n_failed=failed,
    )
