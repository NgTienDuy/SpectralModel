"""Turning a committed config into a runnable benchmark, with the CLI knowing none of them.

`CLAUDE.md`: *every experiment = committed Hydra config + seed + git SHA + a result card.* That
makes the config, not the command line, the thing a comparison is defined by - and it puts one more
requirement on the harness's extension story. Adding a benchmark must touch its own file and this
registry, and never ``usis/cli/main.py``: a command that grew an ``elif`` per benchmark would make
the CLI the place every new dataset has to be merged into.

The corpus filtering here is the part worth reading twice. A spectral window is chosen so that
every spectrum can be put on one grid, and every spectrum that does not cover the window is
dropped. That subset is never random - coverage of 200-1200 cm-1 correlates with instrument, with
grating, and with how the deposit was made - so the count is recorded on the record rather than
being absorbed into "n = 1081".
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from usis.core.errors import UsisError
from usis.data.corpus import CorpusLoad, load_corpus
from usis.data.pull import DEFAULT_DATA_DIR
from usis.eval.tasks.retrieval import OpenSetRetrievalTask
from usis.preprocess.base import PreprocessStep
from usis.preprocess.baseline import PolynomialBaseline, RubberbandBaseline, SnipBaseline
from usis.preprocess.pipeline import Pipeline
from usis.preprocess.scale import Normalisation, NormaliseMode, ResampleToGrid

if TYPE_CHECKING:
    from collections.abc import Callable

    from usis.compare.task import ComparisonTask
    from usis.core.config import ExperimentConfig
    from usis.core.types import Spectrum

__all__ = ["TASK_BUILDERS", "BuildError", "BuiltTask", "build_pipeline", "build_task"]


class BuildError(UsisError):
    """A config does not describe a benchmark that can be built."""


#: Baseline-correction steps a config may name. ``none`` is explicit rather than a missing key,
#: because "no baseline correction" is a decision about the data and should be written down.
_BASELINES: dict[str, Callable[[], PreprocessStep]] = {
    "snip": SnipBaseline,
    "polynomial": PolynomialBaseline,
    "rubberband": RubberbandBaseline,
}


class BuiltTask:
    """A benchmark plus the account of what building it dropped.

    Not a dataclass with the task inlined into the record: the counts here are about the *corpus*,
    not about any method, and they have to reach the record's notes before a single number is read.
    """

    def __init__(
        self,
        task: ComparisonTask,
        *,
        load: CorpusLoad,
        pipeline: Pipeline,
        n_window: int,
        n_unlabelled: int,
    ) -> None:
        self.task = task
        self.load = load
        self.pipeline = pipeline
        self.n_window = n_window
        self.n_unlabelled = n_unlabelled

    @property
    def corpus_notes(self) -> tuple[str, ...]:
        """What the corpus assembly dropped, and why it is not a random subset."""
        return (
            f"Corpus: {self.load.n_files} mirrored record(s) of "
            f"{self.load.dataset_id}, {len(self.load.spectra)} spectra parsed, "
            f"{self.load.n_skipped} record(s) unparseable ({self.load.skip_reasons or 'none'}).",
            f"{self.n_unlabelled} spectrum/spectra carried no identity and were dropped: a "
            "retrieval answer about an unlabelled sample cannot be right or wrong.",
            f"{self.n_window} spectrum/spectra did not cover the analysis window and were "
            "dropped. That subset is NOT random - window coverage correlates with instrument, "
            "grating and deposit practice - so the benchmark is over the covering subset and any "
            "claim from it is conditional on that.",
            f"Preprocessing applied to every spectrum, recorded on each: {self.pipeline.label}",
        )


def build_pipeline(config: ExperimentConfig) -> Pipeline:
    """Build the recorded preprocessing chain a comparison config declares.

    The grid comes first and is stated in absolute wavenumbers, so every spectrum leaves this
    pipeline on the identical axis - which is what makes ``requires_common_axis`` satisfiable at
    all, and what a matrix method needs before channel k means the same thing twice.
    """
    steps: list[PreprocessStep] = [
        ResampleToGrid(
            low_cm1=float(config.require("window.low_cm1")),
            high_cm1=float(config.require("window.high_cm1")),
            step_cm1=float(config.require("window.step_cm1")),
        )
    ]
    baseline = str(config.get("preprocess.baseline", "none")).lower()
    if baseline not in {*_BASELINES, "none"}:
        raise BuildError(
            f"preprocess.baseline={baseline!r} is not one of {sorted({*_BASELINES, 'none'})}"
        )
    if baseline != "none":
        steps.append(_BASELINES[baseline]())
    normalise = str(config.get("preprocess.normalise", "none")).lower()
    if normalise != "none":
        try:
            mode = NormaliseMode(normalise)
        except ValueError as exc:
            raise BuildError(
                f"preprocess.normalise={normalise!r} is not one of "
                f"{[m.value for m in NormaliseMode]}"
            ) from exc
        steps.append(Normalisation(mode=mode))
    return Pipeline(steps=tuple(steps), name=config.name)


def _covers_window(spectrum: Spectrum, low: float, high: float) -> bool:
    values = np.asarray(spectrum.axis.values, dtype=np.float64)
    return float(values.min()) <= low and float(values.max()) >= high


def _build_open_set_retrieval(config: ExperimentConfig, root: Path) -> BuiltTask:
    """Assemble the open-set retrieval benchmark of :mod:`usis.eval.tasks.retrieval`."""
    dataset_id = str(config.require("dataset.id"))
    limit = config.get("dataset.limit")
    load = load_corpus(dataset_id, root=root, limit=None if limit is None else int(limit))

    labelled = [
        s
        for s in load.spectra
        if s.label is not None and (s.label.inchikey or s.label.smiles or s.label.name)
    ]
    n_unlabelled = len(load.spectra) - len(labelled)

    low = float(config.require("window.low_cm1"))
    high = float(config.require("window.high_cm1"))
    covering = [s for s in labelled if _covers_window(s, low, high)]
    n_window = len(labelled) - len(covering)
    if not covering:
        raise BuildError(
            f"no spectrum in {dataset_id} covers [{low:g}, {high:g}] cm-1, so the corpus cannot be "
            "put on one grid. Widen the window, or name a dataset whose range covers it."
        )

    pipeline = build_pipeline(config)
    processed = pipeline.apply_all(covering)
    task = OpenSetRetrievalTask.from_spectra(
        processed,
        task_id=str(config.get("task.id", config.name)),
        dataset_ids=(dataset_id,),
        seed=config.seed,
        held_out_fraction=float(config.get("split.test_fraction", 0.2)),
        pipeline=pipeline,
    )
    return BuiltTask(
        task, load=load, pipeline=pipeline, n_window=n_window, n_unlabelled=n_unlabelled
    )


#: Task kind to builder. A new benchmark registers here and nowhere else; the CLI reads this map.
TASK_BUILDERS: dict[str, Callable[[ExperimentConfig, Path], BuiltTask]] = {
    "open_set_retrieval": _build_open_set_retrieval,
}


def build_task(config: ExperimentConfig, *, root: Path = DEFAULT_DATA_DIR) -> BuiltTask:
    """Build the benchmark ``config`` names.

    Raises:
        BuildError: if ``task.kind`` names no registered builder. The message lists what is
            available, because the alternative - a KeyError - tells a reader nothing about which
            benchmarks this build of the tool actually has.
    """
    kind = str(config.require("task.kind"))
    builder = TASK_BUILDERS.get(kind)
    if builder is None:
        raise BuildError(
            f"task.kind={kind!r} is not a registered benchmark. Available: "
            f"{sorted(TASK_BUILDERS)}. A benchmark is added by writing its module and registering "
            "it in usis.compare.build.TASK_BUILDERS - never by editing the CLI."
        )
    return builder(config, root)
