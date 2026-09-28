"""A pipeline, and the check that its own record reproduces its own output.

The M2 acceptance criterion for this layer is one sentence: *the processing chain in the output
reproduces the spectrum.* :func:`verify_reproduction` is that sentence executed. It takes the
original and the processed spectrum, rebuilds a pipeline **from the record alone** - not from the
pipeline object that happens to be in memory - replays it, and compares the result sample by
sample.

That is a stronger test than it looks, and it has already constrained the design of every step:
parameters must be JSON scalars, numerics must be deterministic, no step may consult a global
random state, and a step whose numerics change must bump its version. A chain which cannot pass
this check is a chain that documents an intention.

The comparison is **exact** by default. Replaying the identical operations in the identical order
on the identical input is bit-reproducible in IEEE arithmetic, so any deviation means the record is
incomplete rather than that floating point is imprecise. ``tolerance`` exists for the case where
the record travelled through a lossy serialisation, and using it is a statement about the transport,
not about the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from usis.core.types import ProcessingStep, Spectrum
from usis.preprocess.base import (
    PreprocessError,
    PreprocessStep,
    added_steps,
    from_recorded,
)

__all__ = ["Pipeline", "Reproduction", "verify_reproduction"]


@dataclass(frozen=True)
class Pipeline:
    """An ordered sequence of preprocessing steps, applied in the order given.

    Unlike :class:`usis.measurement.forward_model.ForwardModel`, this does **not** reorder its
    steps. The measurement chain has a canonical order because the physics has one - light is
    broadened before it is digitised, whatever order the operators were listed in. Preprocessing
    has no such order: smoothing before baseline correction and after it are two different, equally
    defensible pipelines, and which is better is an empirical question this layer exists to ask.
    Sorting them would answer it by fiat.
    """

    steps: tuple[PreprocessStep, ...] = ()
    name: str = ""

    def __len__(self) -> int:
        """Number of steps."""
        return len(self.steps)

    def __iter__(self) -> Iterator[PreprocessStep]:
        """Iterate over the steps in application order."""
        return iter(self.steps)

    @classmethod
    def of(cls, *steps: PreprocessStep, name: str = "") -> Pipeline:
        """Build a pipeline from steps given positionally."""
        return cls(steps=tuple(steps), name=name)

    @property
    def label(self) -> str:
        """A stable, readable identifier: the step names and their parameters, in order.

        This is what a leaderboard row is keyed on, so it must not depend on dict ordering or on
        the ``name`` field, which is free text.
        """
        return " > ".join(_label_of(step) for step in self.steps)

    def apply(self, spectrum: Spectrum) -> Spectrum:
        """Apply every step in order, each appending its own record."""
        current = spectrum
        for step in self.steps:
            current = step.apply(current)
        return current

    def apply_all(self, spectra: Iterable[Spectrum]) -> tuple[Spectrum, ...]:
        """Apply this pipeline to many spectra."""
        return tuple(self.apply(spectrum) for spectrum in spectra)

    def then(self, *steps: PreprocessStep) -> Pipeline:
        """A new pipeline with ``steps`` appended."""
        return Pipeline(steps=(*self.steps, *steps), name=self.name)

    @classmethod
    def from_records(cls, records: Sequence[ProcessingStep], name: str = "") -> Pipeline:
        """Rebuild a pipeline from recorded chain entries.

        Raises:
            PreprocessError: if any record is not a registered preprocessing step. Skipping the
                unrecognised ones would rebuild a *different* pipeline and then verify reproduction
                against it, which is worse than not verifying at all.
        """
        return cls(steps=tuple(from_recorded(record) for record in records), name=name)

    @classmethod
    def from_chain(cls, original: Spectrum, processed: Spectrum) -> Pipeline:
        """Rebuild the pipeline that took ``original`` to ``processed``, from the record alone."""
        return cls.from_records(added_steps(original, processed))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view, so a comparison record can name the pipeline it scored."""
        return {
            "name": self.name,
            "label": self.label,
            "steps": [
                {"name": step.step_name, "version": step.step_version, "params": step.params()}
                for step in self.steps
            ],
        }


def _label_of(step: PreprocessStep) -> str:
    """One step as ``ClassName(param=value,...)``, parameters sorted so the label is stable."""
    parameters = ",".join(f"{key}={value}" for key, value in sorted(step.params().items()))
    return f"{type(step).__name__}({parameters})"


@dataclass(frozen=True)
class Reproduction:
    """The outcome of replaying a recorded chain: whether it reproduced, and how far off it was."""

    reproduced: bool
    n_steps: int
    max_abs_deviation: float
    axis_max_abs_deviation: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view."""
        return {
            "reproduced": self.reproduced,
            "n_steps": self.n_steps,
            "max_abs_deviation": self.max_abs_deviation,
            "axis_max_abs_deviation": self.axis_max_abs_deviation,
            "detail": self.detail,
        }


def verify_reproduction(
    original: Spectrum, processed: Spectrum, tolerance: float = 0.0
) -> Reproduction:
    """Replay ``processed``'s recorded chain on ``original`` and compare with ``processed``.

    Args:
        original: the input the chain was applied to.
        processed: the output, carrying the chain to replay.
        tolerance: maximum permitted absolute deviation. ``0.0`` - the default - demands bit
            equality, which replaying identical deterministic operations delivers.

    Returns:
        A :class:`Reproduction`. A failure is returned rather than raised: whether an
        unreproducible chain is fatal depends on the caller, and an audit that crashed would tell
        the reader less than one that reported the deviation.

    Raises:
        PreprocessError: if the chain cannot be rebuilt at all - an unregistered step, a version
            mismatch, parameters that do not fit the signature. That is a different failure from
            "replayed and differed", and collapsing the two would hide which one happened.
    """
    pipeline = Pipeline.from_chain(original, processed)
    replayed = pipeline.apply(original)
    intensity_deviation = _max_deviation(replayed.intensity, processed.intensity)
    axis_deviation = _max_deviation(replayed.axis.values, processed.axis.values)
    worst = max(intensity_deviation, axis_deviation)
    reproduced = worst <= tolerance
    if reproduced:
        detail = (
            f"replayed {len(pipeline)} recorded step(s) from the chain and reproduced the "
            f"spectrum exactly"
            if tolerance == 0.0
            else f"replayed {len(pipeline)} step(s); worst deviation {worst:g} within "
            f"tolerance {tolerance:g}"
        )
    else:
        detail = (
            f"replaying {len(pipeline)} recorded step(s) gave a spectrum differing by "
            f"{intensity_deviation:g} in intensity and {axis_deviation:g} on the axis, above the "
            f"tolerance {tolerance:g}. The record is incomplete: some input to the computation was "
            "not written down."
        )
    return Reproduction(
        reproduced=reproduced,
        n_steps=len(pipeline),
        max_abs_deviation=intensity_deviation,
        axis_max_abs_deviation=axis_deviation,
        detail=detail,
    )


def _max_deviation(left: Any, right: Any) -> float:
    """Largest absolute elementwise difference; infinity when the shapes disagree."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def require_reproduction(
    original: Spectrum, processed: Spectrum, tolerance: float = 0.0
) -> Reproduction:
    """As :func:`verify_reproduction`, but raise when the chain does not reproduce the spectrum.

    For the places where an unreproducible chain must stop the run - writing a result card, or
    publishing a preprocessed corpus.
    """
    outcome = verify_reproduction(original, processed, tolerance)
    if not outcome.reproduced:
        raise PreprocessError(outcome.detail)
    return outcome
