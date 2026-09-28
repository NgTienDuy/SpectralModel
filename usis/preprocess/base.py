"""One preprocessing step, and the discipline that makes a recorded chain replayable.

Brief §3: *preprocessing is recorded and comparable, never silently baked in.* That sentence is
usually implemented as a log line. Here it has a stronger meaning, and the M2 acceptance criterion
says so: **the processing chain in the output must reproduce the spectrum.** A chain that cannot be
replayed is a description of what someone believes happened, and the difference between that and a
record only shows up when a result is disputed.

Three consequences follow, and they shape the whole module.

**Steps are reconstructible from their own record.** Every step registers itself under a stable
name and declares its parameters as dataclass fields, so :func:`from_recorded` can rebuild the exact
object from a :class:`~usis.core.types.ProcessingStep`. Adding a step means adding a file; nothing
holds a list.

**Recorded parameters must survive JSON.** :meth:`PreprocessStep.apply` refuses to record a
parameter that is not a JSON scalar or a tuple of them. A chain that only replays inside the process
that produced it is not provenance - and provenance is written to disk.

**A step never changes ``origin``.** A processed experimental measurement is still an experimental
measurement, which is the opposite of the measurement forward model, where a degradation marks its
output ``AUGMENTED`` because it fabricated part of it. The asymmetry is deliberate: one module
invents signal, the other only rearranges it.

**Where the ``Method`` boundary falls.** A step is a spectrum-to-spectrum transform, not a
:class:`~usis.methods.base.Method`, because it makes no claim about matter. A *pipeline* becomes
part of a method's identity when one is scored - which is ADR-0001 §S3's fix, that disagreement
must be computed across whole pipelines including the preprocessing choice, not across final-stage
methods sharing one preprocessed input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar, TypeVar

import numpy as np

from usis.core.errors import UsisError
from usis.core.provenance import utc_now
from usis.core.registry import Registry
from usis.core.types import AxisSpec, FloatArray, ProcessingStep, Spectrum

__all__ = [
    "PREPROCESS_STEPS",
    "PreprocessError",
    "PreprocessStep",
    "added_steps",
    "from_recorded",
    "register_step",
    "step_name_of",
]

#: JSON scalar types a recorded parameter may be, so a chain replays after a round trip to disk.
_JSON_SCALARS = (str, int, float, bool, type(None))

S = TypeVar("S", bound="PreprocessStep")


class PreprocessError(UsisError):
    """A preprocessing step was asked for something undefined, or cannot be replayed."""


@dataclass(frozen=True)
class PreprocessStep(ABC):
    """One recorded, replayable spectrum-to-spectrum transform.

    Subclasses implement :meth:`_transform` and declare their parameters as dataclass fields. The
    base handles recording, the JSON-safety check and the append-only chain, so no subclass can
    forget them.
    """

    #: Version of this step's implementation, recorded in the chain. Bump it when the numerics
    #: change, because a chain replayed under different numerics reproduces a different spectrum
    #: and would otherwise fail its own reproduction check with no explanation.
    step_version: ClassVar[str] = "1"

    @property
    def step_name(self) -> str:
        """The name recorded in the chain, e.g. ``preprocess.SnipBaseline``."""
        return step_name_of(type(self))

    def params(self) -> dict[str, Any]:
        """The parameters recorded alongside the step: every dataclass field, in declared order.

        ``dataclasses.fields`` rather than ``__dataclass_fields__``, and the difference is not
        cosmetic: the latter includes ``ClassVar`` entries, so ``step_version`` would be recorded
        as a parameter and then handed back to a constructor that has no such argument.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @abstractmethod
    def _transform(self, axis: AxisSpec, intensity: FloatArray) -> tuple[AxisSpec, FloatArray]:
        """Return the transformed axis and intensity. Must be deterministic."""

    def apply(self, spectrum: Spectrum) -> Spectrum:
        """Apply this step, appending its record to the processing chain.

        Raises:
            PreprocessError: if a recorded parameter would not survive a JSON round trip, so the
                chain could not be replayed from disk.
        """
        recorded = _json_safe(self.params(), self.step_name)
        axis, intensity = self._transform(
            spectrum.axis, np.asarray(spectrum.intensity, dtype=np.float64)
        )
        step = ProcessingStep(
            name=self.step_name,
            params=recorded,
            version=self.step_version,
            timestamp=utc_now(),
        )
        # sigma described the input's uncertainty on the input's axis. Smoothing correlates
        # neighbouring samples and resampling moves them, so the stored per-sample sigma no longer
        # describes the output. Dropping it says so; carrying it forward would overstate precision.
        return replace(
            spectrum,
            axis=axis,
            intensity=intensity,
            processing=(*spectrum.processing, step),
            sigma=None,
        )


def _json_safe(params: dict[str, Any], step_name: str) -> dict[str, Any]:
    """Return ``params`` unchanged, raising if any value would not survive a JSON round trip."""
    for key, value in params.items():
        if isinstance(value, _JSON_SCALARS):
            continue
        if isinstance(value, (tuple, list)) and all(isinstance(v, _JSON_SCALARS) for v in value):
            continue
        raise PreprocessError(
            f"{step_name} would record parameter {key!r} as {type(value).__name__}, which does "
            "not survive a JSON round trip. A chain that only replays in the process that wrote "
            "it is not provenance. Record a scalar, or a tuple of scalars, instead."
        )
    return dict(params)


#: Name to class, so a recorded chain can be turned back into the objects that produced it.
PREPROCESS_STEPS: Registry[type[PreprocessStep]] = Registry("preprocess-steps")


def step_name_of(cls: type[PreprocessStep]) -> str:
    """The registry key and recorded name for a step class."""
    return f"preprocess.{cls.__name__}"


def register_step(cls: type[S]) -> type[S]:
    """Class decorator registering a step under its recorded name."""
    PREPROCESS_STEPS.register(step_name_of(cls), cls)
    return cls


def from_recorded(record: ProcessingStep) -> PreprocessStep:
    """Rebuild the step object that wrote ``record``.

    Raises:
        PreprocessError: if the name is not a registered preprocessing step, or the recorded
            parameters do not match its signature. Both are refusals rather than best-effort
            reconstructions: a pipeline rebuilt from a partially-understood record would verify
            reproduction against a *different* pipeline, which is worse than not checking.
    """
    if record.name not in PREPROCESS_STEPS:
        raise PreprocessError(
            f"{record.name!r} is not a registered preprocessing step, so this chain cannot be "
            "replayed. Steps recorded by other layers - measurement degradations, source-declared "
            "preprocessing - are deliberately not reconstructible here."
        )
    cls = PREPROCESS_STEPS.get(record.name)
    if record.version != cls.step_version:
        raise PreprocessError(
            f"{record.name} was recorded at version {record.version!r} but the installed "
            f"implementation is version {cls.step_version!r}. Replaying it would reproduce a "
            "different spectrum, which is exactly the confusion the version field exists to "
            "prevent."
        )
    accepted = {f.name for f in fields(cls)}
    supplied = dict(record.params)
    unknown = sorted(set(supplied) - accepted)
    if unknown:
        raise PreprocessError(f"{record.name} has no parameter(s) {unknown}")
    try:
        return cls(**supplied)
    except TypeError as exc:
        raise PreprocessError(f"{record.name} cannot be rebuilt from {supplied}: {exc}") from exc


def added_steps(original: Spectrum, processed: Spectrum) -> tuple[ProcessingStep, ...]:
    """The chain entries ``processed`` has and ``original`` does not.

    Relies on the chain being append-only, which :class:`~usis.core.types.Spectrum` enforces, so
    the suffix is exactly what was applied in between. Raises if ``processed``'s chain is not an
    extension of ``original``'s, because then the two spectra are not related by preprocessing and
    any reproduction check on them would be meaningless.
    """
    before, after = original.processing, processed.processing
    if len(after) < len(before) or after[: len(before)] != before:
        raise PreprocessError(
            "the processed spectrum's chain is not an extension of the original's, so they are "
            "not related by a sequence of preprocessing steps"
        )
    return after[len(before) :]


def transform_callable(step: PreprocessStep) -> Callable[[Spectrum], Spectrum]:
    """``step.apply`` as a plain callable, for code that composes transforms generically."""
    return step.apply
