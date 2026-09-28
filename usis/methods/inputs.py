"""What a method is given.

Separated from :mod:`usis.methods.base` for one reason: assumptions are checked against these
inputs, and the ``Method`` protocol in ``base`` declares its assumptions - so a single module
holding both would import itself. The brief's §6.1 file list does not name this module; the
alternative was to declare ``MethodInputs`` inside ``assumptions.py``, which is the wrong home for
it.

The design rule here is the same one that governs :class:`~usis.core.types.Spectrum`: a field that
is absent is *absent*, never defaulted into a claim. ``library=None`` means "no reference library
was supplied", and a retrieval method must refuse; ``LibraryHandle.inchikeys=None`` means
"membership is unknown", which is a different statement from an empty set and is checked
differently.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from usis.core.errors import UsisError
from usis.core.types import Modality, Spectrum

__all__ = ["LibraryHandle", "MethodInputError", "MethodInputs"]


class MethodInputError(UsisError):
    """The inputs handed to a method are structurally unusable."""


@dataclass(frozen=True)
class LibraryHandle:
    """A reference library a retrieval or matching method searches against.

    ``inchikeys`` is the membership set when it is known. It is ``None`` - not ``frozenset()`` -
    when the library exists but its contents have not been enumerated here, because
    "we cannot tell whether the target is in the library" and "the library is empty" lead to
    opposite conclusions about a retrieval miss.
    """

    library_id: str
    n_entries: int
    modality: Modality
    inchikeys: frozenset[str] | None = None
    note: str = ""

    def covers(self, inchikey: str) -> bool | None:
        """Whether ``inchikey`` is in the library, or ``None`` when membership is unknown."""
        if self.inchikeys is None:
            return None
        return inchikey in self.inchikeys


@dataclass(frozen=True)
class MethodInputs:
    """One method invocation's inputs: the spectra, and whatever else was supplied.

    ``calibration`` carries the labelled spectra a supervised or calibration-based method needs
    (PLS, nearest-neighbour matching). It is separate from ``spectra`` so that a method cannot
    accidentally score itself on its own calibration set, and so an assumption check can look at
    the two populations independently.
    """

    spectra: tuple[Spectrum, ...]
    library: LibraryHandle | None = None
    calibration: tuple[Spectrum, ...] = ()
    aux: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject an empty input set rather than let a method return a result about nothing."""
        if not self.spectra:
            raise MethodInputError(
                "MethodInputs.spectra is empty: a method invocation with no spectrum has no "
                "subject, and a result computed from it would be attributed to a measurement "
                "that was never made"
            )

    @classmethod
    def of(cls, *spectra: Spectrum, **kwargs: Any) -> MethodInputs:
        """Build inputs from one or more spectra given positionally."""
        return cls(spectra=tuple(spectra), **kwargs)

    @property
    def primary(self) -> Spectrum:
        """The single spectrum, raising when there is more than one.

        Mirrors :meth:`usis.data.sources.base.Source.parse_one`: a per-spectrum method handed a
        batch must say so, not silently analyse the first element and report it as the answer.
        """
        if len(self.spectra) != 1:
            raise MethodInputError(
                f"primary requested but {len(self.spectra)} spectra were supplied. A method that "
                "answers about one measurement must be mapped over the batch by its caller, so "
                "that every result carries its own spectrum_id."
            )
        return self.spectra[0]

    @property
    def spectrum_ids(self) -> tuple[str, ...]:
        """The ids of the input spectra, in the order supplied."""
        return tuple(s.spectrum_id for s in self.spectra)

    def modalities(self) -> frozenset[Modality]:
        """The distinct modalities present in the inputs."""
        return frozenset(s.modality for s in self.spectra)

    def with_spectra(self, spectra: Iterable[Spectrum]) -> MethodInputs:
        """A copy carrying different spectra and the same auxiliary context."""
        return MethodInputs(
            spectra=tuple(spectra),
            library=self.library,
            calibration=self.calibration,
            aux=self.aux,
        )
