"""The declarative dataset manifest and the vendoring gate.

Brief §8: a dataset is described before it is downloaded — id, homepage, DOI, licence, and how far
its facts have actually been verified. Two properties matter more than the convenience:

* **The licence is a field, not a README sentence.** :meth:`DatasetSpec.assert_vendorable` is what
  CI calls, so an unread licence blocks the build instead of passing by default.
* **Verification status travels with the spec.** ``UNVERIFIED`` is the default, and a loader that
  runs against an unverified dataset is a loader written against assumptions.

Registering a dataset does not fetch it. Sources register themselves at import time; call
:func:`load_sources` to populate the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from usis.core.errors import UsisError
from usis.core.registry import Registry
from usis.core.types import Modality, Origin, SourceRecord
from usis.data.licences import LicenceSpec, assert_vendorable, lookup

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "ReplicateStructure",
    "Verification",
    "get_dataset",
    "load_sources",
    "register_dataset",
]


class ReplicateStructure:
    """Whether repeated spectra of one physical sample can be grouped in this dataset.

    The distinction that matters is ``NONE`` versus ``UNKNOWN``. ``NONE`` is a checked finding that
    every spectrum is an independent sample; ``UNKNOWN`` means nobody has established the replicate
    structure, and treating it as ``NONE`` is how a random split leaks a sample across train and
    test. A ``sample_disjoint`` split refuses on ``UNKNOWN`` rather than guessing.
    """

    KNOWN = "known"
    NONE = "none"
    UNKNOWN = "unknown"

    ALL = frozenset({KNOWN, NONE, UNKNOWN})


class Verification:
    """How far a spec's facts have actually been checked.

    Deliberately a string vocabulary rather than an enum: it mirrors the same three words used in
    ``docs/knowledge/datasets.md``, and the two must stay readable side by side.
    """

    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    UNVERIFIED = "UNVERIFIED"

    ALL = frozenset({VERIFIED, PARTIAL, UNVERIFIED})


@dataclass(frozen=True)
class DatasetSpec:
    """What we know about a dataset before touching a byte of it."""

    dataset_id: str
    name: str
    modality: Modality
    origin: Origin
    licence: str
    homepage: str | None = None
    doi: str | None = None
    citation: str | None = None
    verification: str = Verification.UNVERIFIED
    verified_on: str = ""
    replicate_structure: str = ReplicateStructure.UNKNOWN
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the identifier, the licence key and the verification vocabulary."""
        if not self.dataset_id:
            raise UsisError("DatasetSpec.dataset_id must be non-empty")
        # Raises LicenceError for a key that is not in the vocabulary, which is the point: a
        # dataset cannot enter the registry with an invented licence string.
        lookup(self.licence)
        if self.verification not in Verification.ALL:
            raise UsisError(
                f"{self.dataset_id}: verification must be one of {sorted(Verification.ALL)}, "
                f"got {self.verification!r}"
            )
        if self.replicate_structure not in ReplicateStructure.ALL:
            raise UsisError(
                f"{self.dataset_id}: replicate_structure must be one of "
                f"{sorted(ReplicateStructure.ALL)}, got {self.replicate_structure!r}"
            )
        if self.verification == Verification.VERIFIED and not self.verified_on:
            raise UsisError(
                f"{self.dataset_id}: a VERIFIED spec must record verified_on, so a reader can "
                "tell how stale the check is"
            )

    @property
    def licence_spec(self) -> LicenceSpec:
        """The full licence entry for this dataset."""
        return lookup(self.licence)

    @property
    def vendorable(self) -> bool:
        """Whether the data may be committed into this repository."""
        return self.licence_spec.redistributable

    def assert_vendorable(self) -> LicenceSpec:
        """Raise unless this dataset's licence permits shipping its bytes in the repo."""
        return assert_vendorable(self.licence, self.dataset_id)

    def source_record(
        self, retrieved_at: str, url: str | None = None, sha256: str = ""
    ) -> SourceRecord:
        """Build the :class:`~usis.core.types.SourceRecord` every spectrum from here carries."""
        return SourceRecord(
            dataset_id=self.dataset_id,
            licence=self.licence,
            retrieved_at=retrieved_at,
            sha256=sha256,
            url=url or self.homepage,
            doi=self.doi,
        )


DATASETS: Registry[DatasetSpec] = Registry("datasets")


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    """Add a dataset to the manifest. Duplicate ids are refused, as everywhere else."""
    return DATASETS.register(spec.dataset_id, spec)


def get_dataset(dataset_id: str) -> DatasetSpec:
    """Look up a dataset spec, loading the source modules first."""
    load_sources()
    return DATASETS.get(dataset_id)


def load_sources() -> Mapping[str, DatasetSpec]:
    """Import the source modules so that they register their datasets, and return the manifest.

    The import is deferred rather than top-level because the source modules import this one.
    """
    import usis.data.sources  # noqa: F401  (imported for its registration side effect)

    return DATASETS.as_mapping()
