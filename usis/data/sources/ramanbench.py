"""RamanBench: 93 Raman datasets behind one access layer.

RamanBench (arXiv:2605.02003) unifies datasets scattered across eleven repositories and ships them
through the MIT-licensed ``raman-data`` package. We use it for two things, and they have very
different dependency profiles:

* **The catalogue** — names, licences, replicate-structure flags and descriptions — is committed
  here as a snapshot (:data:`CATALOGUE_PATH`), so the manifest and the licence gate work offline
  and without the dependency. Refresh it with ``scripts/refresh_ramanbench_catalogue.py``.
* **The data** is fetched through ``raman-data``, which is an optional extra and is imported lazily
  inside :meth:`RamanBenchSource.fetch`. Twenty transitive dependencies do not belong in the core
  install for a loader most runs will not touch.

Three findings from auditing the catalogue shape this module:

1. **The eight-instrument collection is real.** ``bioprocess_analytes_*`` is eight datasets, one per
   spectrometer, measuring *nearly* the same glucose / sodium acetate / magnesium sulfate mixtures,
   all CC BY 4.0. That is the only instrument-disjoint split we have found in open data.
2. **Licences are heterogeneous and often not licences at all.** Sixteen distinct strings appear,
   including "Requested from authors", "Free access" and "unknown". Anything that is not an
   unambiguous licence maps to ``UNVERIFIED``, which blocks vendoring — see :data:`LICENCE_MAP`.
3. **``DatasetInfo.id`` is not unique.** Four adenine datasets share Zenodo record ``3572359``.
   The catalogue is therefore keyed by the ``list_datasets()`` name; keying on ``id`` would have
   silently merged six datasets into two.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from usis.core.errors import SpectrumValidationError, UsisError
from usis.core.provenance import utc_now
from usis.core.types import (
    AxisKind,
    AxisSpec,
    GroundTruth,
    InstrumentSpec,
    IntensityKind,
    Modality,
    Origin,
    Spectrum,
)
from usis.data.registry import (
    DatasetSpec,
    ReplicateStructure,
    Verification,
    register_dataset,
)
from usis.data.sources.base import Source

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

__all__ = [
    "BIOPROCESS_ANALYTE_DATASETS",
    "CATALOGUE",
    "CATALOGUE_PATH",
    "LICENCE_MAP",
    "RamanBenchSource",
    "dataset_spec_id",
    "map_licence",
    "unresolved_licences",
]

CATALOGUE_PATH: Final[Path] = Path(__file__).parent / "data" / "ramanbench_catalogue.json"

#: Upstream licence string -> our vocabulary. **Only unambiguous licences appear here.**
#:
#: Everything absent maps to ``UNVERIFIED``, which is non-redistributable, so an unread or
#: ambiguous term blocks vendoring instead of passing. The strings deliberately left out include
#: "Requested from authors", "Authors contacted", "Free access", "Optica OAPA", "BY-NC" (no
#: version), "unknown", and one entry whose own text begins "Unclear." None of those is a licence
#: we can act on, and guessing which CC variant "BY-NC" means is exactly the kind of shortcut that
#: turns into a licence violation.
LICENCE_MAP: Final[dict[str, str]] = {
    "CC BY 4.0": "CC-BY-4.0",
    "CC0 1.0": "CC0-1.0",
    "MIT": "MIT",
    "GPL-3.0": "GPL-3.0",
    "CC BY-NC 3.0": "CC-BY-NC-3.0",
    "CC BY-NC-SA 4.0": "CC-BY-NC-SA-4.0",
    "CC BY-NC-ND 4.0": "CC-BY-NC-ND-4.0",
    "research use only": "ACADEMIC-USE-ONLY",
}


def _load_catalogue() -> dict[str, Any]:
    """Read the committed catalogue snapshot."""
    with CATALOGUE_PATH.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    if "datasets" not in payload or not payload["datasets"]:
        raise UsisError(f"{CATALOGUE_PATH}: catalogue snapshot has no datasets")
    return payload


CATALOGUE: Final[dict[str, Any]] = _load_catalogue()

#: The eight-spectrometer collection: nearly the same samples measured on eight instruments.
BIOPROCESS_ANALYTE_DATASETS: Final[tuple[str, ...]] = tuple(
    sorted(k for k in CATALOGUE["datasets"] if k.startswith("bioprocess_analytes_"))
)

#: Vendor and excitation for the eight-instrument collection, read off the catalogue entries.
#: Excitation is filled in only where the dataset states it; the rest stay ``None`` rather than
#: being inferred from a vendor's typical configuration.
_BIOPROCESS_INSTRUMENTS: Final[dict[str, tuple[str, float | None]]] = {
    "bioprocess_analytes_anton_532": ("Anton Paar", 532.0),
    "bioprocess_analytes_anton_785": ("Anton Paar", 785.0),
    "bioprocess_analytes_kaiser": ("Kaiser Optical Systems", None),
    "bioprocess_analytes_metrohm": ("Metrohm", None),
    "bioprocess_analytes_mettler_toledo": ("Mettler Toledo", None),
    "bioprocess_analytes_tec5": ("tec5", None),
    "bioprocess_analytes_timegate": ("Timegate", None),
    "bioprocess_analytes_tornado": ("Tornado Spectral Systems", None),
}


def map_licence(upstream: str) -> str:
    """Map an upstream licence string to our vocabulary, refusing to guess.

    Anything not in :data:`LICENCE_MAP` becomes ``UNVERIFIED``: non-redistributable, and loud.
    """
    return LICENCE_MAP.get(upstream.strip(), "UNVERIFIED")


def unresolved_licences() -> dict[str, list[str]]:
    """Upstream licence strings we could not map, with the datasets carrying each.

    A worklist, not a failure: each entry is a licence somebody has to read before those datasets
    can be vendored.
    """
    out: dict[str, list[str]] = {}
    for key, entry in sorted(CATALOGUE["datasets"].items()):
        upstream = str(entry.get("license", "unknown"))
        if map_licence(upstream) == "UNVERIFIED":
            out.setdefault(upstream, []).append(key)
    return out


def dataset_spec_id(dataset_key: str) -> str:
    """Our manifest id for a RamanBench dataset."""
    return f"ramanbench:{dataset_key}"


def _modality(dataset_key: str, entry: Mapping[str, Any]) -> Modality:
    """SERS is a separate modality, and under-detection here is a known, stated limit.

    Detection is on the dataset key and short name only. Descriptions are not used: several
    spontaneous-Raman datasets describe their analytes as "SERS-active", and matching on that would
    mislabel the technique. The cost is that a SERS dataset whose name does not say so is read as
    spontaneous Raman, so this is recorded in the spec notes rather than trusted silently.
    """
    haystack = f"{dataset_key} {entry.get('short_name', '')}".lower()
    return Modality.RAMAN_SERS if "sers" in haystack else Modality.RAMAN


def _register_all() -> tuple[DatasetSpec, ...]:
    """Register every catalogued dataset in the manifest."""
    specs: list[DatasetSpec] = []
    for key, entry in sorted(CATALOGUE["datasets"].items()):
        upstream = str(entry.get("license", "unknown"))
        licence = map_licence(upstream)
        grouped = entry.get("is_grouped")
        replicates = {
            True: ReplicateStructure.KNOWN,
            False: ReplicateStructure.NONE,
            None: ReplicateStructure.UNKNOWN,
        }[grouped]
        notes = (
            f"RamanBench dataset {key!r}. Upstream licence string: {upstream!r}"
            + ("" if licence != "UNVERIFIED" else " (not mappable; blocked from vendoring)")
            + f". Task: {entry.get('task_type')}. Replicate structure: "
            + {True: "known", False: "none", None: "UNKNOWN - treat as unsafe to split randomly"}[
                grouped
            ]
            + f". Source: {entry.get('source')}"
        )
        specs.append(
            register_dataset(
                DatasetSpec(
                    dataset_id=dataset_spec_id(key),
                    name=str(entry.get("name") or key),
                    modality=_modality(key, entry),
                    origin=Origin.EXPERIMENTAL,
                    licence=licence,
                    homepage=entry.get("source"),
                    doi=entry.get("doi"),
                    citation=entry.get("paper"),
                    # PARTIAL, not VERIFIED: we read the catalogue, not the spectra.
                    verification=Verification.PARTIAL,
                    verified_on="2026-09-03",
                    replicate_structure=replicates,
                    notes=notes,
                )
            )
        )
    return tuple(specs)


SPECS: Final[tuple[DatasetSpec, ...]] = _register_all()


class RamanBenchSource(Source):
    """Adapter for one RamanBench dataset.

    Unlike the file-based loaders, the payload here is an in-memory array bundle rather than text,
    because that is what ``raman-data`` returns. :meth:`spectra_from_arrays` is the pure part and
    is what the tests exercise; :meth:`parse` exists only to satisfy the interface and refuses.
    """

    def __init__(self, dataset_key: str) -> None:
        if dataset_key not in CATALOGUE["datasets"]:
            raise UsisError(
                f"unknown RamanBench dataset {dataset_key!r}; the catalogue holds "
                f"{len(CATALOGUE['datasets'])} entries"
            )
        self._key = dataset_key
        self._entry = CATALOGUE["datasets"][dataset_key]

    @property
    def dataset_key(self) -> str:
        """The upstream ``list_datasets()`` name."""
        return self._key

    @property
    def spec(self) -> DatasetSpec:
        """The manifest entry for this dataset."""
        from usis.data.registry import DATASETS

        return DATASETS.get(dataset_spec_id(self._key))

    @property
    def is_regression(self) -> bool:
        """Whether the upstream task is regression rather than classification."""
        return "regression" in str(self._entry.get("task_type", "")).lower()

    def instrument(self) -> InstrumentSpec:
        """The instrument key for this dataset.

        A RamanBench dataset is the finest instrument grouping we can justify: the catalogue has no
        instrument field, and within one dataset the spectra come from one setup. So the dataset
        *is* the instrument key, which makes an instrument-disjoint split conservatively equal to a
        dataset-disjoint one — and, for the eight-spectrometer bioprocess collection, gives exactly
        the eight-way split the collection was built for. Vendor and excitation are filled in only
        where the catalogue states them.
        """
        vendor, excitation = _BIOPROCESS_INSTRUMENTS.get(self._key, (None, None))
        return InstrumentSpec(
            instrument_id=f"ramanbench:{self._key}",
            manufacturer=vendor,
            excitation_nm=excitation,
            lab_id=None,
        )

    def spectra_from_arrays(
        self,
        spectra: Any,
        raman_shifts: Any,
        targets: Any = None,
        target_names: Sequence[str] | None = None,
        group_ids: Any = None,
        *,
        url: str | None = None,
        sha256: str = "",
        retrieved_at: str | None = None,
    ) -> Iterator[Spectrum]:
        """Convert a ``raman-data`` array bundle into spectra. Pure; no network, no filesystem.

        Duck-typed on the arrays rather than on ``RamanDataset`` so that the conversion — where
        every mapping decision lives — is testable without the optional dependency installed.
        """
        matrix = np.asarray(spectra, dtype=np.float64)
        shifts = np.asarray(raman_shifts, dtype=np.float64)
        if matrix.ndim != 2:
            raise SpectrumValidationError(
                f"{self._key}: spectra must be 2-D (n_spectra, n_points), got shape {matrix.shape}"
            )
        if shifts.ndim != 1 or shifts.size != matrix.shape[1]:
            raise SpectrumValidationError(
                f"{self._key}: raman_shifts has {shifts.size} points but the spectra have "
                f"{matrix.shape[1]} columns"
            )
        axis = AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, shifts)

        target_array = None if targets is None else np.asarray(targets)
        if target_array is not None and target_array.shape[0] != matrix.shape[0]:
            raise SpectrumValidationError(
                f"{self._key}: {target_array.shape[0]} targets for {matrix.shape[0]} spectra"
            )
        groups = None if group_ids is None else np.asarray(group_ids)
        if groups is not None and groups.shape[0] != matrix.shape[0]:
            raise SpectrumValidationError(
                f"{self._key}: {groups.shape[0]} group ids for {matrix.shape[0]} spectra"
            )

        spec = self.spec
        instrument = self.instrument()
        modality = spec.modality
        source = spec.source_record(
            retrieved_at=retrieved_at or utc_now(), url=url or spec.homepage, sha256=sha256
        )
        names = list(target_names or [])

        for row in range(matrix.shape[0]):
            yield Spectrum(
                spectrum_id=f"ramanbench:{self._key}:{row:06d}",
                modality=modality,
                origin=Origin.EXPERIMENTAL,
                axis=axis,
                intensity=matrix[row],
                intensity_kind=IntensityKind.ARBITRARY,
                sample_id=self._sample_id(row, groups),
                source=source,
                instrument=instrument,
                acquisition=None,
                processing=(),
                label=self._label(row, target_array, names, groups),
            )

    def _sample_id(self, row: int, groups: Any) -> str:
        """Group replicates when the upstream says how, and be explicit when it does not."""
        if groups is not None:
            return f"ramanbench:{self._key}:group:{groups[row]}"
        return f"ramanbench:{self._key}:row:{row}"

    def _label(self, row: int, targets: Any, names: list[str], groups: Any) -> GroundTruth:
        """Build the label, keeping quantitative targets out of the closure-checked field."""
        note_parts: list[str] = []
        if groups is None and self._entry.get("is_grouped") is not False:
            # is_grouped True or None with no group ids means we cannot reconstruct which spectra
            # are replicates. Saying so is what lets a split refuse rather than leak.
            note_parts.append(
                "replicate structure not reconstructible here "
                f"(catalogue is_grouped={self._entry.get('is_grouped')})"
            )
        if targets is None:
            return GroundTruth(annotation_note="; ".join(note_parts) or None)

        value = targets[row]
        if self.is_regression:
            values = np.atleast_1d(np.asarray(value, dtype=np.float64))
            labels = (
                names if len(names) == values.size else [f"target_{i}" for i in range(values.size)]
            )
            finite = {
                name: float(v) for name, v in zip(labels, values, strict=True) if np.isfinite(v)
            }
            if len(finite) != values.size:
                note_parts.append("some targets are missing (NaN) and were dropped")
            return GroundTruth(targets=finite, annotation_note="; ".join(note_parts) or None)

        index = int(np.ravel(value)[0])
        name = names[index] if 0 <= index < len(names) else str(index)
        return GroundTruth(name=name, annotation_note="; ".join(note_parts) or None)

    def parse(
        self,
        payload: str,
        *,
        url: str | None = None,
        sha256: str = "",
        retrieved_at: str | None = None,
    ) -> Iterator[Spectrum]:
        """Not applicable: RamanBench payloads are array bundles, not text."""
        raise NotImplementedError(
            f"{self._key}: RamanBench data arrives as arrays through raman-data, not as a text "
            "payload. Use spectra_from_arrays(), or fetch() to load through the package."
        )

    def fetch(self, record_id: str, dest: Path) -> Path:
        """Not implemented; use :meth:`load`, which goes through ``raman-data``."""
        raise NotImplementedError(
            f"{self._key}: RamanBench is fetched through the raman-data package rather than by "
            "record URL. Install the 'bench' extra and use load()."
        )

    def load(self) -> Iterator[Spectrum]:
        """Download through ``raman-data`` and convert (performs network I/O).

        The import is deliberately lazy: ``raman-data`` pulls in about twenty transitive
        dependencies and is an optional extra, so a run that never touches RamanBench never pays
        for it.
        """
        try:
            from raman_data.datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise UsisError(
                "loading RamanBench data needs the optional 'bench' extra: uv sync --extra bench"
            ) from exc

        dataset = load_dataset(self._key)
        yield from self.spectra_from_arrays(
            dataset.spectra,
            dataset.raman_shifts,
            dataset.targets,
            list(dataset.target_names or []),
            dataset.group_ids,
            url=self.spec.homepage,
        )
