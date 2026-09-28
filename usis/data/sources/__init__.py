"""One module per dataset: fetch, parse, normalise to ``Spectrum``.

Importing this package registers every dataset in :data:`usis.data.registry.DATASETS` and every
loader in :data:`SOURCES`. Parsers live behind :class:`usis.data.sources.base.Source` so that
adding a format is never a change to a science layer.

Milestone: M1.
"""

from __future__ import annotations

from usis.core.registry import Registry
from usis.data.sources.base import Source
from usis.data.sources.qm9s import QM9S_IR_SPEC, QM9S_RAMAN_SPEC, Qm9sSource
from usis.data.sources.ramanbench import (
    BIOPROCESS_ANALYTE_DATASETS,
    CATALOGUE,
    RamanBenchSource,
)
from usis.data.sources.rod import ROD_SPEC, RodSource
from usis.data.sources.scidata_api import SCIDATA_API_SPEC, SciDataApiSource

__all__ = [
    "BIOPROCESS_ANALYTE_DATASETS",
    "CATALOGUE",
    "QM9S_IR_SPEC",
    "QM9S_RAMAN_SPEC",
    "ROD_SPEC",
    "SCIDATA_API_SPEC",
    "SOURCES",
    "Qm9sSource",
    "RamanBenchSource",
    "RodSource",
    "SciDataApiSource",
    "Source",
    "get_source",
]

#: Loader for each registered dataset, keyed by ``dataset_id``.
SOURCES: Registry[Source] = Registry("sources")
SOURCES.register(ROD_SPEC.dataset_id, RodSource())
SOURCES.register(SCIDATA_API_SPEC.dataset_id, SciDataApiSource())
SOURCES.register(QM9S_IR_SPEC.dataset_id, Qm9sSource("ir"))
SOURCES.register(QM9S_RAMAN_SPEC.dataset_id, Qm9sSource("raman"))
for _key in CATALOGUE["datasets"]:
    SOURCES.register(f"ramanbench:{_key}", RamanBenchSource(_key))


def get_source(dataset_id: str) -> Source:
    """Return the loader for ``dataset_id``, listing what is available when it is unknown."""
    return SOURCES.get(dataset_id)
