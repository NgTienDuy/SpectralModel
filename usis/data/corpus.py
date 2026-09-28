"""Loading a mirrored dataset off disk into spectra, and counting what would not load.

``usis data pull`` writes record files; ``usis data verify`` re-parses them and reports. This is
the third thing a benchmark needs and neither of those provides: the spectra themselves, as a list,
with an account of everything that did not make it into that list.

The account is the point. A benchmark assembled from "whatever parsed" is a benchmark over an
unstated subset, and the subset is never random - it is whichever records had the metadata the
loader insisted on, which correlates with age, with lab, and with how carefully the deposit was
made. :class:`CorpusLoad` therefore carries the skipped count and the reason for each skip, and the
comparison record prints them, so a reader can tell a 240-spectrum benchmark drawn from 1,131
records from a 240-record dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from usis.core.errors import UsisError
from usis.data.pull import DEFAULT_DATA_DIR, SIDECAR_NAMES, dataset_dir
from usis.data.sources import get_source

if TYPE_CHECKING:
    from usis.core.types import Spectrum

__all__ = ["CorpusLoad", "load_corpus"]


@dataclass(frozen=True)
class CorpusLoad:
    """The spectra that loaded, and an itemised account of those that did not."""

    dataset_id: str
    spectra: tuple[Spectrum, ...]
    n_files: int
    skipped: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def n_skipped(self) -> int:
        """How many record files did not yield a spectrum."""
        return len(self.skipped)

    @property
    def skip_reasons(self) -> dict[str, int]:
        """Skip counts by exception type, which is usually the level a reader needs."""
        counts: dict[str, int] = {}
        for _, reason in self.skipped:
            kind = reason.split(":", 1)[0]
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable summary for a comparison record."""
        return {
            "dataset_id": self.dataset_id,
            "n_spectra": len(self.spectra),
            "n_files": self.n_files,
            "n_skipped": self.n_skipped,
            "skip_reasons": self.skip_reasons,
        }


def load_corpus(
    dataset_id: str,
    *,
    root: Path = DEFAULT_DATA_DIR,
    limit: int | None = None,
) -> CorpusLoad:
    """Parse every mirrored record of ``dataset_id`` into spectra, in stable filename order.

    Args:
        dataset_id: the registered dataset, e.g. ``"rod"``.
        root: the directory ``usis data pull`` wrote into.
        limit: stop after this many spectra. Applied in filename order, so it is reproducible -
            and it is therefore *not* a random sample, which any result built on it must say.

    Raises:
        UsisError: if the dataset has not been mirrored under ``root``.
    """
    directory = dataset_dir(root, dataset_id)
    if not directory.is_dir():
        raise UsisError(
            f"no mirrored records for {dataset_id!r} at {directory}. Run "
            f"'uv run usis data pull {dataset_id} --all' first; the corpus is not committed."
        )
    source = get_source(dataset_id)
    spectra: list[Spectrum] = []
    skipped: list[tuple[str, str]] = []
    n_files = 0
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        if path.name in SIDECAR_NAMES:
            continue
        n_files += 1
        try:
            parsed = list(source.parse_file(path))
        except (UsisError, ValueError) as exc:
            skipped.append((path.name, f"{type(exc).__name__}: {exc}"))
            continue
        if not parsed:
            skipped.append((path.name, "EmptyRecord: the file parsed but held no spectrum"))
            continue
        spectra.extend(parsed)
        if limit is not None and len(spectra) >= limit:
            spectra = spectra[:limit]
            break
    return CorpusLoad(
        dataset_id=dataset_id,
        spectra=tuple(spectra),
        n_files=n_files,
        skipped=tuple(skipped),
    )
