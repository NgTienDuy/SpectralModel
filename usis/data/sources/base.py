"""The source interface: fetch, parse, normalise.

One rule shapes this file, and it comes from brief §9: **tests never touch the network.** So
fetching and parsing are separate methods with separate contracts:

* :meth:`Source.fetch` performs I/O. It is never called from a test.
* :meth:`Source.parse` is a **pure function of the payload**. It is what the fixtures exercise, and
  it is where every mapping decision — which tag becomes which field — actually lives.

``parse`` yields spectra rather than returning one, because the two shapes both occur: ROD stores
one spectrum per file, while a matrix CSV stores thousands. Use :meth:`Source.parse_one` where
exactly one is expected; it fails loudly rather than silently taking the first.

That split is not fussiness. A loader whose parsing can only be tested by downloading is a loader
nobody re-tests, and silent parse drift is how a corpus quietly becomes wrong.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from usis.core.errors import UsisError
from usis.core.provenance import content_hash, utc_now

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from usis.core.types import Spectrum
    from usis.data.registry import DatasetSpec

__all__ = ["BulkFile", "Source"]


@dataclass(frozen=True)
class BulkFile:
    """One file a publisher distributes whole, rather than record by record.

    Three of the four M1 sources publish this way, and the sizes are why per-record fetch is not
    merely unimplemented but meaningless for them: a single 8.8 GB CSV holds 130,000 spectra and
    there is no URL for spectrum 4,001.

    ``published_bytes`` is the size the publisher's API reports, recorded so that a fetch can say
    what fraction of the file it actually took. A prefix of a published file is a **prefix**, not a
    sample: the rows in it are the first rows, in file order, and any statistic computed over one
    inherits whatever ordering the publisher chose. Nothing here hides that - it is written into
    the fetch record so an experiment has to acknowledge it.
    """

    name: str
    url: str
    published_bytes: int
    note: str = ""


class Source(ABC):
    """One dataset's fetch/parse/normalise implementation.

    Adding a format is a new subclass and nothing else: no science layer knows that vendor formats
    exist.
    """

    @property
    @abstractmethod
    def spec(self) -> DatasetSpec:
        """The dataset manifest entry this source loads."""

    @abstractmethod
    def parse(
        self,
        payload: str,
        *,
        url: str | None = None,
        sha256: str = "",
        retrieved_at: str | None = None,
    ) -> Iterator[Spectrum]:
        """Turn a payload into :class:`~usis.core.types.Spectrum` objects, in file order.

        Pure: no network, no filesystem, no clock beyond the ``retrieved_at`` the caller supplies.

        Args:
            payload: the payload's full text, exactly as fetched.
            url: where it came from, recorded in the provenance chain.
            sha256: checksum of the payload bytes, recorded in the provenance chain.
            retrieved_at: ISO-8601 UTC timestamp; defaults to now when the caller has none.
        """

    @abstractmethod
    def fetch(self, record_id: str, dest: Path) -> Path:
        """Download one record to ``dest`` and return the path (performs network I/O)."""

    def bulk_files(self) -> tuple[BulkFile, ...]:
        """The whole-file downloads this source is distributed as, if any.

        Empty for a per-record source such as ROD. A source has one or the other, never both, and
        which one it is decides whether :func:`~usis.data.pull.pull_records` or
        :func:`~usis.data.pull.pull_bulk` can reproduce the corpus.
        """
        return ()

    def list_record_ids(self) -> tuple[str, ...]:
        """Every record id this source publishes (performs network I/O).

        Only meaningful for per-record sources. The default refuses rather than returning an empty
        tuple, because "this dataset has no records" and "this dataset is not enumerable" are
        different facts and a caller that cannot tell them apart will mirror nothing and report
        success.
        """
        raise UsisError(
            f"{type(self).__name__} does not publish an enumerable record index; it is distributed "
            "as whole files (see bulk_files())"
        )

    def parse_file(self, path: Path, url: str | None = None) -> Iterator[Spectrum]:
        """Parse a payload from disk, checksumming the bytes on the way through.

        The checksum is computed here rather than trusted from a manifest, so that the provenance a
        spectrum carries describes the bytes actually parsed. Subclasses whose payloads are large
        override this to stream rather than hold the whole file in memory.
        """
        yield from self.parse(
            path.read_text(encoding="utf-8", errors="replace"),
            url=url,
            sha256=content_hash(path),
            retrieved_at=utc_now(),
        )

    def parse_one(self, path: Path, url: str | None = None) -> Spectrum:
        """Parse a payload expected to hold exactly one spectrum.

        Raises when it holds none or several, rather than quietly taking the first — a file that
        gained a second record is a change worth noticing.
        """
        spectra = list(self.parse_file(path, url=url))
        if len(spectra) != 1:
            raise UsisError(f"{path}: expected exactly one spectrum, found {len(spectra)}")
        return spectra[0]
