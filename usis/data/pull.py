"""Corpus fetch and verification, behind ``usis data``.

The split mirrors :class:`usis.data.sources.base.Source`: :func:`pull_records` performs network
I/O and is never exercised by a test, while :func:`verify_directory` is offline and re-parses every
record on disk. Verification re-derives the checksum from the bytes rather than trusting a
manifest, because a manifest that agrees with itself proves nothing.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from usis.core.errors import UsisError
from usis.core.provenance import content_hash
from usis.data.registry import get_dataset
from usis.data.sources import get_source
from usis.data.sources.base import BulkFile

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "DEFAULT_DATA_DIR",
    "FETCH_RECORD_NAME",
    "SIDECAR_NAMES",
    "UNAVAILABLE_RECORD_NAME",
    "VerifyResult",
    "dataset_dir",
    "fetch_record",
    "pull_bulk",
    "pull_records",
    "verify_directory",
]

#: Written into each dataset directory by :func:`pull_bulk`, naming what was actually taken.
FETCH_RECORD_NAME = "FETCH.json"

#: Written by :func:`pull_records` when the index names records the publisher does not serve.
UNAVAILABLE_RECORD_NAME = "UNAVAILABLE.json"

#: Files this module writes into a dataset directory. They describe the fetch; they are not records.
SIDECAR_NAMES = frozenset({FETCH_RECORD_NAME, UNAVAILABLE_RECORD_NAME})

#: Where ``usis data`` puts fetched records unless told otherwise. Git-ignored.
DEFAULT_DATA_DIR = Path("data/raw")


def dataset_dir(root: Path, dataset_id: str) -> Path:
    """The directory a dataset's raw records live in."""
    return root / dataset_id


@dataclass(frozen=True)
class VerifyResult:
    """One record's verification outcome."""

    path: Path
    dataset_id: str
    ok: bool
    sha256: str = ""
    n_spectra: int = 0
    first_spectrum_id: str = ""
    n_points: int = 0
    error: str = ""


def pull_records(
    dataset_id: str,
    record_ids: Sequence[str] | None = None,
    root: Path = DEFAULT_DATA_DIR,
    *,
    skip_existing: bool = True,
) -> list[Path]:
    """Fetch records into ``root/<dataset_id>/`` (performs network I/O).

    ``record_ids=None`` means every record the source publishes, taken from its committed index
    (:meth:`~usis.data.sources.base.Source.list_record_ids`). A source with no index refuses rather
    than mirroring nothing and reporting success.

    ``skip_existing`` leaves files already on disk alone, so an interrupted mirror resumes instead
    of re-fetching a thousand records from someone else's server. Verification is what proves a
    file is right; :func:`verify_directory` re-parses and re-checksums every one of them, and it
    does not trust this function's word for anything.
    """
    spec = get_dataset(dataset_id)
    source = get_source(dataset_id)
    if record_ids is not None and not record_ids:
        raise UsisError(
            f"{spec.dataset_id}: no record ids given, so nothing was fetched. Pass ids, or None "
            "for every record in the committed index."
        )
    ids = tuple(record_ids) if record_ids is not None else source.list_record_ids()
    destination = dataset_dir(root, dataset_id)
    paths: list[Path] = []
    unavailable: list[tuple[str, str]] = []
    for record_id in ids:
        target = destination / f"{record_id}.rod"
        if skip_existing and target.exists():
            paths.append(target)
            continue
        try:
            paths.append(source.fetch(record_id, target))
        except (UsisError, OSError) as exc:
            # One record the publisher no longer serves must not abandon the other thousand. It is
            # collected and reported instead, because "1,130 of 1,133" and "1,133" are different
            # corpora and only one of them is what happened.
            unavailable.append((record_id, f"{type(exc).__name__}: {exc}"))
    if unavailable:
        (destination / UNAVAILABLE_RECORD_NAME).write_text(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "index_size": len(ids),
                    "fetched": len(paths),
                    "unavailable": [{"record_id": rid, "error": err} for rid, err in unavailable],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if not paths:
        raise UsisError(
            f"{spec.dataset_id}: nothing was fetched"
            + (f"; {len(unavailable)} record(s) were unavailable" if unavailable else "")
        )
    return paths


def pull_bulk(
    dataset_id: str,
    root: Path = DEFAULT_DATA_DIR,
    *,
    max_bytes: int | None = None,
    chunks: int = 8,
    compress: bool = True,
) -> list[Path]:
    """Fetch the whole files a source is published as (performs network I/O).

    Three of the four M1 sources publish one large file rather than per-record downloads, which is
    why :func:`pull_records` cannot reproduce them. ``max_bytes`` takes a **prefix** over HTTP
    range, for the common case where a corpus of a few dozen spectra is wanted from a file of
    130,000 and downloading 8.8 GB to use 0.04% of it would be rude to the publisher's server.

    The budget is spread over ``chunks`` byte ranges rather than taken as one prefix, because
    both published CSVs are ordered - see :func:`fetch_record`.

    Two things a partial fetch must not be allowed to hide, both written into ``FETCH.json``
    beside the file:

    * **it is not a random sample.** The rows are whichever ones the byte offsets landed in, in the
      publisher's own file order. An experiment that uses one has to say so;
    * **how much of the file it is**, against the size the publisher reports.

    Every range is cut back to whole lines, so a loader never sees a half-written row and reports
    it as a parse failure.
    """
    source = get_source(dataset_id)
    files = source.bulk_files()
    if not files:
        raise UsisError(
            f"{dataset_id} is not published as bulk files; fetch it record by record with "
            "'usis data pull'"
        )
    destination = dataset_dir(root, dataset_id)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    records: list[dict[str, object]] = []
    for bulk in files:
        path, record = fetch_record(
            bulk, destination, max_bytes=max_bytes, chunks=chunks, compress=compress
        )
        written.append(path)
        records.append(record)
    (destination / FETCH_RECORD_NAME).write_text(
        json.dumps({"dataset_id": dataset_id, "files": records}, indent=2), encoding="utf-8"
    )
    return written


def fetch_record(
    bulk: BulkFile, destination: Path, *, max_bytes: int | None, chunks: int, compress: bool
) -> tuple[Path, dict[str, object]]:
    """Fetch one :class:`~usis.data.sources.base.BulkFile` and describe what was taken.

    With ``max_bytes`` the budget is split into ``chunks`` equal ranges spread evenly across the
    published file, and each range is cut back to whole lines. ``chunks=1`` is a plain prefix.

    The default is not 1, and the reason is a property of these particular files rather than a
    general preference. Both published CSVs are **ordered**: QM9S runs in molecule index order, so
    its first rows are its smallest molecules and therefore its sparsest spectra, and the *Sci
    Data* matrix is grouped by compound, so its first rows are all one compound in one vial. A
    prefix of either is not a small version of the corpus; it is one corner of it. Striding does
    not make the sample random - the rows are still whichever ones the byte offsets landed in - but
    it spans the file instead of its first page, and ``FETCH.json`` records which was done.
    """
    if max_bytes is None:
        payload, spans = _get(bulk.url, None), [(0, bulk.published_bytes)]
        is_sample = False
    else:
        if max_bytes < 1:
            raise UsisError(f"max_bytes must be at least 1 byte, got {max_bytes}")
        if chunks < 1:
            raise UsisError(f"chunks must be at least 1, got {chunks}")
        spans = _chunk_spans(bulk.published_bytes, max_bytes, chunks)
        pieces = [
            _whole_lines(_get(bulk.url, span), first=index == 0) for index, span in enumerate(spans)
        ]
        for index, piece in enumerate(pieces):
            if not piece:
                raise UsisError(
                    f"{bulk.name}: chunk {index} of {len(spans)} held no complete line. Raise "
                    "max_bytes or lower chunks."
                )
        payload = b"".join(pieces)
        is_sample = len(payload) < bulk.published_bytes
    target = destination / (f"{bulk.name}.gz" if compress else bulk.name)
    target.write_bytes(gzip.compress(payload) if compress else payload)
    record: dict[str, object] = {
        "name": bulk.name,
        "url": bulk.url,
        "path": target.name,
        "published_bytes": bulk.published_bytes,
        "fetched_bytes": len(payload),
        "fraction_of_published": round(len(payload) / bulk.published_bytes, 6),
        "is_partial": is_sample,
        "selection": "whole file"
        if not is_sample
        else ("prefix" if len(spans) == 1 else f"{len(spans)} byte-strided chunks"),
        "byte_spans": [list(span) for span in spans] if is_sample else [],
        "n_lines": payload.count(b"\n"),
        "sha256_of_stored_file": content_hash(target),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "note": bulk.note,
        "partial_warning": (
            "Not a random sample. The rows are whichever ones the byte offsets landed in, in the "
            "publisher's own file order. Any statistic computed over them inherits that ordering, "
            "and an experiment that uses this corpus has to say so."
        )
        if is_sample
        else "",
    }
    return target, record


def _get(url: str, span: tuple[int, int] | None) -> bytes:
    """One HTTP GET, optionally for a byte range."""
    from urllib.request import Request, urlopen

    request = Request(url)
    if span is not None:
        request.add_header("Range", f"bytes={span[0]}-{span[1] - 1}")
    with urlopen(request, timeout=600) as response:
        return bytes(response.read())


def _chunk_spans(total: int, budget: int, chunks: int) -> list[tuple[int, int]]:
    """``chunks`` equal byte ranges summing to ``budget``, spread evenly across ``total`` bytes.

    The first starts at 0 so that a CSV keeps its header row.
    """
    if budget >= total:
        return [(0, total)]
    size = max(1, budget // chunks)
    stride = (total - size) // max(1, chunks - 1) if chunks > 1 else 0
    return [(index * stride, index * stride + size) for index in range(chunks)]


def _whole_lines(payload: bytes, *, first: bool) -> bytes:
    """Trim a byte range back to complete lines.

    The leading partial line is dropped from every chunk except the first, whose start is the start
    of the file and therefore the header.
    """
    if not first:
        head = payload.find(b"\n")
        if head < 0:
            return b""
        payload = payload[head + 1 :]
    tail = payload.rfind(b"\n")
    return payload[: tail + 1] if tail >= 0 else b""


def verify_directory(
    root: Path = DEFAULT_DATA_DIR, dataset_id: str | None = None
) -> list[VerifyResult]:
    """Re-parse every record under ``root``, reporting one result per file. Offline.

    A record counts as verified only when every spectrum in it satisfies the construction
    invariants — not merely when its bytes are present. A payload may hold one spectrum (ROD) or
    thousands (a matrix CSV); both are reported by count.
    """
    if not root.exists():
        raise UsisError(f"no data directory at {root}; run 'usis data pull' first")
    results: list[VerifyResult] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if dataset_id is not None and directory.name != dataset_id:
            continue
        results.extend(_verify_dataset(directory))
    return results


def _verify_dataset(directory: Path) -> Iterable[VerifyResult]:
    """Verify every record file in one dataset directory."""
    name = directory.name
    try:
        source = get_source(name)
    except UsisError as exc:
        yield VerifyResult(path=directory, dataset_id=name, ok=False, error=str(exc))
        return
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        if path.name in SIDECAR_NAMES:
            # Our own bookkeeping, not a record. Verifying it would report a fetch that succeeded
            # as a corpus of one failed spectrum.
            continue
        first_id = ""
        count = 0
        points = 0
        digest = ""
        try:
            for spectrum in source.parse_file(path):
                count += 1
                if count == 1:
                    first_id = spectrum.spectrum_id
                    points = spectrum.axis.n_points
                    digest = spectrum.source.sha256
        except (UsisError, ValueError) as exc:
            yield VerifyResult(
                path=path,
                dataset_id=name,
                ok=False,
                sha256=content_hash(path),
                n_spectra=count,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        if count == 0:
            yield VerifyResult(
                path=path,
                dataset_id=name,
                ok=False,
                sha256=content_hash(path),
                error="parsed without error but yielded no spectra",
            )
            continue
        yield VerifyResult(
            path=path,
            dataset_id=name,
            ok=True,
            sha256=digest,
            n_spectra=count,
            first_spectrum_id=first_id,
            n_points=points,
        )
