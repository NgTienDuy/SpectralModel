"""Append-only structured event log, and its replay reader.

One JSON object per line, one line per event. Deliberately not a logging-module wrapper: the
purpose is a machine-readable trace that an evaluation run can replay and attribute, and the
human-readable stream is a separate concern.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from usis.core.provenance import utc_now

__all__ = ["Event", "RunLog", "read_events"]


@dataclass(frozen=True)
class Event:
    """One recorded event: when, what kind, and its payload."""

    timestamp: str
    kind: str
    payload: dict[str, Any]


class RunLog:
    """Writes events to a JSONL file. Usable as a context manager."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def event(self, kind: str, **payload: Any) -> Event:
        """Record one event and flush it, so a killed process keeps its trace."""
        record = Event(timestamp=utc_now(), kind=kind, payload=payload)
        self._handle.write(
            json.dumps(
                {"timestamp": record.timestamp, "kind": record.kind, **record.payload},
                default=str,
                sort_keys=True,
            )
            + "\n"
        )
        self._handle.flush()
        return record

    def close(self) -> None:
        """Close the underlying file."""
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> RunLog:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the log on exit, whether or not the block raised."""
        self.close()


def read_events(path: Path) -> Iterator[Event]:
    """Replay a run log. Malformed lines raise rather than being skipped silently."""
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed run-log line") from exc
            timestamp = record.pop("timestamp", "")
            kind = record.pop("kind", "")
            yield Event(timestamp=timestamp, kind=kind, payload=record)
