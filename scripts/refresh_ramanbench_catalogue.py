"""Regenerate the committed RamanBench catalogue snapshot.

Run with the optional extra installed::

    uv sync --extra bench
    uv run python scripts/refresh_ramanbench_catalogue.py

Metadata only — this never downloads spectra. The snapshot exists so that the dataset manifest and
the licence gate work offline and without ``raman-data`` in the core install; it is committed, so a
refresh shows up as a reviewable diff rather than as a silent change under someone's feet.

Keyed by the ``list_datasets()`` name, **not** by ``DatasetInfo.id``: the id is a source record
identifier and is not unique (four adenine datasets share Zenodo record 3572359), so keying on it
would merge six datasets into two.
"""

from __future__ import annotations

import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

OUTPUT = (
    Path(__file__).parent.parent
    / "usis"
    / "data"
    / "sources"
    / "data"
    / ("ramanbench_catalogue.json")
)


def main() -> int:
    """Write the snapshot and report what changed in size."""
    try:
        from raman_data.datasets import get_dataset_info, list_datasets
    except ImportError:
        print("needs the optional extra: uv sync --extra bench")
        return 1

    payload: dict[str, object] = {
        "_snapshot": {
            "package": "raman-data",
            "version": importlib.metadata.metadata("raman-data")["Version"],
            "captured_at": datetime.now(UTC).date().isoformat(),
            "n_datasets": 0,
            "note": (
                "Metadata only, no spectra. Keyed by the list_datasets() name: DatasetInfo.id is "
                "a source record id and is NOT unique (4 adenine datasets share Zenodo 3572359)."
            ),
        }
    }
    datasets: dict[str, dict[str, object]] = {}
    for name in list_datasets():
        info = get_dataset_info(name)
        datasets[name] = {
            "source_id": info.id,
            "name": info.name,
            "short_name": info.short_name,
            "task_type": str(info.task_type),
            "application_type": str(info.application_type),
            "license": info.license,
            "is_grouped": info.is_grouped,
            "has_missing_labels": info.has_missing_labels,
            "source": info.metadata.get("source"),
            "doi": info.metadata.get("doi"),
            "paper": info.metadata.get("paper"),
            "description": info.metadata.get("description", ""),
        }
    payload["datasets"] = datasets
    payload["_snapshot"]["n_datasets"] = len(datasets)  # type: ignore[index]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(datasets)} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
