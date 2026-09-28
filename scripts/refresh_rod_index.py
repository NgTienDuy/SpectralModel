"""Regenerate the committed Raman Open Database record index.

Run::

    uv run python scripts/refresh_rod_index.py

Ids only — this never downloads a spectrum. ROD inherits the Crystallography Open Database's query
interface, whose ``format=lst`` view is a plain-text listing of every published code. The snapshot
is committed so that ``usis data pull rod --all`` fetches the same corpus on two different days: a
corpus that grows silently between runs makes two result cards incomparable without either of them
saying so, and refreshing it here turns that growth into a reviewable diff.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.request import urlopen

from usis.data.sources.rod import ROD_INDEX_SNAPSHOT, ROD_INDEX_URL


def main() -> int:
    """Fetch the listing, sanity-check it, and write the snapshot."""
    with urlopen(ROD_INDEX_URL, timeout=120) as response:
        text = response.read().decode("utf-8")

    ids = [line.strip() for line in text.splitlines() if line.strip()]
    if not all(code.isdigit() for code in ids):
        bad = next(code for code in ids if not code.isdigit())
        print(f"refusing to write: {bad!r} is not a record code, so this is not the id listing")
        return 1
    if len(ids) < 1000:
        print(f"refusing to write: only {len(ids)} ids, which is fewer than ROD publishes")
        return 1

    previous: set[str] = set()
    if ROD_INDEX_SNAPSHOT.exists():
        previous = set(json.loads(ROD_INDEX_SNAPSHOT.read_text(encoding="utf-8"))["ids"])

    payload = {
        "_snapshot": {
            "url": ROD_INDEX_URL,
            "captured_at": datetime.now(UTC).date().isoformat(),
            "n_ids": len(ids),
            "note": (
                "Record codes only, no spectra. ROD is CC0-1.0. The listing is the database's own "
                "index, not a scan of guessed id blocks."
            ),
        },
        "ids": sorted(ids, key=int),
    }
    ROD_INDEX_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    ROD_INDEX_SNAPSHOT.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    added = sorted(set(ids) - previous, key=int)
    removed = sorted(previous - set(ids), key=int)
    print(f"wrote {ROD_INDEX_SNAPSHOT} with {len(ids)} ids (+{len(added)}, -{len(removed)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
