"""Audit instrument, excitation and resolution coverage across a mirrored ROD corpus.

Run after ``uv run usis data pull rod --all``::

    uv run python scripts/audit_rod_instruments.py --corpus data/raw/rod

Offline: it reads records already on disk. The point of the script rather than a one-off command
is that the numbers in ``docs/knowledge/datasets.md`` section 5 can be re-derived, and that a
future mirror shows up as a change in those numbers rather than as a claim nobody can check.

The question it answers is whether ROD can support an `instrument_disjoint` split. An earlier
audit answered "effectively no" from a 240-record scan of guessed id blocks; this one reads every
record in the database's own index.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from usis.data.sources.rod import RodSource


def main() -> int:
    """Count instruments, excitations, resolutions and multi-excitation specimens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/raw/rod", type=Path)
    args = parser.parse_args()

    source = RodSource()
    instruments: Counter[str] = Counter()
    excitations: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    labs: Counter[str] = Counter()
    excitation_by_sample: dict[str, set[str]] = {}
    joint: Counter[tuple[str, str]] = Counter()
    failed = 0
    total = 0

    for path in sorted(args.corpus.glob("*.rod")):
        total += 1
        try:
            spectrum = source.parse_one(path)
        except Exception as exc:
            failed += 1
            print(f"  unreadable {path.name}: {type(exc).__name__}: {exc}")
            continue
        instrument = spectrum.instrument
        model = "unrecorded" if instrument is None else (instrument.model or "unrecorded")
        maker = "" if instrument is None else (instrument.manufacturer or "")
        instruments[f"{maker} {model}".strip()] += 1
        labs[("unrecorded" if instrument is None else instrument.lab_id) or "unrecorded"] += 1
        laser = (
            "unrecorded"
            if instrument is None or instrument.excitation_nm is None
            else (f"{instrument.excitation_nm:g} nm")
        )
        excitations[laser] += 1
        width = spectrum.axis.resolution_fwhm
        resolutions["unrecorded" if width is None else f"{width:g} cm^-1"] += 1
        excitation_by_sample.setdefault(spectrum.sample_id, set()).add(laser)
        joint[
            (
                f"{maker} {model}".strip(),
                ("unrecorded" if instrument is None else instrument.lab_id) or "unrecorded",
            )
        ] += 1

    readable = total - failed
    print(f"\nrecords on disk: {total}, readable: {readable}, unreadable: {failed}\n")
    for title, counter in (
        ("instrument", instruments),
        ("laboratory", labs),
        ("excitation", excitations),
        ("resolution", resolutions),
    ):
        print(f"{title}:")
        for key, count in counter.most_common():
            share = 100.0 * count / readable if readable else 0.0
            print(f"  {count:5d}  {share:5.1f}%  {key}")
        print()

    print("instrument x laboratory (an instrument-disjoint split needs these not to coincide):")
    for (model, lab), count in sorted(joint.items(), key=lambda kv: -kv[1]):
        print(f"  {count:5d}  {model}  @  {lab}")
    print()

    multi = {k: v for k, v in excitation_by_sample.items() if len(v) > 1}
    print(f"specimens: {len(excitation_by_sample)}")
    print(f"specimens measured at >=2 excitations: {len(multi)}")
    print()
    print(
        "NOTE: instrument strings are free text in the source and are NOT normalised here. Some "
        "published records carry a model value with the next CIF tag embedded in it, and there "
        "are spelling variants of one model. Collapsing them is a documented decision, not an "
        "inference a loader may make silently."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
