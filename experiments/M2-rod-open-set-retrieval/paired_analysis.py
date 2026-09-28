"""Post-hoc paired comparison of the three matching metrics. NOT preregistered.

The leaderboard reports each method's selective accuracy with an *unpaired* interval, resampled
over identity groups. Those intervals overlap, and on their own they say the ordering is not
resolved. But the comparison is paired in the strongest sense available: every method saw the same
733 queries against the same library, and ``baseline.nn_match_sca`` returns a candidate list
identical to ``baseline.nn_match_cosine``'s on every one of them. An unpaired interval throws that
structure away and is the wrong test for this question.

This script runs the paired one: bootstrap the *difference* in per-item correctness, resampling
groups rather than items. Read it as exploratory. It was written after the leaderboard was read, it
was not preregistered, and a difference it resolves is a hypothesis for a preregistered run rather
than a settled result.

Usage::

    uv run python experiments/M2-rod-open-set-retrieval/paired_analysis.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NONE_KEY = "__none_of_these__"
N_SAMPLES = 10_000
SEED = 20260911


def main() -> None:
    """Print a paired interval for every pair of methods on the record."""
    record = json.loads((HERE / "comparison.json").read_text(encoding="utf-8"))
    rows = {row["method_key"]: row for row in record["rows"]}
    order = sorted(rows)

    reference_ids = [item["item_id"] for item in rows[order[0]]["items"]]
    for key in order:
        if [item["item_id"] for item in rows[key]["items"]] != reference_ids:
            raise SystemExit(f"{key} was scored on a different item order; the pairing is invalid")

    # The group the leaderboard resampled on is the query's own identity. For a held-out identity
    # the reference is the sentinel, so the item id stands in - each absent query is its own group,
    # which is the conservative reading and matches how the task grouped them.
    keys = [
        item["reference"] if item["reference"] != NONE_KEY else f"absent:{item['item_id']}"
        for item in rows[order[0]]["items"]
    ]
    unique = sorted(set(keys))
    index_of = {key: i for i, key in enumerate(unique)}
    buckets: list[list[int]] = [[] for _ in unique]
    for position, key in enumerate(keys):
        buckets[index_of[key]].append(position)

    correct = {
        key: np.array([item["outcome"] == "correct" for item in rows[key]["items"]], dtype=float)
        for key in order
    }

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(buckets), size=(N_SAMPLES, len(buckets)))
    resamples = [np.concatenate([buckets[i] for i in row]) for row in draws]

    print(f"paired over {len(keys)} items in {len(unique)} groups, {N_SAMPLES} resamples\n")
    for left, right in combinations(order, 2):
        difference = correct[left] - correct[right]
        means = np.array([difference[idx].mean() for idx in resamples])
        low, high = np.quantile(means, [0.025, 0.975])
        verdict = "resolved" if low > 0.0 or high < 0.0 else "NOT resolved at 95%"
        disagree = int(np.sum(correct[left] != correct[right]))
        print(f"{left}  minus  {right}")
        print(
            f"  accuracy difference {difference.mean():+.4f} "
            f"[{low:+.4f}, {high:+.4f}]  -> {verdict}"
        )
        print(f"  the two disagree on {disagree} of {len(keys)} items\n")


if __name__ == "__main__":
    main()
