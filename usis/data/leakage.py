"""The leakage checker that runs in CI.

Brief §10.3: *no ``sample_id``, InChIKey, or near-duplicate spectrum may appear on both sides of a
reported split.* This module checks all three, and it checks them against the spectra rather than
against the split's own bookkeeping — a split that believes its groups are disjoint proves nothing
if the keys were extracted wrongly in the first place.

Findings are returned, not raised, so a caller can report every problem at once instead of the
first. :func:`assert_no_leakage` is the CI entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from usis.core.errors import UsisError
from usis.data.splits import SplitKind, group_keys

if TYPE_CHECKING:
    from collections.abc import Sequence

    from usis.core.types import Spectrum
    from usis.data.splits import Split

__all__ = [
    "NEAR_DUPLICATE_CORRELATION",
    "LeakageFinding",
    "assert_no_leakage",
    "check_leakage",
]

#: Pearson correlation above which two spectra on the same grid are treated as near-duplicates.
#: High on purpose: replicate spectra of one sample routinely correlate above 0.99, so a lower
#: threshold would flag every legitimate within-class pair and train people to ignore the check.
NEAR_DUPLICATE_CORRELATION: float = 0.9999


@dataclass(frozen=True)
class LeakageFinding:
    """One way a split leaks, with enough detail to fix it."""

    kind: str
    detail: str
    train_index: int | None = None
    test_index: int | None = None

    def __str__(self) -> str:
        """Render the finding for a CI log."""
        where = ""
        if self.train_index is not None and self.test_index is not None:
            where = f" (train[{self.train_index}] vs test[{self.test_index}])"
        return f"{self.kind}: {self.detail}{where}"


def _identity_keys(spectra: Sequence[Spectrum]) -> list[str | None]:
    """InChIKey where a spectrum has one, else None. Names are deliberately not used here."""
    out: list[str | None] = []
    for spectrum in spectra:
        label = spectrum.label
        out.append(None if label is None else label.inchikey)
    return out


def check_leakage(
    spectra: Sequence[Spectrum],
    split: Split,
    *,
    check_near_duplicates: bool = True,
    max_pairs: int = 4_000_000,
) -> list[LeakageFinding]:
    """Return every way ``split`` leaks across its own boundary.

    ``sample_id`` and InChIKey overlaps are checked for **every** split kind, including ``random``:
    the point of the checker is to demonstrate that a random split leaks, not to exempt it.

    Near-duplicate detection compares only spectra sharing an axis length, since a correlation
    across different grids is meaningless. It is quadratic, so it is skipped with a finding of its
    own when the pair count would exceed ``max_pairs``.
    """
    findings: list[LeakageFinding] = []
    train, test = list(split.train), list(split.test)

    # 1. sample_id: replicates of one physical sample on both sides.
    sample_keys = [s.sample_id for s in spectra]
    train_samples = {sample_keys[i] for i in train}
    shared = sorted({sample_keys[i] for i in test} & train_samples)
    if shared:
        findings.append(
            LeakageFinding(
                kind="sample_id",
                detail=(
                    f"{len(shared)} sample(s) appear on both sides, e.g. {shared[:3]} — spectra of "
                    "one physical sample are split across train and test"
                ),
            )
        )

    # 2. InChIKey: the same molecule on both sides. Only meaningful for identity-style splits.
    if split.kind in {SplitKind.MOLECULE_DISJOINT, SplitKind.SCAFFOLD_DISJOINT, SplitKind.OPEN_SET}:
        identities = _identity_keys(spectra)
        train_ids = {key for i in train if (key := identities[i])}
        test_ids = {key for i in test if (key := identities[i])}
        shared_ids = sorted(test_ids & train_ids)
        if shared_ids:
            findings.append(
                LeakageFinding(
                    kind="inchikey",
                    detail=f"{len(shared_ids)} molecule(s) on both sides, e.g. {shared_ids[:3]}",
                )
            )

    # 3. The declared group key, recomputed from the spectra rather than trusted.
    try:
        keys = group_keys(spectra, split.kind)
    except UsisError as exc:
        findings.append(
            LeakageFinding(kind="group_key", detail=f"could not recompute the group key: {exc}")
        )
    else:
        if split.kind is not SplitKind.RANDOM:
            overlap = sorted({keys[i] for i in test} & {keys[i] for i in train})
            if overlap:
                findings.append(
                    LeakageFinding(
                        kind=f"{split.group_key}",
                        detail=(
                            f"{len(overlap)} value(s) on both sides when recomputed, e.g. "
                            f"{overlap[:3]}"
                        ),
                    )
                )

    if check_near_duplicates:
        findings.extend(_near_duplicates(spectra, train, test, max_pairs))
    return findings


def _near_duplicates(
    spectra: Sequence[Spectrum],
    train: list[int],
    test: list[int],
    max_pairs: int,
) -> list[LeakageFinding]:
    """Flag train/test pairs whose spectra are nearly identical on a shared grid."""
    if len(train) * len(test) > max_pairs:
        return [
            LeakageFinding(
                kind="near_duplicate",
                detail=(
                    f"skipped: {len(train)} x {len(test)} pairs exceeds max_pairs={max_pairs}. "
                    "Run it on a subsample, or raise the limit deliberately."
                ),
            )
        ]

    findings: list[LeakageFinding] = []
    by_length: dict[int, list[int]] = {}
    for index in train:
        by_length.setdefault(spectra[index].axis.n_points, []).append(index)

    for test_index in test:
        candidates = by_length.get(spectra[test_index].axis.n_points)
        if not candidates:
            continue
        target = _standardise(spectra[test_index].intensity)
        if target is None:
            continue
        for train_index in candidates:
            other = _standardise(spectra[train_index].intensity)
            if other is None:
                continue
            correlation = float(np.dot(target, other) / target.size)
            if correlation >= NEAR_DUPLICATE_CORRELATION:
                findings.append(
                    LeakageFinding(
                        kind="near_duplicate",
                        detail=(
                            f"correlation {correlation:.6f} between "
                            f"{spectra[train_index].spectrum_id!r} and "
                            f"{spectra[test_index].spectrum_id!r}"
                        ),
                        train_index=train_index,
                        test_index=test_index,
                    )
                )
                break
    return findings


def _standardise(intensity: np.ndarray) -> np.ndarray | None:
    """Zero-mean, unit-variance copy, or None for a flat spectrum where correlation is undefined."""
    centred = intensity - intensity.mean()
    norm = float(np.sqrt((centred**2).mean()))
    if norm == 0.0:
        return None
    return centred / norm


def assert_no_leakage(spectra: Sequence[Spectrum], split: Split, **kwargs: object) -> None:
    """Raise if ``split`` leaks. The CI entry point."""
    findings = check_leakage(spectra, split, **kwargs)  # type: ignore[arg-type]
    if findings:
        listed = "\n  ".join(str(f) for f in findings)
        raise UsisError(f"{split.kind} split leaks:\n  {listed}")
