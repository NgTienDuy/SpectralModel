"""Split construction, where refusing is the primary feature.

Brief §3.4 and §10.3: splits are adversarial by default and ``split_kind`` is a required argument
with no default. This module adds the part the brief implies but does not spell out — **a split
that cannot be constructed honestly must raise, not degrade.**

That is not defensive programming, it is the lesson of the four M1 loaders. Every one of them
turned up a way a split could look fine and be wrong:

* the *Sci Data* set has one vial per compound, so a random split puts ~110 scans of the same vial
  on both sides;
* ROD varies excitation, not instrument, so an "instrument-disjoint" split over it would be a
  relabelled excitation split;
* 27 of the 97 registered datasets have **unknown** replicate structure, where treating "no group
  ids" as "no replicates" is exactly the silent failure;
* every source except one has a single instrument, so an instrument-disjoint split over it would
  return an empty test set rather than an error.

So each kind declares the key it groups on, and every way that key can be missing is a named
refusal. A split object also carries ``notes`` — caveats that must travel with any number computed
from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from usis.core.errors import UsisError
from usis.core.types import Origin
from usis.data.registry import DATASETS, ReplicateStructure

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from usis.core.types import Spectrum

__all__ = ["Split", "SplitError", "SplitKind", "group_keys", "make_split"]


class SplitError(UsisError):
    """A split could not be constructed honestly, with the reason in the message."""


class SplitKind(StrEnum):
    """The split matrix of brief §10.3, plus the excitation split ROD actually supports.

    ``RANDOM`` is a smoke-test tool and says so in every split it produces. ``EXCITATION_DISJOINT``
    is an addition: ROD and RRUFF measure one sample at several excitation wavelengths on one
    instrument, which probes wavelength-dependent effects (fluorescence, the nu^4 factor,
    resonance) and is *not* instrument fingerprinting. Giving it its own name is what stops it
    being reported as the stronger claim.
    """

    RANDOM = "random"
    SAMPLE_DISJOINT = "sample_disjoint"
    MOLECULE_DISJOINT = "molecule_disjoint"
    SCAFFOLD_DISJOINT = "scaffold_disjoint"
    INSTRUMENT_DISJOINT = "instrument_disjoint"
    LAB_DISJOINT = "lab_disjoint"
    EXCITATION_DISJOINT = "excitation_disjoint"
    SIM2REAL = "sim2real"
    RESOLUTION_SHIFT = "resolution_shift"
    OPEN_SET = "open_set"


@dataclass(frozen=True)
class Split:
    """One train/test partition, with what it grouped on and what it cannot claim."""

    kind: SplitKind
    train: tuple[int, ...]
    test: tuple[int, ...]
    group_key: str
    train_groups: frozenset[str]
    test_groups: frozenset[str]
    seed: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Assert the partition is a partition and its groups do not straddle the boundary."""
        if not self.train or not self.test:
            raise SplitError(
                f"{self.kind}: a split needs a non-empty train and test side, got "
                f"{len(self.train)} and {len(self.test)}"
            )
        if set(self.train) & set(self.test):
            raise SplitError(f"{self.kind}: an index appears on both sides of the split")
        overlap = self.train_groups & self.test_groups
        if overlap and self.kind is not SplitKind.RANDOM:
            raise SplitError(
                f"{self.kind}: {len(overlap)} {self.group_key} value(s) appear on both sides, "
                f"e.g. {sorted(overlap)[:3]}"
            )

    @property
    def n_train(self) -> int:
        """Number of training spectra."""
        return len(self.train)

    @property
    def n_test(self) -> int:
        """Number of test spectra."""
        return len(self.test)

    @property
    def test_fraction(self) -> float:
        """Realised test fraction, which whole-group splitting rarely makes exact."""
        return self.n_test / (self.n_train + self.n_test)


# --------------------------------------------------------------------------------------------
# Group-key extraction: one function per kind, each failing with the reason
# --------------------------------------------------------------------------------------------
def _require_dataset_replicates_known(spectra: Sequence[Spectrum]) -> None:
    """Refuse a sample-disjoint split when any dataset's replicate structure is unknown."""
    unknown = set()
    for spectrum in spectra:
        try:
            spec = DATASETS.get(spectrum.source.dataset_id)
        except UsisError:
            unknown.add(spectrum.source.dataset_id)
            continue
        if spec.replicate_structure == ReplicateStructure.UNKNOWN:
            unknown.add(spec.dataset_id)
    if unknown:
        raise SplitError(
            "sample_disjoint needs to know which spectra are replicates of one physical sample, "
            f"and {len(unknown)} dataset(s) do not record it: {sorted(unknown)[:3]}. Treating "
            "unknown as 'no replicates' is how a sample leaks across the split. Establish the "
            "replicate structure, or use molecule_disjoint, which is coarser and safe."
        )


def _sample_keys(spectra: Sequence[Spectrum]) -> list[str]:
    _require_dataset_replicates_known(spectra)
    return [s.sample_id for s in spectra]


def _molecule_keys(spectra: Sequence[Spectrum]) -> list[str]:
    keys: list[str] = []
    for index, spectrum in enumerate(spectra):
        label = spectrum.label
        key = None if label is None else (label.inchikey or label.smiles or label.name)
        if not key:
            raise SplitError(
                f"molecule_disjoint needs an identity per spectrum, and spectrum "
                f"{spectrum.spectrum_id!r} (index {index}) has none. An unlabelled spectrum cannot "
                "be held out by molecule."
            )
        keys.append(key)
    return keys


def _scaffold_keys(spectra: Sequence[Spectrum]) -> list[str]:
    raise SplitError(
        "scaffold_disjoint needs Murcko scaffolds, which needs RDKit: uv sync --extra chem. It is "
        "not implemented yet because no registered dataset carries SMILES for every spectrum "
        "(QM9S keeps structures in a separate archive), so the split would refuse on identity "
        "before it ever reached a scaffold."
    )


def _instrument_keys(spectra: Sequence[Spectrum]) -> list[str]:
    keys: list[str] = []
    for spectrum in spectra:
        if spectrum.instrument is None:
            raise SplitError(
                f"instrument_disjoint needs an instrument per spectrum, and "
                f"{spectrum.spectrum_id!r} has none (origin {spectrum.origin}). Simulated spectra "
                "belong in a sim2real split, not an instrument-disjoint one."
            )
        keys.append(spectrum.instrument.instrument_id)
    if len(set(keys)) < 2:
        raise SplitError(
            "instrument_disjoint needs at least two instruments; these spectra come from one "
            f"({next(iter(set(keys)))!r}). Holding out the only instrument leaves nothing to train "
            "on. Of the registered sources only the RamanBench bioprocess_analytes_* collection "
            "spans several instruments."
        )
    return keys


def _lab_keys(spectra: Sequence[Spectrum]) -> list[str]:
    keys: list[str] = []
    for spectrum in spectra:
        lab = None if spectrum.instrument is None else spectrum.instrument.lab_id
        if not lab:
            raise SplitError(
                f"lab_disjoint needs a lab per spectrum, and {spectrum.spectrum_id!r} does not "
                "record one. No registered dataset records labs across more than one site."
            )
        keys.append(lab)
    if len(set(keys)) < 2:
        raise SplitError("lab_disjoint needs at least two labs; these spectra come from one")
    return keys


def _excitation_keys(spectra: Sequence[Spectrum]) -> list[str]:
    keys: list[str] = []
    for spectrum in spectra:
        excitation = spectrum.excitation_nm
        if excitation is None:
            raise SplitError(
                f"excitation_disjoint needs an excitation wavelength, and "
                f"{spectrum.spectrum_id!r} does not record one."
            )
        keys.append(f"{excitation:g}nm")
    if len(set(keys)) < 2:
        raise SplitError(
            "excitation_disjoint needs at least two excitation wavelengths; these spectra were "
            f"all taken at {next(iter(set(keys)))}"
        )
    return keys


def _resolution_keys(spectra: Sequence[Spectrum]) -> list[str]:
    keys: list[str] = []
    for spectrum in spectra:
        fwhm = spectrum.axis.resolution_fwhm
        if fwhm is None:
            raise SplitError(
                f"resolution_shift needs a resolution per spectrum, and {spectrum.spectrum_id!r} "
                "has none. A spectrum whose resolution is unknown cannot be placed on either side "
                "of a resolution shift."
            )
        keys.append(f"{fwhm:g}")
    if len(set(keys)) < 2:
        raise SplitError(
            f"resolution_shift needs at least two resolutions; these spectra are all at "
            f"{next(iter(set(keys)))} cm^-1"
        )
    return keys


_KEY_FUNCTIONS: dict[SplitKind, Callable[[Sequence[Spectrum]], list[str]]] = {
    SplitKind.SAMPLE_DISJOINT: _sample_keys,
    SplitKind.MOLECULE_DISJOINT: _molecule_keys,
    SplitKind.SCAFFOLD_DISJOINT: _scaffold_keys,
    SplitKind.INSTRUMENT_DISJOINT: _instrument_keys,
    SplitKind.LAB_DISJOINT: _lab_keys,
    SplitKind.EXCITATION_DISJOINT: _excitation_keys,
    SplitKind.RESOLUTION_SHIFT: _resolution_keys,
    SplitKind.OPEN_SET: _molecule_keys,
}


def group_keys(spectra: Sequence[Spectrum], kind: SplitKind) -> list[str]:
    """The per-spectrum group key a split of this kind groups on.

    Exposed because the leakage checker needs the same keys the split used, and because it is
    useful to ask "could this split even be built?" without building it.
    """
    if kind is SplitKind.RANDOM:
        return [str(i) for i in range(len(spectra))]
    if kind is SplitKind.SIM2REAL:
        return [str(s.origin) for s in spectra]
    return _KEY_FUNCTIONS[kind](spectra)


# --------------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------------
def _partition_groups(
    keys: Sequence[str], test_fraction: float, seed: int
) -> tuple[frozenset[str], frozenset[str]]:
    """Assign whole groups to test until the target fraction of *spectra* is covered.

    Groups are sorted before shuffling so that the result depends on the seed and the data, never
    on dictionary ordering.
    """
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts)
    if len(ordered) < 2:
        raise SplitError(
            f"cannot split on a key with a single value ({ordered[0]!r}): one side would be empty"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)

    target = max(1, math.floor(test_fraction * len(keys)))
    test: set[str] = set()
    covered = 0
    for key in ordered:
        if covered >= target and test:
            break
        test.add(key)
        covered += counts[key]
    train = set(ordered) - test
    if not train:
        raise SplitError(
            "every group landed in test; the requested test_fraction is too large for "
            f"{len(ordered)} group(s)"
        )
    return frozenset(train), frozenset(test)


def make_split(
    spectra: Sequence[Spectrum],
    kind: SplitKind,
    *,
    seed: int,
    test_fraction: float = 0.2,
) -> Split:
    """Build a split, or raise :class:`SplitError` explaining why it cannot be built honestly.

    Args:
        spectra: the corpus to split.
        kind: **required, with no default.** Choosing a split is a scientific decision, and a
            default would make ``random`` the silent norm.
        seed: makes the partition reproducible; recorded on the result.
        test_fraction: target share of *spectra* on the test side. Whole groups are kept together,
            so the realised fraction is approximate — read it from :attr:`Split.test_fraction`.
    """
    if not spectra:
        raise SplitError("cannot split an empty corpus")
    if not 0.0 < test_fraction < 1.0:
        raise SplitError(f"test_fraction must be in (0, 1), got {test_fraction}")

    notes: list[str] = []
    keys = group_keys(spectra, kind)

    if kind is SplitKind.SIM2REAL:
        train_groups = frozenset({str(Origin.SIMULATED)})
        test_groups = frozenset({str(Origin.EXPERIMENTAL)})
        missing = [g for g in (train_groups | test_groups) if g not in set(keys)]
        if missing:
            raise SplitError(
                f"sim2real needs both simulated and experimental spectra; none are {missing[0]}"
            )
        notes.append(
            "sim2real trains on simulated spectra only; any gap it measures includes the "
            "simulation's own line broadening, not just the instrument."
        )
    elif kind is SplitKind.OPEN_SET:
        train_groups, test_groups = _partition_groups(keys, test_fraction, seed)
        notes.append(
            "open_set holds out whole identities: the test side contains molecules absent from "
            "training, so a closed-set metric over it is meaningless. Score p_none, not accuracy."
        )
    else:
        train_groups, test_groups = _partition_groups(keys, test_fraction, seed)

    if kind is SplitKind.RANDOM:
        notes.append(
            "random is a smoke test only. It controls for nothing: replicates, instruments and "
            "molecules all straddle the boundary. No reported result may use it."
        )
    if kind is SplitKind.EXCITATION_DISJOINT:
        notes.append(
            "excitation_disjoint varies the laser wavelength on one instrument. It is not "
            "instrument shift and must never be reported as such."
        )
    if kind is SplitKind.RESOLUTION_SHIFT:
        notes.append(
            "resolution_shift groups by recorded resolution; for simulated sources that value is "
            "the applied line broadening rather than an instrument response."
        )

    train = tuple(i for i, key in enumerate(keys) if key in train_groups)
    test = tuple(i for i, key in enumerate(keys) if key in test_groups)
    return Split(
        kind=kind,
        train=train,
        test=test,
        group_key=_GROUP_KEY_NAMES[kind],
        train_groups=train_groups,
        test_groups=test_groups,
        seed=seed,
        notes=tuple(notes),
    )


_GROUP_KEY_NAMES: dict[SplitKind, str] = {
    SplitKind.RANDOM: "row index",
    SplitKind.SAMPLE_DISJOINT: "sample_id",
    SplitKind.MOLECULE_DISJOINT: "molecule identity",
    SplitKind.SCAFFOLD_DISJOINT: "Murcko scaffold",
    SplitKind.INSTRUMENT_DISJOINT: "instrument_id",
    SplitKind.LAB_DISJOINT: "lab_id",
    SplitKind.EXCITATION_DISJOINT: "excitation wavelength",
    SplitKind.SIM2REAL: "origin",
    SplitKind.RESOLUTION_SHIFT: "resolution_fwhm",
    SplitKind.OPEN_SET: "molecule identity",
}
