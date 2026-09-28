"""Turning spectra into the matrix the classical methods want, and refusing when it would lie.

Every method in this package indexes channel *k* of one spectrum against channel *k* of another.
Building that matrix is three lines, and the reason it is a module is the fourth: the places where
it must refuse rather than reshape. A stacked matrix of spectra on different axes is arithmetically
valid and physically meaningless, and nothing downstream can detect it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from usis.core.errors import UsisError
from usis.core.types import FloatArray, Spectrum

__all__ = ["MatrixError", "as_matrix", "class_labels", "target_values"]


class MatrixError(UsisError):
    """Spectra cannot be stacked into a matrix, or carry no usable labels."""


def as_matrix(spectra: Sequence[Spectrum]) -> FloatArray:
    """Stack spectra row-wise, refusing when their axes differ.

    A duplicate of what :class:`~usis.methods.assumptions.RequiresCommonAxis` checks, and
    deliberately so: the assumption layer reports the violation in the audit, this raises. A method
    that declared the assumption never reaches here with a bad matrix; a script that calls the
    algorithm directly still cannot build one.
    """
    if not spectra:
        raise MatrixError("no spectra to stack")
    lengths = {int(np.asarray(s.intensity).size) for s in spectra}
    if len(lengths) > 1:
        raise MatrixError(
            f"spectra have {sorted(lengths)} channels: resample onto a common axis first, as a "
            "recorded preprocessing step"
        )
    reference = np.asarray(spectra[0].axis.values, dtype=np.float64)
    for spectrum in spectra[1:]:
        values = np.asarray(spectrum.axis.values, dtype=np.float64)
        if not np.allclose(values, reference, atol=1e-6, rtol=0.0):
            raise MatrixError(
                f"{spectrum.spectrum_id} is on a different axis from {spectra[0].spectrum_id}; "
                "stacking them would add intensities from different wavenumbers"
            )
    return np.asarray(
        np.vstack([np.asarray(s.intensity, dtype=np.float64) for s in spectra]), dtype=np.float64
    )


def class_labels(spectra: Sequence[Spectrum]) -> list[str]:
    """The class label of each spectrum, from ``label.name``, refusing when any is missing.

    Falling back to a placeholder for the unlabelled ones would put a class called "unknown" into
    the training set and then score predictions of it.
    """
    labels = []
    for spectrum in spectra:
        name = spectrum.label.name if spectrum.label is not None else None
        if not name:
            raise MatrixError(
                f"{spectrum.spectrum_id} carries no label.name, so it cannot be part of a "
                "supervised calibration set"
            )
        labels.append(name)
    return labels


def target_values(spectra: Sequence[Spectrum], target: str) -> FloatArray:
    """The named quantitative target of each spectrum, from ``label.targets``.

    Raises when any spectrum lacks it, and names the target: a regression fitted on the subset that
    happened to carry a value is a regression on a different population.
    """
    values = []
    for spectrum in spectra:
        targets = spectrum.label.targets if spectrum.label is not None else None
        if targets is None or target not in targets:
            raise MatrixError(
                f"{spectrum.spectrum_id} carries no label.targets[{target!r}]; the available "
                f"targets are {sorted(targets) if targets else '[]'}"
            )
        values.append(float(targets[target]))
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)
