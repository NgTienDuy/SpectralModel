"""Exception hierarchy for USIS.

Every error raised deliberately by this repo derives from :class:`UsisError`, so a caller can
distinguish "the science layer rejected this input" from "the interpreter fell over".
"""

from __future__ import annotations

__all__ = [
    "AssumptionViolationError",
    "BeliefStateError",
    "LicenceError",
    "MissingMetadataError",
    "ProvenanceError",
    "SpectrumValidationError",
    "UnitConversionError",
    "UsisError",
]


class UsisError(Exception):
    """Base class for all deliberate USIS errors."""


class UnitConversionError(UsisError):
    """A unit or axis conversion was requested that is undefined or non-physical.

    Raised for out-of-domain values (non-positive wavelengths, transmittance outside ``(0, 1]``)
    and for conversions that are mathematically undefined rather than merely inconvenient.
    """


class MissingMetadataError(UsisError):
    """An operation needs a metadata field that the spectrum does not carry.

    Per the brief's input contract, missing metadata must *widen the posterior*, not crash the
    pipeline. This exception is therefore for the low-level primitives (e.g. a Raman-shift
    conversion with no excitation wavelength); the analysis layers catch it and price it.
    """


class SpectrumValidationError(UsisError):
    """A :class:`~usis.core.types.Spectrum` or one of its parts violates a structural invariant."""


class ProvenanceError(UsisError):
    """A provenance invariant was violated (e.g. the processing chain was not append-only)."""


class BeliefStateError(UsisError):
    """A belief state violates its normalisation or open-set invariants."""


class LicenceError(UsisError):
    """A dataset carries an unknown or non-redistributable licence where one is required."""


class AssumptionViolationError(UsisError):
    """A method was executed outside its declared validity domain."""
