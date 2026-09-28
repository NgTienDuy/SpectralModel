"""Explicit, tested unit and axis conversions.

The brief calls this class of bug the one that "silently destroys projects", so every conversion
here is a named function with a round-trip test, and nothing converts implicitly. Two rules
govern the module:

1. **A Raman shift is not a wavenumber.** Both are quoted in cm^-1 and they differ by the
   excitation wavenumber. Converting between them *requires* the excitation wavelength; when it
   is absent we raise :class:`~usis.core.errors.MissingMetadataError` rather than guess.
2. **Out-of-domain input is an error, not a NaN.** ``log10`` of a non-positive transmittance and
   division by a zero wavelength are rejected at the boundary, where the caller still has the
   context to explain what went wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np

from usis.core.constants import CM1_PER_EV, K_B_EV_PER_K, NM_CM1_PRODUCT
from usis.core.errors import MissingMetadataError, UnitConversionError
from usis.core.types import AxisKind, IntensityKind, RamanBranch

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
else:  # pragma: no cover - runtime alias only
    FloatArray = np.ndarray

__all__ = [
    "absolute_wavenumber_cm1_to_raman_shift",
    "absorbance_to_transmittance",
    "anti_stokes_stokes_ratio",
    "convert_axis",
    "convert_intensity",
    "ev_to_wavenumber_cm1",
    "kubelka_munk_to_reflectance",
    "raman_shift_to_absolute_wavenumber_cm1",
    "raman_shift_to_scattered_wavelength_nm",
    "reflectance_to_kubelka_munk",
    "scattered_wavelength_nm_to_raman_shift",
    "temperature_from_anti_stokes_ratio",
    "transmittance_to_absorbance",
    "wavelength_nm_to_ev",
    "wavelength_nm_to_wavenumber_cm1",
    "wavenumber_cm1_to_ev",
    "wavenumber_cm1_to_wavelength_nm",
]

#: Conversions are shape-preserving: a float in gives a float out, an array in gives an array out.
Num = TypeVar("Num", float, "FloatArray")


def _require_positive(value: float | FloatArray, name: str) -> None:
    """Raise :class:`UnitConversionError` unless every element of ``value`` is strictly positive."""
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise UnitConversionError(f"{name} must be finite")
    if np.any(array <= 0.0):
        raise UnitConversionError(f"{name} must be strictly positive")


def _require_excitation(excitation_nm: float | None) -> float:
    """Return the excitation wavelength, or explain precisely why the conversion cannot proceed."""
    if excitation_nm is None:
        raise MissingMetadataError(
            "this conversion needs the excitation wavelength: a Raman shift and an absolute "
            "wavenumber differ by the excitation wavenumber, and guessing it would be a "
            "fabricated axis (failure code F01)"
        )
    _require_positive(excitation_nm, "excitation_nm")
    return float(excitation_nm)


# --------------------------------------------------------------------------------------------
# Axis conversions
# --------------------------------------------------------------------------------------------
def wavelength_nm_to_wavenumber_cm1(wavelength_nm: Num) -> Num:
    """Convert a vacuum wavelength in nm to an absolute wavenumber in cm^-1."""
    _require_positive(wavelength_nm, "wavelength_nm")
    return NM_CM1_PRODUCT / wavelength_nm


def wavenumber_cm1_to_wavelength_nm(wavenumber_cm1: Num) -> Num:
    """Convert an absolute wavenumber in cm^-1 to a vacuum wavelength in nm."""
    _require_positive(wavenumber_cm1, "wavenumber_cm1")
    return NM_CM1_PRODUCT / wavenumber_cm1


def wavenumber_cm1_to_ev(wavenumber_cm1: Num) -> Num:
    """Convert a wavenumber in cm^-1 to a photon energy in eV.

    This is a linear scaling, so it applies equally to an absolute wavenumber (photon energy) and
    to a Raman shift (energy transferred to the sample).
    """
    return wavenumber_cm1 / CM1_PER_EV


def ev_to_wavenumber_cm1(energy_ev: Num) -> Num:
    """Convert an energy in eV to a wavenumber in cm^-1."""
    return energy_ev * CM1_PER_EV


def wavelength_nm_to_ev(wavelength_nm: Num) -> Num:
    """Convert a vacuum wavelength in nm to a photon energy in eV."""
    return wavenumber_cm1_to_ev(wavelength_nm_to_wavenumber_cm1(wavelength_nm))


def ev_to_wavelength_nm(energy_ev: Num) -> Num:
    """Convert a photon energy in eV to a vacuum wavelength in nm."""
    return wavenumber_cm1_to_wavelength_nm(ev_to_wavenumber_cm1(energy_ev))


def raman_shift_to_absolute_wavenumber_cm1(
    shift_cm1: Num,
    excitation_nm: float | None,
    branch: RamanBranch = RamanBranch.STOKES,
) -> Num:
    """Convert a Raman shift to the absolute wavenumber of the scattered photon.

    Stokes scattering loses energy to the sample, so the scattered photon sits *below* the
    excitation line; anti-Stokes sits above it.

    Args:
        shift_cm1: Raman shift in cm^-1, conventionally positive on both branches.
        excitation_nm: Excitation wavelength in nm. Required.
        branch: Which side of the Rayleigh line the shift refers to.

    Returns:
        Absolute wavenumber of the scattered photon in cm^-1.
    """
    excitation_cm1 = wavelength_nm_to_wavenumber_cm1(_require_excitation(excitation_nm))
    if branch is RamanBranch.STOKES:
        return excitation_cm1 - shift_cm1
    return excitation_cm1 + shift_cm1


def absolute_wavenumber_cm1_to_raman_shift(
    wavenumber_cm1: Num,
    excitation_nm: float | None,
    branch: RamanBranch = RamanBranch.STOKES,
) -> Num:
    """Convert the absolute wavenumber of a scattered photon to a Raman shift."""
    excitation_cm1 = wavelength_nm_to_wavenumber_cm1(_require_excitation(excitation_nm))
    if branch is RamanBranch.STOKES:
        return excitation_cm1 - wavenumber_cm1
    return wavenumber_cm1 - excitation_cm1


def raman_shift_to_scattered_wavelength_nm(
    shift_cm1: Num,
    excitation_nm: float | None,
    branch: RamanBranch = RamanBranch.STOKES,
) -> Num:
    """Convert a Raman shift to the scattered wavelength in nm."""
    absolute = raman_shift_to_absolute_wavenumber_cm1(shift_cm1, excitation_nm, branch)
    return wavenumber_cm1_to_wavelength_nm(absolute)


def scattered_wavelength_nm_to_raman_shift(
    wavelength_nm: Num,
    excitation_nm: float | None,
    branch: RamanBranch = RamanBranch.STOKES,
) -> Num:
    """Convert a scattered wavelength in nm to a Raman shift in cm^-1."""
    absolute = wavelength_nm_to_wavenumber_cm1(wavelength_nm)
    return absolute_wavenumber_cm1_to_raman_shift(absolute, excitation_nm, branch)


def convert_axis(
    values: FloatArray,
    from_kind: AxisKind,
    to_kind: AxisKind,
    excitation_nm: float | None = None,
    branch: RamanBranch = RamanBranch.STOKES,
) -> FloatArray:
    """Convert an axis between any two :class:`~usis.core.types.AxisKind` semantics.

    Conversions that cross the Raman-shift boundary require ``excitation_nm``; the others do not.
    The result is not re-sorted: a caller that needs monotonic-ascending output should sort it
    explicitly, together with the intensities, so the pairing is never silently broken.
    """
    array = np.asarray(values, dtype=np.float64)
    if from_kind is to_kind:
        return array

    # Step 1: to absolute wavenumber, the module's pivot representation.
    if from_kind is AxisKind.ABSOLUTE_WAVENUMBER_CM1:
        absolute = array
    elif from_kind is AxisKind.WAVELENGTH_NM:
        absolute = wavelength_nm_to_wavenumber_cm1(array)
    elif from_kind is AxisKind.ENERGY_EV:
        absolute = ev_to_wavenumber_cm1(array)
    elif from_kind is AxisKind.RAMAN_SHIFT_CM1:
        absolute = raman_shift_to_absolute_wavenumber_cm1(array, excitation_nm, branch)
    else:  # pragma: no cover - exhaustive over the enum
        raise UnitConversionError(f"unsupported source axis kind {from_kind}")

    # Step 2: out of the pivot.
    if to_kind is AxisKind.ABSOLUTE_WAVENUMBER_CM1:
        return absolute
    if to_kind is AxisKind.WAVELENGTH_NM:
        return wavenumber_cm1_to_wavelength_nm(absolute)
    if to_kind is AxisKind.ENERGY_EV:
        return wavenumber_cm1_to_ev(absolute)
    if to_kind is AxisKind.RAMAN_SHIFT_CM1:
        return absolute_wavenumber_cm1_to_raman_shift(absolute, excitation_nm, branch)
    raise UnitConversionError(f"unsupported target axis kind {to_kind}")  # pragma: no cover


# --------------------------------------------------------------------------------------------
# Intensity conversions
# --------------------------------------------------------------------------------------------
def transmittance_to_absorbance(transmittance: Num) -> Num:
    """Convert transmittance (a fraction in ``(0, 1]``) to absorbance.

    Transmittance expressed in percent must be divided by 100 first: this function has no way to
    tell 50% from a transmittance of 50, and guessing is failure code ``F01``.
    """
    _require_positive(transmittance, "transmittance")
    if np.any(np.asarray(transmittance) > 1.0):
        raise UnitConversionError(
            "transmittance must be a fraction in (0, 1]; divide percent values by 100 first"
        )
    # numpy and float.__pow__ both widen to Any in the stubs; the runtime types are exact.
    return cast(Num, -np.log10(transmittance))


def absorbance_to_transmittance(absorbance: Num) -> Num:
    """Convert absorbance to transmittance as a fraction."""
    if not np.all(np.isfinite(np.asarray(absorbance, dtype=np.float64))):
        raise UnitConversionError("absorbance must be finite")
    transmittance = 10.0 ** (-np.asarray(absorbance, dtype=np.float64))
    if isinstance(absorbance, np.ndarray):
        return transmittance
    return float(transmittance)


def reflectance_to_kubelka_munk(reflectance: Num) -> Num:
    """Convert diffuse reflectance in ``(0, 1]`` to the Kubelka-Munk remission function.

    ``F(R) = (1 - R)^2 / (2R)``. This is the transform that makes diffuse-reflectance intensity
    approximately proportional to concentration; applying it to a specular or ATR spectrum is a
    category error, which is why the intensity kind is part of the type.
    """
    _require_positive(reflectance, "reflectance")
    if np.any(np.asarray(reflectance) > 1.0):
        raise UnitConversionError("reflectance must be a fraction in (0, 1]")
    return (1.0 - reflectance) ** 2 / (2.0 * reflectance)


def kubelka_munk_to_reflectance(kubelka_munk: Num) -> Num:
    """Invert the Kubelka-Munk function, taking the physical root in ``(0, 1]``."""
    array = np.asarray(kubelka_munk, dtype=np.float64)
    if np.any(array < 0.0):
        raise UnitConversionError("the Kubelka-Munk function is non-negative")
    root = np.sqrt((1.0 + array) ** 2 - 1.0)
    if isinstance(kubelka_munk, np.ndarray):
        return (1.0 + array) - root
    return float((1.0 + array) - root)


def convert_intensity(
    values: FloatArray,
    from_kind: IntensityKind,
    to_kind: IntensityKind,
) -> FloatArray:
    """Convert between intensity conventions where the conversion is defined.

    Only the physically meaningful conversions exist. ``COUNTS`` and ``ARBITRARY`` cannot be
    converted to anything: recovering absorbance from raw counts needs a reference measurement,
    not a formula, and pretending otherwise fabricates a quantitative claim.
    """
    array = np.asarray(values, dtype=np.float64)
    if from_kind is to_kind:
        return array
    pairs: dict[tuple[IntensityKind, IntensityKind], object] = {
        (IntensityKind.TRANSMITTANCE, IntensityKind.ABSORBANCE): transmittance_to_absorbance,
        (IntensityKind.ABSORBANCE, IntensityKind.TRANSMITTANCE): absorbance_to_transmittance,
        (IntensityKind.REFLECTANCE, IntensityKind.KUBELKA_MUNK): reflectance_to_kubelka_munk,
        (IntensityKind.KUBELKA_MUNK, IntensityKind.REFLECTANCE): kubelka_munk_to_reflectance,
    }
    func = pairs.get((from_kind, to_kind))
    if func is None:
        raise UnitConversionError(
            f"no defined conversion from {from_kind} to {to_kind}; this pair needs a reference "
            "measurement or a calibration, not a unit conversion"
        )
    return cast("FloatArray", func(array))  # type: ignore[operator]


# --------------------------------------------------------------------------------------------
# Thermometric relations (unit-level, used by the physics checks and by the forward model)
# --------------------------------------------------------------------------------------------
def anti_stokes_stokes_ratio(
    shift_cm1: float,
    temperature_k: float,
    excitation_nm: float | None = None,
) -> float:
    """Predict the anti-Stokes / Stokes intensity ratio for a band at a given temperature.

    The ratio is the Boltzmann population factor ``exp(-hc*nu / kT)`` times a frequency-dependent
    scattering factor. With ``excitation_nm`` given, the standard ``nu^4`` scattering correction
    is applied using the scattered wavenumbers; without it, only the Boltzmann factor is returned
    and the result is therefore a lower bound on the ratio.

    This is a *verifiable physical law*: measured against a real spectrum it yields the sample
    temperature, and it is one of the cheap consistency checks the measurement audit can run.
    """
    _require_positive(shift_cm1, "shift_cm1")
    _require_positive(temperature_k, "temperature_k")
    boltzmann = float(np.exp(-wavenumber_cm1_to_ev(shift_cm1) / (K_B_EV_PER_K * temperature_k)))
    if excitation_nm is None:
        return boltzmann
    excitation_cm1 = wavelength_nm_to_wavenumber_cm1(float(excitation_nm))
    if shift_cm1 >= excitation_cm1:
        raise UnitConversionError("the Raman shift exceeds the excitation wavenumber")
    anti_stokes_cm1 = excitation_cm1 + shift_cm1
    stokes_cm1 = excitation_cm1 - shift_cm1
    return boltzmann * float((anti_stokes_cm1 / stokes_cm1) ** 4)


def temperature_from_anti_stokes_ratio(
    ratio: float,
    shift_cm1: float,
    excitation_nm: float | None = None,
) -> float:
    """Invert :func:`anti_stokes_stokes_ratio` to estimate the sample temperature in K."""
    _require_positive(ratio, "ratio")
    _require_positive(shift_cm1, "shift_cm1")
    corrected = ratio
    if excitation_nm is not None:
        excitation_cm1 = wavelength_nm_to_wavenumber_cm1(float(excitation_nm))
        if shift_cm1 >= excitation_cm1:
            raise UnitConversionError("the Raman shift exceeds the excitation wavenumber")
        factor = ((excitation_cm1 + shift_cm1) / (excitation_cm1 - shift_cm1)) ** 4
        corrected = ratio / float(factor)
    if corrected >= 1.0:
        raise UnitConversionError(
            "an anti-Stokes/Stokes ratio at or above 1 after the scattering correction implies a "
            "non-positive temperature; the measurement or the correction is wrong"
        )
    energy_ev = wavenumber_cm1_to_ev(float(shift_cm1))
    return float(-energy_ev / (K_B_EV_PER_K * np.log(corrected)))
