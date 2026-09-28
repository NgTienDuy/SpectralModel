"""Unit-conversion round trips. This class of bug silently destroys spectroscopy projects."""

from __future__ import annotations

import numpy as np
import pytest

from usis.core.constants import CM1_PER_EV
from usis.core.errors import MissingMetadataError, UnitConversionError
from usis.core.types import AxisKind, IntensityKind, RamanBranch
from usis.core.units import (
    absolute_wavenumber_cm1_to_raman_shift,
    absorbance_to_transmittance,
    anti_stokes_stokes_ratio,
    convert_axis,
    convert_intensity,
    ev_to_wavenumber_cm1,
    kubelka_munk_to_reflectance,
    raman_shift_to_absolute_wavenumber_cm1,
    raman_shift_to_scattered_wavelength_nm,
    reflectance_to_kubelka_munk,
    scattered_wavelength_nm_to_raman_shift,
    temperature_from_anti_stokes_ratio,
    transmittance_to_absorbance,
    wavelength_nm_to_wavenumber_cm1,
    wavenumber_cm1_to_ev,
    wavenumber_cm1_to_wavelength_nm,
)

WAVELENGTHS_NM = np.array([200.0, 532.0, 785.0, 1064.0, 2500.0])
SHIFTS_CM1 = np.array([100.0, 520.7, 1000.0, 1600.0, 3000.0])


def test_wavelength_wavenumber_round_trip():
    back = wavenumber_cm1_to_wavelength_nm(wavelength_nm_to_wavenumber_cm1(WAVELENGTHS_NM))
    assert np.allclose(back, WAVELENGTHS_NM, rtol=1e-12)


def test_wavenumber_ev_round_trip():
    wavenumbers = np.array([100.0, 1000.0, 20000.0])
    back = ev_to_wavenumber_cm1(wavenumber_cm1_to_ev(wavenumbers))
    assert np.allclose(back, wavenumbers, rtol=1e-12)


def test_ev_conversion_matches_codata():
    # 1 eV = 8065.54... cm^-1 from e / (h c). Checked to 6 significant figures.
    assert pytest.approx(8065.543937, rel=1e-9) == CM1_PER_EV
    assert wavenumber_cm1_to_ev(CM1_PER_EV) == pytest.approx(1.0, rel=1e-12)


def test_raman_shift_round_trip_stokes():
    absolute = raman_shift_to_absolute_wavenumber_cm1(SHIFTS_CM1, 785.0)
    back = absolute_wavenumber_cm1_to_raman_shift(absolute, 785.0)
    assert np.allclose(back, SHIFTS_CM1, rtol=1e-10)


def test_raman_shift_round_trip_anti_stokes():
    absolute = raman_shift_to_absolute_wavenumber_cm1(SHIFTS_CM1, 785.0, RamanBranch.ANTI_STOKES)
    back = absolute_wavenumber_cm1_to_raman_shift(absolute, 785.0, RamanBranch.ANTI_STOKES)
    assert np.allclose(back, SHIFTS_CM1, rtol=1e-10)


def test_stokes_is_below_and_anti_stokes_above_the_excitation_line():
    excitation_cm1 = wavelength_nm_to_wavenumber_cm1(532.0)
    stokes = raman_shift_to_absolute_wavenumber_cm1(1000.0, 532.0, RamanBranch.STOKES)
    anti = raman_shift_to_absolute_wavenumber_cm1(1000.0, 532.0, RamanBranch.ANTI_STOKES)
    assert stokes < excitation_cm1 < anti
    # A Stokes photon is redder than the laser; an anti-Stokes photon bluer.
    assert wavenumber_cm1_to_wavelength_nm(stokes) > 532.0
    assert wavenumber_cm1_to_wavelength_nm(anti) < 532.0


def test_raman_shift_wavelength_round_trip():
    nm = raman_shift_to_scattered_wavelength_nm(SHIFTS_CM1, 785.0)
    back = scattered_wavelength_nm_to_raman_shift(nm, 785.0)
    assert np.allclose(back, SHIFTS_CM1, rtol=1e-9)


def test_raman_conversion_without_excitation_refuses_to_guess():
    with pytest.raises(MissingMetadataError, match="excitation"):
        raman_shift_to_absolute_wavenumber_cm1(1000.0, None)


def test_a_known_shift_maps_to_the_documented_wavelength():
    # The silicon calibration line at 520.7 cm^-1 under 532 nm excitation sits at ~547.1 nm.
    # Computed here rather than quoted: 1e7/532 = 18796.99 cm^-1; minus 520.7 -> 18276.29;
    # 1e7/18276.29 = 547.16 nm.
    nm = raman_shift_to_scattered_wavelength_nm(520.7, 532.0)
    assert nm == pytest.approx(547.16, abs=0.01)


def test_convert_axis_is_identity_for_same_kind():
    values = np.array([1.0, 2.0, 3.0])
    out = convert_axis(values, AxisKind.WAVELENGTH_NM, AxisKind.WAVELENGTH_NM)
    assert np.array_equal(out, values)


@pytest.mark.parametrize(
    "target",
    [AxisKind.WAVELENGTH_NM, AxisKind.ENERGY_EV, AxisKind.ABSOLUTE_WAVENUMBER_CM1],
)
def test_convert_axis_round_trips_through_every_kind(target):
    values = SHIFTS_CM1
    forward = convert_axis(values, AxisKind.RAMAN_SHIFT_CM1, target, excitation_nm=785.0)
    back = convert_axis(forward, target, AxisKind.RAMAN_SHIFT_CM1, excitation_nm=785.0)
    assert np.allclose(back, values, rtol=1e-9)


def test_convert_axis_across_the_raman_boundary_needs_excitation():
    with pytest.raises(MissingMetadataError):
        convert_axis(SHIFTS_CM1, AxisKind.RAMAN_SHIFT_CM1, AxisKind.WAVELENGTH_NM)


def test_absorbance_transmittance_round_trip():
    absorbance = np.array([0.0, 0.3, 1.0, 2.5])
    back = transmittance_to_absorbance(absorbance_to_transmittance(absorbance))
    assert np.allclose(back, absorbance, rtol=1e-12)
    # A = 1 means one decade of attenuation.
    assert absorbance_to_transmittance(1.0) == pytest.approx(0.1)


def test_transmittance_in_percent_is_rejected_rather_than_guessed():
    with pytest.raises(UnitConversionError, match="percent"):
        transmittance_to_absorbance(np.array([50.0]))


def test_zero_transmittance_is_an_error_not_an_infinity():
    with pytest.raises(UnitConversionError):
        transmittance_to_absorbance(np.array([0.0]))


def test_kubelka_munk_round_trip():
    reflectance = np.array([0.05, 0.2, 0.5, 0.9, 1.0])
    back = kubelka_munk_to_reflectance(reflectance_to_kubelka_munk(reflectance))
    assert np.allclose(back, reflectance, rtol=1e-9)
    # Total reflectance means zero remission.
    assert reflectance_to_kubelka_munk(1.0) == pytest.approx(0.0)


def test_convert_intensity_refuses_undefined_pairs():
    with pytest.raises(UnitConversionError, match="no defined conversion"):
        convert_intensity(np.array([1.0]), IntensityKind.COUNTS, IntensityKind.ABSORBANCE)


def test_convert_intensity_dispatches_to_the_defined_pairs():
    out = convert_intensity(
        np.array([0.1, 0.5]), IntensityKind.TRANSMITTANCE, IntensityKind.ABSORBANCE
    )
    assert np.allclose(out, [1.0, np.log10(2.0)], rtol=1e-12)


def test_anti_stokes_ratio_is_below_one_and_increases_with_temperature():
    cold = anti_stokes_stokes_ratio(1000.0, 200.0)
    warm = anti_stokes_stokes_ratio(1000.0, 400.0)
    assert 0.0 < cold < warm < 1.0


def test_temperature_recovered_from_the_anti_stokes_ratio():
    for temperature in (150.0, 300.0, 600.0):
        ratio = anti_stokes_stokes_ratio(800.0, temperature, excitation_nm=532.0)
        recovered = temperature_from_anti_stokes_ratio(ratio, 800.0, excitation_nm=532.0)
        assert recovered == pytest.approx(temperature, rel=1e-9)


def test_impossible_anti_stokes_ratio_is_rejected():
    with pytest.raises(UnitConversionError, match="non-positive temperature"):
        temperature_from_anti_stokes_ratio(2.0, 800.0)


def test_scalar_in_scalar_out():
    assert isinstance(wavelength_nm_to_wavenumber_cm1(532.0), float)
    assert isinstance(wavenumber_cm1_to_ev(1000.0), float)
    assert isinstance(kubelka_munk_to_reflectance(0.5), float)
