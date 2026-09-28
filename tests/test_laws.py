"""Physical laws used as constraints and as falsification probes."""

from __future__ import annotations

import pytest

from usis.knowledge.rules.laws import (
    ATOMIC_MASS_U,
    PhysicalLawError,
    check_closure,
    check_non_negativity,
    harmonic_wavenumber_cm1,
    is_centrosymmetric,
    isotope_shifted_wavenumber,
    mutual_exclusion_violations,
    reduced_mass,
)


def test_reduced_mass_is_symmetric_and_below_both_masses():
    mu = reduced_mass(ATOMIC_MASS_U["O16"], ATOMIC_MASS_U["H"])
    assert mu == reduced_mass(ATOMIC_MASS_U["H"], ATOMIC_MASS_U["O16"])
    assert mu < ATOMIC_MASS_U["H"] < ATOMIC_MASS_U["O16"]


def test_oh_to_od_shift_approaches_the_one_over_root_two_limit():
    mu_oh = reduced_mass(ATOMIC_MASS_U["O16"], ATOMIC_MASS_U["H"])
    mu_od = reduced_mass(ATOMIC_MASS_U["O16"], ATOMIC_MASS_U["D"])
    shifted = isotope_shifted_wavenumber(3600.0, mu_oh, mu_od)
    # The idealised limit for a heavy partner is 1/sqrt(2) = 0.707; oxygen is not infinitely
    # heavy, so the real harmonic factor is slightly larger.
    assert 0.70 < shifted / 3600.0 < 0.74
    assert shifted == pytest.approx(2620.0, abs=20.0)


def test_isotope_substitution_never_raises_a_frequency_for_a_heavier_isotope():
    mu_ch = reduced_mass(ATOMIC_MASS_U["C12"], ATOMIC_MASS_U["H"])
    mu_cd = reduced_mass(ATOMIC_MASS_U["C12"], ATOMIC_MASS_U["D"])
    assert isotope_shifted_wavenumber(3000.0, mu_ch, mu_cd) < 3000.0


def test_isotope_law_is_self_inverse():
    mu_a, mu_b = 0.95, 1.79
    there = isotope_shifted_wavenumber(3600.0, mu_a, mu_b)
    back = isotope_shifted_wavenumber(there, mu_b, mu_a)
    assert back == pytest.approx(3600.0, rel=1e-12)


def test_harmonic_wavenumber_matches_a_hand_computed_case():
    # A 500 N/m bond between C and H. mu = 0.92974 u = 1.5439e-27 kg, so
    # sqrt(k/mu) = 5.691e14 rad/s and nu = 5.691e14 / (2*pi*2.9979e10 cm/s) = 3021 cm^-1 -
    # the right order for a C-H stretch. Computed from the closed form, not fitted.
    mu = reduced_mass(ATOMIC_MASS_U["C12"], ATOMIC_MASS_U["H"])
    assert mu == pytest.approx(0.92974, abs=1e-5)
    assert harmonic_wavenumber_cm1(500.0, mu) == pytest.approx(3021.2, abs=1.0)


def test_stiffer_bonds_and_lighter_atoms_raise_the_frequency():
    mu = reduced_mass(ATOMIC_MASS_U["C12"], ATOMIC_MASS_U["H"])
    assert harmonic_wavenumber_cm1(1000.0, mu) > harmonic_wavenumber_cm1(500.0, mu)
    heavier = reduced_mass(ATOMIC_MASS_U["C12"], ATOMIC_MASS_U["D"])
    assert harmonic_wavenumber_cm1(500.0, heavier) < harmonic_wavenumber_cm1(500.0, mu)


def test_invalid_inputs_raise():
    with pytest.raises(PhysicalLawError):
        reduced_mass(0.0, 1.0)
    with pytest.raises(PhysicalLawError):
        isotope_shifted_wavenumber(-1.0, 1.0, 1.0)
    with pytest.raises(PhysicalLawError):
        harmonic_wavenumber_cm1(-1.0, 1.0)


@pytest.mark.parametrize("group", ["Ci", "C2h", "D2h", "D6h", "D3d", "S6", "Oh", "Ih", "D*h"])
def test_centrosymmetric_groups_are_recognised(group):
    assert is_centrosymmetric(group)


@pytest.mark.parametrize("group", ["C1", "Cs", "C2v", "C3v", "D3h", "D2d", "Td", "S4", "C3h"])
def test_non_centrosymmetric_groups_are_recognised(group):
    assert not is_centrosymmetric(group)


def test_unknown_point_group_does_not_silently_pass_the_rule():
    # An unrecognised label must mean "the rule does not apply", never "the rule was satisfied".
    assert is_centrosymmetric("not-a-point-group") is False
    assert mutual_exclusion_violations("not-a-point-group", {"m1"}, {"m1"}) == set()


def test_mutual_exclusion_flags_a_mode_active_in_both_techniques():
    assert mutual_exclusion_violations("D2h", {"ag", "b1g"}, {"b1u"}) == set()
    assert mutual_exclusion_violations("D2h", {"ag", "b1u"}, {"b1u"}) == {"b1u"}


def test_mutual_exclusion_does_not_apply_without_a_centre_of_symmetry():
    assert mutual_exclusion_violations("C2v", {"a1"}, {"a1"}) == set()


def test_closure_and_non_negativity_checks():
    assert check_closure([0.5, 0.5]) == 0.0
    assert check_closure([0.5, 0.4]) == pytest.approx(-0.1)
    assert check_non_negativity([0.1, -0.2, 0.3]) == [1]
    with pytest.raises(PhysicalLawError):
        check_closure([])
