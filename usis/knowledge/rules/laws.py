"""Physical laws used as inference-time checkers and as falsification probes.

Invariant 7: hard constraints are enforced, not learned. Each function here is either a checker
that a prediction must pass, or the closed-form law that the E0 battery tests a learned model
against. Nothing here is fitted to data.

Scope note: these are laws about *relationships* (how a frequency moves when a mass changes,
which modes may be active in which technique). Absolute band positions are not laws, they are
conditional priors, and they do not belong in this module or anywhere else in the code.
"""

from __future__ import annotations

import math
from typing import Final

from usis.core.constants import SPEED_OF_LIGHT_M_PER_S
from usis.core.errors import UsisError

__all__ = [
    "ATOMIC_MASS_U",
    "CENTROSYMMETRIC_POINT_GROUPS",
    "check_closure",
    "check_non_negativity",
    "harmonic_wavenumber_cm1",
    "is_centrosymmetric",
    "isotope_shifted_wavenumber",
    "mutual_exclusion_violations",
    "reduced_mass",
]

# Atomic masses of the isotopes the isotope-substitution probe uses, in unified atomic mass
# units. These are the standard nuclide masses, not standard atomic weights - the distinction
# matters for the H/D probe, where the natural-abundance average would be meaningless.
ATOMIC_MASS_U: Final[dict[str, float]] = {
    "H": 1.007825,
    "D": 2.014102,
    "T": 3.016049,
    "C12": 12.000000,
    "C13": 13.003355,
    "N14": 14.003074,
    "N15": 15.000109,
    "O16": 15.994915,
    "O18": 17.999160,
    "S32": 31.972071,
    "Cl35": 34.968853,
    "Cl37": 36.965903,
    "F19": 18.998403,
}

# Point groups containing an inversion centre. The mutual exclusion rule applies to exactly
# these: a molecule with a centre of symmetry has no mode that is both Raman- and IR-active.
# Cnh is centrosymmetric for even n, Dnh for even n, Dnd for odd n, S2n for odd n.
CENTROSYMMETRIC_POINT_GROUPS: Final[frozenset[str]] = frozenset(
    {
        "Ci",
        "S2",
        "C2h",
        "C4h",
        "C6h",
        "D2h",
        "D4h",
        "D6h",
        "D8h",
        "D3d",
        "D5d",
        "S6",
        "C3i",
        "Th",
        "Oh",
        "Ih",
        "Dinfh",
        "D*h",
        "Kh",
    }
)

_ATOMIC_MASS_KG: Final[float] = 1.660_539_066_60e-27  # CODATA 2018 unified atomic mass unit


class PhysicalLawError(UsisError):
    """A law was applied outside its domain of validity."""


def reduced_mass(mass_a_u: float, mass_b_u: float) -> float:
    """Reduced mass of a two-body oscillator, in unified atomic mass units."""
    if mass_a_u <= 0 or mass_b_u <= 0:
        raise PhysicalLawError("masses must be positive")
    return (mass_a_u * mass_b_u) / (mass_a_u + mass_b_u)


def isotope_shifted_wavenumber(
    wavenumber_cm1: float,
    reduced_mass_before_u: float,
    reduced_mass_after_u: float,
) -> float:
    """Predict a band position after isotope substitution, from ``nu ~ sqrt(k / mu)``.

    The force constant is assumed unchanged by substitution, which is the Born-Oppenheimer
    result and is accurate to a few percent for X-H/X-D stretches. This is E0 probe T2: the law
    is cheap, unambiguous, and absent from typical training sets, so a model that reproduces it
    has learned something about vibration rather than about the dataset.

    Deviations from this prediction are physically meaningful (anharmonicity, mode coupling,
    Fermi resonance), which is why the probe is scored with a tolerance rather than exactly.
    """
    if wavenumber_cm1 <= 0:
        raise PhysicalLawError("wavenumber must be positive")
    if reduced_mass_before_u <= 0 or reduced_mass_after_u <= 0:
        raise PhysicalLawError("reduced masses must be positive")
    return wavenumber_cm1 * math.sqrt(reduced_mass_before_u / reduced_mass_after_u)


def harmonic_wavenumber_cm1(force_constant_n_per_m: float, reduced_mass_u: float) -> float:
    """Harmonic oscillator wavenumber in cm^-1 from a force constant and a reduced mass."""
    if force_constant_n_per_m <= 0 or reduced_mass_u <= 0:
        raise PhysicalLawError("force constant and reduced mass must be positive")
    mass_kg = reduced_mass_u * _ATOMIC_MASS_KG
    angular = math.sqrt(force_constant_n_per_m / mass_kg)  # rad/s
    c_cm_per_s = SPEED_OF_LIGHT_M_PER_S * 100.0
    return angular / (2.0 * math.pi * c_cm_per_s)


def is_centrosymmetric(point_group: str) -> bool:
    """Whether a Schoenflies point-group label denotes a centrosymmetric group.

    Accepts the common ASCII spellings of the linear group (``D*h``, ``Dinfh``, ``D∞h``).
    Unknown labels return ``False``, so a caller never gets a mutual-exclusion *pass* from a
    label the checker did not understand - it gets "the rule does not apply", which the report
    must then state as such.
    """
    label = point_group.strip().replace("∞", "inf").replace("_", "")
    return label in CENTROSYMMETRIC_POINT_GROUPS


def mutual_exclusion_violations(
    point_group: str,
    raman_active_modes: set[str],
    ir_active_modes: set[str],
) -> set[str]:
    """Modes predicted active in both Raman and IR for a centrosymmetric molecule.

    For a centrosymmetric point group, ``g`` modes are Raman-active and ``u`` modes IR-active, so
    the two sets must be disjoint. Any overlap is a hard physical inconsistency in the
    prediction, reported as failure code ``F10``. For a non-centrosymmetric group the rule does
    not apply and the function returns an empty set.
    """
    if not is_centrosymmetric(point_group):
        return set()
    return raman_active_modes & ir_active_modes


def check_non_negativity(values: list[float], name: str = "values") -> list[int]:
    """Return the indices of negative entries. Used on concentrations and resolved spectra."""
    return [i for i, v in enumerate(values) if v < 0.0]


def check_closure(fractions: list[float], tolerance: float = 1e-6) -> float:
    """Return the closure residual ``sum(fractions) - 1``; the caller decides what to do with it."""
    if not fractions:
        raise PhysicalLawError("closure is undefined for an empty composition")
    residual = math.fsum(fractions) - 1.0
    return 0.0 if abs(residual) < tolerance else residual
