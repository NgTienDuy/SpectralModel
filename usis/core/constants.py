"""Physical constants and instrument-calibration anchors.

Values are CODATA 2018 (as adopted by the SI 2019 redefinition), quoted exactly where the
constant is exact by definition. Every non-exact value carries its source in a comment.

Nothing in this module is a *band assignment*. Band positions are conditional priors and live in
the knowledge layer, never as constants. The only spectroscopic positions here are
instrument-calibration anchors, which are properties of a reference material under a stated
convention, not of chemistry in general.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CM1_PER_EV",
    "EV_PER_CM1",
    "K_B_EV_PER_K",
    "NM_CM1_PRODUCT",
    "PLANCK_J_S",
    "POLYSTYRENE_RAMAN_ANCHORS_CM1",
    "SILICON_RAMAN_CM1",
    "SPEED_OF_LIGHT_M_PER_S",
]

# --- Exact by SI definition (2019 redefinition) ---------------------------------------------
SPEED_OF_LIGHT_M_PER_S: Final[float] = 299_792_458.0
PLANCK_J_S: Final[float] = 6.626_070_15e-34
ELEMENTARY_CHARGE_C: Final[float] = 1.602_176_634e-19
BOLTZMANN_J_PER_K: Final[float] = 1.380_649e-23

# --- Derived conversion factors ---------------------------------------------------------------
# 1 eV expressed in cm^-1: e / (h * c), with c in cm/s.
CM1_PER_EV: Final[float] = ELEMENTARY_CHARGE_C / (PLANCK_J_S * SPEED_OF_LIGHT_M_PER_S * 100.0)
EV_PER_CM1: Final[float] = 1.0 / CM1_PER_EV
K_B_EV_PER_K: Final[float] = BOLTZMANN_J_PER_K / ELEMENTARY_CHARGE_C

# Wavenumber [cm^-1] * wavelength [nm] = 1e7 (exact, definitional unit bridge).
NM_CM1_PRODUCT: Final[float] = 1.0e7

# --- Calibration anchors ----------------------------------------------------------------------
# Crystalline silicon first-order optical phonon. The literature quotes 520.5-520.7 cm^-1
# depending on temperature, strain and doping; 520.7 cm^-1 at ~300 K is the value used for
# routine Raman wavenumber calibration. Treated as an anchor with a stated tolerance, never as
# an exact truth.
SILICON_RAMAN_CM1: Final[float] = 520.7
SILICON_RAMAN_TOLERANCE_CM1: Final[float] = 0.3

# Polystyrene is an ASTM-recognised Raman wavenumber standard (ASTM E1840). The anchor
# positions below are the ones we will calibrate against; their accepted values and
# uncertainties MUST be taken from the standard before this is used in anger.
# STATUS: UNVERIFIED - values not yet checked against ASTM E1840. Do not rely on them.
POLYSTYRENE_RAMAN_ANCHORS_CM1: Final[tuple[float, ...]] = ()
