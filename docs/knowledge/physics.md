# P0 — Physical and molecular foundations

Covers KEYWORD_MAP clusters **A** (light–matter interaction) and **B** (molecular and quantum
physics). Owns: `usis/measurement/`, `usis/simulate/`, `usis/knowledge/rules/`.

## What it is

A vibrational spectrum is a measurement of how a molecule's nuclei move. Nuclear motion is
decomposed into `3N - 6` normal modes (`3N - 5` if linear), each an independent oscillator to first
order. Which modes a technique *sees* is decided by symmetry, and how strongly is decided by how the
molecule's charge distribution responds to that motion.

- **Infrared absorption** happens when a mode changes the dipole moment: intensity `∝ (∂μ/∂Q)²`.
- **Raman scattering** happens when a mode changes the polarisability: activity `∝ (∂α/∂Q)²`.

These are different derivatives of different quantities, which is why the two techniques are
complementary rather than redundant, and why a model trained on one does not automatically transfer
to the other.

## Equations that matter to us

| Relation | Form | Where we use it |
| --- | --- | --- |
| Harmonic oscillator | `ν̃ = (1/2πc)·√(k/μ)` | `laws.harmonic_wavenumber_cm1` |
| Isotope shift | `ν̃'/ν̃ = √(μ/μ')` at fixed `k` | E0 probe T2, `laws.isotope_shifted_wavenumber` |
| Reduced mass | `μ = m₁m₂/(m₁+m₂)` | both of the above |
| Mutual exclusion | centrosymmetric ⇒ Raman-active (`g`) and IR-active (`u`) sets are disjoint | E0 probe T3, `laws.mutual_exclusion_violations` |
| Boltzmann / anti-Stokes ratio | `I_AS/I_S = ((ν̃₀+ν̃)/(ν̃₀−ν̃))⁴ · exp(−hcν̃/k_BT)` | sample thermometry, measurement audit; `units.anti_stokes_stokes_ratio` |
| Raman scattering strength | `∝ ν̃_scattered⁴` | why 532 nm excitation scatters far more strongly than 1064 nm |
| Raman shift definition | `Δν̃ = 1/λ_ex − 1/λ_scat` (λ in cm) | `units.raman_shift_to_absolute_wavenumber_cm1` |

The isotope law is the sharpest cheap probe we have: substituting H→D roughly divides an X–H stretch
frequency by √2 (exactly 0.707 only for an infinitely heavy partner; for O–H→O–D the harmonic factor
is ≈ 0.728, which our test pins). It is unambiguous, computable from first principles, and absent
from typical training sets. A pattern-matcher cannot pass it by memorising.

## Assumptions that break

1. **The harmonic approximation.** Real potentials are anharmonic. Computed harmonic frequencies are
   systematically too high, which is why empirical scaling factors exist — and why a scaling factor
   fitted at one level of theory is not transferable to another. This is the single largest source of
   simulation→experiment bias.
2. **Mode independence.** Fermi resonance and general mode coupling redistribute intensity between
   modes that happen to be near-degenerate. A "band" is then not a mode.
3. **The isolated molecule.** Hydrogen bonding, solvent, crystal packing, phase and temperature all
   shift and broaden bands. A band position is a conditional prior, never a truth — which is why
   `usis/core/constants.py` contains calibration anchors and no band assignments.
4. **A single conformer.** Room-temperature spectra are ensembles. Comparing to one optimised
   geometry compares to something that does not exist.
5. **Selection rules are exact only for the idealised point group.** Real crystals have site symmetry
   and defects; "forbidden" bands appear weakly. The mutual-exclusion checker must therefore report a
   *violation rate*, not a pass/fail, and must return "rule does not apply" for unrecognised groups
   rather than a silent pass. `laws.is_centrosymmetric` is written that way deliberately.

## Implications for our code

- Physical laws are **inference-time checkers** in `usis/knowledge/rules/`, never loss terms, unless
  a measured generalisation benefit is demonstrated (KEYWORD_MAP §M gate).
- Point-group information must survive from a predicted structure to the checker, so any structure
  representation that discards symmetry costs us probe T3.
- The `ν⁴` factor means intensity comparisons across excitation wavelengths are meaningless without
  correction — a cross-instrument trap that the forward model must own.
- Anti-Stokes/Stokes gives a free temperature readout when both branches are measured. Where the
  measurement covers both, the audit should report it and flag disagreement with a stated sample
  temperature.

## Gap statement

Nobody has cleanly decomposed the sim→real gap into *harmonic approximation* vs *solvent* vs
*conformer ensemble* vs *instrument response* for Raman at scale. The 78-compound RRUFF ∩ computed
overlap (`datasets.md` §2) is the smallest asset that could answer it. → `PROJECT_NOVELTY.md` N5,
open question Q2.
