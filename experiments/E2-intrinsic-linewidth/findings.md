# E2 — findings

**Run 2026-09-03** at git `9ba925a`, clean tree. Protocol and thresholds fixed in
`preregistration.md`, committed at `e6e7238` and clarified at `d4a6184`, both **before** the runner
existed and long before any number did. Raw output in `result_card.md` and `comparison.json`.

Corpus: **1,129 ROD** records (the whole published index, minus two that contradict their own
declared range), **226 QM9S** Raman spectra, **279** *Sci Data* spectra. ROD split
`sample_disjoint`, seed 20260903: 564 fit / 565 test, 305 / 316 specimens.

*A debug run was made earlier against a partially-downloaded corpus (210 ROD records) to shake out
crashes. It is disclosed here because it showed held-out numbers. Nothing was changed as a result:
there is no hand-settable parameter in this experiment — the intrinsic width, its spread, the noise
level and the baseline swing are all estimated by fixed procedures from the fit half, and the split
seed was fixed in the preregistration.*

## Answer: the second fix is the whole improvement — and the first one does exactly what it was built to do while making the comparison worse

| Arm | broadening | scale | comparable (of 4) | total \|gap\| |
| --- | --- | --- | --- | --- |
| `A0B0` control (= E1) | off | `MEDIAN` | 2 | 7.67 IQR |
| `A1B0` | **on** | `MEDIAN` | 1 | 6.67 IQR |
| `A0B1` | off | **`AMPLITUDE`** | **3** | **2.09 IQR** |
| `A1B1` | **on** | **`AMPLITUDE`** | 3 | 4.73 IQR |

Under the pre-registered rule — more comparable statistics, or the same number with a strictly
smaller total gap:

* **Factor B (the scale mode) is an improvement in both contrasts.** `A0B1` vs `A0B0`: 2 → 3.
  `A1B1` vs `A1B0`: 1 → 3. It is a main effect, and it is large.
* **Factor A (intrinsic broadening) is not an improvement in either contrast.** `A1B0` vs `A0B0`:
  2 → 1. `A1B1` vs `A0B1`: 3 → 3 but the total gap grows from 2.09 to 4.73 IQR. The two contrasts
  agree in direction, so this is not an interaction — it is a consistent negative result.

## The control reproduces E1 on a corpus sampled differently

E1 reported `noise_variance_slope` and `n_bands` comparable, `dynamic_range` and
`autocorrelation_length_cm1` not. `A0B0` — E1's configuration — reports **exactly that pattern**,
on a ROD corpus 4.7x larger drawn from the database's own index instead of guessed id blocks, and
on QM9S and *Sci Data* samples taken at spread byte offsets instead of as prefixes.

That is worth more than it looks. E1's corpus is gone and could not be reconstructed; the
worry was that its answer was an artifact of a prefix sample. It was not.

## Factor B: the dynamic-range gap was ours, and it is now closed

E1 measured `dynamic_range` at **+18.4 IQR**, its largest failure by a factor of fifteen, and
diagnosed part of it as our own parameterisation: additive amplitudes were expressed against the
median absolute intensity, which on a measured spectrum is the baseline but on a simulated one is
the near-zero floor between bands.

| Arm | `dynamic_range` gap |
| --- | --- |
| `A0B0` (median) | +5.52 IQR |
| `A0B1` (amplitude) | **+0.08 IQR** |
| `A1B1` (amplitude) | **+0.00 IQR** |

Expressing the amplitude against the peak-to-floor signal instead closes it essentially exactly.
The estimated parameters differ by two orders of magnitude between the modes
(`noise_relative` 0.0166 vs 0.000206) — which is the point: the same number meant two different
physical things, and neither corpus could tell you which.

**This was a defect in our code, not in the physics.** E1 said so before this experiment was
designed, and the size of the effect is the reason it was worth an arm of its own rather than a
quiet fix.

## Factor A: the linewidth model is correct and the profile is wrong

The manipulation checks — fitted quantities, not evidence — say the operator does exactly what it
was built to do:

| Check | `A0B0` | `A1B0` | real |
| --- | --- | --- | --- |
| `median_band_fwhm_cm1` | 11.0 (−1.01 IQR) | **23.0 (−0.01 IQR)** | 23.14 [17.84, 29.89] |
| `band_fwhm_iqr_cm1` | 3.0 (−0.59 IQR) | **11.0 (+0.18 IQR)** | 9.16 [5.15, 15.64] |

E1's headline defect is gone. The simulated widths were `[11.00, 11.00]` — every band of every
molecule identical, a perfect shortcut feature (`F14`). They now have both the right median and the
right spread, from a lognormal fitted on a disjoint half of the real corpus
(**median 14.27 cm⁻¹, geometric sd 2.15, from 4,269 bands over 562 spectra**).

And the held-out statistics get **worse**:

| Held out | `A0B0` | `A1B0` | real |
| --- | --- | --- | --- |
| `autocorrelation_length_cm1` | 33.0 (+1.04 IQR) | **73.0 (+4.64 IQR)** | 21.49 [17.36, 28.44] |
| `noise_variance_slope` | 6.8e-4 (+0.48, comparable) | 3.5e-5 (−0.13, not) | 1.69e-4 [3.8e-5, 1.1e-3] |
| `n_bands` | 7.5 (+0.62) | **5.0 (+0.00)** | 5 [4, 8] |

`n_bands` lands exactly, which is the consequence broadening was expected to have and nobody
fitted. But `autocorrelation_length_cm1` goes the wrong way by a factor of three, and it is the
statistic that carries the finding:

> **Matching a band's width does not match its shape.** At the same median FWHM, the simulated
> spectra decorrelate over 73 cm⁻¹ and the real ones over 21 cm⁻¹.

The cause is the profile, not the width. A Lorentzian's wings fall off as `1/x²`: at three
half-widths from the centre a Lorentzian still carries ~10% of its peak, and summed over every band
in a spectrum that builds a broad correlated pedestal. Real ROD bands at the *same* measured width
decorrelate three times faster, so their wings are lighter than a Lorentzian's. That points at
inhomogeneous broadening — site disorder, strain, particle-size distribution in a mineral powder —
dominating over lifetime broadening, which is a Gaussian contribution, not a Lorentzian one.

So the negative result is specific and it names its successor: **the intrinsic profile should be a
Voigt with a fitted Gaussian/Lorentzian mixing ratio, and `autocorrelation_length_cm1` is the
statistic that scores it.** `usis/measurement/broadening.py` already takes a `shape` argument and
already composes widths through the Olivero–Longbothum approximation; what is missing is the mix,
and a way to fit it that does not use a statistic already spent.

**The code stays.** E1's measurement — that real ROD bands carry ≈12–14 cm⁻¹ of width no instrument
put there — is a property of the data and is unaffected by this result. A forward model that cannot
express intrinsic width cannot express that fact at all. What E2 refutes is the claim that a
*Lorentzian* intrinsic width, independent per band, closes the sim-to-real gap. It does not.

## The best arm is not the one with the most physics in it

`A0B1` — no intrinsic broadening, corrected scale — is the best arm at 3/4 and 2.09 IQR. It is also
the arm whose simulated band widths are wrong by a factor of two and have almost no spread. It
would still be trivially separable from real data by band width alone.

That is worth stating plainly because the pre-registered score does not capture it: **an arm can win
on these four statistics while retaining the exact defect E1 identified as the most damaging one for
sim-to-real transfer.** The scoring rule was fixed in advance and is reported as fixed; the
limitation is the rule's, and it is an argument for `band_fwhm_iqr_cm1` becoming a held-out
statistic in a future experiment that does not fit it.

## What this changes

* **`ScaleMode.AMPLITUDE` is the right default for additive terms on simulated spectra**, measured
  rather than argued. There is still no default in the code, and there should not be: the two modes
  mean different things on different corpora, and the caller has to say which.
* **Q2 (`docs/knowledge/open_questions.md`) gets a second quantified target.** The first was
  ≈12.5 cm⁻¹ of unmodelled intrinsic width; it is now closed to −0.01 IQR. The second is the wing
  shape: 73 cm⁻¹ of simulated correlation length against 21 cm⁻¹ real, at matched width.
* **`PROJECT_NOVELTY.md`**: the sim-to-real linewidth candidate moves from "the simulation has no
  linewidth" to "the simulation has the wrong lineshape", which is a narrower and more testable
  claim.

## What this does not show

* Nothing about chemistry: band *positions* were never compared.
* Nothing about a trained model; none was involved.
* The corpora are not matched — ROD is minerals measured on grating spectrometers, QM9S is small
  organic molecules computed at B3LYP/def-TZVP. The linewidth and lineshape findings are properties
  of the two *methods*, not of the molecules, but the `dynamic_range` and `autocorrelation` numbers
  are not cleanly separable from the composition difference.
* The intrinsic width is fitted from ROD only. *Sci Data* records no resolution, so it cannot be
  deconvolved, and it contributes only to the noise and baseline estimates.
* The instrument response is assumed **Gaussian** for the deconvolution. A Lorentzian assumption
  would subtract linearly and return a different intrinsic width. That assumption is stated in the
  runner and is not tested here.
* QM9S's 10 cm⁻¹ of publisher-applied broadening is treated as though it were an instrument
  response by the resolution-matching step. It is not one; it is an arbitrary choice by the
  publisher. This assumption is inherited from E1 and is a candidate for its own experiment.
* The QM9S and *Sci Data* corpora are partial fetches at fixed byte offsets — not random samples.
  `FETCH.json` in each dataset directory records the spans.
