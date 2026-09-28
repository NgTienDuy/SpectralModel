# E2 — Does an intrinsic linewidth drawn from a distribution close the sim-to-real gap?

**Written 2026-09-03, before the comparison was run.** Follow-up to E1
(`../E1-sim2real-comparability/`), which answered its own question with *no* and named two causes.
This experiment tests whether fixing them changes the answer, and attributes the change to one
cause or the other.

*One clarification was made after the first commit of this file and before any E2 number existed:
the sentence naming what the degradation parameters are estimated from said "the fit half only" in
one place and included* Sci Data *in another. It now says the ROD fit half plus the whole* Sci Data
*corpus, both disjoint from the ROD test half. The commit order is the evidence that this preceded
the run.*

## The question

E1 found that 2 of 5 held-out statistics were comparable between degraded simulated spectra and
real ones, and that the three failures shared one cause: **the simulation has no intrinsic
linewidth**. It named two design changes:

1. an intrinsic-broadening term, separate from the instrument response and **drawn from a
   distribution rather than a constant**;
2. a mode for additive amplitudes on spectra whose median is a floor rather than a baseline.

Both are now implemented (`usis/measurement/broadening.py`, `ScaleMode` in
`usis/measurement/forward_model.py`). E2 asks whether they work, and it asks about each separately,
because two changes made at once cannot be attributed.

## Design: 2 x 2 factorial, four arms

| Arm | Intrinsic broadening | Additive scale mode |
| --- | --- | --- |
| `A0B0` — **control** | off | `MEDIAN` (E1's implicit choice) |
| `A1B0` | on | `MEDIAN` |
| `A0B1` | off | `AMPLITUDE` |
| `A1B1` | on | `AMPLITUDE` |

`A0B0` is E1's configuration. It is included so that the comparison to E1 is made against a run on
*this* corpus rather than against E1's numbers, which came from a differently-sampled one.

## Corpora — and how they differ from E1's

Rebuilt through the committed loaders, because E1's fetched corpus was not kept.

| Role | Source | Selection | Notes |
| --- | --- | --- | --- |
| Simulated | QM9S Raman (`CC-BY-4.0`) | 8 byte-strided chunks, ~0.18% of the 8.8 GB file | Lorentzian 10 cm⁻¹ applied by the publisher, 500–4000 cm⁻¹ |
| Real A | Raman Open Database (`CC0-1.0`) | every code in the database's own index | 4 cm⁻¹ declared, 532/780/514 nm |
| Real B | *Sci Data* 12:498 (`CC-BY-4.0`) | 8 byte-strided chunks, ~8% of the file | One instrument, 785 nm, resolution unrecorded |

Three differences from E1, all of which make cross-experiment comparison of raw numbers invalid and
are the reason the control arm exists:

* **ROD is now the whole index (1,133 codes), not a 240-record scan of guessed id blocks.** The
  index also exposed that ROD serves CIF 1.x as well as CIF 2.0 records, which our fetcher had been
  rejecting as if they were failures.
* **The partial fetches are byte-strided, not prefixes.** Both published CSVs are ordered — QM9S by
  molecule index, so its first rows are its smallest molecules, and the *Sci Data* matrix by
  compound, so its first rows are all one compound in one vial. E1 took prefixes of both.
* **Neither partial fetch is a random sample.** The rows are whichever ones the byte offsets landed
  in. `FETCH.json` in each dataset directory records this, and it is a limitation of every number
  below.

## What is estimated, what is tested — and what moved between them

E1 held out five statistics. **One of them is being fitted in E2 and therefore stops being a
test.** Saying so is the whole reason this section exists: silently keeping `median_band_fwhm_cm1`
in the held-out list while setting a parameter from it would make the experiment look successful by
construction.

**Estimated from real spectra that are disjoint from the comparison set — cannot be evidence.**
Concretely: the ROD **fit** half, plus the whole *Sci Data* corpus, which is a different dataset on
a different instrument and never enters the comparison at all. Nothing in this list touches the ROD
test half.

1. `noise_relative`, expressed against the arm's own scale mode.
2. `baseline_swing`, likewise.
3. `median_band_fwhm_cm1` → the intrinsic width, via
   `deconvolved_fwhm(observed, declared instrument resolution)` per band. **Moved out of the
   held-out set for E2.**
4. the spread of those per-band intrinsic widths → the lognormal `geometric_sd`. Also fitted.

**Manipulation checks — fitted quantities, verified on a disjoint half. Not evidence:**

* `median_band_fwhm_cm1`
* `band_fwhm_iqr_cm1` (new): the interquartile range of band widths *within* a spectrum. This is
  the E1 defect in its purest form — QM9S measured `[11.00, 11.00]`, every band identical — and a
  fix that does not move it has not done what it claims. It is a check and not a test because the
  spread was fitted.

**Held out, and the actual test — no parameter is set from any of these:**

5. `noise_variance_slope`
6. `n_bands`
7. `dynamic_range`
8. `autocorrelation_length_cm1`

`n_bands` and `autocorrelation_length_cm1` are the informative tests for factor A: broadening
merges neighbouring bands and lengthens the autocorrelation, and neither consequence was fitted.
`dynamic_range` is the informative test for factor B, and changing the *mode* an amplitude is
expressed against is a structural change, not a fit to it.

## Protocol

1. Load the three corpora through the committed loaders.
2. Split ROD **`sample_disjoint`, seed 20260903, test_fraction 0.5** into a fit half and a test
   half. The sample key is `_rod_data_source.file`, so specimens measured at two excitations stay
   on one side (`PROJECT_DECISIONS.md` D9). Everything in "estimated" above is measured on the
   **ROD fit half plus the *Sci Data* corpus**; every comparison is against the **ROD test half
   only**. *Sci Data* cannot be a comparison set because it records no resolution and so cannot be
   resolution-matched — the same limitation E1 had.
3. Resolution-match the real test half up to the simulation's 10 cm⁻¹, exactly as E1 did: QM9S
   carries 10 cm⁻¹ and ROD declares 4 cm⁻¹, convolution cannot sharpen, so the direction is forced.
4. For each of the four arms, degrade the simulated corpus with the estimated parameters plus that
   arm's two switches, and compute the statistics.
5. Compare each held-out statistic to the real test half under the rule below.

## What counts as comparable — decided now

Unchanged from E1, so the two experiments' verdicts mean the same thing:

> a statistic is **comparable** when the median over the degraded-simulated corpus falls inside the
> [25th, 75th] percentile of the real corpus.

Effect size, likewise unchanged:

```
standardised gap = (median_simulated - median_real) / IQR_real
```

## The pre-registered outcome

**Primary:** the number of the four held-out statistics that are comparable, per arm.

**A change counts as an improvement only if**, relative to the control arm `A0B0`, it either makes
strictly more held-out statistics comparable, or leaves the count unchanged while strictly
reducing the sum of |standardised gap| over all four. Both numbers are reported for every arm
whichever way it comes out.

**Attribution:** factor A's effect is `A1B0` vs `A0B0` and `A1B1` vs `A0B1`; factor B's is `A0B1`
vs `A0B0` and `A1B1` vs `A1B0`. If the two contrasts for one factor disagree in sign, the factors
interact and the result is reported as an interaction rather than as a main effect.

## Interpretation rules — decided now

* **The change helps.** Report by how much, per factor, and note that it is measured on a corpus
  whose real half is minerals and whose simulated half is small organics — the confound E1 already
  flagged and this experiment does not remove.
* **The change does not help.** That is a negative result about *this parameterisation* of
  intrinsic width — a lognormal, independent per band, with no dependence on wavenumber or mode
  symmetry. It would not overturn E1's measurement that real ROD bands carry ≈12.5 cm⁻¹ of
  non-instrumental width, which is a property of the data and stands on its own. The code stays,
  because it is physically right; the claim does not get made.
* **The manipulation checks fail.** Then the operator does not do what it was set to do and the
  held-out results are not interpretable at all. Report that first and stop.

## The rules I am most likely to break, written down so I cannot

1. **No parameter will be adjusted after seeing the held-out statistics.** Same rule as E1, same
   reason. If the arms disagree with the hypothesis, that is the finding.
2. **The manipulation checks will not be reported as evidence.** They are fitted. A table that
   lists them beside the held-out statistics without saying which is which would read as six
   successes when there are at most four.
3. **The four arms will all be reported**, including any that is worse than the control. Reporting
   the best arm is the same error as tuning, taken one step later.

## What this experiment cannot show

* Nothing about chemistry: band *positions* are still never compared.
* Nothing about a trained model's transferability; no model is involved.
* Nothing about matched samples. Real is minerals and solvents, simulated is small organics, and
  the spectral ranges differ.
* Nothing about the *Sci Data* corpus's resolution, which is unrecorded, so it contributes to the
  parameter estimate but cannot be resolution-matched or deconvolved.
* **Nothing about whether the fitted intrinsic width is right for anything but ROD minerals.** It
  is estimated from one corpus on one instrument family and applied to small organic molecules.
* **Nothing that rests on QM9S's 10 cm⁻¹ being physical.** It is a publisher's arbitrary choice,
  and treating it as though it were an instrument response — which the resolution-matching step
  does — is an assumption this experiment inherits from E1 rather than tests.
* Nothing about the modalities other than Raman, or about SERS.
