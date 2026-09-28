# E1 — Are degraded simulated spectra statistically comparable to real ones?

**Written 2026-09-03, before the comparison was run.** Closes acceptance criterion B7 in
`PROJECT_ACCEPTANCE.md`.

## The question

The M1 forward model turns a clean spectrum into an "observed" one. B7 asks whether the result
resembles a real measurement — not whether the *chemistry* is right, which is a different and later
question, but whether the **measurement character** matches: noise, background, band structure,
dynamic range.

This matters because everything downstream of "train on simulated data" depends on it, and because
we have abundant simulated data (QM9S, 133,885 molecules) and scarce experimental data.

## Corpora

| Role | Source | n | Notes |
| --- | --- | --- | --- |
| Simulated | QM9S Raman (`CC-BY-4.0`) | 42 | Broadened, Lorentzian 10 cm⁻¹ (measured by us), 500–4000 cm⁻¹ |
| Real A | Raman Open Database (`CC0-1.0`) | 240 | Thermo Nicolet Almega XR and HORIBA T64000, 532/780/514 nm |
| Real B | *Sci Data* 12:498 API compounds (`CC-BY-4.0`) | ~30 | Endress+Hauser Rxn2, 785 nm, one instrument |

Two confounds are present by construction and are **not** controlled: the corpora cover different
spectral ranges, and different chemistry (minerals vs solvents vs small organics). Any conclusion is
about measurement character on overlapping statistics, not about matched samples.

## The statistics

Fixed in advance, implemented in `usis/measurement/diagnostics.py`, and split into two roles.

**Used to set the forward-model parameters (so they cannot be part of the test):**

1. `noise_relative` — robust noise sigma over the median absolute intensity.
2. `baseline_swing` — peak-to-peak of the fitted lower envelope, relative to signal scale.
3. resolution — taken from each corpus's own recorded `resolution_fwhm`, not fitted.

**Held out, and the actual test:**

4. `noise_variance_slope` — how noise variance scales with signal. Distinguishes photon-limited
   from readout-limited, and is *not* used to set any parameter.
5. `n_bands` — count of local maxima above the extreme-value noise threshold.
6. `median_band_fwhm_cm1` — median band width.
7. `dynamic_range` — 99.9th percentile over median absolute intensity.
8. `autocorrelation_length_cm1` — lag at which the detrended autocorrelation falls below 1/e.

## Protocol

1. Load the three corpora through the committed loaders.
2. Compute diagnostics 1–2 on each **real** corpus; take the median as the degradation parameters.
3. Degrade the simulated corpus with exactly those parameters, plus the resolution step that the
   physics permits. **The direction is forced**: QM9S carries 10 cm⁻¹ of broadening and ROD reports
   4 cm⁻¹, and convolution cannot sharpen, so the real corpus is broadened up to the simulation
   rather than the reverse. Recorded in `docs/knowledge/datasets.md`.
4. Compute the held-out statistics 4–8 on both, and compare.

## What counts as comparable — decided now

For each held-out statistic, **comparable** means the median over the degraded-simulated corpus
falls inside the [25th, 75th] percentile of the real corpus. The reported effect size is

```
standardised gap = (median_simulated - median_real) / IQR_real
```

so a gap of 0 is a perfect match and a gap of 1 means the simulated median sits a full
interquartile range away.

## Interpretation rules — decided now

- **All five comparable.** The forward model reproduces the measurement character of real spectra on
  these axes. It would *not* license any claim about chemistry, band positions, or transferability
  of a trained model.
- **Some comparable.** Name which are not, and in which direction. Each failing statistic points at
  a specific missing piece of physics, and that list is the result.
- **None comparable.** The forward model is not yet a measurement model, and the sim-to-real
  programme needs the gap closed before any sim-trained result is meaningful.

**A negative answer is a valid result.** It feeds open question Q2 directly and is worth more than a
matched pair of histograms obtained by tuning.

## The rule I am most likely to break, written down so I cannot

**No parameter will be adjusted after seeing the held-out statistics.** Parameters come from
statistics 1–3, measured on the real corpora. If the held-out statistics disagree, that is the
finding and it gets reported as such. Tuning until the two look alike is fitting, not measuring, and
it would make this experiment worthless while making it look successful.

## What this experiment cannot show

- Nothing about chemistry: band *positions* are not compared, only their number and width.
- Nothing about a trained model's transferability; no model is involved.
- Nothing about the other modalities, or about SERS.
- Nothing about matched samples: the corpora differ in range and in chemistry, and that confound is
  not removed.
