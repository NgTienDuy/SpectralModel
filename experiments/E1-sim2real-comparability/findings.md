# E1 — findings

**Run 2026-09-03.** Protocol and thresholds fixed in `preregistration.md`, committed before the
run (`4588cec`). Raw output in `result_card.md` and `comparison.json`.

Corpus: **240 ROD** records (Thermo Nicolet Almega XR and HORIBA T64000; 532/780/514 nm),
**45 QM9S** Raman spectra, **67** *Sci Data* API-compound spectra. Larger than the preregistration
estimated, and fetched through the committed loaders.

## Answer: no. Two of five held-out statistics are comparable.

| Statistic | Simulated | Real (median [IQR]) | Gap | Verdict |
| --- | --- | --- | --- | --- |
| `noise_variance_slope` | 1.62e-4 | 1.11e-4 [3.5e-5, 3.8e-4] | +0.15 IQR | **comparable** |
| `n_bands` | 5 | 6 [4, 9] | −0.20 IQR | **comparable** |
| `median_band_fwhm_cm1` | 11.0 | 24.6 [19.0, 29.9] | −1.25 IQR | not comparable |
| `dynamic_range` | 1731 | 50.2 [28.3, 120] | **+18.4 IQR** | not comparable |
| `autocorrelation_length_cm1` | 13.0 | 21.2 [17.7, 26.2] | −0.97 IQR | not comparable |

Per the pre-registered interpretation rules, "some comparable" means the failing statistics name
the missing physics. They do, and all three point the same way.

## The headline: the simulation has no intrinsic linewidth

Measured on the raw corpora, before any matching:

| | Band FWHM, median | IQR | Stated resolution | Implied intrinsic width |
| --- | --- | --- | --- | --- |
| ROD (real) | 16.5 cm⁻¹ | [12.1, 23.1] | 4 cm⁻¹ | **≈ 12.5 cm⁻¹** |
| QM9S (simulated) | 11.00 cm⁻¹ | **[11.00, 11.00]** | 10 cm⁻¹ applied | **≈ 1.0 cm⁻¹** (one grid step) |

Two things are visible there and the second is the more damaging.

1. **Real bands carry about 12 cm⁻¹ of width the simulation does not model.** Real Raman linewidths
   come from phonon lifetime, disorder, inhomogeneous broadening and thermal effects. A harmonic
   DFT stick spectrum has none of that, so its width is *entirely* the broadening parameter.
   Matching the instrument resolution therefore does **not** match the bands, and the residual gap
   is physics rather than instrumentation.

2. **The simulated width has zero spread.** The interquartile range is `[11.00, 11.00]`: every band
   of every molecule has the same width, because one constant was applied to all of them. No real
   spectrum looks like that. That is a perfect shortcut feature — a classifier could separate
   simulated from real on band width alone without learning any chemistry, which is failure code
   `F14`, and it means **any sim-to-real domain adaptation that does not address linewidth is
   adapting the wrong variable.**

## The dynamic-range gap is partly an artifact of our own parameterisation

Simulated dynamic range is 50× the real one. Some of that is real — a simulation has no detector
floor — but part is a defect in how we expressed the degradation, and it is worth recording rather
than quietly fixing:

`BaselineDrift` and `ReadoutNoise` take amplitudes **relative to the median absolute intensity**.
On a real spectrum the median *is* the baseline. On a simulated one the median is the near-zero
floor between bands, so "1.14 × median" adds almost nothing. Measured medians make this concrete:

| | `noise_relative` | `baseline_swing` | `dynamic_range` |
| --- | --- | --- | --- |
| ROD (real) | 0.051 | 1.26 | 57 |
| QM9S (raw) | 5.7e-5 | 5.91 | 2862 |

The simulation is ~900× less noisy in relative terms, and its `baseline_swing` is *larger* than the
real corpus's precisely because the ratio is taken against a near-zero median.

**Per the preregistration, no parameter was adjusted after seeing this.** The fix is a design
change to be made and re-run, not a tuning knob: additive terms should be expressed against a
signal scale that does not collapse when a spectrum has no baseline — the band-region signal, or an
absolute level derived from the instrument model.

## What this changes

- **Q2 (`docs/knowledge/open_questions.md`) gets a quantified starting point.** The sim→real gap for
  Raman band shape is ≈ 12 cm⁻¹ of unmodelled intrinsic linewidth, plus a linewidth *distribution*
  that is degenerate. That is a concrete target: any correction claiming to close the sim→real gap
  can be scored against it.
- **The forward model needs an intrinsic-broadening term**, distinct from the instrument response
  and drawn from a distribution rather than a constant. It does not have one.
- **`reference_scale` needs a second mode** for additive terms, because relative-to-median is wrong
  for spectra whose median is a floor rather than a baseline.

## What this does not show

- Nothing about chemistry: band *positions* were never compared.
- Nothing about a trained model; none was involved.
- The corpora are not matched — minerals and solvents versus small organics, different spectral
  ranges. The linewidth finding is robust to that (it is a property of the simulation method, not of
  the molecules), but the dynamic-range and autocorrelation numbers are not cleanly separable from
  the composition difference.
- The *Sci Data* corpus records no resolution, so it could not be resolution-matched and contributed
  only to the parameter estimate.
