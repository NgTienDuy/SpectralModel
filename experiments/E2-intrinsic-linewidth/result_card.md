# E2 — Does an intrinsic linewidth drawn from a distribution close the sim-to-real gap?

## What this shows

- Control arm A0B0 (E1's configuration on this corpus): 2 of 4 held-out statistics comparable, total |gap| 7.67 IQR.
- Best arm A0B1: 3 of 4 comparable, total |gap| 2.09 IQR.
- Parameters were estimated on the ROD fit half plus the whole Sci Data corpus, both disjoint from the ROD test half every comparison is made against.

## What this does NOT show

- The two manipulation checks (median_band_fwhm_cm1, band_fwhm_iqr_cm1) are FITTED quantities and are not evidence. They verify the operator does what it was set to do.
- Nothing about chemistry: band positions are not compared, only their count and width.
- Nothing about a trained model's transferability; no model is involved.
- Nothing about matched samples: real is minerals and solvents, simulated is small organics, and the spectral ranges differ. That confound is not removed.
- Nothing about whether the fitted intrinsic width generalises past ROD's instrument family: it is estimated from one corpus and applied to a different chemistry.
- Nothing that rests on QM9S's 10 cm^-1 being physical. It is a publisher's arbitrary choice, and the resolution-matching step treats it as an instrument response, which is an assumption inherited from E1 rather than tested here.
- The QM9S and Sci Data corpora are partial fetches taken at fixed byte offsets, not random samples; see FETCH.json in each dataset directory.

## Version lock

- git SHA: `9ba925aa92b51ed9d2f001419960088fcb4c4c41` (dirty: `false`)
- run id: `E2-intrinsic-linewidth`, started `2026-09-03T10:10:58+00:00`
- seed: `0`, seeded: `numpy, python`
- config: `experiments/E2-intrinsic-linewidth/preregistration.md` hash `846bbf9a90a504da`
- environment: machine x86_64, numpy 2.5.2, platform Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39, processor x86_64, python 3.12.3

## Datasets

- `rod`
- `qm9s-raman`
- `scidata-api-raman`

## Metrics

| metric | mean | interval | IQM | n |
| --- | --- | --- | --- | --- |
| A0B0 total |gap| over the held-out statistics (IQR units) | 1.917 | [0.5522, 4.3] @ 95% | 0.8313 | 4 |
| A1B0 total |gap| over the held-out statistics (IQR units) | 1.667 | [0.06276, 3.515] @ 95% | 1.012 | 4 |
| A0B1 total |gap| over the held-out statistics (IQR units) | 0.5225 | [0.182, 0.8972] @ 95% | 0.4882 | 4 |
| A1B1 total |gap| over the held-out statistics (IQR units) | 1.182 | [5.951e-05, 3.484] @ 95% | 0.04144 | 4 |

The mean and the IQM differ by more than 10% for A0B0 total |gap| over the held-out statistics (IQR units), A1B0 total |gap| over the held-out statistics (IQR units), A1B1 total |gap| over the held-out statistics (IQR units). That is a finding, not a rounding artifact: the average is being carried by the tails.

## Failure codes

No failures were classified. An empty histogram usually means the
classifier was not run, not that nothing failed — check which.

## Notes

- intrinsic width fitted: median 14.27 cm^-1, geometric sd 2.148, from 4269 bands over 562 ROD fit-half spectra
- A0B0 (broadening off, scale median): 2/4 comparable, total |gap| 7.67 IQR
-     TEST  noise_variance_slope: sim 0.0006817 vs real 0.000169 [3.77e-05, 0.001107], gap +0.48 IQR - comparable
-     TEST  n_bands: sim 7.5 vs real 5 [4, 8], gap +0.62 IQR - comparable
-     TEST  dynamic_range: sim 649.8 vs real 58.19 [25.75, 132.8], gap +5.52 IQR - NOT comparable
-     TEST  autocorrelation_length_cm1: sim 33 vs real 21.49 [17.36, 28.44], gap +1.04 IQR - NOT comparable
-     check median_band_fwhm_cm1: sim 11 vs real 23.14, gap -1.01 IQR (fitted, not evidence)
-     check band_fwhm_iqr_cm1: sim 3 vs real 9.16, gap -0.59 IQR (fitted, not evidence)
- A1B0 (broadening on, scale median): 1/4 comparable, total |gap| 6.67 IQR
-     TEST  noise_variance_slope: sim 3.478e-05 vs real 0.000169 [3.77e-05, 0.001107], gap -0.13 IQR - NOT comparable
-     TEST  n_bands: sim 5 vs real 5 [4, 8], gap +0.00 IQR - comparable
-     TEST  dynamic_range: sim 261.5 vs real 58.19 [25.75, 132.8], gap +1.90 IQR - NOT comparable
-     TEST  autocorrelation_length_cm1: sim 73 vs real 21.49 [17.36, 28.44], gap +4.64 IQR - NOT comparable
-     check median_band_fwhm_cm1: sim 23 vs real 23.14, gap -0.01 IQR (fitted, not evidence)
-     check band_fwhm_iqr_cm1: sim 11 vs real 9.16, gap +0.18 IQR (fitted, not evidence)
- A0B1 (broadening off, scale amplitude): 3/4 comparable, total |gap| 2.09 IQR
-     TEST  noise_variance_slope: sim 0.0006784 vs real 0.000169 [3.77e-05, 0.001107], gap +0.48 IQR - comparable
-     TEST  n_bands: sim 7 vs real 5 [4, 8], gap +0.50 IQR - comparable
-     TEST  dynamic_range: sim 66.32 vs real 58.19 [25.75, 132.8], gap +0.08 IQR - comparable
-     TEST  autocorrelation_length_cm1: sim 33 vs real 21.49 [17.36, 28.44], gap +1.04 IQR - NOT comparable
-     check median_band_fwhm_cm1: sim 11 vs real 23.14, gap -1.01 IQR (fitted, not evidence)
-     check band_fwhm_iqr_cm1: sim 3 vs real 9.16, gap -0.59 IQR (fitted, not evidence)
- A1B1 (broadening on, scale amplitude): 3/4 comparable, total |gap| 4.73 IQR
-     TEST  noise_variance_slope: sim 8.049e-05 vs real 0.000169 [3.77e-05, 0.001107], gap -0.08 IQR - comparable
-     TEST  n_bands: sim 5 vs real 5 [4, 8], gap +0.00 IQR - comparable
-     TEST  dynamic_range: sim 58.2 vs real 58.19 [25.75, 132.8], gap +0.00 IQR - comparable
-     TEST  autocorrelation_length_cm1: sim 73 vs real 21.49 [17.36, 28.44], gap +4.64 IQR - NOT comparable
-     check median_band_fwhm_cm1: sim 22.25 vs real 23.14, gap -0.07 IQR (fitted, not evidence)
-     check band_fwhm_iqr_cm1: sim 10.5 vs real 9.16, gap +0.13 IQR (fitted, not evidence)

---

Written 2026-09-03T10:11:58+00:00 by `usis.eval.mrep`.
