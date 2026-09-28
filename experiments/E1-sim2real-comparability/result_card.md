# E1 — Are degraded simulated spectra statistically comparable to real ones?

> **This result is not fully attributable.**
>
> The working tree was `true` when it ran, so the code that produced it is not the code at that commit.

## What this shows

- 2 of 5 held-out statistics are comparable under the pre-registered rule (simulated median inside the real interquartile range).
- Parameters came from the real corpora and the held-out statistics were never used to set them, so the comparison is not circular.

## What this does NOT show

- Nothing about chemistry: band positions are not compared, only their count and width.
- Nothing about a trained model's transferability; no model is involved.
- Nothing about matched samples: the corpora differ in spectral range and in chemistry (minerals and solvents versus small organics), and that confound is not removed.
- Nothing about the Sci Data corpus's resolution, which is unrecorded, so it could not be resolution-matched and contributes only to the parameter estimate.

## Version lock

- git SHA: `4588cecb207ffbb507561e2d25004702b5fa600f` (dirty: `true`)
- run id: `E1-sim2real-comparability`, started `2026-09-03T08:16:21+00:00`
- seed: `0`, seeded: `numpy, python`
- config: `experiments/E1-sim2real-comparability/preregistration.md` hash `815584b3c1a4ce55`
- environment: machine x86_64, numpy 2.5.2, platform Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39, processor x86_64, python 3.12.3

## Datasets

- `rod`
- `qm9s-raman`
- `scidata-api-raman`

## Metrics

| metric | mean | interval | IQM | n |
| --- | --- | --- | --- | --- |
| noise_variance_slope (degraded simulation) | 0.0001973 | [0.0001453, 0.0002525] @ 95% | 0.000162 | 45 |
| n_bands (degraded simulation) | 5.133 | [4.556, 5.711] @ 95% | 5.31 | 45 |
| median_band_fwhm_cm1 (degraded simulation) | 11.12 | [11.01, 11.27] @ 95% | 11 | 45 |
| dynamic_range (degraded simulation) | 2394 | [1804, 3103] @ 95% | 1805 | 45 |
| autocorrelation_length_cm1 (degraded simulation) | 17.93 | [15.4, 21.18] @ 95% | 14.18 | 45 |

The mean and the IQM differ by more than 10% for noise_variance_slope (degraded simulation), dynamic_range (degraded simulation), autocorrelation_length_cm1 (degraded simulation). That is a finding, not a rounding artifact: the average is being carried by the tails.

## Failure codes

No failures were classified. An empty histogram usually means the
classifier was not run, not that nothing failed — check which.

## Notes

- noise_variance_slope: simulated 0.0001618 vs real 0.0001113 [3.479e-05, 0.0003798], gap +0.15 IQR — comparable
- n_bands: simulated 5 vs real 6 [4, 9], gap -0.20 IQR — comparable
- median_band_fwhm_cm1: simulated 11 vs real 24.59 [18.98, 29.89], gap -1.25 IQR — NOT comparable
- dynamic_range: simulated 1731 vs real 50.18 [28.3, 119.7], gap +18.38 IQR — NOT comparable
- autocorrelation_length_cm1: simulated 13 vs real 21.21 [17.72, 26.15], gap -0.97 IQR — NOT comparable

---

Written 2026-09-03T08:16:25+00:00 by `usis.eval.mrep`.
