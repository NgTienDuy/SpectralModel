# P0 — Instrumentation and measurement science

Covers KEYWORD_MAP **D** and **E**. Owns: `usis/measurement/`, `usis/eval/`.

## What it is

The layer that answers the question section 1 of the Sample Report exists to answer: **is this signal
chemistry, or is it the instrument?** Most published spectral models skip it entirely, which is why
their numbers do not survive a change of instrument.

The forward measurement model is one operator:

```
observed = f(clean_spectrum, instrument_state, acquisition)
```

composed of, in roughly the order they apply:

1. **Resolution convolution** with the instrument response function (approximately Gaussian for a
   slit-limited disperser; the true shape is measurable from a calibration lamp).
2. **Axis miscalibration** — offset, gain, and low-order nonlinearity in the wavenumber axis.
3. **Intensity response curve** — the wavelength-dependent efficiency of grating, optics and
   detector. Uncorrected, it is a smooth multiplicative distortion that no amount of normalisation
   removes, because it differs per instrument.
4. **Baseline and fluorescence** — additive, broad, sample- and instrument-dependent.
5. **Noise** — Poisson (`σ = √N`, signal-dependent), readout (additive, constant), dark (rate × time).
6. **Spikes** — cosmic rays; sparse, narrow, single-frame.
7. **Saturation / clipping** — a hard nonlinearity that destroys quantitative information silently.

This single module powers augmentation, sim→real transfer, domain-shift stress tests and
simulation-based inference. Building it once and well is worth more than any model in the repo.

## Calibration

- **Wavenumber**: silicon's first-order optical phonon near **520.7 cm⁻¹** at room temperature (the
  literature range 520.5–520.7 reflects temperature, strain and doping — hence
  `SILICON_RAMAN_TOLERANCE_CM1`). Polystyrene is the ASTM-recognised multi-point standard
  (ASTM E1840); **our anchor list is empty and marked `UNVERIFIED` until the standard is read.**
- **Intensity**: NIST SRMs provide a certified relative-response curve. Without one, intensity
  comparisons across instruments are qualitative.
- **Traceability** matters here in the metrological sense: a number without a traceable calibration
  chain is not a measurement, it is a reading.

## Uncertainty, LOD and LOQ

- Random error is reducible by accumulation (`σ ∝ 1/√n`); systematic error is not. Averaging a
  miscalibrated axis gives a precise wrong answer.
- LOD and LOQ are properties of a **method on an instrument on a matrix**, never of a substance.
  Reporting a concentration without its validity domain is a refusal-contract violation.
- Uncertainty must propagate: `Spectrum.sigma` exists so that per-point uncertainty survives
  preprocessing rather than being discarded at the first smoothing step.

## Sampling

Nyquist applies to the *instrument-broadened* spectrum: a band of FWHM `w` needs a sample spacing
well below `w` (3–5 points across the FWHM in practice) to preserve position and area. Below that,
peak position estimates become grid-dependent — a resolution-shift failure (`F07`) that looks like a
chemical shift.

**Zero-filling and interpolation do not add information.** They make a grid denser and a model that
learns the grid learns the instrument (`F14`).

## Implications for our code

- `AxisSpec.resolution_fwhm` is optional in the type but required for cross-instrument work, and the
  absence is priced rather than fatal.
- `InstrumentSpec.instrument_id` and `lab_id` are stable strings because they are split keys, not
  documentation.
- Degradations must be applied **independently and at controllable severity**, because E0 probe T4
  sweeps each one separately. A forward model that only applies "realistic" combined degradation
  cannot produce a degradation curve.
- Saturation must be detectable after the fact: the audit reports clipped-sample counts, and a
  saturated spectrum cannot support a quantitative claim (`F05`).

## Gap statement

Treating instrument state as a **latent variable to infer** rather than a nuisance to normalise away
is `PROJECT_NOVELTY.md` N3, and the forward model above is exactly the generative component such an
inference needs. Whether joint inference beats normalisation is untested and is a clean two-arm
experiment on the Bioprocess Analytes collection, which the Q1 audit confirmed carries the same
analytes across eight instruments — restricted to quantification/classification, since no structure
library measured across instruments was found.
