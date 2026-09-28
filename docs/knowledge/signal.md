# P0 — Spectral signal science and preprocessing

Covers KEYWORD_MAP **F**. Owns: `usis/preprocess/`.

## The governing fact

**No single preprocessing pipeline is universally optimal**, and every choice destroys some
information while revealing other. Preprocessing is therefore a *search space with provenance*, not a
fixed prelude — and how much a conclusion depends on that choice is itself part of the answer
(Sample Report section 2, `PROJECT_NOVELTY.md` N8).

## The operations, and how each fails

| Operation | Common methods | Characteristic failure |
| --- | --- | --- |
| Despiking | median/derivative thresholds, multi-frame comparison | removes genuine sharp bands; leaves spikes that models memorise (`F03`) |
| Baseline / fluorescence removal | polynomial, rubber-band, AsLS, airPLS | over-subtraction produces **negative peaks** — physically impossible, and the clearest `F02` signature |
| Smoothing | Savitzky–Golay | broadens and shifts narrow bands; window and polynomial order interact non-obviously |
| Derivatives | 1st, 2nd (usually via SG) | resolves overlap and removes additive baseline, at a large SNR cost; the sign convention flips band appearance |
| Scatter correction | SNV `(x − mean)/std`, MSC/EMSC against a reference | SNV is per-spectrum and so cannot be inverted to physical units; MSC depends on the reference chosen, which leaks dataset identity |
| Normalisation | max, area, vector, band-ratio | max-normalisation makes everything sensitive to the single noisiest point |
| Alignment | shift/warp to a reference | can absorb a genuine chemical shift into a calibration correction |
| Peak fitting | Lorentzian / Gaussian / **Voigt** | model-order ambiguity: more components always fit better |

**Line shapes.** Homogeneous (lifetime) broadening gives Lorentzian; inhomogeneous (environmental
distribution) and instrument response give Gaussian; their convolution is the **Voigt** profile,
which is what real bands mostly are. Fitting a Gaussian to a Voigt biases FWHM and area
systematically — which matters because area is what quantification uses.

**Noise.** Poisson (`σ = √N`) is signal-dependent, so SNR varies *across* the spectrum; a single
scalar SNR is a summary, not a description. Readout noise is additive and constant, dark noise scales
with exposure. Denoising that assumes homoscedastic Gaussian noise is misspecified for the
photon-limited regime, which is exactly the low-signal regime where denoising is wanted.

## Implications for our code

- Every step emits a `ProcessingStep` with name, params, version and timestamp, and the chain is
  append-only (`Spectrum.with_processing`). Provenance that can be rewritten is not provenance.
- Steps are composable and **comparable**: the pipeline is an object the harness can vary, not a
  function buried in a loader.
- Negative intensities after baseline correction are a detectable, reportable condition — not
  something to clip away silently.
- Preprocessing is never baked into a stored dataset. The raw spectrum plus the chain reproduces the
  processed one; if it does not, the chain is lying.

## Gap statement

Preprocessing comparison studies are common. What we have not found is a system that **propagates
pipeline choice into the reported uncertainty of a specific identity claim**, or that tests whether
pipeline-sensitivity predicts being wrong. If it does, it is a free abstention signal.
→ `PROJECT_NOVELTY.md` N8.
