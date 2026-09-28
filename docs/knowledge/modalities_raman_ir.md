# P0 — Raman and infrared in depth

Covers KEYWORD_MAP **C1** and **C2**. Owns: `usis/measurement/`, `usis/data/sources/`.

## Raman

Inelastic scattering: a tiny fraction of incident photons exchange energy with a vibration. The
observable is the **shift** from the excitation line, not an absolute photon energy — which is why
`AxisKind` separates `RAMAN_SHIFT_CM1` from `ABSOLUTE_WAVENUMBER_CM1` and why converting between
them without the excitation wavelength raises rather than guesses (`F01`).

**What varies between instruments, and therefore between datasets:**

- **Excitation wavelength.** Changes scattering strength (`ν̃⁴`), fluorescence background (worse at
  shorter λ), penetration depth, and whether resonance enhancement occurs at all.
- **Spectral resolution**, set by grating and slit. Convolution with the instrument response
  broadens and can merge bands. Without `resolution_fwhm`, cross-instrument work is guesswork.
- **Low-wavenumber cutoff**, set by the notch/edge filter — an *instrument* limit, not a physical
  one, and a strong instrument fingerprint.
- **Detector**: CCD vs InGaAs changes the accessible range and the noise character.

**Traps.**

- **SERS is a different modality.** Enhancement factors vary by orders of magnitude, depend on
  nanostructure and molecular orientation, and are not transferable. Pooling SERS with spontaneous
  Raman is a domain error; `Modality.RAMAN_SERS` is a separate member for this reason.
- **Fluorescence** is simultaneously a modality and the dominant Raman artifact. It is broad, intense,
  sample-dependent, and its removal is a *choice* with consequences (`F04`, and `F02` when the
  removal is over-aggressive).
- **Cosmic-ray spikes** are narrow, intense, and single-frame. They look like sharp peaks and leak
  into models trained on undespiked data (`F03`).
- **Confocal depth response** distorts relative intensities in layered or inhomogeneous samples.

## Infrared

Absorption of IR photons by dipole-active modes. The intensity convention is part of the type
because the conventions are not interconvertible without extra information:

- **Transmittance** `T ∈ (0,1]` — and percent transmittance is a different number entirely.
- **Absorbance** `A = −log₁₀T`, the convention in which Beer–Lambert is linear.
- **Reflectance** and **Kubelka–Munk** `F(R) = (1−R)²/2R = k/s`, the diffuse-reflectance transform
  that restores approximate proportionality to concentration.

**ATR is not transmission, and the correction is not optional.** In attenuated total reflectance the
evanescent field penetrates a depth

```
d_p = λ / (2π·n₁·√(sin²θ − (n₂/n₁)²))
```

which is **proportional to wavelength**. Longer wavelengths (lower wavenumbers) sample more material,
so an uncorrected ATR spectrum has systematically inflated low-wavenumber bands relative to a
transmission spectrum of the same substance. Comparing them without correction is `F01`/`F06`, and it
looks exactly like chemistry.

**Other traps.**

- Atmospheric **water vapour** (sharp rotational structure ~1400–1900 and 3500–4000 cm⁻¹) and **CO₂**
  (~2350 cm⁻¹) are instrument artifacts that read as sample bands.
- Apodisation and zero-filling in FTIR change apparent line shape and apparent resolution. Zero-filling
  interpolates; it does not add information. A model that learns the sampling grid learns the
  instrument (`F07`, `F14`).

## Implications for our code

- `Modality` distinguishes `IR_ATR`, `IR_TRANSMISSION`, `IR_DRIFT` because they are not comparable
  without a stated correction. The forward model owns those corrections.
- `convert_intensity` refuses `COUNTS → ABSORBANCE`: recovering absorbance from raw counts needs a
  reference measurement, not a formula. Refusing is the whole point.
- `AcquisitionParams.temperature_k` and `InstrumentSpec.excitation_nm` are optional but *priced*: the
  report must state which missing field cost how much posterior width.

## Gap statement

The instrument-dependent parts above (filter cutoff, resolution, response curve, ATR depth) are
exactly the features a model can exploit as a shortcut, and exactly the features an
instrument-disjoint split is meant to defeat. Whether such a split is constructible from open data is
open question Q1 — and it is the reason `datasets.md` records `resolution_fwhm` and `instrument_id`
availability per source rather than just spectrum counts.
