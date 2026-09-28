# Open questions

Answered with evidence, not opinion. Each carries the specific experiment or lookup that would
resolve it, and each is updated as evidence arrives — including when the evidence is disappointing.

**Last updated:** 2026-09-02.

---

## The top five, ranked by how much they change what we build

### Q1 — Can an instrument-disjoint split over the same molecules be constructed from open data?

**ANSWERED 2026-09-03: (b) yes, but only in a narrow domain — and the field has not used it.**
Full audit table in `docs/knowledge/datasets.md` §5.

**What was found.**

1. **The data exists, once.** RamanBench's **Bioprocess Analytes** collection provides
   *"the same analytes measured across eight different instruments"* (`VERIFIED`, paper full text).
   That is exactly the pairing an `instrument_disjoint` split needs.
2. **It is narrow.** The collection is bioprocess quantification/classification, not a structure
   library. An instrument-disjoint **`spectrum_to_structure`** evaluation is still not constructible
   from anything we found.
3. **The evaluation has not been done.** RamanBench's own protocol is *"fixed 80/20 train/test splits
   over 3 different seeds"*, stratified for classification. The paper notes the collection "offers an
   opportunity for transfer learning" and does not take it.
4. **RRUFF and ROD give something adjacent but weaker**: the same sample at different excitation
   wavelengths on the same instrument. That supports an *excitation-disjoint* split — a real probe
   of wavelength-dependent effects (fluorescence, the ν⁴ factor, resonance) — but it is not
   instrument fingerprinting, and it must not be reported as such. **ROD was audited directly on
   2026-09-03**: 240 records sampled, 238 on one instrument and 2 on a second, but 58 specimens
   measured at both 532 and 780 nm. See `docs/knowledge/datasets.md` §5.
5. **`instrument_id` is not a field.** The `raman-data` access package (MIT, 93 datasets) exposes
   spectra, `raman_shifts`, targets and dataset-level metadata; the instrument appears in per-dataset
   free text. Populating the split key is a per-dataset extraction job at M1.

**Consequence — fallback (a) from `docs/adr/0001-brief-critique.md` §S6 is selected**, and it is
stronger than S6 anticipated. M2's headline criterion is answerable in a restricted form:

> *How much of the deep-learning advantage over chemometrics survives a leave-one-instrument-out
> split on a collection where the same analytes were measured on eight instruments?*

That is a real, publishable answer to the brief's open question 8, with an explicitly stated domain
limit. What it cannot answer is the same question for spectrum→structure, which stays open and is
now a data-collection problem rather than a method problem.

**Closed since**: ROD and the *Sci Data* API set were both audited directly (2026-09-03) and
neither supports an instrument-disjoint split — ROD is 238/240 one instrument, the API set is one
instrument outright. Details in `docs/knowledge/datasets.md` §5. That leaves RamanBench's
Bioprocess Analytes as the *only* candidate we have found.

**Now closed** (2026-09-03, via the `raman-data` catalogue): the collection is **eight sibling
datasets**, one per spectrometer — Anton Paar 532, Anton Paar 785, Kaiser, Metrohm, Mettler Toledo,
tec5, Timegate, Tornado — measuring mixtures of **three analytes** (glucose, sodium acetate,
magnesium sulfate) with concentrations verified by enzymatic assay. **All eight are CC BY 4.0**, so
the experiment is both runnable and the data vendorable. Full table in `docs/knowledge/datasets.md`
§5.

**One caveat that must travel with any result from it**: the catalogue states the eight instruments
measure *"nearly the same samples"*, with *"some datasets have a bit more samples than others"*. A
leave-one-instrument-out split is therefore not a perfectly matched design, and part of any
cross-instrument drop is sample composition rather than instrument response. Quantifying that
confound belongs inside the experiment.

**Still unresolved**: `lab_disjoint` (the catalogue records no lab), and an instrument-disjoint
split for `spectrum_to_structure` — the eight-instrument collection is quantitative regression on
three analytes, and nothing in the 93 datasets pairs a *structure library* across instruments.

---

### Q2 — How large is the simulated → experimental gap, and what closes it most per unit of compute?

**Now first among the open questions**, Q1 having been answered — and it now has a quantified
starting point rather than an intuition.

**Measured 2026-09-03** (experiment E1, 240 ROD vs 45 QM9S Raman spectra; full write-up in
`experiments/E1-sim2real-comparability/findings.md`):

- **≈12.5 cm⁻¹ of intrinsic band width is simply absent from the simulation.** Real ROD bands are
  16.5 cm⁻¹ [12.1, 23.1] against a declared 4 cm⁻¹ instrument; QM9S bands are 11.00 cm⁻¹ against
  10 cm⁻¹ of applied broadening. Harmonic DFT has no phonon lifetime, no disorder, no inhomogeneous
  broadening, so its width *is* the broadening constant.
- **The simulated linewidth distribution is degenerate**: interquartile range `[11.00, 11.00]`.
  Every band of every molecule has the same width. No real spectrum does, which makes it a shortcut
  a model can exploit (`F14`) without learning any chemistry.
- Under the pre-registered comparability rule, 2 of 5 held-out statistics matched
  (`noise_variance_slope`, `n_bands`); band width, autocorrelation length and dynamic range did not.
- **Replicated 2026-09-03** by E2's control arm on a corpus 4.7x larger and sampled differently
  (the whole ROD index rather than three guessed id blocks): same pattern, same two statistics.

**That candidate has now been built and tested, and it did not work.** Experiment E2
(`experiments/E2-intrinsic-linewidth/findings.md`, 2026-09-03, 1,129 ROD vs 226 QM9S) sampled the
width per band from a lognormal fitted on a disjoint half of the real corpus. The width itself is
now right — median gap −1.01 → **−0.01 IQR**, spread gap −0.59 → **+0.18 IQR**, so the degenerate
`[11.00, 11.00]` distribution is gone. But the held-out statistics got **worse**, and one of them
says why:

- **`autocorrelation_length_cm1` 33 → 73 cm⁻¹ against 21 cm⁻¹ real** (+1.04 → +4.64 IQR). At the
  same median band width, the simulated spectra decorrelate three times more slowly than the real
  ones.

**So the question sharpens: the gap is not the width, it is the wings.** A Lorentzian still carries
~10% of its peak three half-widths out, and summed over every band that is a broad correlated
pedestal real spectra do not have. Lighter wings at the same FWHM means **inhomogeneous broadening
dominates over lifetime broadening** in these samples — a Gaussian contribution, not a Lorentzian
one. The next candidate is a **Voigt intrinsic profile with a fitted mixing ratio**, scored on
`autocorrelation_length_cm1`, and it is still cheap. Anharmonic corrections and conformer ensembles
remain untried and remain the expensive answers.

**The other half of E1's diagnosis was ours and is now closed.** The `dynamic_range` gap fell from
+18.4 IQR (E1) to **+0.00 IQR** once additive amplitudes were expressed against the peak-to-floor
signal rather than the median, which on a simulated spectrum is the floor between bands. That was a
defect in our parameterisation, not in the physics.

**Still to measure**: how much of the *remaining* gap is harmonic approximation vs solvent vs
conformer ensemble. The 78-compound RRUFF ∩ computed-Raman overlap remains the cleanest asset for
that, and is untouched.

**Direction is forced by physics.** QM9S carries 10 cm⁻¹ of broadening and ROD reports 4 cm⁻¹, and
convolution cannot sharpen — so a comparison must broaden the *experiment* up to the simulation, not
the reverse. Any Q2 result must state which direction it used.

### Q3 — Which spectral representation wins, and does the answer depend on the task?

**Why third.** M3 exists to answer it, and the answer determines the architecture of everything after.

**What resolves it.** The M3 benchmark: raw / patches / peak tokens / wavelet / learned VQ / hybrid,
one protocol, scored on information loss, robustness to each M1 degradation, retrieval quality,
mixture suitability, interpretability and cost.

**Prior evidence that should make us cautious.** RamanBench reports that *tabular* foundation models
with no spectral inductive bias beat domain-specific architectures. If that holds, the honest reading
is that the tasks are dominated by dataset-level structure rather than spectroscopy — which would
make the representation question less important than the *task* question, and would be a finding
worth more than the benchmark itself.

---

### Q4 — Does `p_none` mean anything, or is it a renamed softmax leftover?

**Why fourth.** The open-set residual is the brief's central epistemic commitment. If it cannot be
calibrated, the refusal contract is decoration and the abstention story collapses.

**What resolves it.** An `open_set` split with a chemically disjoint held-out fraction of the
library; measure `p_none` AUROC and calibration on molecules that are genuinely absent, first
in-distribution, then under instrument shift. Until that exists, `p_none` is reported as
`UNCALIBRATED`.

---

### Q5 — Is 3D molecular geometry necessary, or do 2D graphs suffice for spectrum ↔ structure?

**Why fifth.** It decides whether conformer generation — expensive, and a large dependency surface —
is on the critical path.

**What resolves it.** The prior art already offers a natural experiment: Vib2Mol (1D SMILES),
VibraCLIP (2D graph), SMEN and Vib2Conf (3D conformer) are reported on the same VB-Mols benchmark
with Recall@1 89.01 / 91.38 / 94.02 / 97.23 (`PARTIAL` — from the paper claiming the win). The
ordering is suggestive but confounded by encoder capacity and by a single split kind. Resolving it
properly means re-running the family under our split matrix, with capacity controlled — which is a
well-defined M5 experiment and publishable if the ordering inverts.

---

## The brief's own eight, with status

| # | Question | Status | Where it is answered |
| --- | --- | --- | --- |
| 1 | Which representation wins, task-dependent? | open — Q3 | M3 |
| 2 | Is 3D geometry necessary? | open — Q5 | M5 |
| 3 | Size of the sim→real gap and cheapest correction | open — Q2 | M1–M5 |
| 4 | Does any physics-informed constraint improve *generalisation*? | open | M4 (E0b), gated per KEYWORD_MAP §M |
| 5 | Is SBI a better formalisation than contrastive retrieval + reranker? | open | M5, N2 |
| 6 | Where does the data-scale curve flatten? | open | M3+ |
| 7 | Does active measurement beat a fixed protocol at equal budget? | open | M8, N4 |
| 8 | How much published DL advantage survives instrument-disjoint evaluation? | **unblocked** — answerable on the Bioprocess Analytes collection, for classification/regression only | M2 |

## Questions that have been answered

### Can an instrument-disjoint split over the same molecules be built from open data?

**Answered 2026-09-03: partly — see Q1 above.** One collection supports it, in a narrow domain, and
the benchmark that owns it evaluates with random splits instead. The gap is protocol choice, not data
absence, which revises `PROJECT_NOVELTY.md` N9.

### Does a cross-instrument, physics-probing spectral benchmark already exist?

**Answered 2026-09-02: partially, and more than the brief assumed.** RamanBench (arXiv:2605.02003),
SpectrumBench (arXiv:2508.01188) and arXiv:2601.16107 all exist and are verified. None is reported to
probe physical law, score abstention, or split by instrument over the same molecules. Consequence:
novelty candidate N1 narrowed; see `docs/adr/0001-brief-critique.md` §S7.
