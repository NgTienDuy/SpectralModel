# Prior art

**Reviewed:** 2026-09-02. Recording format follows brief §10.2: input, output, dataset,
architecture, objective, evaluation protocol, limitations, reproducibility, code. Every entry ends
with a **gap statement** — what it does *not* answer, and whether we can address that.

Verification statuses are as in `docs/knowledge/datasets.md`. Where a paper's abstract page was
fetched, that is `VERIFIED`; where facts come from search summaries, `PARTIAL`. **Nothing below is
a remembered benchmark number.**

---

## 1. Vib2Mol — the closest prior art to our MVP

`VERIFIED` · arXiv:2503.07014 · "Vib2Mol: from vibrational spectra to molecular structures — a
unified deep learning framework" · Lu, Ma, Li, Li, Rong, Li, Zhu, Liu, Ren · submitted 2025-03-10,
v4 2026-01-20 · code: `github.com/X1nyuLu/vib2mol` (`UNVERIFIED`).

- **Input / output**: IR and Raman spectra → molecular structure, by both retrieval and generation.
- **Architecture / objective**: contrastive spectrum↔structure alignment over **1D SMILES**
  molecular representation, with *coarse-to-fine retrieval* and *generate-then-rerank*.
- **Also claims**: reaction-product prediction and peptide sequencing from vibrational spectra.
- **Evaluation**: theoretical *and* experimental IR/Raman. Exact datasets, splits and metric
  definitions were not obtainable from the abstract page; **must be read in full before we
  reproduce it** (M5 acceptance requires reproducing a published baseline within a stated
  tolerance).

**Gap statement.** Retrieval and generation both presuppose that the answer is *in* the candidate
space; neither emits an open-set residual, a calibrated posterior, or an abstention. Nothing in the
framing treats the instrument as a variable. Addressable by us: yes — this is the direct target of
our M5/M6 comparison.

---

## 2. The contrastive spectrum↔structure family: VibraCLIP, SMEN, Vib2Conf

`PARTIAL` (Vib2Conf abstract fetched via search: arXiv:2604.24310).

| System | Molecular representation | Reported role |
| --- | --- | --- |
| Vib2Mol | 1D SMILES | baseline |
| VibraCLIP | 2D graph + IR + Raman into one latent space | baseline |
| SMEN (Spectra and Molecule Encoder Network) | 3D conformation; scores molecules against an IR spectrum | baseline |
| Vib2Conf | 3D conformation; spectral encoder + molecular encoder, bidirectional retrieval | reports state of the art |

Reported on the **VB-Mols** benchmark, IR input, Recall@1: Vib2Conf 97.23%, SMEN 94.02%,
Vib2Mol 91.38%, VibraCLIP 89.01%. `PARTIAL` — these numbers come from a search summary of the
Vib2Conf paper, i.e. from the paper claiming the win. **Do not quote them as established until the
paper and the benchmark definition are read.**

**Why this family matters to us specifically.** It is a natural experiment on the brief's own open
question 2 (*is 3D geometry necessary?*): the ranking 1D < 2D < 3D on one benchmark is suggestive,
but a single benchmark with a single split kind cannot separate "3D carries vibrational
information" from "3D encoders have more capacity" or "the benchmark's split rewards conformer
memorisation".

**Gap statement.** Recall@1 above 97% on a retrieval benchmark says the candidate set is closed and
probably easy. What none of them reports: performance under instrument-disjoint evaluation, an
open-set residual, degeneracy behaviour on isomer pairs, or calibrated uncertainty. Addressable by
us: yes, and cheaply — re-running this family under our split matrix is a well-defined M5
experiment and is a publishable result on its own if the ranking inverts.

---

## 3. DSCF — the strongest existing claim to a spectral foundation model

`VERIFIED` (search + publisher listing) · "Deep spectral component filtering as a foundation model
for spectral analysis demonstrated in metabolic profiling" · *Nature Machine Intelligence* 7:743–757
(2025) · DOI `10.1038/s42256-025-01027-5`.

- **Objective**: self-supervised *spectral component resolvable learning* over **>1M simulated and
  experimental spectra** spanning UV, IR and Raman.
- **Architecture**: signal transformer with layer normalisation and shifted-window multi-head
  self-attention.
- **Claims**: state of the art on 5 spectral analysis tasks over 11 datasets; zero-shot spectral
  denoising; trace-level quantification in mixtures; metabolic biomarker identification (stroke,
  Alzheimer's, prostate cancer).

**Is it a foundation model or a large multi-task model?** The brief asks this and it is the right
question. On the evidence available from the abstract, the distinguishing test is not the training
scale but whether representations transfer to a task family absent from pretraining *and* to an
instrument absent from pretraining. Neither is claimed. **Our assessment, pending full text:
"large, well-trained multi-task spectral model" is the defensible description; "foundation model"
is not yet demonstrated by the criterion that would matter to us.** This must be revised if the
full text reports held-out-modality or held-out-instrument transfer.

**Gap statement.** Component filtering is a *preprocessing/unmixing* capability, not structure
inference; the claims are about extracting a component's signal, not about identifying an unknown.
The biomarker claims are correlational and, per our invariant 8, would require a `ClaimType` label.
Addressable by us: the honest-transfer evaluation, yes. Competing on pretraining scale, no.

---

## 4. Benchmarks and evaluation infrastructure

### RamanBench — `VERIFIED` (arXiv:2605.02003, submitted 2026-05-03)

"RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy" ·
Koddenbrock, Lange, Legner, Jäger, Kögler, Cruz Bournazou, Neubauer, Biessmann, Rodner.

- **74 datasets** unified across four domains, **16 released for the first time**, **325,668
  spectra**, classification and regression, unified data access + evaluation protocol + a live
  leaderboard.
- **28 models** in four families: classical (PLS), Raman-specific (RamanNet), tabular foundation
  models (TabPFN), time-series (ROCKET).
- **Headline finding**: tabular foundation models consistently beat domain-specific and
  gradient-boosting baselines; time-series models stay competitive; **no method generalises across
  all datasets.**

That last finding is the single most useful external datapoint we have, and it partially
pre-empts our M2 kill criterion. It also raises a question the paper's framing does not:
if a *tabular* foundation model with no spectral inductive bias wins, the tasks may be dominated by
dataset-level shortcuts rather than spectroscopy.

### Benchmarking Deep Learning Models for Raman Spectroscopy — `VERIFIED` (arXiv:2601.16107)

Sineesh & Kamsali, submitted 2026-01-22, revised 2026-07-28, published in *Digital Discovery*
(CC BY 4.0). Three open Raman datasets chosen "to support standard evaluation, fine-tuning, and
explicit distribution-shift testing"; five deep architectures plus two conventional ML methods under
a unified protocol. **No cross-instrument or instrument-disjoint experiment is described in the
abstract** — needs full-text confirmation.

### SpectrumWorld / SpectrumLab / SpectrumBench — `PARTIAL` (arXiv:2508.01188)

A Python library, a benchmark-generating annotator, and **SpectrumBench**: 14 spectroscopic tasks,
>10 spectrum types, curated from >1.2M chemical substances; 18 multimodal LLMs evaluated, reported
as revealing critical limitations.

**Gap statement for the whole group.** Three independent benchmarking efforts appeared within
roughly seven months, which means *benchmark existence* is no longer a gap — the brief's novelty
hypothesis 1 must be narrowed accordingly (see `PROJECT_NOVELTY.md` N1 and
`docs/adr/0001-brief-critique.md` §S13). What none of them is reported to do: **probe physical law**
(isotope shift, mutual exclusion), **construct instrument- or lab-disjoint splits over the same
molecules**, or **score abstention and open-set behaviour**. That intersection is where our
contribution has to live.

---

## 5. Simulation and computed corpora

- **DetaNet / QM9S** — `VERIFIED`. Geometric deep learning model for IR/Raman/UV-Vis; QM9S is
  133,885 molecules at B3LYP/def-TZVP with TD-DFT (*Nat. Comput. Sci.*, DOI
  `10.1038/s43588-023-00550-y`). This is the practical substitute for running our own DFT.
- **QMe14S** — `PARTIAL`, arXiv:2501.18876. 186,102 molecules, 14 elements, 47 functional groups,
  B3LYP/TZVP, IR + Raman + NMR plus tensorial properties.
- **Computed Raman of inorganic compounds with hybrid functionals** — `PARTIAL`, *Sci Data* 11:105
  (2024). 161 compounds, **78 of them also in RRUFF**. Small, but a paired computed/experimental set
  is exactly what a sim2real gap measurement needs, and it is the only such pairing we have found.
- **MLIP-based IR prediction (MACE / NequIP / Allegro families)** — `UNVERIFIED`. Not checked this
  session; only relevant from M5.

**Gap statement.** Computed corpora are abundant and permissively licensed; paired
computed↔experimental sets of the *same* material are rare. The sim→real gap is therefore
under-measured not because it is uninteresting but because the pairing is scarce. Addressable by us:
yes, via the 78-compound RRUFF overlap plus our own forward model — and that is a small, sharp,
feasible experiment rather than a grand one.

---

## 6. The world-model frame

`VERIFIED` · "Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond" ·
arXiv:2604.22748 · submitted 2026-04-24, final 2026-06-16 · ~50 authors. Defines the capability
ladder the brief adopts: **L1 Predictor** (one-step local transition operators) → **L2 Simulator**
(multi-step, action-conditioned rollouts) → **L3 Evolver** (autonomously revises its own model when
predictions fail). It discusses decision-centric evaluation principles and a minimal reproducible
evaluation package.

**Correction to the brief.** The abstract does **not** name *Resolution Success Rate* or
*Counterfactual Outcome Deviation*. The brief presents RSR and COD as "adapted from the world-model
literature". Until the full text is read, treat those two metric names as `UNVERIFIED` attributions.
The *concepts* — did the episode resolve, and does the plan actually depend on the model — are sound
and we can define them ourselves; we just must not cite someone else for them yet.

---

## 7. Tooling

- **ramanchada2** — `VERIFIED`, MIT, `h2020charisma/ramanchada2`, CHARISMA H2020. Vendor-format I/O,
  processing, and **instrument calibration**. Written up in *J. Raman Spectrosc.* (`10.1002/jrs.6789`).
  **Depend on it; do not reimplement calibration.**
- **RamanSPy** — `PARTIAL`, *Anal. Chem.* `10.1021/acs.analchem.4c00383`. Dataset loaders and an
  analysis pipeline. Evaluate as a loader dependency at M1.
- **RAMANMETRIX** — `PARTIAL`, `CC-BY-NC-ND-4.0`. The `ND` clause forbids derivatives: **study, do
  not build on.**

---

## 8. So what is actually left unsolved?

Stated as the gap our thesis addresses, after the review above:

1. **No benchmark probes physical law.** Isotope-shift consistency and the mutual-exclusion rule are
   cheap, unambiguous, and absent from every benchmark found. A model can top RamanBench without
   knowing that deuteration lowers a frequency.
2. **No system reports an open-set residual with measured coverage.** Recall@1 on a closed candidate
   set is the field's default metric; "the molecule is not in the library" is not scored anywhere we
   found.
3. **Instrument-disjoint evaluation is claimed rarely and constructed almost never**, because the
   data for it (the same molecules on named different instruments) is scarce — which is itself an
   unreported finding about the field's evidence base.
4. **The instrument is universally treated as noise to normalise away**, never as a latent variable
   to infer and re-measure against.
5. **Nobody reports whether a failure was method-limited or measurement-limited.** Without that,
   every accuracy number in this literature is uninterpretable — including the ones above.
6. **Nothing plans the next measurement.** All prior art is a single-shot map from a fixed spectrum
   to an answer. The decision-theoretic layer is empty.

Items 1, 2, 4, 5 and 6 are open. Item 3 is open but may be *data*-blocked rather than method-blocked,
which is itself the finding.

---

## Sources

- [Vib2Mol, arXiv:2503.07014](https://arxiv.org/abs/2503.07014)
- [Vib2Conf, arXiv:2604.24310](https://arxiv.org/abs/2604.24310)
- [DSCF, *Nature Machine Intelligence* 7:743–757 (2025)](https://www.nature.com/articles/s42256-025-01027-5)
- [RamanBench, arXiv:2605.02003](https://arxiv.org/abs/2605.02003)
- [Benchmarking DL Models for Raman Spectroscopy, arXiv:2601.16107](https://arxiv.org/abs/2601.16107)
- [SpectrumWorld, arXiv:2508.01188](https://arxiv.org/abs/2508.01188)
- [DetaNet / QM9S, *Nat. Comput. Sci.*](https://www.nature.com/articles/s43588-023-00550-y) · [QM9S on figshare](https://figshare.com/articles/dataset/QM9S_dataset/24235333)
- [QMe14S, arXiv:2501.18876](https://arxiv.org/abs/2501.18876)
- [Computed Raman of inorganic compounds, *Sci Data* 11:105 (2024)](https://www.nature.com/articles/s41597-024-02924-x)
- [Agentic World Modeling, arXiv:2604.22748](https://arxiv.org/abs/2604.22748)
- [ramanchada2](https://github.com/h2020charisma/ramanchada2) · [*J. Raman Spectrosc.* 10.1002/jrs.6789](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/jrs.6789)
- [RamanSPy](https://ramanspy.readthedocs.io/) · [RAMANMETRIX](https://docs.ramanmetrix.eu)
