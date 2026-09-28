# USIS — Domain Knowledge Map

> **Place at `docs/knowledge/KEYWORD_MAP.md`.** This is reference material, loaded on demand — not
> part of the always-on context. `docs/PROJECT_BRIEF.md` §5 is the executive summary; this file is
> the complete territory.
>
> **Priority tags.** `P0` = blocking, must be understood before the corresponding module is written.
> `P1` = needed within the first year. `P2` = needed for the agentic/knowledge layers. `P3` = parked
> on the speculative shelf; recorded so it is not lost, not built.
>
> **How to use this.** When you open a module, read its cluster here first and write
> `docs/knowledge/<cluster>.md`: what it is, which equations matter, which assumptions break, what
> it implies for our code. Do not paste keywords into a note and call it knowledge. Each note ends
> with a **gap statement** feeding `PROJECT_NOVELTY.md`.

---

## A. Physics foundations — `P0`
**Module**: `usis/measurement/`, constraint checkers in `usis/knowledge/rules/`

Electromagnetic radiation, photon, light–matter interaction, absorption, emission, spontaneous and
stimulated emission, elastic scattering, inelastic scattering, Rayleigh scattering, coherence
(spatial, temporal), polarization, dispersion, diffraction, interference, refraction, reflection,
refractive index, complex refractive index, dielectric function, susceptibility, optical
conductivity, optical response, linear optics, nonlinear optics, quantum optics, laser physics,
photon statistics, transition probability, selection rules.

**Load-bearing for us**
- Selection rules become an inference-time validator, not a learned prior.
- Kramers–Kronig relations link real and imaginary optical constants — a hard consistency law.
- The distinction elastic vs inelastic scattering is the whole reason Raman needs notch/edge
  filtering, and why the low-wavenumber region is instrument-limited rather than physics-limited.

**Research questions**: Which of these laws are cheaply checkable on a *predicted structure*?
Which can be made differentiable and used as a training constraint with measured benefit?

---

## B. Molecular & quantum physics — `P0`
**Module**: `usis/simulate/`, `usis/knowledge/rules/`

Molecular energy levels, electronic states, vibrational states, rotational states, normal modes,
harmonic oscillator, anharmonic oscillator, harmonic approximation, anharmonicity, potential energy
surface, force constant, Hessian matrix, normal mode analysis, reduced mass, vibrational coupling,
mode coupling, Fermi resonance, isotope effects, molecular symmetry, point groups, group theory,
symmetry selection rules, Born–Oppenheimer approximation, molecular Hamiltonian, wavefunction,
quantum state, density matrix, perturbation theory, time-dependent perturbation theory, transition
dipole moment, transition polarizability, polarizability tensor, Raman tensor, Raman activity, IR
intensity, dipole moment derivative, molecular orbital, electronic structure.

**Load-bearing for us**
- `ν ∝ √(k/μ)` gives the **isotope-shift falsification probe** (E0/T2) — the sharpest cheap test of
  whether a model learned vibrational structure or dataset statistics.
- Point group → **mutual exclusion rule** (centrosymmetric molecules: Raman-active ⇒ IR-inactive and
  vice versa). Checkable from any predicted structure. This is E0/T3.
- The harmonic approximation is the single largest source of simulation→experiment bias. Scaling
  factors, VPT2, and MD-based approaches are the ladder out.

**Research questions**: How much of the sim→real gap is harmonic approximation vs solvent vs
conformer ensemble vs instrument response? Nobody has cleanly decomposed this for Raman at scale.

---

## C. Modalities

### C1. Raman — `P0` (primary)
Raman scattering, Stokes, anti-Stokes, Raman shift, Raman spectrum/peak/band/intensity, Raman
cross-section, Raman polarizability, depolarization ratio, polarization Raman, resonance Raman,
SERS, TERS, CARS, SRS, coherent Raman, stimulated Raman scattering, SORS, Raman microscopy, confocal
Raman, Raman mapping, Raman imaging, Raman tomography, time-resolved Raman, temperature-dependent
Raman, low-frequency Raman, phonon, Raman of solids/liquids/gases/biological samples.

**Traps**: SERS enhancement factors vary by orders of magnitude and are not transferable — treat
SERS as a distinct sub-modality with its own domain, never pooled naively with spontaneous Raman.
Confocal depth response distorts intensity in layered samples. The anti-Stokes/Stokes ratio gives
sample temperature via Boltzmann statistics — a free, verifiable physical readout.

### C2. Infrared — `P0` (primary)
FTIR, ATR-FTIR, transmission IR, reflection IR, diffuse reflectance, DRIFTS, MIR, far-IR,
vibrational absorption, dipole moment derivative, overtone, combination band, hydrogen bonding,
anharmonicity, interferometer, Michelson interferometer, apodization, zero-filling.

**Traps**: **ATR penetration depth is wavelength-dependent** — an uncorrected ATR spectrum is not
comparable to a transmission spectrum, and the correction is not optional. Water vapour and CO₂
bands are instrument artifacts that look like chemistry.

### C3. NIR — `P1`
Overtones, combination bands, multivariate calibration dependence, high scattering sensitivity,
SNV/MSC/EMSC as near-mandatory preprocessing, calibration transfer as a mature sub-field.

### C4. UV-Vis / Fluorescence / Photoluminescence — `P1`
Electronic transition, absorption spectrum, Beer–Lambert law, chromophore, oscillator strength,
exciton, band gap, charge-transfer transition, fluorescence, phosphorescence, quantum yield, Stokes
shift, inner filter effect, quenching, excitation–emission matrix.

**Note**: fluorescence is simultaneously a *modality* and the dominant *artifact* in Raman. Both
roles must be modelled.

### C5. THz / Hyperspectral / Imaging — `P2`
Terahertz spectroscopy, low-frequency collective modes, hyperspectral imaging, spectral cube,
chemical imaging, spatial×spectral dimensionality.

### C6. NMR — `P2`
Chemical shift, spin, Zeeman splitting, spin–spin coupling, J coupling, relaxation, T1, T2, 1D/2D
NMR, COSY, HSQC, HMBC, NOESY, nmrML.

### C7. Mass spectrometry — `P2`, and architecturally separate
m/z, ionization, ESI, MALDI, fragmentation, CID, isotope pattern, adduct, precursor/product ion,
MS/MS, high-resolution MS, mzML.

**Design decision**: MS is **not** "another spectroscopy". It is a distinct analytical modality with
a different physical observable and a far more mature structure-elucidation literature. Study that
literature for transferable lessons (candidate ranking, retrieval + rerank, open-set behaviour,
benchmark design) — then keep it behind its own adapter.

**Modality taxonomy for the architecture** — do not flatten these into one pipeline:

```
Vibrational (Raman, IR, NIR) │ Electronic (UV-Vis, PL, fluorescence) │ Magnetic resonance (NMR)
                             └──────────── shared scientific representation ────────────┘
                                                     ↑
                              MS / chromatography / imaging / other sensors
```

---

## D. Instrumentation — `P0`
**Module**: `usis/measurement/`

Spectrometer, Raman spectrometer, FTIR spectrometer, spectrograph, monochromator, interferometer,
laser, excitation wavelength, laser power, optical filter, notch filter, edge filter, bandpass
filter, polarizer, beam splitter, objective lens, numerical aperture, optical fiber, detector, CCD,
EMCCD, CMOS, InGaAs, photodiode, APD, PMT, single-photon detector, detector cooling, dark current,
readout noise, shot noise, optical alignment, acquisition parameters, exposure time, integration
time, accumulation, binning, sampling, spatial resolution, spectral resolution.

**Implication**: `InstrumentSpec` is a first-class type and `instrument_id` drives the
instrument-disjoint split. Instrument state is a *modelled latent variable*, not noise to normalise
away (novelty hypothesis 3 in the brief).

---

## E. Measurement science — `P0`
**Module**: `usis/measurement/`, `usis/eval/`

Measurement theory, forward measurement model, inverse measurement model, instrument response
function, detector response, calibration, wavelength/wavenumber calibration, intensity calibration,
frequency calibration, calibration curve, reference standard, standard reference material,
traceability, measurement uncertainty, uncertainty propagation, systematic error, random error,
repeatability, reproducibility, accuracy, precision, sensitivity, specificity, dynamic range,
detection limit (LOD), quantification limit (LOQ), spectral bandwidth, sampling theory, Nyquist
sampling, aliasing, digitization, quantization noise, instrument drift, spectral drift.

**Concrete anchors**: silicon 520.7 cm⁻¹ and polystyrene for Raman calibration; NIST SRMs for
intensity response.

**This layer is what lets the system answer**: *is this signal chemistry, or is it the instrument?*
That question is section 1 of the Sample Report and it is what most published models skip entirely.

---

## F. Spectral signal science — `P0`
**Module**: `usis/preprocess/`

Signal-to-noise ratio, Poisson/shot noise, Gaussian noise, readout noise, dark noise, fluorescence
background, cosmic ray spike, saturation, clipping, baseline, baseline drift, background, spectral
drift, wavelength drift, detector artifact, peak broadening, line shape, Lorentzian, Gaussian, Voigt
profile, FWHM, homogeneous and inhomogeneous broadening, spectral overlap, peak overlap,
convolution, instrument response function, deconvolution.

Processing: digital signal processing, spectral preprocessing, denoising, smoothing, filtering,
Savitzky–Golay, baseline correction (polynomial, rubber-band, asymmetric least squares, airPLS),
background subtraction, fluorescence correction, normalization, standardization, scaling, centering,
SNV, MSC, EMSC, derivative spectroscopy (first, second), Fourier transform, FFT, wavelet transform,
wavelet denoising, interpolation, resampling, spectral alignment, peak detection, peak fitting,
spectral unmixing, signal separation, noise modeling, uncertainty propagation.

**Non-negotiable**: no single preprocessing pipeline is universally optimal. Preprocessing is a
**search space with provenance**, and the report states how sensitive the conclusions are to the
choice. This is section 2 of the Sample Report and is itself a novelty candidate.

---

## G. Chemometrics — `P0`
**Module**: `usis/chemometrics/`

Univariate: peak analysis, peak area, peak ratio, calibration curve, univariate quantification.

Multivariate: multivariate analysis/statistics/calibration, PCA, PLS, PLSR, PLS-DA, PCR, CLS, ECLS,
LDA, factor analysis, ICA, MCR, MCR-ALS, tMCR-ALS, NNMF, hierarchical clustering, k-means, latent
variables, latent space, dimensionality reduction, feature extraction, feature/variable selection,
spectral decomposition.

Mixture: spectral unmixing, blind source separation, source separation, non-negative matrix
factorization, multivariate curve resolution, deconvolution, sparse decomposition,
non-negative constraint correction.

**Position**: chemometrics is a *foundation*, not an obsolete predecessor. It supplies the mandatory
baselines, and MCR-ALS in particular remains the reference method for mixture resolution. A deep
model that cannot beat MCR-ALS on an instrument-disjoint split has not earned its parameters.

---

## H. Inverse problems — `P0`
**Module**: `usis/models/`, `usis/uncertainty/`

Forward problem, inverse problem, ill-posed inverse problem, non-uniqueness, identifiability,
degeneracy, regularization, Tikhonov regularization, sparsity, sparse reconstruction, Bayesian
inversion, Bayesian inverse problem, posterior inference, posterior predictive distribution, latent
variable inference, parameter estimation, system identification, state estimation, reconstruction,
source separation, blind inversion, blind source separation, uncertainty-aware inversion,
uncertainty quantification, spectral inversion, inverse spectroscopy.

**The mathematical core of the project.** `P(structure | spectrum)`, never `structure = f(spectrum)`.
Non-identifiability classes (isomer-degenerate, library gap, resolution-limited,
mixture-underdetermined) are *named outputs*, section 8 of the report.

---

## I. Simulation-based inference — `P1` *(addition beyond the original map)*
**Module**: `usis/models/`, `usis/simulate/`

Simulation-based inference, likelihood-free inference, approximate Bayesian computation, neural
posterior estimation, neural likelihood estimation, neural ratio estimation, sequential/amortised
inference, normalizing flows, score-based posterior sampling, summary statistics sufficiency,
model misspecification in SBI.

**Why it belongs here**: we possess a forward simulator (structure + conditions + instrument →
spectrum) and an intractable likelihood. That is the exact setting SBI was built for. It composes
directly with the agent: posterior → expected information gain → next measurement. The
retrieval/generation literature has largely bypassed this framing — novelty hypothesis 2.

---

## J. Computational spectroscopy — `P1`
**Module**: `usis/simulate/`

Computational/theoretical spectroscopy, spectral simulation, synthetic spectrum generation, forward
simulation, molecular spectrum simulation, Raman/IR spectrum simulation, quantum chemistry
simulation, DFT, TD-DFT, Hartree–Fock, post-Hartree–Fock, Møller–Plesset perturbation theory,
ab initio, first-principles, basis set, geometry optimization, normal mode calculation, vibrational
frequency calculation, Raman activity calculation, IR intensity calculation, polarizability
calculation, harmonic frequency scaling factors, anharmonic corrections, VPT2, molecular dynamics
spectroscopy, dipole/polarizability autocorrelation, finite-temperature spectra, conformational
ensembles, solvent effects, excited-state spectra, spectrum broadening/convolution,
theoretical-to-experimental alignment, synthetic-to-real transfer.

Machine-learned potentials: machine-learned interatomic potentials, neural network potentials,
Behler–Parrinello, equivariant MLIPs (MACE / NequIP / Allegro families), Δ-machine-learning,
foundation-model potentials, active learning for potentials.

**The loop this creates**:
`quantum/molecular physics → simulation → synthetic spectra → ML → experimental spectra → validation`

**Do not assume** millions of simulated spectra automatically produce a useful model. Measure the
domain gap; decompose it; then decide where compute goes.

---

## K. Molecular & chemical informatics — `P1`
**Module**: `usis/molecular/`

Cheminformatics, computational chemistry, molecular representation, molecular descriptor, molecular
fingerprint, chemical fingerprint, SMILES, SELFIES, InChI, InChIKey, molecular graph, 3D molecular
graph, conformer generation, molecular embedding, molecular property prediction,
structure–property relationship, **structure–spectrum relationship**, **spectrum–structure
relationship**, compound identification, chemical classification, chemical similarity, Tanimoto,
scaffold, Murcko scaffold, reaction informatics, materials informatics.

Spectrum↔structure as a field: spectrum–structure correlation/retrieval/matching,
spectrum-to-structure, structure-to-spectrum, forward spectral prediction, inverse spectral
inference, molecular reconstruction, de novo structure generation, molecular graph generation,
graph decoding, cross-modal alignment, contrastive spectrum–molecule learning, generate-then-rerank,
coarse-to-fine retrieval.

**Open question we must answer empirically**: are 2D graphs sufficient, or is 3D geometry essential?
Spectra depend on geometry; the prior art disagrees (1D-SMILES vs 2D-graph vs 3D-conformer models).

---

## L. Machine learning — `P1`
**Module**: `usis/models/`

General: machine learning, deep learning, supervised, unsupervised, semi-supervised,
self-supervised, weakly supervised, reinforcement learning, representation learning, feature
learning, transfer learning, domain adaptation, domain generalization, few-shot, zero-shot,
meta-learning, continual learning, online learning, active learning, federated learning,
multi-task learning, multimodal learning, generative modeling, foundation model, pretrained model,
embedding, latent representation.

Spectral-specific: spectral machine/deep learning, spectral representation learning, spectral
embedding, spectral feature learning, spectral classification/regression/prediction/identification,
spectral matching, spectral retrieval, spectral similarity, spectral anomaly detection, spectral
clustering, spectral segmentation, spectral denoising, spectral reconstruction, spectral
super-resolution, spectral interpolation, spectral synthesis/generation, spectral deconvolution,
spectral unmixing, spectral inverse problem.

Architectures: MLP, 1D/2D/3D CNN, RNN, LSTM, GRU, autoencoder, VAE, VQ-VAE, transformer, spectral
transformer, attention, self-attention, cross-attention, GNN, graph convolutional network,
equivariant GNN, neural operator, Fourier neural operator, diffusion model, GAN, mixture-of-experts,
hybrid networks, PINN.

Representation candidates to **benchmark, not assume**: raw continuous spectrum · spectral patches ·
peak tokens · wavelet/Fourier bases · learned latents · VQ tokens · hybrid. Score on information
loss, physical interpretability, robustness, resolution dependence, compute, transformer
suitability, retrieval suitability, mixture suitability, cross-modal suitability.

Pretraining objectives: masked spectral modeling, next-spectrum prediction, contrastive spectral
learning, spectrum↔molecule joint training, spectral augmentation invariance.

---

## M. Physics-informed & neuro-symbolic AI — `P2`
Physics-informed/guided/aware/constrained/embedded ML, scientific machine learning, AI for science,
physics-based model, data-driven model, hybrid model, mechanistic model, forward model, inverse
model, differentiable physics, differentiable simulation, neural simulation, scientific foundation
model, theory-guided learning, constraint-based learning, symmetry-aware learning, equivariance,
conservation laws, physical priors, inductive bias, domain knowledge injection, concept bottleneck,
neuro-symbolic reasoning, differentiable logic, symbolic solvers.

**Gate**: for every physics-informed method, answer before implementing — what equation is
available? Is it differentiable? Is it relevant to *this* task? Does the constraint improve
generalisation, measured? Is there empirical evidence? No answer, no PINN.

---

## N. Uncertainty & reliability — `P1`
**Module**: `usis/uncertainty/`

Bayesian inference, Bayesian ML, Bayesian neural network, probabilistic modeling, probabilistic
programming, Gaussian process, Gaussian process regression, Bayesian optimization, prior,
likelihood, posterior, epistemic uncertainty, aleatoric uncertainty, predictive uncertainty, deep
ensembles, Monte Carlo dropout, ensemble uncertainty, calibration, calibration error, ECE,
**conformal prediction**, prediction interval, credible interval, confidence interval, coverage,
uncertainty propagation, abstention, selective prediction, risk–coverage curve, AURC, confidence
estimation, OOD detection, open-set recognition, unknown class detection, failure detection.

**Requirement**: the system must be able to say *"I do not have enough evidence"* and name what
would resolve it. Conformal prediction is the natural fit for set-valued candidate output because it
gives distribution-free coverage.

---

## O. Domain shift & instrument intelligence — `P1`
**Module**: `usis/data/splits.py`, `usis/measurement/`, `usis/eval/protocols/`

Instrument drift, instrument variability, instrument-to-instrument variability, cross-instrument
transfer, calibration transfer, piecewise direct standardization, instrument standardization,
harmonization, normalization, spectral alignment, domain adaptation, domain generalization,
covariate shift, concept shift, dataset shift, distribution shift, batch effect, lab-to-lab
variation, device-to-device variation, biological variability, sample variability, synthetic-to-real,
device invariance, environment invariance, invariant representation, invariant risk minimization,
adversarial domain adaptation, test-time adaptation, continual calibration, model transfer.

**Assessment**: this matters *more* than the CNN-vs-Transformer question. Wavelength correspondence,
resolution, intensity response and spectral range differences routinely make a source model fail on
a target instrument. Design experiments that explicitly test A→B for instrument, lab, dataset,
synthetic→experimental, clean→noisy, high→low resolution.

---

## P. Data, quality, and datasets — `P0`
**Module**: `usis/data/`

Spectral data, spectrum, spectral dataset/matrix/cube/time-series/image, wavelength axis, wavenumber
axis, intensity axis, metadata (experimental, instrument, sample), measurement conditions, data
provenance, data lineage, data quality/integrity, normalization, standardization, representation,
format, raw/processed/reference/labeled/unlabeled spectrum.

Quality: quality control, quality assurance, outlier, outlier/anomaly/artifact detection, missing
data, corrupted spectrum, noise, drift, batch effect, reproducibility, repeatability, robustness,
reliability, generalization.

Datasets: reference/benchmark dataset, training/validation/test/calibration set, external
validation, ground truth, labels, weak labels, noisy labels, label uncertainty, data augmentation,
synthetic data, simulated spectra, experimental spectra, synthetic-to-real, real-to-sim, data
curation, data annotation, dataset bias, class imbalance, long-tail distribution, **data leakage**.

Formats and standards worth adopting rather than inventing: JCAMP-DX, mzML, nmrML, FAIR data
principles for spectroscopy.

**Every spectrum ideally carries**: modality, instrument, manufacturer, acquisition parameters,
axis range, resolution, sample identity, sample preparation, environment, temperature, pH,
replicate information, processing history, source, provenance, uncertainty, ground truth, licence.

---

## Q. Spectral libraries & knowledge — `P2`
**Module**: `usis/knowledge/`

Spectral library, Raman/IR spectral library, reference database, molecular/chemical database,
compound identification, spectral search/matching, similarity search, nearest-neighbour search,
approximate nearest neighbour, vector database, graph database, molecular/spectral/chemical
knowledge graph, reaction graph, pathway graph, ontology (spectroscopy, chemical, molecular,
reaction, instrument, metabolic), ontology alignment, entity linking, knowledge graph reasoning,
symbolic chemistry, reaction rules, mechanistic reasoning, retrieval-augmented generation,
retrieval-augmented scientific reasoning.

**Design note**: our RAG retrieves *reference spectra with measurement conditions*, not just text.
A retrieved spectrum whose acquisition conditions differ from the query is weak evidence and must be
down-weighted explicitly.

---

## R. Explainability — `P2`
**Module**: `usis/eval/metrics/`

Explainable AI, interpretable ML, model interpretability, feature importance, feature attribution,
saliency, attention visualization, SHAP, LIME, attribution, counterfactual explanation,
concept-based explanation, concept bottleneck, prototype learning, spectral attribution, **peak
attribution**, molecular interpretation, physically interpretable model, chemically interpretable
model, mechanistic interpretability, faithfulness.

**Rule**: attention ≠ explanation. Faithfulness is measured by *specificity of occlusion* — masking
the diagnostic region for group *g* must degrade *g* specifically (E0/T6).

---

## S. Validation & evaluation — `P0`
**Module**: `usis/eval/`

Prediction: accuracy, precision, recall, F1, AUROC, RMSE, MAE, R², Recall@k, MRR, Tanimoto, exact
match, molecular validity, cross-validation, external validation, leave-one-out, hold-out,
benchmarking, statistical significance, bootstrap confidence intervals, IQM, performance profiles.

Scientific correctness: physical consistency, chemical consistency, spectral reconstruction error,
cycle consistency, reaction validity, mechanistic consistency.

Reliability: calibration, uncertainty quality, selective accuracy, abstention quality, OOD
detection, coverage, set size.

Generalization: cross-instrument, cross-lab, cross-dataset, cross-domain, unseen molecules, unseen
mixtures, unseen modalities, sim→real.

Explanation: faithfulness, attribution accuracy, peak attribution, concept fidelity, counterfactual
consistency.

Agentic: hypothesis quality, experiment selection efficiency, information gain, intervention
success, planning efficiency, tool-use correctness, recovery from failure, measurements-to-decision.

Splits: random, sample-disjoint, molecule-disjoint, scaffold-disjoint, instrument-disjoint,
lab-disjoint, dataset-disjoint, resolution-shift, open-set, synthetic-to-real.

Stress tests: noise, baseline shift, resolution shift, peak shift, mixture, OOD, adversarial spectral
perturbation, counterfactual, permutation control, retrieval ablation, physics-constraint ablation,
knowledge-graph ablation, uncertainty ablation.

---

## T. Model security & robustness — `P2`
Adversarial robustness, adversarial examples, data poisoning, model poisoning, distribution shift,
OOD detection, open-set recognition, unknown class detection, abstention, selective prediction,
confidence estimation, model uncertainty, failure detection, fault diagnosis, safe AI, reliable AI.

---

## U. Multimodal & fusion — `P2`
Multimodal AI/learning, multisensor fusion, sensor fusion, data fusion, early/late/intermediate
fusion, cross-modal learning, cross-modal representation, cross-attention, modality-specific
encoders, missing-modality learning, modality dropout, modality uncertainty. Combinations:
Raman+IR, Raman+fluorescence, Raman+microscopy, spectroscopy+imaging, spectroscopy+MS,
spectroscopy+chromatography, spectroscopy+structure, spectroscopy+clinical data.

**Constraint**: the architecture must never *require* all modalities to be present. Missing-modality
behaviour is a design requirement, not an afterthought.

---

## V. Agentic & autonomous science — `P2`
**Module**: `usis/agent/`

Agentic AI for science, scientific agents, AI scientist, autonomous scientific discovery, scientific
workflow agents, tool-using agents, planning agents, experiment planning, active learning, active
sensing, adaptive spectroscopy, adaptive/intelligent measurement, optimal experimental design,
Bayesian experimental design, expected information gain, closed-loop experimentation, autonomous
laboratory, self-driving laboratory, hypothesis generation, hypothesis testing, scientific decision
making, world model, scientific world model, agentic world model, model-based planning, belief
state, observation–action loop, uncertainty-aware planning, decision-centric evaluation.

**Framing** (from *Agentic World Modeling*, arXiv:2604.22748): L1 Predictor / L2 Simulator /
L3 Evolver × governing-law regimes. Spectroscopy sits in the **Scientific World** — mechanisms are
partially known and constraints are verified empirically — with a measurement layer that behaves
like the Digital World: specifiable, loggable, replayable.

The evaluation question is not *"did it identify the molecule?"* but: does it know when it is
uncertain; does it choose a useful next measurement; does it retrieve the right reference; does it
know when to simulate; does it detect instrument shift; does it revise its hypothesis after new
evidence?

---

## W. Software & computational infrastructure — `P1`
Scientific computing, numerical computing, HPC, GPU computing, parallel/distributed computing,
cloud computing, scientific software, spectroscopy software, instrument control, data acquisition,
data pipeline, ETL, preprocessing pipeline, inference pipeline, ML pipeline, MLOps, model serving,
deployment, monitoring, experiment tracking, version control, dataset versioning, reproducible
computing, containerization, API, SDK, workflow engine.

Tools to evaluate rather than assume: PyTorch, PyTorch Geometric, Hugging Face, RDKit, ASE,
quantum chemistry packages, MLIP libraries, `ramanchada2`, RAMANMETRIX, ONNX, vector databases,
graph databases, ontology tools, scientific RAG frameworks, workflow engines, experiment trackers.
For each: purpose, maturity, licence, research usefulness, production usefulness, alternatives.

---

## X. Application domains — `P2`, opportunistic
Materials, semiconductor, mineral, polymer, pharmaceutical, food, agricultural, biomedical,
bio-spectroscopy, tissue, cell, single-cell, microbial, pathology, cancer, environmental, forensic,
planetary, remote sensing spectroscopy. Also: food authenticity, drug stability, pollutant
detection, process analytical technology.

**Rule**: choose *one* application for validation depth rather than claiming all of them. Breadth
here is the most common way this class of project loses credibility.

---

## Y. Parked — `P3`
Recorded so they are not lost. Not built.

- **Quantum machine learning / quantum computing hardware.** Quantum *chemistry* simulated
  classically is `P0`. Running our inference on quantum hardware has no demonstrated advantage for
  this class of inverse problem. Keep the distinction sharp if asked.
- **Real-time / edge spectroscopy**: edge AI, embedded ML, on-device AI, TinyML, low-latency
  inference, streaming/online/in-line spectroscopy, continuous monitoring, PAT, industrial
  spectroscopy.
- **Hardware–software co-design**: AI accelerator, FPGA, NPU, ASIC, computational sensing,
  computational spectroscopy, smart/intelligent spectrometer, software-defined spectroscopy,
  reconfigurable spectroscopy, adaptive/active/intelligent acquisition.
- **Autonomous laboratory robotics**: robotic spectroscopy, closed-loop experiment execution,
  self-driving laboratory hardware.
- **Biological causal inference**: digestion, metabolism, microbiome interaction, pharmacokinetics,
  metabolic pathways, phenotypic outcomes. The mechanistic-hypothesis pathway may exist in the
  output schema; a causal claim never leaves the system without interventional data.

---

## Recommended reading order

Do **not** learn AI first and spectroscopy second. The dependency order is:

```
Physics (A, B)
  → Measurement science (D, E)
    → Spectral signal science (F)
      → Modalities: Raman & IR in depth (C1, C2)
        → Chemometrics (G)
          → Inverse problems (H)
            → Computational spectroscopy (J) ∥ Molecular informatics (K)
              → Spectral representation learning (L)
                → Uncertainty (N) ∥ Domain shift (O)
                  → Knowledge & retrieval (Q)
                    → Agentic layer (V)
```

Evaluation (S) and data (P) are not a stage — they are built continuously from the first commit.