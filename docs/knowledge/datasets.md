# Dataset matrix

**Verified:** 2026-09-02. **Method:** web search plus direct fetch of the primary source where the
source permitted it. **Nothing in this file is quoted from memory.**

## How to read the status column

| Status | Means |
| --- | --- |
| `VERIFIED` | The primary source (publisher page, repository API, project site) was fetched this session and the stated facts came from it. |
| `PARTIAL` | Facts came from search-result summaries or a secondary source; the primary source was unreachable (paywall redirect, 403) or silent on the point. |
| `UNVERIFIED` | Named in the brief, not yet checked. **No loader may be written against an `UNVERIFIED` row.** |

The `licence` column uses the vocabulary in `usis/data/licences.py`. `TERMS-OF-USE` and
`UNVERIFIED` are both non-redistributable, so the CI gate blocks vendoring either.

---

## 1. Experimental vibrational spectra

| Dataset | Modality | Size | Molecules / phases | Metadata richness | Licence | Status | Use case |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Raman Open Database (ROD)** | Raman | >1100 entries | >500 phases | Good: JCAMP-DX + CIF2, interlinked with the Crystallography Open Database | `CC0-1.0` | `VERIFIED` | The one open corpus we may vendor outright. First loader. |
| **RRUFF** | Raman (multiple excitations), plus XRD and IR | not established | Minerals only | Strong: locality, chemistry, XRD, photographs | `TERMS-OF-USE` (see note) | `PARTIAL` | Instrument/excitation diversity on well-characterised solids. Fetch at build time, never vendor. |
| **NIST Chemistry WebBook (IR)** | IR | Large, per-compound pages | Broad organic coverage | Moderate: phase and instrument notes vary by record | `TERMS-OF-USE` | `PARTIAL` | Reference IR. Query access; bulk redistribution not established. |
| **SDBS (AIST)** | IR, Raman, NMR, MS | ~34,000 organic molecules (secondary sources; not confirmed on the AIST site) | Broad organic | Moderate | `TERMS-OF-USE` | `PARTIAL` | Large and attractive; terms are the blocker, not the data. |
| **Open Raman spectra for API development** (*Sci Data* 12:498, 2025) | Raman | 3,510 spectra | 32 compounds (solvents and reagents) | Per compound: exposure, pixel fill, assay, supplier. **One instrument, stated only in the paper — the CSV has no instrument column** | `CC-BY-4.0` | `VERIFIED` | Replicate-rich, small compound set. Note the replicate structure below: `sample_disjoint` collapses onto `molecule_disjoint` here. |
| **RamanBench** (arXiv:2605.02003) | Raman | 325,668 spectra across 74 datasets (16 first released there) | Four application domains | Varies by constituent dataset | `UNVERIFIED` | `PARTIAL` | A ready-made unified access layer and leaderboard. Read before designing our own splits. |
| **Open Raman spectral library for biomolecules** (*Chemom. Intell. Lab. Syst.*, 2025) | Raman | not established | Biomolecules | not established | `UNVERIFIED` | `UNVERIFIED` | Named in the brief; needs a direct check. |

**RRUFF licence note.** The RRUFF site footer states only `Copyright © 2026 RRUFF - Comprehensive
Database of Mineral Data`; no licence, permitted-use or redistribution statement was found on the
About page fetched this session. A secondary source reports terms limiting a scientist to a single
downloaded copy and up to 20 paper reproductions for non-commercial purposes, but that wording was
not confirmed against an RRUFF page. **Treated as `TERMS-OF-USE`: usable in place, not
redistributable.** *Action to close this: email the RRUFF project for a written statement, or find
the terms page. Until then no RRUFF bytes enter the repository.*

**Why the small numbers matter.** The whole open, well-licensed, experimental Raman corpus we have
actually confirmed is on the order of **10³–10⁴ spectra**, most of it minerals or a few dozen
solvents. See `docs/adr/0001-brief-critique.md` §S6: the binding constraint on the M2 evaluation is
not model capacity, it is whether an `instrument_disjoint` split over *molecules* is constructible
at all from open data.

---

## 2. Simulated / computed vibrational spectra

| Dataset | Content | Size | Level of theory | Licence | Status | Use case |
| --- | --- | --- | --- | --- | --- | --- |
| **QM9S** | IR, Raman, UV-Vis for QM9-scale organics | 133,885 molecules per the paper; the figshare record says ~130,000. ~25.9 GB (broadened IR 8.7 GB, Raman 8.8 GB, UV 1.5 GB) | B3LYP/def-TZVP geometry + frequency + TD-DFT, Gaussian 16 B.01 | `CC-BY-4.0` | `VERIFIED` | The workhorse simulated corpus, **loader written**. Broadened grid 500–4000 cm⁻¹ at 1 cm⁻¹ (3,501 points); the broadened CSVs carry a molecule index but no SMILES. Cite the article DOI `10.1038/s43588-023-00550-y`, which is what the readme asks for. |
| **QMe14S** | IR, Raman, NMR + energies, charges, forces, dipole, polarizability, hyperpolarizability | 186,102 molecules, 14 elements, 47 functional groups | B3LYP/TZVP | `CC-BY-4.0` (paper badge; repository link not stated in the abstract) | `PARTIAL` | Wider element and functional-group coverage than QM9S. arXiv:2501.18876. |
| **Computed Raman of inorganic compounds with hybrid functionals** (*Sci Data* 11:105, 2024) | Raman + phonon properties | 161 compounds, growing | Hybrid functionals; structures from ICSD; 78 overlap with RRUFF | `CC-BY-4.0` (data + code on GitHub) | `PARTIAL` | Small but directly paired with RRUFF experimental spectra → **a ready-made sim2real probe set**. |
| **VB-Mols / ViBench** | Vibrational spectrum↔structure benchmark used by Vib2Mol / Vib2Conf | not established | not established | `UNVERIFIED` | `PARTIAL` | Referenced by the prior art with Recall@k numbers; needs a direct check before we reproduce anything on it. |
| **SPICE, ANI-1x, MD17** | MLIP training corpora | not established | not established | `UNVERIFIED` | `UNVERIFIED` | Only relevant if we take the MLIP route to simulation. Defer to M5. |

---

## 3. Molecular / reaction structure sources

| Source | Content | Licence | Status | Use case |
| --- | --- | --- | --- | --- |
| PubChem | compounds | `UNVERIFIED` | `UNVERIFIED` | Identifier resolution, InChIKey normalisation |
| ChEMBL | bioactive compounds | `UNVERIFIED` | `UNVERIFIED` | Scaffold statistics |
| ZINC | purchasable compounds | `UNVERIFIED` | `UNVERIFIED` | Open-set distractors |
| USPTO reactions | reactions | `UNVERIFIED` | `UNVERIFIED` | Later phases only |
| KEGG | pathways | `UNVERIFIED` | `UNVERIFIED` | **Parked** — biology is P3 |

These are structure sources, not spectral sources. None is on the M1 critical path: our need for
them starts when open-set distractor libraries are built (M6).

---

## 4. Software we should depend on rather than reimplement

| Tool | What it gives us | Licence | Status |
| --- | --- | --- | --- |
| **ramanchada2** (`h2020charisma/ramanchada2`) | Raman I/O across vendor formats, processing, and *instrument calibration* — the part we least want to write ourselves | `MIT` | `VERIFIED` |
| **RamanSPy** | Integrative Raman analysis package with dataset loaders (*Anal. Chem.*, DOI `10.1021/acs.analchem.4c00383`) | `UNVERIFIED` | `PARTIAL` |
| **RAMANMETRIX** | Web platform for Raman analysis | `CC-BY-NC-ND-4.0` — **not open source**, no derivatives | `PARTIAL` |
| **SpectrumLab / SpectrumBench** (arXiv:2508.01188) | 14 spectroscopic tasks, >10 spectrum types, spectra curated from >1.2M substances; evaluated 18 multimodal LLMs | `UNVERIFIED` | `PARTIAL` |

RAMANMETRIX's `ND` clause means we cannot build on it. It is a reference to study, not a dependency.

---

## 5. Instrument coverage — the Q1 audit

**Audited 2026-09-03.** The question (open question Q1): can an `instrument_disjoint` split over the
**same molecules** be constructed from open data? It gates the M2 headline criterion, the brief's
open question 8, and novelty candidate N3.

The measurement: per source, are the same substances measured on **≥2 distinct instruments**, and is
the instrument recoverable as a split key?

| Source | Instrument metadata | Same substance on ≥2 instruments? | Usable as a split key? | Status |
| --- | --- | --- | --- | --- |
| **RamanBench — Bioprocess Analytes collection** | The instrument **is** the dataset: eight sibling datasets, one per spectrometer | **Yes.** Eight datasets measuring *nearly* the same glucose / sodium acetate / magnesium sulfate mixtures | **Yes — and all eight are CC BY 4.0**, so the experiment is runnable and the data vendorable | `VERIFIED` (catalogue + paper) |
| **RamanBench — overall (74 datasets)** | Excitation "532 nm to 1064 nm across benchtop, portable, and process instruments"; instrument named in per-dataset free text, **not a standardised field** | Across datasets, not within: different datasets use different instruments *and* different substances, so the two vary together | Only after per-dataset reconstruction | `VERIFIED` |
| **RRUFF** | Excitation recorded per spectrum | **Different excitation on the same sample** (e.g. R060890 at 532 nm and 785 nm) — but same instrument | Gives an *excitation-disjoint* split, not instrument-disjoint | `PARTIAL` |
| **ROD** | Rich, per record: company, model, location, excitation, resolution, grating, objective, integration time, calibration standard, **and an explicit declaration of whether baseline correction / background subtraction was applied** | **Yes.** Over all 1,129 readable records: six instrument models across seven laboratories, the largest two holding 806 and 152 records | **Yes for instrument, with a caveat that must travel with it**: instrument and laboratory are almost perfectly confounded, so the split is also lab-disjoint and the two effects cannot be separated. **Yes for excitation**: 432 of 621 specimens are measured at ≥2 wavelengths | `VERIFIED` (exhaustive) |
| ***Sci Data* 12:498 (API set)** | **None in the data.** The CSV is 3,276 wavenumber columns plus a `label`; the instrument is named only in the paper | **No — one instrument.** Endress+Hauser (formerly Kaiser Optical Systems) Raman Rxn2, Rxn-10 non-contact probe, 10x objective, 785 nm | No. And **`sample_disjoint` is unavailable as a separate split**: one 4 mL vial per compound, so all ~110 scans of a compound share a physical sample | `VERIFIED` |
| **QM9S / QMe14S** | n/a — simulated, no instrument | n/a | n/a (this is what `sim2real` is for) | `VERIFIED` |

### Verdict: (b) constructible, but only in a narrow domain

**The data is not absent.** RamanBench's Bioprocess Analytes collection provides the same analytes
across eight instruments — exactly the pairing needed. Two limits define what it can support:

1. It is a **bioprocess quantification/classification** collection, not a structure library. It
   supports "how much does performance drop across instruments" on those tasks. It does **not**
   support an instrument-disjoint `spectrum_to_structure` evaluation, which remains unavailable.
2. **RamanBench itself does not run this evaluation.** Its protocol is "fixed 80/20 train/test
   splits over 3 different seeds", stratified for classification; the paper notes the collection
   "offers an opportunity for transfer learning" without taking it.

That second point is the sharper finding, and it changes what N9 claims: the field's gap here is
**not missing data, it is protocol choice.** The benchmark that aggregates 325,668 spectra holds an
eight-instrument same-analyte collection and evaluates with random splits. See
`PROJECT_NOVELTY.md` N9.

### ROD, audited exhaustively (2026-09-03) — this **replaces** an earlier sampled audit

**The earlier audit was wrong, and it was wrong in the way a sample is wrong.** It read 240 records
from the guessed id blocks `8100001–8100600`, `1000001–1000400` and `2000001–2000200`, found 238 of
240 on one instrument, and concluded that ROD "cannot support an instrument-disjoint split. Two
entries on the second instrument is not a split, it is a rounding error."

ROD publishes an index (`result.php?format=lst`, the Crystallography Open Database query interface
ROD inherits). It lists **1,133 codes in nine id blocks**. The old sampling frame covered three of
them, and the block it drew most heavily from is the one Arizona deposited. Mirroring the whole
database and re-running the audit gives a different answer.

Reproduce with `uv run usis data pull rod --all` then
`uv run python scripts/audit_rod_instruments.py`.

| Field | Finding over 1,129 readable records |
| --- | --- |
| Instrument | Thermo Nicolet Almega XR **806** (71.4%) · Thermo DXR **152** (13.5%) · Horiba Aramis **53** · Horiba LabRam HR800 **50** · HORIBA Jobin Yvon T64000 **37** · Thermo DXR2 **16** · two home-made filter systems **15** |
| Laboratory | arizona-university 806 · ensicaen 152 · university-of-verona 66 · university-of-trento 50 · immm-maine-university 35 · okayama 15 · montpellier 2 |
| Excitation | 532 nm 481 · 780 nm 419 · 633 nm 118 · 785 nm 58 · 514 nm 37 · 488 nm 15 · 632.8 nm 1 |
| Resolution | 4 cm⁻¹ 808 · 3 cm⁻¹ 150 · 1 cm⁻¹ 82 · 2 cm⁻¹ 76 · 2.5 cm⁻¹ 5 · 1.5 cm⁻¹ 5 · 9 cm⁻¹ 2 · 6 cm⁻¹ 1 |
| **Specimens at ≥2 excitations** | **432 of 621** |

Five consequences, and the first reverses a recorded conclusion.

1. **ROD *can* support an instrument-disjoint split.** Six named instrument models across seven
   laboratories, with the second-largest holding 152 records — not a rounding error. This
   materially reduces risk S6 in `docs/adr/0001-brief-critique.md`, which said the data for an
   instrument-disjoint split might not exist in the open. For Raman *phase identification* on
   minerals, it does.
2. **But instrument and laboratory are almost perfectly confounded**, and a split that ignores that
   is claiming more than it has. The Almega XR appears only at Arizona and the DXR only at
   ENSICAEN. Only two models cross a lab boundary at all: the T64000 (35 at IMMM Maine, 2 at
   Montpellier) and the Aramis (50 at Trento, 1 at Verona, 2 unrecorded). **An instrument-disjoint
   split over ROD is therefore also a lab-disjoint split, and the two effects cannot be separated
   in this corpus.** Reporting it as instrument shift alone would be the same overclaim as calling
   an excitation split an instrument split.
3. **The instrument strings are free text and need normalisation before they can be a split key —
   and the normalisation is a decision, not an inference.** Three defect families are present in
   the published records and are *not* fixed in the loader:
   - a value with the next CIF tag embedded in it, quoted in the source file itself:
     `'LabRamHR800_raman_measurement_device.optics_type microscope'` (21 records) and
     `'Thermo Scientific_raman_measurement_device.model DXR2 unrecorded'` (16 records);
   - spelling variants of one model: `LabRamHR800`, `LabRamaHR800`, `LabRam HR800`;
   - a laboratory typo that manufactures a spurious group: `university-of-verono` (1) against
     `university-of-verona` (66).
   Collapsing these silently in the loader would bake a judgement into the corpus. The counts above
   are collapsed by hand *for this table only*.
4. **ROD supports an excitation-disjoint split very well**: 432 of 621 specimens are measured at
   two or more excitations, far above the 58-of-240 the sample suggested. That probes the
   wavelength-dependent effects — fluorescence background, the ν⁴ scattering factor, resonance —
   and it must never be labelled instrument shift.
5. **The sample key is `_rod_data_source.file`, not the ROD code** (`PROJECT_DECISIONS.md` D9).
   With 432 multi-excitation specimens rather than 58, the leakage a random split would create is
   far larger than the sample implied.

**Two more things a sample could not have shown.**

*ROD holds computed spectra alongside measured ones.* Records `3500266` and `3500267` carry
`_raman_theoretical_spectrum.raman_shift` and no measured loop: they are DFT predictions deposited
next to the measurements. `usis/data/sources/rod.py` refuses them by name. A loader reaching for
whatever `raman_shift` column it could find would put a DFT spectrum into a corpus labelled
`EXPERIMENTAL` with an instrument attached — `F13` created at load time and invisible everywhere
downstream. Supporting them needs a separate loader with `origin=SIMULATED` and a level of theory.

*Two records contradict their own declared range.* `3500044` spans [20.28, 2999] against a declared
[20.28, 2500]; `3500076` spans [50.17, 2000] against [50.17, 199.9]. The declared-range check was
written against two fixtures and has now earned its keep at 1,131: both records are rejected rather
than silently loaded with a wrong axis.

*Sampling frame: none. This is every code in ROD's published index as of 2026-09-03; 1,131 fetched,
1,129 parsed, 2 refused as computed spectra, 2 refused for the range contradiction.*

### RamanBench, audited through the catalogue (2026-09-03)

Audited offline against the `raman-data` package's own metadata API (MIT, v1.6.2), snapshotted into
`usis/data/sources/data/ramanbench_catalogue.json` so the manifest and the licence gate work
without the dependency.

**The eight-instrument collection, named.** `bioprocess_analytes_*` is eight sibling datasets, one
per spectrometer, all **CC BY 4.0**:

| Dataset | Vendor | Excitation | Replicate structure |
| --- | --- | --- | --- |
| `bioprocess_analytes_anton_532` | Anton Paar | 532 nm | known |
| `bioprocess_analytes_anton_785` | Anton Paar | 785 nm | known |
| `bioprocess_analytes_kaiser` | Kaiser Optical Systems | not stated | none |
| `bioprocess_analytes_metrohm` | Metrohm | not stated | known |
| `bioprocess_analytes_mettler_toledo` | Mettler Toledo | not stated | known |
| `bioprocess_analytes_tec5` | tec5 | not stated | known |
| `bioprocess_analytes_timegate` | Timegate | not stated | none |
| `bioprocess_analytes_tornado` | Tornado Spectral Systems | not stated | known |

**Analytes: three** — glucose, sodium acetate and magnesium sulfate, in mixtures, with
concentrations verified by enzymatic assay. So the task is *regression on three concentrations*,
not compound identification, and every claim drawn from this collection inherits that scope.

**One caveat that must travel with the result**: the catalogue says the eight instruments measure
*"nearly the same samples"*, and *"some datasets have a bit more samples than others"*. So a
leave-one-instrument-out split is **not** a perfectly matched design, and a performance drop across
instruments is partly a sample-composition difference. Quantifying that confound is part of the
experiment, not a footnote to it.

**Licences: 16 distinct strings across 93 datasets, and 8 of them are not licences.** Our gate maps
only the unambiguous ones and blocks the rest:

| Mapped (73 datasets) | Blocked as `UNVERIFIED` (20 datasets) |
| --- | --- |
| CC BY 4.0, CC0 1.0, MIT, GPL-3.0, CC BY-NC 3.0, CC BY-NC-SA 4.0, CC BY-NC-ND 4.0, "research use only" | "Optica OAPA" (5), "Requested from authors" (4), "Authors contacted" (3), "unknown" (3), "Free access" (2), "Unclear. Repo root LICENSE is MIT…" (1), "Authors contacted (data provided by Analyze IQ Limited)" (1), "BY-NC" (1) |

That right-hand column is a worklist, not a failure: eight strings somebody has to read before
20 datasets can be vendored. Guessing which CC variant `"BY-NC"` means is exactly the shortcut that
becomes a licence violation, so the gate refuses instead.

**23 datasets are blocked in total**, not 20: the other three carry licences we *did* map and that
still forbid redistribution here — two are "research use only" (`ACADEMIC-USE-ONLY`) and one is
CC BY-NC-ND 4.0, whose no-derivatives clause blocks us because every artifact this project makes
from a dataset is a derivative. Mapping a licence and being allowed to vendor under it are separate
questions, and the gate keeps them separate.

**Replicate structure is tracked upstream**, which is unusual and welcome: `is_grouped` is `True`
for 45 datasets, `False` for 21, and `None` for 27 — with the package's own documentation warning
that `None` means unknown, not ungrouped. Our loader propagates that distinction into the label,
so a split can refuse rather than leak.

**A trap worth recording**: `DatasetInfo.id` is a *source record* id and is **not unique** — four
adenine datasets share Zenodo record `3572359`, and two sugar-mixture datasets share `10779223`.
Keying the catalogue on it would have silently merged six datasets into two. The snapshot is keyed
by the `list_datasets()` name instead.

**What is still not available anywhere**: an instrument-disjoint split for
`spectrum_to_structure`. The eight-instrument collection is quantitative regression on three
analytes. Nothing in the 93 datasets pairs a *structure library* across instruments.

### The *Sci Data* API set, audited directly (2026-09-03)

Audited by reading the published files rather than the abstract: the figshare API for the licence
and file list, HTTP range requests for the CSV schema, the product-information spreadsheet, the
analysis notebooks, and the paper's full text via Europe PMC.

| Question | Answer |
| --- | --- |
| Instruments | **One.** Endress+Hauser (formerly Kaiser Optical Systems) Raman Rxn2 analyser, Rxn-10 non-contact probe, 10x objective, 785 nm, driven by Mettler Toledo iC Raman 4.1 |
| Same compound on ≥2 instruments | **No** |
| Axis | 3,276 columns, 150–3425 cm⁻¹, exactly 1 cm⁻¹ spacing |
| Replicate structure | One 4 mL amber vial per compound; ~100–145 repeated manual scans with the probe moved laterally and rotationally |
| Per-spectrum metadata in the file | **None** — intensities and a compound label, nothing else |
| Per-compound metadata | Exposure (s), pixel fill %, assay %, supplier, sample count, formula, reference bands |

Three consequences, and the second is the one that would have bitten us:

1. **No instrument-disjoint split**, and none is recoverable: with one instrument there is nothing
   to hold out.
2. **`sample_disjoint` collapses onto `molecule_disjoint`.** One vial per compound means every scan
   of a compound is a replicate of one physical sample. A random split puts ~110 scans of the same
   vial on both sides — the leakage the brief names as the most common silent killer, and nothing
   in the file itself would reveal it. `usis/data/sources/scidata_api.py` sets `sample_id` to the
   compound for exactly this reason.
3. **The data is not raw, though it is widely described that way.** The paper states that automatic
   pre-treatment "included dark noise subtraction, cosmic ray filtering, and intensity correction",
   with no further preprocessing afterwards. All three are recorded in the loader's processing
   chain; an empty chain would assert that nothing was applied.

One measurement caveat worth carrying forward: the paper calls 1 cm⁻¹ the "average spectral
resolution", but 3,276 points across 150–3425 cm⁻¹ *is* a 1 cm⁻¹ grid step, so that figure
describes the sampling interval, not the instrument response. The loader therefore leaves
`resolution_fwhm` unset rather than recording a resolution nobody measured — which would corrupt
any resolution-shift experiment (`F07`).

### QM9S line broadening, measured (2026-09-03)

Not an instrument-coverage question, but it belongs with the audits because it is a published-data
fact that the publisher did not state and that Q2 cannot be answered without.

The QM9S readme gives the broadened grid and calls the spectra "Unnormalized", but never says what
lineshape was applied or how wide. Both are needed before a simulated spectrum can be compared with
a measured one: an instrument response must be convolved *on top of* whatever broadening is already
there, and convolving onto an unknown starting width gives an unknown answer.

Measured over the first 15 molecules of each broadened file, on isolated peaks (no neighbour above
10% of the peak within ±60 cm⁻¹), half-maximum crossings interpolated between grid points:

| | Isolated peaks | FWHM | Lorentzian residual | Gaussian residual |
| --- | --- | --- | --- | --- |
| `ir_boraden.csv` | 25 | **15.085 ± 0.121 cm⁻¹** | 2–4% | 28% |
| `raman_boraden.csv` | 21 | **10.101 ± 0.083 cm⁻¹** | 4–7% | 32% |

**Lorentzian, 15 cm⁻¹ (IR) and 10 cm⁻¹ (Raman).** The spreads are far below one grid step, so the
nominal values are almost certainly exactly 15 and 10; the small excess is the tail contribution of
neighbouring modes.

Why it matters for Q2: 10–15 cm⁻¹ is comparable to, or wider than, the resolution of the
experimental sources in this matrix (ROD reports 4 cm⁻¹, and the *Sci Data* set samples at 1 cm⁻¹).
So a sim→real comparison against those sources is **not** resolution-limited by the instrument — it
is limited by the simulation's own broadening, and matching them means either degrading the
experimental spectra or deconvolving the simulated ones. That is a decomposition step in the Q2
experiment, and it would have been invisible without measuring.

**Confirmed in code (2026-09-03).** The forward model refuses to sharpen, so
`ResolutionConvolution.to_target(10.0, 4.0)` raises: the direction everyone reaches for first —
degrade the simulation down to the experiment — is unavailable. The workable direction is the
reverse, and it now runs end to end on the committed fixtures: resample ROD onto a uniform grid,
convolve 4 → 10 cm⁻¹, and its resolution matches QM9S exactly. Any Q2 result must state which
direction it used, because they are not equivalent: one discards experimental information, the
other would have to invent simulated detail.

Recorded in `usis/data/sources/qm9s.py` as `BROADENING_FWHM_CM1`, with the value re-derived from the
committed fixture in the test suite so the constant cannot drift from the data it came from.

### Real spectrometer axes are not uniform, and that is dispersion (2026-09-03)

Found while making the forward model work on real data, and worth recording because it silently
breaks any operation specified in cm^-1.

A grating disperses light linearly in **wavelength** across the detector, and `1/λ` is not linear,
so the wavenumber step varies across the range. Measured on the two committed ROD records:

| Record | Points | Mean step | Spread |
| --- | --- | --- | --- |
| `8100020.rod` (Thermo Nicolet Almega XR) | 2,297 | 0.482 cm⁻¹ | **0.23%** |
| `1000001.rod` (HORIBA T64000) | 1,024 | 2.130 cm⁻¹ | **25.5%** |

Neither is uniform, and the second is not close. This is not noise or a defect; it is what the
optics do. The consequence is concrete: a convolution kernel specified in cm^-1 has no fixed width
in samples on such a grid, so `ResolutionConvolution` refuses until the spectrum has been resampled,
and resampling is its own recorded step rather than something a loader does quietly.

The published matrix CSVs (*Sci Data*, QM9S) **are** uniform — because their publishers already
resampled them onto a common grid, which is itself a preprocessing step that the raw instrument
output did not have.

### What the splits can actually be built over (2026-09-03)

Now that the split machinery exists, the audit above resolves into a table of what is constructible
from the four loaded sources. This is the practical form of the Q1 answer.

| Split | Constructible from | Not constructible from |
| --- | --- | --- |
| `sample_disjoint` | ROD, *Sci Data*, QM9S, 66 RamanBench datasets | 27 RamanBench datasets — replicate structure `unknown`, and the split refuses rather than guess |
| `molecule_disjoint` | ROD, *Sci Data*, RamanBench classification sets | QM9S (structures live in a separate archive) |
| `instrument_disjoint` | **Only** RamanBench `bioprocess_analytes_*` (8 spectrometers) | Everything else — each is a single instrument |
| `lab_disjoint` | Nothing verified | No source records more than one lab per dataset |
| `excitation_disjoint` | ROD (532 / 780 nm, 58 shared specimens in the sample), RRUFF | The rest are single-excitation |
| `sim2real` | QM9S → any experimental source | — |
| `resolution_shift` | QM9S (10–15 cm⁻¹) vs ROD (4 cm⁻¹) | *Sci Data*, whose resolution is unrecorded |
| `scaffold_disjoint` | Nothing yet — needs RDKit **and** per-spectrum SMILES, which no loaded source has | — |
| `open_set` | ROD, *Sci Data*, RamanBench classification sets | — |

Two entries deserve emphasis because they are easy to misreport:

- **`lab_disjoint` is empty.** Not "hard", empty. Nothing we have verified records spectra from more
  than one lab within a dataset. Any lab-disjoint claim in this project would be fabricated today.
- **`excitation_disjoint` is not `instrument_disjoint`.** ROD gives the former and not the latter,
  and they answer different questions — wavelength-dependent physics versus instrument
  fingerprinting. The split kinds are named separately for exactly this reason.

### What this costs us at M1

- `instrument_id` is **not a field** in the unified schema. The `raman-data` package (MIT, PyPI,
  93 datasets) returns `RamanDataset` objects with spectra, `raman_shifts`, targets and dataset-level
  metadata; instrument appears in per-dataset prose (e.g. "Horiba LabRAM HR Evolution spectrometer,
  633 nm, 13.17 mW"), not standardised across entries. Populating `InstrumentSpec.instrument_id`
  means a per-dataset extraction step, and it must be recorded as `UNVERIFIED` where the prose is
  ambiguous rather than guessed.
- **Licences are per constituent dataset**, with no unified licence across the RamanBench
  collection. Our vendoring gate has to resolve 74 licences, not one. Budget for it.

### Still open

- The Bioprocess Analytes specifics — analyte count, the eight instrument names, excitation
  wavelengths, licence, and whether it is one of the 16 first-released datasets — are in the paper's
  appendix (§A.14.2) / supplementary material, which was not reachable this session.
  **Action to close: read §A.14.2, or inspect the collection through `raman-data` directly.**
- Whether ROD and the *Sci Data* API set carry instrument metadata at all. Both are small enough that
  a direct download settles it at M1.
- `lab_disjoint` was not established anywhere. Eight instruments in one bioprocess collection are
  plausibly one lab, which would leave lab-disjoint unsupported.

---

## Sources

- [Raman Open Database](https://solsa.crystallography.net/rod/index.php)
- [RRUFF project — about](https://www.rruff.net/about/about_general.php) · [zipped data archives](https://www.rruff.net/zipped_data_files/raman/)
- [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) · [NIST licensing statement](https://www.nist.gov/open/license)
- [SDBS](https://sdbs.db.aist.go.jp/) · [SDBS in re3data](https://www.re3data.org/repository/r3d100010822)
- [Open-source Raman spectra for API development, *Sci Data* 12:498 (2025)](https://www.nature.com/articles/s41597-025-04848-6) · [figshare record](https://springernature.figshare.com/articles/dataset/27931131)
- [RamanBench, arXiv:2605.02003](https://arxiv.org/abs/2605.02003) · [full text HTML](https://arxiv.org/html/2605.02003v1) · [code](https://github.com/ml-lab-htw/RamanBench) · [`raman-data` on PyPI](https://pypi.org/project/raman-data/) · [leaderboard](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench)
- [QM9S dataset on figshare](https://figshare.com/articles/dataset/QM9S_dataset/24235333) · [DetaNet paper, *Nat. Comput. Sci.*](https://www.nature.com/articles/s43588-023-00550-y)
- [QMe14S, arXiv:2501.18876](https://arxiv.org/abs/2501.18876)
- [Computed Raman of inorganic compounds, *Sci Data* 11:105 (2024)](https://www.nature.com/articles/s41597-024-02924-x)
- [ramanchada2](https://github.com/h2020charisma/ramanchada2) · [Open Source for Raman Spectroscopy Data Harmonization, *J. Raman Spectrosc.*](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/jrs.6789)
- [RamanSPy](https://ramanspy.readthedocs.io/) · [RAMANMETRIX docs](https://docs.ramanmetrix.eu)
- [SpectrumWorld, arXiv:2508.01188](https://arxiv.org/abs/2508.01188)
