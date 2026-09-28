# Revised plan, M0 → M4

Brief §15.5 asks for a realistic plan for **one person on one 24 GB GPU**, including what we propose
to cut. Written 2026-09-02, after the critique in `docs/adr/0001-brief-critique.md`.

Timeboxes assume a single engineer at roughly full time and **already include** the cuts in §S12 of
the critique. They are working estimates, not commitments; the point of writing them down is to make
slippage visible rather than to be right.

---

## M0 — Foundations · 1.5 weeks · *mostly delivered*

| Work | State |
| --- | --- |
| Repo scaffold, packaging, CI, lint/typecheck/test gates | **done** |
| `usis/core/types.py` — the §7 contracts with invariants enforced at construction | **done** |
| `usis/core/units.py` — axis and intensity conversions, round-trip tested | **done** |
| Provenance, run context, replayable JSONL trace | **done** |
| Licence vocabulary and vendoring gate | **done** |
| Physical-law checkers (isotope law, mutual exclusion) | **done** |
| CLI: implemented verbs work, planned verbs refuse and name their milestone | **done** |
| Brief critique, dataset matrix, prior art, novelty ledger, open questions | **done** |
| P0 knowledge notes | **done** (`docs/knowledge/*.md`) |
| Hydra config skeleton | **deferred to M1** — no experiment exists to configure yet, and an unused config system rots |

**Remaining M0 work: none blocking.** The stack decisions are recorded in `PROJECT_DECISIONS.md`.

---

## M1 — Data and the measurement model · 6–8 weeks

The forward model is the highest-leverage object in the repo and deserves most of this budget.

| # | Work | Estimate | Verified by |
| --- | --- | --- | --- |
| 1.1 | ~~Q1 lookup~~ — **done 2026-09-03**, `docs/knowledge/datasets.md` §5. Residual: read RamanBench §A.14.2 for the Bioprocess Analytes specifics (analyte count, the eight instruments, licence) | 0.5 day | The audit table exists; M2's headline criterion is unblocked in restricted form |
| 1.2 | Loaders: **ROD, *Sci Data* API set, QM9S (IR + Raman) and RamanBench done 2026-09-03** — 97 datasets in the manifest, 116 offline tests. M1's "≥3 experimental sources" is met (ROD, *Sci Data*, RamanBench) and "≥1 simulated" by QM9S | **done** | `usis data verify` re-parses every record; fixtures committed, tests offline |
| 1.3 | Read 8 unresolved RamanBench licence strings (20 datasets blocked from vendoring), and pull the Bioprocess Analytes arrays to confirm the shapes the catalogue implies | 3 days | The blocked count falls, or each string is recorded as genuinely unusable |
| 1.4 | **Forward measurement model — done 2026-09-03.** Resampling, resolution convolution (Gaussian/Lorentzian/Voigt composition), axis miscalibration, intensity response, baseline, fluorescence, dark current, Poisson, readout, cosmic rays, saturation | **done** | Each degradation applies independently at controllable severity (E0/T4); broadening a real QM9S spectrum to a stated target reproduces it within 1% when measured back |
| 1.5 | Splits: `random`, `sample_disjoint`, `molecule_disjoint`, `scaffold_disjoint`, `instrument_disjoint`, `lab_disjoint`, **`excitation_disjoint`**, `sim2real`, `resolution_shift`, `open_set` — **done 2026-09-03** | **done** | `split_kind` is a required argument with no default, and a split that cannot be built honestly raises with the reason |
| 1.6 | Leakage checker in CI — **done 2026-09-03** | **done** | Catches a planted replicate leak, and catches a *random* split over the real *Sci Data* fixtures putting scans of one vial on both sides |
| 1.7 | **Config + result card writer — done 2026-09-03.** OmegaConf loading with `name`/`seed`/`split_kind` required and no defaults; bootstrap CIs and IQM; result cards that refuse to omit their limitations | **done** | A card cannot be written without a "what this does NOT show" section, and no metric can be reported as a bare point estimate |

**Cut from M1:** vendor-format parsers beyond what the chosen sources need. Depend on `ramanchada2`
(MIT, verified) for vendor I/O and calibration rather than writing our own.

**Outcome.** 1.4's engineering landed in a day; the research half of it did not. "Statistically
comparable to real spectra" is acceptance criterion B7 and is **still open** — see
`PROJECT_ACCEPTANCE.md`. That split is the honest one: the operators are built and validated
against measured widths, but nobody has yet asked whether a degraded simulation *looks like* a
measurement under named statistics.

---

## M2 — Method layer, comparison harness, classical baselines · 8–10 weeks

The spine. Everything after this is a plugin.

| # | Work | Estimate | Verified by |
| --- | --- | --- | --- |
| 2.1 | **`Method` protocol, machine-readable assumptions, applicability checking — done 2026-09-03.** Thirteen assumption classes with a four-valued status; one applicability policy (`PROJECT_DECISIONS.md` D13); `execute` as the only entry point | **done** | `execute` refuses with `valid_snr_above=50 - estimated SNR 3.60, required > 50`. A method cannot report its own audit, and cannot emit a causal claim |
| 2.2 | **Registry with maturity levels — done 2026-09-03.** Every level's evidence requirement enforced at registration (D14); `pkgutil` discovery; `usis methods` prints the registry | **done** | A method written into a package outside the repository is registered with no other edit. The pair exists: `audit.saturation` (established) and `shape.broadening_origin` (hypothesised) |
| 2.3 | **Shadow mode with the ADR-0001 §S2 fix — done 2026-09-03.** The fix is in the record type: `reference_answer` and `incumbent_answer` are separate fields and cannot be conflated | **done** | 500 records of perfect agreement with the incumbent score nothing: `0 independently-answered runs out of 500`. No function anywhere changes a maturity level |
| 2.4 | **Composable, provenance-emitting preprocessing pipelines + the pipeline search — done 2026-09-03.** Nine steps, pure numpy and deterministic; nine defensible pipelines including an identity arm; N8's sensitivity readout | **done** | The chain is replayed *from the record*, after a JSON round trip, and compared bit-for-bit: **720/720 on 80 real ROD spectra x 9 pipelines**. Tampering with one recorded parameter fails; a foreign step refuses to replay |
| 2.5 | **Classical baselines: PCA, PLS(R/-DA), MCR-ALS, NNMF, nearest-neighbour matching — algorithms done 2026-09-03**, six registered methods over four task families, all pure numpy. Leaderboard outstanding | **half done** | Every baseline verified against *planted* ground truth (components at r > 0.99, PLS R² > 0.999, retrieval rank 1, and p_none > 0.5 for an absent query); NNLS checked against exhaustive active-set search to 4.4e-16. Leaderboard across the split matrix still needs 2.6 and a dataset decision |
| 2.6 | `usis compare` harness: metric ± CI, risk–coverage curves, failure-code histogram, assumption-violation rates, cost | 2 weeks | One command, N methods, a machine-readable comparison record |
| 2.7 | **E0a as a standing gate**: T8 permutation control, T2 isotope law, T3 selection rules — run in CI from here on | 1 week | Shuffled labels give chance performance; law violations are reported per method |

**Landed already, ahead of the rest of M2:** the hypothesised method registered under 2.2 had its
first control run immediately, and the control went against it (`PROJECT_NOVELTY.md#N11`) — the
statistic is confounded with band spacing as strongly as with the line shape it was meant to read.
That is a result, not a delay: it cost an afternoon and it stopped an unsupported threshold from
being hard-coded. Expect the same from every `hypothesized` registration.

**Ship here:** `usis audit` — sections 1 and 2 of the Sample Report (measurement audit +
preprocessing account) as a real, usable tool. The brief says ship value early; this is the earliest
point where something is genuinely useful and honest, and it does not wait for M8.

**Kill signal (from the brief, kept):** if classical baselines match deep models everywhere on honest
splits, publish that and re-scope. RamanBench's finding that tabular models beat domain-specific
architectures makes this *more* likely than the brief assumed, not less.

---

## M3 — Representation benchmark · 5–6 weeks

Raw / patches / peak tokens / wavelet / learned VQ / hybrid, one protocol, scored on information
loss, robustness to each M1 degradation, retrieval quality, mixture suitability, interpretability and
cost.

**Cut:** the learned-VQ arm goes last and is dropped if M3 overruns — it is the most expensive arm
and the least likely to be decisive given that a *tabular* model already beats spectral architectures
on RamanBench.

**Accept:** the tokenisation question is answered with evidence and written up **even if it
contradicts the peak-token intuition**, which on current evidence it may well.

---

## M4 — E0b, the falsification battery · 6–8 weeks

Pre-registered in `experiments/E0/preregistration.md` **before** running. T1 compositional
generalisation, T4 measurement-shift degradation curves, T5 sim→real, T6 attribution specificity,
T7 degeneracy honesty. T2, T3 and T8 already ran as the E0a gate from M2 and are reported here as a
time series rather than a single measurement — which is strictly more informative.

**Accept:** every probe has a pre-registered threshold and a stated interpretation, written before
the run. A clean negative result is the intended outcome, not a failure.

---

## Totals

| Milestone | Estimate |
| --- | --- |
| M0 | 1.5 weeks (done) |
| M1 | 6–8 weeks |
| M2 | 8–10 weeks |
| M3 | 5–6 weeks |
| M4 | 6–8 weeks |
| **M0 → E0b** | **26–33 weeks ≈ 6.5–8 months** |

The brief hopes for "a clean negative result in the first six months". Six months is reachable only
by cutting M3 to two arms (raw + peaks) and treating the representation question as deferred. **That
trade is worth taking**: E0's answer changes what M3 should even measure, so running E0b earlier with
fewer representations is better science than running M3 exhaustively first.

**Recommended variant:** M0 → M1 → M2 → **M4 (E0b) with two representations** → M3 (full), which
reaches the falsification experiment in ~5.5 months.

## What is cut, consolidated

From `docs/adr/0001-brief-critique.md` §S12: the knowledge-graph/ontology layer before M7; the HTML
report renderer until sections 1–3 are stable; portfolio machinery before three methods exist;
generating a new DFT corpus at all; multimodal fusion before M5 works on one modality.

**Not cut:** the `NMR_1H` and `MS` members of `Modality`. Removing them was proposed and rejected
(`PROJECT_DECISIONS.md` D6) — they stay, and the architectural separation `KEYWORD_MAP.md` §C7 calls
for happens when the first adapter is written.
