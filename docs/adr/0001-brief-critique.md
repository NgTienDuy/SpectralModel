# ADR-0001 — Critique of the project brief

**Status:** accepted (as a critique; the brief is not amended by it)
**Date:** 2026-09-02
**Author:** first-session review, as required by brief §15.1

## Context

The brief (`YEUCAU.md`) asks, in §0, for explicit disagreement: *"When you disagree with a decision
in this brief, say so explicitly and argue the case. Silent compliance is a failure mode."* This ADR
is that argument. It does not change the brief; it records where the brief is wrong, over-scoped, or
internally inconsistent, so that a later session can decide with the reasoning visible.

Findings are ranked by severity: **S1 is the one that most threatens the project's central claim.**

Overall assessment first, because it should not get lost in the list: **the brief is unusually good.**
Its central moves — ill-posedness as a type, abstention as a rewarded output, adversarial splits by
default, mandatory classical baselines, the method registry with maturity levels — are correct and
are not what this literature normally does. The criticisms below are about *specific load-bearing
claims that will not survive contact with data*, not about the direction.

---

## S1 — The per-spectrum information ceiling is not well-posed as stated

**The claim.** Novelty candidate 5: estimate "how much information a given measurement can possibly
yield", and thereby distinguish method-limited from measurement-limited failure. The brief calls
this what makes "94% accurate" interpretable.

**The problem.** There is no such quantity as *the* information content of a measurement. Mutual
information `I(structure; spectrum)` is defined only relative to (a) a hypothesis space, (b) a prior
over it, and (c) a noise/forward model. Change the candidate library and the ceiling changes.
Assume a uniform prior over "all molecules" and you have assumed something that does not exist. As
written, the deliverable is a quantity nobody can compute because nobody can define it.

**Worse:** whatever we do compute is only as good as the forward model. A misspecified instrument
model produces a ceiling estimate that is confidently wrong in *either* direction, and the failure is
invisible — the number looks like physics and is actually our own modelling error reflected back.

**Proposed reformulation** — narrower, computable, and still novel:

> Given a candidate set `C`, a prior over it, and a calibrated instrument forward model `f`, estimate
> the **Bayes-optimal error rate** for distinguishing members of `C` under `f`'s noise. Report it as
> a *conditional, model-relative* bound with the forward model's own validation error attached.

This is computable today: simulate each candidate, push it through `f`, and compute pairwise
distinguishability under the measured noise. It answers the question that actually matters
operationally — *could any method have separated these two candidates on this measurement?* — and it
is honest about being conditional on `C` and `f`.

**Consequence if ignored:** the project's headline interpretability claim rests on an undefined
quantity, and the first reviewer who asks "with respect to what prior?" ends the conversation.

---

## S2 — Shadow-mode scoring is circular as specified

**The claim.** §6.1: hypothesised methods "run on every analysis, are logged and scored against the
accepted answer", accumulating evidence toward promotion.

**The problem.** In routine use there is no ground truth. The "accepted answer" *is* the output of
the established portfolio. So a shadow method is being scored on agreement with the incumbent, and a
shadow method that is **better** than the incumbent scores **worse**. This is the standard
"gold standard is the model" trap, and it inverts the promotion gate exactly when the gate matters.

**Fix.**

1. Shadow scoring counts **only** on runs with an independent answer: labelled benchmark data, a
   re-measurement that resolved the case, or an orthogonal modality that settled it.
2. On unlabelled production runs, a shadow method's disagreement with the portfolio is recorded as a
   **flag for review**, never as a score.
3. The promotion gate reads only the independently-answered subset, and states its size. A promotion
   on `n = 30` labelled cases is not a promotion.

**Severity:** high, because it is a silent failure. The system would keep working and keep promoting
the wrong things.

---

## S3 — Method disagreement is not a calibration-free uncertainty estimate without a control

**The claim.** Novelty candidate 6: run methods with disjoint assumptions; agreement means
confidence, disagreement localises an assumption violation.

**The problem.** The methods are not independent — they share the *input*. A wrong axis unit, an
over-subtracted baseline, a saturated detector or a mislabelled intensity convention corrupts every
downstream method identically. They then **agree**, confidently and wrongly. Agreement measures
"these methods share a preprocessing ancestor" at least as much as it measures truth. The most
dangerous errors in this domain (F01, F02, F05) are exactly the correlated ones.

**Fix.** Compute disagreement across whole **pipelines** including preprocessing choice, not across
final-stage methods on a common preprocessed input — the brief already makes preprocessing a search
space, so this is available. And the novelty claim needs a control experiment: inject known
corruptions and show that the disagreement signal actually rises. Without that control this is an
intuition, not a result.

---

## S4 — The M6 conformal acceptance criterion asks for something the theory forbids

**The claim.** M6 accept: "conformal coverage holds within tolerance under sim→real shift".

**The problem.** Split-conformal coverage guarantees require exchangeability between calibration and
test data. `instrument_disjoint`, `lab_disjoint` and `sim2real` splits are *designed* to break
exchangeability. Standard conformal prediction is therefore expected to lose coverage under exactly
the shifts we care about — not as a bug, but by construction.

**Fix.** Either

- change the criterion to *"measure and report the coverage loss under each shift, with the observed
  gap"* (a good result, and an honest one); or
- adopt a shift-aware variant — weighted/covariate-shift conformal, or Mondrian (group-conditional)
  conformal with the instrument as the group — and then validate the importance weights themselves,
  which is the actual hard part.

The brief's own parenthetical ("or we document exactly where it breaks") already hints at this. Make
that the primary criterion, not the fallback.

---

## S5 — `p_none` is mandated but never given an estimator or a calibration protocol

**The claim.** Invariant 1: every identity output carries `p_none`, the open-set residual mass, and
`sum(posterior) + p_none == 1` is asserted in code. (It is; see `usis/core/types.py`.)

**The problem.** The brief specifies the *field*, not its *meaning* or its *validation*. `p_none`
needs a stated semantics — we propose
`p_none = P(the true structure is absent from the candidate library | spectrum, library, instrument)`
— and it needs held-out open-set data where the answer is genuinely absent, or the number is
decorative. A softmax leftover renamed `p_none` satisfies the assertion and means nothing.

**Fix.** Add to M6: `p_none` is reported as `UNCALIBRATED` until an `open_set` split with measured
detection performance exists. The type system should carry that flag, not just the number.

---

## S6 — The data required for the headline split may not exist in the open

**This is the largest practical risk in the plan and the brief does not name it.**

The brief makes `instrument_disjoint` a default reporting split and asks (open question 8) how much
published deep-learning advantage survives it. Our verified dataset matrix
(`docs/knowledge/datasets.md`) says: the confirmed, openly-licensed *experimental* Raman corpus is on
the order of 10³–10⁴ spectra, largely minerals (RRUFF, restrictively licensed) or a few dozen
solvents (3,510 spectra / 32 compounds). An instrument-disjoint split over **molecules** requires the
same molecules measured on several named instruments. **We have not verified that any open source
provides this.**

**Consequences.**

- The M2 acceptance criterion "state how much of the reported deep-learning gains survive an
  instrument-disjoint split" may be unanswerable from open data.
- If so, the honest options are: (a) `instrument_disjoint` over *classification* tasks using
  RamanBench's 74 datasets; (b) simulate the instrument shift with our own forward model and label
  the result as simulated shift, not measured shift; (c) a small in-house measurement campaign;
  (d) report the *absence of the data* as a finding about the field's evidence base — which it is.
- Option (d) is genuinely publishable and costs nothing. It should be an explicit deliverable, not a
  consolation prize.

**Amended 2026-09-03 — partly withdrawn, and the replacement risk is sharper.** Two audits since
this was written change the picture:

- **RamanBench's Bioprocess Analytes collection** is eight sibling datasets, one per spectrometer,
  measuring the same three analytes, all CC BY 4.0 (`docs/knowledge/datasets.md` §5). The gap there
  is protocol choice, not missing data: the benchmark evaluates it with random 80/20 splits.
- **ROD supports an instrument-disjoint split for mineral phase identification.** The exhaustive
  audit over all 1,129 readable records finds six instrument models across seven laboratories, the
  two largest holding 806 and 152 records. The earlier claim that ROD had "two entries on a second
  instrument" came from a 240-record scan of three guessed id blocks and was an artifact of the
  sampling frame.

**What survives, and it is the part that actually bites:** *instrument and laboratory are almost
perfectly confounded in ROD*, and no open corpus we have verified pairs a **structure library**
across instruments. So the split is constructible, and every claim built on it has to say
"instrument-or-lab", not "instrument". The risk is no longer "the data does not exist"; it is
"the data exists and cannot separate the two effects, and reporting it as though it could would be
the same overclaim as calling an excitation split an instrument split."

**Action:** resolving this is the gating item for M2 planning. Tracked as Q1 in
`docs/knowledge/open_questions.md`.

**Outcome (2026-09-03) — audited, and the finding is sharper than the concern.** The data is not
absent: RamanBench's Bioprocess Analytes collection provides *"the same analytes measured across
eight different instruments"*. But RamanBench evaluates with *"fixed 80/20 train/test splits over 3
different seeds"* and explicitly leaves the cross-instrument study as an "opportunity". So the
field's gap here is **protocol choice, not data scarcity** — which makes option (a) above the
selected fallback and makes it cheap. Two limits stand: the collection supports
classification/regression, not `spectrum_to_structure`, for which the data genuinely is absent; and
RRUFF's multi-excitation records are the *same instrument*, so they give an excitation-disjoint
probe that must never be reported as instrument shift. Audit table:
`docs/knowledge/datasets.md` §5. Revised claim: `PROJECT_NOVELTY.md` N9.

---

## S7 — Novelty hypothesis 1 ("evaluation as the contribution") has been substantially overtaken

The brief asserts that "a cross-instrument, cross-lab, sim→real, physics-law-probing benchmark for
spectral models does not exist in a usable form." Since the brief was written, three independent
efforts have appeared and are verified in `docs/knowledge/prior_art.md`: **RamanBench**
(74 datasets, 325,668 spectra, 28 models, live leaderboard), **SpectrumBench** (14 tasks, >10
spectrum types), and a *Digital Discovery* benchmarking study with explicit distribution-shift
testing.

The claim is not dead, but it must be narrowed to what remains uncovered:

- **physics-law probes** (isotope shift, mutual exclusion) — found in none of them;
- **abstention and open-set scoring** — found in none of them;
- **instrument-/lab-disjoint splits over the same molecules** — claimed in none of them.

Being three benchmarks late to "we built a benchmark" and first to "we built the probes" is a much
stronger position than the brief's framing. `PROJECT_NOVELTY.md` N1 is written accordingly.

One further observation from RamanBench that should worry us productively: **tabular foundation
models with no spectral inductive bias beat the domain-specific architectures.** If true and general,
that is evidence for shortcut structure in the tasks, and it is an argument for our probe battery,
not against it.

---

## S8 — Milestone ordering delays the cheapest and most decisive experiments

E0 sits at M4, after the representation benchmark (M3). But three of its eight probes need almost
nothing:

- **T8 permutation control** (shuffled labels; above-chance ⇒ leakage) needs only a data pipeline. It
  belongs at **M2**, as a standing CI gate, run on every pipeline change — which is what the brief
  itself says ("run this every time the data pipeline changes") while scheduling it at M4.
- **T2 isotope law** and **T3 selection rules** need a forward model and a structure, both of which
  exist at M1. They can be run against *any* method, including the classical baselines.

Running M3 before knowing whether the pipeline leaks means potentially optimising a representation on
a leaking benchmark for six weeks.

**Fix.** Split E0 into **E0a** (T8, T2, T3 — a standing gate from M2 onward) and **E0b** (the full
battery, M4, with pre-registration). The pre-registration discipline applies to E0b; E0a is a
regression test, and regression tests do not need pre-registration.

---

## S9 — The "one curve" framing of risk–coverage understates the reporting burden

The risk–coverage frontier is the right headline. But it is not one curve: it is a curve per
(task × split kind × method × milestone), and at low coverage it is estimated from few accepted
cases, so it is *noisy exactly where the interesting claims live* ("99% correct on accepted cases"
is a claim about the left end of the curve, where n is smallest).

**Fix, all cheap:**

- pre-register the coverage grid points at which risk is reported, so nobody picks the flattering one;
- enforce a minimum accepted-count per grid point and report `n` beside every risk value;
- use stratified bootstrap CIs (already mandated) *on the curve*, not only on the headline metric;
- report a **separate** curve for open-set cases, where the correct behaviour is to abstain, so that
  correct abstention is not scored as lost coverage.

---

## S10 — Internal inconsistencies and type-level gaps in §7

Small individually, but §7 is the contract every module speaks, so they are worth fixing at the
source.

| # | Issue | What we did |
| --- | --- | --- |
| a | §3 invariant 5 mandates typed eV; `AxisKind` in §7 has no eV member | Added `ENERGY_EV`, documented as a deliberate extension |
| b | `Spectrum.label: GroundTruth \| None` — `GroundTruth` is never defined | Defined it, including `label_confidence` and an annotation note, so label noise (F16) is representable rather than assumed away |
| c | `ProcessingStep.params: dict` inside a frozen dataclass — unhashable, and a mutable field in an "immutable provenance" record | Typed as `Mapping`; documented that hashing is unsupported; arrays are made read-only |
| d | The normalisation assertion has no stated tolerance | `POSTERIOR_SUM_TOL = 1e-6`, documented with the reason for the value |
| e | `Modality` includes `NMR_1H` and `MS`, but `KEYWORD_MAP.md` §C7 says MS is "architecturally separate" and must sit behind its own adapter | Members kept; we recommended removing them until an adapter exists, on the grounds that an enum member is a promise and this one is unfunded. **Outcome (2026-09-03): the user rejected the recommendation** — keep the members, handle the architectural separation when the first adapter is written. Recorded as `PROJECT_DECISIONS.md` D6, and as a note on the `Modality` docstring so it is read where a loader would be added |
| f | §7 has no representation for the measurement verdict, non-identifiability class, or failure code, all of which the report contract requires | Added `MeasurementVerdict`, `NonIdentifiability`, `FailureCode`, `TaskKind`, `Maturity` |

---

## S11 — Two attributions could not be verified

- **RSR (Resolution Success Rate)** and **COD (Counterfactual Outcome Deviation)** are presented as
  "adapted from the world-model literature". arXiv:2604.22748 was fetched and its abstract does not
  name either metric. The concepts are sound and we can define them ourselves; **the citation must
  not be made until the full text is read.** Marked `UNVERIFIED` in `prior_art.md` §6.
- The RRUFF licence terms quoted in circulation ("a single copy … up to 20 paper copies") were not
  found on any RRUFF page fetched this session. Treated as `TERMS-OF-USE`, non-redistributable, with
  the action to close it recorded.

---

## S12 — What we propose to cut

Scope, not quality. Each of these can return with evidence.

| Cut | Why | Return condition |
| --- | --- | --- |
| Knowledge graph / ontology layer (`usis/knowledge/graph`, `ontology`) before M7 | Months of work, no measured benefit to any M0–M6 acceptance criterion. Retrieval of *reference spectra* does not need a KG. | A retrieval experiment shows entity linking beats vector search on our task |
| Self-contained HTML report renderer | The report content is the hard part; rendering is not | Sections 1–3 are stable |
| `usis compare --methods all` portfolio machinery before three methods exist | Cannot design a portfolio abstraction against one implementation | Three registered methods, at least two with disjoint assumptions |
| Generating a new DFT corpus | Not a single-GPU activity, and QM9S/QMe14S already exist under CC-BY | The published corpora are shown to be insufficient for a specific question |
| Multimodal fusion (`U` in the keyword map) | Raman+IR joint conditioning is in the input contract but not in any M0–M6 criterion | M5 works on one modality |

One further cut was proposed and **rejected by the user**: removing the `NMR_1H` and `MS` members of
`Modality`. They stay; the architectural separation §C7 calls for happens when the first adapter is
written. See `PROJECT_DECISIONS.md` D6.

---

## S13 — Where the brief is right and the instinct to "improve" it should be resisted

Recorded so that a future session does not relitigate settled ground:

- **Method, not Model, as the central abstraction.** Correct, and unusual.
- **Maturity levels with `REFUTED` retained *with its evidence*.** This is the single cheapest good
  idea in the document.
- **Units as types.** Non-negotiable, and already caught real design questions while implementing
  `usis/core/units.py` (percent transmittance, Raman shift vs absolute wavenumber, undefined
  counts→absorbance).
- **"Band positions are conditional priors, not truth."** Correct, and enforced: `constants.py`
  contains no band assignments, only calibration anchors.
- **Classical baselines as mandatory.** RamanBench's finding that non-spectral models win supports
  this more strongly than the brief could have known.
- **Negative results as first-class, and delete the dead code.** Keep.

---

## Decision

1. Adopt the S1 reformulation of the information ceiling for `PROJECT_NOVELTY.md` N5.
2. Adopt the S2 shadow-mode fix before any shadow scoring is implemented (M2).
3. Restate the M6 conformal criterion per S4 before M6 planning.
4. Treat S6 (data availability for instrument-disjoint splits) as the gating question for M2 and
   resolve it during M1.
5. Split E0 into E0a (standing gate from M2) and E0b (M4 battery) per S8.
6. Apply the S10 type fixes now (done), and take the S12 cuts as the working scope.

## Consequences

No results exist yet, so nothing needs re-running. The cuts in S12 shorten the path to E0 by an
estimated 6–8 weeks. S1 and S2 change what two of the six novelty candidates actually claim, which is
better discovered now than in a rebuttal.

## Alternatives considered

- **Accept the brief as written and discover these in the data.** Rejected: S2 and S6 are the kind of
  problem that costs a quarter before it becomes visible.
- **Rewrite the brief.** Rejected: the brief is the user's document and its hypotheses are worth
  keeping as stated so that this critique remains falsifiable against them.
