# P0 — Inverse problems, ill-posedness and abstention

Covers KEYWORD_MAP **H**. Owns: `usis/models/`, `usis/uncertainty/`, `usis/compare/`.

## The mathematical core of the project

The forward map is well defined: a structure, in an environment, measured by an instrument, produces
a spectrum. The inverse is not. `P(structure | spectrum)` — never `structure = f(spectrum)`.

Ill-posedness here is not a technicality. It has named, physical causes:

| Non-identifiability class | Cause |
| --- | --- |
| `ISOMER_DEGENERATE` | positional isomers whose spectra differ by less than the noise floor |
| `POLYMORPH_DEGENERATE` | same molecule, different packing, differences below resolution |
| `RESOLUTION_LIMITED` | the distinguishing bands are merged by the instrument response |
| `LIBRARY_GAP` | the true structure is not in the candidate set at all |
| `MIXTURE_UNDERDETERMINED` | more components than the data can support (rotational ambiguity) |
| `FLUORESCENCE_DOMINATED` | the signal-carrying region is buried |
| `SNR_LIMITED` / `RANGE_LIMITED` | the information is below the noise floor, or outside the measured window |

**No method recovers information the photons did not carry.** Naming which of these applies is a
successful output, not a failure — it is section 8 of the Sample Report, and it is what makes the
next-measurement recommendation possible.

## Regularisation, and what it costs

Every inverse method chooses a regulariser, and every regulariser is a prior:

- **Tikhonov** `min ‖Ax − b‖² + λ‖Lx‖²` — smoothness. Cheap, and it biases sharp features away.
- **Sparsity** (`ℓ₁`) — few active components. Excellent for mixtures, and it will happily return a
  sparse *wrong* answer when the true mixture is not sparse.
- **Non-negativity** — physical, and it is the constraint doing most of the work in MCR-ALS.
- **A learned prior** — a generative model over structures. The most powerful and the least
  inspectable; its failure mode is confident hallucination of plausible molecules (`F10`, `F17`).

The choice of `λ` is not a hyperparameter to tune on the test set. It sets the bias–variance trade-off
of the *answer*, and tuning it against the metric being reported is a leak.

## Bayesian framing, and where it becomes computable

`p(s | d) ∝ p(d | s)·p(s)`. The likelihood `p(d | s)` is exactly what the M1 forward model provides —
but only up to its own misspecification, which is why forward-model validation is not optional
housekeeping.

When the likelihood is intractable but the simulator is available, this is the textbook setting for
**simulation-based inference** (neural posterior estimation, normalising flows, amortised inference).
It composes naturally with the agent: posterior → expected information gain → next measurement.
→ `PROJECT_NOVELTY.md` N2.

## The decision layer

The interesting question is not "what is the answer" but "**should I answer at all**".

- **Selective prediction**: accept when confidence exceeds a threshold; report the whole
  risk–coverage curve, never one operating point.
- **Conformal prediction** gives distribution-free coverage for *set-valued* outputs, which is the
  natural shape of a candidate list. **Its guarantee requires exchangeability between calibration and
  test** — which `instrument_disjoint` and `sim2real` splits deliberately break. Expect coverage loss
  under exactly the shifts we care about; measure it, or use a shift-aware variant (weighted or
  group-conditional conformal). See `docs/adr/0001-brief-critique.md` §S4.
- **Open-set**: `p_none = P(true structure absent from the library | spectrum, library, instrument)`.
  Mandated by the type system; **meaningless until calibrated against held-out open-set data.** Until
  then it is reported as `UNCALIBRATED`. → N10, open question Q4.

## The bound that makes accuracy interpretable

Given a candidate set, a prior over it and a validated forward model, the **Bayes-optimal error rate**
is computable by simulation: push each candidate through the forward model and measure pairwise
distinguishability under the measured noise. That number separates *"our method is weak"* from
*"this measurement cannot answer the question"*.

It is **conditional** — on the candidate set, on the prior, and on the forward model — and saying so
is what distinguishes it from the brief's original, ill-posed "information ceiling".
→ `docs/adr/0001-brief-critique.md` §S1, `PROJECT_NOVELTY.md` N5.

## Implications for our code

- `BeliefState` cannot represent a bare answer: candidates carry posteriors, `p_none` is required,
  and `sum(posterior) + p_none == 1` is asserted at construction.
- `Candidate` carries **conflicting** evidence as well as supporting evidence. A candidate list with
  no conflicting evidence anywhere is a sign the evidence chain is decorative.
- `NonIdentifiability` is an enum, not a free-text note, so that abstention reasons are countable
  across a benchmark.
- `Action` with expected information gain and expected cost exists in the core types from M0, so that
  "what would resolve this" is a structured output long before the agent that generates it exists.

## Gap statement

Ill-posedness is well understood in inverse-problem theory and essentially absent from the
spectroscopy-AI literature, which reports Recall@1 over closed candidate sets. The three
contributions that follow — a calibrated `p_none`, a named non-identifiability class, and a
conditional Bayes-error bound — are the project's core scientific claims, and all three are testable
before any large model is trained.
