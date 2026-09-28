# M2 leaderboard 1 — open-set compound retrieval on ROD

**Run 2026-09-11** at git `7898437`, **clean tree**, config hash `9f49c3a118a9338a`. Command:

```
uv run usis data pull rod --all
uv run usis compare configs/rod_open_set_retrieval.yaml --dest data/raw \
    --out experiments/M2-rod-open-set-retrieval
```

Raw record in `comparison.json` (per-item detail included), rendered card in `result_card.md`,
resolved config in `config.yaml`. This is the first run of `usis compare` and the first leaderboard
in the project, so it is also the M2 baseline of the risk–coverage frontier.

Not a preregistered hypothesis test. It is a measurement of where classical library matching
actually sits, made under a protocol fixed in the config before the run. The paired comparison in
§4 **is** post-hoc and is labelled as such.

## 0. Setup

- **Corpus.** 1,131 mirrored ROD records → 1,129 spectra (2 records contradict their own declared
  range and refuse to parse). 16 carry no identity and were dropped. 32 more do not cover the
  200–1200 cm⁻¹ analysis window and were dropped. **1,081 spectra, 428 identities** remain.
- **That subset is not random** and the numbers below are conditional on it. Window coverage
  correlates with instrument, grating and deposit practice.
- **Preprocessing**, recorded on every spectrum: `ResampleToGrid(200, 1200, step 2)` → `SnipBaseline`.
  501 channels, identical axis for every spectrum. No normalisation: all three metrics are
  scale-invariant, so it would have changed no number while adding a step to every chain.
- **Split** `open_set`, seed 20260911, holding out whole identities. Library = **348 identities,
  one spectrum each** (the lowest record id of that identity). Queries = **510 of a present
  identity** (never the library entry itself) + **223 of an identity held out entirely**.
- **Field**: every method the registry holds for `G_compound_retrieval`. Three entered.

## 1. The headline: the frontier sits on the floor

| method | coverage at risk ≤ 1% | selective accuracy | AURC | max coverage |
| --- | --- | --- | --- | --- |
| `baseline.nn_match_sca` | **0.008** [0.001, 0.079] | 0.641 [0.580, 0.701] | 0.240 [0.169, 0.330] | 1.000 |
| `baseline.nn_match` (correlation) | 0.003 [0.000, 0.086] | 0.592 [0.527, 0.654] | 0.270 [0.185, 0.376] | 1.000 |
| `baseline.nn_match_cosine` | 0.001 [0.000, 0.042] | 0.592 [0.527, 0.655] | 0.282 [0.189, 0.395] | 1.000 |

**At the project's stated operating point — >99% correct on accepted cases — classical library
matching covers between 0.1% and 0.8% of this corpus.** Six queries out of 733 for the best of
them. That is the number every later method has to beat, and it is a floor rather than a
disappointment: the point of measuring it now is that "we improved retrieval" becomes a claim with
a denominator.

The curve is worse than the single number suggests. For correlation and cosine, coverage at
risk ≤ 1%, ≤ 5% and ≤ 10% are the **same number**: risk climbs past 10% within the first handful of
accepted items, so between the very top of the confidence ordering and one-in-ten wrong there is no
operating point at all. Only at risk ≤ 20% does coverage become meaningful (0.21–0.27).

The three AURC intervals overlap heavily. **AURC does not order these methods** on this data, and
the leaderboard says so rather than printing a ranking.

## 2. The retrieval works. The open-set decision is what fails.

Mean reciprocal rank of the true entry, over the 510 present-identity queries:

| method | MRR | interval (476 groups) |
| --- | --- | --- |
| `baseline.nn_match` | 0.7451 | [0.676, 0.808] |
| `baseline.nn_match_cosine` | 0.7419 | [0.673, 0.805] |
| `baseline.nn_match_sca` | 0.7419 | [0.673, 0.805] |

The right entry is at or near rank 1 most of the time. Selective accuracy is nonetheless 0.59–0.64,
and the gap is entirely the "is it in the library at all?" gate. Decomposed for correlation:

- of **510 present** queries, 170 wrong — 111 named the **wrong entry**, 59 said "none of these";
- of **223 absent** queries, **129 were given a name** — a hallucinated assignment 58% of the time.

A matcher that ranks well and gates badly is exactly the failure mode `p_none` exists to describe,
and `p_none = 1 − best similarity` does not describe it well enough to be useful at a 1% risk
ceiling. **This is direct, measured support for `PROJECT_NOVELTY.md#N10`** (a calibrated open-set
residual, at M6): the thing standing between a 0.74 MRR and a usable answer is the calibration of
one number.

## 3. Cosine and spectral contrast angle return the identical ranking — as predicted

`baseline.nn_match_sca` is a monotone rescaling of the cosine, so it cannot reorder a candidate
list. The prediction was written into its registration before the run; the data confirms it: **MRR
identical to six decimal places (0.741863)**, and the two disagree on only the decision, never on
the order.

That makes this pair a controlled experiment rather than a comparison. Everything that differs
between them isolates one thing: what the open-set gate does with the similarity **value**, since
`p_none = 1 − best` reads the number and not its rank.

## 4. What SCA actually buys: abstention, on a threshold nobody set

*(Post-hoc. `paired_analysis.py`, not preregistered.)*

The unpaired intervals in §1 overlap, so on their own they resolve nothing. But every method saw
the same 733 queries, so the paired test is the right one — bootstrap the per-item difference over
the same 476 groups, 10,000 resamples:

| comparison | accuracy difference | verdict | items they disagree on |
| --- | --- | --- | --- |
| SCA − correlation | **+0.0491** [+0.0167, +0.0842] | resolved | 114 / 733 |
| SCA − cosine | **+0.0491** [+0.0146, +0.0857] | resolved | 120 / 733 |
| correlation − cosine | +0.0000 [−0.0092, +0.0095] | not resolved | 12 / 733 |

So SCA is better here, and correlation and cosine are indistinguishable. The mechanism is visible
in the failure histogram and it is not retrieval:

| method | declared a library gap | named a wrong entry | said "none" about a present one | hallucinated on an absent one |
| --- | --- | --- | --- | --- |
| correlation | 153 | 111 | 59 | 129 |
| cosine | 142 | 116 | 51 | 132 |
| **SCA** | **324** | **54** | **155** | **54** |

SCA declares a gap on more than twice as many queries. The trade is not one-for-one: it says
"none" about 96 more present substances, but 57 fewer of its remaining positive errors are
wrong-entry names, so positive errors rise by only 39 — and hallucinations on absent substances
fall by 75. Net, **36 fewer errors on a corpus that is 70% positives**.

**The metric did not retrieve better. It abstained more, and on this corpus more abstention happened
to pay.** That is a threshold nobody chose: it falls out of where `arccos` rescaling happens to put
`1 − best`. Read forward, it is an argument that the three-way "which similarity metric" question
the field settles by convention is, under open-set conditions, mostly a question about an
uncalibrated threshold — which is N10's subject, not a property of the similarity.

## 5. The validity-domain machinery earned its place

**Every one of the 2,199 method runs is `RUNNABLE_DEGRADED`.** `requires_library_coverage` is
`UNKNOWN` on all of them, because ROD identifies minerals by name and carries no InChIKey, so
library membership cannot be enumerated and coverage of the queried substance cannot be decided.

Consequence, and it is the intended one: **all three methods are `ESTABLISHED`, and not one
individual result from this leaderboard is surfaceable.** The audit also names what would settle it —
"the library's key list" — rather than leaving the numbers looking clean. `PROJECT_NOVELTY.md#N7`
predicted that machine-checkable validity domains would be a research artifact rather than hygiene;
this is the first run where the prediction pays, and the price it charges is the right one.

Violation rate is 0.00 across the board: nothing was checked and found false. The distinction
between *violated* and *undecided* is doing work here — merging them would have reported a corpus
with thin provenance as a corpus full of broken assumptions.

## 6. What this does NOT show

Repeated from `result_card.md`, because it is the part that gets dropped:

- **Nothing about closed-set accuracy.** Every number is conditional on a library missing, by
  construction, one identity in five.
- **Nothing about instrument transfer.** This split holds out identities. ROD's instrument and lab
  fields are near-coincident anyway, so an instrument-disjoint claim from ROD could not be separated
  from a lab-disjoint one.
- **Nothing calibrated.** Every confidence on the curve derives from `p_none = 1 − best similarity`,
  a stated, uncalibrated convention. Each method's position on the frontier reflects that convention
  as much as the method — §4 is the evidence for exactly that.
- **Nothing about mixtures**, nothing outside 200–1200 cm⁻¹, nothing about the 48 spectra the
  window and the labelling dropped.
- **No causal or mechanistic claim.** These are correlations between spectral shapes.
- **§4 is post-hoc.** It was written after the leaderboard was read. Treat the SCA advantage as a
  hypothesis for a preregistered run on a second corpus, not as a settled result.

## 7. What this changes

- **C6 is met**: one command, N methods from the registry, a machine-readable record with intervals,
  a risk–coverage curve, a failure histogram split by source, an assumption-violation rate and cost.
- **The M2 frontier is on the board**: coverage 0.008 at risk ≤ 1%. Every milestone report from here
  carries this curve.
- **N10 moves from argued to evidenced.** The gap between MRR 0.74 and coverage 0.008 is the size of
  the prize for calibrating the open-set residual.
- **N7 has its first field result**: a whole leaderboard of ESTABLISHED methods, none of whose
  results may be surfaced, for a reason the audit names precisely.
- **A second leaderboard is now the obvious next thing.** ROD cannot support quantification or an
  instrument-disjoint split. RamanBench Bioprocess Analytes can, and `usis compare` does not need to
  change to run it — only a new task module and a config.
