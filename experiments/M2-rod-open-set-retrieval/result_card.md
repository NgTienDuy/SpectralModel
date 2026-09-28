# rod-open-set-retrieval

## What this comparison shows

- How three classical spectral-library similarity metrics compare, under one protocol, on an open-set retrieval task built from real deposited Raman spectra: 348 mineral identities in the library, queries of both present and held-out identities.
- Where the risk-coverage frontier of classical library matching actually sits on this corpus, which is the M2 baseline every later method has to beat.
- That the difference between these metrics under open-set conditions is a difference in the open-set decision, not in retrieval ranking: the spectral contrast angle is monotone in the cosine and returns an identical candidate order.

## What it does NOT show

- Nothing about accuracy on a closed-set task. Every number here is conditional on a library that is missing, by construction, one identity in five.
- Nothing about instrument transfer. ROD's instrument and lab fields are near-coincident, so an instrument-disjoint claim cannot be separated from a lab-disjoint one, and this split is neither - it holds out identities.
- Nothing calibrated. Every confidence on the risk-coverage curve derives from p_none = 1 - best similarity, a stated and UNCALIBRATED convention (PROJECT_NOVELTY.md#N10). A method's position on the frontier reflects that convention as much as the method.
- Nothing about mixtures, and nothing about substances outside the 200-1200 cm-1 window or outside the covering subset of ROD.
- No causal or mechanistic claim of any kind. These are correlations between spectral shapes.

## Setup

- task kind: `G_compound_retrieval`
- datasets: `rod`
- items: 733 presented, 0 unscorable
- git SHA: `789843700e4e01a9e12701f56ee5b235b00beee3` (dirty: `false`), seed `20260911`
- intervals: percentile bootstrap, 2000 resamples

- split: `open_set` grouped on **molecule identity**, train 858 / test 223, seed `20260911`

**Caveats carried by this split:**

- open_set holds out whole identities: the test side contains molecules absent from training, so a closed-set metric over it is meaningless. Score p_none, not accuracy.

## Leaderboard

| method | cov@risk<=0.01 | selective accuracy | coverage | AURC | violation rate | errors |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline.nn_match_sca@1.0.0` | 0.008 [0.001, 0.079] | 0.641 [0.580, 0.701] | 1.000 [1.000, 1.000]! | 0.240 [0.169, 0.330] | 0.00 | 0 |
| `baseline.nn_match@1.0.0` | 0.003 [0.000, 0.086] | 0.592 [0.527, 0.654] | 1.000 [1.000, 1.000]! | 0.270 [0.185, 0.376] | 0.00 | 0 |
| `baseline.nn_match_cosine@1.0.0` | 0.001 [0.000, 0.042] | 0.592 [0.527, 0.655] | 1.000 [1.000, 1.000]! | 0.282 [0.189, 0.395] | 0.00 | 0 |

## Failure codes

Split by where the code came from: a method *declares* a code about its own run, a benchmark *scores* one against the reference. The same code can arrive from both and mean different things - `F12` declared is a predicted library gap, `F12` scored is an observed miss - so they are never summed into one bar.

- `baseline.nn_match@1.0.0`
  - declared by the method: `F12_retrieval_miss` x153
  - scored by the benchmark: `F12_retrieval_miss` x170, `F10_hallucinated_assignment` x129
- `baseline.nn_match_cosine@1.0.0`
  - declared by the method: `F12_retrieval_miss` x142
  - scored by the benchmark: `F12_retrieval_miss` x167, `F10_hallucinated_assignment` x132
- `baseline.nn_match_sca@1.0.0`
  - declared by the method: `F12_retrieval_miss` x324
  - scored by the benchmark: `F12_retrieval_miss` x209, `F10_hallucinated_assignment` x54

## Assumption audit (N7)

- `baseline.nn_match@1.0.0`: 0% clean, 0% violated, 100% undecided over 733 run(s)
  - undecided: `requires_library_coverage (unknown)` x733
- `baseline.nn_match_cosine@1.0.0`: 0% clean, 0% violated, 100% undecided over 733 run(s)
  - undecided: `requires_library_coverage (unknown)` x733
- `baseline.nn_match_sca@1.0.0`: 0% clean, 0% violated, 100% undecided over 733 run(s)
  - undecided: `requires_library_coverage (unknown)` x733

## Notes

- Corpus: 1131 mirrored record(s) of rod, 1129 spectra parsed, 2 record(s) unparseable ({'SpectrumValidationError': 2}).
- 16 spectrum/spectra carried no identity and were dropped: a retrieval answer about an unlabelled sample cannot be right or wrong.
- 32 spectrum/spectra did not cover the analysis window and were dropped. That subset is NOT random - window coverage correlates with instrument, grating and deposit practice - so the benchmark is over the covering subset and any claim from it is conditional on that.
- Preprocessing applied to every spectrum, recorded on each: ResampleToGrid(high_cm1=1200.0,low_cm1=200.0,step_cm1=2.0) > SnipBaseline(decreasing=True,iterations=40)
- Library holds 348 identities, one spectrum each. Queries: 510 of a present identity, 223 of an identity held out of the library.
- 'None of these' is scored as a claim with a truth value, not as a refusal to answer. Selective behaviour is measured separately, by the risk-coverage curve over the confidence each method put on the decision it took.
- p_none underlying every decision is an UNCALIBRATED convention (1 - best similarity), so the position of a method on the risk-coverage curve reflects that convention as much as the method. Calibrating it is PROJECT_NOVELTY.md#N10, at M6.
- reciprocal_rank is truncated at the candidate list each method returns. Methods configured to return different numbers of candidates are NOT comparable on that column; they remain comparable on accuracy and on the curve.

