# P0 — Chemometrics

Covers KEYWORD_MAP **G**. Owns: `usis/chemometrics/`.

## Position

Chemometrics is a foundation, not an obsolete predecessor. It supplies the mandatory baselines, and
MCR-ALS remains the reference method for mixture resolution. **A deep model that cannot beat MCR-ALS
on an instrument-disjoint split has not earned its parameters.**

External evidence sharpens this: RamanBench (arXiv:2605.02003) reports that *tabular* foundation
models with no spectral inductive bias beat domain-specific Raman architectures across 74 datasets,
with no method generalising everywhere. Whatever else that means, it is not evidence that classical
methods are behind.

## The methods, and what each assumes

| Method | Model | Load-bearing assumption |
| --- | --- | --- |
| **PCA** | `X = TPᵀ + E` | variance is the interesting direction — often false when the dominant variance is baseline or instrument |
| **PLS / PLS-DA** | latent variables maximising covariance with `y` | linear relation between spectrum and response; overfits fast with many latent variables |
| **CLS / ECLS** | `X = CSᵀ` with `S` **known** | pure-component spectra are known *and measured under the same conditions* |
| **MCR-ALS** | `D = CSᵀ + E`, `C, S ≥ 0` | non-negativity plus constraints resolve the factorisation — see the ambiguity below |
| **NNMF** | non-negative factorisation | same ambiguity, fewer chemical constraints |
| **Nearest-neighbour matching** | similarity to a library | the answer is in the library — precisely what an open-set setting denies |

## The ambiguity that defines the field

For any invertible `T`,

```
D = C Sᵀ = (C T)(T⁻¹ Sᵀ)
```

so bilinear decomposition is **not unique**. This is *rotational ambiguity*, and it is the reason
MCR-ALS is a constrained optimisation rather than a factorisation: non-negativity, unimodality,
closure and known-spectrum constraints shrink the feasible set but generally do not collapse it to a
point. Reporting an MCR-ALS solution without acknowledging the feasible band is reporting one member
of a family as if it were the answer — a mixture-ambiguity failure (`F09`).

This is not a defect of MCR-ALS. It is the ill-posedness of the problem, made visible. A deep model
solving the same problem faces the same ambiguity; it is simply less honest about it.

## Beer–Lambert and closure as hard constraints

- **Beer–Lambert**: `A = ε·c·l`, linear in absorbance, and only within a validity domain (dilute,
  non-scattering, no chemical interaction, no detector saturation). Deviation at high concentration is
  physics, not noise.
- **Closure**: mixture fractions sum to 1. Enforced at construction in `MixtureHypothesis`, not
  learned as a soft penalty.
- **Non-negativity**: concentrations and resolved spectra are non-negative. A negative resolved
  component is a diagnostic, not a rounding artifact.

## Calibration transfer

Piecewise direct standardisation and its successors exist because a model trained on instrument A
routinely fails on instrument B. That the sub-field is mature is itself evidence for how large the
instrument effect is — and an argument for `PROJECT_NOVELTY.md` N3 (infer the instrument rather than
correct it away).

## Two things measured while implementing these, 2026-09-03

**Closure costs a rank, and a scree plot on a closed system is off by one.** Mixtures whose
fractions sum to one lie in an affine subspace one dimension smaller than the number of components.
Measured on three planted components: median PCA reconstruction residual falls from 0.363 at one
component to 0.018 at two, and then barely moves at three. Anyone counting components from where
the residual collapses will count `n - 1` substances on a closed system, every time. Verified in
`tests/test_chemometrics.py::test_pca_recovers_a_planted_subspace_one_rank_short_because_of_closure`.

**PLS does not leave an interferent lying in a late component, so the obvious overfit diagnostic
does not fire.** The intuitive check - "flag a latent variable explaining a lot of `X` and none of
`y`" - almost never triggers, because NIPALS orders components by covariance with the response and
absorbs the interferent into the early ones. Planted interferents at up to eight times the analyte
signal were absorbed into components 1-2 in every configuration tried. The detectable overfit is
the other shape: components that contribute *nothing*, e.g. 0.02% of the response variance for the
third latent variable on a two-direction problem. `PlsModel.capacity_warning` reports that instead,
and was rewritten after the first version was measured and found inert.

## Implications for our code

- The five mandatory baselines are registered as `ESTABLISHED` methods with **explicit machine-readable
  assumptions** (`assumes_linear_mixing`, `requires_baseline_corrected`, `requires_library_coverage`,
  `valid_snr_above`). Their applicability checks are what makes N7 measurable. **Done 2026-09-03**:
  six registered methods over four task families - `baseline.pca_knn`, `baseline.pls_r`,
  `baseline.pls_da`, `baseline.mcr_als`, `baseline.nnmf`, `baseline.nn_match` - each verified
  against planted ground truth, each carrying its declared domain. `baseline.pls_r` and
  `baseline.mcr_als` are `RUNNABLE_DEGRADED` on *every* run, because `assumes_linear_mixing` is
  `NOT_CHECKABLE` from one invocation; that is the intended behaviour rather than a defect.
- Hard constraints are inference-time validators (`usis/knowledge/rules/laws.py`), never loss terms.
- Any mixture result must be able to report the feasible band, not only the point solution.

## Gap statement

The rate at which published spectroscopy results apply a method **outside its declared validity
domain** is widely asserted and, as far as we can find, never measured. The comparison harness
measures it for free. → `PROJECT_NOVELTY.md` N7.
