# Experiments

One directory per experiment, numbered and **immutable once run**. Each contains:

```
E<n>-<slug>/
  preregistration.md   # written and committed BEFORE the run, where the design calls for it
  config.yaml          # the exact committed config
  result_card.md       # what it shows AND what it does not show
  run_context.json     # git SHA, seeds, environment, config hash
```

Rules:

- A result card without a "what this does **not** show" section is incomplete.
- Never a bare point estimate: bootstrap CI or IQM, plus the failure-code histogram.
- Protocol changes need an ADR in `docs/adr/` **and** a re-run of affected baselines.
- Negative results are first-class. Write them up, then delete the dead code.

**E0** is the falsification battery (brief §13). It is split per `docs/adr/0001-brief-critique.md`
§S8:

- **E0a** — probes T8 (permutation control), T2 (isotope law), T3 (selection rules). A *standing CI
  gate* from M2 onward, not a one-off. Regression gates do not need pre-registration.
- **E0b** — the full battery at M4. Thresholds and interpretation rules are pre-registered in
  `E0b/preregistration.md` **before** the first run.
