"""The mandatory classical baselines: PCA, PLS, MCR-ALS, NNMF, nearest-neighbour matching.

A foundation, not an obsolete predecessor. No deep model is reported without these under an
identical split and metric. The algorithms live here as plain, tested numerics; each is wrapped as
a registered `Method` with a declared validity domain in `usis/methods/builtin/`, one file per
method.

Everything is pure numpy. A *mandatory* baseline that needs an optional install is a baseline that
gets skipped, and a comparison missing its baselines is the failure mode the whole package exists
to prevent.

| Module | What it holds |
| --- | --- |
| `matrix` | Stacking spectra into a matrix, and where it must refuse |
| `decomposition` | PCA, NNMF, MCR-ALS, exact NNLS, and the rotational-ambiguity bound |
| `regression` | PLS by NIPALS, with its capacity accounting exposed |
| `matching` | Library matching, and the open-set mass it is forced to state |

Milestone: M2 (C5).
"""
