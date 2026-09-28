# QM9S fixtures

Row subsets of `ir_boraden.csv` and `raman_boraden.csv` from
[figshare 24235333](https://doi.org/10.6084/m9.figshare.24235333) (CC BY 4.0), retrieved
**2026-09-03**. Published with *Nature Computational Science*, DOI `10.1038/s43588-023-00550-y` —
which is the DOI the publisher asks be cited, rather than the figshare one.

The originals are ~8.7 GB each and hold ~130k rows. Each fixture keeps the **real header** and the
**first three real rows** (QM9S molecule indices 0, 1, 2), gzipped from ~237 KB to ~103 KB.

| File | Modality | Axis kind | Points |
| --- | --- | --- | --- |
| `qm9s_ir_subset.csv.gz` | computed IR | absolute wavenumber | 3,501 (500–4000 cm⁻¹) |
| `qm9s_raman_subset.csv.gz` | computed Raman | Raman shift | 3,501 (500–4000 cm⁻¹) |

The same three molecule indices appear in both files, so the tests can check that the IR and Raman
spectra of one molecule share a `sample_id` — the pairing a joint-modality experiment needs and
that a split must not break.

**No value was altered.** Whole lines were selected from the originals; the header and every
intensity are the publisher's bytes.

## The broadening was measured from these bytes

The dataset readme states the grid ("1cm^-1 a point, total 3501 point") but never the lineshape or
its width, and both are needed before these spectra can be compared with a measured one. Measured
over the first 15 molecules of each file, on isolated peaks (no neighbour above 10% of the peak
within ±60 cm⁻¹), with half-maximum crossings interpolated between grid points:

| | Peaks | FWHM | Lorentzian residual | Gaussian residual |
| --- | --- | --- | --- | --- |
| IR | 25 | 15.085 ± 0.121 cm⁻¹ | 2–4% | 28% |
| Raman | 21 | 10.101 ± 0.083 cm⁻¹ | 4–7% | 32% |

So: **Lorentzian, 15 cm⁻¹ for IR and 10 cm⁻¹ for Raman.** The spreads are well under one grid step,
so the nominal values are almost certainly exactly 15 and 10. `tests/test_qm9s.py` re-derives the
IR figure from the committed fixture, so the constant in the loader cannot drift away from the data
it was measured on.
