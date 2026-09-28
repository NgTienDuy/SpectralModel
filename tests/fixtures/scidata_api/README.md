# Sci Data API-compounds fixture

A **row subset** of `raman_spectra_api_compounds.csv` from
[figshare 27931131](https://doi.org/10.6084/m9.figshare.27931131) (CC BY 4.0), retrieved
**2026-09-03**, published with *Sci Data* 12:498 (2025), DOI `10.1038/s41597-025-04848-6`.

The original is 193 MB and holds 3,510 rows. This fixture keeps the **real header** and **six real
rows**, gzipped:

| Rows | Compound | CSV label as published |
| --- | --- | --- |
| 2 | 1,3-Dimethyl-2-imidazolidinone | `"1,3-Dimethyl-2-imidazolidinone"` |
| 2 | 2-Propanol | `2 - Propanol` |
| 2 | 2,2-Dimethoxypropane | `"2, 2 - Dimethoxy Propane"` |

Those three label forms are the point of the selection: the CSV and the published product table
disagree on spacing and quoting, and all three must join. Two rows per compound also let the tests
check that repeated scans of one vial share a `sample_id`.

**No value was altered.** The subset was taken by selecting whole lines from the original — the
header and every intensity are the publisher's bytes. Rows came from byte offsets 0, ~6.2 MB and
~12.6 MB of the original file, which is why the row indices here do not match the original's.

Gzipped because 6 rows × 3,276 float columns is 366 KB plain and 123 KB compressed, and the loader
handles `.gz` transparently anyway — which it must, since the published file is 193 MB.
