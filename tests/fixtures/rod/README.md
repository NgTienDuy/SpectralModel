# ROD fixtures

Two records from the [Raman Open Database](https://solsa.crystallography.net/rod/), retrieved
**2026-09-03**, committed **verbatim and unmodified**.

| File | Compound | Instrument | Lab | Excitation | Resolution |
| --- | --- | --- | --- | --- | --- |
| `8100020.rod` | Chambersite (B7 Cl Mn3 O13) | Thermo Scientific Nicolet Almega XR | Arizona University | 532 nm | 4 cm⁻¹ |
| `1000001.rod` | Hematite (Fe2 O3) | HORIBA Jobin Yvon T64000 | IMMM Maine university | 514 nm | 3 cm⁻¹ |

The pair is chosen for maximum contrast: two instruments, two labs, two excitation wavelengths,
different upstream sources (`R060425.rod` from RRUFF vs `YE0002.rod`), and one entry carrying a
journal DOI while the other does not. A parser that handles both handles the shape of the database.

**Why verbatim rather than trimmed.** ROD dedicates all its data to the public domain under CC0, so
the vendoring gate permits committing them; at ~45 KB and ~24 KB these are small enough that
truncating the spectrum loop would buy nothing and would cost the ability to check a parse against
a byte-exact original.

Provenance: `https://solsa.crystallography.net/rod/<code>.rod`. Checksums are asserted in
`tests/test_rod.py`, so an accidental edit fails the suite rather than silently changing what the
parser is tested against.
