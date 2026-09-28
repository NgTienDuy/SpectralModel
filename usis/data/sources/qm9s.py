"""QM9S: computed IR, Raman and UV-Vis spectra for ~130k QM9 molecules.

Geometries re-optimised and properties computed with Gaussian 16 (B.01) at **B3LYP/def-TZVP**;
frequency analysis and TD-DFT at the same level. Published with *Nature Computational Science*
(DOI ``10.1038/s43588-023-00550-y``), data on figshare under CC BY 4.0.

This is the first **simulated** source in the repo, and simulation changes almost every mapping
decision:

* ``origin`` is ``SIMULATED`` and ``instrument`` is ``None`` — there was no measurement. Anything
  that treats these as experimental spectra is committing failure code ``F13``.
* The provenance that matters is not an instrument but a **level of theory**, so the processing
  chain records the DFT calculation and the line broadening instead of a calibration and a
  baseline correction.
* The axis kind differs between the two modalities even though the numbers coincide. A mode at
  1650 cm⁻¹ appears at *Raman shift* 1650 and at *absolute wavenumber* 1650 in the IR: same value,
  different semantics, and conflating them is ``F01``.

We handle only the broadened spectra (``ir_boraden.csv``, ``raman_boraden.csv`` — the publisher's
spelling of "broadened"). The stick spectra, geometries, Hessians and tensors live in
``qm9s_csv.zip`` / ``qm9s.pt`` and are a separate loader when a task needs them.

**The line broadening is measured here, not quoted.** The readme states the grid but not the
lineshape or its width, so we measured both from the published bytes; see
:data:`BROADENING_FWHM_CM1`.
"""

from __future__ import annotations

import csv
import gzip
import io
from typing import TYPE_CHECKING, Final

import numpy as np

from usis.core.errors import SpectrumValidationError
from usis.core.provenance import content_hash, utc_now
from usis.core.types import (
    AxisKind,
    AxisSpec,
    GroundTruth,
    IntensityKind,
    Modality,
    Origin,
    ProcessingStep,
    Spectrum,
)
from usis.data.registry import (
    DatasetSpec,
    ReplicateStructure,
    Verification,
    register_dataset,
)
from usis.data.sources.base import BulkFile, Source

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

__all__ = [
    "AXIS_MAX_CM1",
    "AXIS_MIN_CM1",
    "AXIS_N_POINTS",
    "BROADENING_FWHM_CM1",
    "LEVEL_OF_THEORY",
    "QM9S_IR_SPEC",
    "QM9S_RAMAN_SPEC",
    "Qm9sSource",
]

#: The broadened grid, stated in the dataset readme and checked against every parsed header.
AXIS_MIN_CM1: Final[float] = 500.0
AXIS_MAX_CM1: Final[float] = 4000.0
AXIS_N_POINTS: Final[int] = 3501

LEVEL_OF_THEORY: Final[dict[str, str]] = {
    "package": "Gaussian 16 (B.01)",
    "functional": "B3LYP",
    "basis_set": "def-TZVP",
    "task": "geometry optimisation + harmonic frequency analysis",
}

#: Lorentzian FWHM of the applied broadening, in cm⁻¹, **measured by us from the published data**.
#:
#: The readme gives the grid ("1cm^-1 a point, total 3501 point") but never the lineshape or its
#: width, and both are needed to compare these spectra with a measured one: a real instrument's
#: response has to be convolved *on top of* whatever is already here. Method: isolated peaks (no
#: neighbour above 10% of the peak within +/-60 cm^-1), half-maximum crossings interpolated
#: linearly between grid points, over the first 15 molecules of each file.
#:
#: * IR - 25 peaks, FWHM 15.085 +/- 0.121 cm^-1
#: * Raman - 21 peaks, FWHM 10.101 +/- 0.083 cm^-1
#:
#: A Lorentzian fits the profile with 2-7% residual against 28-33% for a Gaussian of equal FWHM,
#: so the lineshape is Lorentzian. The spreads are far below one grid step, so the nominal values
#: are almost certainly exactly 15 and 10 cm⁻¹.
BROADENING_FWHM_CM1: Final[dict[str, float]] = {"ir": 15.0, "raman": 10.0}

#: The published broadened CSVs, one per modality. Sizes are the publisher's own figures.
BULK_FILES: Final[dict[str, BulkFile]] = {
    "ir": BulkFile(
        name="ir_boraden.csv",
        url="https://ndownloader.figshare.com/files/43011226",
        published_bytes=8_733_673_295,
        note="broadened IR, 500-4000 cm^-1 at 1 cm^-1; the publisher's spelling of 'broadened'",
    ),
    "raman": BulkFile(
        name="raman_boraden.csv",
        url="https://ndownloader.figshare.com/files/43011415",
        published_bytes=8_784_506_743,
        note="broadened Raman, 500-4000 cm^-1 at 1 cm^-1",
    ),
}

_NOTES = (
    "CC BY 4.0 confirmed via the figshare API (DOI 10.6084/m9.figshare.24235333.v3). "
    "B3LYP/def-TZVP "
    "with Gaussian 16 B.01. Broadened grid 500-4000 cm^-1 at 1 cm^-1, 3501 points, unnormalised. "
    "The broadened CSVs carry a molecule index but no SMILES; SMILES is in qm9s_csv.zip. Line "
    "broadening measured by us: Lorentzian, {fwhm:g} cm^-1 FWHM. The publisher asks that the "
    "article DOI be cited rather than the figshare one."
)

QM9S_IR_SPEC: Final[DatasetSpec] = register_dataset(
    DatasetSpec(
        dataset_id="qm9s-ir",
        name="QM9S computed IR spectra (broadened)",
        modality=Modality.IR_TRANSMISSION,
        origin=Origin.SIMULATED,
        licence="CC-BY-4.0",
        homepage="https://doi.org/10.6084/m9.figshare.24235333",
        doi="10.1038/s43588-023-00550-y",
        citation="Nat. Comput. Sci. (2023)",
        verification=Verification.VERIFIED,
        verified_on="2026-09-03",
        # Simulated: one spectrum per molecule, no physical sample and so no replicates.
        replicate_structure=ReplicateStructure.NONE,
        notes=_NOTES.format(fwhm=BROADENING_FWHM_CM1["ir"]),
    )
)

QM9S_RAMAN_SPEC: Final[DatasetSpec] = register_dataset(
    DatasetSpec(
        dataset_id="qm9s-raman",
        name="QM9S computed Raman spectra (broadened)",
        modality=Modality.RAMAN,
        origin=Origin.SIMULATED,
        licence="CC-BY-4.0",
        homepage="https://doi.org/10.6084/m9.figshare.24235333",
        doi="10.1038/s43588-023-00550-y",
        citation="Nat. Comput. Sci. (2023)",
        verification=Verification.VERIFIED,
        verified_on="2026-09-03",
        # Simulated: one spectrum per molecule, no physical sample and so no replicates.
        replicate_structure=ReplicateStructure.NONE,
        notes=_NOTES.format(fwhm=BROADENING_FWHM_CM1["raman"]),
    )
)


class Qm9sSource(Source):
    """Loader for one of the QM9S broadened-spectrum CSVs.

    Args:
        kind: ``"ir"`` or ``"raman"``. The two files share a schema and differ in what the axis
            *means*, so the kind is chosen explicitly rather than sniffed from a filename.
        smiles_by_index: optional QM9S index → SMILES mapping. The broadened CSVs do not carry
            structures, so without this the label is the molecule's index and says so; supplying it
            (from ``qm9s_csv.zip``) upgrades the label rather than changing anything else.
    """

    def __init__(self, kind: str, smiles_by_index: Mapping[int, str] | None = None) -> None:
        if kind not in BROADENING_FWHM_CM1:
            raise ValueError(f"kind must be 'ir' or 'raman', got {kind!r}")
        self._kind = kind
        self._smiles = dict(smiles_by_index or {})

    @property
    def kind(self) -> str:
        """Which of the two broadened files this instance reads."""
        return self._kind

    @property
    def spec(self) -> DatasetSpec:
        """The dataset manifest entry for this modality."""
        return QM9S_IR_SPEC if self._kind == "ir" else QM9S_RAMAN_SPEC

    @property
    def axis_kind(self) -> AxisKind:
        """The axis semantics for this modality.

        IR absorption is plotted against the absolute wavenumber of the absorbed photon; Raman
        against the shift from the excitation line. The two coincide numerically for a given
        vibrational mode and are not the same quantity.
        """
        return AxisKind.ABSOLUTE_WAVENUMBER_CM1 if self._kind == "ir" else AxisKind.RAMAN_SHIFT_CM1

    def parse(
        self,
        payload: str,
        *,
        url: str | None = None,
        sha256: str = "",
        retrieved_at: str | None = None,
    ) -> Iterator[Spectrum]:
        """Parse a broadened-spectrum CSV, yielding one spectrum per molecule."""
        yield from self._parse_rows(
            io.StringIO(payload), url=url, sha256=sha256, retrieved_at=retrieved_at
        )

    def parse_file(self, path: Path, url: str | None = None) -> Iterator[Spectrum]:
        """Stream from disk, transparently handling ``.gz``.

        Overridden because the published files are 8.7 GB each; holding one in a string to satisfy
        the base class would be absurd.
        """
        digest = content_hash(path)
        stamp = utc_now()
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from self._parse_rows(handle, url=url, sha256=digest, retrieved_at=stamp)

    def _processing(self) -> tuple[ProcessingStep, ...]:
        """The simulation chain: what produced this spectrum, in the order it was applied.

        For a computed spectrum the honest provenance is the level of theory and the broadening,
        not a calibration and a baseline. The broadening width is our measurement, and the step
        records that so nobody later mistakes it for a published figure.
        """
        return (
            ProcessingStep(
                name="simulate.dft_frequency",
                params={**LEVEL_OF_THEORY, "declared_by": "qm9s"},
                version="source-declared",
                timestamp="unknown",
            ),
            ProcessingStep(
                name="simulate.broadening",
                params={
                    "lineshape": "lorentzian",
                    "fwhm_cm1": BROADENING_FWHM_CM1[self._kind],
                    "measured_by": "usis 2026-09-03",
                    "source_states_width": False,
                },
                version="measured",
                timestamp="unknown",
            ),
        )

    def _parse_rows(
        self,
        handle: io.TextIOBase,
        *,
        url: str | None,
        sha256: str,
        retrieved_at: str | None,
    ) -> Iterator[Spectrum]:
        """Shared row loop for the string and file paths."""
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise SpectrumValidationError(f"qm9s-{self._kind}: the CSV is empty") from None

        axis = self._axis_from_header(header)
        spec = self.spec
        source = spec.source_record(retrieved_at=retrieved_at or utc_now(), url=url, sha256=sha256)
        processing = self._processing()

        for line_number, row in enumerate(reader):
            if not row:
                continue
            if len(row) != axis.n_points + 1:
                raise SpectrumValidationError(
                    f"qm9s-{self._kind} line {line_number}: has {len(row)} fields but the header "
                    f"declares an index column plus {axis.n_points} wavenumbers"
                )
            try:
                index = int(row[0])
            except ValueError as exc:
                raise SpectrumValidationError(
                    f"qm9s-{self._kind} line {line_number}: first column {row[0]!r} is not a "
                    "molecule index"
                ) from exc
            yield Spectrum(
                spectrum_id=f"qm9s-{self._kind}:{index}",
                modality=spec.modality,
                origin=Origin.SIMULATED,
                axis=axis,
                intensity=np.array(row[1:], dtype=np.float64),
                # Unnormalised computed intensities: IR in km/mol and Raman activity in A^4/amu
                # before broadening. Neither is absorbance or counts, and calling them either
                # would imply a calibration that does not exist.
                intensity_kind=IntensityKind.ARBITRARY,
                # No physical sample exists. The molecule is the grouping unit, which makes
                # sample_disjoint and molecule_disjoint identical here by construction.
                sample_id=f"qm9s:{index}",
                source=source,
                instrument=None,
                acquisition=None,
                processing=processing,
                label=self._label(index),
            )

    def _label(self, index: int) -> GroundTruth:
        """Label the molecule, stating plainly when the structure is not in this file."""
        smiles = self._smiles.get(index)
        if smiles is None:
            return GroundTruth(
                name=f"QM9S molecule {index}",
                annotation_note=(
                    "SMILES is not present in the broadened CSV; it is line 0 column 2 of the "
                    "molecule's file in qm9s_csv.zip"
                ),
            )
        return GroundTruth(name=f"QM9S molecule {index}", smiles=smiles)

    def _axis_from_header(self, header: list[str]) -> AxisSpec:
        """Read the wavenumber axis from the header, checking it against the published grid."""
        if not header or header[0].strip() != "":
            raise SpectrumValidationError(
                f"qm9s-{self._kind}: the first header cell must be the empty index column; this "
                "does not look like a QM9S broadened CSV"
            )
        try:
            values = np.array(header[1:], dtype=np.float64)
        except ValueError as exc:
            raise SpectrumValidationError(
                f"qm9s-{self._kind}: the header's wavenumber columns are not all numeric ({exc})"
            ) from exc
        if values.size != AXIS_N_POINTS:
            raise SpectrumValidationError(
                f"qm9s-{self._kind}: expected {AXIS_N_POINTS} wavenumber columns, "
                f"found {values.size}"
            )
        span = (float(values.min()), float(values.max()))
        if span != (AXIS_MIN_CM1, AXIS_MAX_CM1):
            raise SpectrumValidationError(
                f"qm9s-{self._kind}: axis spans {span}, but the published range is "
                f"({AXIS_MIN_CM1}, {AXIS_MAX_CM1})"
            )
        # Unlike the instrument-measured sources, resolution_fwhm IS set here, and it describes the
        # applied Lorentzian broadening rather than an instrument response. That is the useful
        # reading: nothing narrower than this survives in the data, so a sim-to-real comparison
        # must convolve on top of it. The number is ours, and the processing chain says so.
        return AxisSpec.from_values(
            self.axis_kind, values, resolution_fwhm=BROADENING_FWHM_CM1[self._kind]
        )

    def bulk_files(self) -> tuple[BulkFile, ...]:
        """The one published CSV for this modality.

        URLs and sizes read from the figshare API for article 24235333 on 2026-09-03 and recorded
        here rather than resolved at fetch time: a URL the code discovers can change what it points
        at between two runs without the change being visible in a diff. Re-derive with
        ``curl https://api.figshare.com/v2/articles/24235333``.
        """
        return (BULK_FILES[self._kind],)

    def fetch(self, record_id: str, dest: Path) -> Path:
        """Not implemented: each broadened file is one ~8.7 GB CSV, not per-record downloads."""
        raise NotImplementedError(
            "qm9s publishes one large CSV per modality (~8.7 GB); download it from "
            f"{self.spec.homepage} and point 'usis data verify' at it. Per-record fetch is not "
            "meaningful for this dataset."
        )
