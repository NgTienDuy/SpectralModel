"""Open-source Raman spectra of chemical compounds for API development (*Sci Data* 12:498, 2025).

3,510 spectra of 32 solvents and reagents, published as one wide CSV: 3,276 wavenumber columns
(150-3425 cm^-1) plus a ``label`` column naming the compound.

Three facts about this dataset shape the loader, and all three come from the paper rather than from
the file, which is why the file alone is not enough to load it honestly:

1. **One instrument.** An Endress+Hauser (formerly Kaiser Optical Systems) Raman Rxn2 analyser with
   an Rxn-10 non-contact probe and a 10x objective, 785 nm excitation. The CSV carries no
   instrument column at all, so every spectrum here shares one ``instrument_id`` — and this dataset
   therefore cannot support an instrument-disjoint split.
2. **Preprocessing was applied, despite the data being commonly called raw.** The paper states that
   automatic pre-treatment via the iC Raman software "included dark noise subtraction, cosmic ray
   filtering, and intensity correction. No other spectral preprocessing was applied after
   collection." All three are recorded in the processing chain; leaving them out would assert that
   nothing was applied, which is false.
3. **One physical sample per compound.** Each compound was sampled once into a 4 mL amber vial, and
   the ~100-145 spectra per compound are repeated manual scans with the probe moved laterally and
   rotationally. So ``sample_id`` is the compound: a random split would put scans of the *same
   vial* on both sides, and ``sample_disjoint`` collapses onto ``molecule_disjoint`` here.

Source: https://doi.org/10.1038/s41597-025-04848-6 · data https://doi.org/10.6084/m9.figshare.27931131
"""

from __future__ import annotations

import csv
import gzip
import io
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from usis.core.errors import SpectrumValidationError
from usis.core.provenance import content_hash, utc_now
from usis.core.types import (
    AcquisitionParams,
    AxisKind,
    AxisSpec,
    GroundTruth,
    InstrumentSpec,
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
    from collections.abc import Iterator
    from pathlib import Path

__all__ = [
    "COMPOUNDS",
    "INSTRUMENT",
    "SCIDATA_API_SPEC",
    "SciDataApiSource",
    "normalise_label",
]

SCIDATA_API_SPEC: Final[DatasetSpec] = register_dataset(
    DatasetSpec(
        dataset_id="scidata-api-raman",
        name="Open-source Raman spectra of chemical compounds for API development",
        modality=Modality.RAMAN,
        origin=Origin.EXPERIMENTAL,
        licence="CC-BY-4.0",
        homepage="https://doi.org/10.6084/m9.figshare.27931131",
        doi="10.1038/s41597-025-04848-6",
        citation="Sci Data 12:498 (2025)",
        verification=Verification.VERIFIED,
        verified_on="2026-09-03",
        # One 4 mL vial per compound, so every scan of a compound is a replicate of one sample.
        replicate_structure=ReplicateStructure.KNOWN,
        notes=(
            "CC BY confirmed via the figshare API. 3,510 spectra of 32 compounds on a single "
            "instrument (Endress+Hauser Raman Rxn2, 785 nm). Per-compound exposure, pixel fill and "
            "sample counts transcribed programmatically from API_Product_Information.xlsx; the "
            "sample counts sum to 3,510, matching the paper. Cannot support an instrument-disjoint "
            "split. See docs/knowledge/datasets.md section 5."
        ),
    )
)

#: The one instrument. Recorded as a constant because the CSV carries no instrument column, and a
#: split key that is absent from the data has to come from somewhere traceable.
INSTRUMENT: Final[InstrumentSpec] = InstrumentSpec(
    instrument_id="scidata-api:endress-hauser__raman-rxn2__rxn-10-probe",
    manufacturer="Endress+Hauser (formerly Kaiser Optical Systems)",
    model="Raman Rxn2 analyser, Rxn-10 non-contact probe",
    excitation_nm=785.0,
    objective="10x",
    lab_id="scidata-api:single-lab",
)

#: The axis the publisher used, stated in the paper and checked against every parsed header.
AXIS_MIN_CM1: Final[float] = 150.0
AXIS_MAX_CM1: Final[float] = 3425.0
AXIS_N_POINTS: Final[int] = 3276

#: Instrument pre-treatment the paper declares, in the order the software applies it.
_DECLARED_PRETREATMENT: Final[tuple[str, ...]] = (
    "dark_noise.subtraction",
    "despike.cosmic_ray_filter",
    "response.intensity_correction",
)


@dataclass(frozen=True)
class _Compound:
    """Per-compound acquisition metadata from the published product-information table."""

    name: str
    formula: str
    exposure_s: float
    pixel_fill_percent: float
    n_samples: int


def normalise_label(label: str) -> str:
    """Normalise a compound label for joining the CSV to the product table.

    The two disagree on spacing: the CSV writes ``2 - Propanol`` and ``"2, 2 - Dimethoxy Propane"``
    where the table writes ``2-Propanol`` and ``2,2-Dimethoxypropane``. Stripping whitespace and
    lowercasing reconciles them; anything subtler would be guessing.
    """
    return re.sub(r"\s+", "", label).strip('"').lower()


#: Transcribed programmatically from ``API_Product_Information.xlsx`` (CC BY, figshare
#: 27931131) on 2026-09-03 rather than by hand, so there is no transcription error to find. The
#: ``n_samples`` column sums to 3,510, which is the spectrum count the paper reports — an
#: independent check that the table is complete.
COMPOUNDS: Final[dict[str, _Compound]] = {
    "1,3-dimethyl-2-imidazolidinone": _Compound(
        "1,3-Dimethyl-2-imidazolidinone", "C5H10N2O", 3.0, 50.0, 109
    ),
    "2,2-dimethoxypropane": _Compound("2,2-Dimethoxypropane", "C5H12O2", 5.0, 60.0, 107),
    "2-propanol": _Compound("2-Propanol", "C3H8O", 5.0, 50.0, 109),
    "4-methyl-2-pentanone": _Compound("4-Methyl-2-pentanone", "C6H12O", 15.0, 50.0, 100),
    "aceticacid": _Compound("Acetic acid", "CH3COOH", 7.5, 60.0, 113),
    "acetone": _Compound("Acetone", "C3H6O", 7.0, 73.0, 120),
    "acetonitrile": _Compound("Acetonitrile", "C2H3N", 10.0, 50.0, 104),
    "benzaldehyde": _Compound("Benzaldehyde", "C7H6O", 2.0, 68.0, 121),
    "benzylbromide": _Compound("Benzyl bromide", "C7H7Br", 2.0, 73.0, 112),
    "butylacetate": _Compound("Butyl acetate", "C6H12O2", 20.0, 51.0, 106),
    "chloroform": _Compound("Chloroform", "CHCl3", 2.0, 70.0, 122),
    "cyclohexane": _Compound("Cyclohexane", "C6H12", 2.0, 70.0, 145),
    "dichloromethane": _Compound("Dichloromethane", "CH2Cl2", 5.0, 55.0, 111),
    "diethylamine": _Compound("Diethylamine", "C4H11N", 15.0, 60.0, 107),
    "diethyleneglycol": _Compound("Diethylene glycol", "C4H10O3", 30.0, 50.0, 105),
    "diethylmalonate": _Compound("Diethyl malonate", "C7H12O4", 30.0, 63.0, 100),
    "dimethylsulfoxide": _Compound("Dimethyl sulfoxide", "C2H6OS", 2.0, 60.0, 105),
    "ethanol": _Compound("Ethanol", "C2H6O", 10.0, 56.0, 109),
    "ethylacetate": _Compound("Ethyl acetate", "C4H8O2", 10.0, 50.0, 110),
    "ethyleneglycol": _Compound("Ethylene glycol", "C2H6O2", 15.0, 70.0, 101),
    "formicacid": _Compound("Formic acid", "CH2O2", 20.0, 55.0, 110),
    "isobutylamine": _Compound("Isobutylamine", "C4H11N", 20.0, 60.0, 104),
    "methanol": _Compound("Methanol", "CH3OH", 15.0, 50.0, 101),
    "methylisobutylketone": _Compound("Methyl isobutyl ketone", "C6H12O", 20.0, 55.0, 105),
    "n,n-dimethylformamide": _Compound("N,N-Dimethylformamide", "C3H7NO", 10.0, 40.0, 105),
    "n-heptane": _Compound("n-Heptane", "C7H16", 45.0, 68.0, 103),
    "n-hexane": _Compound("n-Hexane", "C6H14", 30.0, 55.0, 102),
    "propylacetate": _Compound("Propyl acetate", "C5H10O2", 20.0, 63.0, 107),
    "tert-butanol": _Compound("tert-Butanol", "C4H10O", 3.0, 51.0, 113),
    "tert-butylmethylether": _Compound("tert-Butyl methyl ether", "C5H12O", 3.0, 60.0, 117),
    "tetrahydrofuran": _Compound("Tetrahydrofuran", "C4H8O", 5.0, 70.0, 100),
    "toluene": _Compound("Toluene", "C7H8", 2.0, 67.0, 127),
}


def _pretreatment_steps() -> tuple[ProcessingStep, ...]:
    """The instrument pre-treatment the paper declares, as a recorded processing chain."""
    return tuple(
        ProcessingStep(
            name=name,
            params={"declared_by": "scidata-api", "software": "iC Raman 4.1", "params": "unknown"},
            version="source-declared",
            timestamp="unknown",
        )
        for name in _DECLARED_PRETREATMENT
    )


class SciDataApiSource(Source):
    """Loader for the wide CSV of API-development Raman spectra."""

    @property
    def spec(self) -> DatasetSpec:
        """The dataset manifest entry."""
        return SCIDATA_API_SPEC

    def parse(
        self,
        payload: str,
        *,
        url: str | None = None,
        sha256: str = "",
        retrieved_at: str | None = None,
    ) -> Iterator[Spectrum]:
        """Parse the wide CSV, yielding one spectrum per row in file order."""
        yield from self._parse_rows(
            io.StringIO(payload), url=url, sha256=sha256, retrieved_at=retrieved_at
        )

    def parse_file(self, path: Path, url: str | None = None) -> Iterator[Spectrum]:
        """Stream the CSV from disk, transparently handling ``.gz``.

        Overridden rather than inherited because the published file is 193 MB: reading it into one
        string to satisfy the base class would be a needless memory spike.
        """
        digest = content_hash(path)
        stamp = utc_now()
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from self._parse_rows(handle, url=url, sha256=digest, retrieved_at=stamp)

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
            raise SpectrumValidationError("scidata-api: the CSV is empty") from None

        axis = self._axis_from_header(header)
        source = SCIDATA_API_SPEC.source_record(
            retrieved_at=retrieved_at or utc_now(),
            url=url,
            sha256=sha256,
        )
        processing = _pretreatment_steps()

        for row_index, row in enumerate(reader):
            if not row:
                continue
            if len(row) != axis.n_points + 1:
                raise SpectrumValidationError(
                    f"scidata-api row {row_index}: has {len(row)} fields but the header declares "
                    f"{axis.n_points} wavenumbers plus a label"
                )
            label = row[-1].strip()
            if not label:
                raise SpectrumValidationError(f"scidata-api row {row_index}: empty label")
            compound = COMPOUNDS.get(normalise_label(label))
            yield Spectrum(
                spectrum_id=f"scidata-api:{row_index:05d}",
                modality=Modality.RAMAN,
                origin=Origin.EXPERIMENTAL,
                axis=axis,
                intensity=np.array(row[:-1], dtype=np.float64),
                # The paper reports intensity correction but no absolute radiometric calibration,
                # so these are corrected counts on an arbitrary scale, not photon counts.
                intensity_kind=IntensityKind.ARBITRARY,
                # One 4 mL vial per compound: every scan of a compound shares a physical sample.
                sample_id=f"scidata-api:{normalise_label(label)}",
                source=source,
                instrument=INSTRUMENT,
                acquisition=None
                if compound is None
                else AcquisitionParams(exposure_s=compound.exposure_s, accumulations=1),
                processing=processing,
                label=self._ground_truth(label, compound),
            )

    @staticmethod
    def _ground_truth(label: str, compound: _Compound | None) -> GroundTruth:
        """Build the label, preferring the published compound name over the CSV's spacing."""
        if compound is None:
            # An unjoined label still yields a usable spectrum, with the gap visible in the note
            # rather than silently dropped.
            return GroundTruth(
                name=label.strip('"'),
                annotation_note="not matched to the published product table",
            )
        return GroundTruth(
            name=compound.name,
            annotation_note=f"formula={compound.formula}; csv_label={label}",
        )

    @staticmethod
    def _axis_from_header(header: list[str]) -> AxisSpec:
        """Read the wavenumber axis out of the header row, checking it against the paper."""
        if not header or header[-1].strip().lower() != "label":
            raise SpectrumValidationError(
                "scidata-api: the header's last column must be 'label'; this does not look like "
                "the published matrix CSV"
            )
        try:
            values = np.array(header[:-1], dtype=np.float64)
        except ValueError as exc:
            raise SpectrumValidationError(
                f"scidata-api: the header's wavenumber columns are not all numeric ({exc})"
            ) from exc
        if values.size != AXIS_N_POINTS:
            raise SpectrumValidationError(
                f"scidata-api: expected {AXIS_N_POINTS} wavenumber columns, found {values.size}"
            )
        span = (float(values.min()), float(values.max()))
        if span != (AXIS_MIN_CM1, AXIS_MAX_CM1):
            raise SpectrumValidationError(
                f"scidata-api: axis spans {span}, but the published range is "
                f"({AXIS_MIN_CM1}, {AXIS_MAX_CM1})"
            )
        # resolution_fwhm is deliberately left unset. The paper calls 1 cm^-1 the "average spectral
        # resolution", but 3276 points across 150-3425 cm^-1 is a sample spacing of exactly
        # 1 cm^-1, so that figure describes the grid, not the instrument response. Recording it as
        # an FWHM would assert a resolution nobody measured, and would misinform any
        # resolution-shift experiment (failure code F07).
        return AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, values, resolution_fwhm=None)

    def bulk_files(self) -> tuple[BulkFile, ...]:
        """The one published matrix CSV: 3,510 spectra x 3,276 wavenumbers, plus a label column.

        URL and size read from the figshare API for article 27931131 on 2026-09-03 and recorded
        here rather than resolved at fetch time, so that a change in what the URL points at shows
        up as a diff. Re-derive with ``curl https://api.figshare.com/v2/articles/27931131``.
        """
        return (
            BulkFile(
                name="raman_spectra_api_compounds.csv",
                url="https://ndownloader.figshare.com/files/50881917",
                published_bytes=193_164_182,
                note="3,510 spectra of 32 compounds, one Endress+Hauser Rxn2 at 785 nm",
            ),
        )

    def fetch(self, record_id: str, dest: Path) -> Path:
        """Not implemented: the dataset is one 193 MB file, not per-record downloads.

        Fetching it belongs in a manifest-driven bulk step rather than here; the figshare download
        URL is recorded on the dataset spec.
        """
        raise NotImplementedError(
            "scidata-api publishes a single 193 MB CSV; download it from "
            f"{SCIDATA_API_SPEC.homepage} and point 'usis data verify' at it. Per-record fetch is "
            "not meaningful for this dataset."
        )
