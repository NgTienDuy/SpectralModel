"""Raman Open Database (ROD) loader.

ROD publishes each entry twice: a JCAMP-DX ``.jdx`` file and a CIF 2.0 ``.rod`` file. **We parse
the ``.rod``**, because it is a strict superset: the JCAMP twin carries the axis, the intensity and
a handful of labels, while the CIF carries the instrument, the laser, the resolution, the
integration time, the calibration standard, and an explicit statement of whether the publisher
applied background subtraction or baseline correction. That last pair is provenance we would
otherwise have to guess at, and guessing it is exactly what invariant 3 forbids.

Two mapping decisions are load-bearing and are argued in the functions below: what becomes
``sample_id``, and what becomes ``instrument_id``.

Licence: ROD states that all data are dedicated to the public domain under CC0, which is why the
fixtures for this loader can be committed verbatim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from gemmi import cif

from usis.core.errors import SpectrumValidationError
from usis.core.provenance import utc_now
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
from usis.data.sources.base import Source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gemmi.cif import Block

__all__ = [
    "ROD_INDEX_SNAPSHOT",
    "ROD_INDEX_URL",
    "ROD_SPEC",
    "SPECTRUM_LOOP_TAG",
    "THEORETICAL_LOOP_TAG",
    "UNMAPPED_TAGS",
    "RodSource",
    "rod_record_url",
]

ROD_SPEC: Final[DatasetSpec] = register_dataset(
    DatasetSpec(
        dataset_id="rod",
        name="Raman Open Database",
        modality=Modality.RAMAN,
        origin=Origin.EXPERIMENTAL,
        licence="CC0-1.0",
        homepage="https://solsa.crystallography.net/rod/",
        verification=Verification.VERIFIED,
        verified_on="2026-09-03",
        # _rod_data_source.file groups spectra of one specimen; 58 of 240 sampled specimens
        # appear at two excitations, so the grouping is real and load-bearing (D9).
        replicate_structure=ReplicateStructure.KNOWN,
        notes=(
            "CC0 confirmed on the project site; entries carry instrument, laser wavelength, "
            "resolution, integration time and an explicit preprocessing declaration. A survey of "
            "240 entries on 2026-09-03 found effectively one instrument (Thermo Scientific Nicolet "
            "Almega XR at Arizona University, 238/240) with a second (HORIBA Jobin Yvon T64000 at "
            "IMMM Maine university) at ~1%, and two dominant excitations, 532 and 780 nm. See "
            "docs/knowledge/datasets.md section 5."
        ),
    )
)

#: Base URL for a single record. ROD serves ``<code>.rod`` (CIF 2.0) and ``<code>.jdx`` (JCAMP-DX).
ROD_RECORD_URL: Final[str] = "https://solsa.crystallography.net/rod/{code}.rod"

#: The tag that makes a ROD record a *measured* spectrum.
SPECTRUM_LOOP_TAG: Final[str] = "_raman_spectrum.raman_shift"

#: The tag ROD uses for a **computed** spectrum. Two of the 1,133 published records carry this one
#: and no measured loop: they are DFT predictions deposited alongside the measurements.
#:
#: This is the reason the fetcher names it rather than merely failing. A loader that reached for
#: whatever `raman_shift` column it could find would put DFT spectra into a corpus labelled
#: ``EXPERIMENTAL`` with an instrument attached, which is failure code ``F13`` created at load
#: time and invisible everywhere downstream. Supporting them is a separate loader with
#: ``origin=SIMULATED`` and a level of theory instead of an instrument, the way the QM9S loader
#: does it - not a fallback inside this one.
THEORETICAL_LOOP_TAG: Final[str] = "_raman_theoretical_spectrum.raman_shift"

#: The database's own list of every published code, one per line. ROD inherits the Crystallography
#: Open Database's query interface, and ``format=lst`` is its plain-text id listing.
#:
#: This is what makes the corpus reproducible rather than sampled. The earlier instrument audit
#: walked guessed id blocks (``8100001-8100600`` and so on) and found 240 records; the index says
#: there are 1,133, in nine blocks, most of which that scan never touched.
ROD_INDEX_URL: Final[str] = "https://solsa.crystallography.net/rod/result.php?format=lst"

#: Committed snapshot of that listing, so ``usis data pull rod --all`` is reproducible and the
#: test suite can exercise the index without the network. Refresh with
#: ``scripts/refresh_rod_index.py``, which produces a reviewable diff.
ROD_INDEX_SNAPSHOT: Final[Path] = Path(__file__).parent / "data" / "rod_index.json"

#: Tags present in ROD records that this loader deliberately does not map, with the reason.
#:
#: Recorded as data rather than prose so that the omission is visible and testable. Dropping
#: metadata silently is how a corpus loses the very field a later experiment needs; dropping it
#: loudly is a decision somebody can revisit. Extending the core types to carry these is a
#: considered change to make once all four M1 loaders are written, not one loader at a time.
UNMAPPED_TAGS: Final[dict[str, str]] = {
    "_raman_measurement_device.direction_polarization": (
        "no polarisation field on AcquisitionParams yet; needed for depolarisation ratios"
    ),
    "_raman_measurement_device.spot_size": "no field; relevant to confocal depth response",
    "_raman_measurement_device.optics_type": "no field",
    "_raman_measurement_device.microscope_system": "no field",
    "_raman_measurement_device.configuration": "no field",
    "_raman_measurement_device.excitation_laser_type": "no field; the wavelength is what we use",
    "_raman_measurement.pressure": "no field; all sampled entries report ambient",
    "_raman_measurement.environment": "no field; all sampled entries report air",
    "_raman_complementary.method": "cross-technique link, belongs in the knowledge layer",
    "_rod_related_entry.code": "cross-database link, belongs in the knowledge layer",
    "_publ_author_name": "citation metadata, carried by DatasetSpec rather than per spectrum",
    "_cell_length_a": "crystallographic structure; a later structure layer, not a spectrum field",
    "_space_group_name_H-M_alt": "crystallographic symmetry; wanted later for selection rules",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def rod_record_url(code: str) -> str:
    """The canonical URL for one ROD record."""
    return ROD_RECORD_URL.format(code=code)


def _slug(value: str) -> str:
    """Lowercase, hyphenated slug suitable for use inside a stable identifier."""
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-")


def _text(block: Block, tag: str) -> str | None:
    """Read a CIF tag as unquoted text, mapping CIF nulls (``?`` and ``.``) to ``None``."""
    raw = block.find_value(tag)
    if raw is None or cif.is_null(raw):
        return None
    value = cif.as_string(raw).strip()
    return value or None


def _number(block: Block, tag: str) -> float | None:
    """Read a CIF tag as a float, returning ``None`` when absent, null or non-numeric."""
    value = _text(block, tag)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _declared_no(block: Block, tag: str) -> bool | None:
    """Interpret ROD's ``yes``/``no`` preprocessing declarations."""
    value = _text(block, tag)
    if value is None:
        return None
    return value.strip().lower() in {"no", "none", "false"}


def _sample_id(block: Block, code: str) -> str:
    """The key that groups spectra of one physical specimen.

    ``_rod_data_source.file`` names the upstream record a ROD entry was derived from (an RRUFF
    code such as ``R060425``, or a contributor's own key such as ``YE0002``), and it is the only
    tag observed to be consistent for this purpose. It matters: in a 240-entry survey on
    2026-09-03, **58 upstream records appeared at two different excitation wavelengths**. Group on
    the ROD code instead and a random split puts the 532 nm and 780 nm spectra of the same specimen
    on both sides — the replicate leakage the brief calls the most common silent killer.

    ``_chemical_compound_source`` is *not* usable: it is sometimes an RRUFF code, sometimes a
    locality ("Ojuela mine, Durango, Mexico"), sometimes "Synthesized".
    """
    upstream = _text(block, "_rod_data_source.file")
    if upstream:
        return f"rod:{Path(upstream).stem}"
    return f"rod:entry-{code}"


def _instrument(block: Block) -> InstrumentSpec | None:
    """Build the instrument spec, or ``None`` when the entry declares no measurement device.

    ``instrument_id`` is built from company, model **and location**, because it is the key the
    instrument-disjoint split groups on and two labs running the same model are not the same
    instrument for that purpose. ``lab_id`` is the location alone, for the lab-disjoint split.
    """
    company = _text(block, "_raman_measurement_device.company")
    model = _text(block, "_raman_measurement_device.model")
    location = _text(block, "_raman_measurement_device.location")
    if not any((company, model, location)):
        return None

    parts = [_slug(p) for p in (company, model, location) if p]
    magnification = _number(block, "_raman_measurement_device.microscope_objective_magnification")
    aperture = _number(block, "_raman_measurement_device.microscope_numerical_aperture")
    objective: str | None = None
    if magnification is not None:
        objective = f"{magnification:g}x"
        if aperture is not None:
            objective += f"/NA{aperture:g}"

    grating = _number(block, "_raman_measurement_device.diffraction_grating")
    return InstrumentSpec(
        instrument_id="rod:" + "__".join(parts),
        manufacturer=company,
        model=model,
        excitation_nm=_number(block, "_raman_measurement_device.excitation_laser_wavelength"),
        grating=None if grating is None else f"{grating:g}",
        objective=objective,
        detector=None,
        lab_id=None if location is None else _slug(location),
    )


def _processing(block: Block) -> tuple[ProcessingStep, ...]:
    """Reconstruct the publisher's declared processing chain.

    ROD states whether background subtraction and baseline correction were applied, and which
    calibration standard was used, but not the parameters. Recording "this happened, declared by
    the source, parameters unknown" is the honest form: silence would read as "nothing was applied",
    which is a different and possibly false claim.

    The timestamp is ``"unknown"`` rather than the retrieval time, because the retrieval time is
    not when the step ran and a plausible wrong timestamp is worse than an absent one.
    """
    steps: list[ProcessingStep] = []
    standard = _text(block, "_raman_measurement_device_calibration.standard")
    if standard:
        steps.append(
            ProcessingStep(
                name="calibration.wavenumber",
                params={"standard": standard, "declared_by": "rod"},
                version="source-declared",
                timestamp="unknown",
            )
        )
    for tag, name in (
        ("_raman_measurement.background_subtraction", "background.subtraction"),
        ("_raman_measurement.baseline_correction", "baseline.correction"),
    ):
        applied = _declared_no(block, tag)
        if applied is False:  # the source said "yes"; None means it said nothing at all
            steps.append(
                ProcessingStep(
                    name=name,
                    params={
                        "declared_by": "rod",
                        "params": "unknown",
                        "details": _text(block, f"{tag}_details") or "",
                    },
                    version="source-declared",
                    timestamp="unknown",
                )
            )
    return tuple(steps)


def _label(block: Block) -> GroundTruth | None:
    """Build the ground truth from the mineral or systematic name, with the formula as a note."""
    name = _text(block, "_chemical_name_mineral") or _text(block, "_chemical_name_systematic")
    formula = _text(block, "_chemical_formula_sum")
    if name is None and formula is None:
        return None
    note_parts = [f"formula={formula}" if formula else ""]
    method = _text(block, "_raman_determination_method")
    if method:
        note_parts.append(f"determination={method}")
    return GroundTruth(
        name=name,
        annotation_note="; ".join(p for p in note_parts if p) or None,
    )


class RodSource(Source):
    """Loader for the Raman Open Database."""

    @property
    def spec(self) -> DatasetSpec:
        """The ROD manifest entry."""
        return ROD_SPEC

    def parse(
        self,
        payload: str,
        *,
        url: str | None = None,
        sha256: str = "",
        retrieved_at: str | None = None,
    ) -> Iterator[Spectrum]:
        """Parse one ROD CIF 2.0 record. A record holds exactly one spectrum, so this yields one."""
        document = cif.read_string(payload)
        block = document.sole_block()
        code = _text(block, "_rod_database.code") or block.name

        table = block.find(["_raman_spectrum.raman_shift", "_raman_spectrum.intensity"])
        if len(table) == 0:
            raise SpectrumValidationError(f"ROD {code}: record contains no _raman_spectrum loop")
        shifts = np.array([float(row[0]) for row in table], dtype=np.float64)
        intensities = np.array([float(row[1]) for row in table], dtype=np.float64)

        diffs = np.diff(shifts)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            raise SpectrumValidationError(
                f"ROD {code}: the raman_shift column is not strictly monotonic, so the record "
                "cannot be read as a spectrum without deciding what the duplicates mean"
            )

        # The record states its own range. Disagreement means we are reading a different column
        # than the publisher described, which is worth failing on rather than averaging over.
        self._check_declared_range(block, code, shifts)

        method = (_text(block, "_raman_determination_method") or "experimental").lower()
        origin = Origin.EXPERIMENTAL if method == "experimental" else Origin.SIMULATED
        instrument = _instrument(block)
        if origin is Origin.EXPERIMENTAL and instrument is None:
            raise SpectrumValidationError(
                f"ROD {code}: declared experimental but names no measurement device"
            )

        resolution = _number(block, "_raman_measurement_device.resolution")
        temperature = _number(block, "_raman_measurement.temperature")
        yield Spectrum(
            spectrum_id=f"rod:{code}",
            modality=Modality.RAMAN,
            origin=origin,
            axis=AxisSpec.from_values(AxisKind.RAMAN_SHIFT_CM1, shifts, resolution_fwhm=resolution),
            intensity=intensities,
            # ROD's JCAMP twin declares YUNITS=ARBITRARY UNITS. Calling it counts would imply a
            # Poisson noise model we have no basis for.
            intensity_kind=IntensityKind.ARBITRARY,
            sample_id=_sample_id(block, code),
            source=ROD_SPEC.source_record(
                retrieved_at=retrieved_at or utc_now(),
                url=url or rod_record_url(code),
                sha256=sha256,
            ),
            instrument=instrument,
            acquisition=AcquisitionParams(
                exposure_s=_number(block, "_raman_measurement.integration_time"),
                laser_power_mw=_number(block, "_raman_measurement_device.power_on_sample"),
                # ROD reports temperature in kelvin: entry 8100020 gives 300 here and its JCAMP
                # twin gives 26.85, and 26.85 degC is 300.00 K.
                temperature_k=temperature,
            ),
            processing=_processing(block),
            label=_label(block),
        )

    @staticmethod
    def _check_declared_range(block: Block, code: str, shifts: np.ndarray) -> None:
        """Cross-check the parsed axis against the range the record declares for itself."""
        declared_min = _number(block, "_raman_measurement.range_min")
        declared_max = _number(block, "_raman_measurement.range_max")
        if declared_min is None or declared_max is None:
            return
        actual_min, actual_max = float(shifts.min()), float(shifts.max())
        tolerance = max(1.0, 0.01 * (actual_max - actual_min))
        if abs(actual_min - declared_min) > tolerance or abs(actual_max - declared_max) > tolerance:
            raise SpectrumValidationError(
                f"ROD {code}: parsed axis spans [{actual_min:.4g}, {actual_max:.4g}] but the "
                f"record declares [{declared_min:.4g}, {declared_max:.4g}]"
            )

    def list_record_ids(self) -> tuple[str, ...]:
        """Every ROD code, from the committed snapshot of the database's own index.

        Read from the snapshot rather than the network so that two runs of the same experiment pull
        the same corpus. A code that appeared since the snapshot is therefore *not* fetched, and
        that is the intended behaviour: a corpus that silently grows between runs makes two result
        cards incomparable without either of them saying so.
        """
        payload = json.loads(ROD_INDEX_SNAPSHOT.read_text(encoding="utf-8"))
        return tuple(str(code) for code in payload["ids"])

    def fetch(self, record_id: str, dest: Path) -> Path:
        """Download one ROD record (performs network I/O; never called from a test)."""
        # Imported here rather than at module scope to keep the network out of the import surface.
        from urllib.request import urlopen

        url = rod_record_url(record_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        # Checked on the spectrum loop rather than on a CIF magic string: ROD serves **both** CIF
        # 2.0 records (which start with '#\\#CIF_2.0') and CIF 1.x ones (which start with an
        # ordinary comment banner), and a magic-string check rejected every CIF 1.x record in the
        # database as if it were a fetch failure. What actually has to be present is the data.
        if SPECTRUM_LOOP_TAG.encode() not in payload:
            if THEORETICAL_LOOP_TAG.encode() in payload:
                raise SpectrumValidationError(
                    f"ROD {record_id}: this record holds a {THEORETICAL_LOOP_TAG} loop and no "
                    "measured one - it is a computed spectrum deposited alongside the "
                    "measurements. Loading it here would put a DFT prediction into an "
                    "EXPERIMENTAL corpus with an instrument attached (F13). It needs its own "
                    "loader with origin=SIMULATED and a level of theory."
                )
            preview = payload.lstrip()[:60].decode("utf-8", errors="replace")
            raise SpectrumValidationError(
                f"ROD {record_id}: response from {url} holds no {SPECTRUM_LOOP_TAG} loop, so it "
                f"is not a spectrum record (a missing record returns an HTML error page). "
                f"Starts: {preview!r}"
            )
        dest.write_bytes(payload)
        return dest
