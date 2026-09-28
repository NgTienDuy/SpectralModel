"""Shared fixtures. Nothing here touches the network or the filesystem outside tmp_path."""

from __future__ import annotations

import numpy as np
import pytest

from usis.core.types import (
    AxisKind,
    AxisSpec,
    InstrumentSpec,
    IntensityKind,
    Modality,
    Origin,
    SourceRecord,
    Spectrum,
)


@pytest.fixture
def raman_axis() -> AxisSpec:
    return AxisSpec.from_values(
        AxisKind.RAMAN_SHIFT_CM1, np.linspace(200.0, 1800.0, 401), resolution_fwhm=4.0
    )


@pytest.fixture
def source() -> SourceRecord:
    return SourceRecord(
        dataset_id="fixture",
        licence="CC0-1.0",
        retrieved_at="2026-09-02T00:00:00+00:00",
    )


@pytest.fixture
def instrument() -> InstrumentSpec:
    return InstrumentSpec(
        instrument_id="fixture-raman-01",
        manufacturer="fixture",
        excitation_nm=785.0,
        lab_id="fixture-lab",
    )


@pytest.fixture
def spectrum(raman_axis: AxisSpec, source: SourceRecord, instrument: InstrumentSpec) -> Spectrum:
    rng = np.random.default_rng(0)
    intensity = 100.0 + rng.normal(0.0, 1.0, raman_axis.n_points)
    return Spectrum(
        spectrum_id="fixture-0001",
        modality=Modality.RAMAN,
        origin=Origin.EXPERIMENTAL,
        axis=raman_axis,
        intensity=intensity,
        intensity_kind=IntensityKind.COUNTS,
        sample_id="fixture-sample-A",
        source=source,
        instrument=instrument,
    )
