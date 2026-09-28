"""The dataset manifest and the vendoring gate."""

from __future__ import annotations

import pytest

from usis.core.errors import LicenceError, UsisError
from usis.core.registry import RegistryError
from usis.core.types import Modality, Origin
from usis.data.registry import (
    DATASETS,
    DatasetSpec,
    Verification,
    get_dataset,
    load_sources,
    register_dataset,
)


def _spec(**overrides) -> DatasetSpec:
    base = {
        "dataset_id": "unit-test-ds",
        "name": "Unit test dataset",
        "modality": Modality.RAMAN,
        "origin": Origin.EXPERIMENTAL,
        "licence": "CC0-1.0",
    }
    base.update(overrides)
    return DatasetSpec(**base)


def test_a_dataset_cannot_be_registered_with_an_invented_licence():
    with pytest.raises(LicenceError, match="unknown licence"):
        _spec(licence="CC-BY-9000")


def test_licence_drives_the_vendoring_gate():
    assert _spec(licence="CC0-1.0").vendorable is True
    unverified = _spec(licence="UNVERIFIED")
    assert unverified.vendorable is False
    with pytest.raises(LicenceError, match="redistribution"):
        unverified.assert_vendorable()


def test_verification_vocabulary_is_enforced():
    with pytest.raises(UsisError, match="verification must be one of"):
        _spec(verification="probably-fine")


def test_a_verified_spec_must_say_when_it_was_verified():
    with pytest.raises(UsisError, match="verified_on"):
        _spec(verification=Verification.VERIFIED)
    assert _spec(verification=Verification.VERIFIED, verified_on="2026-09-03").verified_on


def test_unverified_is_the_default():
    assert _spec().verification == Verification.UNVERIFIED


def test_source_record_carries_the_licence_and_the_checksum():
    spec = _spec(homepage="https://example.invalid/", doi="10.0000/x")
    record = spec.source_record(retrieved_at="2026-09-03T00:00:00+00:00", sha256="a" * 64)
    assert record.dataset_id == "unit-test-ds"
    assert record.licence == "CC0-1.0"
    assert record.doi == "10.0000/x"
    assert record.url == "https://example.invalid/"
    assert record.sha256 == "a" * 64


def test_duplicate_registration_is_refused():
    spec = _spec(dataset_id="dup-test-ds")
    register_dataset(spec)
    with pytest.raises(RegistryError, match="already registered"):
        register_dataset(spec)


def test_load_sources_populates_the_manifest():
    manifest = load_sources()
    assert "rod" in manifest
    assert manifest["rod"].licence == "CC0-1.0"
    assert get_dataset("rod").vendorable is True


def test_every_registered_dataset_has_a_known_licence_and_a_status():
    load_sources()
    for dataset_id in DATASETS:
        spec = DATASETS.get(dataset_id)
        assert spec.licence_spec.key == spec.licence
        assert spec.verification in Verification.ALL
