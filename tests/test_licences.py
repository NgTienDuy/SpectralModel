"""The licence gate. An unread licence must block the build, not pass by default."""

from __future__ import annotations

import pytest

from usis.core.errors import LicenceError
from usis.data.licences import KNOWN_LICENCES, assert_vendorable, is_known, lookup


def test_unverified_is_a_known_licence_and_is_not_vendorable():
    assert is_known("UNVERIFIED")
    assert lookup("UNVERIFIED").redistributable is False
    with pytest.raises(LicenceError, match="redistribution"):
        assert_vendorable("UNVERIFIED", "some-dataset")


def test_absence_of_a_licence_is_not_permission():
    assert lookup("NO-LICENCE-STATED").redistributable is False


def test_an_unknown_licence_string_raises_rather_than_defaulting():
    with pytest.raises(LicenceError, match="unknown licence"):
        lookup("CC-BY-9000")


def test_open_licences_pass_the_gate():
    for key in ("CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0"):
        assert assert_vendorable(key, "d").redistributable


def test_non_commercial_is_vendorable_but_flagged():
    spec = assert_vendorable("CC-BY-NC-4.0", "d")
    assert spec.redistributable is True
    assert spec.commercial_use is False


def test_every_spec_key_matches_its_mapping_key():
    for key, spec in KNOWN_LICENCES.items():
        assert spec.key == key
        assert spec.name
