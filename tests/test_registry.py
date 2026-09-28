"""A silently overwritten registration attributes a result to the wrong implementation."""

from __future__ import annotations

import pytest

from usis.core.registry import Registry, RegistryError


def test_register_and_get():
    registry: Registry[int] = Registry("things")
    registry.register("a", 1)
    assert registry.get("a") == 1
    assert "a" in registry
    assert len(registry) == 1
    assert list(registry) == ["a"]


def test_duplicate_registration_is_refused():
    registry: Registry[int] = Registry("things")
    registry.register("a", 1)
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("a", 2)
    assert registry.get("a") == 1


def test_missing_key_lists_what_is_available():
    registry: Registry[int] = Registry("things")
    registry.register("alpha", 1)
    with pytest.raises(RegistryError, match="alpha"):
        registry.get("beta")


def test_empty_key_is_refused():
    registry: Registry[int] = Registry("things")
    with pytest.raises(RegistryError, match="non-empty"):
        registry.register("", 1)


def test_decorator_form_registers_and_returns_the_object():
    registry: Registry[type] = Registry("classes")

    @registry.decorator("thing")
    class Thing:
        pass

    assert registry.get("thing") is Thing


def test_as_mapping_is_a_snapshot():
    registry: Registry[int] = Registry("things")
    registry.register("a", 1)
    snapshot = registry.as_mapping()
    registry.register("b", 2)
    assert "b" not in snapshot
