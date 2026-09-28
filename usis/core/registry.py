"""A typed, duplicate-rejecting registry.

Used for methods, datasets, splits and preprocessing steps. The important property is not
lookup: it is that registering the same key twice raises. In a research repo a silently
overwritten registration produces results attributed to the wrong implementation, and that class
of error is invisible in the output.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Generic, TypeVar

from usis.core.errors import UsisError

__all__ = ["Registry", "RegistryError"]

T = TypeVar("T")


class RegistryError(UsisError):
    """A registration conflicted with an existing entry, or a lookup failed."""


class Registry(Generic[T]):
    """An append-only mapping from stable string keys to entries."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T] = {}

    @property
    def name(self) -> str:
        """The registry's name, used in error messages."""
        return self._name

    def register(self, key: str, entry: T) -> T:
        """Register ``entry`` under ``key``. Raises if the key is taken."""
        if not key:
            raise RegistryError(f"{self._name}: registration key must be non-empty")
        if key in self._entries:
            raise RegistryError(
                f"{self._name}: {key!r} is already registered. Registrations are append-only so "
                "that a result can always be attributed to one implementation; bump the version "
                "in the key instead of overwriting."
            )
        self._entries[key] = entry
        return entry

    def decorator(self, key: str) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under ``key``."""

        def _register(entry: T) -> T:
            return self.register(key, entry)

        return _register

    def get(self, key: str) -> T:
        """Look up an entry, listing the available keys when the lookup fails."""
        try:
            return self._entries[key]
        except KeyError:
            available = ", ".join(sorted(self._entries)) or "<empty>"
            raise RegistryError(
                f"{self._name}: no entry {key!r}. Registered: {available}"
            ) from None

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` has been registered."""
        return key in self._entries

    def __len__(self) -> int:
        """Number of registered entries."""
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered keys in sorted order."""
        return iter(sorted(self._entries))

    def as_mapping(self) -> Mapping[str, T]:
        """A read-only snapshot of the registry contents."""
        return dict(self._entries)
