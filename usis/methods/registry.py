"""The method registry: a research artifact that happens to also be a lookup table.

Brief §6.1: *``REFUTED`` entries stay, with the experiment that refuted them, so nobody re-tries a
dead idea blindly a year later. The registry is a research artifact and should be publishable on
its own.* That sentence is the specification for this module, and it has two consequences that are
enforced here rather than left to discipline.

**A maturity level is a claim, so registration checks the evidence for it.** The brief defines
``ESTABLISHED`` as *peer-reviewed AND reproduced by us*, so an entry cannot claim it without
provenance and at least one reproduction reference. ``HYPOTHESIZED`` needs provenance too - a
conjecture with no pointer to where it is written down cannot be argued with, and within a year is
indistinguishable from folklore. ``REFUTED`` needs the refuting evidence, ``DEPRECATED`` needs the
thing that superseded it. See `PROJECT_DECISIONS.md` D14 for the one place this reading of the
ladder had to be decided rather than read off: a definitional check with no paper behind it.

**Adding a method must never require touching the harness.** :func:`load_methods_from` imports
every module in a package and lets the registration decorators fire, so a new method is a new file
and nothing else. The comparison harness will read the registry, not a list.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from usis.core.registry import Registry, RegistryError
from usis.core.types import Maturity, TaskKind
from usis.methods.base import SURFACEABLE_MATURITIES, Citation, MethodBase

__all__ = [
    "BUILTIN_METHOD_PACKAGE",
    "METHODS",
    "MethodEntry",
    "MethodRegistrationError",
    "entries",
    "load_builtin_methods",
    "load_methods_from",
    "register_method",
    "registers",
    "select_methods",
]

#: Where the repository's own methods live. One module per method; nothing imports them by name.
BUILTIN_METHOD_PACKAGE = "usis.methods.builtin"

M = TypeVar("M", bound=MethodBase)


class MethodRegistrationError(RegistryError):
    """A method's registration does not support the maturity level it claims."""


@dataclass(frozen=True)
class MethodEntry:
    """One registered method, with the evidence for the maturity level it claims.

    ``evidence`` holds references, not prose: result-card paths, ADR numbers, test node ids. The
    point is that a reader can go and check, and that a promotion later has something to compare
    against.
    """

    method: MethodBase
    evidence: tuple[str, ...] = ()
    refuted_by: str = ""
    superseded_by: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""
    shadow_of: str = ""

    @property
    def key(self) -> str:
        """``id@version`` - the registry key."""
        return self.method.method_key

    @property
    def maturity(self) -> Maturity:
        """The maturity level this entry claims."""
        return self.method.maturity

    @property
    def surfaceable(self) -> bool:
        """Whether results from this entry may appear in a report conclusion."""
        return self.maturity in SURFACEABLE_MATURITIES

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable row, which is what makes the registry publishable on its own."""
        provenance = self.method.provenance
        return {
            "key": self.key,
            "task": str(self.method.task),
            "maturity": str(self.maturity),
            "surfaceable": self.surfaceable,
            "provenance": str(provenance) if provenance is not None else None,
            "assumptions": [a.describe() for a in self.method.all_assumptions],
            "cost": self.method.cost.to_dict(),
            "evidence": list(self.evidence),
            "refuted_by": self.refuted_by,
            "superseded_by": self.superseded_by,
            "shadow_of": self.shadow_of,
            "tags": list(self.tags),
            "notes": self.notes,
        }


#: The one registry. Keyed by ``id@version``, so a version bump is a new entry and an old result
#: stays attributable to the implementation that produced it.
METHODS: Registry[MethodEntry] = Registry("methods")


def _require_provenance(method: MethodBase, level: Maturity) -> Citation:
    provenance = method.provenance
    if provenance is None or not provenance.is_attributed:
        raise MethodRegistrationError(
            f"{method.method_key} claims maturity {level} but carries no attributed provenance. "
            "Set Citation(doi=...) for published work, or Citation(internal_ref=...) pointing at "
            "the ledger entry or knowledge note where the claim is written down - a claim with "
            "nowhere to check it cannot be argued with later."
        )
    return provenance


def _check_maturity_evidence(entry: MethodEntry) -> None:
    """Refuse a registration whose evidence does not support the maturity level it claims."""
    method = entry.method
    level = method.maturity
    if level is Maturity.ESTABLISHED:
        _require_provenance(method, level)
        if not entry.evidence:
            raise MethodRegistrationError(
                f"{method.method_key} claims ESTABLISHED, which the brief defines as peer-reviewed "
                "AND reproduced by us, but supplies no evidence reference for the reproduction. "
                "Pass evidence=(...) naming the result card or the test that reproduced it."
            )
    elif level is Maturity.EMERGING or level is Maturity.HYPOTHESIZED:
        _require_provenance(method, level)
    elif level is Maturity.REFUTED:
        if not entry.refuted_by:
            raise MethodRegistrationError(
                f"{method.method_key} claims REFUTED with no refuted_by reference. A refuted entry "
                "is kept precisely so the next person can read the experiment that killed it; "
                "without the reference the entry is just a discouragement."
            )
    elif level is Maturity.DEPRECATED and not entry.superseded_by:
        raise MethodRegistrationError(
            f"{method.method_key} claims DEPRECATED with no superseded_by reference. Name what "
            "replaced it, or the entry tells a reader to stop without telling them where to go."
        )


def register_method(
    method: M,
    *,
    evidence: Sequence[str] = (),
    refuted_by: str = "",
    superseded_by: str = "",
    shadow_of: str = "",
    tags: Sequence[str] = (),
    notes: str = "",
) -> M:
    """Register ``method`` and return it, so a module can register at import and still export it.

    Raises:
        MethodRegistrationError: if the evidence does not support the claimed maturity level, or
            if the same ``id@version`` is already registered.
    """
    entry = MethodEntry(
        method=method,
        evidence=tuple(evidence),
        refuted_by=refuted_by,
        superseded_by=superseded_by,
        shadow_of=shadow_of,
        tags=tuple(tags),
        notes=notes,
    )
    _check_maturity_evidence(entry)
    METHODS.register(entry.key, entry)
    return method


def registers(**kwargs: Any) -> Callable[[type[M]], type[M]]:
    """Class decorator form: instantiate the decorated method class and register the instance."""

    def _register(cls: type[M]) -> type[M]:
        register_method(cls(), **kwargs)
        return cls

    return _register


def select_methods(
    *,
    task: TaskKind | None = None,
    maturities: Sequence[Maturity] | None = None,
    tags: Sequence[str] = (),
) -> tuple[MethodEntry, ...]:
    """The entries matching a task and maturity filter, in key order.

    ``maturities=None`` means *every* level, refuted ones included, because a comparison that
    re-runs a refuted method to confirm it is still refuted is a legitimate query. The harness
    passes the filter explicitly; there is no default that quietly hides an entry.
    """
    wanted = frozenset(maturities) if maturities is not None else None
    required_tags = frozenset(tags)
    selected = []
    for key in METHODS:
        entry = METHODS.get(key)
        if task is not None and entry.method.task is not task:
            continue
        if wanted is not None and entry.maturity not in wanted:
            continue
        if required_tags and not required_tags.issubset(entry.tags):
            continue
        selected.append(entry)
    return tuple(selected)


def entries() -> Iterator[MethodEntry]:
    """Every registered entry, in key order."""
    for key in METHODS:
        yield METHODS.get(key)


def load_methods_from(package: str) -> tuple[str, ...]:
    """Import every module in ``package`` so its registrations fire; return the module names.

    This is what makes "adding a method touches only its own file" true rather than aspirational.
    Import is idempotent, and the registry refuses duplicate keys, so calling this twice is safe
    and calling it after a manual import is too.
    """
    module = importlib.import_module(package)
    paths = getattr(module, "__path__", None)
    if paths is None:
        raise MethodRegistrationError(f"{package} is not a package, so it holds no method modules")
    loaded = []
    for info in pkgutil.iter_modules(list(paths)):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{package}.{info.name}")
        loaded.append(info.name)
    return tuple(sorted(loaded))


def load_builtin_methods() -> tuple[str, ...]:
    """Import the repository's own method modules."""
    return load_methods_from(BUILTIN_METHOD_PACKAGE)
