"""The licence vocabulary and the redistribution gate.

Brief §9: ``usis/data/registry.py`` records a licence per dataset and CI fails when an unknown or
non-redistributable licence is vendored into the repo. This module holds the vocabulary that gate
checks against.

Two flags are tracked separately because they fail differently:

* ``redistributable`` - may we ship the bytes inside this repository?
* ``commercial_use`` - may a downstream user apply the result commercially?

``UNVERIFIED`` is a first-class member and is non-redistributable. That is the whole point: a
dataset whose terms nobody has read must block the build rather than pass by default.

Every entry here is a statement about a *licence*, not about a dataset. Which licence a given
dataset carries is a separate, per-dataset claim that must be verified against the source and
recorded in ``docs/knowledge/datasets.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from usis.core.errors import LicenceError

__all__ = [
    "KNOWN_LICENCES",
    "LicenceSpec",
    "assert_vendorable",
    "is_known",
    "lookup",
]


@dataclass(frozen=True)
class LicenceSpec:
    """What one licence permits, in the two dimensions the repo has to make decisions about."""

    key: str
    name: str
    redistributable: bool
    commercial_use: bool
    requires_attribution: bool = False
    share_alike: bool = False
    note: str = ""


def _spec(
    key: str,
    name: str,
    *,
    redistributable: bool,
    commercial_use: bool,
    requires_attribution: bool = False,
    share_alike: bool = False,
    note: str = "",
) -> tuple[str, LicenceSpec]:
    return key, LicenceSpec(
        key=key,
        name=name,
        redistributable=redistributable,
        commercial_use=commercial_use,
        requires_attribution=requires_attribution,
        share_alike=share_alike,
        note=note,
    )


KNOWN_LICENCES: Final[dict[str, LicenceSpec]] = dict(
    [
        _spec("CC0-1.0", "Creative Commons Zero v1.0", redistributable=True, commercial_use=True),
        _spec(
            "CC-BY-4.0",
            "Creative Commons Attribution 4.0",
            redistributable=True,
            commercial_use=True,
            requires_attribution=True,
        ),
        _spec(
            "CC-BY-SA-4.0",
            "Creative Commons Attribution-ShareAlike 4.0",
            redistributable=True,
            commercial_use=True,
            requires_attribution=True,
            share_alike=True,
        ),
        _spec(
            "CC-BY-NC-4.0",
            "Creative Commons Attribution-NonCommercial 4.0",
            redistributable=True,
            commercial_use=False,
            requires_attribution=True,
            note="Research use only. Anything derived from it inherits the NC restriction.",
        ),
        _spec(
            "CC-BY-NC-SA-4.0",
            "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
            redistributable=True,
            commercial_use=False,
            requires_attribution=True,
            share_alike=True,
        ),
        _spec(
            "CC-BY-NC-3.0",
            "Creative Commons Attribution-NonCommercial 3.0",
            redistributable=True,
            commercial_use=False,
            requires_attribution=True,
        ),
        _spec(
            "CC-BY-NC-ND-4.0",
            "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0",
            redistributable=False,
            commercial_use=False,
            requires_attribution=True,
            note=(
                "ND forbids derivative works. Verbatim redistribution is permitted, but every "
                "artifact this project makes from a dataset is a derivative (resampled, split, "
                "augmented), so the gate blocks it until somebody confirms a verbatim-only use."
            ),
        ),
        _spec(
            "GPL-3.0",
            "GNU General Public License v3.0",
            redistributable=True,
            commercial_use=True,
        ),
        _spec("MIT", "MIT License", redistributable=True, commercial_use=True),
        _spec("Apache-2.0", "Apache License 2.0", redistributable=True, commercial_use=True),
        _spec("BSD-3-Clause", "BSD 3-Clause", redistributable=True, commercial_use=True),
        _spec(
            "ODbL-1.0",
            "Open Database License 1.0",
            redistributable=True,
            commercial_use=True,
            requires_attribution=True,
            share_alike=True,
        ),
        _spec(
            "PDDL-1.0",
            "Open Data Commons Public Domain Dedication",
            redistributable=True,
            commercial_use=True,
        ),
        _spec(
            "US-GOV-PUBLIC-DOMAIN",
            "US Government work, public domain",
            redistributable=True,
            commercial_use=True,
            note="Applies to the work itself; a compilation may still carry separate terms.",
        ),
        # --- gates -------------------------------------------------------------------------
        _spec(
            "UNVERIFIED",
            "Licence not yet verified",
            redistributable=False,
            commercial_use=False,
            note="The loud default. Blocks vendoring until somebody reads the actual terms.",
        ),
        _spec(
            "NO-LICENCE-STATED",
            "Source states no licence",
            redistributable=False,
            commercial_use=False,
            note="Absence of a licence is not permission. Treated as all rights reserved.",
        ),
        _spec(
            "PROPRIETARY",
            "Proprietary or all rights reserved",
            redistributable=False,
            commercial_use=False,
        ),
        _spec(
            "ACADEMIC-USE-ONLY",
            "Custom academic-use terms",
            redistributable=False,
            commercial_use=False,
            note="Usually usable in place but not redistributable. Read the specific terms.",
        ),
        _spec(
            "TERMS-OF-USE",
            "Site terms of use, no open licence",
            redistributable=False,
            commercial_use=False,
            note="Typical for web databases: query access permitted, bulk redistribution not.",
        ),
    ]
)


def is_known(key: str) -> bool:
    """Whether ``key`` is in the licence vocabulary."""
    return key in KNOWN_LICENCES


def lookup(key: str) -> LicenceSpec:
    """Return the spec for ``key``, or raise with the accepted vocabulary."""
    try:
        return KNOWN_LICENCES[key]
    except KeyError:
        accepted = ", ".join(sorted(KNOWN_LICENCES))
        raise LicenceError(
            f"unknown licence {key!r}. Accepted: {accepted}. If a source's terms do not match "
            "any of these, add a spec for them here after reading the actual terms - do not "
            "approximate."
        ) from None


def assert_vendorable(key: str, dataset_id: str = "") -> LicenceSpec:
    """Raise unless ``key`` permits shipping the data inside this repository."""
    spec = lookup(key)
    if not spec.redistributable:
        subject = f"dataset {dataset_id!r}" if dataset_id else "this data"
        raise LicenceError(
            f"{subject} carries licence {key!r} ({spec.name}), which does not permit "
            f"redistribution. Fetch it at build time instead of vendoring it. {spec.note}".strip()
        )
    return spec
