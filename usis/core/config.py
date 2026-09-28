"""Experiment configuration: committed, resolved, hashed.

Brief §9: *every reported number is reproducible from a committed config plus a seed; no magic
constants in scripts.* This module is the loading half of that contract, and the result-card writer
in :mod:`usis.eval.mrep` is the reporting half.

Three properties are enforced here rather than left to convention:

* **A config must name its split kind and its seed.** Both are required with no default, because a
  default split kind would make ``random`` the silent norm and a default seed would make runs
  irreproducible in the way that is hardest to notice.
* **A resolved config is hashed.** The hash goes on the result card, so two cards claiming the same
  configuration can be checked rather than trusted.
* **Overrides are recorded.** A number produced with ``seed=3`` overridden on the command line is
  not reproducible from the committed file alone, and the card has to say so.

OmegaConf rather than Hydra for now: this is Hydra's own config layer, so adding Hydra's groups and
multirun at M2 is additive. See ``PROJECT_DECISIONS.md`` D10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from usis.core.errors import UsisError
from usis.core.provenance import hash_payload

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["ConfigError", "ExperimentConfig", "load_config"]

#: Keys every experiment config must set. Absent defaults are the point, not an oversight.
REQUIRED_KEYS: tuple[str, ...] = ("name", "seed", "split_kind")


class ConfigError(UsisError):
    """A configuration is missing something a reproducible run needs."""


@dataclass(frozen=True)
class ExperimentConfig:
    """A resolved experiment configuration and everything needed to attribute it."""

    name: str
    seed: int
    split_kind: str
    values: dict[str, Any]
    source_path: str = ""
    overrides: tuple[str, ...] = field(default_factory=tuple)

    @property
    def config_hash(self) -> str:
        """SHA-256 of the fully resolved values, including any overrides."""
        return hash_payload(self.values)

    @property
    def is_reproducible_from_file(self) -> bool:
        """Whether the committed file alone reproduces this run.

        False as soon as an override is applied. A card for such a run has to list the overrides,
        because otherwise "reproducible from a committed config" is false in a way nobody can see.
        """
        return not self.overrides

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a nested value by dotted path, e.g. ``dataset.id``."""
        node: Any = self.values
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Read a nested value, raising with the full path when it is absent."""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"{self.name}: config is missing required key {dotted!r}")
        return value


def load_config(
    path: str | Path,
    overrides: Sequence[str] | None = None,
) -> ExperimentConfig:
    """Load, override and resolve a YAML experiment config.

    Args:
        path: the committed config file.
        overrides: OmegaConf dotlist overrides, e.g. ``["seed=3", "split.test_fraction=0.3"]``.
            Recorded on the result so a card can state that the file alone is not enough.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"no config at {config_path}")

    loaded = OmegaConf.load(config_path)
    if not isinstance(loaded, DictConfig):
        raise ConfigError(f"{config_path}: the top level of a config must be a mapping")
    applied = tuple(overrides or ())
    if applied:
        loaded = OmegaConf.merge(loaded, OmegaConf.from_dotlist(list(applied)))

    # resolve=True expands interpolations now, so the hash covers what actually ran rather than
    # the unresolved template.
    values = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(values, dict):
        raise ConfigError(f"{config_path}: resolved config is not a mapping")
    plain: dict[str, Any] = {str(k): v for k, v in values.items()}

    missing = [key for key in REQUIRED_KEYS if key not in plain]
    if missing:
        raise ConfigError(
            f"{config_path}: missing required key(s) {missing}. 'split_kind' and 'seed' have no "
            "defaults on purpose: a default split kind would make 'random' the silent norm, and a "
            "default seed would make the run irreproducible in the way hardest to notice."
        )
    try:
        seed = int(plain["seed"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{config_path}: seed must be an integer, got {plain['seed']!r}") from exc

    return ExperimentConfig(
        name=str(plain["name"]),
        seed=seed,
        split_kind=str(plain["split_kind"]),
        values=plain,
        source_path=str(config_path),
        overrides=applied,
    )
