"""Run provenance: what code, what data, what environment produced a number.

Brief §4.3 and §9: instrument before you iterate. A run that cannot be replayed is an anecdote,
so every entry point captures a :class:`RunContext` before doing work, and every result card
quotes it. Nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "RunContext",
    "capture_environment",
    "content_hash",
    "git_state",
    "set_global_seeds",
    "utc_now",
]

_HASH_CHUNK_BYTES = 1 << 20


def utc_now() -> str:
    """An ISO-8601 UTC timestamp, the repo's one timestamp format."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def content_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed so that large blobs do not need to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_payload(payload: Any) -> str:
    """SHA-256 of a JSON-serialisable payload, used for config and manifest hashes."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def git_state(repo: Path | None = None) -> dict[str, str]:
    """Return the git SHA and working-tree cleanliness, or an explanation of why they are absent.

    A dirty tree is recorded rather than rejected: blocking exploratory runs would only teach us
    to stop recording. What must never happen is a *reported* number from a dirty tree, which is
    a check for the reporting layer, not for this function.
    """
    cwd = repo or Path.cwd()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return {"git_sha": "unavailable", "git_dirty": "unknown", "git_error": str(exc)}
    return {"git_sha": sha, "git_dirty": "true" if status else "false"}


def capture_environment() -> dict[str, str]:
    """Capture the interpreter, platform and hardware facts a replay would need."""
    env: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import numpy

        env["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        env["numpy"] = "missing"
    for optional in ("torch", "scipy", "sklearn", "rdkit"):
        try:
            module = __import__(optional)
        except ImportError:
            continue
        env[optional] = str(getattr(module, "__version__", "unknown"))
    return env


def set_global_seeds(seed: int) -> dict[str, int]:
    """Seed every RNG we can reach, and report which ones were actually seeded.

    Determinism claims are checked against the returned dict rather than assumed: if torch is
    absent, the caller learns that the torch seed was not set instead of believing it was.
    """
    seeded: dict[str, int] = {}
    random.seed(seed)
    seeded["python"] = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = seed
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
        seeded["torch"] = seed
    except ImportError:
        pass
    return seeded


@dataclass(frozen=True)
class RunContext:
    """Everything needed to attribute and replay a run."""

    run_id: str
    started_at: str
    seed: int
    git_sha: str
    git_dirty: str
    environment: dict[str, str] = field(default_factory=dict)
    config_hash: str = ""
    seeded: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def create(
        cls,
        run_id: str,
        seed: int = 0,
        config: Any = None,
        repo: Path | None = None,
        notes: str = "",
    ) -> RunContext:
        """Capture the current context and seed the RNGs in one step."""
        git = git_state(repo)
        return cls(
            run_id=run_id,
            started_at=utc_now(),
            seed=seed,
            git_sha=git["git_sha"],
            git_dirty=git["git_dirty"],
            environment=capture_environment(),
            config_hash=hash_payload(config) if config is not None else "",
            seeded=set_global_seeds(seed),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view, written verbatim into every run directory."""
        return asdict(self)

    def write(self, directory: Path) -> Path:
        """Write ``run_context.json`` into ``directory`` and return the path."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "run_context.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path
