"""The reproducibility package: result cards that state their own limits.

Brief §9 and §11.3. Every reported experiment ships a card carrying the version lock, the split,
interval-valued metrics, the failure histogram, and — the part that is usually missing —
**an explicit statement of what the experiment does and does not show**.

That last section is enforced, not encouraged. :class:`ResultCard` refuses to be constructed
without at least one entry in ``does_not_show``. The reasoning is simple: a boundary-condition
statement written after a reviewer asks for it is a defence; written before the run, it is a
design constraint. And every experiment has limits, so an empty list is always a claim nobody
checked rather than a genuinely unlimited result.

Two more refusals, for the same reason:

* a card with no metrics **and** no failure histogram records nothing and is rejected;
* a card whose run was made from a dirty working tree is written, but says so in bold, because a
  number from uncommitted code is not attributable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from usis.core.errors import UsisError
from usis.core.provenance import utc_now

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from usis.core.config import ExperimentConfig
    from usis.core.provenance import RunContext
    from usis.core.types import FailureCode
    from usis.data.splits import Split
    from usis.eval.metrics.statistics import MetricResult

__all__ = ["ResultCard", "ResultCardError", "failure_histogram"]


class ResultCardError(UsisError):
    """A result card is missing something a reader needs to judge the result."""


@dataclass(frozen=True)
class ResultCard:
    """One experiment's reproducibility package."""

    title: str
    run_context: RunContext
    config: ExperimentConfig
    metrics: tuple[MetricResult, ...] = ()
    failures: Mapping[FailureCode, int] = field(default_factory=dict)
    split: Split | None = None
    datasets: tuple[str, ...] = ()
    shows: tuple[str, ...] = ()
    does_not_show: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce the parts of §11.3 that are usually skipped."""
        if not self.title:
            raise ResultCardError("a result card needs a title")
        if not self.does_not_show:
            raise ResultCardError(
                "a result card must state what the experiment does NOT show. Every experiment has "
                "limits, so an empty list is a claim nobody checked. Written before the run this "
                "is a design constraint; written after a reviewer asks, it is a defence."
            )
        if not self.shows:
            raise ResultCardError(
                "a result card must state what the experiment does show, in plain language"
            )
        if not self.metrics and not self.failures:
            raise ResultCardError(
                "a result card with neither metrics nor a failure histogram records nothing"
            )

    @property
    def is_attributable(self) -> bool:
        """Whether the number can be traced to committed code and a committed config."""
        return self.run_context.git_dirty == "false" and self.config.is_reproducible_from_file

    def render(self) -> str:
        """Render the card as Markdown."""
        lines: list[str] = [f"# {self.title}", ""]

        if not self.is_attributable:
            lines += ["> **This result is not fully attributable.**", ">"]
            if self.run_context.git_dirty != "false":
                lines.append(
                    f"> The working tree was `{self.run_context.git_dirty}` when it ran, so the "
                    "code that produced it is not the code at that commit."
                )
            if self.config.overrides:
                lines.append(
                    "> Command-line overrides were applied, so the committed config alone does "
                    f"not reproduce it: `{'`, `'.join(self.config.overrides)}`"
                )
            lines.append("")

        lines += [
            "## What this shows",
            "",
            *(f"- {item}" for item in self.shows),
            "",
            "## What this does NOT show",
            "",
            *(f"- {item}" for item in self.does_not_show),
            "",
            "## Version lock",
            "",
            f"- git SHA: `{self.run_context.git_sha}` (dirty: `{self.run_context.git_dirty}`)",
            f"- run id: `{self.run_context.run_id}`, started `{self.run_context.started_at}`",
            f"- seed: `{self.run_context.seed}`, seeded: "
            f"`{', '.join(sorted(self.run_context.seeded)) or 'none'}`",
            f"- config: `{self.config.source_path or '<inline>'}` "
            f"hash `{self.config.config_hash[:16]}`",
            f"- environment: "
            f"{', '.join(f'{k} {v}' for k, v in sorted(self.run_context.environment.items()))}",
            "",
        ]

        if self.datasets:
            lines += ["## Datasets", "", *(f"- `{d}`" for d in self.datasets), ""]

        if self.split is not None:
            lines += [
                "## Split",
                "",
                f"- kind: `{self.split.kind}`, grouped on **{self.split.group_key}**",
                f"- train {self.split.n_train} / test {self.split.n_test} "
                f"({self.split.test_fraction:.1%} test), seed `{self.split.seed}`",
                f"- groups: {len(self.split.train_groups)} train / "
                f"{len(self.split.test_groups)} test",
                "",
            ]
            if self.split.notes:
                lines += ["**Caveats carried by this split:**", ""]
                lines += [f"- {note}" for note in self.split.notes]
                lines.append("")

        if self.metrics:
            lines += [
                "## Metrics",
                "",
                "| metric | mean | interval | IQM | n |",
                "| --- | --- | --- | --- | --- |",
            ]
            for metric in self.metrics:
                groups = "" if metric.n_groups is None else f" ({metric.n_groups} groups)"
                lines.append(
                    f"| {metric.name} | {metric.mean:.4g} | "
                    f"[{metric.ci_low:.4g}, {metric.ci_high:.4g}] @ {metric.confidence:.0%} | "
                    f"{metric.iqm:.4g} | {metric.n_items}{groups} |"
                )
            lines.append("")
            degenerate = [m for m in self.metrics if m.is_degenerate]
            if degenerate:
                lines += [
                    "**The interval collapsed to a point for "
                    f"{', '.join(m.name for m in degenerate)}.** That is not an interval: there "
                    "were too few independent units to resample, so the number is a bare point "
                    "estimate and must be read as one.",
                    "",
                ]
            wide = [m for m in self.metrics if m.mean_iqm_gap > 0.1 * max(abs(m.mean), 1e-12)]
            if wide:
                lines += [
                    "The mean and the IQM differ by more than 10% for "
                    f"{', '.join(m.name for m in wide)}. That is a finding, not a rounding "
                    "artifact: the average is being carried by the tails.",
                    "",
                ]

        lines += ["## Failure codes", ""]
        if self.failures:
            lines += ["| code | count |", "| --- | --- |"]
            lines += [
                f"| `{code}` | {count} |"
                for code, count in sorted(self.failures.items(), key=lambda kv: -kv[1])
            ]
        else:
            lines.append("No failures were classified. An empty histogram usually means the")
            lines.append("classifier was not run, not that nothing failed — check which.")
        lines.append("")

        if self.notes:
            lines += ["## Notes", "", *(f"- {note}" for note in self.notes), ""]

        lines += ["---", "", f"Written {utc_now()} by `usis.eval.mrep`."]
        return "\n".join(lines) + "\n"

    def write(self, directory: Path, filename: str = "result_card.md") -> Path:
        """Write the card and the run context into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        self.run_context.write(directory)
        path = directory / filename
        path.write_text(self.render(), encoding="utf-8")
        return path


def failure_histogram(codes: Sequence[FailureCode]) -> dict[FailureCode, int]:
    """Count failure codes into a histogram, the shape every evaluation run must emit."""
    counts: dict[FailureCode, int] = {}
    for code in codes:
        counts[code] = counts.get(code, 0) + 1
    return counts
