"""Rendering a comparison record for a human, without letting the rendering make a claim.

The JSON record is the deliverable; this is the reading of it. Three things it does differently
from the leaderboard convention, each because the convention misleads here:

**It ranks on coverage at a risk ceiling, not on accuracy.** `CLAUDE.md` states the goal as *push
coverage up while holding risk at the floor*, so that is the sort key. Ranking on accuracy would
put a method that answers one item in three at the top of a table whose readers will take it as the
better method.

**It will not rank on AURC across methods that cannot reach the same coverage.** Those areas are
integrated over different intervals, so their ordering is not a comparison. When
:func:`~usis.compare.risk_coverage.comparable_aurc` is false the column is still printed - it is
informative per method - and the incomparability is stated above the table.

**Shadow rows appear, below a heading that says they are not a conclusion.** Deleting them would
lose the scoring that shadow mode exists to accumulate; printing them in the same table would be
the promotion the gate is there to prevent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from usis.compare.risk_coverage import comparable_aurc
from usis.compare.scoring import Outcome

if TYPE_CHECKING:
    from collections.abc import Sequence

    from usis.compare.record import ComparisonRecord, MethodRow

__all__ = ["rank_rows", "render_markdown", "render_table"]


def rank_rows(rows: Sequence[MethodRow], risk_target: float) -> tuple[MethodRow, ...]:
    """Order rows by coverage held at the risk ceiling, then by selective accuracy, then by key."""

    def key(row: MethodRow) -> tuple[float, float, str]:
        return (-row.curve.coverage_at_risk(risk_target), -row.selective_accuracy, row.method_key)

    return tuple(sorted(rows, key=key))


def _interval(row: MethodRow, name: str) -> str:
    metric = row.metric(name)
    if metric is None:
        return "-"
    flag = "!" if metric.is_degenerate else ""
    return f"{metric.mean:.3f} [{metric.ci_low:.3f}, {metric.ci_high:.3f}]{flag}"


def _estimate(row: MethodRow, name: str) -> str:
    estimate = row.estimate(name)
    if estimate is None:
        return "-"
    flag = "!" if estimate.is_degenerate else ""
    return f"{estimate.value:.3f} [{estimate.ci_low:.3f}, {estimate.ci_high:.3f}]{flag}"


def _rows_block(rows: Sequence[MethodRow], risk_target: float) -> list[str]:
    """The table body, as aligned plain-text lines."""
    coverage_name = f"coverage_at_risk<={risk_target:g}"
    header = (
        f"{'method':<30} {'cov@risk':<22} {'sel.acc':<22} {'coverage':<22} "
        f"{'aurc':<22} {'viol':<6} {'undec':<6} {'errs':<5}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.method_key:<30} {_estimate(row, coverage_name):<22} "
            f"{_interval(row, 'selective_accuracy'):<22} {_interval(row, 'coverage'):<22} "
            f"{_estimate(row, 'aurc'):<22} "
            f"{row.applicability.violation_rate:<6.2f} "
            f"{row.applicability.undecided_rate:<6.2f} "
            f"{row.outcomes[Outcome.ERRORED]:<5d}"
        )
    return lines


def render_table(record: ComparisonRecord) -> str:
    """A terminal rendering of the record."""
    lines: list[str] = [
        f"{record.task_id}  ({record.task_kind})",
        f"  datasets   : {', '.join(record.dataset_ids) or 'unnamed'}",
    ]
    split = record.split
    if split is not None:
        lines.append(
            f"  split      : {split.kind} on {split.group_key}, "
            f"train {split.n_train} / test {split.n_test}, seed {split.seed}"
        )
    lines += [
        f"  items      : {record.n_items_presented} presented, "
        f"{record.n_unscorable} unscorable (no independent answer, in no denominator)",
        f"  git        : {record.run_context.git_sha[:12]} (dirty: {record.run_context.git_dirty})",
        "",
    ]
    if not record.is_attributable:
        lines += [
            "  ** NOT FULLY ATTRIBUTABLE: the working tree was dirty, so these numbers do not",
            "     belong to the code at that commit. **",
            "",
        ]

    surfaceable = rank_rows(record.surfaceable_rows, record.risk_target)
    if surfaceable:
        if not comparable_aurc([r.curve for r in surfaceable]):
            lines += [
                "  NOTE: these methods reach different maximum coverage, so their AURCs are areas",
                "  over different intervals and must not be ranked against each other. The ranking",
                f"  below is on coverage held at risk <= {record.risk_target:g}.",
                "",
            ]
        lines += _rows_block(surfaceable, record.risk_target)
    else:
        lines.append("  no surfaceable method ran on this task.")

    shadow = rank_rows(record.shadow_rows, record.risk_target)
    if shadow:
        lines += [
            "",
            "SHADOW - COMPUTED AND SCORED, NOT A CONCLUSION",
            "  These are hypothesised methods. They are kept and scored so a promotion gate has",
            "  something to read; no report section may draw on them.",
            "",
        ]
        lines += _rows_block(shadow, record.risk_target)

    lines += ["", "Legend: intervals are percentile bootstrap; '!' marks an interval that"]
    lines.append("collapsed to a point and is therefore not an interval. 'viol' is the share of")
    lines.append("runs in which a declared assumption was checked and found false; 'undec' the")
    lines.append("share in which one could not be decided at all, which caps every result on that")
    lines.append("row at RUNNABLE_DEGRADED however good the score looks.")
    return "\n".join(lines) + "\n"


def render_markdown(record: ComparisonRecord) -> str:
    """A Markdown rendering, for a result card or a milestone report."""
    lines: list[str] = [f"# {record.task_id}", ""]
    if not record.is_attributable:
        lines += [
            "> **These numbers are not fully attributable.** The working tree was "
            f"`{record.run_context.git_dirty}` when they were produced.",
            "",
        ]
    lines += [
        "## What this comparison shows",
        "",
        *(f"- {item}" for item in record.shows),
        "",
        "## What it does NOT show",
        "",
        *(f"- {item}" for item in record.does_not_show),
        "",
        "## Setup",
        "",
        f"- task kind: `{record.task_kind}`",
        f"- datasets: {', '.join(f'`{d}`' for d in record.dataset_ids) or 'unnamed'}",
        f"- items: {record.n_items_presented} presented, {record.n_unscorable} unscorable",
        f"- git SHA: `{record.run_context.git_sha}` (dirty: `{record.run_context.git_dirty}`), "
        f"seed `{record.run_context.seed}`",
        f"- intervals: percentile bootstrap, {record.bootstrap_samples} resamples",
        "",
    ]
    split = record.split
    if split is not None:
        lines += [
            f"- split: `{split.kind}` grouped on **{split.group_key}**, "
            f"train {split.n_train} / test {split.n_test}, seed `{split.seed}`",
            "",
        ]
        if split.notes:
            lines += ["**Caveats carried by this split:**", ""]
            lines += [f"- {note}" for note in split.notes]
            lines.append("")

    coverage_name = f"coverage_at_risk<={record.risk_target:g}"
    for title, rows in (
        ("Leaderboard", rank_rows(record.surfaceable_rows, record.risk_target)),
        (
            "Shadow - computed and scored, not a conclusion",
            rank_rows(record.shadow_rows, record.risk_target),
        ),
    ):
        if not rows:
            continue
        lines += [
            f"## {title}",
            "",
            f"| method | cov@risk<={record.risk_target:g} | selective accuracy | coverage | "
            "AURC | violation rate | errors |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            lines.append(
                f"| `{row.method_key}` | {_estimate(row, coverage_name)} | "
                f"{_interval(row, 'selective_accuracy')} | {_interval(row, 'coverage')} | "
                f"{_estimate(row, 'aurc')} | {row.applicability.violation_rate:.2f} | "
                f"{row.outcomes[Outcome.ERRORED]} |"
            )
        lines.append("")

    lines += [
        "## Failure codes",
        "",
        "Split by where the code came from: a method *declares* a code about its own run, a "
        "benchmark *scores* one against the reference. The same code can arrive from both and "
        "mean different things - `F12` declared is a predicted library gap, `F12` scored is an "
        "observed miss - so they are never summed into one bar.",
        "",
    ]
    any_failure = False
    for row in record.rows:
        if row.failures_declared or row.failures_scored:
            any_failure = True
            lines.append(f"- `{row.method_key}`")
            for label, histogram in (
                ("declared by the method", row.failures_declared),
                ("scored by the benchmark", row.failures_scored),
            ):
                counts = ", ".join(
                    f"`{code}` x{n}" for code, n in sorted(histogram.items(), key=lambda kv: -kv[1])
                )
                lines.append(f"  - {label}: {counts or 'none'}")
    if not any_failure:
        lines.append("- No failure codes were emitted. An empty histogram usually means the")
        lines.append("  classifier did not run, not that nothing failed — check which.")
    lines.append("")

    lines += ["## Assumption audit (N7)", ""]
    for row in record.rows:
        tally = row.applicability
        lines.append(
            f"- `{row.method_key}`: {tally.clean_rate:.0%} clean, "
            f"{tally.violation_rate:.0%} violated, {tally.undecided_rate:.0%} undecided "
            f"over {tally.n_runs} run(s)"
        )
        for name, count in sorted(tally.violated_assumptions.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - violated: `{name}` x{count}")
        for name, count in sorted(tally.undecided_assumptions.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - undecided: `{name}` x{count}")
    lines.append("")

    if record.notes:
        lines += ["## Notes", "", *(f"- {note}" for note in record.notes), ""]
    return "\n".join(lines) + "\n"
