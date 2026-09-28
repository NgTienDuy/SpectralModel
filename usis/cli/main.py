"""The ``usis`` command line.

Two rules shape this file:

1. **A verb that is not implemented says so, names the milestone that delivers it, and exits
   non-zero.** A stub that prints a plausible-looking result is the fastest way to end up
   believing a number nobody computed.
2. **Every verb that does real work captures a run context first.** Provenance is not an
   afterthought that gets added when the results start to matter.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from omegaconf import OmegaConf

from usis import __version__
from usis.compare.build import build_task
from usis.compare.harness import run_comparison
from usis.compare.leaderboard import render_markdown, render_table
from usis.core.config import load_config
from usis.core.errors import UsisError
from usis.core.provenance import RunContext, capture_environment, git_state
from usis.core.types import AxisKind, IntensityKind, Maturity, RamanBranch
from usis.core.units import convert_axis, convert_intensity
from usis.data.licences import KNOWN_LICENCES
from usis.data.pull import DEFAULT_DATA_DIR, pull_bulk, pull_records, verify_directory
from usis.data.registry import load_sources
from usis.methods.registry import entries, load_builtin_methods

__all__ = ["build_parser", "main"]

#: Exit code for a verb that this milestone does not implement.
EXIT_NOT_IMPLEMENTED: Final[int] = 3
#: Exit code for a deliberate USIS error (bad units, licence gate, invalid spectrum).
EXIT_USIS_ERROR: Final[int] = 4

_NOT_YET: Final[dict[str, str]] = {
    "analyze": "M2 onwards - the Sample Report needs the measurement audit (M1) and the "
    "preprocessing account (M2) before any section of it can be honest.",
    "simulate": "M5 - forward simulation.",
}


def _not_implemented(verb: str) -> int:
    """Report that a verb is not available at this milestone, and why."""
    reason = _NOT_YET.get(verb, "not implemented yet")
    print(f"usis {verb}: not implemented at this milestone.", file=sys.stderr)
    print(f"  Delivered by: {reason}", file=sys.stderr)
    print("  See PROJECT_STATUS.md for the current milestone.", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


def cmd_info(args: argparse.Namespace) -> int:
    """Print the environment and git state a reproduction would need."""
    git = git_state()
    environment = capture_environment()
    if args.json:
        payload = {"usis_version": __version__, **git, "environment": environment}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"usis {__version__}")
    print(f"git sha    : {git['git_sha']}")
    print(f"git dirty  : {git['git_dirty']}")
    for key, value in sorted(environment.items()):
        print(f"{key:<11}: {value}")
    return 0


def cmd_licences(args: argparse.Namespace) -> int:
    """List the licence vocabulary and which entries pass the vendoring gate."""
    if args.json:
        print(
            json.dumps(
                {k: vars(v) for k, v in sorted(KNOWN_LICENCES.items())}, indent=2, sort_keys=True
            )
        )
        return 0
    header = f"{'key':<24} {'vendorable':<11} {'commercial':<11} name"
    print(header)
    print("-" * len(header))
    for key, spec in sorted(KNOWN_LICENCES.items()):
        print(
            f"{key:<24} {'yes' if spec.redistributable else 'NO':<11} "
            f"{'yes' if spec.commercial_use else 'no':<11} {spec.name}"
        )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert values between axis or intensity conventions, explicitly.

    Exposed as a verb because the conversion is exactly the operation people do wrong in a
    spreadsheet, and having one audited implementation is worth a CLI surface.
    """
    values = np.asarray(args.values, dtype=np.float64)
    if args.kind == "axis":
        result = convert_axis(
            values,
            AxisKind(args.source),
            AxisKind(args.target),
            excitation_nm=args.excitation_nm,
            branch=RamanBranch(args.branch),
        )
    else:
        result = convert_intensity(values, IntensityKind(args.source), IntensityKind(args.target))
    if args.json:
        print(json.dumps({"from": args.source, "to": args.target, "values": result.tolist()}))
        return 0
    for original, converted in zip(values, result, strict=True):
        print(f"{original:>16.6g} -> {converted:.6g}")
    return 0


def cmd_run_context(args: argparse.Namespace) -> int:
    """Capture a run context, seed the RNGs, and write it out. The replay unit of the repo."""
    ctx = RunContext.create(run_id=args.run_id, seed=args.seed, notes=args.notes)
    print(json.dumps(ctx.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_methods(args: argparse.Namespace) -> int:
    """List the method registry: what is registered, at what maturity, on what evidence.

    Brief §6.1 calls the registry a research artifact that should be publishable on its own, so
    this verb prints it. The ``surfaceable`` column is the one to read: it is the answer to
    "may this method's output appear in a conclusion?", and a ``NO`` there for a HYPOTHESIZED
    entry is shadow mode being visible rather than implied.
    """
    load_builtin_methods()
    rows = [entry.to_dict() for entry in entries()]
    if args.maturity:
        wanted = set(args.maturity.split(","))
        rows = [row for row in rows if row["maturity"] in wanted]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("no methods registered at this filter")
        return 0
    width = max(len(row["key"]) for row in rows)
    header = f"{'method':<{width}} {'maturity':<14} {'surfaces':<9} task"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['key']:<{width}} {row['maturity']:<14} "
            f"{'yes' if row['surfaceable'] else 'NO':<9} {row['task']}"
        )
    print()
    for row in rows:
        print(f"{row['key']}")
        print(f"  provenance : {row['provenance']}")
        print(f"  assumptions: {', '.join(row['assumptions']) or 'none declared'}")
        if row["evidence"]:
            print(f"  evidence   : {', '.join(row['evidence'])}")
        if row["refuted_by"]:
            print(f"  refuted by : {row['refuted_by']}")
        if row["notes"]:
            print(f"  notes      : {row['notes']}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run every registered method for one benchmark and write the comparison record.

    The verb takes a config file and not a list of methods. That is the whole design: the field is
    whatever the registry holds for the task, so a leaderboard cannot be published having quietly
    left out the baseline that would have beaten it.
    """
    load_builtin_methods()
    config = load_config(args.config, args.overrides)
    built = build_task(config, root=Path(args.dest))
    task = built.task
    maturities = (
        None
        if not args.maturity
        else [Maturity(level.strip()) for level in args.maturity.split(",")]
    )
    ctx = RunContext.create(
        run_id=config.name, seed=config.seed, notes=f"usis compare {args.config}"
    )
    record = run_comparison(
        task,
        ctx,
        maturities=maturities,
        risk_target=float(config.get("compare.risk_target", 0.01)),
        coverage_target=float(config.get("compare.coverage_target", 0.5)),
        bootstrap_samples=int(config.get("compare.bootstrap_samples", 2000)),
        confidence=float(config.get("compare.confidence", 0.95)),
        shows=tuple(config.get("report.shows", ()) or ()),
        does_not_show=tuple(config.get("report.does_not_show", ()) or ()),
        notes=(*built.corpus_notes, *task.notes),
        config_hash=config.config_hash,
    )
    if args.out:
        directory = Path(args.out)
        record_path = record.write(directory)
        card_path = directory / "result_card.md"
        card_path.write_text(render_markdown(record), encoding="utf-8")
        # The resolved config, next to the hash the record carries, so a reader can check the
        # attribution rather than trust it. Resolved, not copied: an override applied on the
        # command line is part of what ran and must appear in the file that claims to reproduce it.
        config_path = directory / "config.yaml"
        config_path.write_text(OmegaConf.to_yaml(config.values), encoding="utf-8")
        for path in (record_path, card_path, config_path):
            print(f"wrote {path}", file=sys.stderr)
    if args.json:
        print(
            json.dumps(record.to_dict(include_items=False), indent=2, sort_keys=True, default=str)
        )
    else:
        print(render_table(record))
    return 0


def cmd_data_list(args: argparse.Namespace) -> int:
    """List the dataset manifest with the vendoring gate and the verification status."""
    manifest = load_sources()
    if args.json:
        print(
            json.dumps(
                {
                    key: {
                        "name": spec.name,
                        "modality": str(spec.modality),
                        "origin": str(spec.origin),
                        "licence": spec.licence,
                        "vendorable": spec.vendorable,
                        "verification": spec.verification,
                        "verified_on": spec.verified_on,
                        "homepage": spec.homepage,
                    }
                    for key, spec in sorted(manifest.items())
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    width = max([len(k) for k in manifest] + [4])
    header = f"{'id':<{width}} {'licence':<14} {'vendorable':<11} {'verified':<11} name"
    print(header)
    print("-" * len(header))
    for key, spec in sorted(manifest.items()):
        print(
            f"{key:<{width}} {spec.licence:<14} {'yes' if spec.vendorable else 'NO':<11} "
            f"{spec.verification:<11} {spec.name}"
        )
    return 0


def cmd_data_pull(args: argparse.Namespace) -> int:
    """Fetch records, or the published bulk files. Performs network I/O."""
    if args.bulk:
        paths = pull_bulk(args.dataset, root=Path(args.dest), max_bytes=args.max_bytes)
        for path in paths:
            print(f"fetched {path} ({path.stat().st_size:,} bytes on disk)")
        if args.max_bytes is not None:
            print(
                "NOTE: only part of the published file was taken, and it is not a random sample - "
                "the rows are whichever ones the byte offsets landed in, in the publisher's own "
                f"file order. See {Path(args.dest) / args.dataset / 'FETCH.json'}.",
                file=sys.stderr,
            )
        return 0
    if not args.records and not args.all:
        print(
            "usis data pull: name the record ids, or pass --all for every record in the "
            "committed index, or --bulk for a source published as whole files",
            file=sys.stderr,
        )
        return EXIT_USIS_ERROR
    paths = pull_records(
        args.dataset, args.records or None, root=Path(args.dest), skip_existing=not args.refetch
    )
    print(f"{len(paths)} record(s) into {args.dest}")
    missing = Path(args.dest) / args.dataset / "UNAVAILABLE.json"
    if missing.exists():
        print(
            f"NOTE: some records in the index could not be fetched; see {missing}", file=sys.stderr
        )
    return 0


def cmd_data_verify(args: argparse.Namespace) -> int:
    """Re-parse every record on disk and report. Offline."""
    results = verify_directory(Path(args.dest), dataset_id=args.dataset)
    if not results:
        print(f"usis data verify: nothing found under {args.dest}", file=sys.stderr)
        return EXIT_USIS_ERROR
    failures = [r for r in results if not r.ok]
    if args.json:
        print(json.dumps([vars(r) | {"path": str(r.path)} for r in results], indent=2, default=str))
    else:
        for result in results:
            if result.ok:
                print(
                    f"OK    {result.first_spectrum_id:<24} {result.n_spectra:>6} spec  "
                    f"{result.n_points:>5} pts  sha256={result.sha256[:12]}  {result.path}"
                )
            else:
                print(f"FAIL  {result.path}: {result.error}", file=sys.stderr)
    print(f"{len(results) - len(failures)}/{len(results)} verified", file=sys.stderr)
    return EXIT_USIS_ERROR if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser, including the verbs that are not implemented yet."""
    parser = argparse.ArgumentParser(
        prog="usis",
        description=(
            "USIS - a physics-grounded, uncertainty-aware research instrument for optical "
            "spectroscopy. A spectrum is a physical measurement of matter; identification is an "
            "ill-posed inverse problem whose answer is a posterior plus an explicit 'none of "
            "these' mass."
        ),
        epilog="Verbs marked (planned) exit non-zero and name the milestone that delivers them.",
    )
    parser.add_argument("--version", action="version", version=f"usis {__version__}")
    subparsers = parser.add_subparsers(dest="verb", metavar="<verb>")

    info = subparsers.add_parser("info", help="print version, git state and environment")
    info.add_argument("--json", action="store_true", help="emit JSON")
    info.set_defaults(func=cmd_info)

    licences = subparsers.add_parser(
        "licences", help="list the licence vocabulary and the vendoring gate"
    )
    licences.add_argument("--json", action="store_true", help="emit JSON")
    licences.set_defaults(func=cmd_licences)

    convert = subparsers.add_parser(
        "convert", help="convert values between axis or intensity conventions"
    )
    convert.add_argument("kind", choices=("axis", "intensity"))
    convert.add_argument("--from", dest="source", required=True, help="source convention")
    convert.add_argument("--to", dest="target", required=True, help="target convention")
    convert.add_argument("values", nargs="+", type=float, help="values to convert")
    convert.add_argument(
        "--excitation-nm",
        dest="excitation_nm",
        type=float,
        default=None,
        help="excitation wavelength; required to cross the Raman-shift boundary",
    )
    convert.add_argument(
        "--branch", choices=tuple(b.value for b in RamanBranch), default=RamanBranch.STOKES.value
    )
    convert.add_argument("--json", action="store_true", help="emit JSON")
    convert.set_defaults(func=cmd_convert)

    run_context = subparsers.add_parser(
        "run-context", help="capture and print a reproducible run context"
    )
    run_context.add_argument("run_id", help="a stable identifier for this run")
    run_context.add_argument("--seed", type=int, default=0)
    run_context.add_argument("--notes", default="")
    run_context.set_defaults(func=cmd_run_context)

    methods = subparsers.add_parser(
        "methods", help="list the method registry with maturity, assumptions and evidence"
    )
    methods.add_argument(
        "--maturity",
        default=None,
        help="comma-separated maturity levels to show, e.g. established,emerging",
    )
    methods.add_argument("--json", action="store_true")
    methods.set_defaults(func=cmd_methods)

    data = subparsers.add_parser("data", help="list, pull and verify the corpus")
    data_sub = data.add_subparsers(dest="data_verb", metavar="<subverb>")
    data.set_defaults(func=lambda _args, parser=data: (parser.print_help(), 0)[1])

    data_list = data_sub.add_parser("list", help="list the dataset manifest and the licence gate")
    data_list.add_argument("--json", action="store_true")
    data_list.set_defaults(func=cmd_data_list)

    data_pull = data_sub.add_parser("pull", help="fetch records or bulk files (network)")
    data_pull.add_argument("dataset", help="dataset id, e.g. rod")
    data_pull.add_argument("records", nargs="*", help="record ids to fetch")
    data_pull.add_argument(
        "--all", action="store_true", help="every record in the committed index for this dataset"
    )
    data_pull.add_argument(
        "--bulk", action="store_true", help="fetch the whole published file(s) instead of records"
    )
    data_pull.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        dest="max_bytes",
        help="with --bulk: take only this many bytes, cut back to the last complete line",
    )
    data_pull.add_argument(
        "--refetch", action="store_true", help="re-download records already on disk"
    )
    data_pull.add_argument("--dest", default=str(DEFAULT_DATA_DIR))
    data_pull.set_defaults(func=cmd_data_pull)

    data_verify = data_sub.add_parser("verify", help="re-parse every record on disk (offline)")
    data_verify.add_argument("--dest", default=str(DEFAULT_DATA_DIR))
    data_verify.add_argument("--dataset", default=None, help="limit to one dataset id")
    data_verify.add_argument("--json", action="store_true")
    data_verify.set_defaults(func=cmd_data_verify)

    compare = subparsers.add_parser(
        "compare", help="run every registered method for a task and write a comparison record"
    )
    compare.add_argument("config", help="committed experiment config, e.g. configs/*.yaml")
    compare.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        dest="overrides",
        help="OmegaConf dotlist override, e.g. -o seed=3. Recorded: a run with an override is "
        "not reproducible from the committed file alone, and the record says so.",
    )
    compare.add_argument("--dest", default=str(DEFAULT_DATA_DIR), help="mirrored corpus root")
    compare.add_argument(
        "--out", default=None, help="directory for comparison.json, the result card and run context"
    )
    compare.add_argument(
        "--maturity",
        default=None,
        help="comma-separated maturity levels to enter, e.g. established,emerging. Omitted, every "
        "registered method for the task runs - which is the point of a leaderboard.",
    )
    compare.add_argument("--json", action="store_true", help="print the record instead of a table")
    compare.set_defaults(func=cmd_compare)

    for verb, help_text in (
        ("analyze", "analyse a spectrum and emit a Sample Report (planned)"),
        ("simulate", "simulate a spectrum from a structure (planned)"),
    ):
        planned = subparsers.add_parser(verb, help=help_text, add_help=True)
        planned.add_argument("args", nargs="*", help=argparse.SUPPRESS)
        planned.set_defaults(func=lambda _args, verb=verb: _not_implemented(verb))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than calling ``sys.exit`` itself."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except UsisError as exc:
        print(f"usis: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USIS_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
