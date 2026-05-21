"""Command-line driver for LATTICE.

Subcommands:

    python -m lattice apply  <workspace> [--action JSON | -]
    python -m lattice intent <workspace> [--intent JSON | -] [--files a.py,b.py]

`apply` takes a single typed Action and prints its diff.
`intent` takes a high-level Intent, expands it into many typed Actions
(the composition spine: intent → primitives), compiles and verifies
each, and prints a multi-file unified diff. No filesystem mutation —
diffs are printed, not applied.

Exit: 0 on success, 1 on compile/verify failure, 2 on input errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import IO

from pydantic import TypeAdapter, ValidationError

from lattice.actions import parse_action
from lattice.compiler import (
    CompileError,
    FilesystemWorkspace,
    compile_action,
)
from lattice.orchestrator import (
    ExecutionReport,
    Intent,
    execute_plan,
    expand_intent,
)
from lattice.verify import verify_syntactic


def _read_action_json(arg: str | None, stdin: IO[str]) -> dict:
    if arg is None or arg == "-":
        raw = stdin.read()
    elif arg.startswith("@"):
        with open(arg[1:], encoding="utf-8") as fh:
            raw = fh.read()
    else:
        raw = arg
    raw = raw.strip()
    if not raw:
        raise SystemExit("error: empty action JSON on stdin")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: action JSON is malformed: {exc}") from exc


def _apply(args: argparse.Namespace) -> int:
    workspace = FilesystemWorkspace(args.workspace)
    payload = _read_action_json(args.action, sys.stdin)
    try:
        action = parse_action(payload)
    except ValidationError as exc:
        sys.stderr.write(f"action did not validate:\n{exc}\n")
        return 2

    try:
        compiled = compile_action(action, workspace)
    except CompileError as exc:
        sys.stderr.write(f"compile error ({type(exc).__name__}): {exc}\n")
        return 1

    outcome = verify_syntactic(compiled)
    if not outcome.ok:
        for path, msg in outcome.errors:
            sys.stderr.write(f"verify failed in {path}: {msg}\n")
        return 1

    if compiled.is_noop:
        sys.stderr.write("(no-op: action did not change the workspace)\n")
        return 0

    for change in compiled.file_changes:
        if change.diff:
            sys.stdout.write(change.diff)
            if not change.diff.endswith("\n"):
                sys.stdout.write("\n")
    return 0


_intent_adapter: TypeAdapter[Intent] = TypeAdapter(Intent)


def _intent(args: argparse.Namespace) -> int:
    workspace = FilesystemWorkspace(args.workspace)
    payload = _read_action_json(args.intent, sys.stdin)
    try:
        intent = _intent_adapter.validate_python(payload)
    except ValidationError as exc:
        sys.stderr.write(f"intent did not validate:\n{exc}\n")
        return 2

    if args.files:
        files = [p.strip() for p in args.files.split(",") if p.strip()]
    else:
        files = workspace.iter_files(suffix=".py")

    actions = expand_intent(intent, workspace, files)
    if not actions:
        sys.stderr.write("(intent expanded to zero actions — no matching symbols)\n")
        return 0

    sys.stderr.write(f"expanded to {len(actions)} action(s)\n")
    report: ExecutionReport = execute_plan(actions, workspace)

    if not report.ok:
        for step in report.steps:
            if step.ok:
                continue
            if step.error:
                sys.stderr.write(f"  step failed: {step.error}\n")
            elif step.verify and step.verify.errors:
                for path, msg in step.verify.errors:
                    sys.stderr.write(f"  verify failed {path}: {msg}\n")
        return 1

    # Consolidated diffs (one per touched file) are more readable than
    # per-step diffs when actions chain. Both are available on the report.
    for diff in report.consolidated_diffs:
        sys.stdout.write(diff)
        if not diff.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lattice")
    sub = parser.add_subparsers(dest="cmd", required=True)

    apply_p = sub.add_parser("apply", help="Compile and print the diff for a typed action.")
    apply_p.add_argument("workspace", help="Repository root the action operates against.")
    apply_p.add_argument(
        "--action",
        default="-",
        help="Action JSON: literal string, '-' for stdin, or '@FILE' to read from a file.",
    )
    apply_p.set_defaults(func=_apply)

    intent_p = sub.add_parser(
        "intent",
        help="Expand a high-level intent into many typed actions and print the diffs.",
    )
    intent_p.add_argument("workspace", help="Repository root the intent operates against.")
    intent_p.add_argument(
        "--intent",
        default="-",
        help="Intent JSON: literal string, '-' for stdin, or '@FILE' to read from a file.",
    )
    intent_p.add_argument(
        "--files",
        default="",
        help="Comma-separated repository-relative .py paths. Default: walk workspace.",
    )
    intent_p.set_defaults(func=_intent)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
