"""Command-line driver for LATTICE.

Currently exposes one subcommand:

    python -m lattice apply <workspace_root> [--action JSON | -]

Reads a typed Action JSON, compiles it against the workspace, verifies
the resulting source parses, and writes the unified diff to stdout.
Exit code is 0 on success, 1 on compile/verify failure, 2 on input
errors. No filesystem mutation — diffs are printed, not applied.

Used as the smoke-test entry point until the full reasoning loop lands.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import IO

from pydantic import ValidationError

from lattice.actions import parse_action
from lattice.compiler import (
    CompileError,
    FilesystemWorkspace,
    compile_action,
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
