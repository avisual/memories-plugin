"""Type-check verification — runs mypy on the changed files.

Goes beyond the syntactic gate (ast.parse) by actually type-checking
the post-edit code. Catches real bugs the parser misses: undefined
names, wrong call signatures, missing imports of types used in
annotations.

mypy is an optional dep — install with `uv pip install -e '.[typecheck]'`.
When unavailable, `verify_types` returns ok=True with a notice so the
orchestrator's stack stays unbroken.

Runs in a tempdir copy of the workspace's edited files only, so it's
fast (no whole-project type-check), and the underlying workspace is
never touched.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from lattice.compiler.types import CompiledAction


class TypeCheckOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    ok: bool
    errors: tuple[tuple[str, str], ...] = ()  # (path, message)
    skipped: bool = False  # True when mypy isn't installed
    notice: str = ""

    def __bool__(self) -> bool:
        return self.ok


def verify_types(
    compiled: CompiledAction,
    *,
    strict: bool = False,
    timeout_s: float = 20.0,
) -> TypeCheckOutcome:
    """Type-check the after-content of every changed file with mypy.

    Returns ok=True if mypy reports no errors. Returns
    ok=True, skipped=True if mypy isn't installed (graceful fallback).
    Returns ok=False with one error tuple per failing file otherwise.
    """
    changed = [c for c in compiled.file_changes if not c.is_noop]
    if not changed:
        return TypeCheckOutcome(ok=True, notice="no-op compile; nothing to type-check")

    try:
        import mypy.api  # noqa: F401
    except ImportError:
        return TypeCheckOutcome(
            ok=True,
            skipped=True,
            notice="mypy not installed (run: uv pip install -e '.[typecheck]')",
        )

    with tempfile.TemporaryDirectory(prefix="lattice-mypy-") as td_str:
        td = Path(td_str)
        for change in changed:
            target = td / change.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(change.after, encoding="utf-8")

        cmd = [sys.executable, "-m", "mypy", "--no-incremental", "--no-error-summary"]
        if strict:
            cmd.append("--strict")
        else:
            cmd += ["--ignore-missing-imports", "--check-untyped-defs"]
        cmd += [str(td / c.path) for c in changed]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=td,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return TypeCheckOutcome(
                ok=False,
                errors=(("", f"mypy timed out after {timeout_s}s"),),
            )

        if result.returncode == 0:
            return TypeCheckOutcome(ok=True)

        errors = _parse_mypy_output(result.stdout, td)
        return TypeCheckOutcome(ok=False, errors=errors)


def _parse_mypy_output(stdout: str, root: Path) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for line in stdout.splitlines():
        # mypy lines look like: /tmp/lattice-.../path/to/file.py:12: error: msg
        if ": error:" not in line and ": note:" not in line:
            continue
        try:
            path_part, _, message = line.partition(": ")
            file_path, _, _ = path_part.rpartition(":")
            rel = str(Path(file_path).relative_to(root))
        except (ValueError, IndexError):
            rel = ""
            message = line
        out.append((rel, message))
    return tuple(out)
