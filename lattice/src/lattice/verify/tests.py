"""Test-execution verify gate — runs pytest on affected tests.

Goes beyond parse + type-check by actually executing the tests that
exercise the changed code. Real ground truth for 'did this work?'.

Strategy:
1. Copy the workspace root to a tempdir.
2. Apply the compiled FileChange diffs to the tempdir.
3. Resolve which test files to run (caller-supplied, or auto-discover
   from the file basenames touched).
4. Invoke pytest in that tempdir with the venv's interpreter, short
   traceback, fail-fast.
5. Parse the result.

Auto-discovery heuristic: for every touched file `<dir>/foo.py`, look
for a sibling `test_foo.py`, a `tests/test_foo.py`, or any test file
that imports `foo` by name. Imperfect but cheap and works for the
common test-mirroring conventions.

pytest is an optional dep; when unavailable the gate gracefully
skips (ok=True, skipped=True) so the orchestrator stack is not
broken.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict

from lattice.compiler.types import CompiledAction


class TestVerifyOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Tell pytest's collector to skip this Pydantic model — the
    # 'Test' prefix would otherwise trigger a PytestCollectionWarning.
    __test__ = False

    ok: bool
    errors: tuple[tuple[str, str], ...] = ()  # (test_id, message)
    skipped: bool = False
    notice: str = ""
    ran: int = 0  # tests executed

    def __bool__(self) -> bool:
        return self.ok


def verify_tests(
    compiled: CompiledAction,
    *,
    workspace_root: str | Path,
    test_paths: Iterable[str] | None = None,
    timeout_s: float = 60.0,
) -> TestVerifyOutcome:
    """Run pytest against the affected tests in a sandbox.

    The sandbox is a tempdir copy of *workspace_root* with the
    compiled FileChanges applied on top. Returns ok=True/skipped=True
    if pytest isn't importable (graceful fallback). Returns ok=False
    with one (test_id, message) tuple per failure on real test
    failures.
    """
    changed = [c for c in compiled.file_changes if not c.is_noop]
    if not changed:
        return TestVerifyOutcome(
            ok=True, notice="no-op compile; nothing to test"
        )

    try:
        import pytest  # noqa: F401
    except ImportError:
        return TestVerifyOutcome(
            ok=True,
            skipped=True,
            notice="pytest not installed; install with `uv pip install pytest` to enable",
        )

    root = Path(workspace_root).resolve()
    if not root.is_dir():
        return TestVerifyOutcome(
            ok=False,
            errors=(("", f"workspace_root {root} is not a directory"),),
        )

    with tempfile.TemporaryDirectory(prefix="lattice-tests-") as td_str:
        td = Path(td_str)
        sandbox = td / "ws"
        # Copy the workspace, skipping conventional non-source dirs.
        shutil.copytree(
            root,
            sandbox,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "node_modules",
                ".pytest_cache", ".lattice", "dist", "build",
                "*.egg-info",
            ),
        )
        # Apply the compiled changes.
        for change in changed:
            target = sandbox / change.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(change.after, encoding="utf-8")

        # Resolve test paths.
        if test_paths is not None:
            tests_to_run = [str(sandbox / p) for p in test_paths]
        else:
            tests_to_run = _discover_affected_tests(changed, sandbox)
        if not tests_to_run:
            return TestVerifyOutcome(
                ok=True,
                skipped=True,
                notice="no tests discovered for the changed files",
            )

        cmd = [
            sys.executable, "-m", "pytest",
            "--no-header", "-q", "-x",
            "--tb=short",
            *tests_to_run,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=sandbox,
                timeout=timeout_s,
                env=_pytest_env(sandbox),
            )
        except subprocess.TimeoutExpired:
            return TestVerifyOutcome(
                ok=False,
                errors=(("", f"pytest timed out after {timeout_s:.0f}s"),),
            )

        ran, failures = _parse_pytest_output(result.stdout)
        if result.returncode == 0:
            return TestVerifyOutcome(ok=True, ran=ran)

        if not failures:
            # Pytest exited non-zero but we couldn't parse a failure;
            # surface a slice of stderr/stdout so the caller sees something.
            tail = (result.stdout + "\n" + result.stderr)[-400:]
            failures = (("pytest", tail.strip()),)
        return TestVerifyOutcome(
            ok=False,
            errors=failures,
            ran=ran,
        )


def _pytest_env(sandbox: Path) -> dict[str, str]:
    """Environment for the pytest subprocess.

    Adds `<sandbox>/src` and `<sandbox>` to PYTHONPATH so simple
    src-layout projects without explicit installation resolve.
    """
    import os

    env = dict(os.environ)
    extra = [str(sandbox / "src"), str(sandbox)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        ":".join([*extra, existing]) if existing else ":".join(extra)
    )
    return env


def _imported_module_names(source: str) -> set[str]:
    """Parse `source` and return the set of module BASENAMES it imports.

    Handles `import foo`, `import foo.bar`, `from foo import x`,
    `from .foo import x`, and aliases via `as`. For each import target,
    we return the *final* dotted-name component because that's what
    the lattice's `touched_modules` set contains (Path.stem of each
    changed source file).

    Returns an empty set on parse error — better to under-trigger
    than to error the test-discovery pass.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # 'import foo.bar.baz' contributes 'foo', 'bar', 'baz'
                # — touched_modules contains basenames, so any final
                # component matching is a hit.
                out.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.update(node.module.split("."))
            for alias in node.names:
                out.add(alias.name)
    return out


def _discover_affected_tests(
    changed: list, sandbox: Path
) -> list[str]:
    """Resolve which test files to run given the changed file paths.

    Two strategies, in order:
    1. Direct mirror: `src/foo.py` -> `tests/test_foo.py` /
       `tests/<dir>/test_foo.py` / `test_foo.py`.
    2. Reference search: any test_*.py file whose contents mention
       a touched module name (basename-without-.py).

    Returns deduplicated paths relative to `sandbox`. Skips test files
    that are themselves touched (they'd be tested by their own change).
    """
    out: list[Path] = []
    seen: set[Path] = set()

    for change in changed:
        rel = Path(change.path)
        if "test_" in rel.name:
            continue  # changes to test files don't trigger further runs
        stem = rel.stem
        candidates = (
            sandbox / "tests" / f"test_{stem}.py",
            sandbox / "tests" / rel.parent.name / f"test_{stem}.py",
            sandbox / f"test_{stem}.py",
            sandbox / rel.parent / f"test_{stem}.py",
        )
        for cand in candidates:
            if cand.is_file() and cand not in seen:
                out.append(cand)
                seen.add(cand)

    # Reference search: walk every test_*.py and look for an actual
    # IMPORT of a touched module. Previously this was a substring
    # match on the basename — that produced both false positives
    # (the basename appears in a comment or docstring) and false
    # negatives (an alias import like 'from . import atom as atm'
    # buries the basename). AST parsing the test file and matching
    # on canonical import targets fixes both directions.
    touched_modules = {Path(c.path).stem for c in changed}
    if touched_modules:
        for test_file in sandbox.rglob("test_*.py"):
            if not test_file.is_file() or test_file in seen:
                continue
            try:
                content = test_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            imported = _imported_module_names(content)
            if imported & touched_modules:
                out.append(test_file)
                seen.add(test_file)

    return [str(p) for p in out]


def _parse_pytest_output(stdout: str) -> tuple[int, tuple[tuple[str, str], ...]]:
    """Pull (test_count, failures) from pytest's stdout.

    Looks for the summary line ('5 passed', '1 failed, 4 passed') and
    the FAILED <nodeid> tail. Imperfect but robust enough for the
    short-traceback format we invoke pytest with.
    """
    import re

    ran = 0
    summary = re.search(
        r"(\d+)\s+(?:passed|failed|errors?)",
        stdout,
    )
    if summary:
        for m in re.finditer(r"(\d+)\s+(passed|failed|errors?)", stdout):
            ran += int(m.group(1))

    failures: list[tuple[str, str]] = []
    for line in stdout.splitlines():
        if line.startswith("FAILED "):
            parts = line.split(" - ", 1)
            test_id = parts[0][len("FAILED "):].strip()
            message = parts[1].strip() if len(parts) > 1 else ""
            failures.append((test_id, message))
        elif line.startswith("ERROR "):
            parts = line.split(" - ", 1)
            test_id = parts[0][len("ERROR "):].strip()
            message = parts[1].strip() if len(parts) > 1 else "error"
            failures.append((test_id, message))

    return ran, tuple(failures)
