"""Tests for the type-check verify gate.

Live mypy tests run only when mypy is importable; everywhere else the
verifier returns ok=True/skipped=True so the orchestrator stack stays
unbroken.
"""

from __future__ import annotations

import textwrap

import pytest

from lattice.compiler.types import CompiledAction, FileChange
from lattice.verify import TypeCheckOutcome, verify_types


def _change(path: str, after: str) -> FileChange:
    return FileChange(path=path, before="", after=after, diff="x")


def _compiled(*changes: FileChange) -> CompiledAction:
    return CompiledAction(verb="AddImport", file_changes=tuple(changes))


def test_noop_returns_ok():
    out = verify_types(_compiled())
    assert out.ok
    assert "no-op" in out.notice


def test_skipped_when_mypy_missing(monkeypatch):
    """If mypy isn't importable, fall back gracefully."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("mypy"):
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = verify_types(_compiled(_change("a.py", "x = 1\n")))
    assert out.ok
    assert out.skipped


# Live tests — run only if mypy is installed.
mypy_installed = pytest.importorskip("mypy", reason="mypy not installed; type-check tests skipped")


def test_clean_file_passes_type_check():
    src = textwrap.dedent("""\
        from typing import Iterable


        def head(items: Iterable[int]) -> int:
            for x in items:
                return x
            return 0
    """)
    out = verify_types(_compiled(_change("a.py", src)))
    assert out.ok, out.errors


def test_type_error_fails_check():
    src = textwrap.dedent("""\
        def add(a: int, b: int) -> int:
            return a + b


        x: int = add("hello", 3)
    """)
    out = verify_types(_compiled(_change("a.py", src)))
    assert not out.ok
    assert any("error" in msg.lower() for _, msg in out.errors)


def test_strict_catches_missing_annotations():
    src = "def f(x):\n    return x\n"
    out = verify_types(_compiled(_change("a.py", src)), strict=True)
    assert not out.ok


def test_non_strict_tolerates_missing_imports():
    src = "from nonexistent_module import thing\n\nx: int = thing()\n"
    out = verify_types(_compiled(_change("a.py", src)))
    # In non-strict mode (default), missing imports are tolerated.
    assert out.ok or out.skipped
