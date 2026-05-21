"""Tests for the test-execution verify gate."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lattice.compiler.types import CompiledAction, FileChange
from lattice.verify import TestVerifyOutcome, verify_tests


def _change(path: str, after: str) -> FileChange:
    return FileChange(path=path, before="", after=after, diff="diff")


def _compiled(*changes: FileChange) -> CompiledAction:
    return CompiledAction(verb="AddImport", file_changes=tuple(changes))


def _project(tmp_path: Path, src: str, tests: str | None = None) -> Path:
    """Set up a tiny src-layout project. Returns workspace root."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.py").write_text(src, encoding="utf-8")
    if tests is not None:
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_foo.py").write_text(tests, encoding="utf-8")
    return tmp_path


def test_no_op_compile_returns_ok(tmp_path: Path):
    """Empty change set short-circuits to ok=True."""
    out = verify_tests(_compiled(), workspace_root=tmp_path)
    assert out.ok
    assert "nothing to test" in out.notice


def test_no_tests_discovered_returns_skipped(tmp_path: Path):
    """No matching test files -> skipped (not a failure)."""
    _project(tmp_path, "def f():\n    return 1\n", tests=None)
    out = verify_tests(
        _compiled(_change("src/foo.py", "def f():\n    return 2\n")),
        workspace_root=tmp_path,
    )
    assert out.ok
    assert out.skipped
    assert out.ran == 0


def test_passes_when_tests_pass(tmp_path: Path):
    """Affected test still passes after the change -> ok."""
    _project(
        tmp_path,
        "def double(x):\n    return x * 2\n",
        tests=textwrap.dedent(
            """\
            from src.foo import double

            def test_double():
                assert double(3) == 6
            """
        ),
    )
    after = "def double(x):\n    return x * 2  # unchanged behaviour\n"
    out = verify_tests(
        _compiled(_change("src/foo.py", after)),
        workspace_root=tmp_path,
    )
    assert out.ok, out.errors
    assert out.ran >= 1
    assert not out.skipped


def test_catches_test_failures(tmp_path: Path):
    """The verify gate's job: catch behavior changes that break tests."""
    _project(
        tmp_path,
        "def double(x):\n    return x * 2\n",
        tests=textwrap.dedent(
            """\
            from src.foo import double

            def test_double():
                assert double(3) == 6
            """
        ),
    )
    # Mutate behavior — return 0 instead of x*2.
    broken = "def double(x):\n    return 0\n"
    out = verify_tests(
        _compiled(_change("src/foo.py", broken)),
        workspace_root=tmp_path,
    )
    assert not out.ok
    assert out.errors
    # At least one failure should mention the failing test by id.
    assert any("test_double" in test_id for test_id, _ in out.errors)


def test_explicit_test_paths(tmp_path: Path):
    """Caller can pass test_paths explicitly; auto-discovery is bypassed."""
    _project(
        tmp_path,
        "def f():\n    return 1\n",
        tests=textwrap.dedent(
            """\
            from src.foo import f

            def test_f():
                assert f() == 1
            """
        ),
    )
    out = verify_tests(
        _compiled(_change("src/foo.py", "def f():\n    return 1\n")),
        workspace_root=tmp_path,
        test_paths=["tests/test_foo.py"],
    )
    assert out.ok


def test_workspace_root_missing(tmp_path: Path):
    """Nonexistent workspace root -> ok=False."""
    bogus = tmp_path / "does_not_exist"
    out = verify_tests(
        _compiled(_change("src/foo.py", "x = 1")),
        workspace_root=bogus,
    )
    assert not out.ok


def test_pytest_not_importable_is_skipped(monkeypatch, tmp_path: Path):
    """Graceful fallback when pytest isn't installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pytest":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _project(tmp_path, "x = 1\n")
    out = verify_tests(
        _compiled(_change("src/foo.py", "x = 2\n")),
        workspace_root=tmp_path,
    )
    assert out.ok
    assert out.skipped
    assert "pytest not installed" in out.notice


def test_outcome_is_truthy_only_when_ok(tmp_path: Path):
    assert bool(TestVerifyOutcome(ok=True))
    assert not bool(TestVerifyOutcome(ok=False, errors=(("t", "m"),)))
