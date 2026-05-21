"""Tests for the apply-to-disk path.

Pure write semantics; we don't go through the compiler here because
its output is already covered by test_compiler.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lattice.actions import AddImport, FileRef
from lattice.compiler import (
    DictWorkspace,
    WorkspaceError,
    compile_action,
)
from lattice.compiler.types import CompiledAction, FileChange
from lattice.apply import write_compiled, write_final


def test_writes_changed_file(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")

    ws = DictWorkspace({"a.py": "x = 1\n"})
    compiled = compile_action(
        AddImport(file=FileRef(path="a.py"), module="json", confidence=0.9), ws
    )
    written = write_compiled(compiled, root=tmp_path)
    assert written == ["a.py"]
    after = target.read_text(encoding="utf-8")
    assert "import json" in after


def test_writes_creates_parent_dirs(tmp_path: Path):
    compiled = CompiledAction(
        verb="AddImport",
        file_changes=(
            FileChange(
                path="nested/new.py",
                before="",
                after="import json\n",
                diff="x",
            ),
        ),
    )
    written = write_compiled(compiled, root=tmp_path)
    assert written == ["nested/new.py"]
    assert (tmp_path / "nested" / "new.py").read_text(encoding="utf-8") == "import json\n"


def test_skips_noop(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")

    compiled = CompiledAction(
        verb="AddImport",
        file_changes=(
            FileChange(path="a.py", before="x = 1\n", after="x = 1\n", diff=""),
        ),
    )
    written = write_compiled(compiled, root=tmp_path)
    assert written == []
    assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_rejects_absolute_path(tmp_path: Path):
    compiled = CompiledAction(
        verb="AddImport",
        file_changes=(
            FileChange(path="/etc/passwd", before="", after="rooted!", diff="x"),
        ),
    )
    with pytest.raises(WorkspaceError):
        write_compiled(compiled, root=tmp_path)


def test_rejects_traversal(tmp_path: Path):
    compiled = CompiledAction(
        verb="AddImport",
        file_changes=(
            FileChange(path="a/../../x.py", before="", after="rooted!", diff="x"),
        ),
    )
    with pytest.raises(WorkspaceError):
        write_compiled(compiled, root=tmp_path)


def test_write_final_from_report_shape(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")

    class FakeReport:
        final_files = {"a.py": "x = 2\n"}

    written = write_final(FakeReport(), root=tmp_path)
    assert written == ["a.py"]
    assert target.read_text(encoding="utf-8") == "x = 2\n"


def test_write_final_empty_report(tmp_path: Path):
    class FakeReport:
        final_files: dict[str, str] = {}

    assert write_final(FakeReport(), root=tmp_path) == []
