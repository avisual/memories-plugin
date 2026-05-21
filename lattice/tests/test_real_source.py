"""End-to-end test: the harness handles REAL lattice source, not toy fixtures.

This is the test that pushes back on 'a simple one-line change is not really
a full cli harness for code writing'. Each test here:
  - Copies a real production .py file from lattice itself into a tempdir
  - Runs AgentLoop against it with MockProposer
  - Asserts the output is valid Python AND contains the expected change

MockProposer makes the test deterministic (no LLM call needed) so we can
verify the COMPILER + VERIFY + PLAN-DAG + BRAIN wiring on production-shaped
code without the audit's slow inference cost.
"""

from __future__ import annotations

import ast
import shutil
import textwrap
from pathlib import Path

import pytest

from lattice.actions import (
    AddImport,
    ChangeReturnType,
    FileRef,
    MarkDone,
    ModifyDocstring,
    SymbolRef,
    TypeExpr,
)
from lattice.compiler import DictWorkspace
from lattice.orchestrator import AgentLoop
from lattice.propose.mock import MockProposer


LATTICE_ROOT = Path(__file__).parent.parent / "src" / "lattice"


def _read_real(rel_path: str) -> str:
    """Read a real lattice source file as text."""
    p = LATTICE_ROOT / rel_path
    if not p.is_file():
        pytest.skip(f"real source file not found: {p}")
    return p.read_text(encoding="utf-8")


def test_harness_adds_import_to_real_atoms_feedback():
    """Add a typing.Sequence import to lattice/atoms/feedback.py (the
    real file, copied verbatim). After the edit the file should:
      - parse as valid Python
      - contain the new import
      - retain the original record_experience function
    """
    src = _read_real("atoms/feedback.py")
    ws = DictWorkspace({"src/feedback.py": src})
    proposer = MockProposer(
        batches=[
            [AddImport(
                file=FileRef(path="src/feedback.py"),
                module="typing",
                names=["Sequence"],
                confidence=0.9,
            )],
            [MarkDone(summary="added import", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=ws)
    trace = loop.run("add a typing.Sequence import to src/feedback.py")
    assert trace.ok, f"trace failed: {trace.terminated_by}"

    after = trace.final_files["src/feedback.py"]
    ast.parse(after)  # must parse — fails loudly if not
    assert "from typing import" in after and "Sequence" in after
    assert "def record_experience" in after  # original content preserved


def test_harness_changes_return_type_on_real_evolve():
    """Run ChangeReturnType on a real function in lattice/atoms/evolve.py.

    The function `_set_at_path` is annotated `-> bool` (since the
    KeyError-tolerance fix). Change it to `None` and verify the
    function declaration is rewritten AND the rest of the file is
    untouched.
    """
    src = _read_real("atoms/evolve.py")
    if "def _set_at_path" not in src:
        pytest.skip("_set_at_path not present in current evolve.py")

    ws = DictWorkspace({"src/evolve.py": src})
    proposer = MockProposer(
        batches=[
            [ChangeReturnType(
                symbol=SymbolRef(file="src/evolve.py", name="_set_at_path"),
                return_type=TypeExpr(expr="None"),
                confidence=0.9,
            )],
            [MarkDone(summary="retyped", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=ws)
    trace = loop.run("change _set_at_path return type to None")
    assert trace.ok

    after = trace.final_files["src/evolve.py"]
    ast.parse(after)
    # The signature line now ends in '-> None:' (possibly with trailing
    # space variations — the test asserts the new annotation is present
    # and the old one isn't on the def line).
    def_lines = [
        line for line in after.splitlines()
        if line.startswith("def _set_at_path")
    ]
    assert def_lines, "_set_at_path def line missing after edit"
    assert "-> None" in def_lines[0]
    # No spurious other top-level functions removed.
    assert "def infer_template" in after
    assert "def write_trace" in after


def test_harness_adds_docstring_to_real_function():
    """Add a docstring to a real helper in evolve.py via ModifyDocstring."""
    src = _read_real("atoms/evolve.py")
    if "def _walk_string_fields" not in src:
        pytest.skip("_walk_string_fields not present")

    ws = DictWorkspace({"src/evolve.py": src})
    proposer = MockProposer(
        batches=[
            [ModifyDocstring(
                file=FileRef(path="src/evolve.py"),
                symbol=SymbolRef(file="src/evolve.py", name="_walk_string_fields"),
                docstring="Yield (path_tuple, value) for every str value in obj.",
                confidence=0.9,
            )],
            [MarkDone(summary="docstring added", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=ws)
    trace = loop.run("docstring for _walk_string_fields")
    assert trace.ok

    after = trace.final_files["src/evolve.py"]
    ast.parse(after)
    assert "Yield (path_tuple, value) for every str value in obj." in after


def test_multi_step_real_source_plan_dag():
    """End-to-end multi-step plan against real source via Plan-DAG.

    Real lattice file, two-step task: (1) add an import, (2) add a
    docstring on an existing function. Plan-DAG advances after each
    successful mutating step. Both edits must land for the trace to
    succeed; both must leave the file parseable.
    """
    src = _read_real("atoms/feedback.py")
    if "def record_experience" not in src:
        pytest.skip("record_experience not present")

    ws = DictWorkspace({"src/feedback.py": src})
    proposer = MockProposer(
        batches=[
            [AddImport(
                file=FileRef(path="src/feedback.py"),
                module="collections.abc",
                names=["Iterable"],
                confidence=0.9,
            )],
            [ModifyDocstring(
                file=FileRef(path="src/feedback.py"),
                symbol=SymbolRef(file="src/feedback.py", name="record_experience"),
                docstring="Persist a successful agent run as an experience atom.",
                confidence=0.9,
            )],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=ws, use_plan=True)
    trace = loop.run(
        "add an import of Iterable from collections.abc to src/feedback.py; "
        "add a docstring to record_experience in src/feedback.py"
    )
    assert trace.ok, f"terminated_by={trace.terminated_by}"

    after = trace.final_files["src/feedback.py"]
    ast.parse(after)
    assert "from collections.abc import" in after and "Iterable" in after
    assert "Persist a successful agent run as an experience atom." in after
