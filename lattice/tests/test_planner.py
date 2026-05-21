"""Tests for the deterministic task decomposer + multi-subtask runner."""

from __future__ import annotations

import textwrap

from lattice.actions import (
    AddImport,
    AddParameter,
    FileRef,
    MarkDone,
    SymbolRef,
    TypeExpr,
)
from lattice.compiler import DictWorkspace
from lattice.orchestrator import decompose, run_subtasks
from lattice.propose.mock import MockProposer


class TestDecompose:
    def test_single_task(self):
        assert decompose("add stripe import") == ["add stripe import"]

    def test_split_on_then(self):
        out = decompose("add stripe import then add dry_run parameter")
        assert out == ["add stripe import", "add dry_run parameter"]

    def test_split_on_and_then(self):
        out = decompose("rename foo to bar and then add a test for it")
        assert out == ["rename foo to bar", "add a test for it"]

    def test_split_on_semicolon(self):
        out = decompose("step one; step two; step three")
        assert out == ["step one", "step two", "step three"]

    def test_numbered_list(self):
        out = decompose("Do these: 1. add import 2. add field 3. add test")
        assert out[-3:] == ["add import", "add field", "add test"]

    def test_empty_returns_empty(self):
        assert decompose("") == []
        assert decompose("   ") == []

    def test_strips_trailing_punctuation(self):
        out = decompose("alpha; beta.")
        assert out == ["alpha", "beta"]


class TestMultiSubtaskRunner:
    def _ws(self) -> DictWorkspace:
        return DictWorkspace(
            {
                "src/payments/charge.py": textwrap.dedent("""\
                    \"\"\"Charge processing.\"\"\"

                    def charge(amount: int) -> None:
                        pass
                """),
            }
        )

    def test_two_subtasks_share_overlay(self):
        # subtask 1: AddImport ; subtask 2: AddParameter ; each ends with MarkDone
        proposer = MockProposer(
            batches=[
                [AddImport(file=FileRef(path="src/payments/charge.py"), module="stripe", confidence=0.9)],
                [MarkDone(summary="import done", confidence=0.9)],
                [
                    AddParameter(
                        function=SymbolRef(file="src/payments/charge.py", name="charge"),
                        name="dry_run",
                        type=TypeExpr(expr="bool"),
                        keyword_only=True,
                        confidence=0.9,
                    )
                ],
                [MarkDone(summary="param done", confidence=0.9)],
            ]
        )
        ws = self._ws()
        report = run_subtasks(
            ["add the stripe import", "add a dry_run keyword-only parameter"],
            proposer=proposer,
            workspace=ws,
        )
        assert report.ok
        assert report.terminated_by == "done"
        assert len(report.traces) == 2
        assert all(t.ok for t in report.traces)
        final = report.final_files["src/payments/charge.py"]
        assert "import stripe" in final
        assert "*, dry_run: bool" in final

    def test_advances_on_stuck_without_mark_done(self):
        """Stack-machine semantics: when the next instruction would be a
        no-op (stuck), the program counter advances to the next subtask.
        The LLM never has to emit MarkDone.
        """
        add_stripe = AddImport(
            file=FileRef(path="src/payments/charge.py"),
            module="stripe",
            confidence=0.9,
        )
        add_param = AddParameter(
            function=SymbolRef(file="src/payments/charge.py", name="charge"),
            name="dry_run",
            type=TypeExpr(expr="bool"),
            keyword_only=True,
            confidence=0.9,
        )
        proposer = MockProposer(
            batches=[
                [add_stripe],   # subtask 1, step 1: applies
                [add_stripe],   # subtask 1, step 2: no-op
                [add_stripe],   # subtask 1, step 3: no-op -> stuck -> advance
                [add_param],    # subtask 2, step 1: applies
                [add_param],    # subtask 2, step 2: no-op
                [add_param],    # subtask 2, step 3: no-op -> stuck -> advance
            ]
        )
        ws = self._ws()
        report = run_subtasks(
            ["add stripe", "add dry_run"],
            proposer=proposer,
            workspace=ws,
            max_steps_per_subtask=3,
        )
        assert report.ok
        assert report.terminated_by == "done"
        assert all(t.terminated_by == "stuck" for t in report.traces)
        final = report.final_files["src/payments/charge.py"]
        assert "import stripe" in final
        assert "*, dry_run: bool" in final

    def test_blocked_subtask_short_circuits(self):
        from lattice.actions import MarkBlocked

        proposer = MockProposer(
            batches=[
                [MarkBlocked(reason_code="needs_human", detail="?", confidence=0.9)],
            ]
        )
        ws = self._ws()
        report = run_subtasks(
            ["subtask one (will block)", "subtask two (never runs)"],
            proposer=proposer,
            workspace=ws,
        )
        assert not report.ok
        assert report.terminated_by == "blocked"
        assert len(report.traces) == 1

    def test_partial_when_subtask_exhausts(self):
        from lattice.actions import Branch

        # Subtask 1 exhausts (Branch every turn, never MarkDone).
        # Subtask 2 completes.
        proposer = MockProposer(
            batches=[
                [Branch(rationale="x", confidence=0.5)],
                [Branch(rationale="x", confidence=0.5)],
                [MarkDone(summary="ok", confidence=0.9)],
            ]
        )
        ws = self._ws()
        report = run_subtasks(
            ["never finishes", "this one does"],
            proposer=proposer,
            workspace=ws,
            max_steps_per_subtask=2,
        )
        assert not report.ok
        assert report.terminated_by == "partial"
