"""Tests for the agent loop.

Uses MockProposer so the LLM is deterministic and tests run in
milliseconds. Real-LLM behavior is exercised by the propose CLI demo.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from lattice.actions import (
    AddImport,
    AddParameter,
    Branch,
    FileRef,
    MarkBlocked,
    MarkDone,
    RecallMore,
    RevealBody,
    SymbolRef,
    TypeExpr,
)
from lattice.compiler import DictWorkspace
from lattice.orchestrator import AgentLoop, AgentTrace
from lattice.propose.mock import MockProposer


def _ws() -> DictWorkspace:
    return DictWorkspace(
        {
            "src/payments/charge.py": textwrap.dedent("""\
                \"\"\"Charge processing.\"\"\"

                def charge(amount: int) -> None:
                    pass
            """),
        }
    )


def test_terminates_on_mark_done():
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/payments/charge.py"), module="stripe", confidence=0.9)],
            [MarkDone(summary="added stripe import", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=_ws())
    trace: AgentTrace = loop.run("add stripe import")
    assert trace.ok
    assert trace.terminated_by == "done"
    assert len(trace.steps) == 2
    assert trace.steps[0].kind == "edit"
    assert "import stripe" in trace.final_files["src/payments/charge.py"]


def test_terminates_on_mark_blocked():
    proposer = MockProposer(
        batches=[
            [MarkBlocked(reason_code="needs_human", detail="ambiguous", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=_ws())
    trace = loop.run("do something ambiguous")
    assert not trace.ok
    assert trace.terminated_by == "blocked"
    assert trace.steps[0].kind == "blocked"


def test_exhausts_max_steps():
    proposer = MockProposer(
        batches=[
            [Branch(rationale="step 1", confidence=0.5)],
            [Branch(rationale="step 2", confidence=0.5)],
            [Branch(rationale="step 3", confidence=0.5)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=_ws(), max_steps=2)
    trace = loop.run("loop forever")
    assert trace.terminated_by == "exhausted"
    assert len(trace.steps) == 2


def test_terminates_on_empty_propose():
    proposer = MockProposer.empty()
    loop = AgentLoop(proposer=proposer, workspace=_ws())
    trace = loop.run("?")
    assert trace.terminated_by == "empty"
    assert trace.steps == ()


def test_chains_two_edits_then_done():
    ws = _ws()
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/payments/charge.py"), module="stripe", confidence=0.9)],
            [
                AddParameter(
                    function=SymbolRef(file="src/payments/charge.py", name="charge"),
                    name="dry_run",
                    type=TypeExpr(expr="bool"),
                    keyword_only=True,
                    confidence=0.9,
                )
            ],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=ws)
    trace = loop.run("add stripe + dry_run")
    assert trace.ok
    assert len(trace.steps) == 3
    final = trace.final_files["src/payments/charge.py"]
    assert "import stripe" in final
    assert "*, dry_run: bool" in final
    # Original workspace untouched.
    assert "import stripe" not in ws.read("src/payments/charge.py")


def test_compile_error_does_not_terminate(caplog):
    bad = AddParameter(
        function=SymbolRef(file="src/payments/charge.py", name="missing"),
        name="x",
        type=TypeExpr(expr="int"),
        confidence=0.8,
    )
    proposer = MockProposer(
        batches=[
            [bad],
            [MarkDone(summary="given up", confidence=0.5)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=_ws())
    trace = loop.run("doomed")
    assert trace.terminated_by == "done"
    assert trace.steps[0].kind == "error"
    assert "SymbolNotFound" in trace.steps[0].error


def test_recall_more_threads_through(tmp_path: Path):
    from lattice.atoms import SQLiteAtomStore

    class StubEmbedder:
        @property
        def dim(self) -> int:
            return 8

        def embed(self, texts):
            import numpy as np
            return np.ones((len(texts), 8), dtype="float32") / (8**0.5)

    db = tmp_path / "atoms.db"
    store = SQLiteAtomStore(db, embedder=StubEmbedder())
    store.add("relevant prior knowledge about stripe")

    proposer = MockProposer(
        batches=[
            [RecallMore(query="stripe", confidence=0.5)],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=_ws(), atom_store=store)
    trace = loop.run("look up stripe")
    assert trace.steps[0].kind == "recall"
    assert "stripe" in trace.steps[0].payload
    store.close()


def test_reveal_body_returns_source_snippet():
    proposer = MockProposer(
        batches=[
            [
                RevealBody(
                    symbol=SymbolRef(file="src/payments/charge.py", name="charge"),
                    confidence=0.5,
                )
            ],
            [MarkDone(summary="seen", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=_ws())
    trace = loop.run("show me charge")
    assert trace.steps[0].kind == "reveal"
    assert "def charge" in trace.steps[0].payload


def test_stuck_detection_breaks_no_op_loop():
    """When the proposer repeats an action that's already been applied,
    the harness should terminate with 'stuck' rather than spinning.
    """
    repeated = AddImport(
        file=FileRef(path="src/payments/charge.py"), module="stripe", confidence=0.9
    )
    proposer = MockProposer(
        batches=[[repeated], [repeated], [repeated], [repeated]]
    )
    loop = AgentLoop(proposer=proposer, workspace=_ws(), max_steps=10)
    trace = loop.run("loop me")
    assert trace.terminated_by == "stuck"
    # The first call applied; subsequent ones were no-ops (empty diff);
    # harness broke out before exhausting all batches.
    assert trace.steps[0].kind == "edit"
    assert trace.steps[0].diff != ""
    # At least one subsequent edit was a no-op.
    assert any(s.kind == "edit" and s.diff == "" for s in trace.steps[1:])
    # The loop terminated earlier than max_steps and earlier than the
    # proposer's 4 batches.
    assert len(trace.steps) < 4


def test_branch_is_noop_in_linear_loop():
    proposer = MockProposer(
        batches=[
            [Branch(rationale="try this way", confidence=0.5)],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=_ws())
    trace = loop.run("?")
    assert trace.steps[0].kind == "branch"
    assert trace.terminated_by == "done"
