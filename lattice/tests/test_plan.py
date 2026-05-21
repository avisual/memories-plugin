"""Tests for Plan / PlanStep — multi-step task tracking."""

from __future__ import annotations

import textwrap

from lattice.actions import (
    AddImport,
    AddParameter,
    Expr,
    FileRef,
    MarkDone,
    SymbolRef,
    TypeExpr,
)
from lattice.compiler import DictWorkspace
from lattice.orchestrator import AgentLoop
from lattice.orchestrator.plan import Plan, PlanStep
from lattice.propose.mock import MockProposer


def test_plan_single_step_for_simple_task():
    """A single-clause task gets a one-element plan; is_multistep() is False."""
    plan = Plan.build("add an import of json to src/main.py")
    assert len(plan.steps) == 1
    assert plan.steps[0].idx == 1
    assert plan.steps[0].status == "pending"
    assert not plan.is_multistep()
    # Single-step plans render empty (no PLAN: header to waste tokens).
    assert plan.render() == ""


def test_plan_splits_on_semicolon():
    """The decompose heuristic splits on '; ' for multi-step tasks."""
    plan = Plan.build(
        "add import json to src/api.py; "
        "add a keyword-only parameter timeout to function get"
    )
    assert len(plan.steps) == 2
    assert plan.is_multistep()
    assert "import json" in plan.steps[0].description
    assert "timeout" in plan.steps[1].description


def test_plan_splits_on_then_keyword():
    plan = Plan.build("first do A then do B and then do C")
    assert len(plan.steps) >= 2  # 'and then' may or may not collapse


def test_plan_advance_walks_to_completion():
    """advance() marks current done and returns next; final advance is None."""
    plan = Plan.build("step one; step two; step three")
    assert len(plan.steps) == 3

    nxt = plan.advance()
    assert nxt is not None and nxt.idx == 2
    assert plan.steps[0].status == "done"

    nxt = plan.advance()
    assert nxt is not None and nxt.idx == 3

    nxt = plan.advance()
    assert nxt is None
    assert plan.is_complete()


def test_plan_render_marks_current_step():
    """The rendered ASCII view shows [x] / [>] / [ ] markers correctly."""
    plan = Plan.build("step A; step B; step C")
    plan.begin(plan.current())  # mark step A in-progress
    plan.advance()  # step A done
    plan.begin(plan.current())  # step B in-progress
    text = plan.render()
    assert "[x] 1. step A" in text
    assert "[>] 2. step B" in text
    assert "[ ] 3. step C" in text


def test_plan_block_marks_current_blocked():
    plan = Plan.build("step A; step B")
    plan.block("ran into something")
    assert plan.steps[0].status == "blocked"
    assert "ran into something" in plan.steps[0].note


def test_plan_current_prefers_in_progress():
    """If both in-progress and pending exist, current() returns in-progress."""
    plan = Plan.build("step A; step B")
    plan.steps[0].status = "in-progress"
    plan.steps[1].status = "pending"
    assert plan.current() is plan.steps[0]


def _two_file_workspace() -> DictWorkspace:
    return DictWorkspace(
        {
            "src/api.py": textwrap.dedent("""\
                class Client:
                    def get(self, path: str) -> str:
                        return ""
            """),
        }
    )


def test_agent_loop_multistep_uses_plan_to_advance():
    """A multi-step task advances the plan after each successful edit.

    Each cycle the proposer sees the CURRENT step description as the
    task. Two successful edits should complete the two-step plan with
    terminated_by='done', even without an explicit MarkDone from the
    proposer.
    """
    ws = _two_file_workspace()
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/api.py"), module="json", confidence=0.9)],
            [
                AddParameter(
                    function=SymbolRef(file="src/api.py", name="Client.get"),
                    name="timeout",
                    type=TypeExpr(expr="float"),
                    default=Expr(code="5.0"),
                    keyword_only=True,
                    confidence=0.9,
                )
            ],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=ws, use_plan=True)
    trace = loop.run(
        "Add an import of json to src/api.py; "
        "add a keyword-only parameter timeout of type float "
        "with default 5.0 to function get of class Client in src/api.py"
    )
    assert trace.ok
    assert trace.terminated_by == "done"
    final = trace.final_files["src/api.py"]
    assert "import json" in final
    assert "timeout: float = 5.0" in final


def test_agent_loop_single_step_plan_still_works():
    """A single-step task with use_plan=True behaves identically to old loop.

    Single-step plans don't change behavior — they still terminate on
    explicit MarkDone, not on plan-completion (a one-step plan IS
    completed by the single edit, but we want the test to confirm the
    MarkDone path also works to avoid surprising single-step users).
    """
    ws = _two_file_workspace()
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/api.py"), module="json", confidence=0.9)],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=ws, use_plan=True)
    trace = loop.run("add an import of json to src/api.py")
    assert trace.ok
    final = trace.final_files["src/api.py"]
    assert "import json" in final


def test_plan_fork_produces_independent_copy():
    """Plan.fork() returns a new Plan whose PlanStep mutations don't
    leak back to the original. Required for beam-search where each
    branch advances its own plan independently.
    """
    original = Plan.build("step A; step B; step C")
    forked = original.fork()

    # Mutate the fork.
    forked.advance()
    assert forked.steps[0].status == "done"
    # Original unchanged.
    assert original.steps[0].status == "pending"

    # Identity check — different list, different step objects.
    assert original.steps is not forked.steps
    assert original.steps[0] is not forked.steps[0]


def test_agent_loop_use_plan_false_keeps_legacy_behavior():
    """When use_plan=False the loop never builds a plan; obs.task is raw."""
    ws = _two_file_workspace()
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/api.py"), module="json", confidence=0.9)],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(proposer=proposer, workspace=ws, use_plan=False)
    trace = loop.run("step one; step two")
    assert trace.terminated_by == "done"
    # With use_plan=False, this is a single-step run terminated by MarkDone.
    assert len(trace.steps) == 2
