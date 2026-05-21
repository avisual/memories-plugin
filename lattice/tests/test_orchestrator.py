"""Tests for the orchestrator: intent expansion + plan execution."""

from __future__ import annotations

import textwrap

from lattice.actions import AddParameter
from lattice.compiler.workspace import DictWorkspace
from lattice.orchestrator import (
    AddParameterToAllMatching,
    execute_plan,
    expand_intent,
)


# ---------------------------------------------------------------------------
# Intent expansion
# ---------------------------------------------------------------------------


def _two_file_workspace() -> DictWorkspace:
    return DictWorkspace(
        {
            "a.py": textwrap.dedent("""\
                def charge(amount: int) -> None:
                    pass


                class Other:
                    def refund(self) -> None:
                        pass
            """),
            "b.py": textwrap.dedent("""\
                class Processor:
                    def charge(self, amount: int) -> None:
                        pass

                    def charge_async(self, amount: int) -> None:  # leaf-name differs
                        pass
            """),
        }
    )


def test_expand_finds_matches_across_files():
    ws = _two_file_workspace()
    intent = AddParameterToAllMatching(
        function_name="charge",
        parameter_name="dry_run",
        parameter_type="bool",
        parameter_default="False",
        keyword_only=True,
    )
    actions = expand_intent(intent, ws, ["a.py", "b.py"])

    assert len(actions) == 2
    targeted = {(a.function.file, a.function.name) for a in actions}
    assert targeted == {("a.py", "charge"), ("b.py", "Processor.charge")}
    for a in actions:
        assert isinstance(a, AddParameter)
        assert a.name == "dry_run"
        assert a.keyword_only is True


def test_expand_respects_include_flags():
    ws = _two_file_workspace()
    only_funcs = expand_intent(
        AddParameterToAllMatching(
            function_name="charge",
            parameter_name="dry_run",
            parameter_type="bool",
            include_functions=True,
            include_methods=False,
        ),
        ws,
        ["a.py", "b.py"],
    )
    assert [(a.function.file, a.function.name) for a in only_funcs] == [("a.py", "charge")]

    only_methods = expand_intent(
        AddParameterToAllMatching(
            function_name="charge",
            parameter_name="dry_run",
            parameter_type="bool",
            include_functions=False,
            include_methods=True,
        ),
        ws,
        ["a.py", "b.py"],
    )
    assert [(a.function.file, a.function.name) for a in only_methods] == [
        ("b.py", "Processor.charge")
    ]


def test_expand_empty_when_no_matches():
    ws = _two_file_workspace()
    actions = expand_intent(
        AddParameterToAllMatching(
            function_name="nope",
            parameter_name="x",
            parameter_type="int",
        ),
        ws,
        ["a.py", "b.py"],
    )
    assert actions == []


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------


def test_execute_plan_compiles_and_verifies():
    ws = _two_file_workspace()
    intent = AddParameterToAllMatching(
        function_name="charge",
        parameter_name="dry_run",
        parameter_type="bool",
        parameter_default="False",
        keyword_only=True,
    )
    actions = expand_intent(intent, ws, ["a.py", "b.py"])
    report = execute_plan(actions, ws)

    assert report.ok
    assert all(step.ok for step in report.steps)
    assert len(report.diffs) == 2
    for diff in report.diffs:
        assert "dry_run: bool" in diff
        assert "*, dry_run" in diff


def test_execute_plan_reports_compile_errors():
    # Build an intent whose match list will include a symbol that does not
    # actually exist when the compiler dispatches — simulated by stale
    # workspace edits would normally cause this; here we hand-craft the
    # situation via expand → and then mutate the workspace before execute.
    ws = _two_file_workspace()
    intent = AddParameterToAllMatching(
        function_name="charge",
        parameter_name="dry_run",
        parameter_type="bool",
    )
    actions = expand_intent(intent, ws, ["a.py", "b.py"])
    # Wipe b.py so the second action can't find its symbol anymore.
    ws._files["b.py"] = ""  # type: ignore[attr-defined]
    report = execute_plan(actions, ws)
    assert not report.ok
    failed = [s for s in report.steps if not s.ok]
    assert len(failed) >= 1
    assert any("SymbolNotFound" in (s.error or "") for s in failed)
