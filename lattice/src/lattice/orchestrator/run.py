"""Plan execution — drives Actions through compile and verify.

ExecutionReport bundles every action's CompiledAction and verify
SyntacticOutcome plus an overall success flag. No filesystem writes —
the report is the deliverable; applying it to disk is downstream.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from lattice.actions import Action
from lattice.compiler import (
    CompileError,
    CompiledAction,
    Workspace,
    compile_action,
)
from lattice.verify import SyntacticOutcome, verify_syntactic


class StepResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    action: Action
    compiled: CompiledAction | None = None
    verify: SyntacticOutcome | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.verify is not None and self.verify.ok


class ExecutionReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    steps: tuple[StepResult, ...]

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def diffs(self) -> tuple[str, ...]:
        out: list[str] = []
        for step in self.steps:
            if step.compiled is None:
                continue
            for change in step.compiled.file_changes:
                if change.diff:
                    out.append(change.diff)
        return tuple(out)


def execute_plan(actions: list[Action], workspace: Workspace) -> ExecutionReport:
    """Compile + verify each action against the in-memory workspace.

    Subsequent actions see the *original* workspace (no chained
    application yet) — that's a deliberate v0 limitation. Once the
    world model lands, chained simulation replaces this and the
    workspace can be updated between steps in imagination.
    """
    steps: list[StepResult] = []
    for action in actions:
        try:
            compiled = compile_action(action, workspace)
        except CompileError as exc:
            steps.append(StepResult(action=action, error=f"{type(exc).__name__}: {exc}"))
            continue
        outcome = verify_syntactic(compiled)
        steps.append(StepResult(action=action, compiled=compiled, verify=outcome))
    return ExecutionReport(steps=tuple(steps))
