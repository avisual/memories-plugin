"""Plan execution — drives Actions through compile and verify, chaining state.

Each action sees the cumulative output of prior successful actions via
an OverlayWorkspace. The original workspace is never mutated — the
overlay holds in-memory edits, and the ExecutionReport exposes both
per-step diffs (each action's local change) and a `final_files` mapping
of path → fully-edited content for downstream apply-to-disk.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from lattice.actions import Action
from lattice.compiler import (
    CompileError,
    CompiledAction,
    OverlayWorkspace,
    Workspace,
    compile_action,
)
from lattice.compiler.diff import unified_diff
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
    final_files: dict[str, str] = {}
    consolidated_diffs: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def diffs(self) -> tuple[str, ...]:
        """Per-step diffs — each action's local change."""
        out: list[str] = []
        for step in self.steps:
            if step.compiled is None:
                continue
            for change in step.compiled.file_changes:
                if change.diff:
                    out.append(change.diff)
        return tuple(out)


def execute_plan(actions: list[Action], workspace: Workspace) -> ExecutionReport:
    """Compile + verify each action against a chained overlay of *workspace*.

    A successful, non-noop action's after-content is staged into the
    overlay so subsequent actions see it. Failed actions do not update
    the overlay; later actions still see the last known good state.
    """
    overlay = OverlayWorkspace(workspace)
    steps: list[StepResult] = []
    for action in actions:
        try:
            compiled = compile_action(action, overlay)
        except CompileError as exc:
            steps.append(StepResult(action=action, error=f"{type(exc).__name__}: {exc}"))
            continue
        outcome = verify_syntactic(compiled)
        if outcome.ok:
            for change in compiled.file_changes:
                if not change.is_noop:
                    overlay.update(change.path, change.after)
        steps.append(StepResult(action=action, compiled=compiled, verify=outcome))

    final_files: dict[str, str] = {}
    consolidated: list[str] = []
    for path in overlay.overlay_paths():
        before = overlay.base_content(path)
        after = overlay.read(path)
        if before == after:
            continue
        final_files[path] = after
        consolidated.append(unified_diff(path=path, before=before, after=after))

    return ExecutionReport(
        steps=tuple(steps),
        final_files=final_files,
        consolidated_diffs=tuple(consolidated),
    )
