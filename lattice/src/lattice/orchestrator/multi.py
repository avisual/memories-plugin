"""Multi-subtask runner — planner-executor split.

Drives the agent loop through a sequence of subtasks, sharing one
overlay so later subtasks see earlier ones' edits. Each subtask is a
self-contained call to the agent loop with its own step budget. The
final report is a single ExecutionReport-shape carrying the
consolidated diffs and final-file map across the whole plan.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from lattice.compiler import OverlayWorkspace, Workspace
from lattice.compiler.diff import unified_diff
from lattice.orchestrator.agent import AgentLoop, AgentTrace
from lattice.propose import Proposer


class MultiSubtaskReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    subtasks: tuple[str, ...]
    traces: tuple[AgentTrace, ...]
    final_files: dict[str, str]
    consolidated_diffs: tuple[str, ...]
    terminated_by: str  # "done" | "blocked" | "partial"

    @property
    def ok(self) -> bool:
        return self.terminated_by == "done"


_ADVANCE_REASONS = frozenset({"done", "stuck"})
"""Subtask outcomes that mean 'goal observably achieved, advance to next'.

'done' is explicit (MarkDone). 'stuck' means the next action would be a
no-op against the overlay — which IS the observable signal that the
goal is met. This is the stack-machine model: the LLM emits one
instruction per cycle; when the next instruction would change nothing,
the program counter advances.
"""


def run_subtasks(
    subtasks: list[str],
    *,
    proposer: Proposer,
    workspace: Workspace,
    atom_store: Any = None,
    max_steps_per_subtask: int = 3,
    stop_on_blocked: bool = True,
) -> MultiSubtaskReport:
    """Run the agent loop once per subtask against a shared overlay.

    Each subtask gets its own AgentLoop with the SAME overlay, so edits
    from subtask 1 are visible when subtask 2 starts.

    Advances to the next subtask when the current one terminates with
    "done" (explicit MarkDone) OR "stuck" (next action would be a
    no-op — the goal is observably achieved). The LLM does not need
    to plan multi-step or emit MarkDone; the harness uses observable
    state change as the program-counter advance signal.

    Stops the whole plan on "blocked" if stop_on_blocked is true.
    """
    shared_overlay = OverlayWorkspace(workspace)

    traces: list[AgentTrace] = []
    terminated_by = "done"

    for subtask in subtasks:
        loop = AgentLoop(
            proposer=proposer,
            workspace=shared_overlay,
            atom_store=atom_store,
            max_steps=max_steps_per_subtask,
        )
        trace = loop.run(subtask)
        traces.append(trace)
        for path in loop.overlay.overlay_paths():
            shared_overlay.update(path, loop.overlay.read(path))

        if trace.terminated_by == "blocked" and stop_on_blocked:
            terminated_by = "blocked"
            break
        if trace.terminated_by not in _ADVANCE_REASONS:
            terminated_by = "partial"

    final_files: dict[str, str] = {}
    diffs: list[str] = []
    for path in shared_overlay.overlay_paths():
        before = shared_overlay.base_content(path)
        after = shared_overlay.read(path)
        if before == after:
            continue
        final_files[path] = after
        diffs.append(unified_diff(path=path, before=before, after=after))

    return MultiSubtaskReport(
        subtasks=tuple(subtasks),
        traces=tuple(traces),
        final_files=final_files,
        consolidated_diffs=tuple(diffs),
        terminated_by=terminated_by,
    )
