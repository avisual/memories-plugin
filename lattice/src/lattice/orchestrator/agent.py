"""Agent loop — multi-step reasoning driven by a Proposer.

One action per turn. The result of each turn (typed outcome record)
becomes a hint in the next turn's Observation, so the proposer can
react to what happened. Terminates on:

- MarkDone (success): proposer is satisfied.
- MarkBlocked: proposer can't make progress; reported as such.
- max_steps reached: hard cap.
- proposer returns no action: empty propose() result.

Non-mutating verbs (RecallMore, RevealBody, Branch) shape the loop
without producing diffs:
- RecallMore: the loop hits the atom store with the query and threads
  the result into next turn's hints.
- RevealBody: the loop fetches the requested symbol's source and
  threads it as a hint.
- Branch: no-op in the linear v0 loop (logged for future population
  search; doesn't terminate).

Verify failures roll back that turn's overlay edit and surface as a
hint to the next turn.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from lattice.actions import (
    Action,
    Branch,
    MarkBlocked,
    MarkDone,
    RecallMore,
    RevealBody,
)
from lattice.compiler import (
    CompileError,
    OverlayWorkspace,
    Workspace,
    compile_action,
)
from lattice.compiler.diff import unified_diff
from lattice.compiler.errors import NonMutatingAction
from lattice.propose import ObservationContext, Proposer
from lattice.sense import Symbol, walk_workspace
from lattice.verify import SyntacticOutcome, verify_syntactic, verify_types


class StepRecord(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    step: int
    action: Action | None = None  # None when the proposer itself raised before emitting
    kind: str  # "edit" | "blocked" | "done" | "recall" | "reveal" | "branch" | "error"
    verb: str = ""
    verify: SyntacticOutcome | None = None
    diff: str = ""
    error: str = ""
    payload: str = ""

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        if not self.verb and self.action is not None:
            object.__setattr__(self, "verb", self.action.verb)


class AgentTrace(BaseModel):
    model_config = ConfigDict(frozen=True)
    task: str
    steps: tuple[StepRecord, ...]
    final_files: dict[str, str]
    consolidated_diffs: tuple[str, ...]
    # "done" | "blocked" | "exhausted" | "empty" | "stuck"
    terminated_by: str

    @property
    def ok(self) -> bool:
        return self.terminated_by == "done"


class AgentLoop:
    """Drives a Proposer through a multi-step task.

    Reads atom-store hints (if provided) and surfaces both prior atoms
    and the running action history as hints into the Observation, so
    the LLM is grounded in what it has tried and what was found.
    """

    def __init__(
        self,
        *,
        proposer: Proposer,
        workspace: Workspace,
        atom_store: Any = None,  # AtomStore Protocol; typed dynamically to avoid import
        max_steps: int = 8,
        files: list[str] | None = None,
        noop_streak_limit: int = 2,
        preflight_candidates: int = 3,
        type_check: bool = False,
    ) -> None:
        self.proposer = proposer
        self.workspace = workspace
        self.overlay = OverlayWorkspace(workspace)
        self.atom_store = atom_store
        self.max_steps = max_steps
        self._files = files
        self._noop_streak_limit = noop_streak_limit
        self._preflight_candidates = preflight_candidates
        self._type_check = type_check

    def run(self, task: str) -> AgentTrace:
        steps: list[StepRecord] = []
        terminated_by = "exhausted"

        for step_idx in range(1, self.max_steps + 1):
            obs = self._build_observation(task, steps)
            try:
                action = self._propose_with_preflight(obs)
            except Exception as exc:  # noqa: BLE001
                steps.append(
                    StepRecord(
                        step=step_idx,
                        action=None,
                        kind="error",
                        verb="proposer",
                        error=f"proposer raised: {exc}",
                    )
                )
                terminated_by = "blocked"
                break
            if action is None:
                terminated_by = "empty"
                break
            record, should_stop, stop_reason = self._handle_action(action, step_idx)
            steps.append(record)
            if should_stop:
                terminated_by = stop_reason
                break
            if self._is_stuck(steps):
                terminated_by = "stuck"
                break

        final_files: dict[str, str] = {}
        diffs: list[str] = []
        for path in self.overlay.overlay_paths():
            before = self.overlay.base_content(path)
            after = self.overlay.read(path)
            if before == after:
                continue
            final_files[path] = after
            diffs.append(unified_diff(path=path, before=before, after=after))

        return AgentTrace(
            task=task,
            steps=tuple(steps),
            final_files=final_files,
            consolidated_diffs=tuple(diffs),
            terminated_by=terminated_by,
        )

    # ----- pre-flight: simulate before commit -----

    def _propose_with_preflight(self, obs: ObservationContext) -> Action | None:
        """Ask the proposer for N candidates; pick the first non-no-op edit.

        Pre-flight uses the real compiler against the current overlay so
        the predicted outcome matches what would actually happen. This
        is the v0 'world model': for mutating verbs, simulate cheaply
        before commit. Non-mutating verbs (MarkDone/MarkBlocked/recall)
        pass through unchanged — there's nothing to simulate.

        Falls back to the first proposed action if every candidate is a
        no-op (so the loop's stuck detector still has signal to break).
        """
        import os
        import sys

        actions = self.proposer.propose(obs, n=self._preflight_candidates)
        if os.environ.get("LATTICE_LLM_DEBUG"):
            sys.stderr.write(
                f"preflight got {len(actions)} candidate(s): "
                + ", ".join(a.verb for a in actions)
                + "\n"
            )
        if not actions:
            return None

        for action in actions:
            if not self._is_mutating(action):
                return action
            try:
                compiled = compile_action(action, self.overlay)
            except (CompileError, NonMutatingAction):
                return action  # let _handle_action surface the error
            if not compiled.is_noop:
                return action

        return actions[0]

    @staticmethod
    def _is_mutating(action: Action) -> bool:
        from lattice.actions import (
            AddField,
            AddImport,
            AddParameter,
            AddTest,
            RenameSymbol,
            WrapInTry,
        )

        return isinstance(
            action,
            (AddImport, AddField, AddParameter, AddTest, WrapInTry, RenameSymbol),
        )

    # ----- per-action dispatch -----

    def _handle_action(
        self, action: Action, step_idx: int
    ) -> tuple[StepRecord, bool, str]:
        verb = action.verb

        if isinstance(action, MarkDone):
            return (
                StepRecord(step=step_idx, action=action, kind="done", payload=action.summary),
                True,
                "done",
            )
        if isinstance(action, MarkBlocked):
            return (
                StepRecord(
                    step=step_idx,
                    action=action,
                    kind="blocked",
                    payload=f"{action.reason_code}: {action.detail}",
                ),
                True,
                "blocked",
            )
        if isinstance(action, Branch):
            return (
                StepRecord(step=step_idx, action=action, kind="branch", payload=action.rationale),
                False,
                "",
            )
        if isinstance(action, RecallMore):
            payload = self._recall_more(action.query)
            return (
                StepRecord(step=step_idx, action=action, kind="recall", payload=payload),
                False,
                "",
            )
        if isinstance(action, RevealBody):
            payload = self._reveal_body(action.symbol.file, action.symbol.name)
            return (
                StepRecord(step=step_idx, action=action, kind="reveal", payload=payload),
                False,
                "",
            )

        # Mutating path.
        try:
            compiled = compile_action(action, self.overlay)
        except (CompileError, NonMutatingAction) as exc:
            return (
                StepRecord(
                    step=step_idx,
                    action=action,
                    kind="error",
                    error=f"{type(exc).__name__}: {exc}",
                ),
                False,
                "",
            )
        outcome = verify_syntactic(compiled)
        if not outcome.ok:
            return (
                StepRecord(
                    step=step_idx,
                    action=action,
                    kind="error",
                    verify=outcome,
                    error="; ".join(f"{p}: {m}" for p, m in outcome.errors),
                ),
                False,
                "",
            )
        if self._type_check:
            type_outcome = verify_types(compiled)
            if not type_outcome.ok:
                return (
                    StepRecord(
                        step=step_idx,
                        action=action,
                        kind="error",
                        verify=outcome,
                        error="type-check failed: "
                        + "; ".join(f"{p}: {m}" for p, m in type_outcome.errors[:3]),
                    ),
                    False,
                    "",
                )
        for change in compiled.file_changes:
            if not change.is_noop:
                self.overlay.update(change.path, change.after)
        joined_diff = "\n".join(c.diff for c in compiled.file_changes if c.diff)
        return (
            StepRecord(
                step=step_idx,
                action=action,
                kind="edit",
                verify=outcome,
                diff=joined_diff,
            ),
            False,
            "",
        )
        return (
            StepRecord(
                step=step_idx,
                action=action,
                kind="error",
                verify=outcome,
                error="; ".join(f"{p}: {m}" for p, m in outcome.errors),
            ),
            False,
            "",
        )

    # ----- helpers -----

    def _is_stuck(self, steps: list[StepRecord]) -> bool:
        """Detect the harness-level no-op cycle (proposer repeating itself).

        Triggers when the most recent N edit-or-error steps all repeat the
        same action verb AND were no-ops or errors. This is the
        scaffolding around small-model planning weakness: even if the LLM
        can't track plan state, the harness will not loop forever.
        """
        limit = self._noop_streak_limit
        recent = [s for s in steps if s.kind in {"edit", "error"}]
        if len(recent) < limit:
            return False
        tail = recent[-limit:]
        verb = tail[0].verb
        if not all(s.verb == verb for s in tail):
            return False
        return all(
            s.kind == "error"
            or (s.diff == "" or "no-op" in _summarize_diff(s.diff))
            for s in tail
        )

    def _build_observation(
        self, task: str, steps: list[StepRecord]
    ) -> ObservationContext:
        symbols = self._gather_symbols()
        hints: list[str] = []

        # Pin the workspace file list explicitly so the model uses real
        # paths, not paths it might hallucinate from prompt examples.
        ws_files = self._files
        if ws_files is None:
            ws_files = self.overlay.iter_files(suffix=".py")
        if ws_files:
            hints.append(
                "WORKSPACE FILES (use these EXACT paths, do not invent paths): "
                + ", ".join(ws_files[:20])
            )

        if self.atom_store is not None and not steps:
            try:
                hits = self.atom_store.recall(task, k=4)
            except Exception:
                hits = []
            for r in hits:
                hints.append(f"{r.atom.type.value} (score {r.score:.2f}): {r.atom.content}")

        for step in steps[-6:]:
            line = f"step {step.step} [{step.kind}] {step.verb}"
            if step.kind == "edit":
                summary = _summarize_diff(step.diff)
                line += f" -> {summary}"
            elif step.kind == "error":
                line += f" -> error: {step.error[:140]}"
            elif step.kind == "recall" and step.payload:
                line += f" -> recalled: {step.payload[:200]}"
            elif step.kind == "reveal" and step.payload:
                line += f" -> revealed: {step.payload[:200]}"
            elif step.kind == "branch":
                line += f" -> branch noted: {step.payload[:120]}"
            hints.append(line)

        if self.overlay.overlay_paths():
            edited = ", ".join(self.overlay.overlay_paths())
            hints.append(f"FILES ALREADY EDITED THIS TASK: {edited}")
        hints.append(
            "Look at the history. If the task is complete, emit MarkDone. "
            "Do not repeat an edit that has already been applied."
        )

        return ObservationContext(task=task, symbols=tuple(symbols[:30]), hints=tuple(hints))

    def _gather_symbols(self) -> list[Symbol]:
        files = self._files if self._files is not None else self.overlay.iter_files(suffix=".py")
        return walk_workspace(self.overlay, files)

    def _recall_more(self, query: str) -> str:
        if self.atom_store is None:
            return "(no atom store configured)"
        try:
            hits = self.atom_store.recall(query, k=3)
        except Exception as exc:  # noqa: BLE001
            return f"recall failed: {exc}"
        if not hits:
            return "(no atoms matched)"
        return "; ".join(f"{r.atom.type.value}: {r.atom.content[:140]}" for r in hits)

    def _reveal_body(self, file: str, dotted_name: str) -> str:
        if not self.overlay.exists(file):
            return f"(file not found: {file})"
        try:
            src = self.overlay.read(file)
        except Exception as exc:  # noqa: BLE001
            return f"(read failed: {exc})"
        lines = src.splitlines()
        leaf = dotted_name.rsplit(".", 1)[-1]
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "async def ")) and leaf in stripped:
                snippet = "\n".join(lines[i : i + 12])
                return f"{file}:{i + 1}\n{snippet}"
        return f"(symbol {dotted_name!r} not found in {file})"


def _summarize_diff(diff: str) -> str:
    """One-line summary of what an edit changed."""
    if not diff.strip():
        return "no-op (already in target state)"
    adds: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            content = line[1:].strip()
            if content:
                adds.append(content)
        if len(adds) >= 3:
            break
    if not adds:
        return "no visible additions"
    return "added " + " | ".join(adds[:3])
