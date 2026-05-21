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
    Research,
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
from lattice.sense.semble_search import SembleCodeSearch
from lattice.verify import (
    SyntacticOutcome,
    verify_syntactic,
    verify_tests,
    verify_types,
)


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
        run_tests: bool = False,
        workspace_root: str | None = None,
        code_search: SembleCodeSearch | None = None,
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
        self._run_tests = run_tests
        self._workspace_root = workspace_root
        self._code_search = code_search

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
        """POPULATION (Organ 6, v0): score all N candidates, pick the best.

        Asks the proposer for N candidates, simulates each via the real
        compiler against the current overlay (the v0 'world model'),
        scores them by a composite (non-no-op + lines-changed + confidence),
        returns the highest-scoring action.

        Earlier this method picked the FIRST non-no-op candidate; the
        beam-style version below uses all signals so a low-confidence
        large-diff candidate can be passed over for a high-confidence
        precise one. Non-mutating verbs (MarkDone/Branch/Research)
        bypass scoring — they're not edits and the world model has
        nothing to simulate; they're returned with their confidence.

        Falls back to the first candidate if scoring rejects everything
        (so the loop's stuck detector still has signal to break).
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

        scored: list[tuple[float, Action, str]] = []
        for action in actions:
            if not self._is_mutating(action):
                # Non-mutating: score by confidence only.
                conf = float(getattr(action, "confidence", 0.5))
                scored.append((conf, action, "non-mutating"))
                continue
            try:
                compiled = compile_action(action, self.overlay)
            except (CompileError, NonMutatingAction) as exc:
                # Compile errors get a tiny positive score so they can
                # still be picked if every candidate failed — the agent
                # loop will surface them as errors and learn from them.
                scored.append((0.01, action, f"compile-error: {exc}"))
                continue
            if compiled.is_noop:
                scored.append((0.05, action, "no-op"))
                continue
            # Composite: confidence + lines-changed bonus (capped).
            conf = float(getattr(action, "confidence", 0.5))
            lines_changed = sum(
                _count_diff_added_lines(c.diff)
                for c in compiled.file_changes
                if not c.is_noop
            )
            change_bonus = min(0.3, 0.02 * lines_changed)
            score = conf + change_bonus + 1.0  # +1.0 ensures real edits beat no-ops
            scored.append((score, action, f"score=conf{conf:.2f}+lines{lines_changed}"))

        if os.environ.get("LATTICE_LLM_DEBUG"):
            for s, a, note in scored:
                sys.stderr.write(f"  candidate {a.verb} score={s:.2f} ({note})\n")

        # Highest score wins; stable order on ties.
        scored.sort(key=lambda t: -t[0])
        return scored[0][1] if scored else actions[0]

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
        if isinstance(action, Research):
            payload = self._research(action.url, action.reason)
            return (
                StepRecord(step=step_idx, action=action, kind="research", payload=payload),
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
        if self._run_tests and self._workspace_root is not None:
            test_outcome = verify_tests(
                compiled, workspace_root=self._workspace_root
            )
            if not test_outcome.ok and not test_outcome.skipped:
                return (
                    StepRecord(
                        step=step_idx,
                        action=action,
                        kind="error",
                        verify=outcome,
                        error="tests failed: "
                        + "; ".join(f"{t}: {m}" for t, m in test_outcome.errors[:3]),
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
        """Detect the harness-level no-progress cycle.

        Two patterns trip this:
        1. The most recent N edit-or-error steps all repeat the same
           verb AND were no-ops or errors. (Original detector.)
        2. The most recent N steps all repeat the same verb,
           regardless of kind. Covers the case where the LLM keeps
           emitting Research (or any non-mutating verb) without
           ever progressing to an edit. N defaults to 3 to give
           legitimate research+act sequences room.
        """
        limit = self._noop_streak_limit

        recent_edits = [s for s in steps if s.kind in {"edit", "error"}]
        if len(recent_edits) >= limit:
            tail = recent_edits[-limit:]
            verb = tail[0].verb
            if all(s.verb == verb for s in tail) and all(
                s.kind == "error"
                or (s.diff == "" or "no-op" in _summarize_diff(s.diff))
                for s in tail
            ):
                return True

        repeat_limit = max(3, limit + 1)
        if len(steps) >= repeat_limit:
            tail = steps[-repeat_limit:]
            verb = tail[0].verb
            if verb and all(s.verb == verb for s in tail):
                return True

        return False

    def _build_observation(
        self, task: str, steps: list[StepRecord]
    ) -> ObservationContext:
        symbols = self._gather_symbols()
        hints: list[str] = []

        # STEER (Organ 4, lightweight v0): high-importance atoms — those
        # EVOLVE has boosted because the system has lived through their
        # pattern — go FIRST in the hint list so they sit in the LLM's
        # most-attended prompt position. The heavyweight path
        # (activation patching) is documented in steer/__init__.py.
        if self.atom_store is not None:
            try:
                from lattice.steer import render_priming_block, top_priming_atoms

                primed = top_priming_atoms(self.atom_store, task, k=2)
                priming = render_priming_block(primed)
                if priming:
                    hints.append(priming)
            except Exception:  # noqa: BLE001
                pass

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

        # Recall atoms EVERY cycle, not just the first. New atoms
        # written by mid-loop Research are exactly the ones that need
        # to surface on the very next turn.
        if self.atom_store is not None:
            try:
                hits = self.atom_store.recall(task, k=4)
            except Exception:
                hits = []
            for r in hits:
                hints.append(
                    f"{r.atom.type.value} (score {r.score:.2f}): {r.atom.content[:600]}"
                )

        # Semble code-search: per-task, focused chunks rather than the
        # whole symbol list. Same per-cycle cadence as atom recall.
        if self._code_search is not None:
            try:
                chunks = self._code_search.search(task, top_k=3)
            except Exception:
                chunks = []
            for c in chunks:
                hints.append(
                    f"CODE {c.file}:{c.start_line}-{c.end_line}\n{c.content[:300]}"
                )

        for step in steps[-6:]:
            line = f"step {step.step} [{step.kind}] {step.verb}"
            if step.kind == "edit":
                summary = _summarize_diff(step.diff)
                line += f" -> {summary}"
            elif step.kind == "error":
                line += f" -> error: {step.error[:140]}"
            elif step.kind == "recall" and step.payload:
                line += f" -> recalled: {step.payload[:200]}"
            elif step.kind == "research" and step.payload:
                line += f" -> {step.payload[:600]}"
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

    def _research(self, url: str, reason: str) -> str:
        """Fetch *url* via curl-cffi, store the cleaned text as a fact atom,
        and return a short payload string that next-turn hint rendering
        will surface to the LLM. The atom carries the URL as a tag so
        the brain can re-fetch / dedup on subsequent runs.
        """
        try:
            from lattice.atoms.web import WebFetchError, fetch_url
        except ImportError as exc:
            return f"(web fetch unavailable: {exc})"
        try:
            page = fetch_url(url)
        except WebFetchError as exc:
            return f"(fetch failed: {exc})"

        snippet = page.text[:2000]
        content = (
            f"Fetched from {page.final_url} (status {page.status})"
            + (f" — title: {page.title}" if page.title else "")
            + f"\n\n{snippet}"
        )

        if self.atom_store is not None:
            try:
                from lattice.atoms import AtomType

                self.atom_store.add(
                    content,
                    type=AtomType.FACT,
                    region="research",
                    tags=("web", url),
                    importance=0.75,
                )
            except Exception:  # noqa: BLE001
                pass

        return (
            f"researched {page.final_url} ({reason}); "
            f"top excerpt: {snippet[:240].replace(chr(10), ' ')}"
        )


def _count_diff_added_lines(diff: str) -> int:
    """Count '+' (added) lines in a unified diff, skipping the header."""
    if not diff:
        return 0
    n = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            n += 1
    return n


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
