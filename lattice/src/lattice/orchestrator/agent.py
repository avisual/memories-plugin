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
from lattice.orchestrator.plan import Plan, PlanStep
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
        use_brain: bool = True,
        use_plan: bool = True,
    ) -> None:
        self.proposer = proposer
        self.workspace = workspace
        self.overlay = OverlayWorkspace(workspace)
        # When use_brain is False, the loop pretends there is no atom
        # store from the OBSERVATION side: no priming, no recall hints,
        # no apprentice influence. Used by the brain-effect audit to
        # measure 'does the substrate actually pay for itself'.
        self.atom_store = atom_store
        self.max_steps = max_steps
        self._files = files
        self._noop_streak_limit = noop_streak_limit
        self._preflight_candidates = preflight_candidates
        self._type_check = type_check
        self._run_tests = run_tests
        self._workspace_root = workspace_root
        self._code_search = code_search
        self._use_brain = use_brain
        self._use_plan = use_plan
        # The plan is built at the start of each run() so a single
        # AgentLoop instance can run many tasks back-to-back.
        self._plan: Plan | None = None
        # Atom IDs recalled into the CURRENT cycle's observation.
        # Captured here so the post-step Hebbian update knows which
        # atoms 'participated' (and should be reinforced/decayed by
        # the outcome of the step). Reset each cycle in
        # _build_observation.
        self._cycle_recalled_ids: list[int] = []
        # Atom IDs that brain_score consulted for the WINNING candidate.
        # These also participated in the decision and should receive
        # the same Hebbian update as observation-recall atoms. Set in
        # _propose_with_preflight after the winner is picked.
        self._winning_brain_ids: list[int] = []
        # The task string the proposer is actively attacking THIS cycle.
        # In single-step runs it == run()'s task argument; in multi-step
        # plan-DAG runs it's the current step's description. Set in
        # _build_observation; used by _record_step_outcome and
        # _brain_score so the SAME query embeds against the SAME atoms.
        self._active_task: str = ""

    def run(self, task: str) -> AgentTrace:
        steps: list[StepRecord] = []
        terminated_by = "exhausted"

        # PLAN-DAG (Organ 6 v0 / Phase 2): build the plan once per task.
        # Even single-step tasks get a one-element plan so the rest of
        # the loop can use a uniform interface. Multi-step tasks gain
        # an explicit current-step pointer the proposer focuses on.
        self._plan = Plan.build(task) if self._use_plan else None
        if self._plan is not None and self._plan.is_multistep():
            self._persist_plan_atom(self._plan)
            cur = self._plan.current()
            if cur is not None:
                self._plan.begin(cur)

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
            self._record_step_outcome(record)
            # HEBBIAN REINFORCEMENT: atoms that participated in THIS
            # cycle's decision (i.e. were recalled into the observation)
            # get nudged up on success or down on failure. The brain
            # actually learns from outcome here — atoms that helped get
            # surfaced more readily next time; atoms that misled get
            # buried. This is the missing teeth that turns recall from
            # decoration into a learning loop.
            self._reinforce_recalled(record)

            # PLAN-DAG advance: when the just-recorded step is a clean
            # mutating edit (no verify error), the CURRENT plan step is
            # considered done and we move the pointer. MarkDone from the
            # LLM is treated as "this step is done" on multi-step plans;
            # only the FINAL step's MarkDone completes the whole plan.
            if self._plan is not None and self._plan.is_multistep():
                if record.kind == "edit" and not record.error:
                    next_step = self._plan.advance()
                    if next_step is None:
                        terminated_by = "done"
                        break
                    self._plan.begin(next_step)
                elif record.kind == "done":
                    # LLM thinks we're done — if more plan steps remain,
                    # advance instead of terminating; only the last
                    # step's MarkDone breaks out.
                    next_step = self._plan.advance()
                    if next_step is None:
                        terminated_by = "done"
                        break
                    self._plan.begin(next_step)
                    # Don't fall through to should_stop for this kind.
                    if self._is_stuck(steps):
                        terminated_by = "stuck"
                        break
                    continue
                elif record.kind == "blocked":
                    self._plan.block(record.error or "blocked")

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

        # Track brain contributors per candidate so we can attribute
        # the winner's Hebbian update correctly. Without this, atoms
        # that scored the WINNING candidate get no reinforcement after
        # a successful step — only observation-recall atoms do. That's
        # a correctness gap: those atoms ARE participating in the
        # decision and should be rewarded by outcome.
        scored: list[tuple[float, Action, str, list[int]]] = []
        for action in actions:
            if not self._is_mutating(action):
                # Non-mutating: score by confidence only.
                conf = float(getattr(action, "confidence", 0.5))
                scored.append((conf, action, "non-mutating", []))
                continue
            try:
                compiled = compile_action(action, self.overlay)
            except (CompileError, NonMutatingAction) as exc:
                # Compile errors get a tiny positive score so they can
                # still be picked if every candidate failed — the agent
                # loop will surface them as errors and learn from them.
                scored.append((0.01, action, f"compile-error: {exc}", []))
                continue
            if compiled.is_noop:
                scored.append((0.05, action, "no-op", []))
                continue
            # Composite: confidence + lines-changed bonus (capped).
            conf = float(getattr(action, "confidence", 0.5))
            lines_changed = sum(
                _count_diff_added_lines(c.diff)
                for c in compiled.file_changes
                if not c.is_noop
            )
            change_bonus = min(0.3, 0.02 * lines_changed)
            # HEBBIAN DECISION-WEIGHTING (Organ 4 / Phase 4).
            # The brain doesn't just decorate the prompt — it shapes
            # which candidate wins. Recall outcomes from prior steps
            # that match this (verb, slots, task) signature; SKILL /
            # EXPERIENCE atoms boost, ANTIPATTERN atoms penalize.
            # Bounded so a single hot atom can't drown out confidence,
            # but a clear repeat-failure pattern CAN flip the winner.
            # Uses self._active_task (set in _build_observation) — same
            # query used here lands near atoms _record_step_outcome
            # wrote with the same signature.
            brain_delta, brain_ids = self._brain_score(action, self._active_task)
            score = conf + change_bonus + 1.0 + brain_delta
            scored.append((
                score, action,
                f"score=conf{conf:.2f}+lines{lines_changed}+brain{brain_delta:+.2f}",
                brain_ids,
            ))

        if os.environ.get("LATTICE_LLM_DEBUG"):
            for s, a, note, _ids in scored:
                sys.stderr.write(f"  candidate {a.verb} score={s:.2f} ({note})\n")

        # Highest score wins; stable order on ties.
        scored.sort(key=lambda t: -t[0])
        if not scored:
            self._winning_brain_ids = []
            return actions[0]
        winner = scored[0]
        # Stash the brain-contributors for the winner so the post-step
        # reinforcement Hebbian-updates them too (not just observation-
        # recall atoms). Atoms that earned the +/- delta are exactly the
        # ones whose importance should move with the outcome.
        self._winning_brain_ids = winner[3]
        return winner[1]

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

    # ----- brain (Hebbian) decision-weighting -----

    @staticmethod
    def _action_signature(action: Action) -> str:
        """Compact (verb, slots) signature.

        Used both as the recall query when scoring a candidate AND as
        the content body of outcome atoms — so an atom written by one
        run embeds near the query a future scorer will issue. Both
        share the same template, so cosine similarity actually does
        what we want it to.
        """
        verb = getattr(action, "verb", "?")
        try:
            dump = action.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            return f"verb={verb}"
        parts = [f"verb={verb}"]
        for k, v in dump.items():
            if k in {"verb", "confidence"}:
                continue
            parts.append(f"{k}={_compact_value(v)}")
        return " ".join(parts)

    def _brain_score(self, action: Action, task: str) -> tuple[float, list[int]]:
        """Brain influence on candidate selection.

        Recalls atoms matching this candidate's signature. SKILL /
        EXPERIENCE atoms (prior successes on similar attempts) boost;
        ANTIPATTERN atoms (prior failures) penalize. Weighting:

          + 0.25 * similarity  per matching success atom
          - 0.40 * similarity  per matching failure atom
          (failures are weighted harder — a repeat-mistake signal
          should be louder than a stale success.)

        Bounded to [-0.5, +0.5] so the brain can break ties and
        overrule small confidence gaps but cannot ride roughshod
        over a clear high-confidence candidate. Similarity threshold
        0.35 filters out near-irrelevant hits.

        Region filter: only consults atoms written by _record_step_outcome
        (region='steps'). Generic seeded atoms in other regions don't
        have the (verb, slots, task) signature shape that the query
        expects, and admitting them produces spurious low-cosine boosts.
        If the user wants seeded knowledge to score candidates they'd
        need to seed under the 'steps' region explicitly.

        Returns (delta, atom_ids_that_contributed) so the caller can
        track which atoms participated in scoring the WINNING candidate
        (for Hebbian reinforcement after verify).
        """
        if not self._use_brain or self.atom_store is None:
            return 0.0, []
        from lattice.atoms import AtomType

        query = _signature_with_task(self._action_signature(action), task)
        try:
            hits = self.atom_store.recall(query, k=4, region="steps")
        except Exception:  # noqa: BLE001
            return 0.0, []
        delta = 0.0
        contributors: list[int] = []
        for hit in hits:
            sim = float(hit.score)
            if sim < 0.35:
                continue
            atype = hit.atom.type
            if atype in (AtomType.SKILL, AtomType.EXPERIENCE):
                delta += 0.25 * sim
                atom_id = getattr(hit.atom, "id", None)
                if atom_id is not None:
                    contributors.append(int(atom_id))
            elif atype == AtomType.ANTIPATTERN:
                delta -= 0.40 * sim
                atom_id = getattr(hit.atom, "id", None)
                if atom_id is not None:
                    contributors.append(int(atom_id))
        return max(-0.5, min(0.5, delta)), contributors

    def _reinforce_recalled(self, record: "StepRecord") -> None:
        """Hebbian update on the atoms recalled into THIS cycle's obs.

        On a clean mutating edit (kind=='edit' and no error) → positive
        nudge: atoms that participated in a winning decision get more
        important and will surface higher in future recall.

        On a verify error → small negative nudge: atoms that showed up
        for a failing decision get pushed down a bit so they don't keep
        misleading the next attempt. Decay is gentler than reinforcement
        because a failed step can still have useful recalled atoms (the
        cause may be elsewhere) — overcorrecting buries them too fast.

        No-ops, non-mutating steps, and the case where brain is off all
        skip the update.
        """
        if not self._use_brain or self.atom_store is None:
            return
        if record.action is None:
            return
        if not self._is_mutating(record.action):
            return
        # Decide direction & magnitude.
        if record.kind == "edit" and not record.error:
            delta = +0.05
        elif record.kind == "error":
            delta = -0.03
        else:
            return
        # Reinforce TWO pools:
        #   (a) atoms recalled into the observation (general task-level
        #       context that the LLM saw),
        #   (b) atoms that brain_score consulted for the WINNING candidate
        #       (specific (verb, slot, task) signature matches that
        #       earned the +/- decision delta).
        # Union prevents double-counting when an atom appears in both.
        participating = list(set(self._cycle_recalled_ids) | set(self._winning_brain_ids))
        if not participating:
            return
        try:
            self.atom_store.reinforce(participating, delta)
        except Exception:  # noqa: BLE001
            pass

    def _persist_plan_atom(self, plan: Plan) -> None:
        """Write a TASK atom that summarizes the plan shape.

        Lets the brain co-activate "tasks that decompose this way"
        with their step atoms, which is the substrate Organ 9 will
        eventually mine for recipes (recurring plan shapes → recipe
        promotion candidates).
        """
        if not self._use_brain or self.atom_store is None:
            return
        from lattice.atoms import AtomType

        step_summary = " | ".join(s.description[:60] for s in plan.steps)
        content = (
            f"Plan for task '{plan.task[:120]}'. "
            f"{len(plan.steps)} steps: {step_summary[:400]}"
        )
        try:
            self.atom_store.add(
                content,
                type=AtomType.TASK,
                region="plans",
                tags=("plan-dag", f"steps={len(plan.steps)}"),
                importance=0.55,
            )
        except Exception:  # noqa: BLE001
            pass

    def _record_step_outcome(self, record: "StepRecord") -> None:
        """Write a SKILL atom on success / ANTIPATTERN on failure.

        Per-step granularity: the brain accumulates fine-grained
        outcome data for THIS verb on THIS kind of slot value, not
        just an aggregate end-of-task summary. The next call to
        `_brain_score` for a similar candidate will surface this atom
        and bias selection accordingly.

        Uses self._active_task (the step text the proposer was given
        THIS cycle) so the atom embeds near where a future cycle's
        brain_score query will land — both sides share the same
        template via _signature_with_task. Without this alignment the
        atom written for step N is in a different part of embedding
        space from the query in step N+1, and brain_score misses it.

        Skipped when brain is off, when there's no atom_store, or when
        the action isn't a mutating one (non-mutating verbs don't
        teach the brain anything that helps future selection).
        """
        if not self._use_brain or self.atom_store is None:
            return
        if record.action is None:
            return
        if not self._is_mutating(record.action):
            return

        from lattice.atoms import AtomType

        sig = self._action_signature(record.action)
        embed_key = _signature_with_task(sig, self._active_task)
        ok = record.kind == "edit" and not record.error
        if ok:
            content = (
                f"{embed_key} :: Success. The attempt passed verification."
            )
            atype = AtomType.SKILL
            importance = 0.65
            tags = ("agent-step", "success", record.action.verb)
        else:
            content = (
                f"{embed_key} :: Failure. {(record.error or 'unknown')[:240]}"
            )
            atype = AtomType.ANTIPATTERN
            importance = 0.6
            tags = ("agent-step", "failure", record.action.verb)
        try:
            self.atom_store.add(
                content,
                type=atype,
                region="steps",
                tags=tags,
                importance=importance,
            )
        except Exception:  # noqa: BLE001
            pass

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

        # PLAN-DAG view (Phase 2): on multi-step tasks, the proposer
        # receives the CURRENT step as the task (not the original
        # blob) and sees the full plan as a hint. This forces the
        # small model to focus on one thing at a time instead of
        # trying to plan AND act in the same turn.
        active_task = task
        if self._plan is not None and self._plan.is_multistep():
            plan_view = self._plan.render()
            if plan_view:
                hints.append(plan_view)
            active_task = self._plan.render_task()
        # Stash for brain_score + record_step_outcome so write & query
        # use the SAME task string (otherwise atoms written for step N
        # don't embed near the brain_score query in step N+1).
        self._active_task = active_task

        # VERIFY-FAILURE FEEDBACK (Phase 3): when the last step was an
        # error from verify (parse / type / tests), surface it as the
        # FIRST hint — top of the prompt, hard to miss. The model
        # sees what broke and the exact action that caused it, so the
        # next propose call is a real self-correction attempt rather
        # than a blind retry.
        if steps and steps[-1].kind == "error" and steps[-1].error:
            failure_block = _format_failure_hint(steps[-1])
            if failure_block:
                hints.append(failure_block)

        # STEER (Organ 4, lightweight v0): high-importance atoms — those
        # EVOLVE has boosted because the system has lived through their
        # pattern — go FIRST in the hint list so they sit in the LLM's
        # most-attended prompt position. The heavyweight path
        # (activation patching) is documented in steer/__init__.py.
        if self._use_brain and self.atom_store is not None:
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
        # Reset the participating-IDs list each cycle; reinforce() at
        # end of step uses what's collected here as 'the atoms that
        # showed up for THIS decision'.
        self._cycle_recalled_ids = []
        if self._use_brain and self.atom_store is not None:
            try:
                hits = self.atom_store.recall(task, k=4)
            except Exception:
                hits = []
            for r in hits:
                hints.append(
                    f"{r.atom.type.value} (score {r.score:.2f}): {r.atom.content[:600]}"
                )
                atom_id = getattr(r.atom, "id", None)
                if atom_id is not None:
                    self._cycle_recalled_ids.append(int(atom_id))

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

        # CONTEXT BUDGET (Phase 3): hints accumulate per-cycle; on long
        # multi-step runs with priming + recall + history + code search
        # they can blow past the small model's context. Apply a final
        # priority-aware budget so the most important hints survive and
        # the rest get truncated or dropped. Rule of thumb: ~4 chars per
        # token; budget ~8000 chars ≈ 2000 hint tokens, leaving room for
        # the system prompt + symbols + few-shot.
        hints = _apply_context_budget(hints, max_chars=8000)

        return ObservationContext(
            task=active_task, symbols=tuple(symbols[:30]), hints=tuple(hints)
        )

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


def _format_failure_hint(step: "StepRecord") -> str:
    """Render the previous step's failure as a 'FIX REQUIRED' block.

    The block sits at the head of the hints list so it lands in the
    LLM's most-attended position. Includes:
      - what was attempted (verb + key slot values from the action)
      - the actual error message (parse / type / test failure)
      - an explicit directive that the next action must AVOID this
        failure mode.

    Empty when the step has no error string.
    """
    if not step.error:
        return ""
    verb = step.verb or "?"
    action_summary = ""
    if step.action is not None:
        try:
            dump = step.action.model_dump(mode="json")
            slots = {k: v for k, v in dump.items() if k not in {"verb", "confidence"}}
            action_summary = "; ".join(
                f"{k}={_compact_value(v)}" for k, v in slots.items()
            )
        except Exception:  # noqa: BLE001
            action_summary = ""

    lines = [
        "FIX REQUIRED — your previous attempt failed verification.",
        f"  attempted: {verb}" + (f" ({action_summary})" if action_summary else ""),
        f"  failed because: {step.error[:600]}",
        "Your next action MUST address this failure — pick a different "
        "approach, fix the broken slot value, or emit MarkBlocked if you "
        "cannot proceed.",
    ]
    return "\n".join(lines)


def _compact_value(v) -> str:
    """One-line repr of a slot value for the failure hint."""
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_compact_value(val)}" for k, val in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ",".join(_compact_value(x) for x in v) + "]"
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def _count_diff_added_lines(diff: str) -> int:
    """Count '+' (added) lines in a unified diff, skipping the header."""
    if not diff:
        return 0
    n = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            n += 1
    return n


def _signature_with_task(signature: str, task: str) -> str:
    """Canonical template the brain uses to embed step outcomes AND
    to query for matching atoms. Both sides MUST use this exact
    template; otherwise the embedding similarity between a stored
    atom and the query that should match it is uselessly low.
    Truncates the task to 120 chars (enough to disambiguate
    different tasks, short enough to keep the embedding stable).
    """
    return f"{signature} :: task '{task[:120]}'"


def _hint_priority(hint: str) -> int:
    """Lower number = higher priority. Used by _apply_context_budget.

    Ordering rationale:
      0  FIX REQUIRED — last cycle's verify failure; the LLM MUST
         see this to self-correct.
      1  PLAN: — current step is the proposer's task; the plan view
         tells it 'where we are' for the rest of the run.
      2  Priming block (steering) — sticky, high-importance atoms.
      3  WORKSPACE FILES — pins paths so the LLM can't hallucinate.
      4  Atom recall hits — fluid, task-specific knowledge.
      5  CODE — code-search chunks; nice-to-have grounding.
      6  step-history lines — useful but tail-droppable.
      7  FILES ALREADY EDITED — bookkeeping.
      8  Coda ('Look at the history…') — instructions, smallest.
      9  Anything else.
    """
    head = hint[:50]
    if head.startswith("FIX REQUIRED"):
        return 0
    if head.startswith("PLAN:"):
        return 1
    if "priming" in head.lower() or head.startswith("STEER"):
        return 2
    if head.startswith("WORKSPACE FILES"):
        return 3
    if " (score " in head and "): " in head:  # atom recall
        return 4
    if head.startswith("CODE "):
        return 5
    if head.startswith("step "):
        return 6
    if head.startswith("FILES ALREADY EDITED"):
        return 7
    if head.startswith("Look at the history"):
        return 8
    return 9


def _apply_context_budget(hints: list[str], *, max_chars: int) -> list[str]:
    """Prune hints down to a character budget, keeping the highest-priority.

    Approach:
      1) Sort by priority (stable so same-class hints stay in original
         order — e.g. step history stays oldest-first).
      2) Accumulate until budget consumed.
      3) If a single high-priority hint would alone exceed the budget,
         truncate it to fit rather than drop entirely (FIX REQUIRED
         must always survive, even truncated).
      4) Return hints in their original ORDER (so the prompt structure
         stays predictable for the LLM) — priority is only used to
         decide WHICH to keep, not what order to render.
    """
    if not hints:
        return hints
    total = sum(len(h) for h in hints)
    if total <= max_chars:
        return hints

    indexed = list(enumerate(hints))
    indexed.sort(key=lambda t: (_hint_priority(t[1]), t[0]))

    kept_indices: set[int] = set()
    remaining = max_chars
    truncated: dict[int, str] = {}
    for orig_idx, h in indexed:
        if len(h) <= remaining:
            kept_indices.add(orig_idx)
            remaining -= len(h)
            continue
        if not kept_indices and remaining > 200:
            # First (highest-priority) hint already too big — truncate.
            truncated[orig_idx] = h[: remaining - 20] + "…[truncated]"
            kept_indices.add(orig_idx)
            remaining = 0
            continue
        if remaining > 300 and _hint_priority(h) <= 1:
            # Always keep FIX REQUIRED / PLAN even if we have to truncate.
            truncated[orig_idx] = h[: remaining - 20] + "…[truncated]"
            kept_indices.add(orig_idx)
            remaining = 0
            continue
        # Out of budget for this hint; drop.
    return [
        truncated.get(i, h)
        for i, h in enumerate(hints)
        if i in kept_indices
    ]


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
