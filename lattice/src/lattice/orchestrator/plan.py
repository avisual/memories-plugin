"""Plan-DAG — multi-step task structure the agent reasons over.

The agent loop is no longer just "one action per turn until MarkDone."
For real coding work (refactor, add feature with tests, multi-file
edit) it needs an explicit *plan*: an ordered list of subtasks with
status, attempt count, and the dependency that step N+1 should not
start until step N either finishes or is explicitly skipped.

v0 of the plan is built deterministically from the task string via
the existing `decompose` heuristic. v1 will replace `Plan.build` with
an LLM planner call (small model, single turn, asked for the step
list) — the loop machinery here is unchanged.

The plan is two things at once:
  - State the loop drives forward (current_step advances on success).
  - A signal *to the LLM* (rendered as a hint in the observation), so
    the executor model knows where it is in the work and doesn't try
    to redo earlier steps or skip ahead.

Persisted as a TASK atom at run start so the brain sees the plan
shape; downstream this becomes the substrate for plan-DAG atoms (a
plan is a co-firing structure of subtask atoms; macros emerge from
recurring plan shapes).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PlanStep(BaseModel):
    """One subtask in a Plan. Mutable so status can advance in place."""

    model_config = ConfigDict()

    idx: int = Field(ge=1)
    description: str
    status: str = "pending"  # 'pending' | 'in-progress' | 'done' | 'blocked'
    attempts: int = 0
    note: str = ""


class Plan(BaseModel):
    """An ordered set of PlanSteps that drive a multi-cycle run.

    Equality / mutation note: PlanSteps are mutated in place by the
    agent loop (status advances). Plan itself is otherwise read-mostly.
    """

    model_config = ConfigDict()

    task: str
    steps: list[PlanStep] = Field(default_factory=list)

    @classmethod
    def build(cls, task: str) -> "Plan":
        """Build a deterministic plan from the task string.

        Uses decompose() — splits on conjunctions and numbered markers.
        A single-step task gets a one-element plan; the loop machinery
        is the same either way (no special-case branch on len==1).
        """
        from lattice.orchestrator.planner import decompose

        parts = decompose(task)
        if not parts:
            parts = [task.strip()]
        steps = [
            PlanStep(idx=i + 1, description=p)
            for i, p in enumerate(parts)
            if p.strip()
        ]
        if not steps:
            steps = [PlanStep(idx=1, description=task.strip() or "?")]
        return cls(task=task, steps=steps)

    def current(self) -> PlanStep | None:
        """Return the first step that is not done/blocked.

        Steps in 'in-progress' are returned ahead of 'pending' — the
        loop sets in-progress when it starts a step and clears to done
        only on a successful verify.
        """
        for s in self.steps:
            if s.status == "in-progress":
                return s
        for s in self.steps:
            if s.status == "pending":
                return s
        return None

    def advance(self) -> PlanStep | None:
        """Mark the current step done; return the next one (or None).

        Called from the agent loop after a successful mutating step.
        Safe to call multiple times — once all steps are done returns
        None and the loop terminates by 'plan complete'.
        """
        cur = self.current()
        if cur is None:
            return None
        cur.status = "done"
        return self.current()

    def begin(self, step: PlanStep) -> None:
        """Move a pending step to in-progress and bump attempt count."""
        if step.status == "pending":
            step.status = "in-progress"
        step.attempts += 1

    def block(self, reason: str = "") -> None:
        """Mark the current step blocked (loop will stop)."""
        cur = self.current()
        if cur is not None:
            cur.status = "blocked"
            cur.note = (reason or cur.note)[:200]

    def is_complete(self) -> bool:
        """All steps either done or blocked → plan finished."""
        return all(s.status in ("done", "blocked") for s in self.steps)

    def is_multistep(self) -> bool:
        return len(self.steps) > 1

    def render(self) -> str:
        """ASCII view for an observation hint.

        The current step is annotated with '>' so the LLM can see at a
        glance where to focus. Single-step plans render as a single
        line (no PLAN: header) so we don't waste prompt tokens on
        trivial decomposition.
        """
        if len(self.steps) <= 1:
            return ""
        marker_for = {
            "pending": "[ ]",
            "in-progress": "[>]",
            "done": "[x]",
            "blocked": "[!]",
        }
        lines = ["PLAN:"]
        for s in self.steps:
            m = marker_for.get(s.status, "[?]")
            attempt_note = f" (attempt {s.attempts})" if s.attempts > 1 else ""
            lines.append(f"  {m} {s.idx}. {s.description}{attempt_note}")
        return "\n".join(lines)

    def render_task(self) -> str:
        """Task string the proposer should attack THIS cycle.

        For a multi-step plan that's the current step. For a one-step
        plan or completed plan, the original task. This makes the
        proposer focus on one thing at a time without hiding the
        broader plan (which lives in render()).
        """
        cur = self.current()
        if cur is None:
            return self.task
        if not self.is_multistep():
            return self.task
        return cur.description
