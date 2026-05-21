"""ApprenticeProposer — propose actions from learned high-importance traces.

When EVOLVE's `boost_recurrent_traces` raises a trace atom's importance
above threshold, the apprentice exposes that trace as a runtime
proposer: a new task whose embedding closely matches the trace's
recorded task text gets the trace's recorded first action as a
candidate. The population scorer (Organ 6) then evaluates whether to
use it — if the recorded action is a no-op against the new workspace,
it's discarded and the LLM gets its turn.

Why this is a real DISTILL:
- Encodes 'what worked last time we did something like this' as a
  runtime rule.
- No LLM call. No model retraining. The proposer chain skips the
  LLM entirely when an apprentice rule fires.
- The rules grow with use (EVOLVE writes new traces; --apply
  boosts them; apprentice rescans next time).

What this is NOT:
- A trained policy net. That's the heavyweight Organ 8 — a tiny
  transformer head learning (obs → action) mappings from the trace
  store. Out of scope for this session.
- A perfect match. The trace's recorded action might have wrong
  file paths for the new task; the population scorer will catch
  the no-op and fall through to the LLM.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from lattice.actions import Action, parse_action
from lattice.propose.base import ObservationContext, Proposer


_TRACE_REGION = "traces"
_DEFAULT_MIN_IMPORTANCE = 0.7
_DEFAULT_MIN_SIMILARITY = 0.6


class ApprenticeProposer:
    """Proposes actions from high-importance learned traces.

    On each `propose` call:
    1. Recall the top-K traces matching the task.
    2. Skip traces below `min_importance` (not yet EVOLVE-boosted).
    3. Parse the trace's JSON payload; rebuild the first action.
    4. If the cosine recall score clears `min_similarity`, return it.

    Returns [] (falls through to the next proposer) when no learned
    trace is confident enough.
    """

    def __init__(
        self,
        atom_store: Any,
        *,
        min_importance: float = _DEFAULT_MIN_IMPORTANCE,
        min_similarity: float = _DEFAULT_MIN_SIMILARITY,
    ) -> None:
        self.atom_store = atom_store
        self.min_importance = min_importance
        self.min_similarity = min_similarity

    def propose(self, obs: ObservationContext, n: int = 1) -> list[Action]:
        if self.atom_store is None or n <= 0 or not obs.task.strip():
            return []
        try:
            results = self.atom_store.recall(obs.task, k=max(n * 2, 4))
        except Exception:  # noqa: BLE001
            return []

        actions: list[Action] = []
        seen_dumps: set[str] = set()
        for r in results:
            atom = r.atom
            if atom.region != _TRACE_REGION:
                continue
            if atom.importance < self.min_importance:
                continue
            # score = cosine + 0.10*importance. Strip the importance
            # bonus before comparing against the similarity threshold.
            cosine_est = r.score - 0.10 * atom.importance
            if cosine_est < self.min_similarity:
                continue
            payload = _parse_trace_payload(atom.content)
            if payload is None:
                continue
            recorded_actions = payload.get("actions") or []
            if not recorded_actions:
                continue
            recorded_first = _action_for_verb(recorded_actions[0], payload)
            if recorded_first is None:
                continue
            dump = recorded_first.model_dump_json()
            if dump in seen_dumps:
                continue
            seen_dumps.add(dump)
            actions.append(recorded_first)
            if len(actions) >= n:
                break
        return actions


def _parse_trace_payload(content: str) -> dict | None:
    marker = "\nJSON: "
    idx = content.rfind(marker)
    if idx < 0:
        return None
    try:
        return json.loads(content[idx + len(marker):])
    except json.JSONDecodeError:
        return None


def _action_for_verb(verb: str, payload: dict) -> Action | None:
    """Best-effort reconstruction of an Action from a verb name + trace.

    The trace records the verb name only (no slots). For an apprentice
    to emit a USEFUL candidate, we need slot values. For v0 we only
    support verbs whose 'slots' are derivable from the task text or
    workspace files via the existing PatternProposer. Falls back to
    None for verbs the apprentice can't reconstruct yet.

    The honest read of this method: v0 apprentice REQUIRES a matching
    pattern at the task-text level to fill in slots. When that exists,
    the apprentice's value is signaling 'this pattern has worked
    before' — which the population scorer picks up via the recurrence
    boost on the candidate's atom. Phase 2 will add slot-substitution
    via pattern templates inferred from the trace's sample tasks.
    """
    # For v0 we delegate to PatternProposer on the trace's original
    # task text: if the trace's task matches a known pattern, the
    # pattern reconstructs the typed Action. This is intentionally
    # narrow — it means the apprentice only fires for tasks the
    # PatternProposer would have handled anyway, but with the bonus
    # that the trace embedding finds them even when the task wording
    # is slightly different.
    try:
        from lattice.propose.pattern import task_to_action

        original_task = payload.get("task", "")
        if not original_task:
            return None
        action = task_to_action(original_task)
        if action is not None and action.verb == verb:
            return action
    except Exception:  # noqa: BLE001
        return None
    return None


# Structural Proposer-protocol check (cheap; doesn't load anything).
def _selfcheck() -> None:  # pragma: no cover
    p: Proposer = ApprenticeProposer(atom_store=None)  # type: ignore[arg-type]
    _ = p
