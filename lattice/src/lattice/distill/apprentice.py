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

        # TEMPLATE PATH (the real DISTILL): learned templates inferred
        # from recurring traces match novel task wording directly. When
        # one matches, we emit the substituted action WITHOUT consulting
        # recall or the LLM. This is the path that actually saves LLM
        # calls on new-but-shaped-the-same tasks.
        actions: list[Action] = []
        seen_dumps: set[str] = set()
        try:
            from lattice.atoms import apply_template, learned_templates, parse_action_from_dict  # noqa: F401
        except ImportError:
            pass
        try:
            from lattice.actions import parse_action
            from lattice.atoms.evolve import apply_template_chain, learned_templates

            for tmpl in learned_templates(
                self.atom_store, min_recurrence=2
            ):
                if tmpl.sample_count < 2:
                    continue
                substituted_chain = apply_template_chain(tmpl, obs.task)
                if not substituted_chain:
                    continue
                # Emit EVERY step in the chain as a candidate. The
                # orchestrator's pre-flight scorer picks whichever step
                # isn't yet a no-op against the overlay — that's how
                # multi-action templates drive across multiple cycles:
                # cycle 1 picks step 0 (only one that isn't no-op);
                # cycle 2 picks step 1 (step 0 is now no-op); etc.
                for substituted in substituted_chain:
                    try:
                        action = parse_action(substituted)
                    except Exception:  # noqa: BLE001
                        continue
                    dump = action.model_dump_json()
                    if dump in seen_dumps:
                        continue
                    seen_dumps.add(dump)
                    actions.append(action)
                    if len(actions) >= n:
                        return actions
        except Exception:  # noqa: BLE001
            pass

        # RECALL PATH (legacy): nearest-neighbour trace replay. Useful
        # when the task is similar but the template path didn't fire
        # (e.g. only one prior trace, or variable-length task tokens).
        try:
            results = self.atom_store.recall(obs.task, k=max(n * 2, 4))
        except Exception:  # noqa: BLE001
            return actions
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
            first = recorded_actions[0]
            # Trace atoms can now hold full action JSON. If they do,
            # rebuild the action directly from that JSON (the slot
            # values are baked in). If they're just verb names (legacy),
            # fall back to pattern-based reconstruction.
            recorded_first = _action_from_dict_or_verb(first, payload)
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


def _action_from_dict_or_verb(first, payload: dict) -> Action | None:
    """Reconstruct an Action from the recorded first-action.

    Two paths:

    1. Structured form (preferred): `first` is a dict with the full
       action JSON — `parse_action` rebuilds the typed Action directly.
       This is the path that lets the apprentice handle NOVEL tasks
       (the slot values come from the trace, not from re-parsing the
       new task text).

    2. Legacy form: `first` is just a verb name string. Fall back to
       reconstructing via PatternProposer on the trace's original
       task text — narrow but honest.

    Returns None when the action can't be rebuilt safely.
    """
    from lattice.actions import parse_action
    from lattice.propose.pattern import task_to_action

    if isinstance(first, dict):
        try:
            return parse_action(first)
        except Exception:  # noqa: BLE001
            verb = first.get("verb")
            if verb is None:
                return None
            first = verb  # fall through to legacy path

    if isinstance(first, str):
        verb = first
        original_task = payload.get("task", "")
        if not original_task:
            return None
        try:
            action = task_to_action(original_task)
        except Exception:  # noqa: BLE001
            return None
        if action is not None and action.verb == verb:
            return action

    return None


# Structural Proposer-protocol check (cheap; doesn't load anything).
def _selfcheck() -> None:  # pragma: no cover
    p: Proposer = ApprenticeProposer(atom_store=None)  # type: ignore[arg-type]
    _ = p
