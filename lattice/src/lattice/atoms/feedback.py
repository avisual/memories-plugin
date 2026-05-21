"""Hebbian feedback — write experience atoms back after agent runs.

When an agent loop succeeds, the harness records what it did as an
`experience` atom so the brain learns. Failed steps that recurred can
optionally be recorded as antipatterns.

This is the simplest possible Hebbian update loop: things that worked
in this context get stored; recall surfaces them next time a similar
task arrives; the brain accumulates lived knowledge.
"""

from __future__ import annotations

from typing import Any

from lattice.atoms.atom import AtomType
from lattice.atoms.store import AtomStore


def record_experience(
    *,
    store: AtomStore,
    task: str,
    actions_summary: list,
    files_touched: list[str],
    project_region: str | None = None,
) -> int:
    """Store a single experience atom summarizing a successful task.

    `actions_summary` now accepts either verb-name strings (legacy) or
    full action dicts (preferred — Organ 8 template inference needs the
    slot values). Stringification for the human-readable summary uses
    only the verb names.

    Returns the atom id. Also writes a parallel TRACE atom in
    region='traces' that EVOLVE mines for macro-promotion candidates.
    """
    region = project_region or "experiences"
    verb_names: list[str] = []
    for a in actions_summary[:5]:
        if isinstance(a, dict):
            verb_names.append(str(a.get("verb", "?")))
        else:
            verb_names.append(str(a))
    actions_clause = "; ".join(verb_names)
    files_clause = ", ".join(files_touched[:5]) if files_touched else "no files"
    content = (
        f"Did task: '{task[:140]}'. "
        f"Actions taken: {actions_clause}. "
        f"Files touched: {files_clause}."
    )
    atom = store.add(
        content,
        type=AtomType.EXPERIENCE,
        region=region,
        tags=("agent-run", "experience"),
        importance=0.65,
    )

    # Parallel structured trace for Organ 9 (macro discovery).
    try:
        from lattice.atoms.evolve import write_trace

        write_trace(
            store=store,
            task=task,
            actions=actions_summary,
            files_touched=files_touched,
        )
    except Exception:  # noqa: BLE001
        # Trace logging is best-effort; never break the user-facing flow.
        pass

    return atom.id  # type: ignore[return-value]


def record_antipattern(
    *,
    store: AtomStore,
    task: str,
    failure_summary: str,
    project_region: str | None = None,
) -> int:
    region = project_region or "antipatterns"
    content = (
        f"While doing '{task[:140]}', repeated failure: {failure_summary[:180]}. "
        "Reconsider approach for this kind of task."
    )
    atom = store.add(
        content,
        type=AtomType.ANTIPATTERN,
        region=region,
        tags=("agent-run", "antipattern"),
        importance=0.7,
    )
    return atom.id  # type: ignore[return-value]


def summarize_trace_for_experience(trace_or_report: Any) -> tuple[list, list[str]]:
    """Extract (action_records, files_touched) from an AgentTrace or
    MultiSubtaskReport. Tolerant of either shape via duck-typing.

    `action_records` is a list of either:
    - dicts (full action JSON: verb + slots), when the step's
      StepRecord carries an Action (which carries model_dump_json),
    - or verb-only strings, when the step had no Action (e.g.
      proposer raised).

    Downstream (Organ 8 apprentice template inference) needs the slot
    values, so we prefer the dict form.
    """
    actions: list = []
    files: list[str] = list(getattr(trace_or_report, "final_files", {}).keys())

    def _record_for(step: Any):
        if step.kind != "edit" or not step.verb or not step.diff:
            return None
        action = getattr(step, "action", None)
        if action is None:
            return step.verb
        try:
            return action.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            return step.verb

    traces = getattr(trace_or_report, "traces", None)
    if traces is not None:
        for t in traces:
            for s in t.steps:
                rec = _record_for(s)
                if rec is not None:
                    actions.append(rec)
        return actions, files

    steps = getattr(trace_or_report, "steps", ())
    for s in steps:
        rec = _record_for(s)
        if rec is not None:
            actions.append(rec)
    return actions, files
