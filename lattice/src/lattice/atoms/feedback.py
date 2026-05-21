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
    actions_summary: list[str],
    files_touched: list[str],
    project_region: str | None = None,
) -> int:
    """Store a single experience atom summarizing a successful task.

    Returns the atom id. Content is structured so it surfaces well
    under future task recall (the task wording stays prominent).
    """
    region = project_region or "experiences"
    actions_clause = "; ".join(actions_summary[:5])
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


def summarize_trace_for_experience(trace_or_report: Any) -> tuple[list[str], list[str]]:
    """Extract (action_summaries, files_touched) from an AgentTrace or
    MultiSubtaskReport. Tolerant of either shape via duck-typing.
    """
    actions: list[str] = []
    files: list[str] = list(getattr(trace_or_report, "final_files", {}).keys())

    traces = getattr(trace_or_report, "traces", None)
    if traces is not None:
        for t in traces:
            for s in t.steps:
                if s.kind == "edit" and s.verb and s.diff:
                    actions.append(s.verb)
        return actions, files

    steps = getattr(trace_or_report, "steps", ())
    for s in steps:
        if s.kind == "edit" and s.verb and s.diff:
            actions.append(s.verb)
    return actions, files
