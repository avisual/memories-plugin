"""Planner — decomposes a task into a sequence of subtasks.

v0 is deterministic: split on explicit conjunctions and step markers
("then", "and then", numbered lists). Quality is bounded by how cleanly
the user writes the task, which is fine for an opt-in flag.

A later iteration replaces this with an LLM planner call: a slightly
larger (or differently prompted) model emits the subtask list once at
the start, and the small executor model runs each subtask. Same shape.
"""

from __future__ import annotations

import re

_CONJUNCTION_RE = re.compile(
    r"(?:\s+then\s+|\s+and\s+then\s+|\s*;\s*|^\s*\d+\.\s+)",
    re.IGNORECASE,
)


def decompose(task: str) -> list[str]:
    """Split *task* into ordered subtasks.

    Heuristic for v0:
    - If the task contains numbered steps ("1. ... 2. ..."), split on
      them.
    - Otherwise split on " then " / "; " / " and then ".
    - Empty pieces dropped; subtasks are stripped of leading/trailing
      whitespace.
    """
    if not task.strip():
        return []

    # Numbered: "1. step one 2. step two"
    numbered_match = re.search(r"\b\d+\.\s+", task)
    if numbered_match:
        parts = re.split(r"\b\d+\.\s+", task)
        return [p.strip() for p in parts if p.strip()]

    parts = _CONJUNCTION_RE.split(task)
    return [p.strip().rstrip(".;,") for p in parts if p.strip()]
