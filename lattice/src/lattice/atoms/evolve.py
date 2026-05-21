"""EVOLVE — Organ 9: macro discovery from successful traces.

After every successful agent run, the harness writes a structured
trace atom with the (task, action sequence, files touched) tuple.
`lattice evolve` later scans those traces, clusters by task-text
similarity (MiniLM embeddings via the existing atom store), and
surfaces recurring patterns as candidate macros — task shapes the
system handles often enough that they could be promoted to first-
class PatternProposer entries.

This is the IMPROVES dial: the system's own pattern library grows
from observed use. v0 stops at discovery (print the candidates);
phase 2 will auto-wire approved candidates into the proposer chain
at startup.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from lattice.atoms.atom import AtomType
from lattice.atoms.store import AtomStore, RecallResult


_TRACE_REGION = "traces"
_TRACE_TAG = "agent-trace"
_TRACE_TYPE = AtomType.EXPERIENCE


def write_trace(
    store: AtomStore,
    *,
    task: str,
    actions: list[str],
    files_touched: list[str],
) -> int | None:
    """Persist a single agent-run trace so EVOLVE can mine it later.

    Stored as an EXPERIENCE atom in the `traces` region. The JSON
    payload (task / actions / files) is appended to the content so
    both readability and downstream structured parsing work.
    """
    if not task or not actions:
        return None
    payload = {
        "task": task,
        "actions": list(actions),
        "files": list(files_touched),
    }
    content = (
        f"TRACE: {task[:200]}\n"
        f"actions: {' → '.join(actions)}\n"
        f"files: {', '.join(files_touched) if files_touched else '-'}\n"
        f"JSON: {json.dumps(payload, ensure_ascii=False)}"
    )
    atom = store.add(
        content,
        type=_TRACE_TYPE,
        region=_TRACE_REGION,
        tags=(_TRACE_TAG,),
        importance=0.55,
    )
    return atom.id


def _parse_trace_payload(content: str) -> dict | None:
    marker = "\nJSON: "
    idx = content.rfind(marker)
    if idx < 0:
        return None
    try:
        return json.loads(content[idx + len(marker):])
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class MacroCandidate:
    """A recurring (task-shape, action-sequence) pair worth promoting."""

    action_sequence: tuple[str, ...]
    sample_tasks: tuple[str, ...]
    sample_count: int
    avg_recall_score: float = 0.0


@dataclass(frozen=True)
class EvolveReport:
    candidates: tuple[MacroCandidate, ...]
    total_traces: int


def _signature(actions: Iterable[str]) -> tuple[str, ...]:
    """Stable verb-sequence key: drop arguments, keep order."""
    return tuple(actions)


def discover(
    store: AtomStore,
    *,
    min_recurrence: int = 3,
    max_candidates: int = 20,
) -> EvolveReport:
    """Scan trace atoms and surface recurring task→action-sequence patterns.

    Algorithm:
    1. Pull every trace atom from the `traces` region.
    2. Parse the embedded JSON payload to recover (task, actions).
    3. Group by action-sequence signature (the ordered verb tuple).
    4. Emit a candidate for every group whose size >= min_recurrence.

    The clustering is exact-sequence for v0 (verb order matters; argument
    differences don't). A future iteration will fuzzy-match task text via
    embedding similarity so 'add an os import' and 'import json please'
    group together when they share the same action shape.
    """
    # Fetch all trace atoms. We use a broad recall — empty-query short-
    # circuits to [] but a generic query like 'trace' surfaces them.
    # Simpler: read the store directly.
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT content, importance FROM atoms WHERE region = ? AND type = ?",
        (_TRACE_REGION, _TRACE_TYPE.value),
    ).fetchall()

    by_sig: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    parsed_count = 0
    for content, _importance in rows:
        payload = _parse_trace_payload(content)
        if not payload:
            continue
        actions = payload.get("actions") or []
        if not actions:
            continue
        sig = _signature(actions)
        by_sig[sig].append(payload)
        parsed_count += 1

    candidates: list[MacroCandidate] = []
    for sig, payloads in by_sig.items():
        if len(payloads) < min_recurrence:
            continue
        sample_tasks = tuple(p.get("task", "") for p in payloads[:5])
        candidates.append(
            MacroCandidate(
                action_sequence=sig,
                sample_tasks=sample_tasks,
                sample_count=len(payloads),
            )
        )

    # Most recurrent first.
    candidates.sort(key=lambda c: c.sample_count, reverse=True)
    return EvolveReport(
        candidates=tuple(candidates[:max_candidates]),
        total_traces=parsed_count,
    )


def boost_recurrent_traces(store: AtomStore, *, min_recurrence: int = 3) -> int:
    """Phase 2 of Organ 9: when a trace pattern reaches min_recurrence,
    raise the importance of those trace atoms so atom recall surfaces
    them as a stronger hint to the LLM.

    The IMPROVES loop closes through the existing observation pipeline:
    high-importance recurring traces flow into the next turn's HINTS
    without any new code path — the LLM sees 'similar past tasks
    consistently used this action sequence' and acts accordingly.

    Returns the number of trace atoms whose importance was raised.
    """
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT id, content FROM atoms WHERE region = ? AND type = ?",
        (_TRACE_REGION, _TRACE_TYPE.value),
    ).fetchall()
    if not rows:
        return 0

    by_sig: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for atom_id, content in rows:
        payload = _parse_trace_payload(content)
        if not payload:
            continue
        sig = _signature(payload.get("actions") or [])
        if not sig:
            continue
        by_sig[sig].append(atom_id)

    boosted = 0
    for ids in by_sig.values():
        if len(ids) < min_recurrence:
            continue
        # Cap at 0.95 so seed atoms with importance 0.95+ still dominate.
        # Recurrence saturates after a while — 3 runs is enough to learn.
        new_importance = min(0.95, 0.55 + 0.1 * len(ids))
        store._conn.executemany(  # type: ignore[attr-defined]
            "UPDATE atoms SET importance = ? WHERE id = ?",
            [(new_importance, aid) for aid in ids],
        )
        boosted += len(ids)
    return boosted
