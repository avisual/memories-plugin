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
import re
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
    actions: list,
    files_touched: list[str],
) -> int | None:
    """Persist a single agent-run trace so EVOLVE can mine it later.

    Accepts either:
    - actions: list[str] (legacy — verb names only)
    - actions: list[dict] (preferred — full action JSON records,
      including slot values; required for template inference in
      Organ 8's apprentice).

    Stored as an EXPERIENCE atom in the `traces` region with both a
    human-readable summary and a JSON payload (parsed by discover()
    and the apprentice).
    """
    if not task or not actions:
        return None

    # Normalise to the structured form so downstream code can rely on
    # action[i] being a dict. Legacy verb-string entries become
    # {"verb": <name>} dicts; slot values remain unknown for those.
    structured: list[dict] = []
    for a in actions:
        if isinstance(a, dict):
            structured.append(a)
        elif isinstance(a, str):
            structured.append({"verb": a})
        else:
            try:
                # Pydantic Action instances.
                structured.append(a.model_dump(mode="json"))
            except Exception:  # noqa: BLE001
                structured.append({"verb": str(a)})

    verbs = " → ".join(s.get("verb", "?") for s in structured)
    payload = {
        "task": task,
        "actions": structured,
        "files": list(files_touched),
    }
    content = (
        f"TRACE: {task[:200]}\n"
        f"actions: {verbs}\n"
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


def _action_verbs(actions: list) -> list[str]:
    """Extract the verb name from each action (str OR dict OR object)."""
    out: list[str] = []
    for a in actions:
        if isinstance(a, dict):
            out.append(str(a.get("verb", "?")))
        elif isinstance(a, str):
            out.append(a)
        else:
            v = getattr(a, "verb", None)
            if v is not None:
                out.append(str(v))
            else:
                out.append("?")
    return out


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


def _signature(actions: Iterable) -> tuple[str, ...]:
    """Stable verb-sequence key: drop arguments, keep order.

    Accepts both the legacy verb-name list and the structured dict list;
    extracts verbs uniformly via `_action_verbs`.
    """
    return tuple(_action_verbs(list(actions)))


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
        if not sig:
            continue
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


@dataclass(frozen=True)
class LearnedTemplate:
    """A regex-shaped pattern inferred from N successful traces.

    The pattern's capture groups map to JSON-paths in the recorded
    action, so applying the template to a new matching task substitutes
    the new task's captures into the right slots.

    Fields:
    - task_regex: compiled regex matching the task-text shape; named
      groups 'c0', 'c1', ... carry the variable parts.
    - action_template: the recorded action dict with placeholder
      strings `__LATTICE_CAPTURE_N__` substituted into the fields
      that varied across samples.
    - signature: the verb sequence (today single-verb only).
    - sample_count: how many traces contributed.
    """

    task_regex: "re.Pattern[str]"  # forward-ref-free at runtime
    action_template: dict
    signature: tuple[str, ...]
    sample_count: int


def _tokenize(task: str) -> list[str]:
    """Whitespace-split tokens; keeps punctuation glued (good enough for v0)."""
    return task.strip().split()


def _walk_string_fields(obj, path=()):
    """Yield (path_tuple, value) for every str value in a nested dict/list."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk_string_fields(value, path + (key,))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _walk_string_fields(item, path + (i,))
    elif isinstance(obj, str):
        yield path, obj


def _set_at_path(obj, path, value) -> None:
    cur = obj
    for step in path[:-1]:
        cur = cur[step]
    cur[path[-1]] = value


def infer_template(traces: list[dict]) -> LearnedTemplate | None:
    """Infer a (task-regex, action-template) from a list of trace payloads.

    Returns None when traces don't agree well enough to template:
    - Fewer than 2 samples.
    - Task token-counts differ across samples.
    - More than one action recorded per trace (v0 is single-action).
    - No string field in the action dict tracks any task capture.

    The honest scope: this v0 handles single-action sequences where
    task words map directly to action string fields (the AddImport /
    AddParameter / AddField families). It does NOT yet handle multi-
    action sequences or non-string slots (ints, booleans, nested
    typed exprs); those land in a follow-up.
    """
    if len(traces) < 2:
        return None

    tokens_per_sample = [_tokenize(t.get("task", "")) for t in traces]
    if any(not toks for toks in tokens_per_sample):
        return None
    token_count = len(tokens_per_sample[0])
    if any(len(toks) != token_count for toks in tokens_per_sample):
        return None  # variable-length tasks — v0 punts

    # Identify capture positions (tokens that vary across samples).
    capture_positions: list[int] = []
    pattern_parts: list[str] = []
    for i in range(token_count):
        values = [toks[i] for toks in tokens_per_sample]
        if all(v == values[0] for v in values):
            pattern_parts.append(re.escape(values[0]))
        else:
            cap_idx = len(capture_positions)
            capture_positions.append(i)
            pattern_parts.append(f"(?P<c{cap_idx}>\\S+)")
    if not capture_positions:
        return None  # all-constant tasks aren't worth templating
    task_regex = re.compile(r"^" + r"\s+".join(pattern_parts) + r"$")

    # Recover capture values per sample, then align to action string slots.
    captures_per_sample: list[list[str]] = [
        [toks[pos] for pos in capture_positions]
        for toks in tokens_per_sample
    ]

    # Use the FIRST trace's action as the template skeleton; require all
    # traces to have exactly one action AND the same verb.
    actions_first = traces[0].get("actions") or []
    if len(actions_first) != 1 or not isinstance(actions_first[0], dict):
        return None
    base_action = json.loads(json.dumps(actions_first[0]))  # deep copy
    base_verb = base_action.get("verb")
    for tr in traces[1:]:
        acts = tr.get("actions") or []
        if len(acts) != 1 or not isinstance(acts[0], dict):
            return None
        if acts[0].get("verb") != base_verb:
            return None

    # For each string field in the base action, check whether its value
    # changes across samples; if so, find which capture index has the
    # same value at that sample index and substitute a placeholder.
    string_field_paths = list(_walk_string_fields(base_action))
    substitution_made = False
    for path, _value in string_field_paths:
        values_at_path: list[str] = []
        for tr in traces:
            cur = tr["actions"][0]
            for step in path:
                cur = cur[step]
            values_at_path.append(cur)
        if all(v == values_at_path[0] for v in values_at_path):
            continue
        # Find which capture column matches these values across samples.
        for cap_i in range(len(capture_positions)):
            if [c[cap_i] for c in captures_per_sample] == values_at_path:
                _set_at_path(base_action, path, f"__LATTICE_CAPTURE_{cap_i}__")
                substitution_made = True
                break

    if not substitution_made:
        return None  # no learnable mapping

    return LearnedTemplate(
        task_regex=task_regex,
        action_template=base_action,
        signature=(base_verb or "?",),
        sample_count=len(traces),
    )


def apply_template(template: LearnedTemplate, task: str) -> dict | None:
    """If *task* matches the template, return the substituted action dict."""
    match = template.task_regex.match(task)
    if not match:
        return None
    out = json.loads(json.dumps(template.action_template))  # deep copy
    for cap_i in range(len(match.groupdict())):
        key = f"c{cap_i}"
        if key not in match.groupdict():
            continue
        cap_value = match.group(key)
        for path, value in list(_walk_string_fields(out)):
            if value == f"__LATTICE_CAPTURE_{cap_i}__":
                _set_at_path(out, path, cap_value)
    return out


def learned_templates(store: AtomStore, *, min_recurrence: int = 3) -> list[LearnedTemplate]:
    """Discover templates from boosted traces in the store.

    Combines `discover()` (find recurring sequences) with `infer_template`
    (template synthesis from each cluster). Used by ApprenticeProposer at
    propose-time to handle novel tasks via learned shapes.
    """
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT content FROM atoms WHERE region = ? AND type = ?",
        (_TRACE_REGION, _TRACE_TYPE.value),
    ).fetchall()
    by_sig: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for (content,) in rows:
        payload = _parse_trace_payload(content)
        if not payload:
            continue
        actions = payload.get("actions") or []
        if not actions:
            continue
        sig = _signature(actions)
        if sig:
            by_sig[sig].append(payload)

    templates: list[LearnedTemplate] = []
    for sig, payloads in by_sig.items():
        if len(payloads) < min_recurrence:
            continue
        tmpl = infer_template(payloads)
        if tmpl is not None:
            templates.append(tmpl)
    # Most-supported first.
    templates.sort(key=lambda t: -t.sample_count)
    return templates


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
