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

    The regex's capture groups map to JSON-paths in the recorded
    action chain, so applying the template to a new matching task
    substitutes the new task's captures into every step.

    Fields:
    - task_regex: compiled regex matching the task-text shape; named
      groups 'c0', 'c1', ... carry the variable parts.
    - action_template: the FIRST step of the chain (legacy single-
      action view). For single-action templates, this is the whole
      learned action.
    - action_template_chain: the full ordered chain of action
      templates (1+). Multi-action templates store all steps here so
      ApprenticeProposer can emit each step as a separate candidate.
    - signature: the verb sequence (e.g. ('AddImport',) or
      ('Research', 'AddImport', 'AddStatement')).
    - sample_count: how many traces contributed to the inference.
    """

    task_regex: "re.Pattern[str]"  # forward-ref-free at runtime
    action_template: dict
    signature: tuple[str, ...]
    sample_count: int
    action_template_chain: tuple[dict, ...] = ()


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
    - Action sequences differ in length / verb order across samples.
    - No string field in any action dict tracks any task capture.

    Handles single-action AND multi-action sequences:
    - Single-action: action_template is the substituted action dict;
      action_template_chain is just [action_template].
    - Multi-action (e.g. AddImport → AddStatement): every step is
      substituted; action_template_chain holds the whole sequence in
      order. ApprenticeProposer emits each step as a separate
      candidate and the agent loop's pre-flight picks whichever isn't
      yet a no-op against the overlay.
    """
    if len(traces) < 2:
        return None

    tokens_per_sample = [_tokenize(t.get("task", "")) for t in traces]
    if any(not toks for toks in tokens_per_sample):
        return None
    token_count = len(tokens_per_sample[0])
    if any(len(toks) != token_count for toks in tokens_per_sample):
        return None

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
        return None
    task_regex = re.compile(r"^" + r"\s+".join(pattern_parts) + r"$")

    captures_per_sample: list[list[str]] = [
        [toks[pos] for pos in capture_positions]
        for toks in tokens_per_sample
    ]

    # Require all traces to have the same action-sequence shape.
    action_lists = [tr.get("actions") or [] for tr in traces]
    seq_len = len(action_lists[0])
    if seq_len < 1:
        return None
    if any(len(al) != seq_len for al in action_lists):
        return None
    if any(not all(isinstance(a, dict) for a in al) for al in action_lists):
        return None
    verbs_per_step = [
        [al[i].get("verb") for al in action_lists] for i in range(seq_len)
    ]
    if any(any(v != vs[0] for v in vs) for vs in verbs_per_step):
        return None
    signature = tuple(vs[0] or "?" for vs in verbs_per_step)

    # Per-step substitution: walk every step's string fields, find the
    # capture column whose values match across samples.
    chain: list[dict] = []
    any_substitution = False
    for step_i in range(seq_len):
        base_step = json.loads(json.dumps(action_lists[0][step_i]))
        for path, _value in list(_walk_string_fields(base_step)):
            values_at_path: list[str] = []
            for tr_actions in action_lists:
                cur = tr_actions[step_i]
                for step in path:
                    cur = cur[step]
                values_at_path.append(cur)
            if all(v == values_at_path[0] for v in values_at_path):
                continue
            for cap_i in range(len(capture_positions)):
                if [c[cap_i] for c in captures_per_sample] == values_at_path:
                    _set_at_path(base_step, path, f"__LATTICE_CAPTURE_{cap_i}__")
                    any_substitution = True
                    break
        chain.append(base_step)

    if not any_substitution:
        return None

    return LearnedTemplate(
        task_regex=task_regex,
        action_template=chain[0],  # legacy single-action view
        action_template_chain=tuple(chain),
        signature=signature,
        sample_count=len(traces),
    )


def apply_template(template: LearnedTemplate, task: str) -> dict | None:
    """If *task* matches the template, return the substituted FIRST action.

    For multi-action templates, this returns step 0 only — convenience
    for single-action callers. Use apply_template_chain to get the
    whole chain.
    """
    chain = apply_template_chain(template, task)
    return chain[0] if chain else None


def apply_template_chain(template: LearnedTemplate, task: str) -> list[dict]:
    """If *task* matches the template, return every step's substituted action.

    Returns [] on no match. For single-action templates, returns a
    one-element list; for multi-action, the full chain in order.
    """
    match = template.task_regex.match(task)
    if not match:
        return []
    chain_in = template.action_template_chain or (template.action_template,)
    out_chain: list[dict] = []
    captures = match.groupdict()
    for step_template in chain_in:
        out = json.loads(json.dumps(step_template))
        for cap_key, cap_value in captures.items():
            if cap_value is None:
                continue
            for path, value in list(_walk_string_fields(out)):
                # placeholder format: __LATTICE_CAPTURE_<n>__
                if value == f"__LATTICE_CAPTURE_{cap_key[1:]}__":
                    _set_at_path(out, path, cap_value)
        out_chain.append(out)
    return out_chain


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
