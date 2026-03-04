"""CLI entry points for hooks, migration, and health checks.

Claude Code hooks cannot make MCP calls -- they run shell commands that
read JSON from stdin and write plain text (or JSON) to stdout.  This
module provides the fast, synchronous-friendly CLI that hooks invoke.

Usage::

    # Hook commands (read JSON from stdin):
    python -m memories hook session-start
    python -m memories hook prompt-submit
    python -m memories hook post-tool
    python -m memories hook stop

    # Utility commands:
    python -m memories migrate --source ~/.claude-mem/claude-mem.db
    python -m memories health
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GEPA-optimized prompt loading (with fallback to hardcoded defaults)
# ---------------------------------------------------------------------------

_gepa_prompts: dict | None = None
_gepa_prompts_loaded_at: float = 0.0
_GEPA_PROMPTS_TTL: float = 300.0  # Re-read every 5 minutes


def _load_gepa_prompts() -> dict:
    """Load GEPA-optimized prompts from ~/.memories/gepa_prompts.json.

    Returns a dict with keys like 'extraction', 'classification'.
    Falls back to empty dict if file is missing or invalid.
    Cached with 5-minute TTL for cross-process updates.
    """
    global _gepa_prompts, _gepa_prompts_loaded_at
    import time
    now = time.monotonic()
    if _gepa_prompts is not None and (now - _gepa_prompts_loaded_at) < _GEPA_PROMPTS_TTL:
        return _gepa_prompts

    try:
        import json
        path = Path.home() / ".memories" / "gepa_prompts.json"
        if path.exists():
            data = json.loads(path.read_text())
            _gepa_prompts = data
            _gepa_prompts_loaded_at = now
        else:
            _gepa_prompts = {}
            _gepa_prompts_loaded_at = now
    except Exception:
        _gepa_prompts = _gepa_prompts or {}

    return _gepa_prompts


def _escape_xml(text: str) -> str:
    """Escape XML special characters in atom content to prevent tag injection."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Tools whose output is too noisy / low-value to auto-capture.
_SKIP_TOOLS: frozenset[str] = frozenset({
    "Read", "Glob", "Grep", "LS",
    "Edit", "Write", "MultiEdit", "NotebookEdit",  # implementation artifacts, not knowledge
    "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
    "Skill", "AskUserQuestion", "TodoWrite",
    "WebSearch", "WebFetch",
})

# Smaller skip set for PostToolUseFailure — failures from Edit/Write/Read etc. are
# tool-constraint violations worth learning from (e.g. "File has not been read yet").
_SKIP_TOOLS_ON_FAILURE: frozenset[str] = frozenset({
    "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
    "Skill", "AskUserQuestion", "TodoWrite",
})

# Tool-constraint error phrases that indicate a workflow mistake (not a runtime error).
# When detected, the content is framed prescriptively so it auto-classifies as antipattern.
_TOOL_CONSTRAINT_PHRASES: tuple[str, ...] = (
    "has not been read yet",       # Edit without prior Read
    "old_string is not unique",    # ambiguous Edit target
    "old_string not found",        # stale Edit target
    "not found in file",           # variant of above
    "file already exists",         # Write collision
    "must read the file",          # generic constraint
)

# Unambiguous shell/runtime error signatures — not generic "error" word matches.
_BASH_REAL_ERROR_SIGS: tuple[str, ...] = (
    "traceback (most recent call last)",  # Python exception
    "command not found",                   # unknown binary
    "permission denied",                   # OS access error
    "no such file or directory",           # missing path
)

# ------------------------------------------------------------------
# Junk content filter (recording pipeline quality gate)
# ------------------------------------------------------------------

# Static junk patterns derived from 331 manually-deleted atoms.
# Each tuple: (substring to match in lowercased content, reason code).
_JUNK_PATTERNS: tuple[tuple[str, str], ...] = (
    # Tool narration / self-talk
    ("delegated to", "tool_narration"),
    ("let me now", "self_talk"),
    ("went idle", "team_noise"),
    ("i have verified", "self_talk"),
    ("i've verified", "self_talk"),
    ("plan is ready", "self_talk"),
    ("submit it for your approval", "self_talk"),
    ("let me check", "self_talk"),
    ("i will now", "self_talk"),
    ("i'll now", "self_talk"),
    ("let me start by reading", "self_talk"),
    ("let me assess", "self_talk"),
    ("let me analyze", "self_talk"),
    ("let me read", "self_talk"),
    ("now i have the full", "self_talk"),
    ("now i need to read", "self_talk"),
    ("now let me read", "self_talk"),
    ("now let me also", "self_talk"),
    ("the user wants me to", "self_talk"),
    ("good, now i have", "self_talk"),
    ("ready to commit whenever", "self_talk"),
    # Agent task instructions (sub-agent prompts stored as atoms)
    ("[explore]", "agent_task_prompt"),
    ("[general-purpose]", "agent_task_prompt"),
    ("[sonnet]", "agent_task_prompt"),
    ("[haiku]", "agent_task_prompt"),
    ("you are a computational neuroscience", "agent_task_prompt"),
    ("you are a senior performance", "agent_task_prompt"),
    ("you are a memory assistant", "agent_task_prompt"),
    # Meta-noise
    ("note: i was only able to extract", "meta_noise"),
    ("the atomic fact", "meta_noise"),
    ("atomic process", "meta_noise"),
    ("note that there is only", "meta_noise"),
    ("an atomic fact:", "meta_noise"),
    ("inner thought:", "meta_noise"),
    ("knowledge capture agent", "meta_noise"),
    ("send completion message", "meta_noise"),
    ("send knowledge capture", "meta_noise"),
    # Generic truisms
    ("a generic framework is necessary", "truism"),
    ("seamless integration of various", "truism"),
    ("without compromising the user", "truism"),
    # Edit descriptions
    ("has been edited", "edit_description"),
    ("the line of code was edited", "edit_description"),
    ("a single line of code", "edit_description"),
    # Stdout narration
    ("the stdout value is", "stdout_narration"),
    # Cross-session noise
    ("message received from", "cross_session_noise"),
    ("full coordination system", "cross_session_noise"),
    # Monitoring / progress-check commands
    ("wait 3 minutes then check", "monitoring_noise"),
    ("wait 5 minutes then check", "monitoring_noise"),
    ("minutes then check progress", "monitoring_noise"),
    # Messaging scripts
    ("agent-messaging/", "messaging_noise"),
)

# Generic-opener regex: "The/A/An/This system/process/..." under 120 chars.
import re as _re
_GENERIC_OPENER_RE = _re.compile(
    r'^(The|A|An|This) (system|process|function|method|user|file|command|output)\b'
)

# Bash command description pattern: "Description (`command args`)"
# Matches content that embeds a shell command in backtick-parentheses.
_BASH_CMD_DESCRIPTION_RE = _re.compile(
    r'\(\s*`[a-z/~$].*`\s*\)\s*$'
)

# Cache for dynamic (learned) rejection patterns with TTL.
_learned_patterns: list[tuple[str, str]] | None = None
_learned_patterns_loaded_at: float = 0.0
_LEARNED_PATTERNS_TTL: float = 300.0  # Re-read from DB every 5 minutes


def _load_learned_patterns() -> list[tuple[str, str]]:
    """Load dynamic rejection patterns from the rejection_patterns table.

    Returns a list of (pattern, reason) tuples, cached with a 5-minute TTL
    so that new patterns learned by consolidation (a separate process) are
    picked up without requiring a hook process restart.
    """
    global _learned_patterns, _learned_patterns_loaded_at
    import time
    now = time.monotonic()
    if _learned_patterns is not None and (now - _learned_patterns_loaded_at) < _LEARNED_PATTERNS_TTL:
        return _learned_patterns

    try:
        from memories.config import get_config
        import sqlite3
        db_path = get_config().db_path
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT pattern, reason FROM rejection_patterns"
            ).fetchall()
            _learned_patterns = [(row["pattern"], row["reason"]) for row in rows]
            _learned_patterns_loaded_at = now
        except sqlite3.OperationalError:
            _learned_patterns = []
            _learned_patterns_loaded_at = now
        finally:
            conn.close()
    except Exception:
        _learned_patterns = _learned_patterns or []

    return _learned_patterns


def _is_junk(content: str, atom_type: str = "experience") -> tuple[bool, str]:
    """Check if content is junk that should be rejected before storing.

    Returns (is_junk, reason) where reason is a short code like "tool_narration".
    """
    if not content:
        return True, "empty"

    lower = content.lower().strip()

    # Static pattern matching (before length checks so patterns get correct reason codes)
    for pattern, reason in _JUNK_PATTERNS:
        if pattern in lower:
            return True, reason

    # Short content by type (30 chars for experience — task completions are concise but valid)
    if len(lower) < 30 and atom_type == "experience":
        return True, "too_short"
    if len(lower) < 60 and atom_type in ("fact", "insight"):
        return True, "too_short"

    # Generic opener + short
    if _GENERIC_OPENER_RE.match(content) and len(content) < 120:
        return True, "generic_opener"

    # Bash command descriptions: "Check foo (`ssh ...`)"
    if _BASH_CMD_DESCRIPTION_RE.search(content):
        return True, "bash_command_description"

    # Dynamic (learned) patterns
    for pattern, reason in _load_learned_patterns():
        if pattern in lower:
            return True, reason

    return False, ""


async def _record_rejection(
    brain,
    content: str,
    reason: str,
    hook: str | None = None,
    source_project: str | None = None,
) -> None:
    """Record a rejected atom in the atom_rejections table."""
    try:
        storage = getattr(brain, "_storage", None)
    except Exception:
        storage = None
    if storage is None:
        log.debug("_record_rejection: storage not available")
        return
    try:
        await brain._storage.execute_write(
            "INSERT INTO atom_rejections (content, reason, hook, source_project) "
            "VALUES (?, ?, ?, ?)",
            (content[:500], reason, hook, source_project),
        )
    except Exception as exc:
        log.debug("Failed to record rejection: %s", exc)


def _format_atom_line(atom: dict[str, Any]) -> str:
    """Format a single atom for hook output injection.

    Uses confidence-based framing (EmotionPrompt research: stronger
    assertive language for high-confidence atoms shifts LLM attention
    toward trusted knowledge).  Antipatterns use consequence framing
    (BCSP research: behavioral consequence scenarios improve adherence).
    """
    content = atom.get("content", "")
    atom_type = atom.get("type", "unknown")
    confidence = atom.get("confidence", 1.0)
    atom_id = atom.get("id", "")

    # Confidence-based framing: high-confidence atoms get assertive
    # language, low-confidence atoms get hedging.
    safe_content = _escape_xml(content)
    if atom_type == "antipattern":
        severity = atom.get("severity", "medium")
        prefix = "KNOWN MISTAKE" if severity in ("high", "critical") else "warning"
        line = f"  [{prefix}] {safe_content}"
        if atom_id:
            line += f"  (id:{atom_id})"
        instead = atom.get("instead")
        if instead:
            line += f"\n    instead: {_escape_xml(instead)}"
        tags = atom.get("tags") or []
        for tag in tags:
            if tag.startswith("category:"):
                line += f"  [{tag[len('category:'):]}]"
                break
    elif confidence >= 0.8:
        line = f"  [{atom_type}] {safe_content}"
        if atom_id:
            line += f"  (id:{atom_id})"
    else:
        line = f"  [{atom_type}|unverified] {safe_content}"
        if atom_id:
            line += f"  (id:{atom_id})"

    return line


def _format_pathways(
    pathways: list[dict[str, Any]],
    result: dict[str, Any] | None = None,
) -> list[str]:
    """Format pathways between result atoms for context injection.

    W16a: renders atom content previews instead of raw IDs for readability.
    Falls back to raw IDs when atom content is unavailable.
    """
    if not pathways:
        return []

    # Build atom_id → content lookup from result atoms + antipatterns.
    atom_map: dict[int, str] = {}
    if result:
        for a in result.get("atoms", []):
            aid = a.get("id")
            if aid is not None:
                atom_map[aid] = a.get("content", "")
        for a in result.get("antipatterns", []):
            aid = a.get("id")
            if aid is not None:
                atom_map[aid] = a.get("content", "")

    def _preview(atom_id: int | str) -> str:
        """Return a truncated content preview or fall back to ID string."""
        content = atom_map.get(atom_id, "") if isinstance(atom_id, int) else ""
        if content:
            truncated = content[:60].rstrip()
            if len(content) > 60:
                truncated += "..."
            return f'"{_escape_xml(truncated)}"'
        return str(atom_id)

    lines = ["  connections:"]
    for pw in pathways[:5]:  # cap at 5 to stay compact
        rel = pw.get("relationship", "related-to")
        src = pw.get("source_id", "?")
        tgt = pw.get("target_id", "?")
        lines.append(f"    {_preview(src)} --[{rel}]--> {_preview(tgt)}")
    return lines


def _read_stdin_json() -> dict[str, Any]:
    """Read and parse JSON from stdin (non-blocking, UTF-8)."""
    raw: str = ""
    try:
        raw = sys.stdin.read()
        if not raw or not raw.strip():
            return {}
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # B4: raise to WARNING with an input preview so hooks proceeding with
        # empty data (and therefore no session_id) are visible in production logs.
        log.warning(
            "Failed to parse stdin JSON (%s). Input preview: %r. "
            "Hook will proceed with empty data.",
            exc,
            raw[:200] if raw else "<unread>",
        )
        return {}


def _project_name(cwd: str | None) -> str | None:
    """Extract a short project name from a working directory path."""
    if not cwd:
        return None
    return Path(cwd).name or None


def _hook_budget(
    hook_type: str = "prompt-submit",
    prompt_length: int = 0,
) -> int:
    """Calculate the hook token budget using ContextBudget priority tiers.

    W16a: dynamic tier selection based on prompt length.

    - prompt length > 500 chars  -> ``critical`` tier (min(8000, 15% remaining))
    - prompt length 100-500      -> ``background`` tier (min(2000, 5% remaining))
    - prompt length < 100        -> ``minimal`` tier (min(500, 2% remaining))
    - session-start always uses  -> ``background`` tier
    - pre-tool always uses       -> ``minimal`` tier

    Uses :data:`memories.context._PRIORITY_PARAMS` for the actual fraction
    and cap values so tier definitions stay in one place.
    """
    from memories.context import ContextBudget

    cb = ContextBudget()

    # Fixed tiers for non-prompt hooks.
    if hook_type == "session-start":
        tier = "background"
    elif hook_type == "pre-tool":
        tier = "minimal"
    else:
        # Dynamic tier based on prompt length.
        if prompt_length > 500:
            tier = "critical"
        elif prompt_length >= 100:
            tier = "background"
        else:
            tier = "minimal"

    return cb.budget_for_recall(estimated_used=0, priority=tier)


_brain_instance: "Brain | None" = None
_brain_lock: "asyncio.Lock | None" = None
_brain_init_error: "Exception | None" = None

# Per-prompt atom timestamp tracking for temporal Hebbian weighting.
# Maps session_id -> atom_id -> ISO timestamp of when the PROMPT that
# retrieved the atom was submitted.  This gives per-prompt granularity
# instead of a single datetime('now') at DB insertion time.
_prompt_atom_timestamps: dict[str, dict[int, str]] = {}

_MAX_PROMPT_TIMESTAMPS_PER_SESSION = 500
"""Cap per-session timestamp entries to prevent unbounded memory growth
in long-running sessions with many recalled atoms."""


# ---------------------------------------------------------------------------
# Confidence calibration from similar_count
# ---------------------------------------------------------------------------

_CONFIDENCE_BY_SIMILAR_COUNT: dict[int, float] = {
    0: 0.5,
    1: 0.6,
    2: 0.6,
    3: 0.7,
    4: 0.7,
    5: 0.7,
}

_CONFIDENCE_HIGH = 0.85


def _confidence_from_similar_count(similar_count: int) -> float:
    """Map a similar_count from assess_novelty to an initial confidence value.

    More existing similar atoms implies the topic is well-represented in the
    graph, so the new atom's confidence starts higher (it corroborates
    existing knowledge).
    """
    return _CONFIDENCE_BY_SIMILAR_COUNT.get(similar_count, _CONFIDENCE_HIGH)


async def _get_brain() -> "Brain":
    """Return the process-level Brain singleton, initialising it on first call.

    Uses double-checked locking with an asyncio.Lock so only one coroutine
    performs the expensive initialization even under concurrent hook invocations
    in the same event loop.

    B6: If a previous initialisation attempt raised, we fast-fail immediately
    instead of re-attempting — every hook call would otherwise retry and
    produce confusing repeated tracebacks.
    """
    global _brain_instance, _brain_lock, _brain_init_error
    # B6: fast-fail if a previous init attempt failed.
    if _brain_init_error is not None:
        raise RuntimeError(f"Brain init previously failed: {_brain_init_error}") from _brain_init_error
    # Lock must be created inside an async context (cannot be module-level
    # because the event loop may not exist at import time).
    if _brain_lock is None:
        _brain_lock = asyncio.Lock()
    if _brain_instance is not None:
        return _brain_instance
    async with _brain_lock:
        # Re-check inside lock: another coroutine may have initialized
        # while we were waiting.
        if _brain_init_error is not None:
            raise RuntimeError(f"Brain init previously failed: {_brain_init_error}") from _brain_init_error
        if _brain_instance is None:
            from memories.brain import Brain
            import atexit

            try:
                b = Brain()
                await b.initialize()
            except Exception as exc:
                _brain_init_error = exc
                log.error(
                    "Brain init failed — all hooks disabled: %s", exc, exc_info=True
                )
                raise

            # atexit: close storage only — Hebbian has already fired via
            # the stop hook.  Do NOT call brain.shutdown() here because that
            # would invoke end_session() and session_end_learning() a second
            # time (double-fire).
            # B1: use asyncio.run() — asyncio.get_event_loop() is deprecated
            # on Python 3.10+ and raises RuntimeError when no loop is running.
            def _atexit_close() -> None:
                try:
                    asyncio.run(_close_brain_storage(b))
                except Exception as exc:
                    sys.stderr.write(
                        f"[memories] atexit: failed to close storage: {exc}\n"
                    )

            atexit.register(_atexit_close)
            _brain_instance = b
    return _brain_instance  # type: ignore[return-value]


async def _close_brain_storage(brain: "Brain") -> None:
    """Close the underlying storage without running end_session.

    This is the atexit path — Hebbian learning has already fired via the
    stop hook so we only need to flush and close the DB connection.
    """
    try:
        if brain._storage is not None:
            await brain._storage.close()
    except Exception:
        pass


async def _reset_brain_singleton() -> None:
    """Reset the process-level singleton so tests can get a fresh Brain.

    FOR TESTING ONLY.  Never call this in production code.
    """
    global _brain_instance, _brain_lock, _brain_init_error
    if _brain_instance is not None:
        await _close_brain_storage(_brain_instance)
    _brain_instance = None
    _brain_lock = None
    _brain_init_error = None


async def _save_hook_atoms(
    brain,
    claude_session_id: str,
    result: dict[str, Any] | None = None,
    extra_atom_id: int | None = None,
) -> None:
    """Persist atom IDs seen in this hook invocation for cross-hook Hebbian learning.

    Writes to the ``hook_session_atoms`` table keyed by the Claude Code
    ``session_id``.  The stop hook later reads this table to run
    ``session_end_learning`` with the full cross-hook atom set.
    """
    if not claude_session_id:
        return
    ids: list[int] = []
    if result:
        ids += [a["id"] for a in result.get("atoms", []) if a.get("id") is not None]
        ids += [a["id"] for a in result.get("antipatterns", []) if a.get("id") is not None]
    if extra_atom_id is not None:
        ids.append(extra_atom_id)
    if not ids:
        return
    try:
        assert brain._storage is not None
        await brain._storage.execute_many(
            "INSERT OR IGNORE INTO hook_session_atoms "
            "(claude_session_id, atom_id, accessed_at) VALUES (?, ?, datetime('now'))",
            [(claude_session_id, aid) for aid in ids],
        )
    except Exception as exc:
        # B7: storage failure means Hebbian runs with an incomplete atom set —
        # this is a data-quality issue that must be visible in production logs.
        log.warning(
            "Failed to save hook atoms for session %s — Hebbian will be incomplete: %s",
            claude_session_id[:8],
            exc,
        )


async def _record_hook_stat(
    brain,
    hook_type: str,
    latency_ms: int,
    project: str | None = None,
    query: str | None = None,
    result: dict[str, Any] | None = None,
    novelty_result: str | None = None,
) -> None:
    """Persist a hook invocation stat to the hook_stats table.

    Silently swallows all exceptions so it never blocks or crashes the hook.
    """
    try:
        atoms = result.get("atoms", []) if result else []
        antipatterns = result.get("antipatterns", []) if result else []
        all_atoms = atoms + antipatterns

        atom_ids = [a.get("id") for a in all_atoms if a.get("id") is not None]
        scores = [a.get("score", 0.0) for a in all_atoms if "score" in a]
        avg_score = sum(scores) / len(scores) if scores else None
        max_score = max(scores) if scores else None

        atoms_returned = len(all_atoms)
        budget_used = result.get("budget_used", 0) if result else 0
        budget_total = result.get("budget_used", 0) + result.get("budget_remaining", 0) if result else 0
        compression_level = result.get("compression_level", 0) if result else 0
        seed_count = result.get("seed_count", 0) if result else 0
        total_activated = result.get("total_activated", 0) if result else 0

        assert brain._storage is not None
        await brain._storage.execute_write(
            """
            INSERT INTO hook_stats
                (hook_type, project, query, atoms_returned, atom_ids,
                 avg_score, max_score, budget_used, budget_total,
                 compression_level, seed_count, total_activated,
                 novelty_result, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hook_type,
                project,
                query[:500] if query else None,
                atoms_returned,
                json.dumps(atom_ids) if atom_ids else None,
                round(avg_score, 6) if avg_score is not None else None,
                round(max_score, 6) if max_score is not None else None,
                budget_used,
                budget_total,
                compression_level,
                seed_count,
                total_activated,
                novelty_result,
                latency_ms,
            ),
        )
    except Exception as exc:
        log.debug("Failed to record hook stat: %s", exc)


# ------------------------------------------------------------------
# Session-start helpers
# ------------------------------------------------------------------


async def _preseed_from_claude_md(
    brain,
    session_id: str,
    cwd: str,
) -> None:
    """W16a: Predictive pre-activation from CLAUDE.md.

    Reads the project's ``.claude/CLAUDE.md`` and uses its first 200 chars
    as a pseudo-query to recall atoms, seeding ``hook_session_atoms`` before
    the first user prompt arrives.
    """
    if not cwd or not session_id:
        return
    claude_md_path = Path(cwd) / ".claude" / "CLAUDE.md"
    try:
        import anyio

        exists = await anyio.to_thread.run_sync(claude_md_path.exists)
        if not exists:
            return
        text = await anyio.to_thread.run_sync(
            lambda: claude_md_path.read_text(encoding="utf-8")[:200]
        )
        if not text or len(text.strip()) < 10:
            return
        preseed_result = await brain.recall(
            query=text,
            budget_tokens=500,
            region=None,
        )
        preseed_ids: list[int] = []
        for a in preseed_result.get("atoms", []):
            aid = a.get("id")
            if aid is not None:
                preseed_ids.append(aid)
        for a in preseed_result.get("antipatterns", []):
            aid = a.get("id")
            if aid is not None:
                preseed_ids.append(aid)
        if preseed_ids:
            if brain._storage is None:
                raise RuntimeError("Brain storage not initialised")
            await brain._storage.execute_many(
                "INSERT OR IGNORE INTO hook_session_atoms "
                "(claude_session_id, atom_id, accessed_at) VALUES (?, ?, datetime('now'))",
                [(session_id, aid) for aid in preseed_ids],
            )
            log.debug(
                "session-start: pre-seeded %d atoms from CLAUDE.md",
                len(preseed_ids),
            )
    except Exception as exc:
        log.debug("session-start: CLAUDE.md pre-seed failed: %s", exc)


# ------------------------------------------------------------------
# Hook: session-start
# ------------------------------------------------------------------

async def _hook_session_start(data: dict[str, Any]) -> str:
    """Handle SessionStart hook.

    Initializes brain, starts session, recalls project-specific memories.
    Returns formatted memory context as plain text.
    """
    t0 = time.monotonic()
    session_id = data.get("session_id", "")
    cwd = data.get("cwd", "")
    project = _project_name(cwd)

    try:
        brain = await _get_brain()
        assert brain._storage is not None

        # Purge stale sessions older than 4 hours (from crashed/killed processes).
        # This prevents session lineage detection from linking to long-dead sessions
        # and keeps the active_sessions table from growing without bound.
        await brain._storage.execute_write(
            "DELETE FROM active_sessions WHERE started_at < datetime('now', '-4 hours')"
        )

        # Bridge the Claude Code session_id to the Brain's internal session
        # tracking so that session priming (recall) and contextual encoding
        # (remember) work correctly through the hook path.
        if session_id:
            brain._current_session_id = session_id

        # Register this session and detect sub-agent lineage.
        is_subagent = False
        if session_id:
            await brain._storage.execute_write(
                "INSERT OR REPLACE INTO active_sessions (session_id, project) VALUES (?, ?)",
                (session_id, project),
            )
            if project:
                rows = await brain._storage.execute(
                    """
                    SELECT session_id FROM active_sessions
                    WHERE project = ?
                      AND session_id != ?
                      AND started_at >= datetime('now', '-2 hours')
                    ORDER BY started_at DESC
                    LIMIT 1
                    """,
                    (project, session_id),
                )
                if rows:
                    parent_id = rows[0]["session_id"]
                    await brain._storage.execute_write(
                        "INSERT OR REPLACE INTO session_lineage "
                        "(child_session_id, parent_session_id) VALUES (?, ?)",
                        (session_id, parent_id),
                    )
                    is_subagent = True
                    log.debug(
                        "session-start: detected sub-agent; parent=%s", parent_id
                    )

        result = None

        # Recall project-specific memories if we have a project context.
        # Two parallel project-scoped recalls with different queries to
        # maximise coverage: one biased toward problems/antipatterns, one
        # broad recall using the project's own domain terms (top tags).
        # Plus a cross-project antipattern recall for high-value warnings.
        if project:
            budget = _hook_budget("session-start")
            ap_budget = min(budget // 3, 500)

            # Build domain-aware query from the project's most-used tags.
            # Tags are stored as JSON arrays in atoms.tags column.
            tag_rows = await brain._storage.execute(
                """
                SELECT j.value as tag, COUNT(*) as cnt
                FROM atoms a, json_each(a.tags) j
                WHERE a.region = ? AND a.is_deleted = 0
                  AND j.value != ''
                GROUP BY j.value
                ORDER BY cnt DESC
                LIMIT 8
                """,
                (f"project:{project}",),
            )
            top_tags = [r["tag"] for r in tag_rows] if tag_rows else []
            if top_tags:
                domain_query = f"{project} {' '.join(top_tags)}"
            else:
                domain_query = project

            bug_query = f"{project} architecture bugs antipatterns known issues"

            # Run four tasks concurrently:
            # 1. Domain-aware recall (project's actual vocabulary)
            # 2. Bug/antipattern-biased recall
            # 3. Cross-project antipattern recall
            # 4. CLAUDE.md pre-seed
            domain_result, bug_result, ap_result, _ = await asyncio.gather(
                brain.recall(
                    query=domain_query,
                    budget_tokens=budget,
                    region=f"project:{project}",
                ),
                brain.recall(
                    query=bug_query,
                    budget_tokens=budget // 2,
                    region=f"project:{project}",
                ),
                brain.recall(
                    query=bug_query,
                    budget_tokens=ap_budget,
                    region=None,
                    types=["antipattern"],
                ),
                _preseed_from_claude_md(brain, session_id, cwd),
            )

            # Merge: domain_result is primary, bug_result and ap_result add extras.
            result = domain_result
            seen_ids = {a.get("id") for a in result.get("atoms", [])} | {
                a.get("id") for a in result.get("antipatterns", [])
            }
            for extra in (bug_result, ap_result):
                if not extra:
                    continue
                for a in extra.get("atoms", []):
                    if a.get("id") not in seen_ids:
                        if a.get("type") == "antipattern":
                            result.setdefault("antipatterns", []).append(a)
                        else:
                            result.setdefault("atoms", []).append(a)
                        seen_ids.add(a.get("id"))
                for a in extra.get("antipatterns", []):
                    if a.get("id") not in seen_ids:
                        result.setdefault("antipatterns", []).append(a)
                        seen_ids.add(a.get("id"))
            query = domain_query  # for stat recording

            latency_ms = int((time.monotonic() - t0) * 1000)
            await _record_hook_stat(
                brain, "session-start", latency_ms,
                project=project, query=query, result=result,
            )
            await _save_hook_atoms(brain, session_id, result)

            atoms = result.get("atoms", [])
            antipatterns = result.get("antipatterns", [])
            if atoms or antipatterns:
                lines = ["<memories>"]
                lines.append(
                    f"[memories] {len(atoms) + len(antipatterns)} prior learnings for {project}"
                    " — apply these to avoid repeating past mistakes:"
                )
                for atom in atoms:
                    lines.append(_format_atom_line(atom))
                for ap in antipatterns:
                    lines.append(_format_atom_line(ap))
                lines.extend(_format_pathways(result.get("pathways", []), result))
                if is_subagent:
                    lines.append(
                        "[memories] You are a sub-agent with access to the memories MCP.\n"
                        "  RECALL: Call recall(query=\"specific topic\") before investigating "
                        "a topic — prior findings may already exist.\n"
                        "  REMEMBER: Call remember(content=\"...\", type=\"insight\") to store "
                        "key findings so the parent session benefits.\n"
                        "  QUALITY: Only store content with specific technical details — "
                        "file paths, function names, error messages, config values, or "
                        "concrete facts. Do NOT store: meta-commentary (\"I checked...\"), "
                        "tool narration (\"Let me now...\"), generic truisms, or "
                        "descriptions of what tools did."
                    )
                lines.append("</memories>")

                return "\n".join(lines)
            else:
                if is_subagent:
                    return (
                        "<memories>\n"
                        f"No prior learnings for {project} yet.\n"
                        "[memories] You are a sub-agent with access to the memories MCP.\n"
                        "  RECALL: Call recall(query=\"specific topic\") before investigating "
                        "a topic — prior findings may already exist.\n"
                        "  REMEMBER: Call remember(content=\"...\", type=\"insight\") to store "
                        "key findings so the parent session benefits.\n"
                        "  QUALITY: Only store content with specific technical details — "
                        "file paths, function names, error messages, config values, or "
                        "concrete facts. Do NOT store: meta-commentary, tool narration, "
                        "generic truisms, or descriptions of what tools did.\n"
                        "</memories>"
                    )
                return (
                    "<memories>\n"
                    "No prior learnings for this project yet.\n"
                    "</memories>"
                )
        else:
            latency_ms = int((time.monotonic() - t0) * 1000)
            await _record_hook_stat(
                brain, "session-start", latency_ms, project=project,
            )

    except (TimeoutError, ConnectionError) as exc:
        log.warning("session-start hook: transient error: %s", exc)
    except Exception as exc:
        log.error("session-start hook: unexpected error: %s", exc, exc_info=True)

    return "Success"


# ------------------------------------------------------------------
# Hook: prompt-submit
# ------------------------------------------------------------------

async def _hook_prompt_submit(data: dict[str, Any]) -> str:
    """Handle UserPromptSubmit hook.

    Uses the user's prompt as a recall query. Returns formatted atoms
    as plain text context, or "Success" if nothing relevant.
    """
    t0 = time.monotonic()
    prompt = data.get("prompt", "")
    if not prompt or len(prompt.strip()) < 5:
        return "Success"

    try:
        brain = await _get_brain()
        cwd = data.get("cwd", "")
        project = _project_name(cwd)
        session_id = data.get("session_id", "")

        budget = _hook_budget("prompt-submit", prompt_length=len(prompt))
        global_result = None
        ap_result = None

        # A2: Run project-scoped, general, and cross-project antipattern
        # recalls concurrently.  The general recall surfaces cross-project
        # knowledge from the 'general' region only.  The antipattern recall
        # searches ALL regions but only returns antipatterns — these are
        # high-value warnings that apply across projects (e.g., JWT mistakes
        # learned in one project should surface in another).
        if project:
            global_budget = max(budget // 3, 500)
            antipattern_budget = 300
            result, global_result, ap_result = await asyncio.gather(
                brain.recall(
                    query=prompt,
                    budget_tokens=budget,
                    region=f"project:{project}",
                ),
                brain.recall(
                    query=prompt,
                    budget_tokens=global_budget,
                    region="general",
                ),
                brain.recall(
                    query=prompt,
                    budget_tokens=antipattern_budget,
                    region=None,
                    types=["antipattern"],
                ),
            )
        else:
            result = await brain.recall(
                query=prompt,
                budget_tokens=budget,
                region=None,
            )

        atoms = result.get("atoms", [])
        antipatterns = result.get("antipatterns", [])

        # Merge global + cross-project antipattern results, deduplicating by ID.
        seen_ids = {a.get("id") for a in atoms} | {a.get("id") for a in antipatterns}
        for extra in (global_result, ap_result):
            if not extra:
                continue
            for ga in extra.get("atoms", []):
                if ga.get("id") not in seen_ids:
                    atoms.append(ga)
                    seen_ids.add(ga.get("id"))
            for gap in extra.get("antipatterns", []):
                if gap.get("id") not in seen_ids:
                    antipatterns.append(gap)
                    seen_ids.add(gap.get("id"))

        latency_ms = int((time.monotonic() - t0) * 1000)
        await _record_hook_stat(
            brain, "prompt-submit", latency_ms,
            project=project, query=prompt, result=result,
        )
        # M1: Save atoms from both results for Hebbian learning.
        await _save_hook_atoms(brain, session_id, result)
        if global_result:
            await _save_hook_atoms(brain, session_id, global_result)

        # W16b: Record per-prompt timestamps for temporal Hebbian weighting.
        # Each recalled atom gets a timestamp of when THIS prompt retrieved it,
        # rather than a single datetime('now') at DB insertion time.
        if session_id:
            from datetime import datetime as _dt
            now_iso = _dt.now().isoformat()
            if session_id not in _prompt_atom_timestamps:
                # M-1: Cap outer dict to prevent unbounded growth from
                # crashed sessions that never fire the stop hook.
                if len(_prompt_atom_timestamps) >= 20:
                    oldest = next(iter(_prompt_atom_timestamps))
                    _prompt_atom_timestamps.pop(oldest, None)
                _prompt_atom_timestamps[session_id] = {}
            all_recalled_atoms = result.get("atoms", []) + result.get("antipatterns", [])
            if global_result:
                all_recalled_atoms += global_result.get("atoms", []) + global_result.get("antipatterns", [])
            for a in all_recalled_atoms:
                aid = a.get("id")
                if aid is not None and aid not in _prompt_atom_timestamps[session_id]:
                    if len(_prompt_atom_timestamps[session_id]) < _MAX_PROMPT_TIMESTAMPS_PER_SESSION:
                        _prompt_atom_timestamps[session_id][aid] = now_iso

        if not atoms and not antipatterns:
            return (
                "<memories>\n"
                "No relevant prior learnings found for this prompt.\n"
                "</memories>"
            )

        # A4: Deduplicate — remove antipatterns that already appear in atoms.
        atom_ids = {a.get("id") for a in atoms}
        antipatterns = [ap for ap in antipatterns if ap.get("id") not in atom_ids]

        # W16a: Primacy-recency sandwich ordering.
        # FIRST: the single highest-severity antipattern (attention anchor).
        # MIDDLE: regular atoms sorted by score descending.
        # LAST: remaining antipatterns sorted by severity.
        _severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        top_antipattern = None
        remaining_antipatterns = list(antipatterns)
        if remaining_antipatterns:
            remaining_antipatterns.sort(
                key=lambda a: _severity_rank.get(a.get("severity", "medium"), 2)
            )
            top_antipattern = remaining_antipatterns.pop(0)

        total_count = len(atoms) + len(remaining_antipatterns) + (1 if top_antipattern else 0)

        lines = ["<memories>"]
        lines.append(
            f"[memories] {total_count} prior learnings"
            " — apply these to avoid repeating past mistakes:"
        )

        # FIRST: top antipattern (primacy position)
        if top_antipattern:
            lines.append(_format_atom_line(top_antipattern))

        # MIDDLE: regular atoms by score descending
        for atom in atoms:
            lines.append(_format_atom_line(atom))
        lines.extend(_format_pathways(result.get("pathways", []), result))

        # LAST: remaining antipatterns by severity (recency position)
        if remaining_antipatterns:
            lines.append(
                f"[memories] {len(remaining_antipatterns)} known pitfalls"
                " — ignoring these has caused failures before:"
            )
            for ap in remaining_antipatterns:
                lines.append(_format_atom_line(ap))

        lines.append("</memories>")
        return "\n".join(lines)

    except (TimeoutError, ConnectionError) as exc:
        log.warning("prompt-submit hook: transient error: %s", exc)
        return "Success"
    except Exception as exc:
        log.error("prompt-submit hook: unexpected error: %s", exc, exc_info=True)
        return "Success"


# ------------------------------------------------------------------
# Hook: post-tool
# ------------------------------------------------------------------

async def _hook_post_tool(data: dict[str, Any]) -> str:
    """Handle PostToolUse hook.

    For Bash commands that produce genuine runtime errors (Python traceback,
    command not found, permission denied, no such file): extract content
    summary, assess novelty, and store as atom if novel. All other tools
    (Edit/Write/etc.) are skipped via _SKIP_TOOLS. Returns JSON with
    hookSpecificOutput.
    """
    t0 = time.monotonic()
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    tool_response = data.get("tool_response", "")

    # Skip noisy tools.
    if tool_name in _SKIP_TOOLS:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse"}})

    # Extract meaningful content from the tool response.
    content_summary = _extract_tool_content(tool_name, tool_input, tool_response)

    if not content_summary or len(content_summary) < 40:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse"}})

    try:
        brain = await _get_brain()

        # Assess novelty before storing.
        from memories.learning import LearningEngine
        is_novel, similar_count = await brain._learning.assess_novelty(content_summary)

        novelty = "pass" if is_novel else "fail"

        new_atom_id: int | None = None
        if is_novel:
            # Infer atom type from tool context.
            atom_type = _infer_atom_type(tool_name, tool_input, tool_response)
            cwd = data.get("cwd", "")
            project = _project_name(cwd)

            # Quality gate: reject junk before storing.
            junk, junk_reason = _is_junk(content_summary, atom_type)
            if junk:
                await _record_rejection(brain, content_summary, junk_reason, "post-tool", project)
            else:
                # Calculate importance based on type and content
                importance = _calculate_importance(atom_type, content_summary, tool_name)

                # For antipatterns, provide a default severity from the error
                # signature so the atom is useful even when heuristic extraction
                # in extract_antipattern_fields() finds no explicit keywords.
                severity = _infer_error_severity(content_summary) if atom_type == "antipattern" else None

                confidence = _confidence_from_similar_count(similar_count)

                remember_result = await brain.remember(
                    content=content_summary[:_MAX_ATOM_CHARS],
                    type=atom_type,
                    source_project=project,
                    importance=importance,
                    severity=severity,
                    confidence=confidence,
                )
                new_atom_id = remember_result.get("atom_id")

        session_id = data.get("session_id", "")
        await _save_hook_atoms(brain, session_id, extra_atom_id=new_atom_id)

        latency_ms = int((time.monotonic() - t0) * 1000)
        cwd = data.get("cwd", "")
        await _record_hook_stat(
            brain, "post-tool", latency_ms,
            project=_project_name(cwd),
            novelty_result=novelty,
        )

    except (TimeoutError, ConnectionError) as exc:
        log.warning("post-tool hook: transient error: %s", exc)
    except Exception as exc:
        log.error("post-tool hook: unexpected error: %s", exc, exc_info=True)

    return json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse"}})


# ------------------------------------------------------------------
# Hook: post-response
# ------------------------------------------------------------------

async def _hook_post_response(data: dict[str, Any]) -> str:
    """Handle post-response hook.

    Extracts key learnings from assistant responses and stores them.
    Looks for: facts learned, skills demonstrated, decisions made,
    errors encountered, insights gained.
    """
    t0 = time.monotonic()
    response = data.get("response", "")

    if not response or len(response) < 100:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "PostResponse"}})

    try:
        brain = await _get_brain()

        # Extract learnable content from the response
        learnings = _extract_response_learnings(response)

        cwd = data.get("cwd", "")
        project = _project_name(cwd)
        for learning in learnings:
            content, atom_type = learning
            if content and len(content) > 40:
                # Quality gate: reject junk before storing.
                junk, junk_reason = _is_junk(content, atom_type)
                if junk:
                    await _record_rejection(brain, content, junk_reason, "post-response", project)
                    continue

                # Check novelty
                from memories.learning import LearningEngine
                is_novel, similar_count = await brain._learning.assess_novelty(content)

                if is_novel:
                    # Calculate importance based on type and content
                    importance = _calculate_importance(atom_type, content)
                    confidence = _confidence_from_similar_count(similar_count)

                    await brain.remember(
                        content=content[:_MAX_ATOM_CHARS],
                        type=atom_type,
                        source_project=project,
                        importance=importance,
                        confidence=confidence,
                    )

        latency_ms = int((time.monotonic() - t0) * 1000)
        cwd = data.get("cwd", "")
        await _record_hook_stat(
            brain, "post-response", latency_ms,
            project=_project_name(cwd),
        )

    except (TimeoutError, ConnectionError) as exc:
        log.warning("post-response hook: transient error: %s", exc)
    except Exception as exc:
        log.error("post-response hook: unexpected error: %s", exc, exc_info=True)

    return json.dumps({"hookSpecificOutput": {"hookEventName": "PostResponse"}})


def _strip_markdown_noise(text: str) -> str:
    """Remove markdown formatting that inflates sentence length.

    Strips fenced code blocks, table rows, HTML tags, header markers,
    and bullet/number prefixes so that downstream sentence splitting
    operates on plain prose.
    """
    import re

    # Remove fenced code blocks (``` … ```)
    text = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.DOTALL)

    # Remove inline code spans
    text = re.sub(r"`[^`]+`", "", text)

    # Remove markdown table rows (lines starting with |)
    text = re.sub(r"^\|.*$", "", text, flags=re.MULTILINE)

    # Remove table separator rows (|---|---|)
    text = re.sub(r"^[\s|:-]+$", "", text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove header markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bullet/number list prefixes
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Collapse multiple blank lines / whitespace
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


# Maximum characters per extracted atom — keeps atoms atomic (~100 tokens).
_MAX_ATOM_CHARS = 200


def _extract_response_learnings(response: str) -> list[tuple[str, str]]:
    """Extract learnable content from an assistant response.

    Strips markdown formatting first, then splits on real sentence
    boundaries (periods, newlines) and classifies each sentence.
    Atoms are capped at ~100 tokens to stay within the 50-200 token
    guideline documented in ``Brain.remember()``.

    Returns list of (content, atom_type) tuples.
    """
    learnings: list[tuple[str, str]] = []

    cleaned = _strip_markdown_noise(response)

    # Look for explicit statements of fact/learning
    fact_indicators = [
        "i found", "i discovered", "i learned", "turns out",
        "the solution is", "the answer is", "this works because",
        "the issue was", "the problem was", "successfully",
    ]

    skill_indicators = [
        "how to", "the way to", "to do this", "the steps are",
        "workflow:", "process:", "method:",
    ]

    error_indicators = [
        "error:", "failed:", "doesn't work", "can't", "cannot",
        "blocked", "issue:", "problem:",
    ]

    insight_indicators = [
        "this suggests", "this means", "the pattern", "the root cause",
        "this is because", "the key insight", "notably", "in other words",
        "the takeaway", "the implication", "this indicates", "this confirms",
    ]

    # Split on sentence-ending punctuation and newlines — not just '. '
    import re
    sentences = re.split(r'(?<=[.!?])\s+|\n', cleaned)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 60 or len(sentence) > 600:
            continue

        sentence_lower = sentence.lower()

        # Check for error/antipattern — only classify as antipattern if the
        # sentence contains genuine negative-assertion vocabulary (Wave 8-C).
        if any(ind in sentence_lower for ind in error_indicators):
            atom_type = "antipattern" if _content_looks_like_antipattern(sentence) else "experience"
            candidate = sentence[:_MAX_ATOM_CHARS]
            is_junk, _ = _is_junk(candidate, atom_type=atom_type)
            if not is_junk:
                learnings.append((candidate, atom_type))
            continue

        # Check for skill/how-to
        if any(ind in sentence_lower for ind in skill_indicators):
            candidate = sentence[:_MAX_ATOM_CHARS]
            is_junk, _ = _is_junk(candidate, atom_type="skill")
            if not is_junk:
                learnings.append((candidate, "skill"))
            continue

        # Check for insight/conclusion
        if any(ind in sentence_lower for ind in insight_indicators):
            candidate = sentence[:_MAX_ATOM_CHARS]
            is_junk, _ = _is_junk(candidate, atom_type="insight")
            if not is_junk:
                learnings.append((candidate, "insight"))
            continue

        # Check for fact/discovery
        if any(ind in sentence_lower for ind in fact_indicators):
            candidate = sentence[:_MAX_ATOM_CHARS]
            is_junk, _ = _is_junk(candidate, atom_type="fact")
            if not is_junk:
                learnings.append((candidate, "fact"))
            continue

    # Limit to avoid noise
    return learnings[:3]


def _extract_tool_content(
    tool_name: str,
    tool_input: dict | str,
    tool_response: str,
) -> str | None:
    """Extract a meaningful content summary from a tool's input/output."""
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            tool_input = {}

    if not isinstance(tool_input, dict):
        tool_input = {}

    response_str = str(tool_response) if tool_response else ""

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        lower = response_str.lower()
        # Only capture genuine runtime errors — not every output containing the
        # word "error" (which includes cat/grep/git output with that word).
        if response_str and any(sig in lower for sig in _BASH_REAL_ERROR_SIGS):
            return f"Command `{command[:100]}` produced error: {response_str[:500]}"
        return None

    return None


# ------------------------------------------------------------------
# Antipattern classification guard (Wave 8-C)
# ------------------------------------------------------------------

# Prescriptive/cautionary keywords — the atom must express guidance about
# what to AVOID, not merely describe an error that occurred.
_ANTIPATTERN_PRESCRIPTIVE = frozenset([
    "should not", "should never", "avoid", "never use", "never do",
    "bad practice", "pitfall", "mistake", "antipattern", "anti-pattern",
    "failure mode", "gotcha", "footgun", "trap", "beware",
    "don't use", "do not use", "don\u2019t use",  # both apostrophe types
    "don't", "don\u2019t",  # bare "don't" is prescriptive
    "must not", "instead of",
    "leads to", "causes", "results in", "will break",
    "creates a", "introduces",  # causal: "creates a bug", "introduces a race"
])

# Descriptive keywords that need a prescriptive companion to qualify.
# "error" alone is descriptive; "avoid this error" is prescriptive.
_ANTIPATTERN_DESCRIPTIVE = frozenset([
    "wrong", "incorrect", "dangerous", "risk", "mistake",
    "bug", "error-prone", "don't", "do not",
])


def _content_looks_like_antipattern(content: str) -> bool:
    """Return True only if *content* contains prescriptive/cautionary language.

    Used as a guard before classifying an atom as ``antipattern``.  Content
    that merely mentions "error" or "failed" (e.g. command output, status
    reports) should **not** be treated as an antipattern — only content that
    expresses a warning, a bad practice, or something to avoid.

    A prescriptive keyword alone qualifies.  A descriptive keyword only
    qualifies when paired with a prescriptive one (e.g. "this is wrong"
    alone is descriptive; "avoid this — it's wrong" is prescriptive).
    """
    lower = content.lower()
    has_prescriptive = any(kw in lower for kw in _ANTIPATTERN_PRESCRIPTIVE)
    if has_prescriptive:
        return True
    # Descriptive keywords only count if there's also a prescriptive signal.
    return False


def _infer_atom_type(
    tool_name: str,
    tool_input: dict | str,
    tool_response: str,
) -> str:
    """Infer an atom type from the tool context."""
    # Bash: error/traceback in response *may* signal an antipattern, but only
    # if the content also contains negative-assertion vocabulary.  Otherwise
    # it is just an experience (a command that errored out).
    response_str = str(tool_response).lower() if tool_response else ""
    if "error" in response_str or "traceback" in response_str:
        if _content_looks_like_antipattern(response_str):
            return "antipattern"
        return "experience"

    return "fact"


def _infer_error_severity(content: str) -> str:
    """Infer a default severity from error content when type is antipattern.

    Assigns severity based on the error signature so auto-captured tool failures
    get meaningful severity even when ``extract_antipattern_fields`` heuristics
    find no explicit keyword.  The brain's ``remember()`` only calls the heuristic
    extractor when severity is ``None``, so explicit values here take precedence.
    """
    lower = content.lower()
    # Permission/security errors are high severity.
    if any(sig in lower for sig in ("permission denied", "access denied", "unauthorized", "forbidden")):
        return "high"
    # Data loss risks are high severity.
    if any(sig in lower for sig in ("rm -rf", "drop table", "delete from", "truncate")):
        return "high"
    # Python tracebacks are medium severity — they indicate code bugs.
    if "traceback (most recent call last)" in lower:
        return "medium"
    # Missing dependencies or commands are medium.
    if any(sig in lower for sig in ("command not found", "module not found", "no such file")):
        return "medium"
    return "medium"


def _calculate_importance(
    atom_type: str,
    content: str,
    tool_name: str | None = None,
) -> float:
    """Calculate importance score (0-1) based on atom type and content.
    
    Higher importance for:
    - Antipatterns/errors (0.8): Critical to avoid repeating mistakes
    - Skills (0.75): Reusable knowledge worth surfacing
    - Insights (0.7): Strategic learnings
    - Decisions (0.65): Context for future decisions
    - Experiences (0.55): Situational learning
    - Facts (0.5): Default baseline
    
    Additional boosts for:
    - Critical/high severity keywords (+0.1)
    - Security-related content (+0.1)
    - API/credential related (+0.05)
    """
    base_importance = {
        "antipattern": 0.8,
        "skill": 0.75,
        "insight": 0.7,
        "decision": 0.65,
        "experience": 0.55,
        "preference": 0.55,
        "fact": 0.5,
    }.get(atom_type, 0.5)
    
    content_lower = content.lower() if content else ""
    
    # Boost for critical/high severity indicators
    if any(word in content_lower for word in ["critical", "important", "never", "always", "must", "security", "breaking"]):
        base_importance = min(1.0, base_importance + 0.1)
    
    # Boost for security/credential related
    if any(word in content_lower for word in ["api key", "credential", "password", "token", "secret", "auth"]):
        base_importance = min(1.0, base_importance + 0.05)
    
    return round(base_importance, 2)


# ------------------------------------------------------------------
# Transcript reader
# ------------------------------------------------------------------

_THINKING_CAP = 5         # max thinking blocks to extract per session
_MIN_BLOCK_CHARS = 50     # blocks shorter than this are noise
_THINKING_TRUNCATE = 200  # chars to keep per thinking block (atomic insight)
_TEXT_TRUNCATE = 200      # chars to keep from the final text block
_DEDUP_PREFIX_LEN = 80    # first-N-chars key for within-session dedup


def _derive_transcript_path(session_id: str, cwd: str) -> Path | None:
    """Return the JSONL transcript path for this session, or None if absent.

    Claude Code stores transcripts at::

        ~/.claude/projects/{cwd.replace('/', '-')}/{session_id}.jsonl
    """
    if not session_id or not cwd:
        return None
    slug = cwd.replace("/", "-")
    path = Path("~/.claude/projects").expanduser() / slug / f"{session_id}.jsonl"
    return path if path.exists() else None


def _parse_transcript_file(path: Path) -> list[tuple[str, str]]:
    """Parse a JSONL transcript file and return atomic insights.

    Returns a list of ``(block_type, text)`` tuples where *block_type* is
    ``"thinking"`` (Claude's reasoning) or ``"text"`` (the final reply).

    Rules applied:

    - Thinking blocks: up to :data:`_THINKING_CAP`, deduplicated by
      first-:data:`_DEDUP_PREFIX_LEN`-char prefix, each truncated to
      :data:`_THINKING_TRUNCATE` chars so each atom represents one
      discrete planning step.
    - Text blocks: only the *last* one in the transcript is kept — the
      final reply is the highest-signal summary of what was concluded.
    - Blocks shorter than :data:`_MIN_BLOCK_CHARS` are skipped as noise.
    """
    thinking_blocks: list[str] = []
    last_text_block: str | None = None
    seen_prefixes: set[str] = set()

    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                if entry.get("type") != "assistant":
                    continue

                message = entry.get("message", {})
                content_blocks = message.get("content", [])
                if not isinstance(content_blocks, list):
                    continue

                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")

                    if btype == "thinking":
                        text = block.get("thinking", "")
                        if not text or len(text) < _MIN_BLOCK_CHARS:
                            continue
                        prefix = text[:_DEDUP_PREFIX_LEN]
                        if prefix in seen_prefixes:
                            continue
                        seen_prefixes.add(prefix)
                        if len(thinking_blocks) < _THINKING_CAP:
                            thinking_blocks.append(text[:_THINKING_TRUNCATE])

                    elif btype == "text":
                        text = block.get("text", "")
                        if text and len(text) >= _MIN_BLOCK_CHARS:
                            last_text_block = text[:_TEXT_TRUNCATE]

    except OSError:
        return []

    result: list[tuple[str, str]] = [("thinking", t) for t in thinking_blocks]
    if last_text_block:
        result.append(("text", last_text_block))
    return result


def _read_transcript_insights(
    session_id: str,
    cwd: str,
) -> list[tuple[str, str]]:
    """Derive the transcript path for this session and parse it.

    Returns the same format as :func:`_parse_transcript_file`, or ``[]``
    when no transcript exists for the given session/cwd.
    """
    path = _derive_transcript_path(session_id, cwd)
    if path is None:
        return []
    return _parse_transcript_file(path)


async def _extract_atomic_facts(
    text: str,
    ollama_url: str,
    model: str,
) -> list[str]:
    """Use a local generative Ollama model to extract 2-5 atomic facts.

    Each returned string is a short, self-contained insight that can be
    independently embedded, linked via synapses, and recalled on its own.
    Storing multiple small atoms instead of one large blob lets the synapse
    graph form richer connections and recall precisely the right facts.

    Fallback: returns ``[text[:_THINKING_TRUNCATE]]`` when the model is
    unavailable, times out, or produces no parseable output.
    """
    import ollama as _ollama

    # Use GEPA-optimized prompt if available, otherwise hardcoded default.
    gepa = _load_gepa_prompts()
    gepa_extraction = gepa.get("extraction", {})
    if isinstance(gepa_extraction, dict) and gepa_extraction.get("prompt"):
        base_prompt = gepa_extraction["prompt"]
    else:
        base_prompt = (
            "Extract 2 to 5 key technical facts from the following reasoning. "
            "Each fact must be specific and actionable — include file names, "
            "function names, error messages, or concrete details. "
            "DO NOT include: generic observations, meta-commentary about the "
            "extraction process, obvious truisms, or descriptions of what tools did. "
            "Output each as a single sentence on its own line, starting with '- '. "
            "If the text contains no specific technical facts, output NOTHING."
        )
    prompt = f"{base_prompt}\n\n{text[:1500]}"
    try:
        client = _ollama.AsyncClient(host=ollama_url)
        response = await asyncio.wait_for(
            client.generate(model=model, prompt=prompt),
            timeout=15.0,
        )
        raw = (response.response or "").strip()
        facts: list[str] = []
        for line in raw.splitlines():
            fact = line.strip().lstrip("-").strip()
            if fact and len(fact) >= 50:
                facts.append(fact[:_THINKING_TRUNCATE])
        if facts:
            return facts[:5]
    except Exception as exc:
        log.debug("Fact extraction failed (%s), using truncated original", exc)
    return [text[:_THINKING_TRUNCATE]]


async def _store_transcript_insights(
    brain,
    session_id: str,
    cwd: str,
    project: str | None,
) -> list[int]:
    """Extract transcript insights and store novel ones as atoms.

    Returns the list of new atom IDs created (excludes deduped or
    non-novel blocks so the caller can add them to ``hook_session_atoms``).

    Each ``"thinking"`` block → one or more ``insight`` atoms (reasoning steps).
    The final ``"text"`` block → ``experience`` atom (conclusion).

    When ``MEMORIES_DISTILL_THINKING=true`` is set, each thinking block is
    decomposed into 2-5 discrete atomic facts by a local generative model
    (default: ``llama3.2:3b``).  Each fact is stored as a separate atom so
    the synapse graph can link them individually.
    """
    from memories.config import get_config as _get_config

    insights = _read_transcript_insights(session_id, cwd)
    if not insights:
        return []

    cfg = _get_config()
    atom_ids: list[int] = []
    for block_type, text in insights:
        try:
            # Expand thinking blocks into multiple atomic facts when enabled.
            if block_type == "thinking" and cfg.distill_thinking:
                texts: list[str] = await _extract_atomic_facts(
                    text, cfg.ollama_url, cfg.distill_model
                )
            else:
                texts = [text]

            atom_type = "insight" if block_type == "thinking" else "experience"
            for fact in texts:
                # Quality gate: reject junk before storing.
                junk, junk_reason = _is_junk(fact, atom_type)
                if junk:
                    await _record_rejection(brain, fact, junk_reason, "stop", project)
                    continue

                is_novel, _sc = await brain._learning.assess_novelty(fact)
                if not is_novel:
                    continue
                result = await brain.remember(
                    content=fact,
                    type=atom_type,
                    source_project=project,
                )
                if not result.get("deduplicated"):
                    aid = result.get("atom_id")
                    if aid is not None:
                        atom_ids.append(aid)
        except Exception as exc:
            log.debug("Failed to store transcript insight: %s", exc)

    return atom_ids


# ------------------------------------------------------------------
# Backfill command
# ------------------------------------------------------------------

def _slug_to_project(slug: str) -> str | None:
    """Best-effort project name from a Claude projects directory slug.

    The slug is ``cwd.replace('/', '-')``.  Since ``-`` can appear in both
    separators and directory names we can't reverse it perfectly, but the
    last ``-``-delimited token is almost always the project name.

    Examples::

        -Users-john-git-memories  →  memories
        -Users-john-work-api      →  api
    """
    parts = slug.strip("-").split("-")
    return parts[-1] if parts and parts[-1] else None


async def _backfill_transcripts(verbose: bool = False) -> str:
    """Scan all Claude Code transcripts and store novel insights as atoms.

    Iterates every ``~/.claude/projects/{slug}/*.jsonl`` file and runs the
    same parsing + novelty-gate + distillation pipeline used by the stop
    hook.  Already-stored atoms are silently skipped by the semantic dedup
    gate so the command is safe to run multiple times.

    Returns a human-readable summary string.
    """
    from memories.config import get_config as _get_config

    projects_dir = Path("~/.claude/projects").expanduser()
    if not projects_dir.exists():
        return "No Claude projects directory found at ~/.claude/projects"

    cfg = _get_config()
    brain = await _get_brain()

    total_files = 0
    total_insights = 0
    total_atoms = 0
    lines: list[str] = []

    try:
        project_dirs = sorted(d for d in projects_dir.iterdir() if d.is_dir())
        for project_dir in project_dirs:
            slug = project_dir.name
            project = _slug_to_project(slug)
            proj_atoms = 0

            jsonl_files = sorted(project_dir.glob("*.jsonl"))
            for jsonl_file in jsonl_files:
                total_files += 1
                insights = _parse_transcript_file(jsonl_file)
                if not insights:
                    continue

                total_insights += len(insights)

                for block_type, text in insights:
                    try:
                        if block_type == "thinking" and cfg.distill_thinking:
                            facts: list[str] = await _extract_atomic_facts(
                                text, cfg.ollama_url, cfg.distill_model
                            )
                        else:
                            facts = [text]

                        atom_type = "insight" if block_type == "thinking" else "experience"
                        for fact in facts:
                            is_novel, _sc = await brain._learning.assess_novelty(fact)
                            if not is_novel:
                                continue
                            result = await brain.remember(
                                content=fact,
                                type=atom_type,
                                source_project=project,
                            )
                            if not result.get("deduplicated"):
                                aid = result.get("atom_id")
                                if aid is not None:
                                    proj_atoms += 1
                                    total_atoms += 1
                    except Exception as exc:
                        log.debug("backfill: failed to store insight: %s", exc)

            if verbose and proj_atoms:
                lines.append(f"  {slug}: +{proj_atoms} atoms")

        # Auto-relink so new atoms form synapses with existing graph.
        if total_atoms > 0:
            if verbose:
                print("\nRe-linking graph...", flush=True)
            n_atoms, n_synapses = await _relink_brain(brain, verbose=verbose)
            lines.append(f"  graph relinked: {n_atoms} atoms, {n_synapses} synapses")

    finally:
        await brain.shutdown()

    summary = (
        f"Backfill complete: {total_files} transcripts scanned, "
        f"{total_insights} insights found, {total_atoms} atoms stored."
    )
    if lines:
        summary = summary + "\n" + "\n".join(lines)
    return summary


def run_backfill(args: list[str]) -> None:
    """Run the transcript backfill command."""
    verbose = "--verbose" in args or "-v" in args
    result = asyncio.run(_backfill_transcripts(verbose=verbose))
    print(result)


async def _relink_brain(brain, verbose: bool = False) -> tuple[int, int]:
    """Re-run auto_link for every atom using an already-open Brain.

    Returns ``(total_atoms, total_synapses)``.  Safe to call multiple times —
    existing synapses are strengthened, not duplicated.
    """
    rows = await brain._storage.execute(
        "SELECT id FROM atoms WHERE is_deleted = 0 ORDER BY id"
    )
    atom_ids = [r["id"] for r in rows]
    total_synapses = 0

    # M-5: Parallelize auto_link calls with a concurrency semaphore.
    sem = asyncio.Semaphore(4)
    results: list[int] = []

    async def _link_one(atom_id: int) -> int:
        async with sem:
            try:
                synapses = await brain._learning.auto_link(atom_id)
                return len(synapses)
            except Exception as exc:
                log.debug("relink: auto_link failed for atom %d: %s", atom_id, exc)
                return 0

    # Process in chunks so we can report progress.
    chunk_size = 50
    for start in range(0, len(atom_ids), chunk_size):
        chunk = atom_ids[start : start + chunk_size]
        chunk_results = await asyncio.gather(
            *[_link_one(aid) for aid in chunk], return_exceptions=True
        )
        for r in chunk_results:
            total_synapses += r if isinstance(r, int) else 0
        if verbose:
            done = min(start + chunk_size, len(atom_ids))
            print(
                f"  {done}/{len(atom_ids)} atoms re-linked, "
                f"{total_synapses} synapses so far...",
                flush=True,
            )

    return len(atom_ids), total_synapses


async def _relink_all_atoms(verbose: bool = False) -> str:
    """Re-run auto_link for every atom in the graph to fill missing synapses.

    Safe to run multiple times — existing synapses are strengthened, not
    duplicated.  Useful after a ``backfill`` or ``reatomise`` run.

    Returns a human-readable summary string.
    """
    brain = await _get_brain()
    try:
        total_atoms, total_synapses = await _relink_brain(brain, verbose=verbose)
    finally:
        await brain.shutdown()

    return (
        f"Relink complete: {total_atoms} atoms processed, "
        f"{total_synapses} new/strengthened synapses created."
    )


def run_relink(args: list[str]) -> None:
    """Run the relink command."""
    verbose = "--verbose" in args or "-v" in args
    result = asyncio.run(_relink_all_atoms(verbose=verbose))
    print(result)


# Region name aliases → canonical names.
# Covers slug variations produced by different cwd depths.
# Populate with your own project aliases after installation, e.g.:
#   "project:my-app":  "project:myapp",
_REGION_ALIASES: dict[str, str] = {
    # Add your own project aliases here, e.g.:
    # "project:my-app": "project:myapp",
}


async def _normalise_regions(verbose: bool = False) -> str:
    """Rename fragmented/alias region names to canonical values.

    Updates every atom row then rebuilds the ``regions`` table counts so
    retrieval diversity-cap and region filtering stay accurate.

    Returns a human-readable summary string.
    """
    brain = await _get_brain()
    total_moved = 0

    try:
        for alias, canonical in _REGION_ALIASES.items():
            rows = await brain._storage.execute(
                "SELECT COUNT(*) AS cnt FROM atoms WHERE region = ? AND is_deleted = 0",
                (alias,),
            )
            cnt = rows[0]["cnt"] if rows else 0
            if cnt == 0:
                continue

            # B2: UPDATE must use execute_write to respect the write lock.
            await brain._storage.execute_write(
                "UPDATE atoms SET region = ? WHERE region = ?",
                (canonical, alias),
            )
            total_moved += cnt
            if verbose:
                print(f"  {alias!r} → {canonical!r}: {cnt} atoms", flush=True)

        # Rebuild the regions table: upsert live counts then prune empty rows.
        await brain._storage.execute_write(
            """
            INSERT OR REPLACE INTO regions (name, atom_count)
            SELECT region, COUNT(*)
            FROM atoms
            WHERE is_deleted = 0
            GROUP BY region
            """,
        )
        await brain._storage.execute_write(
            """
            DELETE FROM regions
            WHERE name NOT IN (
                SELECT DISTINCT region FROM atoms WHERE is_deleted = 0
            )
            """,
        )

    finally:
        await brain.shutdown()

    return f"Normalise complete: {total_moved} atoms renamed across {len(_REGION_ALIASES)} alias mappings."


def run_normalise(args: list[str]) -> None:
    """Run the region normalisation command."""
    verbose = "--verbose" in args or "-v" in args
    result = asyncio.run(_normalise_regions(verbose=verbose))
    print(result)


async def _reatomise_all_atoms(verbose: bool = False) -> str:
    """Split large blob atoms into 2-5 discrete atomic facts using a local LLM.

    For each atom with content longer than 150 characters, the Ollama model
    extracts the individual facts.  If two or more distinct facts are found,
    each is stored as a new atom (same region/type) and the original blob is
    soft-deleted.  Atoms that are already concise, or where distillation yields
    only one fact, are left untouched.

    Automatically re-links the whole graph when finished.

    Returns a human-readable summary string.
    """
    from memories.config import get_config as _get_config

    cfg = _get_config()
    brain = await _get_brain()

    total_checked = 0
    total_split = 0
    total_new = 0

    try:
        rows = await brain._storage.execute(
            """
            SELECT id, content, type, region, source_project
            FROM atoms
            WHERE is_deleted = 0
            ORDER BY id
            """
        )

        for row in rows:
            content: str = row["content"] or ""
            if len(content) < 150:
                continue  # Already short enough — skip.

            total_checked += 1

            try:
                facts = await _extract_atomic_facts(
                    content, cfg.ollama_url, cfg.distill_model
                )
            except Exception as exc:
                log.debug("reatomise: extraction failed for atom %d: %s", row["id"], exc)
                continue

            if len(facts) <= 1:
                continue  # Distillation didn't find multiple concepts.

            # Store each fact as a new atom, preserving region and type.
            new_ids: list[int] = []
            for fact in facts:
                try:
                    result = await brain.remember(
                        content=fact,
                        type=row["type"],
                        region=row["region"],
                        source_project=row["source_project"],
                    )
                    if not result.get("deduplicated"):
                        aid = result.get("atom_id")
                        if aid is not None:
                            new_ids.append(aid)
                except Exception as exc:
                    log.debug("reatomise: failed to store fact for atom %d: %s", row["id"], exc)

            if len(new_ids) >= 2:
                await brain._atoms.soft_delete(row["id"])
                total_split += 1
                total_new += len(new_ids)
                if verbose:
                    print(
                        f"  atom #{row['id']} → {len(new_ids)} facts "
                        f"({content[:60]!r}...)",
                        flush=True,
                    )

        # Auto-relink to wire up all the newly created atoms.
        if verbose:
            print("\nRe-linking graph...", flush=True)
        n_atoms, n_synapses = await _relink_brain(brain, verbose=verbose)

    finally:
        await brain.shutdown()

    return (
        f"Reatomise complete: {total_checked} blobs checked, "
        f"{total_split} split into {total_new} atomic facts. "
        f"Graph relinked: {n_atoms} atoms, {n_synapses} synapses."
    )


def run_reatomise(args: list[str]) -> None:
    """Run the reatomise command."""
    verbose = "--verbose" in args or "-v" in args
    result = asyncio.run(_reatomise_all_atoms(verbose=verbose))
    print(result)


# ------------------------------------------------------------------
# Shared stop logic (used by stop and subagent-stop)
# ------------------------------------------------------------------

async def _run_session_stop_logic(
    brain,
    session_id: str,
    cwd: str = "",
    project: str | None = None,
) -> None:
    """Apply Hebbian learning, propagate atoms to parent, and clean up.

    Shared between the ``stop`` and ``subagent-stop`` hooks so both
    run identical end-of-session bookkeeping.

    Steps:

    1. Read the JSONL transcript and store novel thinking/text blocks as
       atoms, adding their IDs to ``hook_session_atoms`` so they
       participate in this session's Hebbian run.
    2. Run Hebbian co-activation learning over all accumulated atom IDs.
    3. Propagate atom IDs to the parent session (sub-agent case).
    4. Clean up ``hook_session_atoms``, ``active_sessions``, and
       ``session_lineage``.
    """
    assert brain._storage is not None

    # 1. Store transcript insights before Hebbian so they co-activate
    #    with every other atom touched this session.
    if cwd and session_id:
        try:
            transcript_ids = await _store_transcript_insights(
                brain, session_id, cwd, project
            )
            if transcript_ids:
                await brain._storage.execute_many(
                    "INSERT OR IGNORE INTO hook_session_atoms "
                    "(claude_session_id, atom_id) VALUES (?, ?)",
                    [(session_id, aid) for aid in transcript_ids],
                )
                log.debug(
                    "session stop: added %d transcript atoms (session %.8s)",
                    len(transcript_ids),
                    session_id,
                )
        except Exception as exc:
            log.debug("session stop: transcript reader error: %s", exc)

    rows = await brain._storage.execute(
        "SELECT atom_id, accessed_at FROM hook_session_atoms WHERE claude_session_id = ?",
        (session_id,),
    )
    accumulated_ids = [row["atom_id"] for row in rows]

    # Build atom_id -> Unix timestamp map for temporal Hebbian weighting.
    # W16b: Prefer per-prompt timestamps (recorded by _hook_prompt_submit)
    # over the DB accessed_at (which only records when the hook saved the
    # atom, not when the user prompt actually retrieved it).
    from datetime import datetime, timezone as _tz
    atom_timestamps: dict[int, float] = {}

    prompt_ts = _prompt_atom_timestamps.get(session_id, {})
    for row in rows:
        atom_id = row["atom_id"]
        # Prefer per-prompt timestamp if available.
        ts_str = prompt_ts.get(atom_id) or row["accessed_at"]
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                atom_timestamps[atom_id] = dt.timestamp()
            except (ValueError, TypeError):
                pass

    # Clean up the per-prompt timestamps for this session.
    _prompt_atom_timestamps.pop(session_id, None)

    # B5: Wrap Hebbian + parent propagation in try/except so that the
    # cleanup DELETEs always execute even when session_end_learning raises.
    # Stale rows in hook_session_atoms / active_sessions / session_lineage
    # would cause double-counting in the next session.
    try:
        if accumulated_ids:
            assert brain._learning is not None
            await brain._learning.session_end_learning(
                session_id,
                accumulated_ids,
                atom_timestamps=atom_timestamps or None,
            )
            log.debug(
                "session stop: Hebbian from %d atoms (session %.8s)",
                len(accumulated_ids),
                session_id,
            )
        else:
            log.debug("session stop: no accumulated atoms for %.8s", session_id)

        # Propagate to parent session if this is a sub-agent.
        lineage_rows = await brain._storage.execute(
            "SELECT parent_session_id FROM session_lineage WHERE child_session_id = ?",
            (session_id,),
        )
        if lineage_rows and accumulated_ids:
            parent_id = lineage_rows[0]["parent_session_id"]
            await brain._storage.execute_many(
                "INSERT OR IGNORE INTO hook_session_atoms "
                "(claude_session_id, atom_id) VALUES (?, ?)",
                [(parent_id, aid) for aid in accumulated_ids],
            )
            log.debug(
                "session stop: propagated %d atoms to parent %.8s",
                len(accumulated_ids),
                parent_id,
            )
    except Exception as exc:
        log.error(
            "session stop: Hebbian learning failed for session %.8s — "
            "cleanup will still run: %s",
            session_id,
            exc,
            exc_info=True,
        )
    finally:
        # Clean up accumulated rows and lineage entries — always runs.
        await brain._storage.execute_write(
            "DELETE FROM hook_session_atoms WHERE claude_session_id = ?",
            (session_id,),
        )
        await brain._storage.execute_write(
            "DELETE FROM active_sessions WHERE session_id = ?",
            (session_id,),
        )
        await brain._storage.execute_write(
            "DELETE FROM session_lineage WHERE child_session_id = ?",
            (session_id,),
        )


# ------------------------------------------------------------------
# Hook: stop
# ------------------------------------------------------------------

async def _hook_stop(data: dict[str, Any]) -> str:
    """Handle Stop hook.

    Apply Hebbian co-activation learning across the full session using
    atom IDs accumulated in ``hook_session_atoms`` by earlier hooks.
    """
    t0 = time.monotonic()
    try:
        brain = await _get_brain()
        session_id = data.get("session_id", "")
        cwd = data.get("cwd", "")
        project = _project_name(cwd)

        if session_id:
            await _run_session_stop_logic(
                brain, session_id, cwd=cwd, project=project
            )
        else:
            # Fallback: no session_id, end Brain's internal session.
            await brain.end_session()

        latency_ms = int((time.monotonic() - t0) * 1000)
        await _record_hook_stat(brain, "stop", latency_ms, project=project)
        # Do NOT call brain.shutdown() here — the singleton stays alive for
        # subsequent hook invocations.  The atexit handler registered in
        # _get_brain() closes storage when the process exits.
    except (TimeoutError, ConnectionError) as exc:
        log.warning("stop hook: transient error: %s", exc)
    except Exception as exc:
        log.error("stop hook: unexpected error: %s", exc, exc_info=True)

    return ""


# ------------------------------------------------------------------
# Hook: subagent-stop
# ------------------------------------------------------------------

async def _hook_subagent_stop(data: dict[str, Any]) -> str:
    """Handle SubagentStop hook.

    Fires when a Task sub-agent session ends.  Runs identical end-of-session
    bookkeeping as the stop hook, including propagation to the parent session
    via the lineage table written at session-start.
    """
    t0 = time.monotonic()
    try:
        brain = await _get_brain()
        session_id = data.get("session_id", "")
        cwd = data.get("cwd", "")
        project = _project_name(cwd)

        if session_id:
            await _run_session_stop_logic(
                brain, session_id, cwd=cwd, project=project
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        await _record_hook_stat(brain, "subagent-stop", latency_ms, project=project)
        # Do NOT call brain.shutdown() — singleton stays alive for subsequent hooks.
    except (TimeoutError, ConnectionError) as exc:
        log.warning("subagent-stop hook: transient error: %s", exc)
    except Exception as exc:
        log.error("subagent-stop hook: unexpected error: %s", exc, exc_info=True)

    return ""


# ------------------------------------------------------------------
# Hook: pre-tool
# ------------------------------------------------------------------

# Tools whose pre-execution input carries meaningful intent signal.
_PRE_TOOL_CAPTURE: frozenset[str] = frozenset({"Task", "Bash"})


def _extract_pre_tool_content(
    tool_name: str,
    tool_input: dict | str,
) -> str | None:
    """Extract a meaningful content summary from a tool's input before execution.

    Focuses on signals that reveal Claude's *intent* rather than the result:

    - ``Task``: the full sub-agent prompt shows what Claude decided to delegate.
    - ``Bash``: the human-readable description explains the planned action.
    """
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            tool_input = {}
    if not isinstance(tool_input, dict):
        tool_input = {}

    if tool_name == "Task":
        prompt = tool_input.get("prompt", "")
        if not prompt:
            return None
        description = tool_input.get("description", "")
        subagent_type = tool_input.get("subagent_type", "")
        label = f"[{subagent_type}] " if subagent_type else ""
        header = f"{description}: " if description else ""
        return f"{label}{header}{prompt[:600]}"

    if tool_name == "Bash":
        description = tool_input.get("description", "")
        command = tool_input.get("command", "")
        # Only capture when Claude has provided a meaningful description.
        if description and len(description) > 30:
            return f"{description} (`{command[:100]}`)"

    return None


# ---------------------------------------------------------------------------
# Activity detection for pre-tool hook
# ---------------------------------------------------------------------------

# Activity name -> list of substrings to search in the Bash command.
# First match wins; check is case-insensitive on the lowercased command.
_ACTIVITY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("pr-creation", ("bb-pr.sh", "git push")),
    ("commit", ("git commit",)),
    ("deploy", ("deploy", "rsync")),
]


def _detect_activity(tool_name: str, tool_input: dict | str) -> str | None:
    """Detect a high-stakes activity from a Bash tool invocation.

    Returns one of ``"pr-creation"``, ``"commit"``, ``"deploy"``, or
    ``None`` when the tool is not Bash or no activity pattern matches.

    Only ``Bash`` tool calls are examined — other tools carry no command
    string worth pattern-matching.
    """
    if tool_name != "Bash":
        return None

    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(tool_input, dict):
        return None

    command: str = tool_input.get("command", "") or ""
    lower = command.lower()

    for activity, patterns in _ACTIVITY_PATTERNS:
        for pattern in patterns:
            if pattern in lower:
                return activity

    return None


def _format_rules_reminder(
    project: str,
    activity: str,
    rules: list[dict[str, Any]],
) -> list[str]:
    """Format recalled rule atoms as an urgent checklist block.

    Uses assertive, action-oriented language so the reminder is harder
    to ignore than the generic ``[memories]`` prefix used by regular atoms.
    """
    if not rules:
        return []

    lines = [
        f"<memories>",
        f"[rules] BEFORE {activity.upper().replace('-', ' ')}"
        f" in {project} — verify ALL of the following:",
    ]
    for rule in rules:
        content = _escape_xml(rule.get("content", ""))
        lines.append(f"  MUST: {content}")
    lines.append("</memories>")
    return lines


async def _hook_pre_tool_use(data: dict[str, Any]) -> str:
    """Handle PreToolUse hook.

    Captures Claude's intent before tool execution.  Only ``Task`` and
    ``Bash`` (with a descriptive description) are processed -- other tools
    are either noisy, low-signal, or better captured by PostToolUse.

    For Bash commands, also performs a fast antipattern-only recall to
    surface relevant warnings before execution.

    Activity-aware rule surfacing: when the command matches a high-stakes
    activity pattern (PR creation, commit, deploy), fires a targeted recall
    for project rules tagged with that activity.  Rules are surfaced as a
    checklist ahead of the standard antipattern warnings.
    """
    t0 = time.monotonic()
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})

    if tool_name not in _PRE_TOOL_CAPTURE:
        return ""

    content = _extract_pre_tool_content(tool_name, tool_input)
    if not content or len(content) < 20:
        return ""

    output_lines: list[str] = []

    try:
        brain = await _get_brain()

        # Activity-aware rule surfacing: fire a targeted recall when the
        # agent is about to perform a high-stakes activity.  This surfaces
        # project rules stored as skill atoms *before* execution so they
        # act as a checklist the agent must satisfy.
        if tool_name == "Bash":
            cwd = data.get("cwd", "")
            project = _project_name(cwd)
            activity = _detect_activity(tool_name, tool_input)

            if activity and project:
                rule_query = f"{project} {activity} rules checklist"
                rule_result = await brain.recall(
                    query=rule_query,
                    budget_tokens=800,
                    region=f"project:{project}",
                    types=["skill"],
                )
                # Collect skill atoms that carry the "rule" tag.
                raw_rules = rule_result.get("atoms", [])
                rules = [
                    a for a in raw_rules
                    if "rule" in (a.get("tags") or [])
                ]
                if rules:
                    output_lines.extend(
                        _format_rules_reminder(project, activity, rules)
                    )

        # A3: Pre-tool antipattern recall — surface relevant warnings
        # before Bash/Task execution.  Uses a small budget and restricts
        # to antipattern type for speed.
        if tool_name == "Bash":
            cwd = data.get("cwd", "")
            project = _project_name(cwd)
            warning_result = await brain.recall(
                query=content[:300],
                budget_tokens=500,
                region=f"project:{project}" if project else None,
                types=["antipattern"],
                include_antipatterns=True,
            )
            warnings = warning_result.get("atoms", []) + warning_result.get("antipatterns", [])
            if warnings:
                # Deduplicate by ID.
                seen: set[int] = set()
                unique_warnings: list[dict[str, Any]] = []
                for w in warnings:
                    wid = w.get("id")
                    if wid not in seen:
                        unique_warnings.append(w)
                        seen.add(wid)
                output_lines.append(
                    f"[memories] {len(unique_warnings)} known pitfalls"
                    " — ignoring these has caused failures before:"
                )
                for w in unique_warnings[:3]:  # cap at 3
                    output_lines.append(_format_atom_line(w))

        is_novel, similar_count = await brain._learning.assess_novelty(content)
        novelty = "pass" if is_novel else "fail"

        new_atom_id: int | None = None
        if is_novel:
            atom_type = "insight" if tool_name == "Task" else "experience"
            cwd = data.get("cwd", "")
            project = _project_name(cwd)

            # Quality gate: reject junk before storing.
            junk, junk_reason = _is_junk(content, atom_type)
            if junk:
                await _record_rejection(brain, content, junk_reason, "pre-tool", project)
            else:
                confidence = _confidence_from_similar_count(similar_count)
                remember_result = await brain.remember(
                    content=content[:_MAX_ATOM_CHARS],
                    type=atom_type,
                    source_project=project,
                    confidence=confidence,
                )
                new_atom_id = remember_result.get("atom_id")

        session_id = data.get("session_id", "")
        await _save_hook_atoms(brain, session_id, extra_atom_id=new_atom_id)

        latency_ms = int((time.monotonic() - t0) * 1000)
        await _record_hook_stat(
            brain, "pre-tool", latency_ms,
            project=_project_name(data.get("cwd", "")),
            novelty_result=novelty,
        )
    except (TimeoutError, ConnectionError) as exc:
        log.warning("pre-tool hook: transient error: %s", exc)
    except Exception as exc:
        log.error("pre-tool hook: unexpected error: %s", exc, exc_info=True)

    return "\n".join(output_lines) if output_lines else ""


# ------------------------------------------------------------------
# Hook: pre-compact
# ------------------------------------------------------------------

async def _hook_pre_compact(data: dict[str, Any]) -> str:
    """Handle PreCompact hook.

    Checkpoint Hebbian co-activation learning mid-session so atoms formed
    before context compaction are linked even if the stop hook fires late
    or not at all.  Rows are NOT deleted from hook_session_atoms -- the
    stop hook still needs them for its own Hebbian pass.
    """
    try:
        brain = await _get_brain()
        session_id = data.get("session_id", "")

        if session_id:
            assert brain._storage is not None
            rows = await brain._storage.execute(
                "SELECT atom_id, accessed_at FROM hook_session_atoms WHERE claude_session_id = ?",
                (session_id,),
            )
            accumulated_ids = [r["atom_id"] for r in rows]
            if accumulated_ids:
                from datetime import datetime, timezone as _tz
                atom_timestamps: dict[int, float] = {}
                for r in rows:
                    ts_str = r["accessed_at"]
                    if ts_str:
                        try:
                            dt = datetime.fromisoformat(ts_str)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=_tz.utc)
                            atom_timestamps[r["atom_id"]] = dt.timestamp()
                        except (ValueError, TypeError):
                            pass
                assert brain._learning is not None
                await brain._learning.session_end_learning(
                    session_id,
                    accumulated_ids,
                    atom_timestamps=atom_timestamps or None,
                )
                log.debug(
                    "pre-compact hook: Hebbian checkpoint from %d accumulated atoms",
                    len(accumulated_ids),
                )

        # Do NOT call brain.shutdown() — singleton stays alive for subsequent hooks.
    except (TimeoutError, ConnectionError) as exc:
        log.warning("pre-compact hook: transient error: %s", exc)
    except Exception as exc:
        log.error("pre-compact hook: unexpected error: %s", exc, exc_info=True)

    return ""


# ------------------------------------------------------------------
# Hook: post-tool-failure
# ------------------------------------------------------------------


async def _hook_post_tool_failure(data: dict[str, Any]) -> str:
    """Handle PostToolUseFailure hook.

    Tool failures are always worth capturing — the hook only fires when a tool
    errors, so no error-signature matching is needed.  Uses the smaller
    _SKIP_TOOLS_ON_FAILURE set so Edit/Write/Read constraint violations
    (e.g. "File has not been read yet") are captured as learning atoms.

    Tool-constraint errors are framed prescriptively ("Avoid …") so they
    auto-classify as antipatterns rather than plain experiences.
    """
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    error = data.get("error", "") or data.get("tool_response", "")

    if tool_name in _SKIP_TOOLS_ON_FAILURE or not error or len(str(error).strip()) < 20:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUseFailure"}})

    command = ""
    if isinstance(tool_input, dict):
        command = tool_input.get("command", tool_input.get("file_path", ""))
    elif isinstance(tool_input, str):
        try:
            command = json.loads(tool_input).get("command", "")
        except (json.JSONDecodeError, TypeError):
            pass

    error_str = str(error)
    is_constraint = any(phrase in error_str.lower() for phrase in _TOOL_CONSTRAINT_PHRASES)

    if is_constraint:
        # Prescriptive framing → triggers antipattern classification automatically.
        content = f"Avoid using `{tool_name}` without prerequisite — {error_str[:450]}"
    elif command:
        content = f"Command `{command[:100]}` failed: {error_str[:400]}"
    else:
        content = f"Tool `{tool_name}` failed: {error_str[:500]}"

    try:
        brain = await _get_brain()
        is_novel, similar_count = await brain._learning.assess_novelty(content)
        new_atom_id: int | None = None
        if is_novel:
            atom_type = (
                "antipattern"
                if _content_looks_like_antipattern(content)
                else "experience"
            )
            cwd = data.get("cwd", "")
            project = _project_name(cwd)
            # Quality gate: reject junk before storing.
            junk, junk_reason = _is_junk(content, atom_type)
            if junk:
                await _record_rejection(brain, content, junk_reason, "post-tool-failure", project)
            else:
                importance = _calculate_importance(atom_type, content, tool_name)
                severity = _infer_error_severity(content) if atom_type == "antipattern" else None
                confidence = _confidence_from_similar_count(similar_count)
                result = await brain.remember(
                    content=content[:_MAX_ATOM_CHARS],
                    type=atom_type,
                    source_project=project,
                    importance=importance,
                    severity=severity,
                    confidence=confidence,
                )
                new_atom_id = result.get("atom_id")

        session_id = data.get("session_id", "")
        await _save_hook_atoms(brain, session_id, extra_atom_id=new_atom_id)
        cwd = data.get("cwd", "")
        await _record_hook_stat(
            brain,
            "post-tool-failure",
            0,
            project=_project_name(cwd),
            novelty_result="pass" if is_novel else "fail",
        )
    except (TimeoutError, ConnectionError) as exc:
        log.warning("post-tool-failure hook: transient error: %s", exc)
    except Exception as exc:
        log.error("post-tool-failure hook: unexpected error: %s", exc, exc_info=True)

    return json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUseFailure"}})


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------

async def _health() -> str:
    """Run health check and return formatted status."""
    try:
        brain = await _get_brain()
        status = await brain.status()
        await brain.shutdown()

        lines = [
            "memories health check:",
            f"  atoms: {status['total_atoms']}",
            f"  synapses: {status['total_synapses']}",
            f"  regions: {len(status['regions'])}",
            f"  db_size: {status['db_size_mb']:.2f} MB",
            f"  ollama: {'healthy' if status['ollama_healthy'] else 'unhealthy'}",
            f"  model: {status['embedding_model']}",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"Health check failed: {exc}"


# ------------------------------------------------------------------
# Stats command
# ------------------------------------------------------------------

async def _stats() -> str:
    """Query hook_stats and format a human-readable dashboard."""
    try:
        brain = await _get_brain()
        stats = await brain.get_hook_stats()
        await brain.shutdown()

        lines = ["memories stats:", ""]

        # Hook counts by time period.
        lines.append("  Hooks fired:")
        for period_label, counts in [
            ("Last 7 days", stats.get("counts_7d", {})),
            ("Last 30 days", stats.get("counts_30d", {})),
            ("All time", stats.get("counts_all", {})),
        ]:
            parts = [f"{k}: {v}" for k, v in sorted(counts.items())]
            lines.append(f"    {period_label:14s} {', '.join(parts) or 'none'}")

        lines.append("")

        # Avg atoms returned per hook type.
        avg_atoms = stats.get("avg_atoms_returned", {})
        if avg_atoms:
            lines.append("  Avg atoms returned:")
            for hook_type, avg in sorted(avg_atoms.items()):
                lines.append(f"    {hook_type}: {avg:.1f}")
            lines.append("")

        # Avg relevance score.
        avg_rel = stats.get("avg_relevance_score")
        if avg_rel is not None:
            lines.append(f"  Avg relevance score: {avg_rel:.2f}")

        # Budget utilisation.
        budget_pct = stats.get("budget_utilisation_pct")
        if budget_pct is not None:
            lines.append(f"  Budget utilisation: {budget_pct:.0f}%")

        lines.append("")

        # Unique atoms surfaced.
        unique = stats.get("unique_atoms_surfaced", 0)
        total = stats.get("total_atoms", 0)
        impressions = stats.get("total_impressions", 0)
        pct = (unique / total * 100) if total > 0 else 0
        lines.append(f"  Unique atoms surfaced: {unique} / {total} ({pct:.1f}%)")
        lines.append(f"  Total impressions: {impressions:,}")
        lines.append("")

        # Top 10 most recalled atoms.
        top_atoms = stats.get("top_atoms", [])
        if top_atoms:
            lines.append("  Top 10 most-recalled atoms:")
            for entry in top_atoms[:10]:
                lines.append(
                    f"    #{entry['id']} ({entry['count']}x) "
                    f"[{entry['type']}] {entry['content'][:60]}..."
                )
            lines.append("")

        # Novelty pass rate.
        novelty = stats.get("novelty_stats", {})
        if novelty.get("total", 0) > 0:
            rate = novelty["pass"] / novelty["total"] * 100
            lines.append(
                f"  Novelty pass rate: {rate:.0f}% "
                f"({novelty['pass']}/{novelty['total']})"
            )
            lines.append("")

        # Latency by hook type.
        latency = stats.get("latency", {})
        if latency:
            lines.append("  Avg latency (ms):")
            for hook_type, lat in sorted(latency.items()):
                lines.append(
                    f"    {hook_type:16s} avg={lat['avg']:.0f}ms, "
                    f"max={lat['max']:.0f}ms"
                )
            lines.append("")

        # Weekly relevance trend.
        trend = stats.get("relevance_trend", [])
        if trend:
            lines.append("  Relevance trend (prompt-submit, last 8 weeks):")
            for row in trend:
                lines.append(
                    f"    {row['week']}  avg={row['relevance']:.2f}"
                    f"  invocations={row['invocations']}"
                    f"  atoms={row['avg_atoms']:.1f}"
                )
            lines.append("")

        # Feedback summary.
        feedback = stats.get("feedback", {})
        if feedback.get("total", 0) > 0:
            total_fb = feedback["total"]
            good = feedback.get("good", 0)
            bad = feedback.get("bad", 0)
            pct_good = good / total_fb * 100 if total_fb else 0
            lines.append(
                f"  Feedback: {total_fb} total  "
                f"good={good} ({pct_good:.0f}%)  bad={bad}"
            )

        return "\n".join(lines)
    except Exception as exc:
        return f"Stats failed: {exc}"


def run_stats() -> None:
    """Run stats command."""
    result = asyncio.run(_stats())
    print(result)


# ------------------------------------------------------------------
# Rejection stats command
# ------------------------------------------------------------------

async def _rejection_stats(days: int = 30) -> str:
    """Query atom_rejections and rejection_patterns tables for telemetry."""
    brain = await _get_brain()
    try:
        storage = brain._storage
        lines = ["memories rejection-stats:", ""]

        # --- Rejection counts by reason (last 7 / 30 days / all time) ---
        for label, where_clause in [
            ("Last 7 days", "WHERE created_at >= datetime('now', '-7 days')"),
            ("Last 30 days", "WHERE created_at >= datetime('now', '-30 days')"),
            ("All time", ""),
        ]:
            rows = await storage.execute(
                f"SELECT reason, COUNT(*) AS cnt FROM atom_rejections "
                f"{where_clause} GROUP BY reason ORDER BY cnt DESC",
            )
            if rows:
                parts = [f"{r['reason']}: {r['cnt']}" for r in rows]
                total = sum(r["cnt"] for r in rows)
                lines.append(f"  {label:14s}  {total} total — {', '.join(parts)}")
            else:
                lines.append(f"  {label:14s}  0 rejections")

        lines.append("")

        # --- Learned patterns and hit counts ---
        pattern_rows = await storage.execute(
            "SELECT pattern, reason, hit_count, source, last_hit_at "
            "FROM rejection_patterns ORDER BY hit_count DESC LIMIT 20",
        )
        if pattern_rows:
            lines.append("  Learned patterns (top 20 by hits):")
            for p in pattern_rows:
                last = p["last_hit_at"] or "never"
                lines.append(
                    f"    [{p['reason']}] {p['hit_count']:4d} hits  "
                    f"src={p['source']}  last={last}"
                )
                # Truncate long patterns for display.
                pat = p["pattern"]
                if len(pat) > 80:
                    pat = pat[:77] + "..."
                lines.append(f"      pattern: {pat}")
        else:
            lines.append("  Learned patterns: none")

        lines.append("")

        # --- Top rejected content samples per reason (last 30 days) ---
        # Single ROW_NUMBER() query instead of N per-reason queries.
        sample_rows = await storage.execute(
            """
            SELECT reason, content
            FROM (
                SELECT reason, content,
                       ROW_NUMBER() OVER (
                           PARTITION BY reason
                           ORDER BY created_at DESC
                       ) AS rn
                FROM atom_rejections
                WHERE created_at >= datetime('now', '-30 days')
            )
            WHERE rn <= 3
            ORDER BY reason
            """,
        )
        if sample_rows:
            lines.append("  Recent rejected samples (last 30 days):")
            current_reason = None
            for sr in sample_rows:
                if sr["reason"] != current_reason:
                    current_reason = sr["reason"]
                    lines.append(f"    [{current_reason}]:")
                content = sr["content"]
                if len(content) > 100:
                    content = content[:97] + "..."
                lines.append(f"      - {content}")

        lines.append("")

        # --- Daily rejection trend (last 14 days) ---
        trend_rows = await storage.execute(
            "SELECT DATE(created_at) AS day, COUNT(*) AS cnt "
            "FROM atom_rejections "
            "WHERE created_at >= datetime('now', '-14 days') "
            "GROUP BY DATE(created_at) ORDER BY day",
        )
        if trend_rows:
            lines.append("  Daily rejection trend (last 14 days):")
            for tr in trend_rows:
                bar = "#" * min(50, tr["cnt"])
                lines.append(f"    {tr['day']}  {tr['cnt']:4d}  {bar}")

        return "\n".join(lines)
    finally:
        await brain.shutdown()


def run_rejection_stats(args: list[str]) -> None:
    """Run rejection-stats command."""
    result = asyncio.run(_rejection_stats())
    print(result)


# ------------------------------------------------------------------
# Eval command
# ------------------------------------------------------------------

async def _eval_prompt(
    prompt: str,
    project: str | None = None,
    verbose: bool = False,
) -> str:
    """Show exactly what Claude would see for a given prompt.

    Runs recall against the brain and formats atoms with composite score,
    signal breakdown, antipattern warnings, and token budget summary.
    """
    brain = await _get_brain()
    try:
        result = await brain.recall(
            query=prompt,
            budget_tokens=4000,
            region=f"project:{project}" if project else None,
        )

        atoms = result.get("atoms", [])
        antipatterns = result.get("antipatterns", [])
        all_atoms = atoms + antipatterns
        budget_used = result.get("budget_used", 0)
        seed_count = result.get("seed_count", 0)
        total_activated = result.get("total_activated", 0)

        scores = [a.get("score", 0.0) for a in all_atoms if "score" in a]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        proj_label = project or "global"
        lines = [
            f'memories eval: "{prompt[:80]}"',
            f"  project: {proj_label}  |  {len(all_atoms)} atoms"
            f"  |  {budget_used:,} tokens  |  avg score: {avg_score:.2f}",
            f"  seeds: {seed_count}  |  total activated: {total_activated}",
            "",
        ]

        _BREAKDOWN_KEYS = (
            ("vector_similarity", "vector"),
            ("spread_activation", "activation"),
            ("recency", "recency"),
            ("bm25", "bm25"),
        )

        def _format_atom(atom: dict, prefix: str = "") -> None:
            atom_id = atom.get("id", "?")
            atom_type = atom.get("type", "?")
            score = atom.get("score", 0.0)
            content = atom.get("content", "")
            display = content if verbose else (
                content[:80] + ("..." if len(content) > 80 else "")
            )
            lines.append(f"  {prefix}[{atom_type}] #{atom_id}  {score:.2f}  {display}")
            breakdown = atom.get("score_breakdown")
            if breakdown:
                bd_parts = [
                    f"{short}={breakdown[key]:.2f}"
                    for key, short in _BREAKDOWN_KEYS
                    if key in breakdown
                ]
                lines.append(f"    ↳ {' '.join(bd_parts)}")

        for atom in atoms:
            _format_atom(atom)

        if antipatterns:
            lines.append("")
            for ap in antipatterns:
                _format_atom(ap, prefix="⚠ ")

        lines.append("")
        suffix = " (use --verbose for full atom content)" if not verbose else ""
        lines.append(f"  Budget: {budget_used:,} / 4,000 tokens used{suffix}")

    finally:
        await brain.shutdown()

    return "\n".join(lines)


def run_eval(args: list[str]) -> None:
    """Run eval command: show exactly what Claude sees for a prompt."""
    if not args or args[0].startswith("-"):
        print(
            'Usage: memories eval "<prompt>" [--project <name>] [--verbose]',
            file=sys.stderr,
        )
        sys.exit(1)

    prompt = args[0]
    project: str | None = None
    verbose = "--verbose" in args or "-v" in args

    for flag in ("--project", "-p"):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                project = args[idx + 1]
            break

    result = asyncio.run(_eval_prompt(prompt, project=project, verbose=verbose))
    print(result)


# ------------------------------------------------------------------
# Feedback command
# ------------------------------------------------------------------

async def _record_feedback(atom_id: int, signal: str) -> str:
    """Record good/bad feedback for a recalled atom, nudging its confidence."""
    brain = await _get_brain()
    try:
        assert brain._storage is not None

        rows = await brain._storage.execute(
            "SELECT id, confidence FROM atoms WHERE id = ? AND is_deleted = 0",
            (atom_id,),
        )
        if not rows:
            return f"Error: atom #{atom_id} not found"

        current_confidence = rows[0]["confidence"]

        await brain._storage.execute_write(
            "INSERT INTO atom_feedback (atom_id, signal) VALUES (?, ?)",
            (atom_id, signal),
        )

        if signal == "good":
            new_confidence = min(1.0, current_confidence + 0.05)
        else:
            new_confidence = max(0.0, current_confidence - 0.10)

        await brain._storage.execute_write(
            "UPDATE atoms SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
            (new_confidence, atom_id),
        )

        direction = "↑" if signal == "good" else "↓"
        return (
            f"Feedback recorded: atom #{atom_id} marked '{signal}'. "
            f"Confidence: {current_confidence:.2f} {direction} {new_confidence:.2f}"
        )
    finally:
        await brain.shutdown()


def run_feedback(args: list[str]) -> None:
    """Run feedback command: mark a recalled atom as good or bad."""
    if len(args) < 2:
        print("Usage: memories feedback <atom_id> good|bad", file=sys.stderr)
        sys.exit(1)

    try:
        atom_id = int(args[0])
    except ValueError:
        print(f"Error: atom_id must be an integer, got {args[0]!r}", file=sys.stderr)
        sys.exit(1)

    signal = args[1].lower()
    if signal not in ("good", "bad"):
        print(f"Error: signal must be 'good' or 'bad', got {signal!r}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(_record_feedback(atom_id, signal))
    print(result)


# ------------------------------------------------------------------
# Diagnose command
# ------------------------------------------------------------------

async def _diagnose() -> str:
    """Run comprehensive diagnostics on all system components."""
    import subprocess
    from memories.ollama_manager import OllamaManager

    lines = ["memories diagnostics", "=" * 40, ""]

    # Count running memories server processes
    result = subprocess.run(
        ["pgrep", "-f", "python.*-m memories$"],
        capture_output=True, text=True
    )
    pids = [p for p in result.stdout.strip().splitlines() if p]
    if pids:
        lines.append(f"  [ok] {len(pids)} memories server(s) running (PIDs: {', '.join(pids)})")
    else:
        lines.append("  [warn] No memories server currently running")
    lines.append("")

    # Check Ollama installation
    ollama = OllamaManager()
    if ollama.is_installed():
        lines.append("  [ok] Ollama installed")
    else:
        lines.append("  [error] Ollama not installed")
        lines.append(f"         {ollama._install_instructions()}")
        return "\n".join(lines)

    # Check Ollama daemon
    if await ollama.is_daemon_running():
        lines.append("  [ok] Ollama daemon running")
    else:
        lines.append("  [error] Ollama daemon not running")
        lines.append(f"         {ollama._daemon_instructions()}")
        lines.append("")
        return "\n".join(lines)

    # Check required model
    if await ollama.has_model(ollama.required_model):
        lines.append(f"  [ok] Model available: {ollama.required_model}")
    else:
        lines.append(f"  [error] Model missing: {ollama.required_model}")
        lines.append(f"         Run: ollama pull {ollama.required_model}")
        lines.append("")
        return "\n".join(lines)

    lines.append("")

    # Run health check
    try:
        brain = await _get_brain()
        status = await brain.status()
        await brain.shutdown()

        lines.append("Database:")
        lines.append(f"  Total atoms: {status['total_atoms']}")
        lines.append(f"  Total synapses: {status['total_synapses']}")
        lines.append(f"  Regions: {len(status['regions'])}")
        lines.append(f"  DB size: {status['db_size_mb']:.2f} MB")
        lines.append(f"  Ollama healthy: {status['ollama_healthy']}")
        lines.append("")

        # Check configuration files
        from pathlib import Path
        claude_json = Path("~/.claude.json").expanduser()
        claude_settings = Path("~/.claude/settings.json").expanduser()

        if claude_json.exists():
            lines.append("  [ok] ~/.claude.json exists")
        else:
            lines.append("  [warn] ~/.claude.json not found")

        if claude_settings.exists():
            lines.append("  [ok] ~/.claude/settings.json exists")
        else:
            lines.append("  [warn] ~/.claude/settings.json not found")

        lines.append("")
        lines.append("All systems operational!")

    except Exception as exc:
        lines.append(f"Health check failed: {exc}")

    return "\n".join(lines)


def run_diagnose() -> None:
    """Run diagnose command."""
    result = asyncio.run(_diagnose())
    print(result)


# ------------------------------------------------------------------
# Hook: task-completed
# ------------------------------------------------------------------


async def _hook_task_completed(data: dict[str, Any]) -> str:
    """Handle TaskCompleted hook.

    Fires when a task in the Claude Code task tracker is marked done.
    ``task_subject`` is a concise description of what was accomplished —
    a low-noise, high-signal completion milestone.  Stored as an experience
    atom so the system accumulates a record of what was built/fixed over time.
    """
    task_subject = data.get("task_subject", "")
    task_description = data.get("task_description", "")

    if not task_subject or len(task_subject.strip()) < 5:
        return ""

    team_name = data.get("team_name", "")
    content = f"Completed: {task_subject}"
    if task_description and len(task_description.strip()) > 20:
        content = f"Completed: {task_subject} — {task_description[:150]}"
    if team_name:
        content = f"[team:{team_name}] {content}"

    try:
        brain = await _get_brain()
        cwd = data.get("cwd", "")
        project = _project_name(cwd)
        session_id = data.get("session_id", "")

        is_novel, similar_count = await brain._learning.assess_novelty(content)
        new_atom_id: int | None = None
        if is_novel:
            # Quality gate: reject junk before storing.
            junk, junk_reason = _is_junk(content, "experience")
            if junk:
                await _record_rejection(brain, content, junk_reason, "task-completed", project)
            else:
                confidence = _confidence_from_similar_count(similar_count)
                result = await brain.remember(
                    content=content[:_MAX_ATOM_CHARS],
                    type="experience",
                    source_project=project,
                    importance=0.6,
                    confidence=confidence,
                )
                new_atom_id = result.get("atom_id")

        await _save_hook_atoms(brain, session_id, extra_atom_id=new_atom_id)
        await _record_hook_stat(
            brain, "task-completed", 0, project=project,
            novelty_result="pass" if is_novel else "fail",
        )
    except (TimeoutError, ConnectionError) as exc:
        log.warning("task-completed hook: transient error: %s", exc)
    except Exception as exc:
        log.error("task-completed hook: unexpected error: %s", exc, exc_info=True)

    return ""


# ------------------------------------------------------------------
# Hook: teammate-idle
# ------------------------------------------------------------------


async def _hook_teammate_idle(data: dict[str, Any]) -> str:
    """Handle TeammateIdle hook.

    Fires when a teammate in an agent team goes idle (completes a turn).
    Records a rejection (not a stored atom) for team-idle events since
    the 'went idle' content matches _JUNK_PATTERNS. If a meaningful
    summary is provided, it is preserved in the rejection record.
    """
    teammate_name = data.get("teammate_name", "")
    team_name = data.get("team_name", "")
    summary = data.get("summary", "")

    if not teammate_name:
        return ""

    content = f"Agent team teammate '{teammate_name}' went idle"
    if summary:
        content += f": {summary[:200]}"
    if team_name:
        content = f"[team:{team_name}] {content}"

    try:
        brain = await _get_brain()
        cwd = data.get("cwd", "")
        project = _project_name(cwd)
        session_id = data.get("session_id", "")

        is_novel, similar_count = await brain._learning.assess_novelty(content)
        if is_novel:
            # Quality gate: reject junk before storing.
            junk, junk_reason = _is_junk(content, "experience")
            if junk:
                await _record_rejection(brain, content, junk_reason, "teammate-idle", project)
            else:
                confidence = _confidence_from_similar_count(similar_count)
                await brain.remember(
                    content=content[:_MAX_ATOM_CHARS],
                    type="experience",
                    source_project=project,
                    importance=0.4,
                    confidence=confidence,
                )

        await _record_hook_stat(brain, "teammate-idle", 0, project=project)
    except (TimeoutError, ConnectionError) as exc:
        log.warning("teammate-idle hook: transient error: %s", exc)
    except Exception as exc:
        log.error("teammate-idle hook: unexpected error: %s", exc, exc_info=True)

    return ""


# ------------------------------------------------------------------
# Hook: notification (elicitation_dialog only)
# ------------------------------------------------------------------


async def _hook_notification(data: dict[str, Any]) -> str:
    """Handle Notification hook — elicitation_dialog type only.

    Fires when Claude Code surfaces a dialog requiring user input.
    ``elicitation_dialog`` means Claude hit an ambiguity or decision point
    and is asking for clarification.  The ``message`` field contains the
    question — capturing these reveals recurring sources of uncertainty
    in a project over time.

    All other notification types (auth_success, idle_prompt, permission_prompt)
    are ignored as low-signal noise.
    """
    notification_type = data.get("notification_type", "")
    if notification_type != "elicitation_dialog":
        return ""

    message = data.get("message", "")
    if not message or len(message.strip()) < 20:
        return ""

    content = f"Claude asked for clarification: {message[:150]}"

    try:
        brain = await _get_brain()
        cwd = data.get("cwd", "")
        project = _project_name(cwd)
        session_id = data.get("session_id", "")

        is_novel, similar_count = await brain._learning.assess_novelty(content)
        new_atom_id: int | None = None
        if is_novel:
            # Quality gate: reject junk before storing.
            junk, junk_reason = _is_junk(content, "insight")
            if junk:
                await _record_rejection(brain, content, junk_reason, "notification", project)
            else:
                confidence = _confidence_from_similar_count(similar_count)
                result = await brain.remember(
                    content=content[:_MAX_ATOM_CHARS],
                    type="insight",
                    source_project=project,
                    importance=0.55,
                    confidence=confidence,
                )
                new_atom_id = result.get("atom_id")

        await _save_hook_atoms(brain, session_id, extra_atom_id=new_atom_id)
        await _record_hook_stat(
            brain, "notification", 0, project=project,
            novelty_result="pass" if is_novel else "fail",
        )
    except (TimeoutError, ConnectionError) as exc:
        log.warning("notification hook: transient error: %s", exc)
    except Exception as exc:
        log.error("notification hook: unexpected error: %s", exc, exc_info=True)

    return ""


# ------------------------------------------------------------------
# Hook: session-end
# ------------------------------------------------------------------


async def _hook_session_end(data: dict[str, Any]) -> str:
    """Handle SessionEnd hook.

    Fires when the Claude Code session terminates (after Stop).  Acts as a
    safety net: if any session atoms remain (Stop hook missed or skipped),
    run a final Hebbian pass so no learning is lost.  Also records the
    termination reason as an observability stat.
    """
    try:
        brain = await _get_brain()
        session_id = data.get("session_id", "")
        cwd = data.get("cwd", "")
        project = _project_name(cwd)
        reason = data.get("reason", "other")

        if session_id:
            assert brain._storage is not None
            rows = await brain._storage.execute(
                "SELECT atom_id FROM hook_session_atoms WHERE claude_session_id = ?",
                (session_id,),
            )
            leftover_ids = [r["atom_id"] for r in rows]
            if leftover_ids:
                # Stop hook didn't clean up — run Hebbian as fallback.
                assert brain._learning is not None
                await brain._learning.session_end_learning(session_id, leftover_ids)
                await brain._storage.execute_write(
                    "DELETE FROM hook_session_atoms WHERE claude_session_id = ?",
                    (session_id,),
                )
                log.debug(
                    "session-end hook: fallback Hebbian from %d leftover atoms "
                    "(reason=%s, session=%.8s)",
                    len(leftover_ids),
                    reason,
                    session_id,
                )

        await _record_hook_stat(brain, "session-end", 0, project=project)
    except (TimeoutError, ConnectionError) as exc:
        log.warning("session-end hook: transient error: %s", exc)
    except Exception as exc:
        log.error("session-end hook: unexpected error: %s", exc, exc_info=True)

    return ""


# ------------------------------------------------------------------
# Hook: subagent-start
# ------------------------------------------------------------------


async def _hook_subagent_start(data: dict[str, Any]) -> str:
    """Handle SubagentStart hook.

    Fires when the parent session spawns a Task sub-agent.  Tracks the
    delegation event for hook stats but does NOT store an atom — "Delegated
    to X sub-agent" is noise that clutters the graph.  The sub-agent's own
    findings (stored via mcp__memories__remember) are the valuable signal.
    """
    agent_type = data.get("agent_type", "")

    # Skip unnamed/empty agent types.
    if not agent_type or agent_type in ("Bash",):
        return ""

    try:
        brain = await _get_brain()
        cwd = data.get("cwd", "")
        project = _project_name(cwd)

        await _record_hook_stat(
            brain, "subagent-start", 0, project=project,
            novelty_result=None,
        )
    except (TimeoutError, ConnectionError) as exc:
        log.warning("subagent-start hook: transient error: %s", exc)
    except Exception as exc:
        log.error("subagent-start hook: unexpected error: %s", exc, exc_info=True)

    return ""


# ------------------------------------------------------------------
# Hook: permission-request
# ------------------------------------------------------------------


async def _hook_permission_request(data: dict[str, Any]) -> str:
    """Handle PermissionRequest hook.

    Fires when Claude Code is about to show a permission dialog.  Captures
    what operation triggered the permission request — these are high-signal
    learning moments because repeated permission requests reveal patterns in
    how the agent operates in a project.

    Stored as antipattern when the operation contains dangerous vocabulary
    (sudo, rm -rf, etc.), otherwise as experience.
    """
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})

    if tool_name in _SKIP_TOOLS:
        return ""

    # Extract the command or file path from tool_input.
    command = ""
    if isinstance(tool_input, dict):
        command = tool_input.get("command", tool_input.get("file_path", ""))
    elif isinstance(tool_input, str):
        try:
            command = json.loads(tool_input).get("command", "")
        except (json.JSONDecodeError, TypeError):
            pass

    if not command or len(command.strip()) < 10:
        return ""

    content = f"Permission requested for `{tool_name}`: {command[:300]}"

    try:
        brain = await _get_brain()
        cwd = data.get("cwd", "")
        project = _project_name(cwd)
        session_id = data.get("session_id", "")

        is_novel, similar_count = await brain._learning.assess_novelty(content)
        new_atom_id: int | None = None
        if is_novel:
            atom_type = (
                "antipattern"
                if _content_looks_like_antipattern(content)
                else "experience"
            )
            # Quality gate: reject junk before storing.
            junk, junk_reason = _is_junk(content, atom_type)
            if junk:
                await _record_rejection(brain, content, junk_reason, "permission-request", project)
            else:
                importance = _calculate_importance(atom_type, content, tool_name)
                confidence = _confidence_from_similar_count(similar_count)
                result = await brain.remember(
                    content=content[:_MAX_ATOM_CHARS],
                    type=atom_type,
                    source_project=project,
                    importance=importance,
                    confidence=confidence,
                )
                new_atom_id = result.get("atom_id")

        await _save_hook_atoms(brain, session_id, extra_atom_id=new_atom_id)
        await _record_hook_stat(
            brain, "permission-request", 0, project=project,
            novelty_result="pass" if is_novel else "fail",
        )
    except (TimeoutError, ConnectionError) as exc:
        log.warning("permission-request hook: transient error: %s", exc)
    except Exception as exc:
        log.error("permission-request hook: unexpected error: %s", exc, exc_info=True)

    return ""


# ------------------------------------------------------------------
# Seed-rules: store project rules as skill atoms
# ------------------------------------------------------------------

# Rule sets keyed by (project, activity).  Add your own project rules here.
# Example:
#   "my-project": {
#       "commit": ["MUST use Conventional Commits format"],
#       "pr-creation": ["MUST run tests before creating a PR"],
#   },
_PROJECT_RULES: dict[str, dict[str, list[str]]] = {
    "example": {
        "pr-creation": [
            "MUST run tests and verify all tests pass before creating any pull request",
            "MUST review changes before creating a pull request",
        ],
        "commit": [
            "MUST use Conventional Commits format for all commit messages",
        ],
        "implementation": [
            "MUST handle errors gracefully with appropriate error messages",
        ],
    },
}


async def _seed_project_rules(project: str) -> dict[str, Any]:
    """Store foundational project rules as skill atoms in the memory graph.

    Rules are stored with:
    - ``type``: ``"skill"``
    - ``tags``: ``["rule", "<activity>", "<project>"]``
    - ``importance``: ``0.95`` (high — rules should not decay easily)
    - ``region``: ``"project:<project>"``

    Returns a summary dict with counts of atoms stored and skipped.

    This function is intentionally NOT called automatically by any hook.
    Invoke via CLI: ``uv run python -m memories seed-rules --project <name>``
    """
    brain = await _get_brain()

    rules_by_activity = _PROJECT_RULES.get(project)
    if not rules_by_activity:
        available = ", ".join(_PROJECT_RULES.keys()) or "<none>"
        return {
            "project": project,
            "stored": 0,
            "skipped": 0,
            "error": f"No rules defined for project '{project}'. Available: {available}",
        }

    stored = 0
    skipped = 0
    region = f"project:{project}"

    for activity, rule_texts in rules_by_activity.items():
        tags = ["rule", activity, project]
        for rule_text in rule_texts:
            # Novelty check — avoid duplicate seeding on repeated calls.
            is_novel, _ = await brain._learning.assess_novelty(rule_text)
            if not is_novel:
                skipped += 1
                continue

            await brain.remember(
                content=rule_text,
                type="skill",
                region=region,
                tags=tags,
                source_project=project,
                importance=0.95,
                confidence=1.0,
            )
            stored += 1
            log.debug("seed-rules: stored rule for %s/%s: %s", project, activity, rule_text[:60])

    return {
        "project": project,
        "stored": stored,
        "skipped": skipped,
    }


def run_seed_rules(args: list[str]) -> None:
    """Seed project rules as skill atoms in the memory graph.

    Usage::

        uv run python -m memories seed-rules --project myproject
        uv run python -m memories seed-rules --project myproject --list
    """
    # --list: show available projects without seeding.
    if "--list" in args:
        projects = list(_PROJECT_RULES.keys())
        if projects:
            print("Available projects:")
            for p in projects:
                activities = list(_PROJECT_RULES[p].keys())
                total_rules = sum(len(r) for r in _PROJECT_RULES[p].values())
                print(f"  {p}: {total_rules} rules across {activities}")
        else:
            print("No project rules defined yet.")
        return

    project: str | None = None
    for i, arg in enumerate(args):
        if arg == "--project" and i + 1 < len(args):
            project = args[i + 1]

    if not project:
        print(
            "Usage: python -m memories seed-rules --project <name>\n"
            "       python -m memories seed-rules --list",
            file=sys.stderr,
        )
        sys.exit(1)

    result = asyncio.run(_seed_project_rules(project))

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(
        f"seed-rules: project={result['project']} "
        f"stored={result['stored']} skipped={result['skipped']}"
    )


# ------------------------------------------------------------------
# CLI dispatch
# ------------------------------------------------------------------

def run_hook(subcommand: str) -> None:
    """Dispatch a hook subcommand. Reads JSON from stdin."""
    data = _read_stdin_json()

    handlers = {
        "session-start": _hook_session_start,
        "prompt-submit": _hook_prompt_submit,
        "pre-tool": _hook_pre_tool_use,
        "post-tool": _hook_post_tool,
        "post-response": _hook_post_response,
        "stop": _hook_stop,
        "subagent-stop": _hook_subagent_stop,
        "pre-compact": _hook_pre_compact,
        "post-tool-failure": _hook_post_tool_failure,
        "session-end": _hook_session_end,
        "subagent-start": _hook_subagent_start,
        "permission-request": _hook_permission_request,
        "task-completed": _hook_task_completed,
        "teammate-idle": _hook_teammate_idle,
        "notification": _hook_notification,
    }

    handler = handlers.get(subcommand)
    if handler is None:
        print(f"Unknown hook subcommand: {subcommand}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(handler(data))
    if result:
        print(result)


def run_health() -> None:
    """Run health check command."""
    result = asyncio.run(_health())
    print(result)


def run_migrate(args: list[str]) -> None:
    """Run migration from claude-mem. Delegates to migrate module."""
    from memories.migrate import run_migration
    run_migration(args)


def run_setup(args: list[str]) -> None:
    """Run setup or uninstall. Delegates to setup module."""
    from memories.setup import run_setup as _run_setup
    _run_setup(args)


def run_reflect(args: list[str]) -> None:
    """Run memory consolidation (decay, prune, merge, promote).

    Usage::

        python -m memories reflect
        python -m memories reflect --dry-run
        python -m memories reflect --scope project:myapp
    """
    dry_run = "--dry-run" in args
    scope = "all"
    for i, arg in enumerate(args):
        if arg == "--scope" and i + 1 < len(args):
            scope = args[i + 1]

    async def _reflect() -> dict[str, Any]:
        brain = await _get_brain()
        return await brain.reflect(scope=scope, dry_run=dry_run)

    result = asyncio.run(_reflect())
    print(json.dumps(result, indent=2))


def run_generate_skill(args: list[str]) -> None:
    """Generate a SKILL.md for a project from memories atoms.

    Usage::

        python -m memories generate-skill myproject
        python -m memories generate-skill myproject --output .claude/skills/myproject/SKILL.md
    """
    if not args or args[0].startswith("-"):
        print(
            "Usage: memories generate-skill <project> [--output <path>]",
            file=sys.stderr,
        )
        sys.exit(1)

    project = args[0]
    output_path: str | None = None
    for i, arg in enumerate(args):
        if arg == "--output" and i + 1 < len(args):
            output_path = args[i + 1]

    async def _gen() -> str:
        from memories.skill_gen import generate_skill
        return await generate_skill(project, output_path=output_path)

    result = asyncio.run(_gen())
    if not output_path:
        print(result)


def dispatch(args: list[str]) -> None:
    """Main CLI dispatcher.

    Parameters
    ----------
    args:
        Command-line arguments after ``python -m memories``.
        e.g. ``["hook", "session-start"]`` or ``["migrate", "--source", "..."]``
    """
    if not args:
        return  # Fall through to MCP server.

    command = args[0]

    if command == "hook":
        if len(args) < 2:
            print("Usage: python -m memories hook <subcommand>", file=sys.stderr)
            sys.exit(1)
        run_hook(args[1])
        sys.exit(0)

    elif command == "migrate":
        run_migrate(args[1:])
        sys.exit(0)

    elif command == "health":
        run_health()
        sys.exit(0)

    elif command == "setup":
        run_setup(args[1:])
        sys.exit(0)

    elif command == "stats":
        run_stats()
        sys.exit(0)

    elif command == "diagnose":
        run_diagnose()
        sys.exit(0)

    elif command == "backfill":
        run_backfill(args[1:])
        sys.exit(0)

    elif command == "relink":
        run_relink(args[1:])
        sys.exit(0)

    elif command == "normalise" or command == "normalize":
        run_normalise(args[1:])
        sys.exit(0)

    elif command == "reatomise" or command == "reatomize":
        run_reatomise(args[1:])
        sys.exit(0)

    elif command == "eval":
        run_eval(args[1:])
        sys.exit(0)

    elif command == "feedback":
        run_feedback(args[1:])
        sys.exit(0)

    elif command == "reflect":
        run_reflect(args[1:])
        sys.exit(0)

    elif command == "generate-skill":
        run_generate_skill(args[1:])
        sys.exit(0)

    elif command == "seed-rules":
        run_seed_rules(args[1:])
        sys.exit(0)

    elif command == "rejection-stats":
        run_rejection_stats(args[1:])
        sys.exit(0)

    # Unknown command -- don't exit, fall through to MCP server.
