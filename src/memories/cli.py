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

# Tools whose output is too noisy / low-value to auto-capture.
_SKIP_TOOLS: frozenset[str] = frozenset({
    "Read", "Glob", "Grep", "LS",
    "Edit", "Write", "MultiEdit", "NotebookEdit",  # implementation artifacts, not knowledge
    "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
    "Skill", "AskUserQuestion", "TodoWrite",
    "WebSearch", "WebFetch",
})

# Unambiguous shell/runtime error signatures — not generic "error" word matches.
_BASH_REAL_ERROR_SIGS: tuple[str, ...] = (
    "traceback (most recent call last)",  # Python exception
    "command not found",                   # unknown binary
    "permission denied",                   # OS access error
    "no such file or directory",           # missing path
)


def _format_atom_line(atom: dict[str, Any]) -> str:
    """Format a single atom for hook output injection.

    Includes type, confidence, atom ID, and content.  Antipatterns
    additionally show severity and "instead" fields when present.
    """
    content = atom.get("content", "")
    atom_type = atom.get("type", "unknown")
    confidence = atom.get("confidence", 1.0)
    atom_id = atom.get("id", "")

    line = f"  [{atom_type}|{confidence:.1f}] {content}"
    if atom_id:
        line += f"  (id:{atom_id})"

    # Structured antipattern metadata.
    if atom_type == "antipattern":
        severity = atom.get("severity")
        instead = atom.get("instead")
        if severity:
            line += f"\n    severity: {severity}"
        if instead:
            line += f"\n    instead: {instead}"

    return line


def _format_pathways(pathways: list[dict[str, Any]]) -> list[str]:
    """Format pathways between result atoms for context injection."""
    if not pathways:
        return []
    lines = ["  connections:"]
    for pw in pathways[:5]:  # cap at 5 to stay compact
        rel = pw.get("relationship", "related-to")
        src = pw.get("source_id", "?")
        tgt = pw.get("target_id", "?")
        strength = pw.get("strength", 0.0)
        lines.append(f"    {src} --[{rel}|{strength:.2f}]--> {tgt}")
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


def _hook_budget() -> int:
    """Calculate the hook token budget from config (default 5% of context)."""
    from memories.config import get_config
    cfg = get_config()
    return int(cfg.context_window_tokens * cfg.hook_budget_pct)


_brain_instance: "Brain | None" = None
_brain_lock: "asyncio.Lock | None" = None
_brain_init_error: "Exception | None" = None


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

        # Bridge the Claude Code session_id to the Brain's internal session
        # tracking so that session priming (recall) and contextual encoding
        # (remember) work correctly through the hook path.
        if session_id:
            brain._current_session_id = session_id

        # Register this session and detect sub-agent lineage.
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
                    log.debug(
                        "session-start: detected sub-agent; parent=%s", parent_id
                    )

        result = None

        # Recall project-specific memories if we have a project context.
        # Use the project name itself as the query (not a meta-description
        # like "project context for X") — this has better semantic overlap
        # with actual stored content that mentions the project.
        if project:
            query = project
            result = await brain.recall(
                query=query,
                budget_tokens=_hook_budget(),
                region=f"project:{project}",
            )

            latency_ms = int((time.monotonic() - t0) * 1000)
            await _record_hook_stat(
                brain, "session-start", latency_ms,
                project=project, query=query, result=result,
            )
            await _save_hook_atoms(brain, session_id, result)

            atoms = result.get("atoms", [])
            if atoms:
                lines = [
                    f"[memories] {len(atoms)} recalled memories for {project}"
                    " (verify before acting on stale information):"
                ]
                for atom in atoms:
                    lines.append(_format_atom_line(atom))
                lines.extend(_format_pathways(result.get("pathways", [])))

                return "\n".join(lines)
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

        budget = _hook_budget()

        # Primary recall: project-scoped for relevance.
        result = await brain.recall(
            query=prompt,
            budget_tokens=budget,
            region=f"project:{project}" if project else None,
        )

        atoms = result.get("atoms", [])
        antipatterns = result.get("antipatterns", [])

        # A2: Secondary global recall (no region) to surface cross-project
        # knowledge — general antipatterns, skills, and facts that are
        # relevant to the query but stored in a different project.
        if project:
            seen_ids = {a.get("id") for a in atoms} | {a.get("id") for a in antipatterns}
            global_budget = max(budget // 3, 500)
            global_result = await brain.recall(
                query=prompt,
                budget_tokens=global_budget,
                region=None,  # search all regions
            )
            for ga in global_result.get("atoms", []):
                if ga.get("id") not in seen_ids:
                    atoms.append(ga)
                    seen_ids.add(ga.get("id"))
            for gap in global_result.get("antipatterns", []):
                if gap.get("id") not in seen_ids:
                    antipatterns.append(gap)
                    seen_ids.add(gap.get("id"))

        latency_ms = int((time.monotonic() - t0) * 1000)
        await _record_hook_stat(
            brain, "prompt-submit", latency_ms,
            project=project, query=prompt, result=result,
        )
        await _save_hook_atoms(brain, session_id, result)

        if not atoms and not antipatterns:
            return "Success"

        # A4: Deduplicate — remove antipatterns that already appear in atoms.
        atom_ids = {a.get("id") for a in atoms}
        antipatterns = [ap for ap in antipatterns if ap.get("id") not in atom_ids]

        lines = []
        if atoms:
            lines.append(
                f"[memories] {len(atoms)} recalled memories"
                " (verify before acting on stale information):"
            )
            for atom in atoms:
                lines.append(_format_atom_line(atom))
            lines.extend(_format_pathways(result.get("pathways", [])))

        if antipatterns:
            lines.append(f"[memories] {len(antipatterns)} warnings:")
            for ap in antipatterns:
                lines.append(_format_atom_line(ap))

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

    if not content_summary or len(content_summary) < 20:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse"}})

    try:
        brain = await _get_brain()

        # Assess novelty before storing.
        from memories.learning import LearningEngine
        is_novel = await brain._learning.assess_novelty(content_summary)

        novelty = "pass" if is_novel else "fail"

        new_atom_id: int | None = None
        if is_novel:
            # Infer atom type from tool context.
            atom_type = _infer_atom_type(tool_name, tool_input, tool_response)
            cwd = data.get("cwd", "")
            project = _project_name(cwd)
            
            # Calculate importance based on type and content
            importance = _calculate_importance(atom_type, content_summary, tool_name)

            remember_result = await brain.remember(
                content=content_summary[:800],
                type=atom_type,
                source_project=project,
                importance=importance,
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

        for learning in learnings:
            content, atom_type = learning
            if content and len(content) > 20:
                # Check novelty
                from memories.learning import LearningEngine
                is_novel = await brain._learning.assess_novelty(content)

                if is_novel:
                    cwd = data.get("cwd", "")
                    # Calculate importance based on type and content
                    importance = _calculate_importance(atom_type, content)
                    
                    await brain.remember(
                        content=content[:800],
                        type=atom_type,
                        source_project=_project_name(cwd),
                        importance=importance,
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


def _extract_response_learnings(response: str) -> list[tuple[str, str]]:
    """Extract learnable content from an assistant response.
    
    Returns list of (content, atom_type) tuples.
    """
    learnings = []
    response_lower = response.lower()

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

    # Split into sentences and classify
    sentences = response.replace('\n', '. ').split('. ')

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 30:
            continue

        sentence_lower = sentence.lower()

        # Check for error/antipattern — only classify as antipattern if the
        # sentence contains genuine negative-assertion vocabulary (Wave 8-C).
        if any(ind in sentence_lower for ind in error_indicators):
            if _content_looks_like_antipattern(sentence):
                learnings.append((sentence[:500], "antipattern"))
            else:
                learnings.append((sentence[:500], "experience"))
            continue

        # Check for skill/how-to
        if any(ind in sentence_lower for ind in skill_indicators):
            learnings.append((sentence[:500], "skill"))
            continue

        # Check for fact/discovery
        if any(ind in sentence_lower for ind in fact_indicators):
            learnings.append((sentence[:500], "fact"))
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

_ANTIPATTERN_KEYWORDS = frozenset([
    "should not", "should never", "avoid", "never use", "never do",
    "wrong", "incorrect", "bad practice", "pitfall", "mistake",
    "dangerous", "risk", "don't", "do not", "bug", "error-prone",
    "antipattern", "anti-pattern", "failure mode", "gotcha",
    "footgun", "trap", "beware",
])


def _content_looks_like_antipattern(content: str) -> bool:
    """Return True only if *content* contains negative-assertion vocabulary.

    Used as a guard before classifying an atom as ``antipattern``.  Content
    that merely mentions "error" or "failed" (e.g. command output, status
    reports) should **not** be treated as an antipattern -- only content that
    expresses a warning, a bad practice, or something to avoid.
    """
    lower = content.lower()
    return any(kw in lower for kw in _ANTIPATTERN_KEYWORDS)


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
_THINKING_TRUNCATE = 300  # chars to keep per thinking block (atomic insight)
_TEXT_TRUNCATE = 800      # chars to keep from the final text block
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

    prompt = (
        "Extract 2 to 5 key atomic facts or insights from the following reasoning. "
        "Output each as a single, self-contained sentence on its own line, "
        "starting with '- '. No preamble, no explanation.\n\n"
        f"{text[:1500]}"
    )
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
            if fact and len(fact) >= 20:
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
                is_novel = await brain._learning.assess_novelty(fact)
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
                            is_novel = await brain._learning.assess_novelty(fact)
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

    for i, atom_id in enumerate(atom_ids):
        try:
            synapses = await brain._learning.auto_link(atom_id)
            total_synapses += len(synapses)
        except Exception as exc:
            log.debug("relink: auto_link failed for atom %d: %s", atom_id, exc)

        if verbose and (i + 1) % 50 == 0:
            print(
                f"  {i + 1}/{len(atom_ids)} atoms re-linked, "
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
    "general": "project:utils",
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

    # Build atom_id → Unix timestamp map for temporal Hebbian weighting.
    from datetime import datetime, timezone as _tz
    atom_timestamps: dict[int, float] = {}
    for row in rows:
        ts_str = row["accessed_at"]
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                atom_timestamps[row["atom_id"]] = dt.timestamp()
            except (ValueError, TypeError):
                pass

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


async def _hook_pre_tool_use(data: dict[str, Any]) -> str:
    """Handle PreToolUse hook.

    Captures Claude's intent before tool execution.  Only ``Task`` and
    ``Bash`` (with a descriptive description) are processed -- other tools
    are either noisy, low-signal, or better captured by PostToolUse.

    For Bash commands, also performs a fast antipattern-only recall to
    surface relevant warnings before execution.
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
                output_lines.append(f"[memories] {len(unique_warnings)} relevant warnings:")
                for w in unique_warnings[:3]:  # cap at 3
                    output_lines.append(_format_atom_line(w))

        is_novel = await brain._learning.assess_novelty(content)
        novelty = "pass" if is_novel else "fail"

        new_atom_id: int | None = None
        if is_novel:
            atom_type = "insight" if tool_name == "Task" else "experience"
            cwd = data.get("cwd", "")
            project = _project_name(cwd)
            remember_result = await brain.remember(
                content=content[:800],
                type=atom_type,
                source_project=project,
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
    errors, so no error-signature matching is needed.  Content is stored as
    antipattern if it contains negative-assertion vocabulary, otherwise experience.
    """
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    error = data.get("error", "") or data.get("tool_response", "")

    if tool_name in _SKIP_TOOLS or not error or len(str(error).strip()) < 20:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUseFailure"}})

    command = ""
    if isinstance(tool_input, dict):
        command = tool_input.get("command", tool_input.get("file_path", ""))
    elif isinstance(tool_input, str):
        try:
            command = json.loads(tool_input).get("command", "")
        except (json.JSONDecodeError, TypeError):
            pass

    content = f"Tool `{tool_name}` failed: {str(error)[:500]}"
    if command:
        content = f"Command `{command[:100]}` failed: {str(error)[:400]}"

    try:
        brain = await _get_brain()
        is_novel = await brain._learning.assess_novelty(content)
        new_atom_id: int | None = None
        if is_novel:
            atom_type = (
                "antipattern"
                if _content_looks_like_antipattern(content)
                else "experience"
            )
            cwd = data.get("cwd", "")
            project = _project_name(cwd)
            importance = _calculate_importance(atom_type, content, tool_name)
            result = await brain.remember(
                content=content,
                type=atom_type,
                source_project=project,
                importance=importance,
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

    content = f"Completed: {task_subject}"
    if task_description and len(task_description.strip()) > 20:
        content = f"Completed: {task_subject} — {task_description[:300]}"

    try:
        brain = await _get_brain()
        cwd = data.get("cwd", "")
        project = _project_name(cwd)
        session_id = data.get("session_id", "")

        is_novel = await brain._learning.assess_novelty(content)
        new_atom_id: int | None = None
        if is_novel:
            result = await brain.remember(
                content=content,
                type="experience",
                source_project=project,
                importance=0.6,
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

    content = f"Claude asked for clarification: {message[:400]}"

    try:
        brain = await _get_brain()
        cwd = data.get("cwd", "")
        project = _project_name(cwd)
        session_id = data.get("session_id", "")

        is_novel = await brain._learning.assess_novelty(content)
        new_atom_id: int | None = None
        if is_novel:
            result = await brain.remember(
                content=content,
                type="insight",
                source_project=project,
                importance=0.55,
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

    Fires when the parent session spawns a Task sub-agent.  Captures the
    delegation pattern as an insight so the system learns which agent types
    handle which kinds of work over time.
    """
    agent_type = data.get("agent_type", "")
    agent_id = data.get("agent_id", "")

    # Only capture meaningful agent types — skip unnamed/empty.
    if not agent_type or agent_type in ("Bash",):
        return ""

    try:
        brain = await _get_brain()
        cwd = data.get("cwd", "")
        project = _project_name(cwd)
        session_id = data.get("session_id", "")

        content = f"Delegated to {agent_type} sub-agent"
        if project:
            content = f"Delegated to {agent_type} sub-agent in project {project}"

        is_novel = await brain._learning.assess_novelty(content)
        new_atom_id: int | None = None
        if is_novel:
            result = await brain.remember(
                content=content,
                type="insight",
                source_project=project,
                importance=0.4,
            )
            new_atom_id = result.get("atom_id")

        await _save_hook_atoms(brain, session_id, extra_atom_id=new_atom_id)
        await _record_hook_stat(
            brain, "subagent-start", 0, project=project,
            novelty_result="pass" if is_novel else "fail",
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

        is_novel = await brain._learning.assess_novelty(content)
        new_atom_id: int | None = None
        if is_novel:
            atom_type = (
                "antipattern"
                if _content_looks_like_antipattern(content)
                else "experience"
            )
            importance = _calculate_importance(atom_type, content, tool_name)
            result = await brain.remember(
                content=content,
                type=atom_type,
                source_project=project,
                importance=importance,
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

    # Unknown command -- don't exit, fall through to MCP server.
