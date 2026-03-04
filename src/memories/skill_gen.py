"""Skill generator: produce a SKILL.md for a project from memories atoms.

Queries the memories graph for the highest-value atoms in a project region
and formats them into a Claude Code SKILL.md that any agent can use.

Usage:
    uv run python -m memories generate-skill myproject
    uv run python -m memories generate-skill myproject --output .claude/skills/myproject/SKILL.md
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Maximum display length for atom content in SKILL.md.
_MAX_CONTENT_LEN = 300

# Substrings that indicate research/external-framework noise rather than
# actionable project knowledge.  Case-insensitive matching.
_RESEARCH_NOISE_PATTERNS: tuple[str, ...] = (
    "gepa adapter",
    "gepa skill format",
    "gepa integration",
    "gepa actionable side information",
    "gepa proposer mechanism",
    "gepa cross-model",
    "gepaadapter protocol",
    "gskill pipeline",
    "gskill learns",
    "pareto front selection",
    "pareto frontier",
    "cross-model skill transfer",
    "swe-smith mines",
    "swe-smith task generation",
    "mini-swe-agent",
    "asi (actionable side information)",
    "evaluationbatch.trajectories",
    "evaluationbatch.objective_scores",
    "ingest_gepa_evaluation",
    "memoriesreflectiveproposer",
    "proposenewcandidate protocol",
    "asi→atoms",
)


async def generate_skill(
    project: str,
    output_path: str | None = None,
    db_path: Path | None = None,
) -> str:
    """Generate a SKILL.md for a project from its highest-value atoms.

    Uses a lightweight read-only SQLite connection instead of
    ``Storage.initialize()`` to avoid running migrations, backups, or
    other write-side setup on what is a purely read-only code path.

    Parameters
    ----------
    project:
        The project name.  Atoms are queried from the region
        ``project:{project}``.
    output_path:
        If provided, the generated Markdown is written to this file.
        Parent directories are created automatically.
    db_path:
        Path to the SQLite database.  Defaults to ``cfg.db_path``.
        Pass the storage's ``db_path`` when calling from the consolidation
        engine so tests can use a temporary database.

    Returns
    -------
    str
        The generated SKILL.md Markdown content.
    """
    if db_path is None:
        from memories.config import get_config
        db_path = get_config().db_path

    # Lightweight read-only approach — don't run migrations or backup.
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            region = f"project:{project}"
            # Exclude atoms that have been superseded by a live atom.
            rows = conn.execute(
                """
                SELECT a.content, a.type, a.importance, a.confidence, a.severity
                FROM atoms a
                LEFT JOIN synapses s_sup
                    ON s_sup.target_id = a.id
                   AND s_sup.relationship = 'supersedes'
                LEFT JOIN atoms a_sup
                    ON a_sup.id = s_sup.source_id
                   AND a_sup.is_deleted = 0
                WHERE a.region = ?
                  AND a.is_deleted = 0
                  AND a.type IN ('antipattern', 'insight', 'skill', 'fact')
                  AND a.importance >= 0.6
                  AND a.confidence >= 0.3
                  AND length(a.content) >= 60
                  AND a_sup.id IS NULL
                ORDER BY a.importance DESC, a.confidence DESC
                LIMIT 80
                """,
                (region,),
            ).fetchall()

            # Post-process: remove research noise and deduplicate.
            rows = _filter_research_noise(rows)
            rows = _deduplicate(rows)
            # Cap at 50 after filtering.
            rows = rows[:50]

            if not rows:
                md = _format_empty(project)
            else:
                md = _format_skill_md(project, rows)

            if output_path:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(md, encoding="utf-8")
                log.info("SKILL.md written to %s", out)

            return md
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        return f"# {project}\n\nCannot read memories DB: {exc}\n"


def _filter_research_noise(rows: list[Any]) -> list[Any]:
    """Remove atoms about external research frameworks (GEPA, gskill, etc.)."""
    filtered = []
    for row in rows:
        content_lower = row["content"].lower()
        if any(pat in content_lower for pat in _RESEARCH_NOISE_PATTERNS):
            continue
        filtered.append(row)
    return filtered


def _word_set(text: str) -> set[str]:
    """Extract lowercase word tokens from text."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


# Technical identifiers: function names (_foo_bar), file refs (foo.py),
# method calls (foo.bar()), and PascalCase class names.
_TECH_ID_RE = re.compile(
    r"_[a-z][a-z0-9_]+|[a-z_]+\.py|[A-Z][a-z]+[A-Z][a-z]+\w*"
)


def _tech_ids(text: str) -> set[str]:
    """Extract technical identifiers from text."""
    return set(m.lower() for m in _TECH_ID_RE.findall(text))


def _deduplicate(rows: list[Any], threshold: float = 0.5) -> list[Any]:
    """Remove near-duplicate atoms using word overlap + technical identifier matching.

    Two tiers:
    1. Jaccard word overlap >= threshold → duplicate
    2. 2+ shared technical identifiers AND Jaccard >= 0.2 → duplicate

    Rows are already sorted by importance DESC — the first (highest-importance)
    atom in each duplicate cluster is kept.
    """
    kept: list[Any] = []
    kept_words: list[set[str]] = []
    kept_tech: list[set[str]] = []

    for row in rows:
        words = _word_set(row["content"])
        if len(words) < 3:
            continue

        tech = _tech_ids(row["content"])
        is_dup = False

        for i, existing_words in enumerate(kept_words):
            overlap = len(words & existing_words)
            union = len(words | existing_words)
            jaccard = overlap / union if union > 0 else 0.0

            # Tier 1: high word overlap.
            if jaccard >= threshold:
                is_dup = True
                break

            # Tier 2: shared technical identifiers + moderate overlap.
            shared_tech = len(tech & kept_tech[i])
            if shared_tech >= 2 and jaccard >= 0.2:
                is_dup = True
                break

        if not is_dup:
            kept.append(row)
            kept_words.append(words)
            kept_tech.append(tech)

    return kept


def _truncate(content: str, max_len: int = _MAX_CONTENT_LEN) -> str:
    """Truncate content to max_len, breaking at a sentence boundary if possible."""
    if len(content) <= max_len:
        return content

    # Try to break at a sentence boundary within the limit.
    truncated = content[:max_len]
    last_period = truncated.rfind(". ")
    if last_period >= max_len // 3:
        return truncated[: last_period + 1]
    return truncated.rstrip() + "..."


def _format_skill_md(project: str, rows: list[Any]) -> str:
    """Format atom rows into a SKILL.md with YAML frontmatter and sections."""
    # Group by type.
    grouped: dict[str, list[Any]] = {
        "antipattern": [],
        "skill": [],
        "insight": [],
        "fact": [],
    }
    for row in rows:
        atom_type = row["type"]
        if atom_type in grouped:
            grouped[atom_type].append(row)

    total = len(rows)

    lines: list[str] = [
        "---",
        f"name: {project}",
        f"description: Repository-specific knowledge for {project} — antipatterns to avoid, proven patterns, and key architectural decisions. Auto-generated from {total} memories atoms.",
        "---",
        "",
        f"# {project} — Learned Knowledge",
        "",
    ]

    # Antipatterns.
    if grouped["antipattern"]:
        lines.append("## Antipatterns to Avoid")
        for row in grouped["antipattern"]:
            severity = row["severity"] or "medium"
            lines.append(f"- **AVOID** ({severity}): {_truncate(row['content'])}")
        lines.append("")

    # Proven Patterns.
    if grouped["skill"]:
        lines.append("## Proven Patterns")
        for row in grouped["skill"]:
            lines.append(f"- {_truncate(row['content'])}")
        lines.append("")

    # Key Insights.
    if grouped["insight"]:
        lines.append("## Key Insights")
        for row in grouped["insight"][:10]:
            lines.append(f"- {_truncate(row['content'])}")
        lines.append("")

    # Known Facts (top 10 only).
    if grouped["fact"]:
        lines.append("## Known Facts")
        for row in grouped["fact"][:10]:
            lines.append(f"- {_truncate(row['content'])}")
        lines.append("")

    return "\n".join(lines)


def _count_project_atoms(
    project: str,
    min_importance: float = 0.6,
    db_path: "Path | None" = None,
) -> int:
    """Count high-importance, non-deleted atoms in a project region.

    Used as a lightweight dirty-check before triggering auto-skill generation.
    Opens a read-only connection independent of the main Storage pool.

    Parameters
    ----------
    project:
        The project name — atoms are queried from ``project:{project}``.
    min_importance:
        Only atoms with ``importance >= min_importance`` are counted.
    db_path:
        Path to the SQLite database.  Defaults to ``cfg.db_path``.

    Returns
    -------
    int
        Number of qualifying atoms, or 0 if the database is inaccessible.
    """
    if db_path is None:
        from memories.config import get_config
        db_path = get_config().db_path
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            region = f"project:{project}"
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM atoms
                WHERE region = ?
                  AND is_deleted = 0
                  AND importance >= ?
                """,
                (region, min_importance),
            ).fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return 0


def _format_empty(project: str) -> str:
    """Format an empty SKILL.md when no atoms exist for the project."""
    return "\n".join([
        "---",
        f"name: {project}",
        f"description: Repository-specific knowledge for {project}. No atoms found yet.",
        "---",
        "",
        f"# {project} — Learned Knowledge",
        "",
        "No memories atoms found for this project yet. Use `memories remember` to start building knowledge.",
        "",
    ])
