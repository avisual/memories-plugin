"""Migration from claude-mem observations to memories atoms.

Reads observations from the claude-mem SQLite database and creates
corresponding atoms in the memories system, preserving timestamps
and mapping types appropriately.

Usage::

    python -m memories migrate --source ~/.claude-mem/claude-mem.db --dry-run
    python -m memories migrate --source ~/.claude-mem/claude-mem.db
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Type mapping
# ------------------------------------------------------------------

_TYPE_MAP: dict[str, str] = {
    "bugfix": "antipattern",
    "feature": "experience",
    "discovery": "fact",  # default for discovery; may be upgraded to insight
    "decision": "insight",
    "refactor": "experience",
    "change": "fact",
}

_INSIGHT_KEYWORDS: frozenset[str] = frozenset({
    "because", "reason", "rationale", "trade-off", "tradeoff",
    "conclusion", "learned", "realised", "realized", "insight",
    "key finding", "important", "principle",
})


def _map_type(obs_type: str, content: str) -> str:
    """Map a claude-mem observation type to an atom type."""
    atom_type = _TYPE_MAP.get(obs_type, "fact")

    # Upgrade discovery -> insight if content has rationale keywords.
    if obs_type == "discovery" and atom_type == "fact":
        content_lower = content.lower()
        if any(kw in content_lower for kw in _INSIGHT_KEYWORDS):
            return "insight"

    return atom_type


def _assemble_content(title: str | None, narrative: str | None) -> str | None:
    """Assemble atom content from observation title and narrative."""
    parts = []
    if title and title.strip():
        parts.append(title.strip())
    if narrative and narrative.strip():
        parts.append(narrative.strip())

    if not parts:
        return None

    content = "\n".join(parts)
    # Truncate to ~800 chars (~200 tokens).
    if len(content) > 800:
        content = content[:797] + "..."
    return content


def _parse_json_field(value: str | None) -> list[str]:
    """Parse a JSON array field, returning an empty list on failure."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def _project_basename(project: str | None) -> str | None:
    """Extract the basename from a project path."""
    if not project:
        return None
    return Path(project).name or None


# ------------------------------------------------------------------
# Read observations from claude-mem
# ------------------------------------------------------------------

def _read_observations(db_path: Path) -> list[dict[str, Any]]:
    """Read all observations from the claude-mem database."""
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    cursor = conn.execute(
        """
        SELECT id, project, type, title, narrative, concepts,
               files_modified, files_read, created_at
        FROM observations
        ORDER BY id
        """
    )

    observations = []
    for row in cursor:
        observations.append(dict(row))

    conn.close()
    return observations


# ------------------------------------------------------------------
# Migration core
# ------------------------------------------------------------------

async def _migrate(
    source_path: Path,
    dry_run: bool = False,
    batch_size: int = 32,
) -> dict[str, int]:
    """Run the migration from claude-mem to memories.

    Returns statistics dict with counts of processed, skipped, created,
    and errored observations.
    """
    from memories.brain import Brain
    from memories.storage import Storage, serialize_embedding

    observations = _read_observations(source_path)
    print(f"Read {len(observations)} observations from {source_path}")

    # Filter out empty observations.
    usable = []
    skipped_empty = 0
    for obs in observations:
        content = _assemble_content(obs.get("title"), obs.get("narrative"))
        if content is None:
            skipped_empty += 1
            continue
        obs["_content"] = content
        usable.append(obs)

    print(f"Usable: {len(usable)}, Skipped (empty): {skipped_empty}")

    if dry_run:
        # Show a preview of what would be migrated.
        type_counts: dict[str, int] = {}
        for obs in usable:
            atom_type = _map_type(obs["type"], obs["_content"])
            type_counts[atom_type] = type_counts.get(atom_type, 0) + 1

        print("\nDry run — type distribution:")
        for t, c in sorted(type_counts.items()):
            print(f"  {t}: {c}")
        print(f"\nTotal atoms that would be created: {len(usable)}")
        return {
            "total": len(observations),
            "usable": len(usable),
            "skipped_empty": skipped_empty,
            "dry_run": True,
        }

    # Initialize the brain for real migration.
    brain = Brain()
    await brain.initialize()

    created = 0
    errored = 0
    batch_contents: list[str] = []
    batch_obs: list[dict] = []

    # Process in batches for embedding efficiency.
    for i, obs in enumerate(usable):
        batch_contents.append(obs["_content"])
        batch_obs.append(obs)

        if len(batch_contents) >= batch_size or i == len(usable) - 1:
            # Batch embed.
            try:
                embeddings = await brain._embeddings.embed_batch(batch_contents)
            except Exception as exc:
                log.warning("Batch embed failed: %s", exc)
                embeddings = [None] * len(batch_contents)

            # Create atoms one by one (to get IDs and timestamps right).
            for j, (obs_item, content) in enumerate(zip(batch_obs, batch_contents)):
                try:
                    atom_type = _map_type(obs_item["type"], content)
                    tags = _parse_json_field(obs_item.get("concepts"))
                    files_modified = _parse_json_field(obs_item.get("files_modified"))
                    source_file = files_modified[0] if files_modified else None
                    project_path = obs_item.get("project")
                    project_name = _project_basename(project_path)
                    region = f"project:{project_name}" if project_name else "general"
                    created_at = obs_item.get("created_at", "")

                    # Insert atom directly via storage for timestamp control.
                    tags_json = json.dumps(tags) if tags else None
                    severity = None
                    instead = None

                    atom_id = await brain._storage.execute_write(
                        """
                        INSERT INTO atoms
                            (content, type, region, confidence, tags,
                             severity, instead, source_project, source_session,
                             source_file, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            content,
                            atom_type,
                            region,
                            1.0,
                            tags_json,
                            severity,
                            instead,
                            project_name,
                            None,
                            source_file,
                            created_at,
                            created_at,
                        ),
                    )

                    # Store the pre-computed embedding.
                    if embeddings[j] is not None and brain._storage.vec_available:
                        blob = serialize_embedding(embeddings[j])
                        await brain._storage.execute_write(
                            "INSERT OR REPLACE INTO atoms_vec(atom_id, embedding) VALUES (?, ?)",
                            (atom_id, blob),
                        )

                    # Ensure region exists.
                    await brain._storage.execute_write(
                        "INSERT OR IGNORE INTO regions (name) VALUES (?)",
                        (region,),
                    )
                    await brain._storage.execute_write(
                        "UPDATE regions SET atom_count = atom_count + 1 WHERE name = ?",
                        (region,),
                    )

                    created += 1

                except Exception as exc:
                    log.warning(
                        "Failed to migrate observation %d: %s",
                        obs_item.get("id", "?"),
                        exc,
                    )
                    errored += 1

            # Progress reporting.
            total_processed = created + errored
            if total_processed % 100 == 0 or i == len(usable) - 1:
                print(f"  Progress: {total_processed}/{len(usable)} "
                      f"(created={created}, errors={errored})")

            batch_contents.clear()
            batch_obs.clear()

    # Auto-link in batches (lighter pass — just top connections).
    print("\nAuto-linking atoms...")
    linked = 0
    for atom_id_row in await brain._storage.execute(
        "SELECT id FROM atoms ORDER BY id"
    ):
        try:
            synapses = await brain._learning.auto_link(atom_id_row["id"])
            if synapses:
                linked += len(synapses)
        except Exception:
            pass

        if atom_id_row["id"] % 100 == 0:
            print(f"  Linked through atom {atom_id_row['id']}... ({linked} synapses)")

    print(f"Auto-linking complete: {linked} synapses created")

    # Run initial consolidation to merge near-duplicates.
    print("\nRunning consolidation...")
    try:
        consolidation_result = await brain.reflect(scope="all", dry_run=False)
        print(f"  Merged: {consolidation_result.get('merged', 0)}")
        print(f"  Pruned: {consolidation_result.get('pruned', 0)}")
    except Exception as exc:
        print(f"  Consolidation error: {exc}")

    await brain.shutdown()

    stats = {
        "total": len(observations),
        "usable": len(usable),
        "skipped_empty": skipped_empty,
        "created": created,
        "errored": errored,
        "synapses_linked": linked,
    }

    print(f"\nMigration complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def run_migration(args: list[str]) -> None:
    """Parse arguments and run the migration."""
    parser = argparse.ArgumentParser(
        description="Migrate claude-mem observations to memories atoms",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("~/.claude-mem/claude-mem.db").expanduser(),
        help="Path to the claude-mem SQLite database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be migrated without making changes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )

    parsed = parser.parse_args(args)
    asyncio.run(_migrate(parsed.source, parsed.dry_run, parsed.batch_size))
