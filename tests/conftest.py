"""Shared fixtures and helpers for the memories test suite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from memories.brain import Brain
from memories.embeddings import EmbeddingEngine
from memories.storage import Storage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def brain(tmp_path: Path) -> Brain:
    """Provide an initialized Brain instance backed by a temp database.

    Uses a fresh SQLite file in ``tmp_path`` so tests never touch the
    user's production database at ``~/.memories/memories.db``.
    """
    b = Brain(db_path=tmp_path / "brain.db")
    await b.initialize()
    yield b  # type: ignore[misc]
    await b.shutdown()


@pytest.fixture
async def storage(tmp_path: Path) -> Storage:
    """Provide an initialized Storage instance backed by a temp directory.

    The database file, backup directory, and all related artefacts live
    entirely inside ``tmp_path`` so tests never touch the user's real data.
    """
    db_path = tmp_path / "test.db"
    s = Storage(db_path)
    # Point backups at a temp sub-directory as well.
    s._backup_dir = tmp_path / "backups"
    s._backup_dir.mkdir(exist_ok=True)
    await s.initialize()
    yield s  # type: ignore[misc]
    await s.close()


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Return a MagicMock standing in for EmbeddingEngine.

    Pre-configured with sensible async defaults so tests can override
    per-method return values as needed.
    """
    engine = MagicMock(spec=EmbeddingEngine)
    engine.search_similar = AsyncMock(return_value=[])
    engine.embed_text = AsyncMock(return_value=[0.0] * 768)
    engine.embed_and_store = AsyncMock(return_value=[0.0] * 768)
    engine.embed_batch = AsyncMock(return_value=[])
    engine.health_check = AsyncMock(return_value=True)
    engine.cosine_similarity = MagicMock(return_value=0.5)  # sync function (M-6)
    return engine


# ---------------------------------------------------------------------------
# Shared test helpers -- direct SQL insertion bypassing managers
# ---------------------------------------------------------------------------


async def insert_atom(
    storage: Storage,
    content: str,
    atom_type: str = "fact",
    region: str = "general",
    confidence: float = 1.0,
    importance: float = 0.5,
    access_count: int = 0,
    last_accessed_at: str | None = None,
    tags: list[str] | None = None,
    is_deleted: bool = False,
    severity: str | None = None,
    instead: str | None = None,
    created_at: str | None = None,
    source_project: str | None = None,
    source_session: str | None = None,
    source_file: str | None = None,
) -> int:
    """Insert an atom directly via SQL, bypassing AtomManager (no embedding).

    Returns the auto-generated atom ID.
    """
    now = datetime.now(tz=timezone.utc).isoformat()
    if last_accessed_at is None:
        last_accessed_at = now
    tags_json = json.dumps(tags) if tags else None

    atom_id = await storage.execute_write(
        """
        INSERT INTO atoms
            (content, type, region, confidence, importance, access_count,
             last_accessed_at, tags, is_deleted, severity, instead,
             source_project, source_session, source_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            content,
            atom_type,
            region,
            confidence,
            importance,
            access_count,
            last_accessed_at,
            tags_json,
            int(is_deleted),
            severity,
            instead,
            source_project,
            source_session,
            source_file,
        ),
    )

    # Override created_at if requested (for supersession testing).
    if created_at is not None:
        await storage.execute_write(
            "UPDATE atoms SET created_at = ? WHERE id = ?",
            (created_at, atom_id),
        )

    return atom_id


async def insert_synapse(
    storage: Storage,
    source_id: int,
    target_id: int,
    relationship: str = "related-to",
    strength: float = 0.5,
    bidirectional: bool = True,
) -> int:
    """Insert a synapse directly via SQL.

    Returns the auto-generated synapse ID.
    """
    return await storage.execute_write(
        """
        INSERT INTO synapses
            (source_id, target_id, relationship, strength, bidirectional)
        VALUES (?, ?, ?, ?, ?)
        """,
        (source_id, target_id, relationship, strength, int(bidirectional)),
    )


async def count_synapses(
    storage: Storage,
    source_id: int | None = None,
    target_id: int | None = None,
    relationship: str | None = None,
) -> int:
    """Count synapses matching the given filters."""
    clauses: list[str] = []
    params: list[Any] = []

    if source_id is not None:
        clauses.append("source_id = ?")
        params.append(source_id)
    if target_id is not None:
        clauses.append("target_id = ?")
        params.append(target_id)
    if relationship is not None:
        clauses.append("relationship = ?")
        params.append(relationship)

    where = " AND ".join(clauses) if clauses else "1=1"
    rows = await storage.execute(
        f"SELECT COUNT(*) AS cnt FROM synapses WHERE {where}",
        tuple(params),
    )
    return rows[0]["cnt"]
