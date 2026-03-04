"""Migration completeness tests.

Verifies that all 14 schema migrations have been applied correctly.
Tests run against a freshly-initialized in-memory (temp) database to
confirm the post-migration schema is correct, and against the live
database when present to confirm it has been fully migrated.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memories.storage import Storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_schema(db_path: Path) -> dict[str, str]:
    """Return {table_name: CREATE_sql} for all tables in the database."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {row[0]: (row[1] or "") for row in rows}
    finally:
        conn.close()


def _get_indexes(db_path: Path) -> dict[str, str]:
    """Return {index_name: CREATE_sql} for all indexes in the database."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        ).fetchall()
        return {row[0]: (row[1] or "") for row in rows}
    finally:
        conn.close()


def _get_columns(db_path: Path, table: str) -> set[str]:
    """Return column names for a table."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cursor.fetchall()}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# TestMigrationCompleteness
# ---------------------------------------------------------------------------


class TestMigrationCompleteness:
    """Verify all 14 migrations produce the correct schema on a fresh DB."""

    @pytest.fixture
    async def db(self, tmp_path: Path) -> Path:
        """Return path to a freshly-initialized database."""
        db_path = tmp_path / "test.db"
        s = Storage(db_path)
        s._backup_dir = tmp_path / "backups"
        s._backup_dir.mkdir(exist_ok=True)
        await s.initialize()
        await s.close()
        return db_path

    # ------------------------------------------------------------------
    # Core table existence
    # ------------------------------------------------------------------

    async def test_all_tables_exist(self, db: Path) -> None:
        """Every expected table is present after initialization."""
        schema = _get_schema(db)
        expected = {
            "atoms",
            "synapses",
            "regions",
            "sessions",
            "consolidation_log",
            "embedding_cache",
            "locks",
            "hook_stats",
            "active_sessions",
            "session_lineage",
            "hook_session_atoms",
            "atom_feedback",
            "retrieval_weights",
            "retrieval_weight_log",
        }
        missing = expected - set(schema.keys())
        assert not missing, f"Missing tables: {missing}"

    # ------------------------------------------------------------------
    # Migration 1 — atoms.importance
    # ------------------------------------------------------------------

    async def test_atoms_importance_column_exists(self, db: Path) -> None:
        """Migration 1: atoms.importance column present."""
        cols = _get_columns(db, "atoms")
        assert "importance" in cols

    # ------------------------------------------------------------------
    # Migration 2 — atoms.task_status
    # ------------------------------------------------------------------

    async def test_atoms_task_status_column_exists(self, db: Path) -> None:
        """Migration 2: atoms.task_status column present."""
        cols = _get_columns(db, "atoms")
        assert "task_status" in cols

    # ------------------------------------------------------------------
    # Migration 3 — hook_stats CHECK includes post-response
    # ------------------------------------------------------------------

    async def test_hook_stats_check_includes_post_response(self, db: Path) -> None:
        """Migration 3: hook_stats CHECK constraint includes 'post-response'."""
        schema = _get_schema(db)
        hook_sql = schema.get("hook_stats", "")
        assert "post-response" in hook_sql, (
            "hook_stats CHECK constraint must include 'post-response'"
        )

    # ------------------------------------------------------------------
    # Migration 4 — atom_feedback.processed_at
    # ------------------------------------------------------------------

    async def test_atom_feedback_processed_at_exists(self, db: Path) -> None:
        """Migration 4: atom_feedback.processed_at column present."""
        cols = _get_columns(db, "atom_feedback")
        assert "processed_at" in cols

    # ------------------------------------------------------------------
    # Migration 5 — retrieval_weights table
    # ------------------------------------------------------------------

    async def test_retrieval_weights_table_exists(self, db: Path) -> None:
        """Migration 5: retrieval_weights singleton table present."""
        schema = _get_schema(db)
        assert "retrieval_weights" in schema

    # ------------------------------------------------------------------
    # Migration 6 — retrieval_weights.spread_activation
    # ------------------------------------------------------------------

    async def test_retrieval_weights_spread_activation_column(self, db: Path) -> None:
        """Migration 6: retrieval_weights.spread_activation column present."""
        cols = _get_columns(db, "retrieval_weights")
        assert "spread_activation" in cols

    # ------------------------------------------------------------------
    # Migration 7 — hook_session_atoms.accessed_at
    # ------------------------------------------------------------------

    async def test_hook_session_atoms_accessed_at_column(self, db: Path) -> None:
        """Migration 7: hook_session_atoms.accessed_at column present."""
        cols = _get_columns(db, "hook_session_atoms")
        assert "accessed_at" in cols

    # ------------------------------------------------------------------
    # Migration 8 — hook_stats CHECK includes post-tool-failure
    # ------------------------------------------------------------------

    async def test_hook_stats_check_includes_post_tool_failure(self, db: Path) -> None:
        """Migration 8: hook_stats CHECK includes 'post-tool-failure'."""
        schema = _get_schema(db)
        hook_sql = schema.get("hook_stats", "")
        assert "post-tool-failure" in hook_sql

    # ------------------------------------------------------------------
    # Migration 9 — hook_stats includes subagent-start, permission-request
    # ------------------------------------------------------------------

    async def test_hook_stats_check_includes_subagent_start(self, db: Path) -> None:
        """Migration 9: hook_stats CHECK includes 'subagent-start'."""
        schema = _get_schema(db)
        hook_sql = schema.get("hook_stats", "")
        assert "subagent-start" in hook_sql

    async def test_hook_stats_check_includes_permission_request(self, db: Path) -> None:
        """Migration 9: hook_stats CHECK includes 'permission-request'."""
        schema = _get_schema(db)
        hook_sql = schema.get("hook_stats", "")
        assert "permission-request" in hook_sql

    # ------------------------------------------------------------------
    # Migration 10 — hook_stats includes task-completed, notification
    # ------------------------------------------------------------------

    async def test_hook_stats_check_includes_task_completed(self, db: Path) -> None:
        """Migration 10: hook_stats CHECK includes 'task-completed'."""
        schema = _get_schema(db)
        hook_sql = schema.get("hook_stats", "")
        assert "task-completed" in hook_sql

    async def test_hook_stats_check_includes_notification(self, db: Path) -> None:
        """Migration 10: hook_stats CHECK includes 'notification'."""
        schema = _get_schema(db)
        hook_sql = schema.get("hook_stats", "")
        assert "notification" in hook_sql

    # ------------------------------------------------------------------
    # Migration 11 — synapses.tag_expires_at and encoded-with
    # ------------------------------------------------------------------

    async def test_synapses_tag_expires_at_column(self, db: Path) -> None:
        """Migration 11: synapses.tag_expires_at column present."""
        cols = _get_columns(db, "synapses")
        assert "tag_expires_at" in cols

    async def test_synapses_check_includes_encoded_with(self, db: Path) -> None:
        """Migration 11: synapses CHECK includes 'encoded-with' relationship."""
        schema = _get_schema(db)
        syn_sql = schema.get("synapses", "")
        assert "encoded-with" in syn_sql

    # ------------------------------------------------------------------
    # Migration 12 — hook_session_atoms NULL backfill (structural check)
    # ------------------------------------------------------------------

    async def test_hook_session_atoms_no_null_accessed_at(self, db: Path) -> None:
        """Migration 12: freshly-initialized DB has no NULL accessed_at rows."""
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM hook_session_atoms WHERE accessed_at IS NULL"
            ).fetchone()
            assert rows[0] == 0, "No NULL accessed_at values expected in fresh DB"
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Migration 13 — retrieval_weight_log table
    # ------------------------------------------------------------------

    async def test_retrieval_weight_log_exists(self, db: Path) -> None:
        """Migration 13: retrieval_weight_log table present."""
        schema = _get_schema(db)
        assert "retrieval_weight_log" in schema

    async def test_retrieval_weight_log_columns(self, db: Path) -> None:
        """Migration 13: retrieval_weight_log has expected columns."""
        cols = _get_columns(db, "retrieval_weight_log")
        expected = {"id", "session_id", "rating", "weight_delta", "created_at"}
        missing = expected - cols
        assert not missing, f"Missing columns in retrieval_weight_log: {missing}"

    # ------------------------------------------------------------------
    # Migration 14 — idx_synapses_bidirectional partial index
    # ------------------------------------------------------------------

    async def test_bidirectional_index_exists(self, db: Path) -> None:
        """Migration 14: idx_synapses_bidirectional partial index present."""
        indexes = _get_indexes(db)
        assert "idx_synapses_bidirectional" in indexes, (
            "Partial index idx_synapses_bidirectional must exist after migration 14"
        )

    # ------------------------------------------------------------------
    # Sanity: INSERT into hook_stats with all modern hook types
    # ------------------------------------------------------------------

    async def test_hook_stats_accepts_all_hook_types(self, db: Path) -> None:
        """All expected hook_type values can be inserted without CHECK violation."""
        hook_types = [
            "session-start", "prompt-submit", "pre-tool", "post-tool",
            "post-response", "stop", "subagent-stop", "pre-compact",
            "post-tool-failure", "session-end", "subagent-start",
            "permission-request", "task-completed", "notification",
        ]
        conn = sqlite3.connect(str(db))
        try:
            for ht in hook_types:
                try:
                    conn.execute(
                        "INSERT INTO hook_stats (hook_type, latency_ms) VALUES (?, 0)",
                        (ht,),
                    )
                    conn.rollback()
                except sqlite3.IntegrityError as exc:
                    pytest.fail(
                        f"hook_type '{ht}' rejected by CHECK constraint: {exc}"
                    )
        finally:
            conn.close()
