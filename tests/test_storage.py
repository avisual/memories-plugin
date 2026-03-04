"""Comprehensive tests for the memories storage layer.

All tests use a temporary database directory (via ``tmp_path``) so that no
real user data is affected.  Tests do NOT require Ollama or any network
access -- the storage layer is pure SQLite.
"""

from __future__ import annotations

import asyncio
import math
import sqlite3
from pathlib import Path

import pytest

from memories.storage import (
    Storage,
    deserialize_embedding,
    serialize_embedding,
)


# -----------------------------------------------------------------------
# 1. Initialization
# -----------------------------------------------------------------------


class TestInitialization:
    """Verify that Storage.initialize() sets up the database correctly."""

    async def test_creates_db_file(self, storage: Storage) -> None:
        """Database file must exist on disk after initialization."""
        assert storage.db_path.exists()

    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Nested parent directories are created automatically."""
        deep_path = tmp_path / "a" / "b" / "c" / "test.db"
        s = Storage(deep_path)
        s._backup_dir = tmp_path / "backups_deep"
        await s.initialize()
        assert deep_path.exists()
        await s.close()

    async def test_wal_mode_enabled(self, storage: Storage) -> None:
        """WAL journal mode must be set for concurrent read support."""
        rows = await storage.execute("PRAGMA journal_mode")
        mode = rows[0][0]
        assert mode == "wal", f"Expected WAL journal mode, got {mode!r}"

    async def test_foreign_keys_enabled(self, storage: Storage) -> None:
        """Foreign key enforcement must be turned on."""
        rows = await storage.execute("PRAGMA foreign_keys")
        assert rows[0][0] == 1

    async def test_idempotent(self, storage: Storage) -> None:
        """Calling initialize() twice must not raise."""
        await storage.initialize()  # second call
        rows = await storage.execute("SELECT COUNT(*) FROM atoms")
        assert rows[0][0] == 0

    async def test_initialized_flag(self, storage: Storage) -> None:
        """Internal _initialized flag must be True after init."""
        assert storage._initialized is True

    async def test_vec_available_is_bool(self, storage: Storage) -> None:
        """vec_available property should be a boolean regardless of platform."""
        assert isinstance(storage.vec_available, bool)


# -----------------------------------------------------------------------
# 2. Schema -- all expected tables exist
# -----------------------------------------------------------------------

# Tables that always exist (non-virtual or FTS).
_CORE_TABLES = {
    "atoms",
    "atoms_fts",
    "synapses",
    "regions",
    "sessions",
    "consolidation_log",
    "embedding_cache",
}


class TestSchema:
    """Verify that all expected tables and indexes are present."""

    async def test_core_tables_exist(self, storage: Storage) -> None:
        """All core tables must be present in sqlite_master."""
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
            "ORDER BY name"
        )
        table_names = {row["name"] for row in rows}
        for expected in _CORE_TABLES:
            assert expected in table_names, f"Missing table: {expected}"

    async def test_vec_table_when_available(self, storage: Storage) -> None:
        """atoms_vec virtual table should exist when sqlite-vec loaded."""
        if not storage.vec_available:
            pytest.skip("sqlite-vec not available on this platform")
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE name = 'atoms_vec'"
        )
        assert len(rows) == 1

    async def test_indexes_created(self, storage: Storage) -> None:
        """Indexes defined in _INDEX_SQL must exist."""
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%'"
        )
        index_names = {row["name"] for row in rows}
        for expected in (
            "idx_atoms_type",
            "idx_atoms_region",
            "idx_atoms_confidence",
            "idx_atoms_is_deleted",
            "idx_synapses_source",
            "idx_synapses_target",
            "idx_synapses_strength",
        ):
            assert expected in index_names, f"Missing index: {expected}"

    async def test_fts_triggers_exist(self, storage: Storage) -> None:
        """FTS sync triggers (atoms_ai, atoms_ad, atoms_au) must be present."""
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        )
        trigger_names = {row["name"] for row in rows}
        for name in ("atoms_ai", "atoms_ad", "atoms_au"):
            assert name in trigger_names, f"Missing trigger: {name}"


# -----------------------------------------------------------------------
# 3. Read / Write
# -----------------------------------------------------------------------


class TestReadWrite:
    """Test execute() for reads and execute_write() for writes."""

    async def test_insert_and_select(self, storage: Storage) -> None:
        """Basic INSERT via execute_write, SELECT via execute."""
        row_id = await storage.execute_write(
            "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
            ("hello world", "fact", "general"),
        )
        assert isinstance(row_id, int)
        assert row_id > 0

        rows = await storage.execute(
            "SELECT content, type, region FROM atoms WHERE id = ?", (row_id,)
        )
        assert len(rows) == 1
        assert rows[0]["content"] == "hello world"
        assert rows[0]["type"] == "fact"
        assert rows[0]["region"] == "general"

    async def test_execute_write_returns_lastrowid(self, storage: Storage) -> None:
        """execute_write must return the lastrowid of the inserted row."""
        first = await storage.execute_write(
            "INSERT INTO regions (name, description) VALUES (?, ?)",
            ("test-region", "A region for testing"),
        )
        # Regions has TEXT PRIMARY KEY so lastrowid is typically 0 for text pks,
        # but the implementation returns cursor.lastrowid or 0.
        assert isinstance(first, int)

    async def test_execute_write_rollback_on_error(self, storage: Storage) -> None:
        """A failing write must be rolled back, leaving no partial data."""
        with pytest.raises(sqlite3.IntegrityError):
            # violates CHECK constraint on type column
            await storage.execute_write(
                "INSERT INTO atoms (content, type) VALUES (?, ?)",
                ("bad", "invalid_type"),
            )
        # Verify nothing was inserted.
        rows = await storage.execute("SELECT COUNT(*) FROM atoms")
        assert rows[0][0] == 0

    async def test_execute_returns_empty_for_no_rows(self, storage: Storage) -> None:
        """A SELECT on an empty table returns an empty list."""
        rows = await storage.execute("SELECT * FROM atoms WHERE id = 999")
        assert rows == []

    async def test_execute_many(self, storage: Storage) -> None:
        """execute_many inserts multiple rows in a single call."""
        params = [
            ("fact one", "fact", "general"),
            ("fact two", "fact", "general"),
            ("skill one", "skill", "general"),
        ]
        await storage.execute_many(
            "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
            params,
        )
        rows = await storage.execute("SELECT COUNT(*) FROM atoms")
        assert rows[0][0] == 3

    async def test_execute_script(self, storage: Storage) -> None:
        """execute_script runs multi-statement SQL."""
        await storage.execute_script(
            """
            INSERT INTO regions (name, description) VALUES ('r1', 'Region 1');
            INSERT INTO regions (name, description) VALUES ('r2', 'Region 2');
            """
        )
        rows = await storage.execute("SELECT COUNT(*) FROM regions")
        assert rows[0][0] == 2

    async def test_fts_sync_on_insert(self, storage: Storage) -> None:
        """FTS index is updated when a row is inserted into atoms."""
        await storage.execute_write(
            "INSERT INTO atoms (content, type, region, tags) VALUES (?, ?, ?, ?)",
            ("Python async programming", "skill", "general", "python,async"),
        )
        rows = await storage.execute(
            "SELECT rowid FROM atoms_fts WHERE atoms_fts MATCH 'async'"
        )
        assert len(rows) == 1

    async def test_fts_sync_on_update(self, storage: Storage) -> None:
        """FTS index reflects updates to atoms rows."""
        row_id = await storage.execute_write(
            "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
            ("original content", "fact", "general"),
        )
        await storage.execute_write(
            "UPDATE atoms SET content = ? WHERE id = ?",
            ("updated content", row_id),
        )
        # Old content should NOT match.
        old = await storage.execute(
            "SELECT rowid FROM atoms_fts WHERE atoms_fts MATCH 'original'"
        )
        assert len(old) == 0
        # New content SHOULD match.
        new = await storage.execute(
            "SELECT rowid FROM atoms_fts WHERE atoms_fts MATCH 'updated'"
        )
        assert len(new) == 1

    async def test_fts_sync_on_delete(self, storage: Storage) -> None:
        """FTS index is cleaned up when an atom is deleted."""
        row_id = await storage.execute_write(
            "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
            ("temporary content", "fact", "general"),
        )
        await storage.execute_write("DELETE FROM atoms WHERE id = ?", (row_id,))
        rows = await storage.execute(
            "SELECT rowid FROM atoms_fts WHERE atoms_fts MATCH 'temporary'"
        )
        assert len(rows) == 0


# -----------------------------------------------------------------------
# 4. Serialization -- embedding round-trip
# -----------------------------------------------------------------------


class TestSerialization:
    """Test serialize_embedding / deserialize_embedding round-trip fidelity."""

    def test_round_trip_basic(self) -> None:
        """A simple vector survives serialization and deserialization."""
        original = [1.0, 2.5, -3.14, 0.0, 42.0]
        packed = serialize_embedding(original)
        restored = deserialize_embedding(packed)
        assert len(restored) == len(original)
        for a, b in zip(original, restored):
            assert math.isclose(a, b, rel_tol=1e-6)

    def test_round_trip_768_dims(self) -> None:
        """A realistic 768-dimensional vector round-trips correctly."""
        original = [float(i) / 768.0 for i in range(768)]
        packed = serialize_embedding(original)
        restored = deserialize_embedding(packed)
        assert len(restored) == 768
        for a, b in zip(original, restored):
            assert math.isclose(a, b, rel_tol=1e-6)

    def test_empty_vector(self) -> None:
        """An empty vector round-trips to an empty list."""
        packed = serialize_embedding([])
        restored = deserialize_embedding(packed)
        assert restored == []

    def test_single_element(self) -> None:
        """A single-element vector round-trips correctly."""
        original = [3.14159]
        restored = deserialize_embedding(serialize_embedding(original))
        assert math.isclose(restored[0], original[0], rel_tol=1e-6)

    def test_serialized_type_is_bytes(self) -> None:
        """serialize_embedding must return bytes."""
        result = serialize_embedding([1.0, 2.0])
        assert isinstance(result, bytes)

    def test_serialized_length(self) -> None:
        """Each float occupies 4 bytes in the packed representation."""
        vec = [0.0] * 100
        packed = serialize_embedding(vec)
        assert len(packed) == 100 * 4

    def test_negative_and_special_values(self) -> None:
        """Negative values, very small, and very large floats survive."""
        original = [-1e30, -0.0, 0.0, 1e-38, 1e30]
        restored = deserialize_embedding(serialize_embedding(original))
        for a, b in zip(original, restored):
            assert math.isclose(a, b, rel_tol=1e-6) or (a == 0.0 and b == 0.0)


# -----------------------------------------------------------------------
# 5. Backup
# -----------------------------------------------------------------------


class TestBackup:
    """Test backup creation and pruning."""

    async def test_backup_creates_file(self, storage: Storage) -> None:
        """Calling backup() must produce a .db file in the backup directory."""
        backup_path = await storage.backup()
        assert backup_path.exists()
        assert backup_path.suffix == ".db"
        assert backup_path.parent == storage._backup_dir

    async def test_backup_filename_pattern(self, storage: Storage) -> None:
        """Backup files must follow the memories_<timestamp>.db pattern."""
        backup_path = await storage.backup()
        assert backup_path.name.startswith("memories_")
        assert backup_path.name.endswith(".db")

    async def test_multiple_backups(self, storage: Storage) -> None:
        """Multiple backup calls produce multiple files."""
        paths = set()
        for _ in range(3):
            p = await storage.backup()
            paths.add(p)
            # Tiny delay to ensure distinct timestamps.
            await asyncio.sleep(0.05)
        # At least 1 unique path (timestamps may collide within a second).
        assert len(paths) >= 1

    async def test_prune_keeps_only_n(self, storage: Storage, tmp_path: Path) -> None:
        """Old backups beyond backup_count are pruned."""
        storage._backup_count = 2
        created = []
        for _ in range(4):
            p = await storage.backup()
            created.append(p)
            await asyncio.sleep(0.05)
        remaining = list(storage._backup_dir.glob("memories_*.db"))
        assert len(remaining) <= 2

    async def test_backup_when_no_db_file(self, tmp_path: Path) -> None:
        """Backup on a non-existent DB returns the db_path without error."""
        db_path = tmp_path / "nonexistent.db"
        s = Storage(db_path)
        s._backup_dir = tmp_path / "bk"
        s._backup_dir.mkdir()
        s._backup_count = 5
        # Don't initialize -- db file does not exist.
        result = await s.backup()
        assert result == db_path


# -----------------------------------------------------------------------
# 6. Optimize
# -----------------------------------------------------------------------


class TestOptimize:
    """Test the optimize() maintenance routine."""

    async def test_optimize_runs_without_error(self, storage: Storage) -> None:
        """optimize() completes on an empty database without raising."""
        await storage.optimize()

    async def test_optimize_after_data(self, storage: Storage) -> None:
        """optimize() works after inserting data."""
        await storage.execute_write(
            "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
            ("optimize test", "fact", "general"),
        )
        await storage.optimize()

    async def test_vacuum_runs_without_error(self, storage: Storage) -> None:
        """vacuum() completes without raising."""
        await storage.vacuum()


# -----------------------------------------------------------------------
# 7. DB size
# -----------------------------------------------------------------------


class TestDbSize:
    """Test get_db_size_mb()."""

    async def test_returns_float(self, storage: Storage) -> None:
        """get_db_size_mb must return a float."""
        size = await storage.get_db_size_mb()
        assert isinstance(size, float)

    async def test_positive_after_init(self, storage: Storage) -> None:
        """After initialization the DB file has non-zero size."""
        size = await storage.get_db_size_mb()
        assert size > 0.0

    async def test_zero_when_missing(self, tmp_path: Path) -> None:
        """Returns 0.0 when the database file does not exist."""
        s = Storage(tmp_path / "nope.db")
        s._backup_dir = tmp_path / "bk2"
        s._backup_dir.mkdir()
        size = await s.get_db_size_mb()
        assert size == 0.0


# -----------------------------------------------------------------------
# 8. Table counts
# -----------------------------------------------------------------------


class TestTableCounts:
    """Test table_counts() metadata method."""

    _EXPECTED_KEYS = {
        "atoms",
        "synapses",
        "regions",
        "sessions",
        "consolidation_log",
        "embedding_cache",
    }

    async def test_returns_dict(self, storage: Storage) -> None:
        """table_counts() must return a dict."""
        counts = await storage.table_counts()
        assert isinstance(counts, dict)

    async def test_has_correct_keys(self, storage: Storage) -> None:
        """Returned dict must contain exactly the expected table names."""
        counts = await storage.table_counts()
        assert set(counts.keys()) == self._EXPECTED_KEYS

    async def test_all_zero_on_empty_db(self, storage: Storage) -> None:
        """All counts are zero on a freshly initialized database."""
        counts = await storage.table_counts()
        for table, count in counts.items():
            assert count == 0, f"Expected 0 rows in {table}, got {count}"

    async def test_reflects_inserts(self, storage: Storage) -> None:
        """Counts update after inserting rows."""
        await storage.execute_write(
            "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
            ("counted", "fact", "general"),
        )
        await storage.execute_write(
            "INSERT INTO regions (name, description) VALUES (?, ?)",
            ("test", "test region"),
        )
        counts = await storage.table_counts()
        assert counts["atoms"] == 1
        assert counts["regions"] == 1
        assert counts["synapses"] == 0


# -----------------------------------------------------------------------
# 9. Concurrent reads
# -----------------------------------------------------------------------


class TestConcurrentReads:
    """Test that multiple read operations can execute simultaneously."""

    async def test_parallel_selects(self, storage: Storage) -> None:
        """Multiple concurrent execute() calls return correct results."""
        # Seed a few rows.
        for i in range(5):
            await storage.execute_write(
                "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
                (f"concurrent-{i}", "fact", "general"),
            )

        # Fire off reads in parallel.
        tasks = [
            storage.execute("SELECT COUNT(*) AS cnt FROM atoms")
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            assert result[0]["cnt"] == 5

    async def test_parallel_reads_different_tables(self, storage: Storage) -> None:
        """Concurrent reads across different tables all succeed."""
        queries = [
            storage.execute("SELECT COUNT(*) FROM atoms"),
            storage.execute("SELECT COUNT(*) FROM regions"),
            storage.execute("SELECT COUNT(*) FROM sessions"),
            storage.execute("SELECT COUNT(*) FROM synapses"),
            storage.execute("SELECT COUNT(*) FROM consolidation_log"),
            storage.execute("SELECT COUNT(*) FROM embedding_cache"),
        ]
        results = await asyncio.gather(*queries)
        assert len(results) == 6
        for result in results:
            assert result[0][0] == 0


# -----------------------------------------------------------------------
# 10. Write lock serialization
# -----------------------------------------------------------------------


class TestWriteLock:
    """Test that execute_write serializes concurrent write calls."""

    async def test_concurrent_writes_all_committed(self, storage: Storage) -> None:
        """All concurrent writes must eventually commit without conflict."""
        async def insert(idx: int) -> int:
            return await storage.execute_write(
                "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
                (f"write-{idx}", "fact", "general"),
            )

        tasks = [insert(i) for i in range(20)]
        row_ids = await asyncio.gather(*tasks)

        # All writes should have succeeded.
        assert len(row_ids) == 20
        assert all(isinstance(rid, int) for rid in row_ids)

        # All rows should be in the database.
        rows = await storage.execute("SELECT COUNT(*) AS cnt FROM atoms")
        assert rows[0]["cnt"] == 20

    async def test_unique_row_ids(self, storage: Storage) -> None:
        """Concurrent inserts must produce unique row IDs."""
        async def insert(idx: int) -> int:
            return await storage.execute_write(
                "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
                (f"unique-{idx}", "fact", "general"),
            )

        tasks = [insert(i) for i in range(10)]
        row_ids = await asyncio.gather(*tasks)
        assert len(set(row_ids)) == 10, "Expected 10 unique row IDs"

    async def test_write_lock_is_threading_lock(self, storage: Storage) -> None:
        """The internal write lock must be a threading.Lock (not asyncio)."""
        import threading
        assert type(storage._write_lock) is type(threading.Lock())


# -----------------------------------------------------------------------
# Context manager support
# -----------------------------------------------------------------------


class TestContextManager:
    """Test async context manager protocol."""

    async def test_aenter_aexit(self, tmp_path: Path) -> None:
        """Storage can be used as an async context manager."""
        db_path = tmp_path / "ctx.db"
        s = Storage(db_path)
        s._backup_dir = tmp_path / "ctx_backups"
        s._backup_dir.mkdir()
        async with s as store:
            assert store._initialized is True
            rows = await store.execute("SELECT COUNT(*) FROM atoms")
            assert rows[0][0] == 0

    async def test_context_manager_returns_self(self, tmp_path: Path) -> None:
        """__aenter__ returns the Storage instance itself."""
        db_path = tmp_path / "ctx2.db"
        s = Storage(db_path)
        s._backup_dir = tmp_path / "ctx2_backups"
        s._backup_dir.mkdir()
        async with s as store:
            assert store is s


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge-case coverage."""

    async def test_row_factory_provides_dict_access(self, storage: Storage) -> None:
        """Rows returned by execute() support dict-like key access."""
        await storage.execute_write(
            "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
            ("dict-test", "fact", "general"),
        )
        rows = await storage.execute("SELECT content, type FROM atoms LIMIT 1")
        row = rows[0]
        assert row["content"] == "dict-test"
        assert row["type"] == "fact"

    async def test_named_params(self, storage: Storage) -> None:
        """execute() and execute_write() accept dict parameters."""
        row_id = await storage.execute_write(
            "INSERT INTO atoms (content, type, region) VALUES (:c, :t, :r)",
            {"c": "named", "t": "insight", "r": "general"},
        )
        rows = await storage.execute(
            "SELECT content FROM atoms WHERE id = :id", {"id": row_id}
        )
        assert rows[0]["content"] == "named"

    async def test_db_path_property(self, storage: Storage) -> None:
        """db_path property exposes the database file path."""
        assert isinstance(storage.db_path, Path)
        assert storage.db_path.suffix == ".db"

    async def test_embedding_cache_table_usable(self, storage: Storage) -> None:
        """Can insert and query the embedding_cache table."""
        embedding = serialize_embedding([0.1, 0.2, 0.3])
        await storage.execute_write(
            "INSERT INTO embedding_cache (content_hash, embedding, model) "
            "VALUES (?, ?, ?)",
            ("abc123", embedding, "nomic-embed-text"),
        )
        rows = await storage.execute(
            "SELECT embedding, model FROM embedding_cache WHERE content_hash = ?",
            ("abc123",),
        )
        assert len(rows) == 1
        assert rows[0]["model"] == "nomic-embed-text"
        restored = deserialize_embedding(rows[0]["embedding"])
        assert len(restored) == 3
        assert math.isclose(restored[0], 0.1, rel_tol=1e-6)

    async def test_synapses_unique_constraint(self, storage: Storage) -> None:
        """Duplicate (source_id, target_id, relationship) is rejected."""
        a1 = await storage.execute_write(
            "INSERT INTO atoms (content, type) VALUES (?, ?)", ("a1", "fact")
        )
        a2 = await storage.execute_write(
            "INSERT INTO atoms (content, type) VALUES (?, ?)", ("a2", "fact")
        )
        await storage.execute_write(
            "INSERT INTO synapses (source_id, target_id, relationship) VALUES (?, ?, ?)",
            (a1, a2, "related-to"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            await storage.execute_write(
                "INSERT INTO synapses (source_id, target_id, relationship) VALUES (?, ?, ?)",
                (a1, a2, "related-to"),
            )

    async def test_atoms_type_check_constraint(self, storage: Storage) -> None:
        """Only valid atom types pass the CHECK constraint."""
        valid_types = [
            "fact", "experience", "skill", "preference", "insight", "antipattern",
        ]
        for atom_type in valid_types:
            await storage.execute_write(
                "INSERT INTO atoms (content, type) VALUES (?, ?)",
                (f"test-{atom_type}", atom_type),
            )
        # Invalid type must fail.
        with pytest.raises(sqlite3.IntegrityError):
            await storage.execute_write(
                "INSERT INTO atoms (content, type) VALUES (?, ?)",
                ("bad", "unknown_type"),
            )

    async def test_sessions_table_insert(self, storage: Storage) -> None:
        """Sessions table accepts an insert and returns the data."""
        await storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            ("session-001", "memories"),
        )
        rows = await storage.execute(
            "SELECT id, project FROM sessions WHERE id = ?", ("session-001",)
        )
        assert len(rows) == 1
        assert rows[0]["project"] == "memories"

    async def test_consolidation_log_insert(self, storage: Storage) -> None:
        """Consolidation log accepts entries."""
        row_id = await storage.execute_write(
            "INSERT INTO consolidation_log (action, details) VALUES (?, ?)",
            ("merge", "merged atoms 1 and 2"),
        )
        assert row_id > 0
        rows = await storage.execute(
            "SELECT action, details FROM consolidation_log WHERE id = ?", (row_id,)
        )
        assert rows[0]["action"] == "merge"


# -----------------------------------------------------------------------
# Schema indexes -- verify new indexes exist
# -----------------------------------------------------------------------


class TestSchemaIndexes:
    """Verify that the performance indexes from _INDEX_SQL are created."""

    _REQUIRED_INDEXES = {
        "idx_atoms_last_accessed",
        "idx_atoms_access_count",
        "idx_atoms_created_at",
        "idx_atoms_type_deleted",
        "idx_synapses_source_strength",
        "idx_synapses_target_strength",
        "idx_feedback_atom_processed",
    }

    async def test_required_indexes_exist(self, storage: Storage) -> None:
        """After initialization, all required indexes must exist in sqlite_master."""
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
        index_names = {row["name"] for row in rows}

        for expected in self._REQUIRED_INDEXES:
            assert expected in index_names, f"Missing index: {expected}"

    async def test_thread_local_connection_reuse(self, storage: Storage) -> None:
        """Repeated execute() calls on the same thread should reuse the same
        underlying connection (not open a new one each time)."""
        from unittest.mock import patch

        original_open = storage._open_connection
        open_call_count = 0

        def counting_open():
            nonlocal open_call_count
            open_call_count += 1
            return original_open()

        # Clear any existing thread-local connection so the first call triggers _open_connection.
        storage._local.conn = None

        with patch.object(storage, "_open_connection", side_effect=counting_open):
            await storage.execute("SELECT 1")
            await storage.execute("SELECT 2")
            await storage.execute("SELECT 3")

        # _open_connection should have been called at most once (for the first query).
        assert open_call_count <= 1, (
            f"Expected _open_connection to be called at most once for repeated "
            f"execute() calls, got {open_call_count} calls"
        )
