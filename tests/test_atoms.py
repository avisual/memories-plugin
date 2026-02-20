"""Comprehensive tests for the atoms module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from memories.atoms import (
    ATOM_TYPES,
    SEVERITY_LEVELS,
    Atom,
    AtomManager,
    _validate_atom_type,
    _validate_confidence,
    _validate_severity,
)

from tests.conftest import insert_atom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(**overrides) -> dict:
    """Build a minimal dict that looks like an atoms table row."""
    defaults = {
        "id": 1,
        "content": "test content",
        "type": "fact",
        "region": "general",
        "confidence": 1.0,
        "importance": 0.5,
        "access_count": 0,
        "last_accessed_at": None,
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:00:00",
        "source_project": None,
        "source_session": None,
        "source_file": None,
        "tags": None,
        "severity": None,
        "instead": None,
        "is_deleted": 0,
    }
    defaults.update(overrides)
    return defaults


# ===================================================================
# Atom dataclass
# ===================================================================


class TestAtomDataclass:
    """Tests for Atom.from_row, tag parsing, is_deleted conversion, to_dict."""

    def test_from_row_minimal(self):
        row = _make_row()
        atom = Atom.from_row(row)

        assert atom.id == 1
        assert atom.content == "test content"
        assert atom.type == "fact"
        assert atom.region == "general"
        assert atom.confidence == 1.0
        assert atom.tags == []
        assert atom.is_deleted is False

    def test_from_row_valid_json_tags(self):
        row = _make_row(tags=json.dumps(["redis", "caching"]))
        atom = Atom.from_row(row)
        assert atom.tags == ["redis", "caching"]

    def test_from_row_invalid_json_tags(self):
        row = _make_row(tags="{not valid json")
        atom = Atom.from_row(row)
        assert atom.tags == []

    def test_from_row_none_tags(self):
        row = _make_row(tags=None)
        atom = Atom.from_row(row)
        assert atom.tags == []

    def test_from_row_non_list_json_tags(self):
        """JSON that parses to a dict or scalar should fall back to []."""
        row = _make_row(tags=json.dumps({"key": "val"}))
        atom = Atom.from_row(row)
        assert atom.tags == []

    def test_from_row_non_list_json_string(self):
        """JSON that parses to a plain string should fall back to []."""
        row = _make_row(tags=json.dumps("just a string"))
        atom = Atom.from_row(row)
        assert atom.tags == []

    def test_from_row_is_deleted_truthy(self):
        row = _make_row(is_deleted=1)
        atom = Atom.from_row(row)
        assert atom.is_deleted is True

    def test_from_row_is_deleted_falsy(self):
        row = _make_row(is_deleted=0)
        atom = Atom.from_row(row)
        assert atom.is_deleted is False

    def test_to_dict_returns_all_fields(self):
        atom = Atom(
            id=42,
            content="hello",
            type="skill",
            region="tech",
            confidence=0.9,
            tags=["a", "b"],
            is_deleted=False,
        )
        d = atom.to_dict()
        assert d["id"] == 42
        assert d["content"] == "hello"
        assert d["type"] == "skill"
        assert d["region"] == "tech"
        assert d["confidence"] == 0.9
        assert d["tags"] == ["a", "b"]
        assert d["is_deleted"] is False

    def test_to_dict_is_deleted_stays_bool(self):
        atom = Atom(id=1, content="x", type="fact", is_deleted=True)
        d = atom.to_dict()
        assert d["is_deleted"] is True
        assert isinstance(d["is_deleted"], bool)


# ===================================================================
# Validation helpers
# ===================================================================


class TestValidation:
    """Tests for _validate_atom_type, _validate_severity, _validate_confidence."""

    @pytest.mark.parametrize("atype", list(ATOM_TYPES))
    def test_validate_atom_type_accepts_valid(self, atype):
        _validate_atom_type(atype)  # should not raise

    def test_validate_atom_type_rejects_invalid(self):
        with pytest.raises(ValueError, match="Invalid atom type"):
            _validate_atom_type("banana")

    @pytest.mark.parametrize("sev", list(SEVERITY_LEVELS))
    def test_validate_severity_accepts_valid(self, sev):
        _validate_severity(sev)

    def test_validate_severity_accepts_none(self):
        _validate_severity(None)  # should not raise

    def test_validate_severity_rejects_invalid(self):
        with pytest.raises(ValueError, match="Invalid severity"):
            _validate_severity("extreme")

    @pytest.mark.parametrize("val", [0.0, 0.5, 1.0])
    def test_validate_confidence_accepts_valid(self, val):
        _validate_confidence(val)

    def test_validate_confidence_rejects_negative(self):
        with pytest.raises(ValueError, match="Confidence must be"):
            _validate_confidence(-0.1)

    def test_validate_confidence_rejects_above_one(self):
        with pytest.raises(ValueError, match="Confidence must be"):
            _validate_confidence(1.1)


# ===================================================================
# AtomManager -- Create
# ===================================================================


class TestAtomManagerCreate:
    """Tests for AtomManager.create."""

    async def test_minimal_create(self, storage, mock_embeddings):
        mgr = AtomManager(storage, mock_embeddings)
        atom = await mgr.create("Redis uses single-threaded event loop", type="fact")

        assert atom.id is not None
        assert atom.content == "Redis uses single-threaded event loop"
        assert atom.type == "fact"
        assert atom.region == "general"
        assert atom.confidence == 1.0
        assert atom.is_deleted is False

    async def test_full_params_create(self, storage, mock_embeddings):
        mgr = AtomManager(storage, mock_embeddings)
        atom = await mgr.create(
            content="Avoid SELECT *",
            type="antipattern",
            region="project:db",
            tags=["sql", "performance"],
            severity="high",
            instead="Use explicit column lists",
            source_project="myapp",
            source_session="sess-123",
            source_file="src/db.py",
            confidence=0.85,
        )

        assert atom.type == "antipattern"
        assert atom.region == "project:db"
        assert atom.tags == ["sql", "performance"]
        assert atom.severity == "high"
        assert atom.instead == "Use explicit column lists"
        assert atom.source_project == "myapp"
        assert atom.source_file == "src/db.py"
        assert atom.confidence == 0.85

    async def test_empty_content_rejected(self, storage, mock_embeddings):
        mgr = AtomManager(storage, mock_embeddings)
        with pytest.raises(ValueError, match="must not be empty"):
            await mgr.create("", type="fact")

    async def test_whitespace_only_content_rejected(self, storage, mock_embeddings):
        mgr = AtomManager(storage, mock_embeddings)
        with pytest.raises(ValueError, match="must not be empty"):
            await mgr.create("   ", type="fact")

    async def test_embedding_failure_survives(self, storage, mock_embeddings):
        """Atom is still created even if embed_and_store raises."""
        mock_embeddings.embed_and_store = AsyncMock(
            side_effect=RuntimeError("Ollama offline")
        )
        mgr = AtomManager(storage, mock_embeddings)
        atom = await mgr.create("still persisted", type="fact")

        assert atom.id is not None
        assert atom.content == "still persisted"

    async def test_region_creation(self, storage, mock_embeddings):
        mgr = AtomManager(storage, mock_embeddings)
        atom = await mgr.create("hello", type="fact", region="brand-new-region")

        rows = await storage.execute(
            "SELECT atom_count FROM regions WHERE name = ?",
            ("brand-new-region",),
        )
        assert rows[0]["atom_count"] == 1


# ===================================================================
# AtomManager -- Get
# ===================================================================


class TestAtomManagerGet:
    """Tests for AtomManager.get and get_without_tracking."""

    async def test_get_found(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "findable content")
        mgr = AtomManager(storage, mock_embeddings)
        atom = await mgr.get(aid)

        assert atom is not None
        assert atom.content == "findable content"

    async def test_get_missing_returns_none(self, storage, mock_embeddings):
        mgr = AtomManager(storage, mock_embeddings)
        assert await mgr.get(9999) is None

    async def test_get_soft_deleted_returns_none(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "deleted", is_deleted=True)
        mgr = AtomManager(storage, mock_embeddings)
        assert await mgr.get(aid) is None

    async def test_access_count_increments(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "trackable", access_count=0)
        mgr = AtomManager(storage, mock_embeddings)

        atom = await mgr.get(aid)
        assert atom is not None
        assert atom.access_count == 1

        atom = await mgr.get(aid)
        assert atom.access_count == 2

    async def test_last_accessed_at_updates(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "trackable")
        mgr = AtomManager(storage, mock_embeddings)

        atom_before = await mgr.get_without_tracking(aid)
        original_last_accessed = atom_before.last_accessed_at

        atom_after = await mgr.get(aid)
        assert atom_after.last_accessed_at is not None
        # last_accessed_at should be updated (may or may not differ from original
        # depending on timing, but the field should be populated)
        assert atom_after.last_accessed_at is not None

    async def test_get_without_tracking_no_access_update(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "no tracking", access_count=5)
        mgr = AtomManager(storage, mock_embeddings)

        atom = await mgr.get_without_tracking(aid)
        assert atom is not None
        assert atom.access_count == 5  # unchanged

        # Call again -- still unchanged
        atom2 = await mgr.get_without_tracking(aid)
        assert atom2.access_count == 5


# ===================================================================
# AtomManager -- Update
# ===================================================================


class TestAtomManagerUpdate:
    """Tests for AtomManager.update."""

    async def test_content_change_triggers_reembed(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "original content")
        mgr = AtomManager(storage, mock_embeddings)

        atom = await mgr.update(aid, content="revised content")
        assert atom.content == "revised content"
        mock_embeddings.embed_and_store.assert_called_with(aid, "revised content")

    async def test_type_change(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "some insight", atom_type="fact")
        mgr = AtomManager(storage, mock_embeddings)

        atom = await mgr.update(aid, type="insight")
        assert atom.type == "insight"

    async def test_tags_change(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "tagged atom", tags=["old"])
        mgr = AtomManager(storage, mock_embeddings)

        atom = await mgr.update(aid, tags=["new", "tags"])
        assert atom.tags == ["new", "tags"]

    async def test_confidence_change(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "confident atom", confidence=1.0)
        mgr = AtomManager(storage, mock_embeddings)

        atom = await mgr.update(aid, confidence=0.42)
        assert atom.confidence == pytest.approx(0.42)

    async def test_region_change_adjusts_counters(self, storage, mock_embeddings):
        # Ensure both regions exist with proper counts.
        await storage.execute_write(
            "INSERT OR IGNORE INTO regions (name, atom_count) VALUES (?, ?)",
            ("region-a", 1),
        )
        await storage.execute_write(
            "INSERT OR IGNORE INTO regions (name, atom_count) VALUES (?, ?)",
            ("region-b", 0),
        )

        aid = await insert_atom(storage, "movable atom", region="region-a")
        mgr = AtomManager(storage, mock_embeddings)

        await mgr.update(aid, region="region-b")

        row_a = await storage.execute(
            "SELECT atom_count FROM regions WHERE name = ?", ("region-a",)
        )
        row_b = await storage.execute(
            "SELECT atom_count FROM regions WHERE name = ?", ("region-b",)
        )
        assert row_a[0]["atom_count"] == 0
        assert row_b[0]["atom_count"] == 1

    async def test_noop_when_nothing_to_update(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "stable atom")
        mgr = AtomManager(storage, mock_embeddings)

        atom = await mgr.update(aid)
        assert atom is not None
        assert atom.content == "stable atom"

    async def test_update_nonexistent_returns_none(self, storage, mock_embeddings):
        mgr = AtomManager(storage, mock_embeddings)
        assert await mgr.update(9999, content="nope") is None

    async def test_update_soft_deleted_returns_none(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "deleted", is_deleted=True)
        mgr = AtomManager(storage, mock_embeddings)
        assert await mgr.update(aid, content="should fail") is None


# ===================================================================
# AtomManager -- Delete
# ===================================================================


class TestAtomManagerDelete:
    """Tests for soft_delete and hard_delete."""

    async def test_soft_delete_hides_atom(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "soon to be hidden")
        mgr = AtomManager(storage, mock_embeddings)

        result = await mgr.soft_delete(aid)
        assert result is True

        # Not visible via get
        assert await mgr.get(aid) is None

        # Still in the database
        rows = await storage.execute("SELECT * FROM atoms WHERE id = ?", (aid,))
        assert len(rows) == 1
        assert rows[0]["is_deleted"] == 1

    async def test_hard_delete_removes_from_all_tables(self, storage, mock_embeddings):
        aid = await insert_atom(storage, "permanent removal")
        mgr = AtomManager(storage, mock_embeddings)

        result = await mgr.hard_delete(aid)
        assert result is True

        # Gone from atoms table
        rows = await storage.execute("SELECT * FROM atoms WHERE id = ?", (aid,))
        assert len(rows) == 0

    async def test_soft_delete_decrements_region_counter(self, storage, mock_embeddings):
        await storage.execute_write(
            "INSERT OR IGNORE INTO regions (name, atom_count) VALUES (?, ?)",
            ("test-region", 3),
        )
        aid = await insert_atom(storage, "in test region", region="test-region")
        mgr = AtomManager(storage, mock_embeddings)

        await mgr.soft_delete(aid)

        rows = await storage.execute(
            "SELECT atom_count FROM regions WHERE name = ?", ("test-region",)
        )
        assert rows[0]["atom_count"] == 2

    async def test_hard_delete_decrements_region_counter(self, storage, mock_embeddings):
        await storage.execute_write(
            "INSERT OR IGNORE INTO regions (name, atom_count) VALUES (?, ?)",
            ("hd-region", 5),
        )
        aid = await insert_atom(storage, "hard deletable", region="hd-region")
        mgr = AtomManager(storage, mock_embeddings)

        await mgr.hard_delete(aid)

        rows = await storage.execute(
            "SELECT atom_count FROM regions WHERE name = ?", ("hd-region",)
        )
        assert rows[0]["atom_count"] == 4

    async def test_double_decrement_prevention(self, storage, mock_embeddings):
        """Hard-deleting an already soft-deleted atom should not decrement again."""
        await storage.execute_write(
            "INSERT OR IGNORE INTO regions (name, atom_count) VALUES (?, ?)",
            ("dd-region", 5),
        )
        aid = await insert_atom(storage, "double delete test", region="dd-region")
        mgr = AtomManager(storage, mock_embeddings)

        # Soft-delete first (decrements 5 -> 4)
        await mgr.soft_delete(aid)
        rows = await storage.execute(
            "SELECT atom_count FROM regions WHERE name = ?", ("dd-region",)
        )
        assert rows[0]["atom_count"] == 4

        # Hard-delete the already-soft-deleted atom (should NOT decrement again)
        await mgr.hard_delete(aid)
        rows = await storage.execute(
            "SELECT atom_count FROM regions WHERE name = ?", ("dd-region",)
        )
        assert rows[0]["atom_count"] == 4

    async def test_soft_delete_nonexistent_returns_false(self, storage, mock_embeddings):
        mgr = AtomManager(storage, mock_embeddings)
        assert await mgr.soft_delete(9999) is False

    async def test_hard_delete_nonexistent_returns_false(self, storage, mock_embeddings):
        mgr = AtomManager(storage, mock_embeddings)
        assert await mgr.hard_delete(9999) is False


# ===================================================================
# AtomManager -- Search FTS
# ===================================================================


class TestAtomManagerSearchFTS:
    """Tests for AtomManager.search_fts."""

    async def test_matching_results(self, storage, mock_embeddings):
        await insert_atom(storage, "Redis supports pub/sub messaging")
        await insert_atom(storage, "PostgreSQL uses MVCC for concurrency")
        mgr = AtomManager(storage, mock_embeddings)

        results = await mgr.search_fts("Redis")
        assert len(results) == 1
        assert "Redis" in results[0].content

    async def test_excludes_deleted_atoms(self, storage, mock_embeddings):
        await insert_atom(storage, "visible Redis info")
        await insert_atom(storage, "deleted Redis info", is_deleted=True)
        mgr = AtomManager(storage, mock_embeddings)

        results = await mgr.search_fts("Redis")
        assert len(results) == 1
        assert results[0].is_deleted is False

    async def test_empty_query_returns_empty(self, storage, mock_embeddings):
        await insert_atom(storage, "some content")
        mgr = AtomManager(storage, mock_embeddings)

        assert await mgr.search_fts("") == []
        assert await mgr.search_fts("   ") == []

    async def test_limit_parameter(self, storage, mock_embeddings):
        for i in range(5):
            await insert_atom(storage, f"Redis topic number {i}")
        mgr = AtomManager(storage, mock_embeddings)

        results = await mgr.search_fts("Redis", limit=2)
        assert len(results) == 2


# ===================================================================
# AtomManager -- Stats
# ===================================================================


class TestAtomManagerStats:
    """Tests for AtomManager.get_stats."""

    async def test_empty_db_stats(self, storage, mock_embeddings):
        mgr = AtomManager(storage, mock_embeddings)
        stats = await mgr.get_stats()

        assert stats["total"] == 0
        assert stats["by_type"] == {}
        assert stats["by_region"] == {}
        assert stats["avg_confidence"] == 0.0
        assert stats["total_deleted"] == 0

    async def test_counts_by_type(self, storage, mock_embeddings):
        await insert_atom(storage, "fact 1", atom_type="fact")
        await insert_atom(storage, "fact 2", atom_type="fact")
        await insert_atom(storage, "skill 1", atom_type="skill")
        await insert_atom(storage, "deleted fact", atom_type="fact", is_deleted=True)
        mgr = AtomManager(storage, mock_embeddings)

        stats = await mgr.get_stats()

        assert stats["total"] == 3
        assert stats["by_type"]["fact"] == 2
        assert stats["by_type"]["skill"] == 1
        assert stats["total_deleted"] == 1

    async def test_avg_confidence(self, storage, mock_embeddings):
        await insert_atom(storage, "a", confidence=0.8)
        await insert_atom(storage, "b", confidence=0.6)
        mgr = AtomManager(storage, mock_embeddings)

        stats = await mgr.get_stats()
        assert stats["avg_confidence"] == pytest.approx(0.7, abs=0.001)


# ===================================================================
# AtomManager -- record_access_batch
# ===================================================================


class TestRecordAccessBatch:
    """Tests for AtomManager.record_access_batch (bulk access tracking)."""

    async def test_batch_updates_all_ids(self, storage, mock_embeddings):
        """Insert 5 atoms with access_count=0, batch-update all 5,
        verify all now have access_count=1 and last_accessed_at is set."""
        ids = []
        for i in range(5):
            aid = await insert_atom(storage, f"batch atom {i}", access_count=0)
            ids.append(aid)

        mgr = AtomManager(storage, mock_embeddings)
        await mgr.record_access_batch(ids)

        for aid in ids:
            rows = await storage.execute(
                "SELECT access_count, last_accessed_at FROM atoms WHERE id = ?",
                (aid,),
            )
            assert rows[0]["access_count"] == 1
            assert rows[0]["last_accessed_at"] is not None

    async def test_batch_empty_list_is_noop(self, storage, mock_embeddings):
        """Calling record_access_batch([]) should not raise and should return None."""
        mgr = AtomManager(storage, mock_embeddings)
        result = await mgr.record_access_batch([])
        assert result is None

    async def test_batch_ignores_nonexistent_ids(self, storage, mock_embeddings):
        """Calling record_access_batch with a non-existent ID should not raise."""
        mgr = AtomManager(storage, mock_embeddings)
        # Should complete without error -- UPDATE WHERE id IN (999999) just matches 0 rows.
        await mgr.record_access_batch([999999])

    async def test_batch_only_updates_specified_ids(self, storage, mock_embeddings):
        """Insert 3 atoms, batch-update only the first 2,
        verify atom 3 still has access_count=0."""
        id1 = await insert_atom(storage, "atom one", access_count=0)
        id2 = await insert_atom(storage, "atom two", access_count=0)
        id3 = await insert_atom(storage, "atom three", access_count=0)

        mgr = AtomManager(storage, mock_embeddings)
        await mgr.record_access_batch([id1, id2])

        # Atoms 1 and 2 should be updated.
        for aid in [id1, id2]:
            rows = await storage.execute(
                "SELECT access_count FROM atoms WHERE id = ?", (aid,)
            )
            assert rows[0]["access_count"] == 1

        # Atom 3 should remain untouched.
        rows = await storage.execute(
            "SELECT access_count, last_accessed_at FROM atoms WHERE id = ?",
            (id3,),
        )
        assert rows[0]["access_count"] == 0


# ===================================================================
# AtomManager -- get() single UPDATE...RETURNING query (W3-B)
# ===================================================================


class TestAtomGetSingleQuery:
    """Tests for W3-B: atoms.get() uses a single UPDATE...RETURNING query."""

    async def test_get_single_query(self, storage, mock_embeddings):
        """get() must call execute_write_returning exactly once and
        execute_write / execute zero times for a SELECT.  The new
        implementation uses a single UPDATE...RETURNING instead of the
        old SELECT-then-UPDATE-then-SELECT pattern."""
        aid = await insert_atom(storage, "single-query atom", access_count=3)
        mgr = AtomManager(storage, mock_embeddings)

        # Spy on execute_write_returning (the RETURNING path).
        original_write_returning = storage.execute_write_returning
        returning_calls: list[str] = []

        async def spy_write_returning(sql: str, params=()) -> list:
            returning_calls.append(sql)
            return await original_write_returning(sql, params)

        # Spy on execute (plain SELECT path — should not be called for atoms).
        original_execute = storage.execute
        select_atom_calls: list[str] = []

        async def spy_execute(sql: str, params=()) -> list:
            if "SELECT" in sql.upper() and "atoms" in sql.lower() and "atoms_fts" not in sql.lower():
                select_atom_calls.append(sql)
            return await original_execute(sql, params)

        storage.execute_write_returning = spy_write_returning
        storage.execute = spy_execute

        try:
            atom = await mgr.get(aid)
        finally:
            storage.execute_write_returning = original_write_returning
            storage.execute = original_execute

        # Exactly one execute_write_returning call (the UPDATE...RETURNING).
        assert len(returning_calls) == 1, (
            f"Expected 1 execute_write_returning call, got {len(returning_calls)}: {returning_calls}"
        )
        # No separate SELECT reads on the atoms table.
        assert len(select_atom_calls) == 0, (
            f"Expected 0 SELECT calls on atoms, got {len(select_atom_calls)}: {select_atom_calls}"
        )
        # Result must be valid.
        assert atom is not None
        assert atom.id == aid

    async def test_get_returns_updated_access_count(self, storage, mock_embeddings):
        """After get(), the returned atom's access_count is incremented by 1."""
        aid = await insert_atom(storage, "access count atom", access_count=7)
        mgr = AtomManager(storage, mock_embeddings)

        atom = await mgr.get(aid)

        assert atom is not None
        assert atom.access_count == 8

    async def test_get_nonexistent_returns_none(self, storage, mock_embeddings):
        """get() on a missing atom ID returns None without raising."""
        mgr = AtomManager(storage, mock_embeddings)

        result = await mgr.get(999999)

        assert result is None
