"""Tests for rejection tracking and consolidation learning."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from memories.storage import Storage


@pytest.fixture
async def storage(tmp_path: Path):
    """Create a temporary storage instance with all tables."""
    db = tmp_path / "test.db"
    store = Storage(db_path=db)
    await store.initialize()
    yield store
    await store.close()


class TestAtomRejectionsTable:
    """Verify the atom_rejections table is created by migrations."""

    async def test_table_exists(self, storage: Storage):
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='atom_rejections'"
        )
        assert len(rows) == 1

    async def test_insert_rejection(self, storage: Storage):
        await storage.execute_write(
            "INSERT INTO atom_rejections (content, reason, hook, source_project) "
            "VALUES (?, ?, ?, ?)",
            ("The system processes data.", "generic_opener", "post-tool", "myproject"),
        )
        rows = await storage.execute("SELECT * FROM atom_rejections")
        assert len(rows) == 1
        assert rows[0]["reason"] == "generic_opener"
        assert rows[0]["hook"] == "post-tool"

    async def test_rejection_index_on_reason(self, storage: Storage):
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_rejections_reason'"
        )
        assert len(rows) == 1


class TestRejectionPatternsTable:
    """Verify the rejection_patterns table is created by migrations."""

    async def test_table_exists(self, storage: Storage):
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rejection_patterns'"
        )
        assert len(rows) == 1

    async def test_insert_pattern(self, storage: Storage):
        await storage.execute_write(
            "INSERT INTO rejection_patterns (pattern, reason, source) VALUES (?, ?, ?)",
            ("delegated to sub-agent", "tool_narration", "learned"),
        )
        rows = await storage.execute("SELECT * FROM rejection_patterns")
        assert len(rows) == 1
        assert rows[0]["pattern"] == "delegated to sub-agent"
        assert rows[0]["source"] == "learned"

    async def test_unique_pattern_constraint(self, storage: Storage):
        await storage.execute_write(
            "INSERT INTO rejection_patterns (pattern, reason) VALUES (?, ?)",
            ("test pattern", "test_reason"),
        )
        # Inserting the same pattern should fail or be ignored with OR IGNORE.
        with pytest.raises(Exception):
            await storage.execute_write(
                "INSERT INTO rejection_patterns (pattern, reason) VALUES (?, ?)",
                ("test pattern", "test_reason"),
            )


class TestLearnFromRejections:
    """Test the consolidation engine's _learn_from_rejections method."""

    async def test_learns_common_patterns(self, storage: Storage):
        """When 10+ rejections share a reason, common substrings are learned."""
        from memories.consolidation import ConsolidationEngine, ConsolidationResult

        # Insert 15 rejections with a common substring.
        for i in range(15):
            await storage.execute_write(
                "INSERT INTO atom_rejections (content, reason, hook) VALUES (?, ?, ?)",
                (f"the agent completed task {i} successfully and reported back", "custom_noise", "post-tool"),
            )

        # Build a minimal engine with just storage.
        engine = ConsolidationEngine.__new__(ConsolidationEngine)
        engine._storage = storage
        engine._cfg = type("cfg", (), {"skill_gen_min_atoms": 0})()

        result = ConsolidationResult()
        await engine._learn_from_rejections(result)

        # Should have learned at least one pattern.
        rows = await storage.execute("SELECT * FROM rejection_patterns")
        assert len(rows) >= 1
        assert result.rejection_patterns_learned >= 1

    async def test_no_learning_below_threshold(self, storage: Storage):
        """Fewer than 10 rejections for a reason should not trigger learning."""
        from memories.consolidation import ConsolidationEngine, ConsolidationResult

        for i in range(5):
            await storage.execute_write(
                "INSERT INTO atom_rejections (content, reason) VALUES (?, ?)",
                (f"some content {i}", "rare_reason"),
            )

        engine = ConsolidationEngine.__new__(ConsolidationEngine)
        engine._storage = storage
        engine._cfg = type("cfg", (), {"skill_gen_min_atoms": 0})()

        result = ConsolidationResult()
        await engine._learn_from_rejections(result)
        assert result.rejection_patterns_learned == 0

    async def test_empty_rejections_no_error(self, storage: Storage):
        """Empty atom_rejections table should not raise errors."""
        from memories.consolidation import ConsolidationEngine, ConsolidationResult

        engine = ConsolidationEngine.__new__(ConsolidationEngine)
        engine._storage = storage
        engine._cfg = type("cfg", (), {"skill_gen_min_atoms": 0})()

        result = ConsolidationResult()
        await engine._learn_from_rejections(result)
        assert result.rejection_patterns_learned == 0


class TestFindCommonSubstrings:
    """Test the static helper for finding common ngrams."""

    def test_finds_common_ngram(self):
        from memories.consolidation import ConsolidationEngine

        contents = [
            "the agent went idle and stopped working",
            "the agent went idle and resumed later",
            "the agent went idle after completing task",
        ]
        result = ConsolidationEngine._find_common_substrings(contents, min_frequency=0.6)
        assert any("agent went idle" in r for r in result)

    def test_no_common_substring(self):
        from memories.consolidation import ConsolidationEngine

        contents = [
            "completely different text here",
            "nothing in common at all",
            "unique content every time",
        ]
        result = ConsolidationEngine._find_common_substrings(contents, min_frequency=0.6)
        assert len(result) == 0

    def test_respects_min_frequency(self):
        from memories.consolidation import ConsolidationEngine

        # Only 2 of 5 share a substring — below 60% threshold.
        contents = [
            "the agent went idle again",
            "the agent went idle once more",
            "something else entirely",
            "another different sentence",
            "yet another unrelated text",
        ]
        result = ConsolidationEngine._find_common_substrings(contents, min_frequency=0.6)
        # "agent went idle" appears in 2/5 = 40%, below 60% threshold.
        assert not any("agent went idle" in r for r in result)
