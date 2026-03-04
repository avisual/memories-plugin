"""Tests for the hook_stats observability system.

Covers the hook_stats table schema, the _record_hook_stat helper,
the CLI stats command, and the Brain.get_hook_stats() method.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.storage import Storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_hook_stat(
    storage: Storage,
    hook_type: str = "prompt-submit",
    project: str | None = "myapp",
    query: str | None = "test query",
    atoms_returned: int = 3,
    atom_ids: list[int] | None = None,
    avg_score: float | None = 0.72,
    max_score: float | None = 0.95,
    budget_used: int = 500,
    budget_total: int = 1000,
    compression_level: int = 0,
    seed_count: int = 2,
    total_activated: int = 5,
    novelty_result: str | None = None,
    latency_ms: int = 150,
    created_at: str | None = None,
) -> int:
    """Insert a hook_stats row directly via SQL."""
    row_id = await storage.execute_write(
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
            query,
            atoms_returned,
            json.dumps(atom_ids) if atom_ids else None,
            avg_score,
            max_score,
            budget_used,
            budget_total,
            compression_level,
            seed_count,
            total_activated,
            novelty_result,
            latency_ms,
        ),
    )
    if created_at is not None:
        await storage.execute_write(
            "UPDATE hook_stats SET created_at = ? WHERE id = ?",
            (created_at, row_id),
        )
    return row_id


# ---------------------------------------------------------------------------
# 1. Table existence
# ---------------------------------------------------------------------------


class TestHookStatsTable:
    """Verify hook_stats table is created by Storage.initialize()."""

    async def test_hook_stats_table_exists(self, storage: Storage) -> None:
        """The hook_stats table should exist after initialization."""
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='hook_stats'"
        )
        assert len(rows) == 1
        assert rows[0]["name"] == "hook_stats"

    async def test_hook_stats_indexes_exist(self, storage: Storage) -> None:
        """Both indexes should be created."""
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_hook_stats%'"
        )
        index_names = {r["name"] for r in rows}
        assert "idx_hook_stats_type" in index_names
        assert "idx_hook_stats_created" in index_names


# ---------------------------------------------------------------------------
# 2. Insert and read round-trip
# ---------------------------------------------------------------------------


class TestInsertAndRead:
    """Verify round-trip INSERT + SELECT on hook_stats."""

    async def test_insert_and_read_stat(self, storage: Storage) -> None:
        """A row inserted via SQL should be readable."""
        row_id = await _insert_hook_stat(
            storage,
            hook_type="session-start",
            project="testproj",
            atoms_returned=5,
            atom_ids=[1, 2, 3, 4, 5],
            latency_ms=250,
        )
        assert row_id > 0

        rows = await storage.execute(
            "SELECT * FROM hook_stats WHERE id = ?", (row_id,)
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["hook_type"] == "session-start"
        assert row["project"] == "testproj"
        assert row["atoms_returned"] == 5
        assert row["latency_ms"] == 250
        assert json.loads(row["atom_ids"]) == [1, 2, 3, 4, 5]

    async def test_hook_type_check_constraint(self, storage: Storage) -> None:
        """Invalid hook_type should be rejected by CHECK constraint."""
        import sqlite3
        with pytest.raises(Exception):
            await storage.execute_write(
                """
                INSERT INTO hook_stats (hook_type, latency_ms)
                VALUES ('invalid-type', 100)
                """
            )

    async def test_novelty_result_check_constraint(self, storage: Storage) -> None:
        """Only 'pass', 'fail', or NULL should be accepted for novelty_result."""
        # Valid values should work.
        for val in ("pass", "fail", None):
            await _insert_hook_stat(storage, novelty_result=val)

        # Invalid value should fail.
        with pytest.raises(Exception):
            await storage.execute_write(
                """
                INSERT INTO hook_stats (hook_type, novelty_result, latency_ms)
                VALUES ('post-tool', 'invalid', 100)
                """
            )


# ---------------------------------------------------------------------------
# 3. _record_hook_stat helper
# ---------------------------------------------------------------------------


class TestRecordHookStat:
    """Test the _record_hook_stat helper from cli.py."""

    async def test_record_hook_stat_computes_scores(self, storage: Storage) -> None:
        """avg_score and max_score should be computed from atom dicts."""
        from memories.cli import _record_hook_stat

        # Create a mock brain with the storage.
        brain = MagicMock()
        brain._storage = storage

        result = {
            "atoms": [
                {"id": 1, "score": 0.8},
                {"id": 2, "score": 0.6},
            ],
            "antipatterns": [
                {"id": 3, "score": 0.9},
            ],
            "budget_used": 300,
            "budget_remaining": 700,
            "compression_level": 1,
            "seed_count": 2,
            "total_activated": 5,
        }

        await _record_hook_stat(
            brain, "prompt-submit", 200,
            project="test", query="test query", result=result,
        )

        rows = await storage.execute(
            "SELECT * FROM hook_stats ORDER BY id DESC LIMIT 1"
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["atoms_returned"] == 3
        assert row["atom_ids"] is not None
        ids = json.loads(row["atom_ids"])
        assert set(ids) == {1, 2, 3}
        # avg of 0.8, 0.6, 0.9 = 0.766...
        assert abs(row["avg_score"] - 0.7667) < 0.01
        assert abs(row["max_score"] - 0.9) < 0.001
        assert row["budget_used"] == 300
        assert row["budget_total"] == 1000
        assert row["latency_ms"] == 200

    async def test_record_hook_stat_silent_on_failure(self, storage: Storage) -> None:
        """No exception should propagate when storage is broken."""
        from memories.cli import _record_hook_stat

        brain = MagicMock()
        # Simulate broken storage.
        brain._storage = MagicMock()
        brain._storage.execute_write = AsyncMock(side_effect=Exception("DB error"))

        # Should not raise.
        await _record_hook_stat(brain, "prompt-submit", 100)

    async def test_record_hook_stat_with_no_result(self, storage: Storage) -> None:
        """Recording with result=None should store zeros for atom fields."""
        from memories.cli import _record_hook_stat

        brain = MagicMock()
        brain._storage = storage

        await _record_hook_stat(
            brain, "stop", 50, project="myproj",
        )

        rows = await storage.execute(
            "SELECT * FROM hook_stats ORDER BY id DESC LIMIT 1"
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["hook_type"] == "stop"
        assert row["atoms_returned"] == 0
        assert row["atom_ids"] is None
        assert row["avg_score"] is None
        assert row["latency_ms"] == 50


# ---------------------------------------------------------------------------
# 4. Stats with empty database
# ---------------------------------------------------------------------------


class TestStatsEmpty:
    """Verify stats work on an empty hook_stats table."""

    async def test_get_hook_stats_empty(self, storage: Storage) -> None:
        """get_hook_stats should return valid structure with no data."""
        from memories.brain import Brain

        brain = Brain()
        brain._storage = storage
        brain._initialized = True
        # Mock other components not needed for stats.
        brain._atoms = MagicMock()
        brain._synapses = MagicMock()
        brain._embeddings = MagicMock()

        stats = await brain.get_hook_stats()

        assert stats["counts_7d"] == {}
        assert stats["counts_30d"] == {}
        assert stats["counts_all"] == {}
        assert stats["avg_atoms_returned"] == {}
        assert stats["avg_relevance_score"] is None
        assert stats["unique_atoms_surfaced"] == 0
        assert stats["total_impressions"] == 0
        assert stats["top_atoms"] == []
        assert stats["novelty_stats"]["total"] == 0


# ---------------------------------------------------------------------------
# 5. Stats with sample data
# ---------------------------------------------------------------------------


class TestStatsWithData:
    """Verify stats computation with sample hook_stats rows."""

    async def test_stats_with_sample_data(self, storage: Storage) -> None:
        """Insert sample rows and verify stats sections."""
        from memories.brain import Brain

        # Insert some test atoms so top_atoms lookup works.
        await storage.execute_write(
            "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
            ("Redis SCAN fact", "fact", "technical"),
        )
        await storage.execute_write(
            "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
            ("Redis KEYS warning", "antipattern", "technical"),
        )

        # Insert hook stats.
        await _insert_hook_stat(
            storage, hook_type="session-start", atoms_returned=5,
            atom_ids=[1, 2], avg_score=0.8, budget_used=400, budget_total=1000,
            latency_ms=300,
        )
        await _insert_hook_stat(
            storage, hook_type="prompt-submit", atoms_returned=3,
            atom_ids=[1], avg_score=0.7, budget_used=200, budget_total=1000,
            latency_ms=150,
        )
        await _insert_hook_stat(
            storage, hook_type="post-tool", atoms_returned=0,
            novelty_result="pass", latency_ms=80,
        )
        await _insert_hook_stat(
            storage, hook_type="post-tool", atoms_returned=0,
            novelty_result="fail", latency_ms=90,
        )

        brain = Brain()
        brain._storage = storage
        brain._initialized = True
        brain._atoms = MagicMock()
        brain._synapses = MagicMock()
        brain._embeddings = MagicMock()

        stats = await brain.get_hook_stats()

        # Counts.
        assert stats["counts_all"]["session-start"] == 1
        assert stats["counts_all"]["prompt-submit"] == 1
        assert stats["counts_all"]["post-tool"] == 2

        # Avg atoms returned (only hooks with atoms_returned > 0).
        assert "session-start" in stats["avg_atoms_returned"]

        # Avg relevance score.
        assert stats["avg_relevance_score"] is not None
        assert 0.7 <= stats["avg_relevance_score"] <= 0.8

        # Budget utilisation.
        assert stats["budget_utilisation_pct"] is not None

        # Unique atoms and impressions.
        assert stats["unique_atoms_surfaced"] == 2  # atoms 1 and 2
        assert stats["total_impressions"] == 3  # [1,2] + [1]

        # Top atoms.
        assert len(stats["top_atoms"]) >= 1
        assert stats["top_atoms"][0]["id"] == 1  # appears 2x

        # Novelty.
        assert stats["novelty_stats"]["pass"] == 1
        assert stats["novelty_stats"]["fail"] == 1
        assert stats["novelty_stats"]["total"] == 2

        # Latency.
        assert "session-start" in stats["latency"]
        assert stats["latency"]["session-start"]["avg"] == 300.0

    async def test_stats_top_10_ranking(self, storage: Storage) -> None:
        """Verify most-recalled atoms are ranked by frequency."""
        from memories.brain import Brain

        # Create atoms.
        for i in range(5):
            await storage.execute_write(
                "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
                (f"Atom {i}", "fact", "general"),
            )

        # Atom 1 appears 5 times, atom 2 appears 3 times, atom 3 once.
        for _ in range(5):
            await _insert_hook_stat(storage, atom_ids=[1])
        for _ in range(3):
            await _insert_hook_stat(storage, atom_ids=[2])
        await _insert_hook_stat(storage, atom_ids=[3])

        brain = Brain()
        brain._storage = storage
        brain._initialized = True
        brain._atoms = MagicMock()
        brain._synapses = MagicMock()
        brain._embeddings = MagicMock()

        stats = await brain.get_hook_stats()

        top = stats["top_atoms"]
        assert len(top) >= 3
        assert top[0]["id"] == 1
        assert top[0]["count"] == 5
        assert top[1]["id"] == 2
        assert top[1]["count"] == 3


# ---------------------------------------------------------------------------
# W3-F: get_hook_stats memory fix — SQL aggregation tests
# ---------------------------------------------------------------------------


class TestHookStatsMemoryFix:
    """W3-F: JSON aggregation pushed into SQL via json_each() instead of Python O(N) loop."""

    async def test_hook_stats_uses_sql_aggregation(self, storage: Storage) -> None:
        """The unique-atoms and impressions calculation uses SQL, not O(total_impressions) Python.

        We confirm that the number of storage.execute() calls is small (bounded
        by a constant, not by the number of impressions).  Before W3-F, one
        query fetched ALL atom_ids rows then Python iterated them.  After W3-F
        that query is replaced by a sql json_each() aggregation.
        """
        from memories.brain import Brain
        from unittest.mock import AsyncMock

        # Insert 20 hook_stats rows each with atom_ids.
        for i in range(20):
            await _insert_hook_stat(
                storage,
                hook_type="prompt-submit",
                atom_ids=[i, i + 1, i + 2],
                latency_ms=100,
            )

        brain = Brain()
        brain._storage = storage
        brain._initialized = True
        brain._atoms = MagicMock()
        brain._synapses = MagicMock()
        brain._embeddings = MagicMock()

        # Count execute() calls during get_hook_stats.
        original_execute = storage.execute
        execute_call_count = 0

        async def counting_execute(sql: str, params=()) -> list:
            nonlocal execute_call_count
            execute_call_count += 1
            return await original_execute(sql, params)

        with patch.object(storage, "execute", side_effect=counting_execute):
            stats = await brain.get_hook_stats()

        # The O(N) implementation would issue 1 query for atom_ids rows +
        # potentially N queries per top atom.  The SQL-aggregated version must
        # keep total queries small — we allow up to 15 (generous upper bound)
        # regardless of impression count.
        assert execute_call_count <= 15, (
            f"get_hook_stats issued {execute_call_count} SQL queries for 20 impressions; "
            f"expected <= 15 (SQL aggregation should be O(1), not O(impressions))"
        )

        # Counts must still be correct.
        assert stats["unique_atoms_surfaced"] >= 3
        assert stats["total_impressions"] == 60  # 20 rows * 3 ids each

    async def test_hook_stats_returns_correct_counts(self, storage: Storage) -> None:
        """With known fixture data, unique_atoms_surfaced and total_impressions are correct."""
        from memories.brain import Brain

        # Insert atoms so top_atoms lookup works.
        for i in range(5):
            await storage.execute_write(
                "INSERT INTO atoms (content, type, region) VALUES (?, ?, ?)",
                (f"test atom {i}", "fact", "general"),
            )

        # Row 1: atom_ids [1, 2, 3]  → 3 impressions, 3 unique IDs
        await _insert_hook_stat(storage, hook_type="prompt-submit", atom_ids=[1, 2, 3])
        # Row 2: atom_ids [2, 3, 4]  → 3 impressions, 2 new unique IDs
        await _insert_hook_stat(storage, hook_type="prompt-submit", atom_ids=[2, 3, 4])
        # Row 3: atom_ids [1]        → 1 impression, 0 new unique IDs
        await _insert_hook_stat(storage, hook_type="post-tool", atom_ids=[1])
        # Row 4: None                → 0 impressions
        await _insert_hook_stat(storage, hook_type="stop", atom_ids=None)

        brain = Brain()
        brain._storage = storage
        brain._initialized = True
        brain._atoms = MagicMock()
        brain._synapses = MagicMock()
        brain._embeddings = MagicMock()

        stats = await brain.get_hook_stats()

        assert stats["total_impressions"] == 7, (
            f"Expected 7 total impressions (3+3+1), got {stats['total_impressions']}"
        )
        assert stats["unique_atoms_surfaced"] == 4, (
            f"Expected 4 unique atoms (1,2,3,4), got {stats['unique_atoms_surfaced']}"
        )

    async def test_hook_stats_sql_aggregation_no_python_list_accumulation(
        self, storage: Storage
    ) -> None:
        """The atom_ids rows are NOT fetched wholesale into Python memory.

        After W3-F, no query should SELECT atom_ids FROM hook_stats without
        aggregation — the raw blob-per-row pattern is replaced by sql json_each.
        """
        from memories.brain import Brain

        await _insert_hook_stat(storage, hook_type="prompt-submit", atom_ids=[10, 20])

        brain = Brain()
        brain._storage = storage
        brain._initialized = True
        brain._atoms = MagicMock()
        brain._synapses = MagicMock()
        brain._embeddings = MagicMock()

        # Spy on execute and capture all SQL strings.
        original_execute = storage.execute
        executed_sqls: list[str] = []

        async def capturing_execute(sql: str, params=()) -> list:
            executed_sqls.append(sql.strip())
            return await original_execute(sql, params)

        with patch.object(storage, "execute", side_effect=capturing_execute):
            await brain.get_hook_stats()

        # Detect the old O(N) pattern: raw SELECT atom_ids FROM hook_stats
        # without any json_each() or GROUP BY / COUNT aggregation.
        bad_pattern_found = any(
            "atom_ids" in sql.lower()
            and "json_each" not in sql.lower()
            and "count" not in sql.lower()
            and "group" not in sql.lower()
            for sql in executed_sqls
        )
        assert not bad_pattern_found, (
            "Found a raw SELECT atom_ids FROM hook_stats query without SQL aggregation. "
            "W3-F requires json_each() or COUNT/GROUP BY to push aggregation into SQL.\n"
            f"SQLs issued: {executed_sqls}"
        )
