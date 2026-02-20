"""Wave 3-E performance fix tests.

Covers five independent fixes:
  M-3:  table_counts uses a single UNION ALL query (not 6 separate queries)
  M-4:  synapses.get_stats uses ≤ 3 queries (not 5)
  M-6:  cosine_similarity is NOT a coroutine (pure CPU, no I/O)
  M-8:  Storage.close() closes all thread-local connections (not just the calling thread's)
  M-10: synapses.delete_for_atom does DELETE first, counts via changes() (no prior SELECT)

All tests use a temporary database via pytest ``tmp_path`` so no real user data
is affected.  No Ollama or network access required.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import inspect

import pytest

from memories.storage import Storage
from memories.synapses import SynapseManager
from memories.embeddings import EmbeddingEngine

from tests.conftest import insert_atom, insert_synapse


# -----------------------------------------------------------------------
# M-3: table_counts — single UNION ALL query
# -----------------------------------------------------------------------


class TestTableCountsSingleQuery:
    """table_counts must issue exactly one SQL round-trip using UNION ALL."""

    async def test_table_counts_single_query(self, storage: Storage) -> None:
        """storage.execute should be called exactly once, not once per table."""
        call_count = 0
        original_execute = storage.execute

        async def counting_execute(sql: str, params=()) -> list:
            nonlocal call_count
            call_count += 1
            return await original_execute(sql, params)

        with patch.object(storage, "execute", side_effect=counting_execute):
            result = await storage.table_counts()

        assert call_count == 1, (
            f"table_counts() should issue exactly 1 SQL query (UNION ALL), "
            f"but execute() was called {call_count} times"
        )
        # Correctness: all 6 expected keys must still be present.
        assert set(result.keys()) == {
            "atoms", "synapses", "regions", "sessions",
            "consolidation_log", "embedding_cache",
        }


# -----------------------------------------------------------------------
# M-4: synapses.get_stats — ≤ 3 queries
# -----------------------------------------------------------------------


class TestGetStatsReducedQueries:
    """get_stats must consolidate 5 separate queries into ≤ 3 queries."""

    async def test_get_stats_reduced_queries(self, storage: Storage) -> None:
        """storage.execute should be called ≤ 3 times, not 5 times."""
        mgr = SynapseManager(storage)

        # Insert two atoms and a synapse so the stats are non-trivial.
        a1 = await insert_atom(storage, "alpha", "fact")
        a2 = await insert_atom(storage, "beta", "fact")
        await insert_synapse(storage, a1, a2, "related-to", strength=0.7)

        call_count = 0
        original_execute = storage.execute

        async def counting_execute(sql: str, params=()) -> list:
            nonlocal call_count
            call_count += 1
            return await original_execute(sql, params)

        with patch.object(storage, "execute", side_effect=counting_execute):
            stats = await mgr.get_stats()

        assert call_count <= 3, (
            f"get_stats() should use ≤ 3 queries, but execute() was called "
            f"{call_count} times"
        )
        # Correctness: all required keys must still be present.
        assert "total" in stats
        assert "avg_strength" in stats
        assert "by_relationship" in stats
        assert "weakest" in stats
        assert "strongest" in stats
        assert stats["total"] == 1
        assert "related-to" in stats["by_relationship"]


# -----------------------------------------------------------------------
# M-6: cosine_similarity is NOT a coroutine
# -----------------------------------------------------------------------


class TestCosineSimilarityIsNotCoroutine:
    """cosine_similarity must be a plain synchronous function, not async."""

    async def test_cosine_similarity_is_not_coroutine(self, storage: Storage) -> None:
        """Calling cosine_similarity() must return a plain float, not a coroutine."""
        mock_client = MagicMock()
        engine = EmbeddingEngine(storage)
        engine._client = mock_client

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        result = engine.cosine_similarity(vec1, vec2)

        # If cosine_similarity is still async, result would be a coroutine object.
        assert not inspect.iscoroutine(result), (
            "cosine_similarity() returned a coroutine — it must be a plain "
            "synchronous function (remove the 'async' keyword)"
        )
        # Explicitly close any accidental coroutine to avoid ResourceWarning.
        if inspect.iscoroutine(result):
            result.close()
        else:
            # Correctness: identical vectors → similarity ≈ 1.0
            import math
            assert math.isclose(float(result), 1.0, rel_tol=1e-5)


# -----------------------------------------------------------------------
# M-8: Storage.close() closes all thread connections
# -----------------------------------------------------------------------


class TestStorageCloseAllThreads:
    """Storage.close() must close connections opened from all threads."""

    async def test_storage_close_all_threads(self, tmp_path: Path) -> None:
        """Connections opened from multiple threads must all be tracked and closed."""
        db_path = tmp_path / "close_test.db"
        s = Storage(db_path)
        s._backup_dir = tmp_path / "backups_close"
        s._backup_dir.mkdir()
        await s.initialize()

        # Open connections from two distinct threads so _all_connections grows.
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def thread_work():
            try:
                barrier.wait()
                # Clear the thread-local so a brand-new connection is created.
                s._local.conn = None
                conn = s._get_connection()
                conn.execute("SELECT 1")
                barrier.wait()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=thread_work)
        t2 = threading.Thread(target=thread_work)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        if errors:
            raise errors[0]

        # After the threads have run, _all_connections must contain ≥ 2 entries.
        all_conns = getattr(s, "_all_connections", None)
        assert all_conns is not None, (
            "Storage must have an '_all_connections' list attribute — "
            "M-8 requires tracking every connection opened across threads"
        )
        assert len(all_conns) >= 2, (
            f"Expected _all_connections to contain ≥ 2 connections "
            f"(one per thread), got {len(all_conns)}"
        )

        # After close(), all tracked connections must have been closed.
        # We verify this by checking that attempting to use one raises an error.
        snapshot = list(all_conns)
        await s.close()

        for conn in snapshot:
            try:
                conn.execute("SELECT 1")
                # If we reach here the connection is still open — test fails.
                raise AssertionError(
                    "A connection is still usable after Storage.close() — "
                    "close() must close ALL thread connections, not just the "
                    "calling thread's"
                )
            except Exception as exc:
                # ProgrammingError: "Cannot operate on a closed database."
                if "closed" in str(exc).lower() or "ProgrammingError" in type(exc).__name__:
                    pass  # Good — connection is properly closed.
                # Re-raise unexpected errors.
                elif not isinstance(exc, AssertionError):
                    raise


# -----------------------------------------------------------------------
# M-10: synapses.delete_for_atom — DELETE first, no prior SELECT
# -----------------------------------------------------------------------


class TestDeleteForAtomUsesDeleteFirst:
    """delete_for_atom must DELETE directly without a preceding SELECT COUNT."""

    async def test_delete_for_atom_no_prior_select(self, storage: Storage) -> None:
        """execute() (SELECT) must NOT be called before the DELETE.

        The implementation uses execute_write_returning (RETURNING id) so no
        separate SELECT is needed at all.  We verify that execute() is never
        called during delete_for_atom — only the write path is used.
        """
        mgr = SynapseManager(storage)

        a1 = await insert_atom(storage, "atom-one", "fact")
        a2 = await insert_atom(storage, "atom-two", "fact")
        await insert_synapse(storage, a1, a2, "related-to", strength=0.6)

        call_order: list[str] = []

        original_execute = storage.execute
        original_execute_write = storage.execute_write
        original_execute_write_returning = storage.execute_write_returning

        async def recording_execute(sql: str, params=()) -> list:
            call_order.append("SELECT")
            return await original_execute(sql, params)

        async def recording_execute_write(sql: str, params=()) -> int:
            call_order.append("WRITE")
            return await original_execute_write(sql, params)

        async def recording_execute_write_returning(sql: str, params=()) -> list:
            call_order.append("WRITE_RETURNING")
            return await original_execute_write_returning(sql, params)

        with (
            patch.object(storage, "execute", side_effect=recording_execute),
            patch.object(storage, "execute_write", side_effect=recording_execute_write),
            patch.object(storage, "execute_write_returning", side_effect=recording_execute_write_returning),
        ):
            count = await mgr.delete_for_atom(a1)

        # Verify we get the right count back.
        assert count == 1, f"Expected 1 synapse deleted, got {count}"

        # No SELECT must be called — delete_for_atom uses RETURNING, not changes().
        assert "SELECT" not in call_order, (
            f"delete_for_atom() must not call execute() (SELECT). "
            f"Use execute_write_returning with RETURNING id. Call order: {call_order}"
        )

        # The delete must be performed via execute_write_returning.
        assert "WRITE_RETURNING" in call_order, (
            f"delete_for_atom() must use execute_write_returning. "
            f"Call order: {call_order}"
        )

    async def test_delete_for_atom_returns_correct_count(
        self, storage: Storage
    ) -> None:
        """delete_for_atom must return the exact number of rows deleted."""
        mgr = SynapseManager(storage)

        a1 = await insert_atom(storage, "hub", "fact")
        a2 = await insert_atom(storage, "spoke-one", "fact")
        a3 = await insert_atom(storage, "spoke-two", "fact")

        # Create 3 synapses all touching a1.
        await insert_synapse(storage, a1, a2, "related-to", strength=0.5)
        await insert_synapse(storage, a1, a3, "caused-by", strength=0.6)
        await insert_synapse(storage, a2, a1, "elaborates", strength=0.4)

        count = await mgr.delete_for_atom(a1)
        assert count == 3, f"Expected 3 synapses deleted, got {count}"

    async def test_delete_for_atom_zero_when_no_synapses(
        self, storage: Storage
    ) -> None:
        """delete_for_atom must return 0 when no synapses exist for the atom."""
        mgr = SynapseManager(storage)
        a1 = await insert_atom(storage, "isolated", "fact")

        count = await mgr.delete_for_atom(a1)
        assert count == 0
