"""Load and concurrency tests for the memories system.

These tests validate that the concurrency hardening in storage, atoms,
synapses, and consolidation behaves correctly under concurrent access.
All tests are marked with ``@pytest.mark.load`` so they can be run or
skipped independently.

Run only load tests::

    uv run pytest -m load

Skip load tests::

    uv run pytest -m "not load"
"""

from __future__ import annotations

import asyncio
import multiprocessing
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from memories.atoms import AtomManager
from memories.consolidation import ConsolidationEngine
from memories.embeddings import EmbeddingEngine
from memories.storage import Storage
from memories.synapses import SynapseManager

pytestmark = pytest.mark.load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_embeddings() -> MagicMock:
    engine = MagicMock(spec=EmbeddingEngine)
    engine.search_similar = AsyncMock(return_value=[])
    engine.embed_text = AsyncMock(return_value=[0.0] * 768)
    engine.embed_and_store = AsyncMock(return_value=[0.0] * 768)
    engine.embed_batch = AsyncMock(return_value=[])
    engine.health_check = AsyncMock(return_value=True)
    engine.cosine_similarity = MagicMock(return_value=0.5)  # sync function (M-6)
    return engine


def _worker_insert_atoms(db_path: str, n: int, worker_id: int) -> int:
    """Subprocess worker: insert *n* atoms using raw sqlite3.

    Returns the number of successfully inserted atoms.
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")

    inserted = 0
    for i in range(n):
        try:
            conn.execute(
                """
                INSERT INTO atoms
                    (content, type, region, confidence, is_deleted)
                VALUES (?, 'fact', 'test', 1.0, 0)
                """,
                (f"worker-{worker_id}-atom-{i}",),
            )
            conn.commit()
            inserted += 1
        except sqlite3.OperationalError:
            # Busy / locked — retry once after a short sleep.
            time.sleep(0.01)
            try:
                conn.execute(
                    """
                    INSERT INTO atoms
                        (content, type, region, confidence, is_deleted)
                    VALUES (?, 'fact', 'test', 1.0, 0)
                    """,
                    (f"worker-{worker_id}-atom-{i}-retry",),
                )
                conn.commit()
                inserted += 1
            except sqlite3.OperationalError:
                pass  # accept the loss under extreme contention

    conn.close()
    return inserted


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestMultiProcessWrites:
    """Spawn N subprocesses each inserting M atoms; verify total = N*M."""

    NUM_WORKERS = 4
    ATOMS_PER_WORKER = 25

    async def test_multi_process_no_corruption(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mp_test.db"
        store = Storage(db_path)
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        await store.close()

        with multiprocessing.Pool(self.NUM_WORKERS) as pool:
            results = pool.starmap(
                _worker_insert_atoms,
                [
                    (str(db_path), self.ATOMS_PER_WORKER, wid)
                    for wid in range(self.NUM_WORKERS)
                ],
            )

        total_inserted = sum(results)

        # Verify by reading back.
        conn = sqlite3.connect(str(db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM atoms WHERE is_deleted = 0"
        ).fetchone()[0]
        conn.close()

        assert count == total_inserted
        # We expect all inserts to succeed (WAL + busy_timeout should handle it).
        assert total_inserted == self.NUM_WORKERS * self.ATOMS_PER_WORKER

    async def test_no_database_corruption(self, tmp_path: Path) -> None:
        """Integrity check after multi-process writes."""
        db_path = tmp_path / "integrity_test.db"
        store = Storage(db_path)
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        await store.close()

        with multiprocessing.Pool(2) as pool:
            pool.starmap(
                _worker_insert_atoms,
                [(str(db_path), 20, wid) for wid in range(2)],
            )

        conn = sqlite3.connect(str(db_path))
        result = conn.execute("PRAGMA integrity_check(1)").fetchone()[0]
        conn.close()
        assert result == "ok"


class TestConcurrentAsyncOps:
    """asyncio.gather of interleaved creates + reads within a single process."""

    async def test_interleaved_creates_and_reads(self, tmp_path: Path) -> None:
        store = Storage(tmp_path / "async_test.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()

        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)

        async def create_and_read(idx: int) -> None:
            atom = await atoms.create(
                content=f"concurrent atom {idx}",
                type="fact",
                region="concurrent",
            )
            fetched = await atoms.get(atom.id)
            assert fetched is not None
            assert fetched.content == f"concurrent atom {idx}"

        await asyncio.gather(*[create_and_read(i) for i in range(20)])

        rows = await store.execute(
            "SELECT COUNT(*) AS cnt FROM atoms WHERE is_deleted = 0 AND region = 'concurrent'"
        )
        assert rows[0]["cnt"] == 20

        await store.close()

    async def test_concurrent_synapse_upserts(self, tmp_path: Path) -> None:
        store = Storage(tmp_path / "syn_upsert.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()

        mock_emb = _make_mock_embeddings()
        atom_mgr = AtomManager(store, mock_emb)
        syn_mgr = SynapseManager(store)

        a1 = await atom_mgr.create(content="node A", type="fact")
        a2 = await atom_mgr.create(content="node B", type="fact")

        async def upsert(_: int) -> None:
            await syn_mgr.create(
                source_id=a1.id,
                target_id=a2.id,
                relationship="related-to",
            )

        # 10 concurrent upserts on the same triple.
        await asyncio.gather(*[upsert(i) for i in range(10)])

        rows = await store.execute(
            "SELECT COUNT(*) AS cnt FROM synapses "
            "WHERE source_id = ? AND target_id = ? AND relationship = 'related-to'",
            (a1.id, a2.id),
        )
        # Exactly one synapse should exist (no duplicates).
        assert rows[0]["cnt"] == 1

        await store.close()


class TestHookLatency:
    """Insert many atoms, then verify query time stays reasonable."""

    ATOM_COUNT = 2000
    MAX_QUERY_SECONDS = 0.5

    async def test_recall_latency_under_load(self, tmp_path: Path) -> None:
        store = Storage(tmp_path / "latency_test.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()

        # Bulk-insert atoms directly via SQL for speed.
        params = [
            (f"latency test atom number {i}", "fact", "perf", 1.0, 0)
            for i in range(self.ATOM_COUNT)
        ]
        await store.execute_many(
            """
            INSERT INTO atoms (content, type, region, confidence, is_deleted)
            VALUES (?, ?, ?, ?, ?)
            """,
            params,
        )

        # Verify count.
        rows = await store.execute(
            "SELECT COUNT(*) AS cnt FROM atoms WHERE is_deleted = 0 AND region = 'perf'"
        )
        assert rows[0]["cnt"] == self.ATOM_COUNT

        # Time a FTS query.
        start = time.monotonic()
        await store.execute(
            """
            SELECT a.* FROM atoms_fts f
            JOIN atoms a ON a.id = f.rowid
            WHERE atoms_fts MATCH 'latency'
              AND a.is_deleted = 0
            LIMIT 20
            """,
        )
        elapsed = time.monotonic() - start

        assert elapsed < self.MAX_QUERY_SECONDS, (
            f"FTS query took {elapsed:.3f}s, exceeding {self.MAX_QUERY_SECONDS}s threshold"
        )

        # Time a region-filtered query.
        start = time.monotonic()
        await store.execute(
            """
            SELECT * FROM atoms
            WHERE region = 'perf' AND is_deleted = 0
            ORDER BY confidence DESC, created_at DESC
            LIMIT 50
            """,
        )
        elapsed = time.monotonic() - start

        assert elapsed < self.MAX_QUERY_SECONDS, (
            f"Region query took {elapsed:.3f}s, exceeding {self.MAX_QUERY_SECONDS}s threshold"
        )

        await store.close()


class TestConsolidationConcurrency:
    """Two concurrent reflect() calls — mutex prevents double-execution."""

    async def test_concurrent_reflect_mutex(self, tmp_path: Path) -> None:
        store = Storage(tmp_path / "consol_mutex.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()

        mock_emb = _make_mock_embeddings()
        atom_mgr = AtomManager(store, mock_emb)
        syn_mgr = SynapseManager(store)
        engine = ConsolidationEngine(store, mock_emb, atom_mgr, syn_mgr)

        # Insert a few atoms so consolidation has something to process.
        for i in range(5):
            await atom_mgr.create(
                content=f"mutex test atom {i}", type="fact", region="mutex"
            )

        # Launch two concurrent consolidation runs.
        results = await asyncio.gather(
            engine.reflect(dry_run=True),
            engine.reflect(dry_run=True),
        )

        # At least one should have run (the other may have been skipped).
        ran_count = sum(1 for r in results if r.dry_run)
        assert ran_count >= 1

        # Verify lock is released after completion.
        rows = await store.execute(
            "SELECT COUNT(*) AS cnt FROM locks WHERE name = 'consolidation'"
        )
        assert rows[0]["cnt"] == 0, "Consolidation lock was not released"

        await store.close()


class TestRegionCounterConsistency:
    """Create atoms, concurrently delete some, verify region counter == actual count."""

    async def test_region_counter_matches_actual(self, tmp_path: Path) -> None:
        store = Storage(tmp_path / "region_count.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()

        mock_emb = _make_mock_embeddings()
        atom_mgr = AtomManager(store, mock_emb)

        total = 20
        delete_count = 10

        atom_ids: list[int] = []
        for i in range(total):
            atom = await atom_mgr.create(
                content=f"region counter atom {i}",
                type="fact",
                region="counter-test",
            )
            atom_ids.append(atom.id)

        # Verify initial region counter.
        rows = await store.execute(
            "SELECT atom_count FROM regions WHERE name = 'counter-test'"
        )
        assert rows[0]["atom_count"] == total

        # Concurrently delete half the atoms.
        ids_to_delete = atom_ids[:delete_count]
        await asyncio.gather(
            *[atom_mgr.soft_delete(aid) for aid in ids_to_delete]
        )

        # Verify region counter matches actual count.
        rows = await store.execute(
            "SELECT atom_count FROM regions WHERE name = 'counter-test'"
        )
        region_counter = rows[0]["atom_count"]

        actual_rows = await store.execute(
            "SELECT COUNT(*) AS cnt FROM atoms "
            "WHERE region = 'counter-test' AND is_deleted = 0"
        )
        actual_count = actual_rows[0]["cnt"]

        expected = total - delete_count
        assert actual_count == expected, (
            f"Expected {expected} active atoms, got {actual_count}"
        )
        assert region_counter == expected, (
            f"Region counter is {region_counter}, expected {expected}"
        )

        await store.close()
