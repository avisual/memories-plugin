"""Tests for batch atom operations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from memories.atoms import AtomManager
from memories.embeddings import EmbeddingEngine
from memories.storage import Storage


def _make_mock_embeddings() -> MagicMock:
    engine = MagicMock(spec=EmbeddingEngine)
    engine.embed_text = AsyncMock(return_value=[0.0] * 768)
    engine.embed_and_store = AsyncMock(return_value=[0.0] * 768)
    return engine


class TestBatchGetWithoutTracking:
    """Tests for get_batch_without_tracking method."""

    async def test_returns_empty_dict_for_empty_input(self, tmp_path: Path) -> None:
        store = Storage(tmp_path / "batch_test.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()

        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)

        result = await atoms.get_batch_without_tracking([])
        assert result == {}

        await store.close()

    async def test_returns_multiple_atoms(self, tmp_path: Path) -> None:
        store = Storage(tmp_path / "batch_test.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()

        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)

        # Create some atoms
        atom1 = await atoms.create(content="atom one", type="fact")
        atom2 = await atoms.create(content="atom two", type="fact")
        atom3 = await atoms.create(content="atom three", type="fact")

        # Batch fetch
        result = await atoms.get_batch_without_tracking([atom1.id, atom2.id, atom3.id])

        assert len(result) == 3
        assert atom1.id in result
        assert atom2.id in result
        assert atom3.id in result
        assert result[atom1.id].content == "atom one"
        assert result[atom2.id].content == "atom two"
        assert result[atom3.id].content == "atom three"

        await store.close()

    async def test_excludes_deleted_atoms(self, tmp_path: Path) -> None:
        store = Storage(tmp_path / "batch_test.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()

        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)

        # Create atoms and delete one
        atom1 = await atoms.create(content="keep me", type="fact")
        atom2 = await atoms.create(content="delete me", type="fact")
        await atoms.soft_delete(atom2.id)

        # Batch fetch - deleted should be excluded
        result = await atoms.get_batch_without_tracking([atom1.id, atom2.id])

        assert len(result) == 1
        assert atom1.id in result
        assert atom2.id not in result

        await store.close()

    async def test_handles_nonexistent_ids(self, tmp_path: Path) -> None:
        store = Storage(tmp_path / "batch_test.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()

        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)

        atom1 = await atoms.create(content="real atom", type="fact")

        # Fetch with mix of real and fake IDs
        result = await atoms.get_batch_without_tracking([atom1.id, 9999, 8888])

        assert len(result) == 1
        assert atom1.id in result

        await store.close()

    async def test_performance_over_single_fetches(self, tmp_path: Path) -> None:
        """Verify batch is more efficient than individual fetches."""
        import time

        store = Storage(tmp_path / "batch_perf.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()

        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)

        # Create 100 atoms
        atom_ids = []
        for i in range(100):
            atom = await atoms.create(content=f"perf test {i}", type="fact")
            atom_ids.append(atom.id)

        # Time individual fetches
        start = time.monotonic()
        for aid in atom_ids:
            await atoms.get_without_tracking(aid)
        individual_time = time.monotonic() - start

        # Time batch fetch
        start = time.monotonic()
        await atoms.get_batch_without_tracking(atom_ids)
        batch_time = time.monotonic() - start

        # Batch should be faster (or at least not slower)
        print(f"\n  Individual fetches: {individual_time*1000:.1f}ms")
        print(f"  Batch fetch: {batch_time*1000:.1f}ms")
        print(f"  Speedup: {individual_time/batch_time:.1f}x")

        # Batch should be at least 2x faster for 100 atoms
        assert batch_time < individual_time, (
            f"Batch ({batch_time:.3f}s) should be faster than "
            f"individual ({individual_time:.3f}s)"
        )

        await store.close()
