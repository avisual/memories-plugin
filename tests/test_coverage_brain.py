"""Coverage tests for brain.py — status(), update_task(), get_stale_memories().

These methods were identified as untested in the coverage analysis (brain.py: 78%).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from memories.atoms import AtomManager
from memories.brain import Brain
from memories.config import get_config
from memories.synapses import SynapseManager

from tests.conftest import insert_atom, insert_synapse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def brain(storage, mock_embeddings):
    """Brain with real storage + managers, mocked external services."""
    b = Brain.__new__(Brain)
    b._config = get_config()
    b._storage = storage
    b._embeddings = mock_embeddings
    b._atoms = AtomManager(storage, mock_embeddings)
    b._synapses = SynapseManager(storage)
    b._context = MagicMock()
    b._retrieval = MagicMock()
    b._learning = MagicMock()
    b._learning.suggest_region = AsyncMock(return_value="general")
    b._learning.extract_antipattern_fields = AsyncMock(
        return_value=("medium", "do Y instead")
    )
    b._learning.auto_link = AsyncMock(return_value=[])
    b._learning.detect_supersedes = AsyncMock(return_value=0)
    b._consolidation = MagicMock()
    b._current_session_id = "test-session"
    b._initialized = True

    await storage.execute_write(
        "INSERT INTO sessions (id, project) VALUES (?, ?)",
        ("test-session", None),
    )

    yield b
    b._initialized = False


# ===========================================================================
# TestBrainStatus
# ===========================================================================


class TestBrainStatus:
    """Tests for Brain.status() — lines 715-760."""

    async def test_status_returns_all_expected_keys(self, brain):
        """status() returns a dict with all documented keys."""
        result = await brain.status()

        expected_keys = {
            "total_atoms", "total_synapses", "regions",
            "avg_confidence", "stale_atoms", "orphan_atoms",
            "db_size_mb", "embedding_model", "current_session_id",
            "ollama_healthy",
        }
        assert expected_keys.issubset(result.keys())

    async def test_status_empty_db_returns_zeros(self, brain):
        """On an empty database, counts should be zero."""
        result = await brain.status()

        assert result["total_atoms"] == 0
        assert result["total_synapses"] == 0
        assert result["stale_atoms"] == 0
        assert result["orphan_atoms"] == 0

    async def test_status_counts_atoms(self, brain, storage):
        """status() correctly counts atoms after insertion."""
        await insert_atom(storage, "fact one")
        await insert_atom(storage, "fact two")

        result = await brain.status()
        assert result["total_atoms"] == 2

    async def test_status_counts_synapses(self, brain, storage):
        """status() correctly counts synapses."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2)

        result = await brain.status()
        assert result["total_synapses"] >= 1

    async def test_status_reports_orphan_atoms(self, brain, storage):
        """Atoms with no synapses are counted as orphans."""
        await insert_atom(storage, "lonely atom")

        result = await brain.status()
        assert result["orphan_atoms"] == 1

    async def test_status_connected_atom_not_orphan(self, brain, storage):
        """Atoms with synapses are NOT orphans."""
        a1 = await insert_atom(storage, "connected one")
        a2 = await insert_atom(storage, "connected two")
        await insert_synapse(storage, a1, a2)

        result = await brain.status()
        assert result["orphan_atoms"] == 0

    async def test_status_reports_regions(self, brain, storage):
        """status() lists regions with atom counts."""
        await insert_atom(storage, "tech fact", region="technical")
        await insert_atom(storage, "personal pref", region="personal")

        result = await brain.status()
        assert isinstance(result["regions"], list)

    async def test_status_db_size_is_positive(self, brain, storage):
        """Database size should be > 0 after initialization."""
        await insert_atom(storage, "some data")
        result = await brain.status()
        assert result["db_size_mb"] >= 0

    async def test_status_embedding_model_from_config(self, brain):
        """Embedding model should match configuration."""
        result = await brain.status()
        assert result["embedding_model"] == brain._config.embedding_model

    async def test_status_session_id_set(self, brain):
        """Current session ID should be reported."""
        result = await brain.status()
        assert result["current_session_id"] == "test-session"


# ===========================================================================
# TestBrainUpdateTask
# ===========================================================================


class TestBrainUpdateTask:
    """Tests for Brain.update_task() — lines 1247-1292."""

    async def _create_task(self, storage, content="Fix the bug", status="pending"):
        """Helper: insert a task atom directly."""
        return await insert_atom(
            storage, content, atom_type="task", region="tasks",
        )

    async def test_update_task_changes_status(self, brain, storage):
        """update_task() changes the task status."""
        task_id = await self._create_task(storage)
        # Set task_status in DB (insert_atom doesn't set it).
        await storage.execute_write(
            "UPDATE atoms SET task_status = 'pending' WHERE id = ?", (task_id,)
        )

        result = await brain.update_task(task_id, status="active")
        assert result["task"]["task_status"] == "active"

    async def test_update_task_flags_linked_memories_on_done(self, brain, storage):
        """Completing a task reduces confidence of linked non-task atoms."""
        task_id = await self._create_task(storage)
        await storage.execute_write(
            "UPDATE atoms SET task_status = 'pending' WHERE id = ?", (task_id,)
        )
        linked_id = await insert_atom(storage, "related knowledge", confidence=0.8)
        await insert_synapse(storage, task_id, linked_id)

        result = await brain.update_task(task_id, status="done", flag_linked_memories=True)

        assert result["linked_memories_flagged"] >= 1

        # Verify confidence was reduced.
        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (linked_id,)
        )
        assert rows[0]["confidence"] < 0.8

    async def test_update_task_no_flag_when_not_done(self, brain, storage):
        """Setting status to 'active' does NOT flag linked memories."""
        task_id = await self._create_task(storage)
        await storage.execute_write(
            "UPDATE atoms SET task_status = 'pending' WHERE id = ?", (task_id,)
        )
        linked_id = await insert_atom(storage, "related knowledge", confidence=0.8)
        await insert_synapse(storage, task_id, linked_id)

        result = await brain.update_task(task_id, status="active", flag_linked_memories=True)

        assert result["linked_memories_flagged"] == 0

        # Confidence unchanged.
        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (linked_id,)
        )
        assert rows[0]["confidence"] == pytest.approx(0.8)

    async def test_update_task_not_found_raises(self, brain):
        """update_task() raises ValueError for non-existent task."""
        with pytest.raises(ValueError, match="not found"):
            await brain.update_task(99999, status="done")


# ===========================================================================
# TestBrainGetStalememories
# ===========================================================================


class TestBrainGetStaleMemories:
    """Tests for Brain.get_stale_memories() — lines 1347-1414."""

    async def test_no_completed_tasks_returns_empty(self, brain):
        """No completed tasks → no stale memories."""
        result = await brain.get_stale_memories()
        assert result["count"] == 0
        assert result["memories"] == []

    async def test_returns_memories_linked_to_completed_task(self, brain, storage):
        """Memories linked to done tasks are flagged as stale."""
        task_id = await insert_atom(storage, "Build feature X", atom_type="task")
        await storage.execute_write(
            "UPDATE atoms SET task_status = 'done' WHERE id = ?", (task_id,)
        )
        mem_id = await insert_atom(storage, "Feature X uses REST API", atom_type="fact")
        await insert_synapse(storage, task_id, mem_id)

        result = await brain.get_stale_memories(min_completed_tasks=1)

        assert result["count"] >= 1
        stale_ids = [m["id"] for m in result["memories"]]
        assert mem_id in stale_ids

    async def test_min_completed_tasks_filter(self, brain, storage):
        """Increasing min_completed_tasks filters out memories with fewer links."""
        task1 = await insert_atom(storage, "Task 1", atom_type="task")
        await storage.execute_write(
            "UPDATE atoms SET task_status = 'done' WHERE id = ?", (task1,)
        )
        mem_id = await insert_atom(storage, "Some knowledge", atom_type="fact")
        await insert_synapse(storage, task1, mem_id)

        # With min_completed_tasks=1, the memory is stale.
        result1 = await brain.get_stale_memories(min_completed_tasks=1)
        assert result1["count"] >= 1

        # With min_completed_tasks=2, it's NOT stale (only linked to 1 task).
        result2 = await brain.get_stale_memories(min_completed_tasks=2)
        stale_ids = [m["id"] for m in result2["memories"]]
        assert mem_id not in stale_ids

    async def test_linked_tasks_mapping(self, brain, storage):
        """The linked_tasks dict maps memory ID → list of completed task IDs."""
        task_id = await insert_atom(storage, "Deploy v2", atom_type="task")
        await storage.execute_write(
            "UPDATE atoms SET task_status = 'archived' WHERE id = ?", (task_id,)
        )
        mem_id = await insert_atom(storage, "v2 deployment uses Docker", atom_type="fact")
        await insert_synapse(storage, task_id, mem_id)

        result = await brain.get_stale_memories()

        linked = result["linked_tasks"]
        assert str(mem_id) in linked
        assert task_id in linked[str(mem_id)]

    async def test_task_atoms_not_in_stale_memories(self, brain, storage):
        """Task atoms themselves should not appear in stale memories."""
        task1 = await insert_atom(storage, "Task A", atom_type="task")
        task2 = await insert_atom(storage, "Task B", atom_type="task")
        await storage.execute_write(
            "UPDATE atoms SET task_status = 'done' WHERE id IN (?, ?)",
            (task1, task2),
        )
        await insert_synapse(storage, task1, task2)

        result = await brain.get_stale_memories()
        stale_types = [m["type"] for m in result["memories"]]
        assert "task" not in stale_types
