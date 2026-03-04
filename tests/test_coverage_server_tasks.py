"""Coverage tests for server.py — task management MCP tools.

These tools (create_task, update_task, list_tasks, stale_memories, stats)
were identified as untested in the coverage analysis (server.py: 77%).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memories.brain import Brain
import memories.server as server_module
from memories.server import (
    create_task,
    update_task,
    list_tasks,
    stale_memories,
    stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_brain():
    """Mock Brain with task management methods."""
    brain = MagicMock(spec=Brain)
    brain._initialized = True
    brain.initialize = AsyncMock()

    brain.create_task = AsyncMock(return_value={
        "atom_id": 42,
        "atom": {"id": 42, "content": "Fix the bug", "type": "task", "task_status": "pending"},
        "synapses_created": 2,
        "related_atoms": [],
    })
    brain.update_task = AsyncMock(return_value={
        "task": {"id": 42, "task_status": "done"},
        "linked_memories_flagged": 3,
    })
    brain.get_tasks = AsyncMock(return_value={
        "tasks": [
            {"id": 42, "content": "Fix the bug", "task_status": "pending"},
            {"id": 43, "content": "Write tests", "task_status": "active"},
        ],
        "count": 2,
    })
    brain.get_stale_memories = AsyncMock(return_value={
        "memories": [{"id": 10, "content": "old fact", "type": "fact"}],
        "count": 1,
        "linked_tasks": {"10": [42]},
    })
    brain.get_hook_stats = AsyncMock(return_value={
        "counts_7d": {"prompt-submit": 15},
        "counts_30d": {"prompt-submit": 50},
        "total_atoms": 100,
    })
    return brain


@pytest.fixture(autouse=True)
def patch_brain(mock_brain):
    with patch.object(server_module, "_brain", mock_brain):
        yield mock_brain


# ===========================================================================
# TestCreateTask
# ===========================================================================


class TestCreateTask:
    """Tests for the create_task MCP tool — server.py lines 520-531."""

    async def test_create_task_returns_atom_id(self, mock_brain):
        result = await create_task(content="Fix the bug")
        assert result["atom_id"] == 42

    async def test_create_task_passes_parameters(self, mock_brain):
        await create_task(
            content="Deploy v2",
            region="devops",
            tags=["deploy", "v2"],
            importance=0.9,
            status="active",
        )
        mock_brain.create_task.assert_awaited_once_with(
            content="Deploy v2",
            region="devops",
            tags=["deploy", "v2"],
            importance=0.9,
            status="active",
        )

    async def test_create_task_empty_region_becomes_none(self, mock_brain):
        await create_task(content="Some task", region="")
        _, kwargs = mock_brain.create_task.call_args
        assert kwargs["region"] is None

    async def test_create_task_handles_exception(self, mock_brain):
        mock_brain.create_task = AsyncMock(side_effect=ValueError("bad input"))
        result = await create_task(content="oops")
        assert "error" in result


# ===========================================================================
# TestUpdateTask
# ===========================================================================


class TestUpdateTask:
    """Tests for the update_task MCP tool — server.py lines 559-568."""

    async def test_update_task_returns_result(self, mock_brain):
        result = await update_task(task_id=42, status="done")
        assert result["linked_memories_flagged"] == 3

    async def test_update_task_passes_parameters(self, mock_brain):
        await update_task(task_id=42, status="done", flag_linked_memories=False)
        mock_brain.update_task.assert_awaited_once_with(
            task_id=42,
            status="done",
            flag_linked_memories=False,
        )

    async def test_update_task_handles_exception(self, mock_brain):
        mock_brain.update_task = AsyncMock(side_effect=ValueError("not found"))
        result = await update_task(task_id=99, status="done")
        assert "error" in result


# ===========================================================================
# TestListTasks
# ===========================================================================


class TestListTasks:
    """Tests for the list_tasks MCP tool — server.py lines 590-598."""

    async def test_list_tasks_returns_tasks(self, mock_brain):
        result = await list_tasks()
        assert result["count"] == 2

    async def test_list_tasks_with_status_filter(self, mock_brain):
        await list_tasks(status="pending")
        _, kwargs = mock_brain.get_tasks.call_args
        assert kwargs["status"] == "pending"

    async def test_list_tasks_empty_status_becomes_none(self, mock_brain):
        await list_tasks(status="", region="")
        _, kwargs = mock_brain.get_tasks.call_args
        assert kwargs["status"] is None
        assert kwargs["region"] is None

    async def test_list_tasks_handles_exception(self, mock_brain):
        mock_brain.get_tasks = AsyncMock(side_effect=RuntimeError("db error"))
        result = await list_tasks()
        assert "error" in result


# ===========================================================================
# TestStaleMemories
# ===========================================================================


class TestStaleMemories:
    """Tests for the stale_memories MCP tool — server.py lines 626-633."""

    async def test_stale_memories_returns_result(self, mock_brain):
        result = await stale_memories()
        assert result["count"] == 1

    async def test_stale_memories_passes_min_completed(self, mock_brain):
        await stale_memories(min_completed_tasks=3)
        mock_brain.get_stale_memories.assert_awaited_once_with(
            min_completed_tasks=3,
        )

    async def test_stale_memories_handles_exception(self, mock_brain):
        mock_brain.get_stale_memories = AsyncMock(side_effect=RuntimeError("oops"))
        result = await stale_memories()
        assert "error" in result


# ===========================================================================
# TestStats
# ===========================================================================


class TestStats:
    """Tests for the stats MCP tool — server.py lines 473-478."""

    async def test_stats_returns_result(self, mock_brain):
        result = await stats()
        assert "total_atoms" in result

    async def test_stats_handles_exception(self, mock_brain):
        mock_brain.get_hook_stats = AsyncMock(side_effect=RuntimeError("db error"))
        result = await stats()
        assert "error" in result
