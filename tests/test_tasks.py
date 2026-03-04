"""Tests for the task management system.

Tests cover:
- Task creation with auto-status
- Task status updates
- Flagging linked memories when tasks complete
- Querying stale memories linked to completed tasks
"""

from __future__ import annotations

import pytest

from memories.atoms import Atom, AtomManager, TASK_STATUSES
from memories.storage import Storage
from memories.synapses import SynapseManager


# ---------------------------------------------------------------------------
# Additional fixtures for atom_manager
# ---------------------------------------------------------------------------


@pytest.fixture
async def atom_manager(storage, mock_embeddings):
    """Create an AtomManager with mocked embeddings."""
    return AtomManager(storage, mock_embeddings)


@pytest.fixture
async def synapse_manager(storage):
    """Create a SynapseManager."""
    return SynapseManager(storage)


# ---------------------------------------------------------------------------
# Helper to make a row dict
# ---------------------------------------------------------------------------


def _make_task_row(**overrides) -> dict:
    """Build a minimal dict that looks like an atoms table row with task_status."""
    defaults = {
        "id": 1,
        "content": "test task",
        "type": "task",
        "region": "tasks",
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
        "task_status": "pending",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Task Type Tests
# ---------------------------------------------------------------------------


class TestTaskType:
    """Test that 'task' is a valid atom type."""

    @pytest.mark.asyncio
    async def test_task_in_atom_types(self):
        """Task should be a valid atom type."""
        from memories.atoms import ATOM_TYPES
        assert "task" in ATOM_TYPES

    @pytest.mark.asyncio
    async def test_task_statuses_defined(self):
        """Task statuses should be defined."""
        assert TASK_STATUSES == ("pending", "active", "done", "archived")


# ---------------------------------------------------------------------------
# Task Creation Tests
# ---------------------------------------------------------------------------


class TestTaskCreation:
    """Test task atom creation."""

    @pytest.mark.asyncio
    async def test_create_task_default_status(self, atom_manager):
        """Creating a task without status defaults to 'pending'."""
        atom = await atom_manager.create(
            content="Implement feature X",
            type="task",
        )
        assert atom.type == "task"
        assert atom.task_status == "pending"

    @pytest.mark.asyncio
    async def test_create_task_with_status(self, atom_manager):
        """Creating a task with explicit status."""
        atom = await atom_manager.create(
            content="Review PR",
            type="task",
            task_status="active",
        )
        assert atom.type == "task"
        assert atom.task_status == "active"

    @pytest.mark.asyncio
    async def test_create_non_task_no_status(self, atom_manager):
        """Non-task atoms should have None task_status."""
        atom = await atom_manager.create(
            content="Redis is fast",
            type="fact",
        )
        assert atom.type == "fact"
        assert atom.task_status is None

    @pytest.mark.asyncio
    async def test_invalid_task_status(self, atom_manager):
        """Invalid task status should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid task_status"):
            await atom_manager.create(
                content="Bad task",
                type="task",
                task_status="invalid",
            )


# ---------------------------------------------------------------------------
# Task Status Update Tests
# ---------------------------------------------------------------------------


class TestTaskStatusUpdate:
    """Test task status updates."""

    @pytest.mark.asyncio
    async def test_update_task_status(self, atom_manager):
        """Updating task status should work."""
        atom = await atom_manager.create(
            content="Do something",
            type="task",
            task_status="pending",
        )
        updated = await atom_manager.update_task_status(atom.id, "active")
        assert updated is not None
        assert updated.task_status == "active"

    @pytest.mark.asyncio
    async def test_update_task_status_to_done(self, atom_manager):
        """Task can be marked done."""
        atom = await atom_manager.create(
            content="Complete task",
            type="task",
        )
        updated = await atom_manager.update_task_status(atom.id, "done")
        assert updated is not None
        assert updated.task_status == "done"

    @pytest.mark.asyncio
    async def test_update_non_task_returns_none(self, atom_manager):
        """Updating status on non-task returns None."""
        atom = await atom_manager.create(
            content="Not a task",
            type="fact",
        )
        result = await atom_manager.update_task_status(atom.id, "done")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_invalid_status(self, atom_manager):
        """Invalid status should raise ValueError."""
        atom = await atom_manager.create(
            content="Task",
            type="task",
        )
        with pytest.raises(ValueError, match="Invalid task_status"):
            await atom_manager.update_task_status(atom.id, "bogus")


# ---------------------------------------------------------------------------
# Task Query Tests
# ---------------------------------------------------------------------------


class TestTaskQueries:
    """Test task listing and filtering."""

    @pytest.mark.asyncio
    async def test_get_tasks_all(self, atom_manager):
        """Get all tasks regardless of status."""
        await atom_manager.create(content="Task 1", type="task", task_status="pending")
        await atom_manager.create(content="Task 2", type="task", task_status="active")
        await atom_manager.create(content="Task 3", type="task", task_status="done")

        tasks = await atom_manager.get_tasks()
        assert len(tasks) == 3

    @pytest.mark.asyncio
    async def test_get_tasks_by_status(self, atom_manager):
        """Filter tasks by status."""
        await atom_manager.create(content="Task 1", type="task", task_status="pending")
        await atom_manager.create(content="Task 2", type="task", task_status="pending")
        await atom_manager.create(content="Task 3", type="task", task_status="done")

        pending = await atom_manager.get_tasks(status="pending")
        assert len(pending) == 2

        done = await atom_manager.get_tasks(status="done")
        assert len(done) == 1

    @pytest.mark.asyncio
    async def test_get_tasks_excludes_non_tasks(self, atom_manager):
        """Task query should not return non-task atoms."""
        await atom_manager.create(content="Task", type="task")
        await atom_manager.create(content="Fact", type="fact")

        tasks = await atom_manager.get_tasks()
        assert len(tasks) == 1
        assert tasks[0].type == "task"


# ---------------------------------------------------------------------------
# Task Serialization Tests
# ---------------------------------------------------------------------------


class TestTaskSerialization:
    """Test that task_status is properly serialized."""

    @pytest.mark.asyncio
    async def test_to_dict_includes_task_status(self, atom_manager):
        """to_dict should include task_status."""
        atom = await atom_manager.create(
            content="Task",
            type="task",
            task_status="active",
        )
        d = atom.to_dict()
        assert "task_status" in d
        assert d["task_status"] == "active"

    @pytest.mark.asyncio
    async def test_to_dict_non_task_has_null_status(self, atom_manager):
        """Non-task atoms should have null task_status in dict."""
        atom = await atom_manager.create(
            content="Fact",
            type="fact",
        )
        d = atom.to_dict()
        assert "task_status" in d
        assert d["task_status"] is None

    def test_from_row_parses_task_status(self):
        """Atom.from_row should parse task_status."""
        row = _make_task_row(task_status="done")
        atom = Atom.from_row(row)
        assert atom.task_status == "done"

    def test_from_row_handles_missing_task_status(self):
        """Atom.from_row should handle rows without task_status column."""
        # Simulate an old row without task_status
        row = {
            "id": 1,
            "content": "old content",
            "type": "fact",
            "region": "general",
            "confidence": 1.0,
            "importance": 0.5,
            "access_count": 0,
            "last_accessed_at": None,
            "created_at": "2025-01-01",
            "updated_at": "2025-01-01",
            "source_project": None,
            "source_session": None,
            "source_file": None,
            "tags": None,
            "severity": None,
            "instead": None,
            "is_deleted": 0,
            # task_status intentionally missing
        }
        atom = Atom.from_row(row)
        assert atom.task_status is None
