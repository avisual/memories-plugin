"""Wave 13b — TaskCompleted and Notification hooks.

Changes validated:
- TaskCompleted: task_subject stored as experience atom on completion.
- Notification (elicitation_dialog only): clarification questions stored as insights.
- Storage migration 10: adds task-completed + notification to hook_stats.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# TaskCompleted hook
# ---------------------------------------------------------------------------


class TestTaskCompletedHook:
    def test_hook_registered_in_hook_entries(self):
        from memories.setup import _HOOK_ENTRIES

        assert "TaskCompleted" in _HOOK_ENTRIES
        cmd = _HOOK_ENTRIES["TaskCompleted"][0]["hooks"][0]["command"]
        assert "task-completed" in cmd

    def test_handler_exists(self):
        from memories.cli import _hook_task_completed

        assert callable(_hook_task_completed)

    def test_hook_in_dispatch(self):
        import inspect
        from memories import cli

        source = inspect.getsource(cli.run_hook)
        assert "task-completed" in source

    @pytest.mark.asyncio
    async def test_short_subject_skipped(self):
        """Tasks with very short subjects (< 5 chars) are ignored."""
        from memories.cli import _hook_task_completed

        data = {"task_subject": "x", "session_id": "s1"}
        result = await _hook_task_completed(data)
        assert result == ""

    @pytest.mark.asyncio
    async def test_missing_subject_skipped(self):
        from memories.cli import _hook_task_completed

        data = {"session_id": "s1"}
        result = await _hook_task_completed(data)
        assert result == ""

    @pytest.mark.asyncio
    async def test_completion_stored_as_experience(self, tmp_path):
        """A completed task is stored as an experience atom."""
        from memories.cli import _hook_task_completed

        stored = []

        async def fake_remember(**kwargs):
            stored.append(kwargs)
            return {"atom_id": 90}

        async def fake_novelty(_):
            return True

        brain_mock = MagicMock()
        brain_mock.remember = fake_remember
        brain_mock._learning = MagicMock()
        brain_mock._learning.assess_novelty = fake_novelty
        brain_mock._storage = MagicMock()
        brain_mock._storage.execute_many = AsyncMock()

        with patch("memories.cli._get_brain", return_value=brain_mock):
            with patch("memories.cli._save_hook_atoms", new_callable=AsyncMock):
                with patch("memories.cli._record_hook_stat", new_callable=AsyncMock):
                    data = {
                        "task_subject": "Implement Wave 13 hooks",
                        "task_description": "Add TaskCompleted and Notification handlers",
                        "session_id": "s1",
                        "cwd": str(tmp_path),
                    }
                    await _hook_task_completed(data)

        assert stored
        assert stored[0]["type"] == "experience"
        assert "Implement Wave 13 hooks" in stored[0]["content"]

    @pytest.mark.asyncio
    async def test_description_included_when_long_enough(self, tmp_path):
        """task_description is appended when it's longer than 20 chars."""
        from memories.cli import _hook_task_completed

        stored = []

        async def fake_remember(**kwargs):
            stored.append(kwargs)
            return {"atom_id": 91}

        async def fake_novelty(_):
            return True

        brain_mock = MagicMock()
        brain_mock.remember = fake_remember
        brain_mock._learning = MagicMock()
        brain_mock._learning.assess_novelty = fake_novelty
        brain_mock._storage = MagicMock()
        brain_mock._storage.execute_many = AsyncMock()

        with patch("memories.cli._get_brain", return_value=brain_mock):
            with patch("memories.cli._save_hook_atoms", new_callable=AsyncMock):
                with patch("memories.cli._record_hook_stat", new_callable=AsyncMock):
                    data = {
                        "task_subject": "Fix authentication bug",
                        "task_description": "JWT tokens were not being refreshed correctly",
                        "session_id": "s1",
                        "cwd": str(tmp_path),
                    }
                    await _hook_task_completed(data)

        assert stored
        assert "JWT tokens" in stored[0]["content"]

    @pytest.mark.asyncio
    async def test_duplicate_not_stored(self, tmp_path):
        """Novelty gate prevents storing duplicate completions."""
        from memories.cli import _hook_task_completed

        stored = []

        async def fake_novelty(_):
            return False

        brain_mock = MagicMock()
        brain_mock.remember = AsyncMock(side_effect=lambda **kw: stored.append(kw) or {"atom_id": None})
        brain_mock._learning = MagicMock()
        brain_mock._learning.assess_novelty = fake_novelty
        brain_mock._storage = MagicMock()
        brain_mock._storage.execute_many = AsyncMock()

        with patch("memories.cli._get_brain", return_value=brain_mock):
            with patch("memories.cli._save_hook_atoms", new_callable=AsyncMock):
                with patch("memories.cli._record_hook_stat", new_callable=AsyncMock):
                    data = {
                        "task_subject": "Implement same thing again",
                        "session_id": "s1",
                        "cwd": str(tmp_path),
                    }
                    await _hook_task_completed(data)

        assert not stored


# ---------------------------------------------------------------------------
# Notification hook
# ---------------------------------------------------------------------------


class TestNotificationHook:
    def test_hook_registered_in_hook_entries(self):
        from memories.setup import _HOOK_ENTRIES

        assert "Notification" in _HOOK_ENTRIES
        entry = _HOOK_ENTRIES["Notification"][0]
        assert entry.get("matcher") == "elicitation_dialog"
        cmd = entry["hooks"][0]["command"]
        assert "notification" in cmd

    def test_handler_exists(self):
        from memories.cli import _hook_notification

        assert callable(_hook_notification)

    def test_hook_in_dispatch(self):
        import inspect
        from memories import cli

        source = inspect.getsource(cli.run_hook)
        assert "notification" in source

    @pytest.mark.asyncio
    async def test_auth_success_ignored(self):
        """auth_success notifications are always silently ignored."""
        from memories.cli import _hook_notification

        data = {
            "notification_type": "auth_success",
            "message": "Authentication succeeded for github.com",
            "session_id": "s1",
        }
        result = await _hook_notification(data)
        assert result == ""

    @pytest.mark.asyncio
    async def test_idle_prompt_ignored(self):
        """idle_prompt notifications are always silently ignored."""
        from memories.cli import _hook_notification

        data = {
            "notification_type": "idle_prompt",
            "message": "Claude has been idle for 5 minutes",
            "session_id": "s1",
        }
        result = await _hook_notification(data)
        assert result == ""

    @pytest.mark.asyncio
    async def test_permission_prompt_ignored(self):
        """permission_prompt notifications are ignored (PermissionRequest handles this)."""
        from memories.cli import _hook_notification

        data = {
            "notification_type": "permission_prompt",
            "message": "Allow bash command?",
            "session_id": "s1",
        }
        result = await _hook_notification(data)
        assert result == ""

    @pytest.mark.asyncio
    async def test_short_elicitation_skipped(self):
        """Elicitation messages under 20 chars are ignored."""
        from memories.cli import _hook_notification

        data = {
            "notification_type": "elicitation_dialog",
            "message": "Which file?",
            "session_id": "s1",
        }
        result = await _hook_notification(data)
        assert result == ""

    @pytest.mark.asyncio
    async def test_elicitation_stored_as_insight(self, tmp_path):
        """A substantive elicitation dialog is stored as an insight."""
        from memories.cli import _hook_notification

        stored = []

        async def fake_remember(**kwargs):
            stored.append(kwargs)
            return {"atom_id": 95}

        async def fake_novelty(_):
            return True

        brain_mock = MagicMock()
        brain_mock.remember = fake_remember
        brain_mock._learning = MagicMock()
        brain_mock._learning.assess_novelty = fake_novelty
        brain_mock._storage = MagicMock()
        brain_mock._storage.execute_many = AsyncMock()

        with patch("memories.cli._get_brain", return_value=brain_mock):
            with patch("memories.cli._save_hook_atoms", new_callable=AsyncMock):
                with patch("memories.cli._record_hook_stat", new_callable=AsyncMock):
                    data = {
                        "notification_type": "elicitation_dialog",
                        "message": (
                            "Should I use the existing database connection pool "
                            "or create a new one for the background worker?"
                        ),
                        "session_id": "s1",
                        "cwd": str(tmp_path),
                    }
                    await _hook_notification(data)

        assert stored
        assert stored[0]["type"] == "insight"
        assert "clarification" in stored[0]["content"].lower()


# ---------------------------------------------------------------------------
# Storage migration 10
# ---------------------------------------------------------------------------


class TestMigration10TaskNotification:
    @pytest.mark.asyncio
    async def test_migration_adds_task_notification_types(self, tmp_path):
        """Migration 10 adds task-completed + notification to hook_stats."""
        import sqlite3
        from memories.storage import Storage

        db = tmp_path / "legacy10.db"
        # Simulate a Wave 13 database (has Wave 13 types but not task-completed).
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE hook_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hook_type TEXT NOT NULL CHECK(hook_type IN (
                    'session-start','prompt-submit','pre-tool','post-tool',
                    'post-response','stop','subagent-stop','pre-compact',
                    'post-tool-failure','session-end','subagent-start','permission-request'
                )),
                project TEXT,
                query TEXT,
                atoms_returned INTEGER NOT NULL DEFAULT 0,
                atom_ids TEXT,
                avg_score REAL,
                max_score REAL,
                budget_used INTEGER NOT NULL DEFAULT 0,
                budget_total INTEGER NOT NULL DEFAULT 0,
                compression_level INTEGER NOT NULL DEFAULT 0,
                seed_count INTEGER NOT NULL DEFAULT 0,
                total_activated INTEGER NOT NULL DEFAULT 0,
                novelty_result TEXT,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

        storage = Storage(db)
        await storage.initialize()
        await storage.close()

        conn = sqlite3.connect(str(db))
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='hook_stats'"
        ).fetchone()[0]
        conn.close()
        assert "task-completed" in schema
        assert "notification" in schema
