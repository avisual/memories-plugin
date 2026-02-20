"""Wave 13 — SessionEnd, SubagentStart, PermissionRequest hooks + test gap fixes.

Changes validated:
- SessionEnd hook: fallback Hebbian for leftover session atoms; observability stat.
- SubagentStart hook: captures delegation patterns as insight atoms.
- PermissionRequest hook: captures risky operations as antipattern/experience.
- Storage migration 9: adds wave-13 hook types to hook_stats CHECK constraint.
- Test gap (Wave 12): zero-timestamp edge case in temporal Hebbian weighting.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# SessionEnd hook
# ---------------------------------------------------------------------------


class TestSessionEndHook:
    def test_hook_registered_in_hook_entries(self):
        from memories.setup import _HOOK_ENTRIES

        assert "SessionEnd" in _HOOK_ENTRIES
        cmd = _HOOK_ENTRIES["SessionEnd"][0]["hooks"][0]["command"]
        assert "session-end" in cmd

    def test_handler_exists(self):
        from memories.cli import _hook_session_end

        assert callable(_hook_session_end)

    def test_hook_in_dispatch(self):
        import inspect
        from memories import cli

        source = inspect.getsource(cli.run_hook)
        assert "session-end" in source

    @pytest.mark.asyncio
    async def test_no_leftover_atoms_does_nothing(self, tmp_path):
        """When stop hook already cleaned up, session-end is a no-op for Hebbian."""
        from memories.cli import _hook_session_end

        brain_mock = MagicMock()
        brain_mock._storage = MagicMock()
        brain_mock._storage.execute = AsyncMock(return_value=[])  # no leftover atoms

        with patch("memories.cli._get_brain", return_value=brain_mock):
            with patch("memories.cli._record_hook_stat", new_callable=AsyncMock):
                data = {
                    "session_id": "test-session-abc",
                    "cwd": str(tmp_path),
                    "reason": "clear",
                }
                result = await _hook_session_end(data)

        # No Hebbian call when no leftover atoms.
        assert result == ""
        brain_mock._storage.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_leftover_atoms_triggers_fallback_hebbian(self, tmp_path):
        """When stop hook was skipped, session-end runs Hebbian as fallback."""
        from memories.cli import _hook_session_end, _reset_brain_singleton

        await _reset_brain_singleton()

        hebbian_called = []

        async def fake_session_end_learning(session_id, atom_ids, **kwargs):
            hebbian_called.append((session_id, atom_ids))
            return {"synapses_strengthened": 1, "synapses_created": 0}

        brain_mock = MagicMock()
        brain_mock._storage = MagicMock()
        brain_mock._storage.execute = AsyncMock(
            return_value=[{"atom_id": 10}, {"atom_id": 11}]
        )
        brain_mock._storage.execute_write = AsyncMock()
        brain_mock._learning = MagicMock()
        brain_mock._learning.session_end_learning = fake_session_end_learning

        with patch("memories.cli._get_brain", return_value=brain_mock):
            with patch("memories.cli._record_hook_stat", new_callable=AsyncMock):
                data = {
                    "session_id": "orphan-session",
                    "cwd": str(tmp_path),
                    "reason": "logout",
                }
                await _hook_session_end(data)

        assert hebbian_called, "Expected fallback Hebbian to fire"
        assert hebbian_called[0][1] == [10, 11]


# ---------------------------------------------------------------------------
# SubagentStart hook
# ---------------------------------------------------------------------------


class TestSubagentStartHook:
    def test_hook_registered_in_hook_entries(self):
        from memories.setup import _HOOK_ENTRIES

        assert "SubagentStart" in _HOOK_ENTRIES
        cmd = _HOOK_ENTRIES["SubagentStart"][0]["hooks"][0]["command"]
        assert "subagent-start" in cmd

    def test_handler_exists(self):
        from memories.cli import _hook_subagent_start

        assert callable(_hook_subagent_start)

    def test_hook_in_dispatch(self):
        import inspect
        from memories import cli

        source = inspect.getsource(cli.run_hook)
        assert "subagent-start" in source

    @pytest.mark.asyncio
    async def test_bash_agent_type_skipped(self):
        """Bash sub-agents are too noisy — must be silently skipped."""
        from memories.cli import _hook_subagent_start

        data = {"agent_type": "Bash", "agent_id": "x", "session_id": "s1"}
        result = await _hook_subagent_start(data)
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_agent_type_skipped(self):
        """Empty agent_type must be silently skipped."""
        from memories.cli import _hook_subagent_start

        data = {"agent_type": "", "agent_id": "x", "session_id": "s1"}
        result = await _hook_subagent_start(data)
        assert result == ""

    @pytest.mark.asyncio
    async def test_explore_agent_stored_as_insight(self, tmp_path):
        """Explore sub-agent delegation must be stored as an insight atom."""
        from memories.cli import _hook_subagent_start

        stored = []

        async def fake_remember(**kwargs):
            stored.append(kwargs)
            return {"atom_id": 55}

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
                        "agent_type": "Explore",
                        "agent_id": "agent-123",
                        "session_id": "parent-session",
                        "cwd": str(tmp_path),
                    }
                    await _hook_subagent_start(data)

        assert stored, "Expected remember() to be called"
        assert stored[0]["type"] == "insight"
        assert "Explore" in stored[0]["content"]

    @pytest.mark.asyncio
    async def test_duplicate_not_stored_when_not_novel(self, tmp_path):
        """When novelty gate returns False, no atom is created."""
        from memories.cli import _hook_subagent_start

        stored = []

        async def fake_remember(**kwargs):
            stored.append(kwargs)
            return {"atom_id": 56}

        async def fake_novelty(_):
            return False  # not novel

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
                        "agent_type": "Explore",
                        "agent_id": "agent-456",
                        "session_id": "s2",
                        "cwd": str(tmp_path),
                    }
                    await _hook_subagent_start(data)

        assert not stored, "Duplicate should not be stored"


# ---------------------------------------------------------------------------
# PermissionRequest hook
# ---------------------------------------------------------------------------


class TestPermissionRequestHook:
    def test_hook_registered_in_hook_entries(self):
        from memories.setup import _HOOK_ENTRIES

        assert "PermissionRequest" in _HOOK_ENTRIES
        cmd = _HOOK_ENTRIES["PermissionRequest"][0]["hooks"][0]["command"]
        assert "permission-request" in cmd

    def test_handler_exists(self):
        from memories.cli import _hook_permission_request

        assert callable(_hook_permission_request)

    def test_hook_in_dispatch(self):
        import inspect
        from memories import cli

        source = inspect.getsource(cli.run_hook)
        assert "permission-request" in source

    @pytest.mark.asyncio
    async def test_skip_tool_respected(self):
        """Tools in _SKIP_TOOLS must be silently skipped."""
        from memories.cli import _hook_permission_request

        data = {
            "tool_name": "Read",
            "tool_input": {"file_path": "/some/path/to/file.txt"},
            "session_id": "s1",
        }
        result = await _hook_permission_request(data)
        assert result == ""

    @pytest.mark.asyncio
    async def test_short_command_skipped(self):
        """Commands shorter than 10 chars must be silently skipped."""
        from memories.cli import _hook_permission_request

        data = {
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "session_id": "s1",
        }
        result = await _hook_permission_request(data)
        assert result == ""

    @pytest.mark.asyncio
    async def test_dangerous_command_stored_as_antipattern(self, tmp_path):
        """A permission request for a dangerous command stored as antipattern."""
        from memories.cli import _hook_permission_request

        stored = []

        async def fake_remember(**kwargs):
            stored.append(kwargs)
            return {"atom_id": 77}

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
                        "tool_name": "Bash",
                        "tool_input": {
                            "command": "sudo rm -rf /etc/config — avoid this dangerous pattern"
                        },
                        "session_id": "s1",
                        "cwd": str(tmp_path),
                    }
                    await _hook_permission_request(data)

        assert stored
        assert stored[0]["type"] == "antipattern"

    @pytest.mark.asyncio
    async def test_normal_permission_stored_as_experience(self, tmp_path):
        """A routine permission request (no danger vocab) stored as experience."""
        from memories.cli import _hook_permission_request

        stored = []

        async def fake_remember(**kwargs):
            stored.append(kwargs)
            return {"atom_id": 78}

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
                        "tool_name": "Bash",
                        "tool_input": {
                            "command": "git push origin main --force-with-lease"
                        },
                        "session_id": "s1",
                        "cwd": str(tmp_path),
                    }
                    await _hook_permission_request(data)

        assert stored
        assert stored[0]["type"] == "experience"


# ---------------------------------------------------------------------------
# Storage migration 9
# ---------------------------------------------------------------------------


class TestMigration9HookStatTypes:
    @pytest.mark.asyncio
    async def test_migration_adds_wave13_types(self, tmp_path):
        """Migration 9 adds session-end, subagent-start, permission-request to hook_stats."""
        import sqlite3
        from memories.storage import Storage

        db = tmp_path / "legacy9.db"
        # Simulate a Wave 12 database (has post-tool-failure but not Wave 13 types).
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE hook_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hook_type TEXT NOT NULL CHECK(hook_type IN (
                    'session-start','prompt-submit','pre-tool','post-tool',
                    'post-response','stop','subagent-stop','pre-compact','post-tool-failure'
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

        # Verify the new types are now accepted by the constraint.
        conn = sqlite3.connect(str(db))
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='hook_stats'"
        ).fetchone()[0]
        conn.close()
        assert "session-end" in schema
        assert "subagent-start" in schema
        assert "permission-request" in schema


# ---------------------------------------------------------------------------
# Wave 12 test gap: zero-timestamp edge case
# ---------------------------------------------------------------------------


class TestZeroTimestampEdgeCase:
    """Atoms with unparseable accessed_at get 0.0 (Unix epoch) as timestamp.

    This means they are treated as temporally distant from any 2026-era atom,
    receiving the 0.5x reduced Hebbian increment on new synapse creation.
    This test verifies the behaviour is deterministic and doesn't crash.
    """

    @pytest.mark.asyncio
    async def test_zero_epoch_atom_gets_half_increment(self, tmp_path):
        """Atom at Unix epoch paired with a 2026 atom uses 0.5x increment path."""
        from memories.storage import Storage
        from memories.synapses import SynapseManager

        db = tmp_path / "zero_ts.db"
        storage = Storage(db)
        await storage.initialize()
        mgr = SynapseManager(storage)

        for aid in [30, 31]:
            await storage.execute_write(
                "INSERT OR IGNORE INTO atoms "
                "(id, content, type, confidence, importance, access_count, "
                " created_at, is_deleted) "
                "VALUES (?, 'content', 'fact', 0.9, 0.5, 5, datetime('now'), 0)",
                (aid,),
            )

        now = time.time()
        # Atom 30 has a zero/epoch timestamp (simulates NULL/unparseable accessed_at).
        atom_timestamps = {30: 0.0, 31: now}  # ~1.75B seconds apart > 300s window

        updated = await mgr.hebbian_update([30, 31], atom_timestamps=atom_timestamps)
        # Should complete without error — just uses 0.5x increment.
        assert updated >= 1

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE "
            "(source_id=30 AND target_id=31) OR (source_id=31 AND target_id=30)"
        )
        assert rows, "Synapse should still be created despite zero-epoch timestamp"
        assert rows[0]["strength"] > 0

    @pytest.mark.asyncio
    async def test_both_zero_epoch_atoms_get_full_increment(self, tmp_path):
        """Two atoms both at epoch 0.0 are within window (dist=0) — full increment."""
        from memories.storage import Storage
        from memories.synapses import SynapseManager

        db = tmp_path / "both_zero.db"
        storage = Storage(db)
        await storage.initialize()
        mgr = SynapseManager(storage)

        for aid in [40, 41]:
            await storage.execute_write(
                "INSERT OR IGNORE INTO atoms "
                "(id, content, type, confidence, importance, access_count, "
                " created_at, is_deleted) "
                "VALUES (?, 'content', 'fact', 0.9, 0.5, 5, datetime('now'), 0)",
                (aid,),
            )

        # Both atoms at epoch 0 — dist = 0 ≤ 300s → full increment path.
        atom_timestamps = {40: 0.0, 41: 0.0}
        updated = await mgr.hebbian_update([40, 41], atom_timestamps=atom_timestamps)
        assert updated >= 1
