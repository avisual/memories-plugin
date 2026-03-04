"""Tests for cross-hook Hebbian learning via hook_session_atoms — Bug 2 fix.

Covers:
- hook_session_atoms table exists and enforces the PRIMARY KEY constraint
- _save_hook_atoms writes correct IDs, deduplicates, skips bad inputs
- _hook_stop reads accumulated IDs and calls session_end_learning
- _hook_stop deletes only its own session's rows
- _hook_stop falls back gracefully when no session_id is present
- Full cross-hook cycle: session-start + prompt-submit + post-tool → stop
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.storage import Storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brain_with(storage: Storage) -> Any:
    """Return a minimal brain-like mock backed by the real storage."""
    brain = MagicMock()
    brain._storage = storage
    brain._learning = MagicMock()
    brain._learning.session_end_learning = AsyncMock(return_value={})
    brain.end_session = AsyncMock(return_value={})
    brain.shutdown = AsyncMock()
    return brain


# ---------------------------------------------------------------------------
# 1. Table schema
# ---------------------------------------------------------------------------


class TestHookSessionAtomsTable:
    async def test_table_exists_after_init(self, storage: Storage) -> None:
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='hook_session_atoms'"
        )
        assert len(rows) == 1

    async def test_primary_key_deduplicates(self, storage: Storage) -> None:
        """INSERT OR IGNORE with duplicate (session, atom) keeps exactly one row."""
        for _ in range(3):
            await storage.execute_write(
                "INSERT OR IGNORE INTO hook_session_atoms (claude_session_id, atom_id) VALUES (?, ?)",
                ("sess-pk", 1),
            )
        rows = await storage.execute(
            "SELECT * FROM hook_session_atoms WHERE claude_session_id = 'sess-pk'"
        )
        assert len(rows) == 1

    async def test_same_atom_in_different_sessions(self, storage: Storage) -> None:
        """The same atom_id under two different session_ids is two rows."""
        for sess in ("sess-a", "sess-b"):
            await storage.execute_write(
                "INSERT INTO hook_session_atoms (claude_session_id, atom_id) VALUES (?, ?)",
                (sess, 42),
            )
        rows = await storage.execute("SELECT * FROM hook_session_atoms")
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# 2. _save_hook_atoms helper
# ---------------------------------------------------------------------------


class TestSaveHookAtoms:
    async def test_saves_atoms_and_antipatterns(self, storage: Storage) -> None:
        from memories.cli import _save_hook_atoms

        brain = _brain_with(storage)
        result = {"atoms": [{"id": 1}, {"id": 2}], "antipatterns": [{"id": 3}]}
        await _save_hook_atoms(brain, "sess-1", result)

        rows = await storage.execute(
            "SELECT atom_id FROM hook_session_atoms WHERE claude_session_id = 'sess-1'"
        )
        assert {r["atom_id"] for r in rows} == {1, 2, 3}

    async def test_saves_extra_atom_id(self, storage: Storage) -> None:
        from memories.cli import _save_hook_atoms

        brain = _brain_with(storage)
        await _save_hook_atoms(brain, "sess-extra", extra_atom_id=99)

        rows = await storage.execute(
            "SELECT atom_id FROM hook_session_atoms WHERE claude_session_id = 'sess-extra'"
        )
        assert len(rows) == 1
        assert rows[0]["atom_id"] == 99

    async def test_deduplicates_across_calls(self, storage: Storage) -> None:
        from memories.cli import _save_hook_atoms

        brain = _brain_with(storage)
        result = {"atoms": [{"id": 5}], "antipatterns": []}
        await _save_hook_atoms(brain, "sess-dup", result)
        await _save_hook_atoms(brain, "sess-dup", result)  # exact repeat

        rows = await storage.execute(
            "SELECT * FROM hook_session_atoms WHERE claude_session_id = 'sess-dup'"
        )
        assert len(rows) == 1

    async def test_empty_session_id_writes_nothing(self, storage: Storage) -> None:
        from memories.cli import _save_hook_atoms

        brain = _brain_with(storage)
        await _save_hook_atoms(brain, "", {"atoms": [{"id": 7}], "antipatterns": []})

        rows = await storage.execute("SELECT * FROM hook_session_atoms")
        assert len(rows) == 0

    async def test_atoms_without_id_are_skipped(self, storage: Storage) -> None:
        from memories.cli import _save_hook_atoms

        brain = _brain_with(storage)
        result = {"atoms": [{"content": "no id key here"}], "antipatterns": []}
        await _save_hook_atoms(brain, "sess-noid", result)

        rows = await storage.execute("SELECT * FROM hook_session_atoms")
        assert len(rows) == 0

    async def test_storage_failure_is_silent(self, storage: Storage) -> None:
        from memories.cli import _save_hook_atoms

        brain = MagicMock()
        brain._storage = MagicMock()
        brain._storage.execute_many = AsyncMock(side_effect=Exception("DB exploded"))

        # Must not raise.
        await _save_hook_atoms(brain, "sess-err", {"atoms": [{"id": 1}], "antipatterns": []})

    async def test_none_result_with_extra_id(self, storage: Storage) -> None:
        from memories.cli import _save_hook_atoms

        brain = _brain_with(storage)
        await _save_hook_atoms(brain, "sess-combo", result=None, extra_atom_id=77)

        rows = await storage.execute(
            "SELECT atom_id FROM hook_session_atoms WHERE claude_session_id = 'sess-combo'"
        )
        assert len(rows) == 1
        assert rows[0]["atom_id"] == 77


# ---------------------------------------------------------------------------
# 3. _hook_stop Hebbian integration
# ---------------------------------------------------------------------------


class TestHookStopHebbian:
    async def test_calls_session_end_learning_with_accumulated_ids(
        self, storage: Storage
    ) -> None:
        from memories.cli import _hook_stop

        for atom_id in (10, 20, 30):
            await storage.execute_write(
                "INSERT INTO hook_session_atoms (claude_session_id, atom_id) VALUES (?, ?)",
                ("sess-stop-1", atom_id),
            )

        brain = _brain_with(storage)
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _hook_stop({"session_id": "sess-stop-1", "cwd": "/tmp/proj"})

        brain._learning.session_end_learning.assert_awaited_once()
        called_session_id, called_atom_ids = (
            brain._learning.session_end_learning.call_args[0]
        )
        assert called_session_id == "sess-stop-1"
        assert set(called_atom_ids) == {10, 20, 30}

    async def test_cleans_up_rows_after_learning(self, storage: Storage) -> None:
        from memories.cli import _hook_stop

        await storage.execute_write(
            "INSERT INTO hook_session_atoms (claude_session_id, atom_id) VALUES (?, ?)",
            ("sess-clean", 42),
        )

        brain = _brain_with(storage)
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _hook_stop({"session_id": "sess-clean", "cwd": "/tmp"})

        rows = await storage.execute(
            "SELECT * FROM hook_session_atoms WHERE claude_session_id = 'sess-clean'"
        )
        assert len(rows) == 0

    async def test_does_not_call_learning_when_no_atoms(
        self, storage: Storage
    ) -> None:
        """Stop hook with no accumulated atoms skips session_end_learning."""
        from memories.cli import _hook_stop

        brain = _brain_with(storage)
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _hook_stop({"session_id": "sess-empty", "cwd": "/tmp"})

        brain._learning.session_end_learning.assert_not_awaited()

    async def test_does_not_delete_other_sessions_rows(
        self, storage: Storage
    ) -> None:
        from memories.cli import _hook_stop

        for sess, atom in (("sess-mine", 1), ("sess-other", 2)):
            await storage.execute_write(
                "INSERT INTO hook_session_atoms (claude_session_id, atom_id) VALUES (?, ?)",
                (sess, atom),
            )

        brain = _brain_with(storage)
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _hook_stop({"session_id": "sess-mine", "cwd": "/tmp"})

        remaining = await storage.execute("SELECT * FROM hook_session_atoms")
        assert len(remaining) == 1
        assert remaining[0]["claude_session_id"] == "sess-other"

    async def test_falls_back_to_end_session_when_no_session_id(
        self, storage: Storage
    ) -> None:
        """When session_id is absent, end_session() is called as fallback."""
        from memories.cli import _hook_stop

        brain = _brain_with(storage)
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _hook_stop({"cwd": "/tmp"})  # no session_id key

        brain.end_session.assert_awaited_once()
        brain._learning.session_end_learning.assert_not_awaited()

    async def test_does_not_raise_on_learning_failure(
        self, storage: Storage
    ) -> None:
        """Errors inside session_end_learning are swallowed by the outer try/except."""
        from memories.cli import _hook_stop

        await storage.execute_write(
            "INSERT INTO hook_session_atoms (claude_session_id, atom_id) VALUES (?, ?)",
            ("sess-fail", 1),
        )

        brain = _brain_with(storage)
        brain._learning.session_end_learning = AsyncMock(
            side_effect=Exception("Hebbian failure")
        )
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            # Should not propagate the exception.
            await _hook_stop({"session_id": "sess-fail", "cwd": "/tmp"})


# ---------------------------------------------------------------------------
# 4. Full cross-hook cycle (integration)
# ---------------------------------------------------------------------------


class TestFullCrossHookCycle:
    async def test_atoms_accumulate_across_hooks_and_trigger_hebbian(
        self, storage: Storage
    ) -> None:
        """End-to-end: atoms from session-start, prompt-submit, post-tool all
        feed the stop hook's Hebbian learning call."""
        from memories.cli import _hook_stop, _save_hook_atoms

        sess = "sess-full-cycle"
        brain = _brain_with(storage)

        # Simulate session-start surfacing atoms 1, 2.
        await _save_hook_atoms(
            brain, sess, result={"atoms": [{"id": 1}, {"id": 2}], "antipatterns": []}
        )
        # Simulate prompt-submit surfacing atoms 3, 4 (antipattern).
        await _save_hook_atoms(
            brain, sess, result={"atoms": [{"id": 3}], "antipatterns": [{"id": 4}]}
        )
        # Simulate post-tool storing atom 5.
        await _save_hook_atoms(brain, sess, extra_atom_id=5)

        # Verify accumulation before stop.
        rows = await storage.execute(
            "SELECT atom_id FROM hook_session_atoms WHERE claude_session_id = ?",
            (sess,),
        )
        assert {r["atom_id"] for r in rows} == {1, 2, 3, 4, 5}

        # Run stop hook — should see all 5 atom IDs.
        brain2 = _brain_with(storage)
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain2)):
            await _hook_stop({"session_id": sess, "cwd": "/tmp/myproject"})

        brain2._learning.session_end_learning.assert_awaited_once()
        _, called_ids = brain2._learning.session_end_learning.call_args[0]
        assert set(called_ids) == {1, 2, 3, 4, 5}

        # Table must be clean after stop.
        remaining = await storage.execute(
            "SELECT * FROM hook_session_atoms WHERE claude_session_id = ?", (sess,)
        )
        assert len(remaining) == 0

    async def test_multiple_sessions_do_not_cross_contaminate(
        self, storage: Storage
    ) -> None:
        """Two concurrent Claude sessions accumulate separate atom sets."""
        from memories.cli import _save_hook_atoms

        brain = _brain_with(storage)
        await _save_hook_atoms(brain, "sess-A", {"atoms": [{"id": 1}], "antipatterns": []})
        await _save_hook_atoms(brain, "sess-B", {"atoms": [{"id": 2}], "antipatterns": []})

        for sess, expected in (("sess-A", {1}), ("sess-B", {2})):
            rows = await storage.execute(
                "SELECT atom_id FROM hook_session_atoms WHERE claude_session_id = ?",
                (sess,),
            )
            assert {r["atom_id"] for r in rows} == expected
