"""Wave 4 — Brain singleton tests for CLI hooks.

Covers:
- _get_brain() returns the same instance on consecutive calls (singleton)
- initialize() is idempotent (calling it twice does not double-migrate or raise)
- stop hook triggers session_end_learning exactly once (no double-fire)
- _close_brain_storage does not call session_end_learning

These tests are written BEFORE the implementation so they fail on the current
code, confirming the TDD red-green cycle.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.storage import Storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brain_with(storage: Storage) -> Any:
    """Return a minimal brain-like mock backed by real storage."""
    brain = MagicMock()
    brain._storage = storage
    brain._learning = MagicMock()
    brain._learning.session_end_learning = AsyncMock(return_value={})
    brain.end_session = AsyncMock(return_value={})
    brain.shutdown = AsyncMock()
    return brain


async def _reset_singleton() -> None:
    """Reset the module-level singleton so each test starts clean.

    Calls _reset_brain_singleton when available (post-implementation);
    falls back to direct attribute manipulation for the pre-implementation
    failing tests.  Also forces a config reload so monkeypatched env vars
    (like MEMORIES_DB_PATH) take effect for the next _get_brain() call.
    """
    import memories.cli as cli_mod

    # Post-implementation path: use the official reset helper.
    if hasattr(cli_mod, "_reset_brain_singleton"):
        await cli_mod._reset_brain_singleton()
    else:
        # Pre-implementation path: manually zero out whatever is there.
        if hasattr(cli_mod, "_brain_instance"):
            cli_mod._brain_instance = None
        if hasattr(cli_mod, "_brain_lock"):
            cli_mod._brain_lock = None

    # Force config reload so monkeypatched env vars take effect.
    from memories.config import get_config
    get_config(reload=True)


# ---------------------------------------------------------------------------
# TestBrainSingleton
# ---------------------------------------------------------------------------


class TestBrainSingleton:
    """Verify that _get_brain() implements process-level singleton semantics."""

    async def test_singleton_same_instance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two consecutive _get_brain() calls must return the exact same object."""
        # Point the Brain's DB to a temp path so we never touch production data.
        monkeypatch.setenv("MEMORIES_DB_PATH", str(tmp_path / "brain.db"))

        # Ensure a clean slate — no leftover singleton from a previous test.
        await _reset_singleton()

        from memories.cli import _get_brain

        brain_a = await _get_brain()
        brain_b = await _get_brain()

        # `is` check: must be the identical Python object, not just equal.
        assert brain_a is brain_b, (
            "_get_brain() returned two different Brain instances; "
            "singleton not implemented yet"
        )

    async def test_initialize_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling initialize() on the singleton a second time must not raise."""
        monkeypatch.setenv("MEMORIES_DB_PATH", str(tmp_path / "brain.db"))
        await _reset_singleton()

        from memories.cli import _get_brain

        brain = await _get_brain()

        # Call initialize() a second time — must be a no-op (idempotent).
        try:
            await brain.initialize()
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"Second initialize() raised unexpectedly: {exc}")

        # Brain should still be usable.
        assert brain._initialized is True

    async def test_no_double_hebbian_firing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After the stop hook runs session_end_learning, the atexit cleanup path
        (_close_brain_storage) must NOT fire session_end_learning again.

        This validates the double-fire fix: the atexit handler closes storage
        only — it does NOT call brain.shutdown() or end_session().
        """
        monkeypatch.setenv("MEMORIES_DB_PATH", str(tmp_path / "brain.db"))
        await _reset_singleton()

        from memories.cli import _get_brain, _run_session_stop_logic

        brain = await _get_brain()

        # Seed the hook_session_atoms table so _run_session_stop_logic has
        # something to work with.
        session_id = "test-singleton-sess"
        await brain._storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            (session_id, None),
        )
        await brain._storage.execute_write(
            "INSERT INTO hook_session_atoms (claude_session_id, atom_id) VALUES (?, ?)",
            (session_id, 1),
        )

        # Patch session_end_learning with an async counter.
        call_count = 0

        async def _count_calls(*args: Any, **kwargs: Any) -> dict:
            nonlocal call_count
            call_count += 1
            return {"synapses_strengthened": 0, "synapses_created": 0}

        brain._learning.session_end_learning = _count_calls  # type: ignore[method-assign]

        # Step 1: Run stop-hook logic (fires Hebbian once).
        await _run_session_stop_logic(brain, session_id=session_id)

        # Step 2: Run the atexit cleanup path — must NOT fire Hebbian again.
        from memories.cli import _close_brain_storage
        await _close_brain_storage(brain)

        assert call_count == 1, (
            f"session_end_learning was called {call_count} time(s); "
            "expected exactly 1. Double-fire fix not yet in place."
        )

    async def test_reset_brain_singleton_exists(self) -> None:
        """The module must expose _reset_brain_singleton for test isolation."""
        import memories.cli as cli_mod

        assert hasattr(cli_mod, "_reset_brain_singleton"), (
            "_reset_brain_singleton not found in memories.cli; "
            "add it so tests can reset state between runs"
        )

    async def test_close_brain_storage_exists(self) -> None:
        """The module must expose _close_brain_storage for the atexit path."""
        import memories.cli as cli_mod

        assert hasattr(cli_mod, "_close_brain_storage"), (
            "_close_brain_storage not found in memories.cli"
        )


# ---------------------------------------------------------------------------
# TestHookStopNoBrainShutdown
# ---------------------------------------------------------------------------


class TestHookStopNoBrainShutdown:
    """Verify that _hook_stop no longer calls brain.shutdown()."""

    async def test_hook_stop_does_not_call_shutdown(
        self, storage: Storage
    ) -> None:
        """With the singleton, _hook_stop must NOT call brain.shutdown()."""
        from memories.cli import _hook_stop

        brain = _brain_with(storage)
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _hook_stop({"session_id": "sess-no-shutdown", "cwd": "/tmp"})

        brain.shutdown.assert_not_awaited()

    async def test_hook_subagent_stop_does_not_call_shutdown(
        self, storage: Storage
    ) -> None:
        """With the singleton, _hook_subagent_stop must NOT call brain.shutdown()."""
        from memories.cli import _hook_subagent_stop

        brain = _brain_with(storage)
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _hook_subagent_stop({"session_id": "sess-sub-no-shutdown", "cwd": "/tmp"})

        brain.shutdown.assert_not_awaited()
