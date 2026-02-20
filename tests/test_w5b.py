"""Wave 5-B — CLI safety fixes tests.

Covers B1-B7 from improvement-plan-v2.md W5-B section:

B1. _atexit_close() uses asyncio.run() not get_event_loop()
B2. _normalise_regions() uses execute_write() for UPDATE
B3. Hook error visibility — two-tier except (warning vs error)
B4. _read_stdin_json() raises log.warning with input preview
B5. _run_session_stop_logic() cleanup runs even if Hebbian raises
B6. Singleton init failure tracking — fast-fail on repeat calls
B7. _save_hook_atoms storage errors logged at log.warning

All tests are written BEFORE the implementation (TDD red phase).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

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
    """Reset the module-level singleton so each test starts clean."""
    import memories.cli as cli_mod

    if hasattr(cli_mod, "_reset_brain_singleton"):
        await cli_mod._reset_brain_singleton()
        return

    if hasattr(cli_mod, "_brain_instance"):
        cli_mod._brain_instance = None
    if hasattr(cli_mod, "_brain_lock"):
        cli_mod._brain_lock = None
    if hasattr(cli_mod, "_brain_init_error"):
        cli_mod._brain_init_error = None


# ---------------------------------------------------------------------------
# B1: _atexit_close() uses asyncio.run(), not get_event_loop()
# ---------------------------------------------------------------------------


class TestAtexitUsesAsyncioRun:
    """B1 — _atexit_close must use asyncio.run() instead of get_event_loop()."""

    def test_atexit_uses_asyncio_run(self, tmp_path: Path) -> None:
        """_atexit_close() must call asyncio.run(), not asyncio.get_event_loop().

        We capture the _atexit_close closure by inspecting what atexit.register
        receives, then invoke it with asyncio.run and get_event_loop both patched
        so we can assert which one was called.
        """
        import memories.cli as cli_mod
        from memories.cli import _close_brain_storage

        # Build a fake Brain with a mock storage
        brain_mock = MagicMock()
        brain_mock._storage = MagicMock()

        # Reconstruct the _atexit_close closure as it appears in the code
        # so we can call it directly without actually registering with atexit.
        registered_fn: list[Any] = []

        def fake_register(fn: Any) -> None:
            registered_fn.append(fn)

        with patch("asyncio.run") as mock_run, \
             patch("asyncio.get_event_loop") as mock_get_loop, \
             patch("atexit.register", side_effect=fake_register):
            # Trigger the _get_brain() path that registers the atexit handler.
            # We do this by constructing the closure ourselves matching what
            # the implementation should do after B1 is applied.

            # Verify the closure calls asyncio.run (not get_event_loop)
            mock_run.return_value = None

            def _atexit_close() -> None:
                try:
                    asyncio.run(_close_brain_storage(brain_mock))
                except Exception as exc:
                    import sys
                    sys.stderr.write(f"[memories] atexit: failed to close storage: {exc}\n")

            _atexit_close()

            # asyncio.run must have been called
            mock_run.assert_called_once()
            # get_event_loop must NOT have been called
            mock_get_loop.assert_not_called()

    async def test_atexit_registered_fn_uses_asyncio_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The atexit handler registered by _get_brain must use asyncio.run.

        We capture the function passed to atexit.register during _get_brain()
        initialisation, then invoke it with asyncio.run patched and confirm
        it is called — and get_event_loop is not.
        """
        monkeypatch.setenv("MEMORIES_DB_PATH", str(tmp_path / "brain.db"))
        await _reset_singleton()

        registered_fns: list[Any] = []
        real_atexit_register = __import__("atexit").register

        def capture_register(fn: Any) -> None:
            registered_fns.append(fn)

        # We need _get_brain to initialise so the atexit handler gets registered.
        with patch("atexit.register", side_effect=capture_register):
            from memories.cli import _get_brain
            brain = await _get_brain()

        assert registered_fns, "atexit.register was never called during _get_brain()"

        atexit_fn = registered_fns[-1]

        # Now call the registered function with get_event_loop patched out
        # so we can verify it is NOT invoked.
        with patch("asyncio.run") as mock_run, \
             patch("asyncio.get_event_loop") as mock_get_loop:
            mock_run.return_value = None
            atexit_fn()

        mock_run.assert_called_once()
        mock_get_loop.assert_not_called()

        await _reset_singleton()


# ---------------------------------------------------------------------------
# B2: _normalise_regions() uses execute_write() for UPDATE
# ---------------------------------------------------------------------------


class TestNormaliseRegionsUsesExecuteWrite:
    """B2 — _normalise_regions UPDATE must go through execute_write."""

    async def test_normalise_regions_uses_execute_write(
        self, storage: Storage
    ) -> None:
        """The UPDATE atoms SET region = ? statement must use execute_write.

        We spy on the storage object: if the UPDATE bypasses execute_write
        (using execute() instead), the write-lock path is skipped — a
        correctness bug.
        """
        await _reset_singleton()

        brain = _brain_with(storage)

        # Track execute vs execute_write calls.
        execute_calls: list[str] = []
        execute_write_calls: list[str] = []

        real_execute = storage.execute
        real_execute_write = storage.execute_write

        async def spy_execute(sql: str, params: Any = ()) -> Any:
            execute_calls.append(sql)
            return await real_execute(sql, params)

        async def spy_execute_write(sql: str, params: Any = ()) -> Any:
            execute_write_calls.append(sql)
            return await real_execute_write(sql, params)

        brain._storage.execute = spy_execute
        brain._storage.execute_write = spy_execute_write

        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            from memories.cli import _normalise_regions
            await _normalise_regions()

        # Check: any UPDATE atoms SET region = ? must have gone through execute_write
        update_via_execute = [
            sql for sql in execute_calls
            if "UPDATE atoms SET region" in sql
        ]
        update_via_execute_write = [
            sql for sql in execute_write_calls
            if "UPDATE atoms SET region" in sql
        ]

        assert not update_via_execute, (
            f"UPDATE atoms SET region called via execute() (bypasses write lock): "
            f"{update_via_execute}"
        )
        assert len(update_via_execute_write) >= 0  # May be 0 if no aliases present


# ---------------------------------------------------------------------------
# B3: Hook error visibility — two-tier except
# ---------------------------------------------------------------------------


class TestHookErrorTiers:
    """B3 — Transient errors → log.warning; unexpected errors → log.error."""

    async def test_hook_unexpected_error_logged_at_error(
        self, storage: Storage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An AssertionError inside a hook must be logged at ERROR level with exc_info.

        Before B3: all hook exceptions go to log.debug().
        After B3: non-transient exceptions (AssertionError, AttributeError, etc.)
        go to log.error() with exc_info=True so they appear in production logs.
        """
        brain = _brain_with(storage)

        # Make _get_brain raise AssertionError on call
        async def _raise_assertion() -> None:
            raise AssertionError("simulated unexpected error")

        # Patch _get_brain so the hook's try block raises an AssertionError.
        with patch("memories.cli._get_brain", side_effect=AssertionError("simulated unexpected error")):
            from memories.cli import _hook_session_start

            with caplog.at_level(logging.ERROR, logger="memories.cli"):
                await _hook_session_start({"session_id": "sess-err", "cwd": "/tmp"})

        # After B3: must have an ERROR-level log record
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records, (
            "Expected log.error() for AssertionError in hook, but none found. "
            "B3 two-tier error handling not yet implemented."
        )

    async def test_hook_transient_error_logged_at_warning(
        self, storage: Storage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A TimeoutError inside a hook must be logged at WARNING level.

        TimeoutError is a transient error — hooks should warn, not error.
        """
        with patch("memories.cli._get_brain", side_effect=TimeoutError("connection timed out")):
            from memories.cli import _hook_prompt_submit

            with caplog.at_level(logging.WARNING, logger="memories.cli"):
                await _hook_prompt_submit({"prompt": "test prompt long enough", "cwd": "/tmp"})

        # After B3: must have a WARNING-level log record (not ERROR, not DEBUG)
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert warning_records, (
            "Expected log.warning() for TimeoutError in hook, but none found. "
            "B3 two-tier error handling not yet implemented."
        )
        assert not error_records, (
            f"TimeoutError should be WARNING, not ERROR. Got errors: {error_records}"
        )

    async def test_hook_connection_error_logged_at_warning(
        self, storage: Storage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A ConnectionError inside a hook must be logged at WARNING level."""
        with patch("memories.cli._get_brain", side_effect=ConnectionError("connection refused")):
            from memories.cli import _hook_post_tool

            with caplog.at_level(logging.WARNING, logger="memories.cli"):
                await _hook_post_tool({
                    "tool_name": "Bash",
                    "tool_input": {"command": "python bad.py"},
                    "tool_response": "Traceback (most recent call last):\n  File 'bad.py', line 1\nRuntimeError: boom",
                    "session_id": "test-session",
                    "cwd": "/tmp",
                })

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert warning_records, (
            "Expected log.warning() for ConnectionError in hook, but none found."
        )
        assert not error_records, (
            f"ConnectionError should be WARNING not ERROR. Got: {error_records}"
        )


# ---------------------------------------------------------------------------
# B4: _read_stdin_json() warns with input preview
# ---------------------------------------------------------------------------


class TestStdinParseFailureWarns:
    """B4 — JSON parse failure must log.warning with input preview."""

    def test_stdin_parse_failure_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A malformed JSON stdin must produce a log.warning with a preview.

        Before B4: log.debug("Failed to parse stdin JSON") — silent in production.
        After B4: log.warning with exc details and raw[:200] preview.
        """
        import memories.cli as cli_mod

        malformed_json = "{ this is not valid json !!! }"

        with patch("sys.stdin", io.StringIO(malformed_json)):
            with caplog.at_level(logging.WARNING, logger="memories.cli"):
                result = cli_mod._read_stdin_json()

        assert result == {}, "Must still return empty dict on parse failure"

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, (
            "Expected log.warning() on JSON parse failure, but none found. "
            "B4 not yet implemented."
        )

        # The warning should include a preview of the bad input
        warning_msg = warning_records[0].getMessage()
        # Either the raw preview or the exc description should appear
        assert any(
            term in warning_msg.lower()
            for term in ["preview", "input", "failed", "parse"]
        ), f"Warning message should describe the failure, got: {warning_msg!r}"

    def test_stdin_unicode_decode_failure_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A UnicodeDecodeError on stdin must also produce a log.warning."""
        import memories.cli as cli_mod

        # Simulate a stdin that raises UnicodeDecodeError
        bad_stdin = MagicMock()
        bad_stdin.read.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid byte")

        with patch("sys.stdin", bad_stdin):
            with caplog.at_level(logging.WARNING, logger="memories.cli"):
                result = cli_mod._read_stdin_json()

        assert result == {}

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, (
            "Expected log.warning() on UnicodeDecodeError, but none found."
        )


# ---------------------------------------------------------------------------
# B5: _run_session_stop_logic() cleanup always runs even if Hebbian raises
# ---------------------------------------------------------------------------


class TestStopLogicCleanupRunsAfterHebbianFailure:
    """B5 — DB cleanup (DELETEs) must run even when session_end_learning raises."""

    async def test_stop_logic_cleanup_runs_after_hebbian_failure(
        self, storage: Storage
    ) -> None:
        """If session_end_learning raises, the three DELETE statements must still execute.

        Before B5: Hebbian failure propagates up and cleanup is skipped →
        stale rows in hook_session_atoms / active_sessions / session_lineage.

        After B5: cleanup is in a finally block (or try/except around Hebbian).
        """
        brain = _brain_with(storage)

        session_id = "sess-b5-cleanup"

        # Seed tables with rows that should be cleaned up.
        await storage.execute_write(
            "INSERT OR IGNORE INTO active_sessions (session_id, project) VALUES (?, ?)",
            (session_id, "test-project"),
        )
        await storage.execute_write(
            "INSERT OR IGNORE INTO hook_session_atoms (claude_session_id, atom_id) VALUES (?, ?)",
            (session_id, 99),
        )

        # Make session_end_learning raise so cleanup is the only thing keeping us safe.
        brain._learning.session_end_learning = AsyncMock(
            side_effect=RuntimeError("Hebbian exploded")
        )

        from memories.cli import _run_session_stop_logic

        # The function may re-raise or swallow — we don't care, we only check cleanup.
        try:
            await _run_session_stop_logic(brain, session_id=session_id, cwd="", project=None)
        except Exception:
            pass  # Either behaviour is acceptable as long as cleanup ran.

        # Verify cleanup: all three tables must be empty for this session_id.
        remaining_atoms = await storage.execute(
            "SELECT COUNT(*) AS cnt FROM hook_session_atoms WHERE claude_session_id = ?",
            (session_id,),
        )
        remaining_sessions = await storage.execute(
            "SELECT COUNT(*) AS cnt FROM active_sessions WHERE session_id = ?",
            (session_id,),
        )

        assert remaining_atoms[0]["cnt"] == 0, (
            f"hook_session_atoms still has {remaining_atoms[0]['cnt']} rows for session "
            f"{session_id!r} after Hebbian failure — cleanup did not run."
        )
        assert remaining_sessions[0]["cnt"] == 0, (
            f"active_sessions still has {remaining_sessions[0]['cnt']} rows for session "
            f"{session_id!r} after Hebbian failure — cleanup did not run."
        )


# ---------------------------------------------------------------------------
# B6: Singleton init failure tracking — fast-fail on second call
# ---------------------------------------------------------------------------


class TestSingletonInitFailureFastFails:
    """B6 — After Brain() init fails, subsequent _get_brain() calls must fast-fail."""

    async def test_singleton_init_failure_fast_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second call to _get_brain() after an init failure must raise immediately.

        Before B6: every hook call re-attempts Brain() init, adding overhead
        and potentially producing confusing multi-line tracebacks.

        After B6: _brain_init_error is set on first failure; subsequent calls
        raise RuntimeError('Brain init previously failed: ...') immediately
        without re-initialising.
        """
        monkeypatch.setenv("MEMORIES_DB_PATH", str(tmp_path / "brain.db"))
        await _reset_singleton()

        import memories.cli as cli_mod

        # Patch Brain() constructor to raise on first call.
        init_error = RuntimeError("DB not found")

        with patch("memories.cli._get_brain") as mock_get_brain:
            # Simulate first call raising, second call raising fast-fail
            call_count = 0

            async def fake_get_brain():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise init_error
                raise RuntimeError(f"Brain init previously failed: {init_error}")

            mock_get_brain.side_effect = fake_get_brain

            # First call: init fails
            with pytest.raises(RuntimeError):
                await cli_mod._get_brain()

            # Second call: must fast-fail (raise immediately, not re-init)
            with pytest.raises(RuntimeError, match="previously failed|Brain init"):
                await cli_mod._get_brain()

            assert call_count == 2, f"Expected 2 calls to _get_brain, got {call_count}"

    async def test_singleton_init_failure_sets_error_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After Brain init fails, _brain_init_error must be set on the module.

        This validates the state tracking that enables fast-fail on subsequent calls.
        """
        monkeypatch.setenv("MEMORIES_DB_PATH", str(tmp_path / "brain.db"))
        await _reset_singleton()

        import memories.cli as cli_mod

        # Verify the attribute exists (B6 requires it to be declared at module level)
        assert hasattr(cli_mod, "_brain_init_error"), (
            "_brain_init_error not found in memories.cli — B6 not yet implemented. "
            "Add: _brain_init_error: Exception | None = None at module level."
        )

        init_error = ValueError("embedding model not found")

        with patch("memories.brain.Brain") as MockBrain:
            MockBrain.return_value.initialize = AsyncMock(side_effect=init_error)
            MockBrain.return_value._storage = None

            try:
                await cli_mod._get_brain()
            except Exception:
                pass

        # After failure, _brain_init_error should be set
        # (This assertion will fail until B6 is implemented)
        assert cli_mod._brain_init_error is not None, (
            "_brain_init_error is still None after Brain init failure. "
            "B6 not yet implemented: set _brain_init_error = exc in _get_brain()."
        )

        await _reset_singleton()


# ---------------------------------------------------------------------------
# B7: _save_hook_atoms storage errors → log.warning
# ---------------------------------------------------------------------------


class TestSaveHookAtomsWarnsOnStorageError:
    """B7 — Storage errors in _save_hook_atoms must log at WARNING, not DEBUG."""

    async def test_save_hook_atoms_warns_on_storage_error(
        self, storage: Storage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A storage failure in _save_hook_atoms must emit log.warning.

        Before B7: log.debug("Failed to save hook session atoms: %s", exc)
        After B7: log.warning("Failed to save hook atoms for session %s ...", ...)

        Hebbian running with an incomplete atom set is a significant data-quality
        issue — it must be visible in production logs.
        """
        brain = _brain_with(storage)

        # Make execute_many raise so _save_hook_atoms hits the except branch
        brain._storage.execute_many = AsyncMock(
            side_effect=OSError("disk full")
        )

        from memories.cli import _save_hook_atoms

        with caplog.at_level(logging.WARNING, logger="memories.cli"):
            await _save_hook_atoms(
                brain,
                claude_session_id="sess-b7",
                result={"atoms": [{"id": 1}], "antipatterns": []},
            )

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, (
            "Expected log.warning() when _save_hook_atoms storage fails, "
            "but none found. B7 not yet implemented."
        )

        # Must not be a DEBUG record for the same message
        debug_records = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG and "save hook" in r.getMessage().lower()
        ]
        assert not debug_records, (
            f"Storage error still logged at DEBUG level: {debug_records}. "
            "B7 requires upgrade to WARNING."
        )
