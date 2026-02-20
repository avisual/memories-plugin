"""Tests for ProcessManager.acquire_lock(force=True) — Bug 1 fix.

Covers:
- Default (force=False) returns False when a live process holds the lock
- force=True terminates a live process and acquires the lock
- Stale (dead-PID) lock files are cleaned up in both modes
- Corrupted lock files are handled gracefully
- _terminate_process sends SIGTERM (and, if needed, SIGKILL)
- release_lock only removes our own lock
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from memories.process_manager import ProcessManager


# ---------------------------------------------------------------------------
# Fixture: redirect LOCK_FILE to a temp path so tests never touch
# ~/.memories/server.pid
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path) -> ProcessManager:
    m = ProcessManager()
    m.LOCK_FILE = tmp_path / "server.pid"
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_sleeper() -> subprocess.Popen:
    """Launch a subprocess that sleeps for 60 s."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# 1. No existing lock file
# ---------------------------------------------------------------------------


class TestFreshLock:
    def test_fresh_acquire(self, manager: ProcessManager) -> None:
        assert manager.acquire_lock() is True
        assert manager.LOCK_FILE.exists()
        assert int(manager.LOCK_FILE.read_text()) == os.getpid()

    def test_fresh_acquire_force_flag_has_no_effect(self, manager: ProcessManager) -> None:
        assert manager.acquire_lock(force=True) is True
        assert int(manager.LOCK_FILE.read_text()) == os.getpid()


# ---------------------------------------------------------------------------
# 2. Stale lock (dead PID)
# ---------------------------------------------------------------------------


class TestStaleLock:
    def test_stale_lock_acquired_without_force(self, manager: ProcessManager) -> None:
        manager.LOCK_FILE.write_text("999999")
        assert manager.acquire_lock() is True
        assert int(manager.LOCK_FILE.read_text()) == os.getpid()

    def test_stale_lock_acquired_with_force(self, manager: ProcessManager) -> None:
        manager.LOCK_FILE.write_text("999999")
        assert manager.acquire_lock(force=True) is True
        assert int(manager.LOCK_FILE.read_text()) == os.getpid()

    def test_corrupted_lock_file_is_cleaned_up(self, manager: ProcessManager) -> None:
        manager.LOCK_FILE.write_text("not-a-pid")
        assert manager.acquire_lock() is True
        assert int(manager.LOCK_FILE.read_text()) == os.getpid()


# ---------------------------------------------------------------------------
# 3. Live process — force=False
# ---------------------------------------------------------------------------


class TestLiveProcessNoForce:
    def test_returns_false_for_own_pid(self, manager: ProcessManager) -> None:
        """Our own PID is alive, so force=False should return False."""
        manager.LOCK_FILE.write_text(str(os.getpid()))
        assert manager.acquire_lock(force=False) is False

    def test_returns_false_for_live_subprocess(self, manager: ProcessManager) -> None:
        proc = _start_sleeper()
        try:
            manager.LOCK_FILE.write_text(str(proc.pid))
            assert manager.acquire_lock(force=False) is False
        finally:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# 4. Live process — force=True
# ---------------------------------------------------------------------------


class TestLiveProcessForce:
    def test_terminates_subprocess_and_acquires(self, manager: ProcessManager) -> None:
        proc = _start_sleeper()
        try:
            manager.LOCK_FILE.write_text(str(proc.pid))
            result = manager.acquire_lock(force=True)

            assert result is True
            assert int(manager.LOCK_FILE.read_text()) == os.getpid()
            # The subprocess must have been terminated.
            proc.wait(timeout=3)
        finally:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass

    def test_old_lock_replaced_with_our_pid(self, manager: ProcessManager) -> None:
        proc = _start_sleeper()
        try:
            manager.LOCK_FILE.write_text(str(proc.pid))
            manager.acquire_lock(force=True)
            assert int(manager.LOCK_FILE.read_text()) == os.getpid()
        finally:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 5. _terminate_process unit tests
# ---------------------------------------------------------------------------


class TestTerminateProcess:
    def test_dead_pid_does_not_raise(self, manager: ProcessManager) -> None:
        """Calling _terminate_process with a non-existent PID is a no-op."""
        manager._terminate_process(999999)  # must not raise

    def test_live_process_is_terminated(self, manager: ProcessManager) -> None:
        proc = _start_sleeper()
        try:
            manager._terminate_process(proc.pid)
            proc.wait(timeout=3)
            # returncode is set once the process exits (SIGTERM → negative code on Unix)
            assert proc.returncode is not None
        finally:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 6. release_lock safety
# ---------------------------------------------------------------------------


class TestReleaseLock:
    def test_release_removes_our_lock(self, manager: ProcessManager) -> None:
        manager.acquire_lock()
        manager.release_lock()
        assert not manager.LOCK_FILE.exists()

    def test_release_does_not_remove_foreign_lock(self, manager: ProcessManager) -> None:
        """release_lock must not remove another process's lock file."""
        manager.LOCK_FILE.write_text("999999")
        manager.release_lock()
        # File should still exist because 999999 != our PID.
        assert manager.LOCK_FILE.exists()

    def test_release_is_idempotent(self, manager: ProcessManager) -> None:
        manager.acquire_lock()
        manager.release_lock()
        manager.release_lock()  # second call must not raise


# ---------------------------------------------------------------------------
# 7. get_running_server_pid
# ---------------------------------------------------------------------------


class TestGetRunningServerPid:
    def test_returns_none_when_no_lock(self, manager: ProcessManager) -> None:
        assert manager.get_running_server_pid() is None

    def test_returns_none_for_stale_lock(self, manager: ProcessManager) -> None:
        manager.LOCK_FILE.write_text("999999")
        assert manager.get_running_server_pid() is None

    def test_returns_pid_for_live_process(self, manager: ProcessManager) -> None:
        proc = _start_sleeper()
        try:
            manager.LOCK_FILE.write_text(str(proc.pid))
            assert manager.get_running_server_pid() == proc.pid
        finally:
            proc.kill()
            proc.wait()
