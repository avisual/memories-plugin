"""Process-level singleton enforcement via PID lock file.

Ensures only one memories MCP server runs per machine. Uses a PID lock file
to detect and prevent multiple simultaneous server instances, which can cause
database conflicts and memory corruption.

Usage::

    from memories.process_manager import ProcessManager

    manager = ProcessManager()
    if not manager.acquire_lock():
        print("Another server is already running")
        sys.exit(1)

    try:
        # Run server
        pass
    finally:
        manager.release_lock()
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path


class ProcessManager:
    """Ensures only one memories MCP server runs per machine.

    Uses a PID lock file to coordinate between multiple potential server
    processes. The lock file stores the PID of the currently running server
    and is automatically cleaned up if the process exits normally.
    """

    LOCK_FILE = Path("~/.memories/server.pid").expanduser()

    def acquire_lock(self, force: bool = False) -> bool:
        """Acquire process lock. Returns False if another server is running.

        Parameters
        ----------
        force:
            When ``True`` and an existing server process is alive, send
            SIGTERM (then SIGKILL after 1 s) before acquiring the lock.
            This is used by the MCP server on reconnect so that Claude Code
            can always start a fresh server without manual intervention.

        Returns
        -------
        bool
            ``True`` if the lock was acquired, ``False`` if another server
            is already running and ``force`` is ``False``.
        """
        if self.LOCK_FILE.exists():
            try:
                pid = int(self.LOCK_FILE.read_text().strip())
                if self._is_process_alive(pid):
                    if not force:
                        return False  # Another server is running
                    # Force mode: gracefully replace the existing process.
                    self._terminate_process(pid)
                # Stale or just-terminated lock file — remove it.
                try:
                    self.LOCK_FILE.unlink()
                except FileNotFoundError:
                    pass
            except (ValueError, FileNotFoundError):
                # Corrupted lock file - remove it
                try:
                    self.LOCK_FILE.unlink()
                except FileNotFoundError:
                    pass

        # Write our PID
        self.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.LOCK_FILE.write_text(str(os.getpid()))
        return True

    def _terminate_process(self, pid: int) -> None:
        """Send SIGTERM to *pid*, then SIGKILL after 1 second if still alive."""
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            return  # Already gone

        # Poll for up to 1 second for graceful exit.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            time.sleep(0.1)
            if not self._is_process_alive(pid):
                return

        # Force kill if the process is still alive.
        try:
            if self._is_process_alive(pid):
                os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    def release_lock(self) -> None:
        """Release process lock on shutdown.

        Only removes the lock file if it contains our own PID, preventing
        accidental removal of another process's lock.
        """
        if self.LOCK_FILE.exists():
            try:
                pid = int(self.LOCK_FILE.read_text().strip())
                if pid == os.getpid():
                    self.LOCK_FILE.unlink()
            except (ValueError, FileNotFoundError):
                pass

    def get_running_server_pid(self) -> int | None:
        """Get PID of running server, or None.

        Returns
        -------
        int | None
            The PID of the currently running server, or ``None`` if no
            server is running or the lock file is stale.
        """
        if not self.LOCK_FILE.exists():
            return None
        try:
            pid = int(self.LOCK_FILE.read_text().strip())
            return pid if self._is_process_alive(pid) else None
        except (ValueError, FileNotFoundError):
            return None

    def _is_process_alive(self, pid: int) -> bool:
        """Check if process with given PID is still running.

        Uses ``os.kill(pid, 0)`` which sends a null signal. If the process
        exists and is owned by the current user, this succeeds. Otherwise
        it raises an exception.

        Parameters
        ----------
        pid:
            Process ID to check.

        Returns
        -------
        bool
            ``True`` if the process is running, ``False`` otherwise.
        """
        try:
            os.kill(pid, 0)  # Signal 0 = check existence
            return True
        except (OSError, ProcessLookupError):
            return False
