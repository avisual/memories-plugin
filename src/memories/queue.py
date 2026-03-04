"""Async hook observation queue.

Hooks append observations instantly (<5ms JSONL append).
A background worker drains the queue and processes each observation
through the full hook pipeline (embed, store, auto-link).

This decouples hook capture from embedding/storage, dropping hook
latency from 200-800ms to <10ms.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

try:
    import fcntl as _fcntl

    _HAS_FCNTL = True
except ImportError:
    _fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False

log = logging.getLogger(__name__)

_DEFAULT_QUEUE_PATH = Path.home() / ".memories" / "hook_queue.jsonl"


class HookQueue:
    """Line-delimited JSON queue for async hook observation capture.

    Parameters
    ----------
    queue_path:
        Filesystem path for the JSONL queue file.  Defaults to
        ``~/.memories/hook_queue.jsonl``.
    """

    def __init__(self, queue_path: Path | None = None) -> None:
        self._queue_path = queue_path or _DEFAULT_QUEUE_PATH
        self._worker_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def queue_path(self) -> Path:
        """Filesystem path of the JSONL queue file."""
        return self._queue_path

    def append(self, event_name: str, data: dict[str, Any]) -> None:
        """Append an event to the queue as a single JSON line.

        This is a synchronous, atomic operation using ``fcntl.flock``
        for append safety on macOS/Linux.  Designed to complete in <5ms.

        On Windows, ``fcntl`` is unavailable; concurrent appends from
        multiple processes are not protected by OS-level locking.  Within
        a single process, ``"a"`` mode provides atomic appends.

        Parameters
        ----------
        event_name:
            The hook event name (e.g. ``"post-tool"``).
        data:
            Arbitrary event payload.
        """
        entry = {
            "event": event_name,
            "data": data,
            "queued_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        line = json.dumps(entry, separators=(",", ":")) + "\n"

        self._queue_path.parent.mkdir(parents=True, exist_ok=True)

        fd = open(self._queue_path, "a", encoding="utf-8")
        try:
            if _HAS_FCNTL and _fcntl:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
            # Cross-process atomic lock. Within one process, "a" mode
            # appends are atomic up to PIPE_BUF on POSIX.
            fd.write(line)
            fd.flush()
        finally:
            if _HAS_FCNTL and _fcntl:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            fd.close()

    def drain(self) -> list[tuple[str, dict[str, Any]]]:
        """Read all entries from the queue, clear the file, and return parsed entries.

        Corrupt or partial JSON lines are skipped with a warning logged.

        Returns
        -------
        list[tuple[str, dict]]
            Each entry is ``(event_name, data)`` in queue order.
        """
        if not self._queue_path.exists():
            return []

        entries: list[tuple[str, dict[str, Any]]] = []

        fd = open(self._queue_path, "r+", encoding="utf-8")
        try:
            if _HAS_FCNTL and _fcntl:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
            lines = fd.readlines()
            # Truncate the file to clear it.
            fd.seek(0)
            fd.truncate()
            # Lock held through truncate so racing appenders see the
            # empty file after we release.
        finally:
            if _HAS_FCNTL and _fcntl:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            fd.close()

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                event_name = obj.get("event", "unknown")
                data = obj.get("data", {})
                entries.append((event_name, data))
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning(
                    "Skipping corrupt queue entry at line %d: %s (content: %r)",
                    i + 1,
                    exc,
                    line[:200],
                )

        return entries

    def start_worker(
        self,
        handler: Callable[[str, dict[str, Any]], Awaitable[None]],
        interval: float = 2.0,
    ) -> asyncio.Task:
        """Start a background asyncio task that drains the queue periodically.

        Parameters
        ----------
        handler:
            An async callable ``handler(event_name, data)`` invoked for
            each queue entry.
        interval:
            Seconds between drain cycles (default 2.0).

        Returns
        -------
        asyncio.Task
            The background worker task.
        """
        self._stop_event = asyncio.Event()

        async def _worker() -> None:
            assert self._stop_event is not None
            while not self._stop_event.is_set():
                try:
                    entries = self.drain()
                    for event_name, data in entries:
                        try:
                            await handler(event_name, data)
                        except Exception:
                            log.exception(
                                "Queue worker: handler failed for event %r",
                                event_name,
                            )
                except Exception:
                    log.exception("Queue worker: drain cycle failed")

                # Wait for interval or until stopped.
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=interval
                    )
                except asyncio.TimeoutError:
                    pass  # Normal — timeout means we loop again.

            # Final drain on shutdown.
            try:
                entries = self.drain()
                for event_name, data in entries:
                    try:
                        await handler(event_name, data)
                    except Exception:
                        log.exception(
                            "Queue worker (final drain): handler failed for event %r",
                            event_name,
                        )
            except Exception:
                log.exception("Queue worker: final drain failed")

        self._worker_task = asyncio.create_task(_worker())
        return self._worker_task

    async def stop_worker(self) -> None:
        """Graceful shutdown: signal stop, drain once more, then await completion."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None
