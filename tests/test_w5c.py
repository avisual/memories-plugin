"""Wave 5-C — consolidation.py + storage.py correctness fixes.

Covers C1-C5 from the improvement plan:

C1. Lazy ollama import — ConsolidationEngine.__init__ must not import ollama eagerly
C2. log.warning in _distill_cluster() exception handler
C3. isinstance(res, BaseException) in _distill_clusters_concurrent()
C4. storage.close() differentiated error handling (log per-conn failures)
C5. SQLite version assertion for RETURNING support (>= 3.35.0)

All tests are written BEFORE the implementation (TDD red phase).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.storage import Storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_storage(tmp_path: Path) -> Storage:
    """Create and initialise a real Storage instance for testing."""
    storage = Storage(db_path=tmp_path / "test.db")
    await storage.initialize()
    return storage


# ===========================================================================
# C1 — Lazy ollama import
# ===========================================================================


class TestLazyOllamaImport:
    """ConsolidationEngine.__init__ must NOT import or construct ollama.AsyncClient."""

    async def test_init_does_not_create_llm_client(self, tmp_path: Path) -> None:
        """Creating a ConsolidationEngine should NOT eagerly construct
        ollama.AsyncClient. The _llm_client attribute should not exist
        or be None after __init__."""
        storage = await _make_storage(tmp_path)

        try:
            from memories.consolidation import ConsolidationEngine

            engine = ConsolidationEngine(
                storage=storage,
                embeddings=MagicMock(),
                atoms=MagicMock(),
                synapses=MagicMock(),
            )
            # After the fix, _llm_client should not be created in __init__.
            assert not hasattr(engine, "_llm_client") or engine._llm_client is None
        finally:
            await storage.close()

    async def test_distill_cluster_creates_client_lazily(self, tmp_path: Path) -> None:
        """_distill_cluster() should create the ollama client on first call,
        not at __init__ time."""
        storage = await _make_storage(tmp_path)

        try:
            from memories.consolidation import ConsolidationEngine
            from memories.atoms import Atom

            engine = ConsolidationEngine(
                storage=storage,
                embeddings=MagicMock(),
                atoms=MagicMock(),
                synapses=MagicMock(),
            )

            mock_atom = MagicMock(spec=Atom)
            mock_atom.content = "test content for distillation"

            # Patch ollama at module level so _distill_cluster can lazily import it
            mock_ollama_mod = MagicMock()
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(
                return_value=MagicMock(response="distilled insight")
            )
            mock_ollama_mod.AsyncClient.return_value = mock_client

            with patch.dict("sys.modules", {"ollama": mock_ollama_mod}):
                result = await engine._distill_cluster([mock_atom])

            # The method should have created the client and used it
            assert isinstance(result, str)
        finally:
            await storage.close()

    async def test_no_top_level_ollama_import(self) -> None:
        """The consolidation module must NOT have a top-level
        ``import ollama`` statement.  The import must only happen inside
        _distill_cluster."""
        import importlib
        import memories.consolidation as mod

        source = importlib.util.find_spec("memories.consolidation")
        assert source is not None and source.origin is not None

        with open(source.origin) as f:
            source_code = f.read()

        # Check that there is no top-level "import ollama" (outside functions).
        # Lines starting with "import ollama" at module level are forbidden.
        # Lines inside functions (indented) are fine.
        lines = source_code.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "import ollama" or stripped.startswith("import ollama "):
                # Must be indented (inside a function), not at top level
                assert line[0] in (" ", "\t"), (
                    f"Line {i}: found top-level 'import ollama' — "
                    f"must be moved inside _distill_cluster()"
                )


# ===========================================================================
# C2 — log.warning in _distill_cluster() exception handler
# ===========================================================================


class TestDistillClusterLogsWarning:
    """The except block in _distill_cluster() must log a warning, not silently
    swallow exceptions."""

    async def test_exception_logged_as_warning(self, tmp_path: Path) -> None:
        """When the LLM call raises, _distill_cluster must log.warning."""
        storage = await _make_storage(tmp_path)

        try:
            from memories.consolidation import ConsolidationEngine
            from memories.atoms import Atom

            engine = ConsolidationEngine(
                storage=storage,
                embeddings=MagicMock(),
                atoms=MagicMock(),
                synapses=MagicMock(),
            )

            mock_atom = MagicMock(spec=Atom)
            mock_atom.content = "test fallback content"

            # Force the LLM call to raise by injecting a broken client.
            # After C1 fix, _llm_client won't exist yet, so we set it
            # directly so the method uses it instead of importing ollama.
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(
                side_effect=ConnectionError("ollama down")
            )
            engine._llm_client = mock_client

            with patch("memories.consolidation.logger") as mock_logger:
                result = await engine._distill_cluster([mock_atom])

            # Must return the fallback
            assert result == "test fallback content"

            # Must have logged a warning (not silently swallowed)
            mock_logger.warning.assert_called_once()
            # The warning message should mention distillation
            call_args = mock_logger.warning.call_args
            msg = call_args[0][0].lower()
            assert "distil" in msg, (
                f"Expected warning about distillation, got: {call_args}"
            )
        finally:
            await storage.close()

    async def test_timeout_exception_logged_as_warning(self, tmp_path: Path) -> None:
        """Timeout errors should also be logged at WARNING."""
        storage = await _make_storage(tmp_path)

        try:
            from memories.consolidation import ConsolidationEngine
            from memories.atoms import Atom

            engine = ConsolidationEngine(
                storage=storage,
                embeddings=MagicMock(),
                atoms=MagicMock(),
                synapses=MagicMock(),
            )

            mock_atom = MagicMock(spec=Atom)
            mock_atom.content = "timeout fallback"

            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(
                side_effect=asyncio.TimeoutError("LLM too slow")
            )
            engine._llm_client = mock_client

            with patch("memories.consolidation.logger") as mock_logger:
                result = await engine._distill_cluster([mock_atom])

            assert result == "timeout fallback"
            mock_logger.warning.assert_called_once()
        finally:
            await storage.close()


# ===========================================================================
# C3 — isinstance(res, BaseException) for CancelledError
# ===========================================================================


class TestDistillClustersConcurrentBaseException:
    """_distill_clusters_concurrent must catch BaseException (not just
    Exception) so that asyncio.CancelledError is handled correctly."""

    async def test_cancelled_error_is_caught(self, tmp_path: Path) -> None:
        """asyncio.CancelledError (a BaseException, not Exception) must be
        caught and produce a fallback string, not treated as a result."""
        storage = await _make_storage(tmp_path)

        try:
            from memories.consolidation import ConsolidationEngine
            from memories.atoms import Atom

            engine = ConsolidationEngine(
                storage=storage,
                embeddings=MagicMock(),
                atoms=MagicMock(),
                synapses=MagicMock(),
            )

            mock_atom1 = MagicMock(spec=Atom)
            mock_atom1.content = "fallback content 1"
            mock_atom2 = MagicMock(spec=Atom)
            mock_atom2.content = "fallback content 2"

            cluster1 = [mock_atom1]
            cluster2 = [mock_atom2]

            # Patch _distill_cluster to raise CancelledError for cluster1
            # and return normally for cluster2.
            async def side_effect(cluster):
                if cluster is cluster1:
                    raise asyncio.CancelledError()
                return "distilled 2"

            engine._distill_cluster = AsyncMock(side_effect=side_effect)

            results = await engine._distill_clusters_concurrent([cluster1, cluster2])

            # cluster1 raised CancelledError (BaseException) — should get fallback
            assert results[0] == "fallback content 1"
            # cluster2 succeeded normally
            assert results[1] == "distilled 2"
        finally:
            await storage.close()

    async def test_regular_exception_still_caught(self, tmp_path: Path) -> None:
        """Regular Exception subclasses must still produce fallbacks too."""
        storage = await _make_storage(tmp_path)

        try:
            from memories.consolidation import ConsolidationEngine
            from memories.atoms import Atom

            engine = ConsolidationEngine(
                storage=storage,
                embeddings=MagicMock(),
                atoms=MagicMock(),
                synapses=MagicMock(),
            )

            mock_atom = MagicMock(spec=Atom)
            mock_atom.content = "regular exception fallback"
            cluster = [mock_atom]

            async def raise_value_error(c):
                raise ValueError("bad value")

            engine._distill_cluster = AsyncMock(side_effect=raise_value_error)

            results = await engine._distill_clusters_concurrent([cluster])
            assert results[0] == "regular exception fallback"
        finally:
            await storage.close()


# ===========================================================================
# C4 — storage.close() differentiated error handling
# ===========================================================================


class TestStorageCloseDifferentiatedLogging:
    """storage.close() must log per-connection failures at WARNING level
    with error detail, rather than silently swallowing."""

    async def test_close_logs_warning_on_connection_error(self, tmp_path: Path) -> None:
        """When a connection.close() raises, the error must be logged at
        WARNING with the exception detail."""
        storage = await _make_storage(tmp_path)

        # Inject a mock connection that raises on close
        bad_conn = MagicMock()
        bad_conn.close.side_effect = sqlite3.OperationalError("disk I/O error")

        with storage._connections_lock:
            storage._all_connections.append(bad_conn)

        with patch("memories.storage.log") as mock_log:
            await storage.close()

        # Must have called log.warning (not silently passed)
        mock_log.warning.assert_called()
        # The warning should include the error detail
        warning_calls = mock_log.warning.call_args_list
        found = any("disk I/O error" in str(c) for c in warning_calls)
        assert found, f"Expected warning about 'disk I/O error', got: {warning_calls}"

    async def test_close_does_not_raise_on_connection_error(self, tmp_path: Path) -> None:
        """close() must be best-effort — it must not re-raise even if a
        connection.close() fails."""
        storage = await _make_storage(tmp_path)

        bad_conn = MagicMock()
        bad_conn.close.side_effect = RuntimeError("already closed")

        with storage._connections_lock:
            storage._all_connections.append(bad_conn)

        # Must not raise
        await storage.close()

    async def test_close_still_closes_remaining_after_one_fails(self, tmp_path: Path) -> None:
        """If one connection fails to close, remaining connections must still
        be attempted."""
        storage = await _make_storage(tmp_path)

        bad_conn = MagicMock()
        bad_conn.close.side_effect = OSError("broken pipe")
        good_conn = MagicMock()

        with storage._connections_lock:
            storage._all_connections.clear()
            storage._all_connections.extend([bad_conn, good_conn])

        await storage.close()

        # The good connection must still have been closed
        good_conn.close.assert_called_once()


# ===========================================================================
# C5 — SQLite version assertion for RETURNING support
# ===========================================================================


class TestSqliteVersionAssertion:
    """Storage must check that SQLite >= 3.35.0 at startup."""

    async def test_current_sqlite_passes(self, tmp_path: Path) -> None:
        """On this machine (SQLite 3.51.2), the assertion must pass silently."""
        storage = Storage(db_path=tmp_path / "test.db")
        await storage.initialize()
        await storage.close()

    async def test_old_sqlite_raises_runtime_error(self, tmp_path: Path) -> None:
        """When SQLite version is too old, a RuntimeError must be raised
        before any DB operations."""
        storage = Storage(db_path=tmp_path / "test_old.db")

        # Patch the version string that the check reads.
        # We patch at the module level where storage.py imports sqlite3.
        with patch("memories.storage.sqlite3") as mock_sqlite3:
            mock_sqlite3.sqlite_version = "3.34.1"
            # Preserve real classes/functions needed for the rest
            mock_sqlite3.connect = sqlite3.connect
            mock_sqlite3.Row = sqlite3.Row
            mock_sqlite3.IntegrityError = sqlite3.IntegrityError
            mock_sqlite3.OperationalError = sqlite3.OperationalError
            mock_sqlite3.Connection = sqlite3.Connection

            with pytest.raises(RuntimeError, match="3.35.0"):
                await storage.initialize()

    async def test_exact_minimum_version_passes(self, tmp_path: Path) -> None:
        """SQLite 3.35.0 exactly should pass the check (boundary condition)."""
        storage = Storage(db_path=tmp_path / "test_exact.db")

        with patch("memories.storage.sqlite3") as mock_sqlite3:
            mock_sqlite3.sqlite_version = "3.35.0"
            mock_sqlite3.connect = sqlite3.connect
            mock_sqlite3.Row = sqlite3.Row
            mock_sqlite3.IntegrityError = sqlite3.IntegrityError
            mock_sqlite3.OperationalError = sqlite3.OperationalError
            mock_sqlite3.Connection = sqlite3.Connection

            # Should NOT raise
            await storage.initialize()
            await storage.close()
