"""Coverage tests for embeddings.py — error paths and edge cases.

These error handling paths (ConnectionError, ResponseError, batch mismatch)
were identified as untested in the coverage analysis (embeddings.py: 78%).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import ollama as ollama_lib

from memories.embeddings import EmbeddingEngine
from memories.storage import Storage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(storage):
    """Provide an EmbeddingEngine with mocked Ollama client."""
    e = EmbeddingEngine(storage)
    e._ollama_checked = True  # Skip real Ollama readiness check
    return e


# ===========================================================================
# TestEmbedTextErrors
# ===========================================================================


class TestEmbedTextErrors:
    """Tests for embed_text() error handling — lines 240-246."""

    async def test_connection_error_raises_runtime(self, engine):
        """ConnectionError from Ollama raises RuntimeError."""
        engine._client = MagicMock()
        engine._client.embed = AsyncMock(side_effect=ConnectionError("refused"))

        with pytest.raises(RuntimeError, match="not running"):
            await engine.embed_text("test text")

    async def test_response_error_raises_runtime(self, engine):
        """ollama.ResponseError raises RuntimeError."""
        engine._client = MagicMock()
        engine._client.embed = AsyncMock(
            side_effect=ollama_lib.ResponseError("model not found")
        )

        with pytest.raises(RuntimeError, match="failed"):
            await engine.embed_text("test text")


# ===========================================================================
# TestEmbedBatchErrors
# ===========================================================================


class TestEmbedBatchErrors:
    """Tests for _embed_batch_via_ollama() error handling — lines 484-533."""

    async def test_batch_connection_error(self, engine):
        """ConnectionError during batch embed raises RuntimeError."""
        engine._client = MagicMock()
        engine._client.embed = AsyncMock(side_effect=ConnectionError("refused"))

        with pytest.raises(RuntimeError, match="not running"):
            await engine._embed_batch_via_ollama(["text one", "text two"])

    async def test_batch_response_error(self, engine):
        """ollama.ResponseError during batch embed raises RuntimeError."""
        engine._client = MagicMock()
        engine._client.embed = AsyncMock(
            side_effect=ollama_lib.ResponseError("model error")
        )

        with pytest.raises(RuntimeError, match="failed"):
            await engine._embed_batch_via_ollama(["text one", "text two"])

    async def test_batch_length_mismatch(self, engine):
        """Mismatched embedding count raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 768]  # Only 1 embedding for 2 texts

        engine._client = MagicMock()
        engine._client.embed = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="returned 1 embeddings for 2"):
            await engine._embed_batch_via_ollama(["text one", "text two"])

    async def test_batch_empty_list_returns_empty(self, engine):
        """Empty input list returns empty output without calling Ollama."""
        result = await engine._embed_batch_via_ollama([])
        assert result == []


# ===========================================================================
# TestSearchSimilarNoVec
# ===========================================================================


class TestSearchSimilarNoVec:
    """Tests for search_similar() when sqlite-vec is unavailable — line 340-342."""

    async def test_no_vec_returns_empty_list(self, engine, storage):
        """When vec_available is False, search returns empty list."""
        # Temporarily override vec_available.
        original = storage.vec_available
        try:
            storage._vec_available = False
            # Override the property if it exists, or set attribute.
            if hasattr(type(storage), 'vec_available') and isinstance(
                getattr(type(storage), 'vec_available'), property
            ):
                # The property is read-only; patch it.
                with patch.object(type(storage), 'vec_available', new_callable=PropertyMock, return_value=False):
                    result = await engine.search_similar("test", k=5)
                    assert result == []
            else:
                storage.vec_available = False
                result = await engine.search_similar("test", k=5)
                assert result == []
        finally:
            if hasattr(storage, '_vec_available'):
                storage._vec_available = original


# ===========================================================================
# TestEnsureOllamaReady
# ===========================================================================


class TestEnsureOllamaReady:
    """Tests for _ensure_ollama_ready() — lines 153-167."""

    async def test_failure_raises_runtime_error(self, storage):
        """If OllamaManager.ensure_ready() fails, RuntimeError is raised."""
        engine = EmbeddingEngine(storage)
        engine._ollama_checked = False  # Force the check

        mock_manager = MagicMock()
        mock_manager.ensure_ready = AsyncMock(
            return_value=(False, "Ollama not installed")
        )

        with patch("memories.embeddings.OllamaManager", return_value=mock_manager):
            with pytest.raises(RuntimeError, match="not ready"):
                await engine.ensure_ollama_ready()

    async def test_success_sets_checked_flag(self, storage):
        """On success, _ollama_checked is set to True."""
        engine = EmbeddingEngine(storage)
        engine._ollama_checked = False

        mock_manager = MagicMock()
        mock_manager.ensure_ready = AsyncMock(
            return_value=(True, "All good")
        )

        with patch("memories.embeddings.OllamaManager", return_value=mock_manager):
            await engine.ensure_ollama_ready()
            assert engine._ollama_checked is True

    async def test_skips_when_already_checked(self, storage):
        """If already checked, does not re-check."""
        engine = EmbeddingEngine(storage)
        engine._ollama_checked = True

        # This should return immediately without creating an OllamaManager.
        with patch("memories.embeddings.OllamaManager") as mock_cls:
            await engine.ensure_ollama_ready()
            mock_cls.assert_not_called()
