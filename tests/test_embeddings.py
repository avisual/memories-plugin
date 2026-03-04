"""Comprehensive tests for the embedding engine.

All tests mock the ``ollama.AsyncClient`` so that Ollama does not need to be
running.  The storage fixture provides a real SQLite backend (in a temp
directory) so cache and vector persistence can be verified end-to-end.
"""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.embeddings import EmbeddingEngine, _SlidingWindowRateLimiter
from memories.storage import Storage, deserialize_embedding, serialize_embedding


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_DIMS = 768  # default embedding dimensions from config


def _fake_embedding(seed: float = 1.0) -> list[float]:
    """Generate a deterministic fake embedding vector of the configured dims."""
    return [seed * (i + 1) / _DIMS for i in range(_DIMS)]


def _mock_embed_response(*embeddings: list[float]) -> SimpleNamespace:
    """Build an object mimicking the Ollama embed response."""
    return SimpleNamespace(embeddings=list(embeddings))


def _build_engine(storage: Storage, mock_client: AsyncMock) -> EmbeddingEngine:
    """Build an EmbeddingEngine with an injected mock client."""
    engine = EmbeddingEngine(storage)
    engine._client = mock_client
    return engine


# -----------------------------------------------------------------------
# 1. embed_text
# -----------------------------------------------------------------------


class TestEmbedText:
    """Verify embed_text produces and caches embeddings correctly."""

    async def test_returns_correct_dimensions(self, storage: Storage) -> None:
        """embed_text must return a vector with the configured number of dims."""
        mock_client = AsyncMock()
        fake_vec = _fake_embedding()
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(fake_vec))

        engine = _build_engine(storage, mock_client)
        result = await engine.embed_text("hello world")

        assert len(result) == _DIMS
        assert result == fake_vec

    async def test_calls_ollama_on_cache_miss(self, storage: Storage) -> None:
        """On a cache miss, the Ollama client embed method is invoked."""
        mock_client = AsyncMock()
        fake_vec = _fake_embedding()
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(fake_vec))

        engine = _build_engine(storage, mock_client)
        await engine.embed_text("cache miss text")

        mock_client.embed.assert_called_once()

    async def test_cache_hit_avoids_ollama_call(self, storage: Storage) -> None:
        """After the first embed, a second call with the same text uses the cache."""
        mock_client = AsyncMock()
        fake_vec = _fake_embedding()
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(fake_vec))

        engine = _build_engine(storage, mock_client)

        # First call -- cache miss.
        first = await engine.embed_text("cached text")
        assert mock_client.embed.call_count == 1

        # Second call -- cache hit.
        second = await engine.embed_text("cached text")
        assert mock_client.embed.call_count == 1  # not called again

        # The cached value round-trips through float32 serialization so we
        # compare with tolerance rather than exact equality.
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert math.isclose(a, b, rel_tol=1e-6)

    async def test_cache_stored_in_database(self, storage: Storage) -> None:
        """After embedding, the cache table must contain the result."""
        mock_client = AsyncMock()
        fake_vec = _fake_embedding()
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(fake_vec))

        engine = _build_engine(storage, mock_client)
        text = "store in db"
        await engine.embed_text(text)

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rows = await storage.execute(
            "SELECT embedding, model FROM embedding_cache WHERE content_hash = ?",
            (content_hash,),
        )
        assert len(rows) == 1
        restored = deserialize_embedding(rows[0]["embedding"])
        assert len(restored) == _DIMS

    async def test_runtime_error_on_connection_failure(self, storage: Storage) -> None:
        """A ConnectionError from Ollama is wrapped in a RuntimeError."""
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=ConnectionError("refused"))

        engine = _build_engine(storage, mock_client)

        with pytest.raises(RuntimeError, match="Ollama server is not running"):
            await engine.embed_text("fail text")


# -----------------------------------------------------------------------
# 2. embed_batch
# -----------------------------------------------------------------------


class TestEmbedBatch:
    """Verify batch embedding with caching and batching logic."""

    async def test_batch_returns_all_embeddings(self, storage: Storage) -> None:
        """embed_batch returns one embedding per input text in order."""
        mock_client = AsyncMock()
        vecs = [_fake_embedding(seed=float(i)) for i in range(3)]
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(*vecs))

        engine = _build_engine(storage, mock_client)
        results = await engine.embed_batch(["a", "b", "c"])

        assert len(results) == 3
        for i, result in enumerate(results):
            assert len(result) == _DIMS
            assert result == vecs[i]

    async def test_batch_uses_cache_for_known_texts(self, storage: Storage) -> None:
        """Already-cached texts are skipped in the Ollama batch call."""
        mock_client = AsyncMock()
        vec_a = _fake_embedding(seed=1.0)
        vec_b = _fake_embedding(seed=2.0)
        vec_c = _fake_embedding(seed=3.0)

        # First: embed "a" individually to populate the cache.
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(vec_a))
        engine = _build_engine(storage, mock_client)
        await engine.embed_text("a")
        assert mock_client.embed.call_count == 1

        # Now batch-embed ["a", "b", "c"].  Only "b" and "c" should go to Ollama.
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(vec_b, vec_c))
        results = await engine.embed_batch(["a", "b", "c"])

        assert len(results) == 3
        # The cached value round-trips through float32 serialization so we
        # compare with tolerance rather than exact equality.
        for a, b in zip(results[0], vec_a):
            assert math.isclose(a, b, rel_tol=1e-6)
        # Ollama was called once for the uncached batch.
        assert mock_client.embed.call_count == 1

    async def test_batch_empty_list(self, storage: Storage) -> None:
        """embed_batch with an empty list returns an empty list."""
        mock_client = AsyncMock()
        engine = _build_engine(storage, mock_client)
        results = await engine.embed_batch([])
        assert results == []
        mock_client.embed.assert_not_called()

    async def test_batch_respects_batch_size(self, storage: Storage) -> None:
        """When inputs exceed batch_size, multiple Ollama calls are made."""
        mock_client = AsyncMock()

        engine = _build_engine(storage, mock_client)
        engine._batch_size = 2  # force small batches

        # 5 uncached texts -> ceil(5/2) = 3 batch calls.
        def make_response(*args, **kwargs):
            inp = kwargs.get("input", args[0] if args else [])
            if isinstance(inp, str):
                inp = [inp]
            vecs = [_fake_embedding(seed=float(hash(t) % 100)) for t in inp]
            return _mock_embed_response(*vecs)

        mock_client.embed = AsyncMock(side_effect=make_response)

        texts = ["t0", "t1", "t2", "t3", "t4"]
        results = await engine.embed_batch(texts)

        assert len(results) == 5
        assert mock_client.embed.call_count == 3  # 2+2+1


# -----------------------------------------------------------------------
# 3. embed_and_store
# -----------------------------------------------------------------------


class TestEmbedAndStore:
    """Verify embed_and_store writes to the atoms_vec table."""

    async def test_stores_embedding_in_atoms_vec(self, storage: Storage) -> None:
        """After embed_and_store, atoms_vec contains the embedding for the atom."""
        if not storage.vec_available:
            pytest.skip("sqlite-vec not available on this platform")

        mock_client = AsyncMock()
        fake_vec = _fake_embedding()
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(fake_vec))

        engine = _build_engine(storage, mock_client)

        # Insert a real atom first.
        atom_id = await storage.execute_write(
            "INSERT INTO atoms (content, type) VALUES (?, ?)",
            ("embed me", "fact"),
        )

        result = await engine.embed_and_store(atom_id, "embed me")
        assert result == fake_vec

        rows = await storage.execute(
            "SELECT atom_id FROM atoms_vec WHERE atom_id = ?", (atom_id,)
        )
        assert len(rows) == 1

    async def test_embed_and_store_returns_vector(self, storage: Storage) -> None:
        """embed_and_store returns the embedding vector."""
        if not storage.vec_available:
            pytest.skip("sqlite-vec not available on this platform")

        mock_client = AsyncMock()
        fake_vec = _fake_embedding(seed=42.0)
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(fake_vec))

        engine = _build_engine(storage, mock_client)

        atom_id = await storage.execute_write(
            "INSERT INTO atoms (content, type) VALUES (?, ?)",
            ("vector return test", "fact"),
        )

        result = await engine.embed_and_store(atom_id, "vector return test")
        assert len(result) == _DIMS
        assert result == fake_vec


# -----------------------------------------------------------------------
# 4. health_check
# -----------------------------------------------------------------------


class TestHealthCheck:
    """Verify the Ollama health check probe."""

    async def test_returns_true_when_healthy(self, storage: Storage) -> None:
        """health_check returns True when Ollama responds normally."""
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(
            return_value=_mock_embed_response(_fake_embedding())
        )

        engine = _build_engine(storage, mock_client)
        assert await engine.health_check() is True

    async def test_returns_false_on_connection_error(self, storage: Storage) -> None:
        """health_check returns False when Ollama is unreachable."""
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=ConnectionError("refused"))

        engine = _build_engine(storage, mock_client)
        assert await engine.health_check() is False

    async def test_returns_false_on_os_error(self, storage: Storage) -> None:
        """health_check returns False on OSError."""
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=OSError("network down"))

        engine = _build_engine(storage, mock_client)
        assert await engine.health_check() is False

    async def test_returns_false_on_unexpected_error(self, storage: Storage) -> None:
        """health_check returns False on arbitrary exceptions."""
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=RuntimeError("boom"))

        engine = _build_engine(storage, mock_client)
        assert await engine.health_check() is False


# -----------------------------------------------------------------------
# 5. cosine_similarity
# -----------------------------------------------------------------------


class TestCosineSimilarity:
    """Verify the cosine similarity calculation.

    cosine_similarity is a pure synchronous function (no I/O) — callers
    invoke it directly without ``await``.
    """

    def test_identical_vectors(self, storage: Storage) -> None:
        """Cosine similarity of identical vectors is 1.0."""
        mock_client = MagicMock()
        engine = _build_engine(storage, mock_client)

        vec = [1.0, 2.0, 3.0]
        sim = engine.cosine_similarity(vec, vec)
        # numpy float32 precision requires a looser tolerance than float64.
        assert math.isclose(sim, 1.0, rel_tol=1e-5)

    def test_orthogonal_vectors(self, storage: Storage) -> None:
        """Cosine similarity of orthogonal vectors is 0.0."""
        mock_client = MagicMock()
        engine = _build_engine(storage, mock_client)

        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        sim = engine.cosine_similarity(vec_a, vec_b)
        assert math.isclose(sim, 0.0, abs_tol=1e-9)

    def test_opposite_vectors(self, storage: Storage) -> None:
        """Cosine similarity of opposite vectors is -1.0."""
        mock_client = MagicMock()
        engine = _build_engine(storage, mock_client)

        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        sim = engine.cosine_similarity(vec_a, vec_b)
        assert math.isclose(sim, -1.0, rel_tol=1e-9)

    def test_zero_vector_returns_zero(self, storage: Storage) -> None:
        """Cosine similarity with a zero vector returns 0.0."""
        mock_client = MagicMock()
        engine = _build_engine(storage, mock_client)

        sim = engine.cosine_similarity([0.0, 0.0], [1.0, 2.0])
        assert sim == 0.0

    def test_mismatched_lengths_raises(self, storage: Storage) -> None:
        """Vectors of different lengths raise ValueError."""
        mock_client = MagicMock()
        engine = _build_engine(storage, mock_client)

        with pytest.raises(ValueError, match="Vector length mismatch"):
            engine.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_known_value(self, storage: Storage) -> None:
        """Verify a hand-computed cosine similarity value."""
        mock_client = MagicMock()
        engine = _build_engine(storage, mock_client)

        # cos([1, 2, 3], [4, 5, 6]) = 32 / (sqrt(14) * sqrt(77))
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [4.0, 5.0, 6.0]
        expected = 32.0 / (math.sqrt(14) * math.sqrt(77))
        sim = engine.cosine_similarity(vec_a, vec_b)
        # numpy float32 precision requires a looser tolerance than float64.
        assert math.isclose(sim, expected, rel_tol=1e-5)


# -----------------------------------------------------------------------
# 6. distance_to_similarity
# -----------------------------------------------------------------------


class TestDistanceToSimilarity:
    """Verify L2 distance to similarity conversion."""

    def test_zero_distance_gives_one(self) -> None:
        """Distance 0 maps to similarity 1.0 (identical vectors)."""
        assert EmbeddingEngine.distance_to_similarity(0.0) == 1.0

    def test_distance_two_gives_zero(self) -> None:
        """Distance 2.0 maps to similarity 0.0."""
        assert EmbeddingEngine.distance_to_similarity(2.0) == 0.0

    def test_distance_one_gives_half(self) -> None:
        """Distance 1.0 maps to similarity 0.5."""
        assert math.isclose(
            EmbeddingEngine.distance_to_similarity(1.0), 0.5, rel_tol=1e-9
        )

    def test_large_distance_clamps_to_zero(self) -> None:
        """Distances above 2.0 are clamped to similarity 0.0."""
        assert EmbeddingEngine.distance_to_similarity(10.0) == 0.0

    def test_small_distance_high_similarity(self) -> None:
        """A small distance maps to a high similarity."""
        sim = EmbeddingEngine.distance_to_similarity(0.1)
        assert sim == pytest.approx(0.95, abs=1e-9)


# -----------------------------------------------------------------------
# 7. Content hash
# -----------------------------------------------------------------------


class TestContentHash:
    """Verify the internal content hashing mechanism."""

    def test_deterministic(self) -> None:
        """The same text always produces the same hash."""
        engine = EmbeddingEngine.__new__(EmbeddingEngine)
        h1 = engine._content_hash("hello")
        h2 = engine._content_hash("hello")
        assert h1 == h2

    def test_different_text_different_hash(self) -> None:
        """Different texts produce different hashes."""
        engine = EmbeddingEngine.__new__(EmbeddingEngine)
        h1 = engine._content_hash("hello")
        h2 = engine._content_hash("world")
        assert h1 != h2

    def test_matches_sha256(self) -> None:
        """The hash matches a direct SHA-256 computation."""
        engine = EmbeddingEngine.__new__(EmbeddingEngine)
        text = "test content"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert engine._content_hash(text) == expected

    def test_empty_string(self) -> None:
        """An empty string produces a valid SHA-256 hash."""
        engine = EmbeddingEngine.__new__(EmbeddingEngine)
        h = engine._content_hash("")
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected


# -----------------------------------------------------------------------
# 8. Rate limiter
# -----------------------------------------------------------------------


class TestSlidingWindowRateLimiter:
    """Verify the sliding-window rate limiter does not block under limit."""

    async def test_under_limit_no_blocking(self) -> None:
        """Acquiring fewer slots than max_rps does not sleep."""
        limiter = _SlidingWindowRateLimiter(max_rps=100)
        # Acquiring 10 times with a limit of 100 should never block.
        for _ in range(10):
            await limiter.acquire()
        # If we got here without hanging, the test passes.

    async def test_timestamps_tracked(self) -> None:
        """Each acquire() records a timestamp in the internal deque."""
        limiter = _SlidingWindowRateLimiter(max_rps=50)
        await limiter.acquire()
        await limiter.acquire()
        assert len(limiter._timestamps) == 2


# -----------------------------------------------------------------------
# 9. Cache model mismatch
# -----------------------------------------------------------------------


class TestCacheModelMismatch:
    """Verify that stale cache entries (different model) are treated as misses."""

    async def test_stale_model_triggers_cache_miss(self, storage: Storage) -> None:
        """An embedding cached under a different model name is evicted."""
        text = "model mismatch test"
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Pre-populate the cache with a different model name.
        old_vec = _fake_embedding(seed=99.0)
        blob = serialize_embedding(old_vec)
        await storage.execute_write(
            "INSERT INTO embedding_cache (content_hash, embedding, model) "
            "VALUES (?, ?, ?)",
            (content_hash, blob, "old-model-v1"),
        )

        mock_client = AsyncMock()
        new_vec = _fake_embedding(seed=1.0)
        mock_client.embed = AsyncMock(return_value=_mock_embed_response(new_vec))

        engine = _build_engine(storage, mock_client)
        result = await engine.embed_text(text)

        # Should have called Ollama since the cached entry used a different model.
        mock_client.embed.assert_called_once()
        assert result == new_vec

        # The cache should now contain the new model entry.
        rows = await storage.execute(
            "SELECT model FROM embedding_cache WHERE content_hash = ?",
            (content_hash,),
        )
        assert len(rows) == 1
        assert rows[0]["model"] == "nomic-embed-text"
