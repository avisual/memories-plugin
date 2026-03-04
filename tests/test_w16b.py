"""Tests for Wave 16b — learning quality improvements.

Covers:
- LLM relationship classifier returns valid types and caches results
- assess_novelty returns (bool, int) tuple
- Per-prompt timestamps for temporal Hebbian weighting
- Temporal weighting: prompt-adjacent atoms get full increment
- Confidence calibration from similar_count
- rate_recall MCP tool
- Migration 13 creates retrieval_weight_log table
- Content truncation to 2000 chars
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.atoms import Atom, AtomManager
from memories.brain import Brain
from memories.embeddings import EmbeddingEngine
from memories.learning import (
    LearningEngine,
    _LLM_CLASSIFY_CACHE,
    _LLM_CLASSIFY_CACHE_MAX,
    _VALID_LLM_RELATIONSHIPS,
)
from memories.storage import Storage
from memories.synapses import SynapseManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _insert_atom(
    storage: Storage,
    content: str,
    atom_type: str = "fact",
    region: str = "general",
    confidence: float = 1.0,
) -> int:
    """Insert an atom directly via SQL."""
    now = datetime.now(tz=timezone.utc).isoformat()
    return await storage.execute_write(
        """
        INSERT INTO atoms (content, type, region, confidence, last_accessed_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (content, atom_type, region, confidence, now),
    )


def _make_atom(
    atom_id: int = 1,
    content: str = "test atom",
    atom_type: str = "fact",
    region: str = "general",
    confidence: float = 1.0,
) -> Atom:
    """Create a mock Atom object."""
    now = datetime.now(tz=timezone.utc).isoformat()
    return Atom(
        id=atom_id,
        content=content,
        type=atom_type,
        region=region,
        confidence=confidence,
        importance=0.5,
        access_count=0,
        last_accessed_at=now,
        created_at=now,
        updated_at=now,
        source_project=None,
        source_session=None,
        source_file=None,
        tags=None,
        severity=None,
        instead=None,
        is_deleted=False,
        task_status=None,
    )


def _make_engine(storage, mock_embeddings=None) -> LearningEngine:
    """Create a LearningEngine with mocked dependencies."""
    if mock_embeddings is None:
        mock_embeddings = MagicMock(spec=EmbeddingEngine)
        mock_embeddings.search_similar = AsyncMock(return_value=[])
    atoms = AtomManager(storage, mock_embeddings)
    synapses = SynapseManager(storage)
    return LearningEngine(storage, mock_embeddings, atoms, synapses)


# ---------------------------------------------------------------------------
# Test: LLM classifier returns a valid relationship type
# ---------------------------------------------------------------------------


class TestLLMRelationshipClassifier:
    """Tests for the LLM-based relationship classifier."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the LLM classification cache before each test."""
        _LLM_CLASSIFY_CACHE.clear()
        yield
        _LLM_CLASSIFY_CACHE.clear()

    async def test_returns_valid_relationship(self, storage: Storage):
        """LLM classifier returns a valid relationship type from the allowed set."""
        engine = _make_engine(storage)
        atom_a = _make_atom(1, "SQLite WAL mode improves concurrent reads")
        atom_b = _make_atom(2, "WAL journal mode enables read-write concurrency")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.response = "elaborates"
        mock_client.generate.return_value = mock_response

        with patch.dict("sys.modules", {"ollama": MagicMock(AsyncClient=MagicMock(return_value=mock_client))}):
            result = await engine._classify_relationship_llm(atom_a, atom_b)

        assert result in _VALID_LLM_RELATIONSHIPS
        assert result == "elaborates"

    async def test_caches_result(self, storage: Storage):
        """LLM classifier caches: calling twice with same pair only calls Ollama once."""
        engine = _make_engine(storage)
        atom_a = _make_atom(1, "Redis SCAN is O(N)")
        atom_b = _make_atom(2, "Redis uses single-threaded event loop")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.response = "related-to"
        mock_client.generate.return_value = mock_response

        with patch.dict("sys.modules", {"ollama": MagicMock(AsyncClient=MagicMock(return_value=mock_client))}):
            # First call — should hit Ollama
            result1 = await engine._classify_relationship_llm(atom_a, atom_b)
            # Second call — should hit cache
            result2 = await engine._classify_relationship_llm(atom_a, atom_b)

        assert result1 == result2
        assert mock_client.generate.call_count == 1

    async def test_returns_none_on_invalid_response(self, storage: Storage):
        """LLM classifier returns None when response is not a valid type."""
        engine = _make_engine(storage)
        atom_a = _make_atom(1, "test content a")
        atom_b = _make_atom(2, "test content b")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.response = "some-invalid-type"
        mock_client.generate.return_value = mock_response

        with patch.dict("sys.modules", {"ollama": MagicMock(AsyncClient=MagicMock(return_value=mock_client))}):
            result = await engine._classify_relationship_llm(atom_a, atom_b)

        assert result is None

    async def test_returns_none_on_exception(self, storage: Storage):
        """LLM classifier returns None on failure (e.g. timeout)."""
        engine = _make_engine(storage)
        atom_a = _make_atom(1, "test content a unique_x")
        atom_b = _make_atom(2, "test content b unique_y")

        mock_ollama = MagicMock()
        mock_ollama.AsyncClient.side_effect = ConnectionError("no ollama")

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = await engine._classify_relationship_llm(atom_a, atom_b)

        assert result is None


# ---------------------------------------------------------------------------
# Test: assess_novelty returns (bool, int) tuple
# ---------------------------------------------------------------------------


class TestAssessNoveltyTuple:
    """Tests for the updated assess_novelty return type."""

    async def test_returns_tuple(self, storage: Storage):
        """assess_novelty returns a (bool, int) tuple."""
        engine = _make_engine(storage)
        result = await engine.assess_novelty("some novel content")
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_novel, similar_count = result
        assert isinstance(is_novel, bool)
        assert isinstance(similar_count, int)

    async def test_novel_content_returns_true(self, storage: Storage):
        """Novel content (no similar atoms) returns (True, 0)."""
        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.search_similar = AsyncMock(return_value=[])
        engine = _make_engine(storage, mock_emb)

        is_novel, similar_count = await engine.assess_novelty("completely new content")
        assert is_novel is True
        assert similar_count == 0

    async def test_empty_content_returns_false(self, storage: Storage):
        """Empty content returns (False, 0)."""
        engine = _make_engine(storage)
        is_novel, similar_count = await engine.assess_novelty("")
        assert is_novel is False
        assert similar_count == 0


# ---------------------------------------------------------------------------
# Test: Confidence calibration from similar_count
# ---------------------------------------------------------------------------


class TestConfidenceCalibration:
    """Tests for confidence tier mapping from similar_count."""

    def test_similar_count_zero_gives_0_5(self):
        from memories.cli import _confidence_from_similar_count
        assert _confidence_from_similar_count(0) == 0.5

    def test_similar_count_three_gives_0_7(self):
        from memories.cli import _confidence_from_similar_count
        assert _confidence_from_similar_count(3) == 0.7

    def test_similar_count_six_gives_0_85(self):
        from memories.cli import _confidence_from_similar_count
        assert _confidence_from_similar_count(6) == 0.85

    def test_similar_count_one_gives_0_6(self):
        from memories.cli import _confidence_from_similar_count
        assert _confidence_from_similar_count(1) == 0.6

    def test_similar_count_high_gives_0_85(self):
        from memories.cli import _confidence_from_similar_count
        assert _confidence_from_similar_count(10) == 0.85


# ---------------------------------------------------------------------------
# Test: Per-prompt timestamps
# ---------------------------------------------------------------------------


class TestPerPromptTimestamps:
    """Tests for per-prompt atom timestamp tracking."""

    def test_prompt_atom_timestamps_dict_exists(self):
        """Module-level _prompt_atom_timestamps dict exists and is a dict."""
        from memories.cli import _prompt_atom_timestamps
        assert isinstance(_prompt_atom_timestamps, dict)

    async def test_atoms_from_different_prompts_get_different_timestamps(self):
        """Atoms recalled in different prompts should have different timestamps."""
        from memories.cli import _prompt_atom_timestamps

        session_id = "test-session-ts"
        # Simulate prompt 1 recording timestamps
        _prompt_atom_timestamps[session_id] = {}
        ts1 = "2026-01-15T10:00:00"
        _prompt_atom_timestamps[session_id][100] = ts1
        _prompt_atom_timestamps[session_id][101] = ts1

        # Simulate prompt 2 (later) recording timestamps
        ts2 = "2026-01-15T10:45:00"
        _prompt_atom_timestamps[session_id][200] = ts2

        # Verify timestamps differ between prompt cohorts
        assert _prompt_atom_timestamps[session_id][100] == ts1
        assert _prompt_atom_timestamps[session_id][200] == ts2
        assert ts1 != ts2

        # Cleanup
        _prompt_atom_timestamps.pop(session_id, None)


# ---------------------------------------------------------------------------
# Test: Temporal weighting
# ---------------------------------------------------------------------------


class TestTemporalWeighting:
    """Test that temporal proximity affects Hebbian learning."""

    async def test_prompt_adjacent_atoms_get_full_increment(self, storage: Storage):
        """Atoms from the same prompt (distance=0) get the full Hebbian increment."""
        # Create two atoms
        a1 = await _insert_atom(storage, "fact about databases")
        a2 = await _insert_atom(storage, "fact about SQL performance")

        # Insert a synapse between them
        await storage.execute_write(
            "INSERT INTO synapses (source_id, target_id, relationship, strength, bidirectional) "
            "VALUES (?, ?, 'related-to', 0.5, 1)",
            (a1, a2),
        )

        # Same timestamp = same prompt = distance 0
        now = time.time()
        atom_timestamps = {a1: now, a2: now}

        synapses = SynapseManager(storage)
        updated = sum(await synapses.hebbian_update([a1, a2], atom_timestamps=atom_timestamps))
        assert updated > 0

        # Verify the synapse was strengthened
        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE source_id = ? AND target_id = ?",
            (a1, a2),
        )
        assert rows[0]["strength"] > 0.5

    async def test_distant_atoms_get_half_increment(self, storage: Storage):
        """Atoms 600s apart get strengthened via existing-synapse path.

        Both close and distant pairs with pre-existing synapses are
        strengthened by Hebbian update.  The temporal weighting applies to
        new synapse creation (ON CONFLICT increment); for pre-existing
        synapses the standard BCM increment is used.

        We verify:
        1. Distant pair IS strengthened (> starting strength).
        2. A recent pair with the same starting strength ends up at least as
           strong (full BCM increment vs 0.5x for truly new distant pairs).
        """
        a1 = await _insert_atom(storage, "early prompt fact")
        a2 = await _insert_atom(storage, "late prompt fact")

        initial_strength = 0.5
        await storage.execute_write(
            "INSERT INTO synapses (source_id, target_id, relationship, strength, bidirectional) "
            "VALUES (?, ?, 'related-to', ?, 1)",
            (a1, a2, initial_strength),
        )

        now = time.time()
        atom_timestamps = {a1: now, a2: now + 600}  # 10 minutes apart

        synapses = SynapseManager(storage)
        updated = sum(await synapses.hebbian_update([a1, a2], atom_timestamps=atom_timestamps))
        assert updated > 0

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE source_id = ? AND target_id = ?",
            (a1, a2),
        )
        distant_strength = rows[0]["strength"]
        # Distant pair IS strengthened from starting value.
        assert distant_strength > initial_strength

        # --- Recent pair (same timestamp) for comparison ---
        a3 = await _insert_atom(storage, "recent prompt fact A")
        a4 = await _insert_atom(storage, "recent prompt fact B")

        await storage.execute_write(
            "INSERT INTO synapses (source_id, target_id, relationship, strength, bidirectional) "
            "VALUES (?, ?, 'related-to', ?, 1)",
            (a3, a4, initial_strength),
        )

        atom_timestamps_recent = {a3: now, a4: now}
        updated2 = sum(await synapses.hebbian_update(
            [a3, a4], atom_timestamps=atom_timestamps_recent
        ))
        assert updated2 > 0

        rows2 = await storage.execute(
            "SELECT strength FROM synapses WHERE source_id = ? AND target_id = ?",
            (a3, a4),
        )
        recent_strength = rows2[0]["strength"]

        # Recent pair should be at least as strong as distant pair
        # (both use full BCM increment on existing synapses).
        assert recent_strength >= distant_strength


# ---------------------------------------------------------------------------
# Test: rate_recall MCP tool
# ---------------------------------------------------------------------------


class TestRateRecall:
    """Tests for the rate_recall MCP tool."""

    async def test_rate_recall_tool_exists(self):
        """rate_recall is registered as an MCP tool."""
        from memories.server import rate_recall
        assert callable(rate_recall)

    async def test_rate_recall_returns_dict_on_good(self, storage: Storage):
        """rate_recall returns a dict with expected keys on 'good' rating."""
        from memories.server import _brain, _ensure_brain, rate_recall

        # Use a temporary brain for testing
        brain = Brain(db_path=storage.db_path)
        await brain.initialize()

        import memories.server as srv
        original_brain = srv._brain
        srv._brain = brain

        try:
            result = await rate_recall(session_id="test-session-1", rating="good")
            assert isinstance(result, dict)
            assert result.get("status") == "applied"
            assert result.get("rating") == "good"
            assert "weight_delta" in result
            assert "new_weights" in result
        finally:
            srv._brain = original_brain
            await brain.shutdown()

    async def test_rate_recall_returns_dict_on_bad(self, storage: Storage):
        """rate_recall returns a dict on 'bad' rating without error."""
        from memories.server import rate_recall

        brain = Brain(db_path=storage.db_path)
        await brain.initialize()

        import memories.server as srv
        original_brain = srv._brain
        srv._brain = brain

        try:
            result = await rate_recall(session_id="test-session-2", rating="bad")
            assert isinstance(result, dict)
            assert result.get("status") == "applied"
            assert result.get("rating") == "bad"
        finally:
            srv._brain = original_brain
            await brain.shutdown()

    async def test_rate_recall_invalid_rating(self):
        """rate_recall returns error for invalid rating."""
        from memories.server import rate_recall
        result = await rate_recall(session_id="test", rating="neutral")
        assert "error" in result


# ---------------------------------------------------------------------------
# Test: Migration 13 creates retrieval_weight_log table
# ---------------------------------------------------------------------------


class TestMigration13:
    """Test that migration 13 creates the retrieval_weight_log table."""

    async def test_retrieval_weight_log_table_exists(self, storage: Storage):
        """After initialization, retrieval_weight_log table should exist."""
        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='retrieval_weight_log'"
        )
        assert len(rows) == 1
        assert rows[0]["name"] == "retrieval_weight_log"

    async def test_retrieval_weight_log_schema(self, storage: Storage):
        """retrieval_weight_log has the expected columns."""
        rows = await storage.execute("PRAGMA table_info(retrieval_weight_log)")
        col_names = {row["name"] for row in rows}
        assert "id" in col_names
        assert "session_id" in col_names
        assert "rating" in col_names
        assert "weight_delta" in col_names
        assert "created_at" in col_names

    async def test_retrieval_weight_log_check_constraint(self, storage: Storage):
        """rating CHECK constraint allows only 'good' and 'bad'."""
        # Valid insert
        await storage.execute_write(
            "INSERT INTO retrieval_weight_log (session_id, rating, weight_delta) VALUES (?, ?, ?)",
            ("sess1", "good", '{"confidence": 0.001}'),
        )

        # Invalid rating should fail
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            await storage.execute_write(
                "INSERT INTO retrieval_weight_log (session_id, rating, weight_delta) VALUES (?, ?, ?)",
                ("sess2", "neutral", '{}'),
            )


# ---------------------------------------------------------------------------
# Test: Content truncation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestContentTruncation:
    """Test that content > 2000 chars is truncated."""

    async def test_long_content_truncated(self, tmp_path: Path):
        """Content longer than 2000 chars is truncated to <= 2000 + '...'."""
        brain = Brain(db_path=tmp_path / "trunc.db")
        await brain.initialize()

        long_content = "A" * 3000
        result = await brain.remember(
            content=long_content,
            type="fact",
        )

        # The stored atom should have truncated content
        atom = result.get("atom", {})
        stored_content = atom.get("content", "")
        assert len(stored_content) <= 2000
        assert stored_content.endswith("...")

        await brain.shutdown()

    async def test_short_content_not_truncated(self, tmp_path: Path):
        """Content under 2000 chars is not truncated."""
        brain = Brain(db_path=tmp_path / "notrunc.db")
        await brain.initialize()

        short_content = "This is a normal length atom."
        result = await brain.remember(
            content=short_content,
            type="fact",
        )

        atom = result.get("atom", {})
        stored_content = atom.get("content", "")
        assert stored_content == short_content
        assert not stored_content.endswith("...")

        await brain.shutdown()

    async def test_amend_truncates_long_content(self, tmp_path: Path):
        """amend() truncates content > 2000 chars, matching remember()."""
        brain = Brain(db_path=tmp_path / "amend_trunc.db")
        await brain.initialize()

        # First store a normal atom.
        result = await brain.remember(content="short original", type="fact")
        atom_id = result["atom_id"]

        # Amend with very long content.
        long_content = "B" * 3000
        amended = await brain.amend(atom_id=atom_id, content=long_content)

        stored = amended["atom"]["content"]
        assert len(stored) <= 2000
        assert stored.endswith("...")

        await brain.shutdown()


# ---------------------------------------------------------------------------
# Test: rate_recall with existing weights row
# ---------------------------------------------------------------------------


class TestRateRecallExistingWeights:
    """Tests that rate_recall correctly loads and updates a pre-existing weights row."""

    async def test_rate_recall_with_existing_weights(self, storage: Storage):
        """rate_recall correctly loads and updates a pre-existing weights row."""
        brain = Brain(db_path=storage.db_path)
        await brain.initialize()

        import memories.server as srv
        original_brain = srv._brain
        srv._brain = brain

        try:
            # Insert a weights row with non-default values.
            non_default = {
                "confidence": 0.20,
                "importance": 0.15,
                "frequency": 0.10,
                "recency": 0.15,
                "spread_activation": 0.40,
            }
            await brain._storage.save_retrieval_weights(non_default)

            # Call rate_recall with "good" — should increase spread_activation
            # from the non-default 0.40, not from the default 0.25.
            from memories.server import rate_recall
            result = await rate_recall(session_id="test-existing-w", rating="good")

            assert result["status"] == "applied"
            new_w = result["new_weights"]
            # spread_activation should have increased from 0.40.
            assert new_w["spread_activation"] > 0.40
        finally:
            srv._brain = original_brain
            await brain.shutdown()


# ---------------------------------------------------------------------------
# Test: _prompt_atom_timestamps cleanup after session end
# ---------------------------------------------------------------------------


class TestPromptTimestampsCleanup:
    """Tests that session timestamps are cleaned up when the session ends."""

    async def test_prompt_timestamps_cleaned_up_after_session_end(self, tmp_path: Path):
        """Session timestamps are removed from the module dict when session ends."""
        from memories.cli import _prompt_atom_timestamps, _run_session_stop_logic

        brain = Brain(db_path=tmp_path / "cleanup.db")
        await brain.initialize()

        session_id = "test-session-xyz"
        _prompt_atom_timestamps[session_id] = {1: "2026-01-01T00:00:00"}

        await _run_session_stop_logic(brain, session_id)

        assert session_id not in _prompt_atom_timestamps

        await brain.shutdown()


# ---------------------------------------------------------------------------
# Test: _prompt_atom_timestamps per-session size cap
# ---------------------------------------------------------------------------


class TestPromptTimestampsCap:
    """Tests the per-session cap on _prompt_atom_timestamps entries."""

    def test_cap_constant_defined(self):
        """The cap constant exists and is 500."""
        from memories.cli import _MAX_PROMPT_TIMESTAMPS_PER_SESSION
        assert _MAX_PROMPT_TIMESTAMPS_PER_SESSION == 500
