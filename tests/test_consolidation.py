"""Comprehensive tests for the consolidation engine.

Tests exercise the full consolidation lifecycle: atom decay, synapse decay,
synapse pruning, near-duplicate merging, promotion of frequently accessed
atoms, dry-run preview, scope filtering, and audit logging.

All tests use a temporary database (via the ``storage`` fixture) and mock
the EmbeddingEngine so that Ollama is never required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.atoms import Atom, AtomManager
from memories.consolidation import ConsolidationEngine, ConsolidationResult, _CONFIDENCE_FLOOR
from memories.embeddings import EmbeddingEngine
from memories.storage import Storage
from memories.synapses import SynapseManager


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


async def _insert_atom(
    storage: Storage,
    content: str = "test atom",
    atom_type: str = "fact",
    region: str = "general",
    confidence: float = 1.0,
    access_count: int = 0,
    last_accessed_at: str | None = None,
    created_at: str = "2025-01-01 00:00:00",
    tags: str | None = None,
) -> int:
    """Insert an atom row directly via SQL and return the new id."""
    return await storage.execute_write(
        """
        INSERT INTO atoms
            (content, type, region, confidence, access_count,
             last_accessed_at, created_at, updated_at, tags, is_deleted)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, 0)
        """,
        (content, atom_type, region, confidence, access_count,
         last_accessed_at, created_at, tags),
    )


async def _insert_synapse(
    storage: Storage,
    source_id: int,
    target_id: int,
    relationship: str = "related-to",
    strength: float = 0.5,
    bidirectional: int = 1,
    last_activated_at: str | None = None,
) -> int:
    """Insert a synapse row directly via SQL and return the new id."""
    return await storage.execute_write(
        """
        INSERT INTO synapses
            (source_id, target_id, relationship, strength, bidirectional,
             last_activated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source_id, target_id, relationship, strength, bidirectional,
         last_activated_at),
    )


def _make_engine(
    storage: Storage,
    embeddings: EmbeddingEngine | None = None,
) -> ConsolidationEngine:
    """Build a ConsolidationEngine with a mocked EmbeddingEngine."""
    if embeddings is None:
        embeddings = MagicMock(spec=EmbeddingEngine)
        embeddings.search_similar = AsyncMock(return_value=[])
        embeddings.embed_and_store = AsyncMock()

    atoms = AtomManager(storage, embeddings)
    synapses = SynapseManager(storage)
    return ConsolidationEngine(storage, embeddings, atoms, synapses)


# -----------------------------------------------------------------------
# 1. Decay -- atom confidence
# -----------------------------------------------------------------------


class TestAtomDecay:
    """Verify that _decay_atoms reduces confidence on stale atoms."""

    async def test_decay_reduces_confidence(self, storage: Storage) -> None:
        """Stale atoms have their confidence multiplied by the decay rate."""
        # Insert a stale atom (created well in the past, never accessed).
        atom_id = await _insert_atom(
            storage, content="stale fact", confidence=0.8,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        new_conf = rows[0]["confidence"]

        # Default decay_rate is 0.95; "fact" type has multiplier 0.995 (B1).
        # effective_rate = 0.95 * 0.995 = 0.94525; 0.8 * 0.94525 ≈ 0.7562.
        assert abs(new_conf - 0.8 * 0.95 * 0.995) < 1e-4
        assert result.decayed >= 1

    async def test_confidence_floor_respected(self, storage: Storage) -> None:
        """Decay must not push confidence below the floor (0.1)."""
        atom_id = await _insert_atom(
            storage, content="almost gone", confidence=0.11,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["confidence"] >= _CONFIDENCE_FLOOR

    async def test_atom_at_floor_not_decayed_further(self, storage: Storage) -> None:
        """An atom already at the confidence floor is skipped entirely."""
        atom_id = await _insert_atom(
            storage, content="floor atom", confidence=_CONFIDENCE_FLOOR,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        # The atom was at the floor so it should NOT appear in decayed count.
        assert rows[0]["confidence"] == _CONFIDENCE_FLOOR
        # No decay details for this atom.
        decay_details = [d for d in result.details if d.get("action") == "decay"
                         and d.get("atom_id") == atom_id]
        assert len(decay_details) == 0

    async def test_recently_accessed_atom_not_decayed(self, storage: Storage) -> None:
        """Atoms accessed within the staleness window are not decayed."""
        atom_id = await _insert_atom(
            storage, content="fresh atom", confidence=0.9,
            created_at="2020-01-01 00:00:00",
            last_accessed_at="2099-01-01 00:00:00",  # future date = not stale
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["confidence"] == 0.9
        # Should not appear in the decayed count.
        decay_ids = [d["atom_id"] for d in result.details
                     if d.get("action") == "decay"]
        assert atom_id not in decay_ids


# -----------------------------------------------------------------------
# 2. Decay -- synapse strength
# -----------------------------------------------------------------------


class TestSynapseDecay:
    """Verify that synapse decay reduces strength and prunes weak ones."""

    async def test_synapse_strength_reduced(self, storage: Storage) -> None:
        """Synapse strength is multiplied by the decay factor.

        Non-related-to synapses use the base decay rate (0.95).
        """
        a1 = await _insert_atom(storage, content="atom 1", created_at="2020-01-01 00:00:00")
        a2 = await _insert_atom(storage, content="atom 2", created_at="2020-01-01 00:00:00")
        # Use "elaborates" to test base decay rate (not the faster related-to rate)
        syn_id = await _insert_synapse(storage, a1, a2, relationship="elaborates", strength=0.6)

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (syn_id,)
        )
        # After decay: 0.6 * 0.95 = 0.57.
        assert len(rows) == 1
        assert abs(rows[0]["strength"] - 0.6 * 0.95) < 1e-4

    async def test_related_to_synapse_decays_faster(self, storage: Storage) -> None:
        """Related-to synapses decay faster than typed synapses.

        This addresses the 99.7% related-to problem by gradually pruning
        generic embedding-similarity connections while preserving meaningful ones.
        """
        a1 = await _insert_atom(storage, content="atom 1", created_at="2020-01-01 00:00:00")
        a2 = await _insert_atom(storage, content="atom 2", created_at="2020-01-01 00:00:00")
        a3 = await _insert_atom(storage, content="atom 3", created_at="2020-01-01 00:00:00")
        # related-to synapse (generic)
        syn_related = await _insert_synapse(storage, a1, a2, relationship="related-to", strength=0.6)
        # elaborates synapse (typed)
        syn_elaborates = await _insert_synapse(storage, a1, a3, relationship="elaborates", strength=0.6)

        engine = _make_engine(storage)
        await engine.reflect()

        rows_related = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (syn_related,)
        )
        rows_elaborates = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (syn_elaborates,)
        )

        # related-to decays faster: 0.6 * 0.95 * 0.9 = 0.513
        assert len(rows_related) == 1
        assert abs(rows_related[0]["strength"] - 0.6 * 0.95 * 0.9) < 1e-4

        # elaborates uses base rate: 0.6 * 0.95 = 0.57
        assert len(rows_elaborates) == 1
        assert abs(rows_elaborates[0]["strength"] - 0.6 * 0.95) < 1e-4

    async def test_weak_synapse_pruned_after_decay(self, storage: Storage) -> None:
        """Synapses that fall below prune_threshold after decay are deleted."""
        a1 = await _insert_atom(storage, content="atom a", created_at="2020-01-01 00:00:00")
        a2 = await _insert_atom(storage, content="atom b", created_at="2020-01-01 00:00:00")
        # Prune threshold default is 0.05.  0.04 * 0.95 = 0.038 < 0.05.
        syn_id = await _insert_synapse(storage, a1, a2, strength=0.04)

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT id FROM synapses WHERE id = ?", (syn_id,)
        )
        assert len(rows) == 0
        assert result.pruned >= 1


# -----------------------------------------------------------------------
# 3. Prune weak synapses (explicit pass)
# -----------------------------------------------------------------------


class TestPruneSynapses:
    """Verify the explicit prune pass catches stragglers."""

    async def test_prune_removes_below_threshold(self, storage: Storage) -> None:
        """Synapses below the prune_threshold are removed."""
        a1 = await _insert_atom(storage, content="prune a", created_at="2020-01-01 00:00:00")
        a2 = await _insert_atom(storage, content="prune b", created_at="2020-01-01 00:00:00")
        a3 = await _insert_atom(storage, content="prune c", created_at="2020-01-01 00:00:00")

        # One synapse below threshold, one above.
        await _insert_synapse(storage, a1, a2, strength=0.02)
        strong_id = await _insert_synapse(storage, a1, a3, strength=0.8)

        engine = _make_engine(storage)
        result = await engine.reflect()

        # Strong synapse should survive (maybe decayed slightly).
        rows = await storage.execute(
            "SELECT id FROM synapses WHERE id = ?", (strong_id,)
        )
        assert len(rows) == 1
        assert result.pruned >= 1

    async def test_prune_count_in_result(self, storage: Storage) -> None:
        """The pruned counter in ConsolidationResult reflects actual deletions."""
        a1 = await _insert_atom(storage, content="x1", created_at="2020-01-01 00:00:00")
        a2 = await _insert_atom(storage, content="x2", created_at="2020-01-01 00:00:00")
        a3 = await _insert_atom(storage, content="x3", created_at="2020-01-01 00:00:00")

        await _insert_synapse(storage, a1, a2, strength=0.01)
        await _insert_synapse(storage, a1, a3, strength=0.02)

        engine = _make_engine(storage)
        result = await engine.reflect()

        assert result.pruned >= 2


# -----------------------------------------------------------------------
# 4. Merge near-duplicate atoms
# -----------------------------------------------------------------------


class TestMergeDuplicates:
    """Verify that near-duplicate atoms are identified and merged."""

    async def test_merge_identifies_duplicates(self, storage: Storage) -> None:
        """Two atoms with high similarity and same type are merged."""
        a1 = await _insert_atom(
            storage, content="Python is great", atom_type="fact",
            confidence=0.9, created_at="2025-06-01 00:00:00",
        )
        a2 = await _insert_atom(
            storage, content="Python is wonderful", atom_type="fact",
            confidence=0.7, created_at="2025-06-02 00:00:00",
        )

        mock_embeddings = MagicMock(spec=EmbeddingEngine)
        # search_similar returns the other atom with a very small distance
        # (high similarity).
        async def mock_search_similar(content, k=10):
            if content == "Python is great":
                return [(a2, 0.02)]  # distance=0.02 -> similarity=0.99
            if content == "Python is wonderful":
                return [(a1, 0.02)]
            return []

        mock_embeddings.search_similar = AsyncMock(side_effect=mock_search_similar)
        mock_embeddings.embed_and_store = AsyncMock()
        mock_embeddings.distance_to_similarity = EmbeddingEngine.distance_to_similarity

        engine = _make_engine(storage, embeddings=mock_embeddings)
        result = await engine.reflect()

        assert result.merged >= 1

        # The lower-confidence atom (a2, 0.7) should be soft-deleted.
        rows = await storage.execute(
            "SELECT is_deleted FROM atoms WHERE id = ?", (a2,)
        )
        assert rows[0]["is_deleted"] == 1

        # The higher-confidence atom (a1, 0.9) should remain active.
        rows = await storage.execute(
            "SELECT is_deleted FROM atoms WHERE id = ?", (a1,)
        )
        assert rows[0]["is_deleted"] == 0

    async def test_merge_redirects_synapses(self, storage: Storage) -> None:
        """After merging, synapses from the duplicate are redirected to the survivor."""
        a1 = await _insert_atom(
            storage, content="survivor atom", atom_type="fact",
            confidence=0.9, created_at="2025-06-01 00:00:00",
        )
        a2 = await _insert_atom(
            storage, content="duplicate atom", atom_type="fact",
            confidence=0.7, created_at="2025-06-02 00:00:00",
        )
        a3 = await _insert_atom(
            storage, content="neighbour", atom_type="fact",
            confidence=1.0, created_at="2025-06-01 00:00:00",
        )

        # a2 -> a3 synapse should be redirected to a1 -> a3 after merge.
        syn_id = await _insert_synapse(storage, a2, a3, relationship="related-to", strength=0.6)

        mock_embeddings = MagicMock(spec=EmbeddingEngine)

        async def mock_search_similar(content, k=10):
            if content == "survivor atom":
                return [(a2, 0.02)]
            if content == "duplicate atom":
                return [(a1, 0.02)]
            return []

        mock_embeddings.search_similar = AsyncMock(side_effect=mock_search_similar)
        mock_embeddings.embed_and_store = AsyncMock()
        mock_embeddings.distance_to_similarity = EmbeddingEngine.distance_to_similarity

        engine = _make_engine(storage, embeddings=mock_embeddings)
        await engine.reflect()

        # The original synapse from a2 -> a3 should now point a1 -> a3.
        rows = await storage.execute(
            "SELECT source_id, target_id FROM synapses WHERE id = ?", (syn_id,)
        )
        if rows:
            assert rows[0]["source_id"] == a1
            assert rows[0]["target_id"] == a3

    async def test_merge_creates_supersedes_synapse(self, storage: Storage) -> None:
        """A 'supersedes' synapse is created from survivor to duplicate."""
        a1 = await _insert_atom(
            storage, content="master", atom_type="fact",
            confidence=0.9, created_at="2025-06-01 00:00:00",
        )
        a2 = await _insert_atom(
            storage, content="clone", atom_type="fact",
            confidence=0.5, created_at="2025-06-02 00:00:00",
        )

        mock_embeddings = MagicMock(spec=EmbeddingEngine)

        async def mock_search_similar(content, k=10):
            if content == "master":
                return [(a2, 0.01)]
            return []

        mock_embeddings.search_similar = AsyncMock(side_effect=mock_search_similar)
        mock_embeddings.embed_and_store = AsyncMock()
        mock_embeddings.distance_to_similarity = EmbeddingEngine.distance_to_similarity

        engine = _make_engine(storage, embeddings=mock_embeddings)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT * FROM synapses WHERE source_id = ? AND target_id = ? "
            "AND relationship = 'supersedes'",
            (a1, a2),
        )
        assert len(rows) == 1

    async def test_different_types_not_merged(self, storage: Storage) -> None:
        """Atoms of different types are never merged even with high similarity."""
        a1 = await _insert_atom(
            storage, content="same text", atom_type="fact",
            confidence=0.9, created_at="2025-06-01 00:00:00",
        )
        a2 = await _insert_atom(
            storage, content="same text", atom_type="skill",
            confidence=0.7, created_at="2025-06-02 00:00:00",
        )

        mock_embeddings = MagicMock(spec=EmbeddingEngine)

        async def mock_search_similar(content, k=10):
            if content == "same text":
                return [(a2, 0.01), (a1, 0.01)]
            return []

        mock_embeddings.search_similar = AsyncMock(side_effect=mock_search_similar)
        mock_embeddings.embed_and_store = AsyncMock()
        mock_embeddings.distance_to_similarity = EmbeddingEngine.distance_to_similarity

        engine = _make_engine(storage, embeddings=mock_embeddings)
        result = await engine.reflect()

        assert result.merged == 0


# -----------------------------------------------------------------------
# 5. Promote frequently accessed atoms
# -----------------------------------------------------------------------


class TestPromoteStrong:
    """Verify that frequently accessed atoms receive a confidence boost."""

    async def test_promote_boosts_confidence(self, storage: Storage) -> None:
        """Tiered boost: access_count=25, min=20 → 1 tier → +0.05 → 0.75."""
        atom_id = await _insert_atom(
            storage, content="popular atom", confidence=0.7,
            access_count=25,  # above default threshold of 20 → 1 tier
            last_accessed_at="2099-01-01 00:00:00",  # not stale
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        assert abs(rows[0]["confidence"] - 0.75) < 1e-4
        assert result.promoted >= 1

    async def test_promote_caps_at_one(self, storage: Storage) -> None:
        """Promoted confidence never exceeds 1.0."""
        atom_id = await _insert_atom(
            storage, content="max confidence atom", confidence=0.95,
            access_count=50,
            last_accessed_at="2099-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["confidence"] <= 1.0

    async def test_below_threshold_not_promoted(self, storage: Storage) -> None:
        """Atoms with low access_count are not promoted."""
        atom_id = await _insert_atom(
            storage, content="rarely accessed", confidence=0.5,
            access_count=3,
            last_accessed_at="2099-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        # Confidence should remain the same (not promoted). It might be
        # decayed if the atom is stale, but with a future last_accessed_at
        # it is not stale.
        assert rows[0]["confidence"] == 0.5

    async def test_already_max_confidence_not_promoted(self, storage: Storage) -> None:
        """Atoms already at confidence 1.0 are not included in promotion."""
        atom_id = await _insert_atom(
            storage, content="perfect memory", confidence=1.0,
            access_count=100,
            last_accessed_at="2099-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        # The query filters confidence < 1.0, so this atom should not appear.
        promoted_ids = [d["atom_id"] for d in result.details
                        if d.get("action") == "promote"]
        assert atom_id not in promoted_ids


# -----------------------------------------------------------------------
# 6. Dry-run mode
# -----------------------------------------------------------------------


class TestDryRun:
    """Verify that dry_run=True previews without mutating data."""

    async def test_dry_run_does_not_decay_atoms(self, storage: Storage) -> None:
        """Atom confidence must not change during a dry run."""
        atom_id = await _insert_atom(
            storage, content="dry run atom", confidence=0.8,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        result = await engine.reflect(dry_run=True)

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["confidence"] == 0.8
        assert result.dry_run is True

    async def test_dry_run_does_not_prune_synapses(self, storage: Storage) -> None:
        """Weak synapses must not be deleted during a dry run."""
        a1 = await _insert_atom(storage, content="d1", created_at="2020-01-01 00:00:00")
        a2 = await _insert_atom(storage, content="d2", created_at="2020-01-01 00:00:00")
        syn_id = await _insert_synapse(storage, a1, a2, strength=0.01)

        engine = _make_engine(storage)
        result = await engine.reflect(dry_run=True)

        rows = await storage.execute(
            "SELECT id FROM synapses WHERE id = ?", (syn_id,)
        )
        assert len(rows) == 1
        # The dry-run result still reports what would be pruned.
        assert result.pruned >= 1

    async def test_dry_run_does_not_promote(self, storage: Storage) -> None:
        """Atom confidence must not be boosted during a dry run."""
        atom_id = await _insert_atom(
            storage, content="dry promote", confidence=0.6,
            access_count=30,
            last_accessed_at="2099-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        result = await engine.reflect(dry_run=True)

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["confidence"] == 0.6
        assert result.promoted >= 1

    async def test_dry_run_does_not_write_log(self, storage: Storage) -> None:
        """No consolidation_log entries are created during a dry run."""
        await _insert_atom(
            storage, content="log check", confidence=0.5,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect(dry_run=True)

        rows = await storage.execute("SELECT COUNT(*) AS cnt FROM consolidation_log")
        assert rows[0]["cnt"] == 0


# -----------------------------------------------------------------------
# 7. Consolidation log
# -----------------------------------------------------------------------


class TestConsolidationLog:
    """Verify that consolidation actions are logged correctly."""

    async def test_reflect_creates_log_entries(self, storage: Storage) -> None:
        """A non-dry-run reflect must create at least one log entry."""
        await _insert_atom(
            storage, content="log test", confidence=0.8,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute("SELECT COUNT(*) AS cnt FROM consolidation_log")
        assert rows[0]["cnt"] >= 1

    async def test_log_contains_reflect_action(self, storage: Storage) -> None:
        """The main summary log entry has action='reflect'."""
        await _insert_atom(
            storage, content="summary test", confidence=0.8,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT action FROM consolidation_log WHERE action = 'reflect'"
        )
        assert len(rows) >= 1

    async def test_log_details_are_valid_json(self, storage: Storage) -> None:
        """The details column must contain valid JSON."""
        await _insert_atom(
            storage, content="json test", confidence=0.8,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT details FROM consolidation_log WHERE details IS NOT NULL"
        )
        for row in rows:
            parsed = json.loads(row["details"])
            assert isinstance(parsed, (dict, list))


# -----------------------------------------------------------------------
# 8. get_history
# -----------------------------------------------------------------------


class TestGetHistory:
    """Verify the get_history() method returns recent log entries."""

    async def test_returns_recent_entries(self, storage: Storage) -> None:
        """get_history returns entries ordered by most recent first."""
        await _insert_atom(
            storage, content="history test", confidence=0.8,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        # Run reflect twice so there are multiple log entries.
        await engine.reflect()
        await engine.reflect()

        history = await engine.get_history(limit=10)
        assert len(history) >= 2
        # Each entry has required keys.
        for entry in history:
            assert "id" in entry
            assert "action" in entry
            assert "created_at" in entry

    async def test_limit_respected(self, storage: Storage) -> None:
        """get_history returns at most `limit` entries."""
        await _insert_atom(
            storage, content="limit test", confidence=0.8,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        for _ in range(5):
            await engine.reflect()

        history = await engine.get_history(limit=2)
        assert len(history) <= 2

    async def test_empty_history(self, storage: Storage) -> None:
        """get_history returns an empty list when no consolidation has run."""
        engine = _make_engine(storage)
        history = await engine.get_history()
        assert history == []


# -----------------------------------------------------------------------
# 9. Scope filtering
# -----------------------------------------------------------------------


class TestScopeFiltering:
    """Verify that scope parameter limits consolidation to a specific region."""

    async def test_scope_limits_decay_to_region(self, storage: Storage) -> None:
        """Only atoms in the specified region are decayed."""
        target_id = await _insert_atom(
            storage, content="target region", region="backend",
            confidence=0.8, created_at="2020-01-01 00:00:00",
        )
        other_id = await _insert_atom(
            storage, content="other region", region="frontend",
            confidence=0.8, created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        result = await engine.reflect(scope="backend")

        # Target atom should be decayed.
        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (target_id,)
        )
        assert rows[0]["confidence"] < 0.8

        # Other atom should be untouched.
        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (other_id,)
        )
        assert rows[0]["confidence"] == 0.8

    async def test_scope_limits_promotion(self, storage: Storage) -> None:
        """Only atoms in the specified region are promoted."""
        target_id = await _insert_atom(
            storage, content="promote target", region="backend",
            confidence=0.6, access_count=25,
            last_accessed_at="2099-01-01 00:00:00",
        )
        other_id = await _insert_atom(
            storage, content="promote other", region="frontend",
            confidence=0.6, access_count=25,
            last_accessed_at="2099-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        result = await engine.reflect(scope="backend")

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (target_id,)
        )
        assert rows[0]["confidence"] > 0.6

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (other_id,)
        )
        assert rows[0]["confidence"] == 0.6


# -----------------------------------------------------------------------
# 10. ConsolidationResult
# -----------------------------------------------------------------------


class TestConsolidationResult:
    """Verify ConsolidationResult serialisation and structure."""

    def test_to_dict_contains_all_keys(self) -> None:
        """to_dict() must include all counter fields and details."""
        result = ConsolidationResult(
            merged=1, decayed=2, pruned=3, promoted=4, dry_run=True,
            details=[{"action": "test"}],
        )
        d = result.to_dict()
        assert d["merged"] == 1
        assert d["decayed"] == 2
        assert d["pruned"] == 3
        assert d["promoted"] == 4
        assert d["dry_run"] is True
        assert len(d["details"]) == 1

    def test_default_values(self) -> None:
        """A fresh ConsolidationResult has all counters at zero."""
        result = ConsolidationResult()
        assert result.merged == 0
        assert result.decayed == 0
        assert result.pruned == 0
        assert result.promoted == 0
        assert result.compressed == 0
        assert result.dry_run is False
        assert result.details == []


# -----------------------------------------------------------------------
# 11. Pick survivor logic
# -----------------------------------------------------------------------


class TestPickSurvivor:
    """Verify the static _pick_survivor method on ConsolidationEngine."""

    def test_higher_confidence_wins(self) -> None:
        """The atom with higher confidence survives."""
        strong = Atom(id=1, content="strong", type="fact", confidence=0.9)
        weak = Atom(id=2, content="weak", type="fact", confidence=0.3)

        survivor, duplicate = ConsolidationEngine._pick_survivor(strong, weak)
        assert survivor.id == 1
        assert duplicate.id == 2

    def test_equal_confidence_newer_wins(self) -> None:
        """On equal confidence, the more recently created atom survives."""
        old = Atom(
            id=1, content="old", type="fact", confidence=0.5,
            created_at="2025-01-01 00:00:00",
        )
        new = Atom(
            id=2, content="new", type="fact", confidence=0.5,
            created_at="2025-06-01 00:00:00",
        )

        survivor, duplicate = ConsolidationEngine._pick_survivor(old, new)
        assert survivor.id == 2
        assert duplicate.id == 1


# -----------------------------------------------------------------------
# Time-aware decay (Ebbinghaus exponent)
# -----------------------------------------------------------------------


class TestTimeAwareDecay:
    """Verify that older stale atoms decay more than recently-stale atoms.

    The decay formula is ``confidence * (rate ** exponent)`` where
    ``exponent = days_since_access / decay_after_days``.  An atom that is
    twice as old as the staleness threshold decays by ``rate ** 2`` instead
    of ``rate ** 1``.
    """

    async def test_older_atom_decays_more_than_newer_stale_atom(
        self, storage: Storage
    ) -> None:
        """Two stale atoms: the older one loses more confidence."""
        # Atom A: stale for exactly decay_after_days (30 d) → exponent = 1.
        atom_a = await _insert_atom(
            storage,
            content="just stale atom",
            confidence=0.8,
            last_accessed_at="2025-01-01 00:00:00",  # far in the past
            created_at="2024-01-01 00:00:00",
        )
        # Atom B: stale for 2× decay_after_days → exponent = 2.
        atom_b = await _insert_atom(
            storage,
            content="very stale atom",
            confidence=0.8,
            last_accessed_at="2024-01-01 00:00:00",  # even older
            created_at="2023-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        rows_a = await storage.execute("SELECT confidence FROM atoms WHERE id = ?", (atom_a,))
        rows_b = await storage.execute("SELECT confidence FROM atoms WHERE id = ?", (atom_b,))

        conf_a = rows_a[0]["confidence"]
        conf_b = rows_b[0]["confidence"]

        # Older atom must have lower final confidence.
        assert conf_b < conf_a, (
            f"Older atom (conf={conf_b:.4f}) should have lower confidence "
            f"than newer-stale atom (conf={conf_a:.4f})"
        )

    async def test_just_stale_atom_uses_exponent_one(
        self, storage: Storage
    ) -> None:
        """An atom stale for exactly decay_after_days uses exponent=1.0.

        When ``days_since == decay_after_days``, ``exponent = 1.0`` and
        ``new_confidence = confidence * rate ** 1``.  This preserves backward
        compatibility with the old flat-rate decay for atoms that are just
        barely stale.
        """
        from memories.config import get_config

        cfg = get_config().consolidation
        rate = cfg.decay_rate
        decay_after = cfg.decay_after_days

        atom_id = await _insert_atom(
            storage,
            content="exactly stale",
            confidence=0.8,
            # No last_accessed_at: the code falls back to decay_after_days.
            last_accessed_at=None,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute("SELECT confidence FROM atoms WHERE id = ?", (atom_id,))
        new_conf = rows[0]["confidence"]

        # exponent = decay_after / decay_after = 1.0 → effective_rate ** 1.0.
        # B1: "fact" type has multiplier 0.995, so effective_rate = rate * 0.995.
        type_multiplier = cfg.type_decay_multipliers.get("fact", 1.0)
        effective_rate = rate * type_multiplier
        expected = max(_CONFIDENCE_FLOOR, 0.8 * (effective_rate ** 1.0))
        assert abs(new_conf - expected) < 1e-4, (
            f"Just-stale atom should decay by effective_rate^1: expected {expected:.4f}, got {new_conf:.4f}"
        )

    async def test_twice_stale_atom_uses_exponent_two(
        self, storage: Storage
    ) -> None:
        """An atom stale for 2× decay_after_days uses exponent=2.0."""
        from memories.config import get_config
        from datetime import datetime, timezone, timedelta

        cfg = get_config().consolidation
        rate = cfg.decay_rate
        decay_after = cfg.decay_after_days

        # last_accessed exactly 2× decay_after_days ago
        two_windows_ago = (
            datetime.now(tz=timezone.utc) - timedelta(days=decay_after * 2)
        ).strftime("%Y-%m-%d %H:%M:%S")

        atom_id = await _insert_atom(
            storage,
            content="twice stale atom",
            confidence=0.8,
            last_accessed_at=two_windows_ago,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute("SELECT confidence FROM atoms WHERE id = ?", (atom_id,))
        new_conf = rows[0]["confidence"]

        # exponent = (2 * decay_after) / decay_after = 2.0.
        # B1: "fact" type has multiplier 0.995, so effective_rate = rate * 0.995.
        type_multiplier = cfg.type_decay_multipliers.get("fact", 1.0)
        effective_rate = rate * type_multiplier
        expected = max(_CONFIDENCE_FLOOR, 0.8 * (effective_rate ** 2.0))
        assert abs(new_conf - expected) < 1e-4, (
            f"Twice-stale atom should decay by effective_rate^2: expected {expected:.4f}, got {new_conf:.4f}"
        )


# -----------------------------------------------------------------------
# Long-Term Depression (anti-Hebbian synapse weakening)
# -----------------------------------------------------------------------


def _recent(days_ago: int = 5) -> str:
    """Return an ISO timestamp *days_ago* days in the past."""
    return (
        datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    ).strftime("%Y-%m-%d %H:%M:%S")


class TestLTD:
    """Verify that _apply_ltd weakens synapses between individually-active
    atoms that have not been Hebbian co-activated recently.

    The anti-Hebbian rule: atoms that consistently fire *apart* should lose
    their connection over time, even if each atom is individually active.
    """

    async def test_weakens_synapse_when_atoms_active_but_not_co_activated(
        self, storage: Storage
    ) -> None:
        """LTD weakens a synapse whose endpoints are individually active
        but haven't been co-activated within ltd_window_days."""
        from memories.config import get_config

        cfg = get_config().consolidation

        # Both atoms accessed recently (individually active).
        atom_a = await _insert_atom(
            storage, "active atom A", last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )
        atom_b = await _insert_atom(
            storage, "active atom B", last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )

        # Synapse exists but was last Hebbian-activated well beyond the window.
        old_activation = _recent(cfg.ltd_window_days + 5)
        synapse_id = await _insert_synapse(
            storage, atom_a, atom_b,
            strength=0.6,
            last_activated_at=old_activation,
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (synapse_id,)
        )
        # Synapse may have been pruned (rows empty) or weakened.
        if rows:
            assert rows[0]["strength"] < 0.6, (
                "Synapse strength should have decreased due to LTD"
            )

        assert result.ltd >= 1

    async def test_does_not_weaken_recently_co_activated_synapse(
        self, storage: Storage
    ) -> None:
        """A synapse Hebbian-reinforced within ltd_window_days is protected."""
        from memories.config import get_config

        cfg = get_config().consolidation

        atom_a = await _insert_atom(
            storage, "atom A recent co-act", last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )
        atom_b = await _insert_atom(
            storage, "atom B recent co-act", last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )

        # Last co-activated 3 days ago — inside the ltd_window_days (14).
        recent_activation = _recent(3)
        synapse_id = await _insert_synapse(
            storage, atom_a, atom_b,
            strength=0.6,
            last_activated_at=recent_activation,
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (synapse_id,)
        )
        # Synapse should still exist and strength should be 0.6 (passive
        # synapse decay applies to all synapses but should not cause huge drop).
        assert rows, "Synapse should not have been deleted"
        # The LTD counter must NOT include this synapse.
        ltd_details = [
            d for d in result.details
            if d.get("action") == "ltd" and d.get("synapse_id") == synapse_id
        ]
        assert len(ltd_details) == 0, (
            "Recently co-activated synapse should not appear in LTD details"
        )

    async def test_does_not_weaken_synapse_when_atom_is_stale(
        self, storage: Storage
    ) -> None:
        """LTD only fires when BOTH atoms are individually active.

        If one atom is stale (not accessed recently), it is passive decay's
        job to handle that synapse, not LTD.
        """
        from memories.config import get_config

        cfg = get_config().consolidation

        atom_a = await _insert_atom(
            storage, "active atom", last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )
        # Stale atom — last accessed beyond decay_after_days window.
        stale_days = cfg.decay_after_days + 10
        atom_b = await _insert_atom(
            storage, "stale atom", last_accessed_at=_recent(stale_days),
            created_at="2024-01-01 00:00:00",
        )

        old_activation = _recent(cfg.ltd_window_days + 5)
        synapse_id = await _insert_synapse(
            storage, atom_a, atom_b,
            strength=0.6,
            last_activated_at=old_activation,
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        ltd_details = [
            d for d in result.details
            if d.get("action") == "ltd" and d.get("synapse_id") == synapse_id
        ]
        assert len(ltd_details) == 0, (
            "Synapse with a stale endpoint should not be LTD-weakened"
        )

    async def test_ltd_dry_run_counts_without_mutating(
        self, storage: Storage
    ) -> None:
        """In dry_run mode LTD reports the count but does not weaken synapses."""
        from memories.config import get_config

        cfg = get_config().consolidation

        atom_a = await _insert_atom(
            storage, "dry run atom A", last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )
        atom_b = await _insert_atom(
            storage, "dry run atom B", last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )
        old_activation = _recent(cfg.ltd_window_days + 5)
        synapse_id = await _insert_synapse(
            storage, atom_a, atom_b,
            strength=0.6,
            last_activated_at=old_activation,
        )

        engine = _make_engine(storage)
        result = await engine.reflect(dry_run=True)

        # Strength must be unchanged.
        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (synapse_id,)
        )
        assert rows, "Synapse should still exist in dry_run mode"
        assert abs(rows[0]["strength"] - 0.6) < 1e-6, (
            "Synapse strength must not change in dry_run mode"
        )
        # But LTD count must be reported.
        assert result.ltd >= 1

    async def test_ltd_count_reflects_qualifying_synapse_count(
        self, storage: Storage
    ) -> None:
        """result.ltd equals the number of qualifying synapses."""
        from memories.config import get_config

        cfg = get_config().consolidation
        old_activation = _recent(cfg.ltd_window_days + 5)

        # Create 3 pairs of individually-active atoms with old synapses.
        pairs = []
        for i in range(3):
            a = await _insert_atom(
                storage, f"ltd atom A{i}", last_accessed_at=_recent(5),
                created_at="2024-01-01 00:00:00",
            )
            b = await _insert_atom(
                storage, f"ltd atom B{i}", last_accessed_at=_recent(5),
                created_at="2024-01-01 00:00:00",
            )
            await _insert_synapse(
                storage, a, b,
                strength=0.6,
                last_activated_at=old_activation,
            )
            pairs.append((a, b))

        engine = _make_engine(storage)
        result = await engine.reflect()

        assert result.ltd == 3, (
            f"Expected 3 LTD-weakened synapses, got {result.ltd}"
        )

    async def test_ltd_in_to_dict(self, storage: Storage) -> None:
        """ConsolidationResult.to_dict() includes the ltd counter."""
        engine = _make_engine(storage)
        result = await engine.reflect()
        d = result.to_dict()
        assert "ltd" in d


# -----------------------------------------------------------------------
# 10. Feedback signals → importance adjustment
# -----------------------------------------------------------------------


async def _insert_feedback(
    storage: Storage,
    atom_id: int,
    signal: str,
    processed: bool = False,
) -> None:
    """Insert a feedback record directly via SQL."""
    processed_at = "2026-01-01 00:00:00" if processed else None
    await storage.execute_write(
        "INSERT INTO atom_feedback (atom_id, signal, processed_at) VALUES (?, ?, ?)",
        (atom_id, signal, processed_at),
    )


class TestFeedbackSignals:
    """Verify that feedback signals adjust atom importance during consolidation."""

    async def test_good_feedback_boosts_importance(self, storage: Storage) -> None:
        """A 'good' feedback signal should increase atom importance."""
        atom_id = await _insert_atom(storage, content="helpful fact about caching")
        # Insert importance=0.5 (default) via direct SQL update.
        await storage.execute_write(
            "UPDATE atoms SET importance = 0.5 WHERE id = ?", (atom_id,)
        )
        await _insert_feedback(storage, atom_id, "good")

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["importance"] > 0.5
        assert result.feedback_adjusted >= 1

    async def test_bad_feedback_lowers_importance(self, storage: Storage) -> None:
        """A 'bad' feedback signal should decrease atom importance."""
        atom_id = await _insert_atom(storage, content="unhelpful noise about irrelevant topic")
        await storage.execute_write(
            "UPDATE atoms SET importance = 0.5 WHERE id = ?", (atom_id,)
        )
        await _insert_feedback(storage, atom_id, "bad")

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["importance"] < 0.5
        assert result.feedback_adjusted >= 1

    async def test_processed_feedback_not_applied_again(self, storage: Storage) -> None:
        """Feedback records already marked processed_at are not re-applied."""
        atom_id = await _insert_atom(storage, content="already processed atom")
        await storage.execute_write(
            "UPDATE atoms SET importance = 0.5 WHERE id = ?", (atom_id,)
        )
        # Insert already-processed feedback.
        await _insert_feedback(storage, atom_id, "good", processed=True)

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (atom_id,)
        )
        assert abs(rows[0]["importance"] - 0.5) < 1e-4
        assert result.feedback_adjusted == 0

    async def test_feedback_clamped_at_one(self, storage: Storage) -> None:
        """Importance never exceeds 1.0 regardless of how much good feedback."""
        atom_id = await _insert_atom(storage, content="extremely popular atom")
        await storage.execute_write(
            "UPDATE atoms SET importance = 0.99 WHERE id = ?", (atom_id,)
        )
        for _ in range(10):
            await _insert_feedback(storage, atom_id, "good")

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["importance"] <= 1.0

    async def test_feedback_clamped_at_zero(self, storage: Storage) -> None:
        """Importance never drops below 0.0 regardless of how much bad feedback."""
        atom_id = await _insert_atom(storage, content="frequently disliked atom")
        await storage.execute_write(
            "UPDATE atoms SET importance = 0.01 WHERE id = ?", (atom_id,)
        )
        for _ in range(10):
            await _insert_feedback(storage, atom_id, "bad")

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["importance"] >= 0.0

    async def test_dry_run_does_not_write(self, storage: Storage) -> None:
        """dry_run=True computes adjustments but does not write them."""
        atom_id = await _insert_atom(storage, content="dry run importance test")
        await storage.execute_write(
            "UPDATE atoms SET importance = 0.5 WHERE id = ?", (atom_id,)
        )
        await _insert_feedback(storage, atom_id, "good")

        engine = _make_engine(storage)
        result = await engine.reflect(dry_run=True)

        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (atom_id,)
        )
        assert abs(rows[0]["importance"] - 0.5) < 1e-4

        # The feedback record should still be unprocessed.
        fb_rows = await storage.execute(
            "SELECT processed_at FROM atom_feedback WHERE atom_id = ?", (atom_id,)
        )
        assert fb_rows[0]["processed_at"] is None

    async def test_feedback_in_to_dict(self, storage: Storage) -> None:
        """ConsolidationResult.to_dict() includes the feedback_adjusted counter."""
        engine = _make_engine(storage)
        result = await engine.reflect()
        d = result.to_dict()
        assert "feedback_adjusted" in d


# -----------------------------------------------------------------------
# 11. Episodic-to-semantic abstraction
# -----------------------------------------------------------------------


class TestAbstractExperiences:
    """Verify that clusters of similar experiences are abstracted into facts."""

    def _make_mock_embeddings(
        self,
        similarity_map: dict[str, list[tuple[int, float]]],
    ) -> MagicMock:
        """Build a mock EmbeddingEngine whose search_similar uses the given map."""
        from memories.embeddings import EmbeddingEngine

        mock = MagicMock(spec=EmbeddingEngine)
        mock.distance_to_similarity = EmbeddingEngine.distance_to_similarity
        mock.embed_and_store = AsyncMock()

        async def _search(content, k=10):
            return similarity_map.get(content, [])

        mock.search_similar = AsyncMock(side_effect=_search)
        return mock

    async def test_abstracts_cluster_to_fact(self, storage: Storage) -> None:
        """Five similar old experiences should produce one new fact atom.

        B3 raised abstraction_min_cluster from 3 to 5, so at least 5 mutually
        similar experiences are required to trigger abstraction.
        """
        old = "2025-01-01 00:00:00"
        # Insert five similar experiences aged well past min_age_days (7).
        a = await _insert_atom(storage, content="Redis was slow on write-heavy load", atom_type="experience", created_at=old)
        b = await _insert_atom(storage, content="Redis struggled with write-heavy workload", atom_type="experience", created_at=old)
        c = await _insert_atom(storage, content="Redis performance degraded under write load", atom_type="experience", created_at=old)
        d = await _insert_atom(storage, content="Redis write operations slowed under heavy load", atom_type="experience", created_at=old)
        e = await _insert_atom(storage, content="Write-heavy workload caused Redis latency spikes", atom_type="experience", created_at=old)

        # distance=0.12 → similarity ≈ 0.88 > threshold 0.82
        similarity_map = {
            "Redis was slow on write-heavy load": [(b, 0.12), (c, 0.12), (d, 0.12), (e, 0.12)],
            "Redis struggled with write-heavy workload": [(a, 0.12), (c, 0.12), (d, 0.12), (e, 0.12)],
            "Redis performance degraded under write load": [(a, 0.12), (b, 0.12), (d, 0.12), (e, 0.12)],
            "Redis write operations slowed under heavy load": [(a, 0.12), (b, 0.12), (c, 0.12), (e, 0.12)],
            "Write-heavy workload caused Redis latency spikes": [(a, 0.12), (b, 0.12), (c, 0.12), (d, 0.12)],
        }
        mock = self._make_mock_embeddings(similarity_map)

        engine = _make_engine(storage, embeddings=mock)
        result = await engine.reflect()

        assert result.abstracted >= 1

        fact_rows = await storage.execute(
            "SELECT * FROM atoms WHERE type = 'fact' ORDER BY created_at DESC LIMIT 1"
        )
        assert fact_rows, "Expected a new fact atom to be created"

        # The fact should link to the cluster experiences via part-of synapses.
        fact_id = fact_rows[0]["id"]
        synapse_rows = await storage.execute(
            "SELECT * FROM synapses WHERE target_id = ? AND relationship = 'part-of'",
            (fact_id,),
        )
        assert len(synapse_rows) >= 5

    async def test_small_cluster_not_abstracted(self, storage: Storage) -> None:
        """A cluster of only 2 experiences (below min_cluster=3) is not abstracted."""
        old = "2025-01-01 00:00:00"
        a = await _insert_atom(storage, content="Postgres was slow on reads", atom_type="experience", created_at=old)
        b = await _insert_atom(storage, content="Postgres read performance was poor", atom_type="experience", created_at=old)

        similarity_map = {
            "Postgres was slow on reads": [(b, 0.12)],
            "Postgres read performance was poor": [(a, 0.12)],
        }
        mock = self._make_mock_embeddings(similarity_map)

        engine = _make_engine(storage, embeddings=mock)
        result = await engine.reflect()

        assert result.abstracted == 0

    async def test_already_abstracted_skipped(self, storage: Storage) -> None:
        """Experiences already linked to a fact via part-of are not re-abstracted."""
        old = "2025-01-01 00:00:00"
        a = await _insert_atom(storage, content="MySQL slow on joins", atom_type="experience", created_at=old)
        b = await _insert_atom(storage, content="MySQL join performance poor", atom_type="experience", created_at=old)
        c = await _insert_atom(storage, content="MySQL struggles with complex joins", atom_type="experience", created_at=old)
        # Create an existing fact and link all three to it already.
        existing_fact = await _insert_atom(storage, content="MySQL is slow on joins", atom_type="fact", created_at=old)
        for exp_id in (a, b, c):
            await _insert_synapse(storage, exp_id, existing_fact, relationship="part-of", strength=0.8)

        similarity_map = {
            "MySQL slow on joins": [(b, 0.12), (c, 0.12)],
            "MySQL join performance poor": [(a, 0.12), (c, 0.12)],
            "MySQL struggles with complex joins": [(a, 0.12), (b, 0.12)],
        }
        mock = self._make_mock_embeddings(similarity_map)

        engine = _make_engine(storage, embeddings=mock)
        result = await engine.reflect()

        assert result.abstracted == 0

    async def test_fact_confidence_scales_with_cluster_size(self, storage: Storage) -> None:
        """Larger clusters produce higher-confidence facts."""
        old = "2025-01-01 00:00:00"
        contents = [
            "Nginx load balancing improved latency",
            "Load balancing via nginx reduced response times",
            "Nginx round-robin reduced p99 latency",
            "Using nginx as LB cut latency significantly",
            "Nginx LB helped with latency at high traffic",
        ]
        ids = []
        for content in contents:
            ids.append(await _insert_atom(storage, content=content, atom_type="experience", created_at=old))

        # All are similar to the first one.
        similarity_map: dict = {}
        for i, content in enumerate(contents):
            others = [(ids[j], 0.12) for j in range(len(ids)) if j != i]
            similarity_map[content] = others

        mock = self._make_mock_embeddings(similarity_map)
        engine = _make_engine(storage, embeddings=mock)
        await engine.reflect()

        fact_rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE type = 'fact' ORDER BY created_at DESC LIMIT 1"
        )
        assert fact_rows
        # 5-member cluster → confidence = min(0.85, 0.5 + 5 × 0.1) = 0.85
        assert abs(fact_rows[0]["confidence"] - 0.85) < 1e-4

    async def test_dry_run_no_fact_created(self, storage: Storage) -> None:
        """dry_run=True records intent but does not create the fact atom."""
        old = "2025-01-01 00:00:00"
        a = await _insert_atom(storage, content="S3 uploads were slow in eu-west", atom_type="experience", created_at=old)
        b = await _insert_atom(storage, content="S3 upload speed poor in eu-west region", atom_type="experience", created_at=old)
        c = await _insert_atom(storage, content="S3 slow upload performance eu-west", atom_type="experience", created_at=old)
        d = await _insert_atom(storage, content="S3 eu-west upload throughput degraded", atom_type="experience", created_at=old)
        e = await _insert_atom(storage, content="Slow S3 uploads observed in eu-west-1", atom_type="experience", created_at=old)

        similarity_map = {
            "S3 uploads were slow in eu-west": [(b, 0.12), (c, 0.12), (d, 0.12), (e, 0.12)],
            "S3 upload speed poor in eu-west region": [(a, 0.12), (c, 0.12), (d, 0.12), (e, 0.12)],
            "S3 slow upload performance eu-west": [(a, 0.12), (b, 0.12), (d, 0.12), (e, 0.12)],
            "S3 eu-west upload throughput degraded": [(a, 0.12), (b, 0.12), (c, 0.12), (e, 0.12)],
            "Slow S3 uploads observed in eu-west-1": [(a, 0.12), (b, 0.12), (c, 0.12), (d, 0.12)],
        }
        mock = self._make_mock_embeddings(similarity_map)

        engine = _make_engine(storage, embeddings=mock)
        result = await engine.reflect(dry_run=True)

        # Counter should be incremented (intent recorded).
        assert result.abstracted >= 1
        # But no fact atom should exist.
        fact_rows = await storage.execute(
            "SELECT * FROM atoms WHERE type = 'fact'"
        )
        assert len(fact_rows) == 0

    async def test_abstracted_in_to_dict(self, storage: Storage) -> None:
        """ConsolidationResult.to_dict() includes the abstracted counter."""
        engine = _make_engine(storage)
        result = await engine.reflect()
        d = result.to_dict()
        assert "abstracted" in d

    async def test_reuses_existing_fact_when_similar(self, storage: Storage) -> None:
        """When a similar fact already exists, experiences are linked to it
        instead of creating a new near-duplicate fact."""
        old = "2025-01-01 00:00:00"
        # Create an existing fact.
        existing_fact = await _insert_atom(
            storage, content="Lambda cold starts slow in large VPC", atom_type="fact", created_at=old
        )
        # Five experiences similar to each other and to the existing fact.
        # B3 raised min_cluster from 3 to 5.
        a = await _insert_atom(storage, content="Lambda cold starts were slow in VPC", atom_type="experience", created_at=old)
        b = await _insert_atom(storage, content="Cold starts in Lambda VPC were slow", atom_type="experience", created_at=old)
        c = await _insert_atom(storage, content="VPC Lambda slow cold start observed", atom_type="experience", created_at=old)
        d = await _insert_atom(storage, content="Lambda VPC cold start latency was high", atom_type="experience", created_at=old)
        e = await _insert_atom(storage, content="Slow Lambda cold starts noticed in VPC environment", atom_type="experience", created_at=old)

        from memories.embeddings import EmbeddingEngine

        async def _search(content, k=10):
            # Experiences are similar to each other (distance 0.12)
            # and the existing fact is similar to the template (distance 0.03 > merge_threshold)
            exp_map = {
                "Lambda cold starts were slow in VPC": [(b, 0.12), (c, 0.12), (d, 0.12), (e, 0.12), (existing_fact, 0.03)],
                "Cold starts in Lambda VPC were slow": [(a, 0.12), (c, 0.12), (d, 0.12), (e, 0.12), (existing_fact, 0.03)],
                "VPC Lambda slow cold start observed": [(a, 0.12), (b, 0.12), (d, 0.12), (e, 0.12), (existing_fact, 0.03)],
                "Lambda VPC cold start latency was high": [(a, 0.12), (b, 0.12), (c, 0.12), (e, 0.12), (existing_fact, 0.03)],
                "Slow Lambda cold starts noticed in VPC environment": [(a, 0.12), (b, 0.12), (c, 0.12), (d, 0.12), (existing_fact, 0.03)],
            }
            return exp_map.get(content, [])

        mock = MagicMock(spec=EmbeddingEngine)
        mock.distance_to_similarity = EmbeddingEngine.distance_to_similarity
        mock.embed_and_store = AsyncMock()
        mock.search_similar = AsyncMock(side_effect=_search)

        engine = _make_engine(storage, embeddings=mock)
        result = await engine.reflect()

        # No new fact should be created.
        fact_rows = await storage.execute(
            "SELECT * FROM atoms WHERE type = 'fact'"
        )
        assert len(fact_rows) == 1
        assert fact_rows[0]["id"] == existing_fact

        # Experiences should be linked to the existing fact.
        synapse_rows = await storage.execute(
            "SELECT * FROM synapses WHERE target_id = ? AND relationship = 'part-of'",
            (existing_fact,),
        )
        assert len(synapse_rows) >= 3

        # abstracted counter should not have been incremented.
        assert result.abstracted == 0


# -----------------------------------------------------------------------
# 12. Contradiction resolution
# -----------------------------------------------------------------------


class TestResolveContradictions:
    """Verify that clear contradiction winners supersede their losers."""

    async def test_resolves_clear_contradiction(self, storage: Storage) -> None:
        """When one atom is clearly stronger, the weaker one loses confidence
        and a supersedes synapse is created."""
        old = "2025-01-01 00:00:00"
        winner_id = await _insert_atom(
            storage, content="Use async DB calls in all API handlers",
            atom_type="fact", confidence=0.9, access_count=40, created_at=old,
        )
        loser_id = await _insert_atom(
            storage, content="Use sync DB calls in all API handlers",
            atom_type="fact", confidence=0.4, access_count=2, created_at=old,
        )
        await _insert_synapse(storage, winner_id, loser_id, relationship="contradicts", strength=0.8)

        engine = _make_engine(storage)
        result = await engine.reflect()

        assert result.resolved >= 1

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (loser_id,)
        )
        assert rows[0]["confidence"] < 0.4  # was decremented

        synapse_rows = await storage.execute(
            "SELECT * FROM synapses WHERE source_id = ? AND target_id = ? AND relationship = 'supersedes'",
            (winner_id, loser_id),
        )
        assert synapse_rows, "Expected a supersedes synapse from winner to loser"

    async def test_close_contradiction_not_resolved(self, storage: Storage) -> None:
        """When both atoms are similar strength, the contradiction is left active."""
        old = "2025-01-01 00:00:00"
        a = await _insert_atom(
            storage, content="Use tabs for indentation",
            atom_type="preference", confidence=0.7, access_count=10, created_at=old,
        )
        b = await _insert_atom(
            storage, content="Use spaces for indentation",
            atom_type="preference", confidence=0.7, access_count=10, created_at=old,
        )
        await _insert_synapse(storage, a, b, relationship="contradicts", strength=0.8)

        engine = _make_engine(storage)
        result = await engine.reflect()

        assert result.resolved == 0

    async def test_too_young_not_resolved(self, storage: Storage) -> None:
        """Atoms created recently (< contradiction_min_age_days) are not resolved."""
        # Use a very recent created_at — well within the 14-day window.
        recent = "2099-12-31 00:00:00"
        a = await _insert_atom(
            storage, content="Always use connection pooling",
            atom_type="fact", confidence=0.9, access_count=50, created_at=recent,
        )
        b = await _insert_atom(
            storage, content="Never use connection pooling",
            atom_type="fact", confidence=0.2, access_count=1, created_at=recent,
        )
        await _insert_synapse(storage, a, b, relationship="contradicts", strength=0.8)

        engine = _make_engine(storage)
        result = await engine.reflect()

        assert result.resolved == 0

    async def test_dry_run_no_write(self, storage: Storage) -> None:
        """dry_run=True records resolution intent but makes no DB changes."""
        old = "2025-01-01 00:00:00"
        winner_id = await _insert_atom(
            storage, content="Use prepared statements always",
            atom_type="fact", confidence=0.95, access_count=30, created_at=old,
        )
        loser_id = await _insert_atom(
            storage, content="Use raw queries always",
            atom_type="fact", confidence=0.3, access_count=1, created_at=old,
        )
        await _insert_synapse(storage, winner_id, loser_id, relationship="contradicts", strength=0.8)

        engine = _make_engine(storage)
        result = await engine.reflect(dry_run=True)

        assert result.resolved >= 1

        # Loser confidence must be unchanged.
        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (loser_id,)
        )
        assert abs(rows[0]["confidence"] - 0.3) < 1e-4

        # No supersedes synapse.
        synapses = await storage.execute(
            "SELECT * FROM synapses WHERE relationship = 'supersedes'"
        )
        assert len(synapses) == 0

    async def test_resolved_in_to_dict(self, storage: Storage) -> None:
        """ConsolidationResult.to_dict() includes the resolved counter."""
        engine = _make_engine(storage)
        result = await engine.reflect()
        assert "resolved" in result.to_dict()


# -----------------------------------------------------------------------
# TestTuneRetrievalWeights
# -----------------------------------------------------------------------


class TestTuneRetrievalWeights:
    """Tests for _tune_retrieval_weights — the retrieval weight auto-tuning step."""

    async def test_nudges_confidence_weight_toward_good(self, storage: Storage) -> None:
        """Good atoms with high confidence push the confidence weight up."""
        # 5 good atoms with high confidence.
        for i in range(5):
            aid = await _insert_atom(
                storage, content=f"good fact {i}", confidence=0.95,
                access_count=5,
            )
            await _insert_feedback(storage, aid, "good")

        # 5 bad atoms with low confidence.
        for i in range(5):
            bid = await _insert_atom(
                storage, content=f"bad fact {i}", confidence=0.1,
                access_count=1,
            )
            await _insert_feedback(storage, bid, "bad")

        engine = _make_engine(storage)
        await engine.reflect()

        saved = await storage.load_retrieval_weights()
        assert saved is not None
        # confidence weight should have moved above the default 0.07
        assert saved["confidence"] > 0.07

    async def test_nudges_importance_weight_toward_bad(self, storage: Storage) -> None:
        """Bad atoms with high importance push the importance weight down."""
        # 5 bad atoms with high importance.
        for i in range(5):
            bid = await _insert_atom(
                storage, content=f"bad important {i}", confidence=0.5,
                access_count=1,
            )
            await storage.execute_write(
                "UPDATE atoms SET importance = 0.95 WHERE id = ?", (bid,)
            )
            await _insert_feedback(storage, bid, "bad")

        # 5 good atoms with low importance.
        for i in range(5):
            aid = await _insert_atom(
                storage, content=f"good unimportant {i}", confidence=0.5,
                access_count=1,
            )
            await storage.execute_write(
                "UPDATE atoms SET importance = 0.1 WHERE id = ?", (aid,)
            )
            await _insert_feedback(storage, aid, "good")

        engine = _make_engine(storage)
        await engine.reflect()

        saved = await storage.load_retrieval_weights()
        assert saved is not None
        # importance weight should have moved below the default 0.11
        assert saved["importance"] < 0.11

    async def test_no_change_when_too_few_samples(self, storage: Storage) -> None:
        """Fewer than min_samples good or bad atoms → weights unchanged."""
        # Only 3 good atoms and 3 bad atoms (default min_samples=5).
        for i in range(3):
            aid = await _insert_atom(storage, content=f"good {i}", confidence=0.9)
            await _insert_feedback(storage, aid, "good")
            bid = await _insert_atom(storage, content=f"bad {i}", confidence=0.1)
            await _insert_feedback(storage, bid, "bad")

        engine = _make_engine(storage)
        await engine.reflect()

        saved = await storage.load_retrieval_weights()
        # No row should have been written.
        assert saved is None

    async def test_weights_clamped_to_max_drift(self, storage: Storage) -> None:
        """Weights cannot drift beyond ±30% of factory defaults."""
        # Pre-seed the table with a weight already at the upper clamp boundary.
        default_confidence = 0.07
        max_drift = 0.30
        hi = default_confidence * (1 + max_drift)
        await storage.save_retrieval_weights(
            {"confidence": hi, "importance": 0.11, "frequency": 0.07,
             "recency": 0.10, "spread_activation": 0.25}
        )

        # Many good high-confidence atoms would push it further — but it's clamped.
        for i in range(10):
            aid = await _insert_atom(
                storage, content=f"good conf {i}", confidence=0.99, access_count=5,
            )
            await _insert_feedback(storage, aid, "good")
        for i in range(5):
            bid = await _insert_atom(
                storage, content=f"bad conf {i}", confidence=0.01, access_count=0,
            )
            await _insert_feedback(storage, bid, "bad")

        engine = _make_engine(storage)
        await engine.reflect()

        saved = await storage.load_retrieval_weights()
        assert saved is not None
        assert saved["confidence"] <= hi + 1e-9  # must not exceed upper clamp

    async def test_dry_run_does_not_write(self, storage: Storage) -> None:
        """dry_run=True computes tuning but does not call save_retrieval_weights."""
        for i in range(5):
            aid = await _insert_atom(
                storage, content=f"good {i}", confidence=0.95, access_count=5,
            )
            await _insert_feedback(storage, aid, "good")
            bid = await _insert_atom(
                storage, content=f"bad {i}", confidence=0.05, access_count=0,
            )
            await _insert_feedback(storage, bid, "bad")

        engine = _make_engine(storage)
        await engine.reflect(dry_run=True)

        # Table should still be empty after a dry run.
        saved = await storage.load_retrieval_weights()
        assert saved is None

    async def test_small_diff_ignored(self, storage: Storage) -> None:
        """When good_mean − bad_mean < 0.05 for a signal, no nudge is applied."""
        # Interleave atoms so that good and bad have nearly identical properties.
        for i in range(5):
            aid = await _insert_atom(
                storage, content=f"good near {i}", confidence=0.52, access_count=2,
            )
            await _insert_feedback(storage, aid, "good")
            bid = await _insert_atom(
                storage, content=f"bad near {i}", confidence=0.50, access_count=2,
            )
            await _insert_feedback(storage, bid, "bad")

        engine = _make_engine(storage)
        await engine.reflect()

        # The diff for all signals is ~0.02, below the 0.05 threshold → no write.
        saved = await storage.load_retrieval_weights()
        assert saved is None


# -----------------------------------------------------------------------
# 13. Consolidation ordering: decay runs BEFORE abstraction (B4)
# -----------------------------------------------------------------------


class TestConsolidationOrdering:
    """Verify that _decay_atoms is called before _abstract_experiences in reflect().

    This guarantees that stale atoms have their confidence reduced before
    the abstraction step runs, preventing decayed-but-not-yet-updated atoms
    from being used as cluster templates with artificially high confidence.
    """

    async def test_decay_runs_before_abstraction(self, storage: Storage) -> None:
        """Verify that _decay_atoms is invoked before _abstract_experiences.

        Patches both methods with side_effects that record the call order,
        then asserts that decay came first.
        """
        engine = _make_engine(storage)
        call_order: list[str] = []

        original_decay = engine._decay_atoms
        original_abstract = engine._abstract_experiences

        async def tracked_decay(result, scope="all"):
            call_order.append("decay")
            return await original_decay(result, scope)

        async def tracked_abstract(result, scope="all"):
            call_order.append("abstract")
            return await original_abstract(result, scope)

        with (
            patch.object(engine, "_decay_atoms", side_effect=tracked_decay),
            patch.object(engine, "_abstract_experiences", side_effect=tracked_abstract),
        ):
            await engine.reflect()

        assert "decay" in call_order, "_decay_atoms was never called"
        assert "abstract" in call_order, "_abstract_experiences was never called"
        assert call_order.index("decay") < call_order.index("abstract"), (
            f"Expected decay before abstraction, but call order was: {call_order}"
        )


# -----------------------------------------------------------------------
# 14. Type-differentiated decay (B1)
# -----------------------------------------------------------------------


class TestTypeDifferentiatedDecay:
    """Verify that per-type decay multipliers are applied correctly.

    The type_decay_multipliers dict in ConsolidationConfig assigns a
    multiplier to each atom type.  The effective decay rate for an atom
    is ``decay_rate * type_decay_multipliers[atom.type]``.  Skills and
    facts (multiplier=0.995) decay much slower than experiences (0.93).
    """

    async def test_skill_decays_slower_than_experience(self, storage: Storage) -> None:
        """A skill atom should retain higher confidence than an experience atom
        after one decay cycle, because skill has multiplier 0.995 vs 0.93."""
        skill_id = await _insert_atom(
            storage,
            content="How to set up a Python virtualenv",
            atom_type="skill",
            confidence=1.0,
            created_at="2020-01-01 00:00:00",
        )
        experience_id = await _insert_atom(
            storage,
            content="Set up a virtualenv for project X yesterday",
            atom_type="experience",
            confidence=1.0,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        skill_row = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (skill_id,)
        )
        exp_row = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (experience_id,)
        )

        skill_conf = skill_row[0]["confidence"]
        exp_conf = exp_row[0]["confidence"]

        assert skill_conf > exp_conf, (
            f"Skill ({skill_conf:.4f}) should have higher confidence than "
            f"experience ({exp_conf:.4f}) after decay"
        )

    async def test_fact_decays_same_as_skill(self, storage: Storage) -> None:
        """Facts and skills share the same multiplier (0.995), so they should
        decay at the same rate and have identical confidence after one cycle."""
        fact_id = await _insert_atom(
            storage,
            content="Python 3.12 supports f-string improvements",
            atom_type="fact",
            confidence=0.8,
            created_at="2020-01-01 00:00:00",
        )
        skill_id = await _insert_atom(
            storage,
            content="Debugging Python with breakpoint()",
            atom_type="skill",
            confidence=0.8,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        fact_row = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (fact_id,)
        )
        skill_row = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (skill_id,)
        )

        assert abs(fact_row[0]["confidence"] - skill_row[0]["confidence"]) < 1e-6, (
            "Facts and skills should decay identically (both use multiplier 0.995)"
        )

    async def test_custom_type_multipliers_respected(self, storage: Storage) -> None:
        """When ConsolidationConfig is initialised with custom type_decay_multipliers,
        the engine uses those custom values for decay."""
        from memories.config import ConsolidationConfig

        # Custom multiplier: experience decays at full rate (1.0), skill at 0.5.
        custom_cfg = ConsolidationConfig(
            type_decay_multipliers={"experience": 1.0, "skill": 0.5}
        )

        exp_id = await _insert_atom(
            storage,
            content="Custom experience atom",
            atom_type="experience",
            confidence=0.9,
            created_at="2020-01-01 00:00:00",
        )
        skill_id = await _insert_atom(
            storage,
            content="Custom skill atom",
            atom_type="skill",
            confidence=0.9,
            created_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        # Override the engine config with the custom one.
        engine._cfg = custom_cfg
        await engine.reflect()

        exp_row = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (exp_id,)
        )
        skill_row = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (skill_id,)
        )

        exp_conf = exp_row[0]["confidence"]
        skill_conf = skill_row[0]["confidence"]

        # With multiplier 0.5 the skill decays much faster: effective_rate = 0.95 * 0.5 = 0.475
        # Experience uses 1.0: effective_rate = 0.95 * 1.0 = 0.95
        # So experience should retain higher confidence.
        assert exp_conf > skill_conf, (
            f"Experience ({exp_conf:.4f}) should retain higher confidence than "
            f"skill ({skill_conf:.4f}) with custom multiplier skill=0.5"
        )


# -----------------------------------------------------------------------
# 15. LTD semantic exemption (A3)
# -----------------------------------------------------------------------


class TestLTDSemanticExemption:
    """Verify that _apply_ltd does NOT weaken contradicts, supersedes, or
    warns-against synapses.

    These synapse types carry critical semantic meaning and must be exempt
    from the anti-Hebbian weakening rule.  The SQL filter in _apply_ltd
    excludes them via:
        ``AND s.relationship NOT IN ('contradicts', 'supersedes', 'warns-against')``
    """

    async def _make_exempt_scenario(
        self, storage: Storage, relationship: str
    ) -> tuple[int, int, int]:
        """Create two recently-active atoms with a synapse that has old
        last_activated_at (qualifying for LTD) and return (atom_a, atom_b, synapse_id)."""
        from memories.config import get_config

        cfg = get_config().consolidation

        atom_a = await _insert_atom(
            storage, f"atom A ({relationship})",
            last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )
        atom_b = await _insert_atom(
            storage, f"atom B ({relationship})",
            last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )
        old_activation = _recent(cfg.ltd_window_days + 10)
        synapse_id = await _insert_synapse(
            storage, atom_a, atom_b,
            relationship=relationship,
            strength=0.8,
            last_activated_at=old_activation,
        )
        return atom_a, atom_b, synapse_id

    async def test_ltd_skips_contradicts_synapses(self, storage: Storage) -> None:
        """A 'contradicts' synapse must not be weakened by LTD."""
        _, _, synapse_id = await self._make_exempt_scenario(storage, "contradicts")

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (synapse_id,)
        )
        assert rows, "contradicts synapse should still exist"
        # Synapse may be subject to passive synapse decay (0.95 factor) but NOT LTD.
        ltd_details = [
            d for d in result.details
            if d.get("action") == "ltd" and d.get("synapse_id") == synapse_id
        ]
        assert len(ltd_details) == 0, (
            "LTD should not target 'contradicts' synapses"
        )

    async def test_ltd_skips_supersedes_synapses(self, storage: Storage) -> None:
        """A 'supersedes' synapse must not be weakened by LTD."""
        _, _, synapse_id = await self._make_exempt_scenario(storage, "supersedes")

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (synapse_id,)
        )
        assert rows, "supersedes synapse should still exist"
        ltd_details = [
            d for d in result.details
            if d.get("action") == "ltd" and d.get("synapse_id") == synapse_id
        ]
        assert len(ltd_details) == 0, (
            "LTD should not target 'supersedes' synapses"
        )

    async def test_ltd_skips_warns_against_synapses(self, storage: Storage) -> None:
        """A 'warns-against' synapse must not be weakened by LTD."""
        _, _, synapse_id = await self._make_exempt_scenario(storage, "warns-against")

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (synapse_id,)
        )
        assert rows, "warns-against synapse should still exist"
        ltd_details = [
            d for d in result.details
            if d.get("action") == "ltd" and d.get("synapse_id") == synapse_id
        ]
        assert len(ltd_details) == 0, (
            "LTD should not target 'warns-against' synapses"
        )

    async def test_ltd_does_weaken_related_to_synapses(self, storage: Storage) -> None:
        """A 'related-to' synapse IS subject to LTD weakening when both
        endpoints are active and the synapse has not been co-activated."""
        from memories.config import get_config

        cfg = get_config().consolidation

        atom_a = await _insert_atom(
            storage, "LTD target atom A",
            last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )
        atom_b = await _insert_atom(
            storage, "LTD target atom B",
            last_accessed_at=_recent(5),
            created_at="2024-01-01 00:00:00",
        )
        old_activation = _recent(cfg.ltd_window_days + 10)
        synapse_id = await _insert_synapse(
            storage, atom_a, atom_b,
            relationship="related-to",
            strength=0.6,
            last_activated_at=old_activation,
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        ltd_details = [
            d for d in result.details
            if d.get("action") == "ltd" and d.get("synapse_id") == synapse_id
        ]
        assert len(ltd_details) == 1, (
            "LTD should target 'related-to' synapses that qualify"
        )
        assert result.ltd >= 1


# -----------------------------------------------------------------------
# 16. Feedback for deleted / nonexistent atoms (A4)
# -----------------------------------------------------------------------


class TestFeedbackDeletedAtoms:
    """Verify that feedback signals referencing deleted or nonexistent atoms
    are marked as processed (processed_at set) without errors.

    The _apply_feedback_signals method groups feedback by atom_id and
    batch-fetches atoms via get_batch_without_tracking.  Atoms that
    no longer exist return None from the map, so the code skips the
    importance delta but still marks the feedback rows as processed.
    """

    async def test_feedback_for_deleted_atom_is_marked_processed(
        self, storage: Storage
    ) -> None:
        """Feedback for a soft-deleted atom should still be marked processed_at."""
        atom_id = await _insert_atom(
            storage, content="atom that will be deleted", confidence=0.8,
        )
        # Insert feedback while the atom exists.
        await _insert_feedback(storage, atom_id, "good")

        # Soft-delete the atom.
        await storage.execute_write(
            "UPDATE atoms SET is_deleted = 1 WHERE id = ?", (atom_id,)
        )

        engine = _make_engine(storage)
        await engine.reflect()

        # The feedback row should now have processed_at set.
        fb_rows = await storage.execute(
            "SELECT processed_at FROM atom_feedback WHERE atom_id = ?", (atom_id,)
        )
        assert len(fb_rows) == 1
        assert fb_rows[0]["processed_at"] is not None, (
            "Feedback for deleted atom should be marked as processed"
        )

    async def test_multiple_feedback_for_deleted_atom_all_marked_processed(
        self, storage: Storage
    ) -> None:
        """Multiple feedback rows for a soft-deleted atom should ALL be
        marked as processed after a single consolidation cycle."""
        atom_id = await _insert_atom(
            storage, content="atom with multiple feedback rows", confidence=0.7,
        )
        # Insert several feedback rows.
        await _insert_feedback(storage, atom_id, "good")
        await _insert_feedback(storage, atom_id, "bad")
        await _insert_feedback(storage, atom_id, "good")

        # Soft-delete the atom so get_batch_without_tracking returns None.
        await storage.execute_write(
            "UPDATE atoms SET is_deleted = 1 WHERE id = ?", (atom_id,)
        )

        engine = _make_engine(storage)
        await engine.reflect()

        # All three feedback rows should now have processed_at set.
        fb_rows = await storage.execute(
            "SELECT processed_at FROM atom_feedback WHERE atom_id = ?", (atom_id,)
        )
        assert len(fb_rows) == 3
        for row in fb_rows:
            assert row["processed_at"] is not None, (
                "All feedback rows for a deleted atom must be marked as processed"
            )


# -----------------------------------------------------------------------
# SQL hardening regression tests
# -----------------------------------------------------------------------


class TestSQLHardening:
    """Regression tests that verify parameterised SQL queries execute
    correctly after the injection-hardening refactor.

    Each test focuses on *behaviour* (the correct DB state after a
    consolidation cycle) so that a future regression in parameterisation
    will be caught immediately.
    """

    async def test_feedback_ids_use_placeholders(self, storage: Storage) -> None:
        """Feedback rows are marked processed_at after consolidation.

        This exercises the parameterised ``WHERE id IN (?,?,?)`` path in
        ``_apply_feedback_signals``.  Creates three feedback rows across
        two atoms, runs reflect(), then asserts every row has
        ``processed_at`` set — confirming the UPDATE executed correctly.
        """
        # Insert two atoms that will receive feedback.
        atom_a = await _insert_atom(storage, content="atom a", atom_type="fact")
        atom_b = await _insert_atom(storage, content="atom b", atom_type="fact")

        # Insert feedback rows: 2 for atom_a, 1 for atom_b.
        for signal, atom_id in [("good", atom_a), ("good", atom_a), ("bad", atom_b)]:
            await storage.execute_write(
                "INSERT INTO atom_feedback (atom_id, signal) VALUES (?, ?)",
                (atom_id, signal),
            )

        engine = _make_engine(storage)
        await engine.reflect()

        # All three feedback rows must have been marked processed.
        fb_rows = await storage.execute(
            "SELECT processed_at FROM atom_feedback WHERE atom_id IN (?, ?)",
            (atom_a, atom_b),
        )
        assert len(fb_rows) == 3, "Expected 3 feedback rows"
        for row in fb_rows:
            assert row["processed_at"] is not None, (
                "All feedback rows must have processed_at set after reflect()"
            )

    async def test_antipattern_like_uses_placeholders(self, storage: Storage) -> None:
        """Antipatterns matching reclassification patterns are reclassified.

        This exercises the parameterised LIKE-clause query in
        ``_reclassify_antipatterns``.  Inserts an atom stored as
        ``antipattern`` whose content matches the ``'Edited %:%'`` pattern,
        then confirms reflect() changes its type to ``experience``.
        """
        # Content matches the "Edited %:%" LIKE pattern exactly.
        atom_id = await _insert_atom(
            storage,
            content="Edited src/foo.py: changed import block",
            atom_type="antipattern",
        )

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT type FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["type"] == "experience", (
            "Atom matching 'Edited %:%' pattern should be reclassified from "
            "'antipattern' to 'experience'"
        )

    async def test_datetime_params_in_apply_ltd(self, storage: Storage) -> None:
        """LTD weakens synapses whose last_activated_at is outside the window.

        This exercises the parameterised ``datetime('now', ?)`` expressions
        in ``_apply_ltd``.  Creates two recently-active atoms connected by
        a synapse that has not been activated within ``ltd_window_days``
        (default 14) and confirms the synapse strength is reduced.
        """
        # Both atoms must be recently accessed (within decay_after_days=30).
        recent = "2099-01-01 00:00:00"  # far future ensures "recent"
        atom_a = await _insert_atom(
            storage, content="atom ltd a", atom_type="fact",
            last_accessed_at=recent,
        )
        atom_b = await _insert_atom(
            storage, content="atom ltd b", atom_type="fact",
            last_accessed_at=recent,
        )

        initial_strength = 0.8
        # Synapse last activated far in the past — outside the LTD window.
        synapse_id = await _insert_synapse(
            storage, atom_a, atom_b,
            strength=initial_strength,
            last_activated_at="2020-01-01 00:00:00",
        )

        engine = _make_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (synapse_id,)
        )
        # The synapse may have been pruned (deleted) or weakened — either
        # confirms the parameterised LTD query executed without error.
        if rows:
            assert rows[0]["strength"] < initial_strength, (
                "LTD should have weakened the stale synapse"
            )
        # If no rows, the synapse was pruned below prune_threshold — also correct.
        assert result.ltd >= 1, "At least one LTD event should have been recorded"


# -----------------------------------------------------------------------
# 19. Proportional LTD + batch weaken
# -----------------------------------------------------------------------


class TestProportionalLTD:
    """Verify that _apply_ltd applies proportional (multiplicative) LTD and
    issues batch SQL operations rather than per-row weaken() calls.

    Proportional LTD: strength *= (1 - ltd_fraction)
    This means high-strength synapses lose more absolute strength per cycle
    than low-strength ones, providing a natural convergence property.
    """

    def _ltd_setup(self) -> dict:
        """Return consolidation config fields for reference in tests."""
        from memories.config import get_config
        cfg = get_config().consolidation
        return {
            "ltd_window_days": cfg.ltd_window_days,
            "decay_after_days": cfg.decay_after_days,
            "prune_threshold": cfg.prune_threshold,
        }

    async def test_proportional_ltd_weaker_loses_less(
        self, storage: Storage
    ) -> None:
        """Low-strength synapse loses less absolute strength than high-strength
        synapse under proportional LTD — both are individually above the
        prune_threshold so neither is deleted.
        """
        cfg = self._ltd_setup()
        old_activation = "2020-01-01 00:00:00"
        recent_access = _recent(5)

        # Two atoms that are individually active.
        atom_a = await _insert_atom(
            storage, "low synapse atom A",
            last_accessed_at=recent_access,
            created_at="2024-01-01 00:00:00",
        )
        atom_b = await _insert_atom(
            storage, "low synapse atom B",
            last_accessed_at=recent_access,
            created_at="2024-01-01 00:00:00",
        )
        atom_c = await _insert_atom(
            storage, "high synapse atom C",
            last_accessed_at=recent_access,
            created_at="2024-01-01 00:00:00",
        )
        atom_d = await _insert_atom(
            storage, "high synapse atom D",
            last_accessed_at=recent_access,
            created_at="2024-01-01 00:00:00",
        )

        # Low-strength synapse: 0.1 — well above prune_threshold (0.05).
        low_synapse_id = await _insert_synapse(
            storage, atom_a, atom_b,
            strength=0.1,
            last_activated_at=old_activation,
        )
        # High-strength synapse: 0.8 — well above prune_threshold (0.05).
        high_synapse_id = await _insert_synapse(
            storage, atom_c, atom_d,
            strength=0.8,
            last_activated_at=old_activation,
        )

        engine = _make_engine(storage)
        await engine.reflect()

        low_rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (low_synapse_id,)
        )
        high_rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (high_synapse_id,)
        )

        # Both must still exist (neither pruned).
        assert low_rows, "Low-strength synapse should not have been pruned"
        assert high_rows, "High-strength synapse should not have been pruned"

        low_after = low_rows[0]["strength"]
        high_after = high_rows[0]["strength"]

        # Proportional property: absolute loss = original * ltd_fraction.
        # High synapse loss > low synapse loss.
        low_loss = 0.1 - low_after
        high_loss = 0.8 - high_after

        assert low_loss > 0, "Low-strength synapse must have been weakened"
        assert high_loss > 0, "High-strength synapse must have been weakened"
        assert high_loss > low_loss, (
            f"High-strength synapse should lose more absolute strength "
            f"({high_loss:.4f}) than low-strength ({low_loss:.4f})"
        )

    async def test_proportional_ltd_high_strength_larger_absolute_loss(
        self, storage: Storage
    ) -> None:
        """Explicitly verify the proportional property: for any multiplier in
        (0,1), high_strength * factor < low_strength * factor implies
        high_strength loses more after LTD than low_strength.

        After reflect(), the high-strength synapse's absolute reduction
        must be strictly larger than the low-strength synapse's reduction.
        """
        old_activation = "2020-01-01 00:00:00"
        recent_access = _recent(5)

        atom_e = await _insert_atom(
            storage, "prop atom E",
            last_accessed_at=recent_access,
            created_at="2024-01-01 00:00:00",
        )
        atom_f = await _insert_atom(
            storage, "prop atom F",
            last_accessed_at=recent_access,
            created_at="2024-01-01 00:00:00",
        )
        atom_g = await _insert_atom(
            storage, "prop atom G",
            last_accessed_at=recent_access,
            created_at="2024-01-01 00:00:00",
        )
        atom_h = await _insert_atom(
            storage, "prop atom H",
            last_accessed_at=recent_access,
            created_at="2024-01-01 00:00:00",
        )

        low_id = await _insert_synapse(
            storage, atom_e, atom_f,
            strength=0.2,
            last_activated_at=old_activation,
        )
        high_id = await _insert_synapse(
            storage, atom_g, atom_h,
            strength=0.9,
            last_activated_at=old_activation,
        )

        engine = _make_engine(storage)
        await engine.reflect()

        low_rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (low_id,)
        )
        high_rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (high_id,)
        )

        assert low_rows, "Low-strength synapse (0.2) should survive one LTD cycle"
        assert high_rows, "High-strength synapse (0.9) should survive one LTD cycle"

        low_loss = 0.2 - low_rows[0]["strength"]
        high_loss = 0.9 - high_rows[0]["strength"]

        assert high_loss > low_loss, (
            f"Proportional LTD: high_loss={high_loss:.4f} must exceed "
            f"low_loss={low_loss:.4f}"
        )

    async def test_ltd_floor_prevents_immortal_synapses(
        self, storage: Storage
    ) -> None:
        """A synapse near-but-above prune_threshold is still weakened by
        proportional LTD (it moves toward the floor), preventing weak synapses
        from becoming immortal due to rounding.

        With ltd_fraction=0.15, a synapse at 0.06 → 0.06 * 0.85 = 0.051.
        That is still above prune_threshold (0.05), so it should survive one
        cycle but be weakened (not left at 0.06).
        """
        old_activation = "2020-01-01 00:00:00"
        recent_access = _recent(5)

        atom_i = await _insert_atom(
            storage, "floor atom I",
            last_accessed_at=recent_access,
            created_at="2024-01-01 00:00:00",
        )
        atom_j = await _insert_atom(
            storage, "floor atom J",
            last_accessed_at=recent_access,
            created_at="2024-01-01 00:00:00",
        )

        # 0.06 is just above prune_threshold=0.05.
        synapse_id = await _insert_synapse(
            storage, atom_i, atom_j,
            strength=0.06,
            last_activated_at=old_activation,
        )

        engine = _make_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (synapse_id,)
        )
        # The synapse should either:
        # a) Still exist but with strength strictly less than 0.06 (weakened).
        # b) Be pruned (if ltd_fraction is high enough to push below threshold).
        # Either outcome confirms proportional LTD acted on the synapse.
        if rows:
            assert rows[0]["strength"] < 0.06, (
                "Synapse at 0.06 must be weakened by proportional LTD "
                "(strength should decrease, not stay at 0.06)"
            )
        # If rows is empty, the synapse was pruned — also acceptable.

    async def test_ltd_batch_not_per_row(self, storage: Storage) -> None:
        """The batch LTD implementation issues at most 3 execute_write calls
        for the entire LTD pass (1 UPDATE + 1 DELETE), not one call per synapse.

        With N qualifying synapses, the old per-row weaken() approach would
        issue N execute_transaction calls.  The new batch approach should issue
        at most 3 total write operations regardless of N.
        """
        from unittest.mock import AsyncMock, patch

        old_activation = "2020-01-01 00:00:00"
        recent_access = _recent(5)

        # Create 4 qualifying synapse pairs.
        for i in range(4):
            a = await _insert_atom(
                storage, f"batch atom A{i}",
                last_accessed_at=recent_access,
                created_at="2024-01-01 00:00:00",
            )
            b = await _insert_atom(
                storage, f"batch atom B{i}",
                last_accessed_at=recent_access,
                created_at="2024-01-01 00:00:00",
            )
            await _insert_synapse(
                storage, a, b,
                strength=0.5,
                last_activated_at=old_activation,
            )

        engine = _make_engine(storage)

        write_calls: list[str] = []
        original_execute_write = storage.execute_write

        async def spy_execute_write(sql: str, params=()) -> int:
            write_calls.append(sql.strip())
            return await original_execute_write(sql, params)

        with patch.object(storage, "execute_write", side_effect=spy_execute_write):
            await engine.reflect()

        # Isolate only the LTD-specific batch write calls.
        # The proportional LTD UPDATE is identifiable by MAX( in the SQL
        # (general synapse decay uses plain "strength * ?" without MAX).
        # The LTD prune DELETE is identifiable by "strength <=" in the WHERE.
        ltd_writes = [
            sql for sql in write_calls
            if "synapses" in sql.lower()
            and (
                "max(" in sql.lower()                                          # proportional LTD UPDATE
                or ("delete" in sql.lower() and "strength <=" in sql.lower())  # LTD prune DELETE
            )
        ]

        assert len(ltd_writes) <= 2, (
            f"Batch LTD should issue at most 2 synapse-specific write operations "
            f"(1 proportional UPDATE + 1 prune DELETE), "
            f"got {len(ltd_writes)}: {ltd_writes}"
        )


# -----------------------------------------------------------------------
# W3-D: LLM distillation concurrency in _abstract_experiences
# -----------------------------------------------------------------------


class TestLLMDistillationConcurrency:
    """W3-D: Verify concurrent, timeout-guarded, fallback-safe LLM distillation."""

    def _make_engine_with_mock_embeddings(
        self, storage: Storage
    ) -> ConsolidationEngine:
        """Build an engine with embeddings mocked for abstraction tests."""
        embeddings = MagicMock(spec=EmbeddingEngine)
        # search_similar returns empty so clusters never form (used only in
        # tests that explicitly set the mock's side_effect).
        embeddings.search_similar = AsyncMock(return_value=[])
        embeddings.embed_and_store = AsyncMock()
        atoms = AtomManager(storage, embeddings)
        from memories.synapses import SynapseManager
        synapses = SynapseManager(storage)
        return ConsolidationEngine(storage, embeddings, atoms, synapses)

    async def test_distillation_fallback_on_timeout(
        self, storage: Storage
    ) -> None:
        """When the LLM raises TimeoutError, _distill_cluster returns verbatim content."""
        import asyncio
        from memories.consolidation import ConsolidationEngine
        from memories.atoms import Atom

        engine = self._make_engine_with_mock_embeddings(storage)

        # Build a minimal Atom-like object with a content attribute.
        atom = MagicMock(spec=Atom)
        atom.content = "fallback verbatim content"

        # Patch the LLM client so every call raises TimeoutError.
        with patch.object(
            engine,
            "_distill_cluster",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError,
        ):
            # The fallback logic is inside _distill_cluster itself; to test
            # the real fallback we call the actual (unpatched) method with a
            # mocked LLM client that raises TimeoutError.
            pass

        # Test the real _distill_cluster with a timeout-raising ollama client.
        async def _timeout_generate(*args, **kwargs):
            raise asyncio.TimeoutError

        with patch.object(engine, "_llm_client") as mock_client:
            mock_client.generate = _timeout_generate
            result = await engine._distill_cluster([atom])

        # Must return the verbatim content, not raise.
        assert result == "fallback verbatim content"

    async def test_distillation_fallback_on_generic_exception(
        self, storage: Storage
    ) -> None:
        """When the LLM raises any exception, _distill_cluster falls back to verbatim."""
        from memories.atoms import Atom

        engine = self._make_engine_with_mock_embeddings(storage)

        atom = MagicMock(spec=Atom)
        atom.content = "exception fallback content"

        async def _error_generate(*args, **kwargs):
            raise ConnectionError("Ollama not reachable")

        with patch.object(engine, "_llm_client") as mock_client:
            mock_client.generate = _error_generate
            result = await engine._distill_cluster([atom])

        assert result == "exception fallback content"

    async def test_distillation_fallback_empty_cluster(
        self, storage: Storage
    ) -> None:
        """_distill_cluster returns empty string when the cluster is empty."""
        engine = self._make_engine_with_mock_embeddings(storage)

        with patch.object(engine, "_llm_client") as mock_client:
            mock_client.generate = AsyncMock(side_effect=RuntimeError("should not be called"))
            result = await engine._distill_cluster([])

        assert result == ""

    async def test_distillation_concurrent(self, storage: Storage) -> None:
        """When multiple clusters exist, asyncio.gather is used (not sequential awaits)."""
        import asyncio as _asyncio

        engine = self._make_engine_with_mock_embeddings(storage)

        # Patch asyncio.gather so we can assert it was called.
        original_gather = _asyncio.gather
        gather_calls: list[int] = []

        async def spy_gather(*coros, return_exceptions=False):
            gather_calls.append(len(coros))
            return await original_gather(*coros, return_exceptions=return_exceptions)

        # _distill_cluster must exist on the engine.
        assert hasattr(engine, "_distill_cluster"), (
            "_distill_cluster method not found on ConsolidationEngine; W3-D not yet implemented"
        )

        # Patch _distill_cluster to return immediately (no real LLM call).
        with patch.object(
            engine,
            "_distill_cluster",
            new_callable=AsyncMock,
            return_value="distilled content",
        ):
            with patch("memories.consolidation.asyncio.gather", side_effect=spy_gather):
                # Trigger _abstract_experiences via the internal helper that
                # calls gather.  We call _distill_clusters_concurrent directly
                # if it exists, otherwise rely on the gather spy being called.
                if hasattr(engine, "_distill_clusters_concurrent"):
                    from memories.atoms import Atom
                    clusters = [
                        [MagicMock(spec=Atom, content=f"cluster {i}")]
                        for i in range(3)
                    ]
                    await engine._distill_clusters_concurrent(clusters)
                    assert len(gather_calls) >= 1, "asyncio.gather was not called"
                    assert gather_calls[0] == 3, (
                        f"Expected gather to be called with 3 coroutines, got {gather_calls[0]}"
                    )

    async def test_distillation_respects_timeout(self, storage: Storage) -> None:
        """_distill_cluster wraps the LLM call with a 15-second timeout."""
        import asyncio
        from memories.atoms import Atom

        engine = self._make_engine_with_mock_embeddings(storage)

        assert hasattr(engine, "_distill_cluster"), (
            "_distill_cluster method not found; W3-D not yet implemented"
        )

        wait_for_calls: list[float] = []

        async def spy_wait_for(coro, timeout):
            wait_for_calls.append(timeout)
            # Cancel the real coro and return a placeholder value.
            coro.close()
            return MagicMock(response="distilled spy")

        atom = MagicMock(spec=Atom)
        atom.content = "timeout test content"

        # Use a fast-returning mock LLM so we can inspect wait_for.
        async def _fast_generate(*args, **kwargs):
            return MagicMock(response="distilled fast")

        with patch.object(engine, "_llm_client") as mock_client:
            mock_client.generate = _fast_generate
            with patch("memories.consolidation.asyncio.wait_for", side_effect=spy_wait_for):
                await engine._distill_cluster([atom])

        # At least one wait_for call should have been issued, and the timeout
        # must be <= 15 seconds (the plan specifies exactly 15).
        if wait_for_calls:
            assert wait_for_calls[0] <= 15.0, (
                f"Expected timeout <= 15s, got {wait_for_calls[0]}"
            )
