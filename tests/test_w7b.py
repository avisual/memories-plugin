"""Wave 7-C+D: double-decay exclusion, config-driven dormant params, batch merge.

Fixes
-----
C1. _prune_dormant_synapses excludes synapses already covered by _apply_ltd
    (both endpoint atoms recently active) to prevent double-decay.
C2. dormant_cutoff_days and dormant_multiplier are now config-driven via
    ConsolidationConfig instead of hardcoded.
D1. _merge_duplicates uses get_batch_without_tracking instead of per-item
    get_without_tracking, eliminating N+1 reads.
M1. _abstract_experiences dedup guard uses batch fetch for similar-fact check.

Tests
-----
1. _prune_dormant_synapses excludes synapses whose BOTH endpoints are recently
   accessed (those are handled by _apply_ltd).
2. _prune_dormant_synapses still decays synapses with at least one stale
   endpoint atom.
3. dormant_cutoff_days and dormant_multiplier are config-driven.
4. _merge_duplicates calls get_batch_without_tracking, not per-item
   get_without_tracking.
5. _abstract_experiences dedup guard uses batch fetch.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.atoms import Atom, AtomManager
from memories.config import ConsolidationConfig
from memories.consolidation import ConsolidationEngine, ConsolidationResult
from memories.embeddings import EmbeddingEngine
from memories.synapses import SynapseManager

from tests.conftest import insert_atom, insert_synapse


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_NOW_ISO = datetime.now(tz=timezone.utc).isoformat()
_OLD_ISO = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()
_RECENT_ISO = (datetime.now(tz=timezone.utc) - timedelta(days=5)).isoformat()


def _mock_embeddings() -> MagicMock:
    """Return a MagicMock standing in for EmbeddingEngine."""
    engine = MagicMock(spec=EmbeddingEngine)
    engine.search_similar = AsyncMock(return_value=[])
    engine.embed_text = AsyncMock(return_value=[0.0] * 768)
    engine.embed_and_store = AsyncMock(return_value=[0.0] * 768)
    engine.embed_batch = AsyncMock(return_value=[])
    engine.health_check = AsyncMock(return_value=True)
    engine.cosine_similarity = MagicMock(return_value=0.5)
    return engine


async def _insert_dormant_synapse(
    storage,
    source_id: int,
    target_id: int,
    strength: float = 0.5,
    created_days_ago: int = 45,
) -> int:
    """Insert a synapse that has never been activated, created N days ago."""
    sid = await insert_synapse(
        storage, source_id, target_id, relationship="related-to", strength=strength,
    )
    created_at = (
        datetime.now(tz=timezone.utc) - timedelta(days=created_days_ago)
    ).isoformat()
    await storage.execute_write(
        "UPDATE synapses SET last_activated_at = NULL, created_at = ? WHERE id = ?",
        (created_at, sid),
    )
    return sid


# -----------------------------------------------------------------------
# C1. Exclude _apply_ltd candidates from _prune_dormant_synapses
# -----------------------------------------------------------------------


class TestDormantExcludesLtdCandidates:
    """Synapses where BOTH endpoint atoms are recently active must not be
    decayed by _prune_dormant_synapses (they are handled by _apply_ltd).
    """

    async def test_both_endpoints_recent_excluded(self, storage) -> None:
        """A dormant synapse between two recently-accessed atoms is skipped."""
        a1 = await insert_atom(storage, "recent atom 1", last_accessed_at=_RECENT_ISO)
        a2 = await insert_atom(storage, "recent atom 2", last_accessed_at=_RECENT_ISO)
        sid = await _insert_dormant_synapse(storage, a1, a2, strength=0.5)

        engine = ConsolidationEngine(
            storage,
            MagicMock(spec=EmbeddingEngine),
            AtomManager(storage, _mock_embeddings()),
            SynapseManager(storage),
        )
        result = ConsolidationResult()
        await engine._prune_dormant_synapses(result)

        # Synapse should NOT have been decayed.
        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (sid,),
        )
        assert len(rows) == 1
        assert rows[0]["strength"] == pytest.approx(0.5, abs=1e-6), (
            "Synapse between two recently-active atoms should be untouched "
            "by dormant pruning (handled by LTD instead)"
        )

    async def test_one_endpoint_stale_still_decayed(self, storage) -> None:
        """A dormant synapse with at least one stale endpoint IS decayed."""
        a_recent = await insert_atom(storage, "recent atom", last_accessed_at=_RECENT_ISO)
        a_stale = await insert_atom(storage, "stale atom", last_accessed_at=_OLD_ISO)
        sid = await _insert_dormant_synapse(storage, a_recent, a_stale, strength=0.5)

        engine = ConsolidationEngine(
            storage,
            MagicMock(spec=EmbeddingEngine),
            AtomManager(storage, _mock_embeddings()),
            SynapseManager(storage),
        )
        result = ConsolidationResult()
        await engine._prune_dormant_synapses(result)

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (sid,),
        )
        assert len(rows) == 1
        assert rows[0]["strength"] < 0.5, (
            "Synapse with one stale endpoint should be decayed by dormant pruning"
        )

    async def test_both_endpoints_stale_still_decayed(self, storage) -> None:
        """A dormant synapse with both endpoints stale IS decayed."""
        a1 = await insert_atom(storage, "stale atom 1", last_accessed_at=_OLD_ISO)
        a2 = await insert_atom(storage, "stale atom 2", last_accessed_at=_OLD_ISO)
        sid = await _insert_dormant_synapse(storage, a1, a2, strength=0.5)

        engine = ConsolidationEngine(
            storage,
            MagicMock(spec=EmbeddingEngine),
            AtomManager(storage, _mock_embeddings()),
            SynapseManager(storage),
        )
        result = ConsolidationResult()
        await engine._prune_dormant_synapses(result)

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (sid,),
        )
        assert len(rows) == 1
        assert rows[0]["strength"] < 0.5, (
            "Synapse with both stale endpoints should be decayed by dormant pruning"
        )

    async def test_null_last_accessed_endpoint_still_decayed(self, storage) -> None:
        """A dormant synapse where one endpoint has NULL last_accessed_at IS decayed."""
        a_recent = await insert_atom(storage, "recent atom", last_accessed_at=_RECENT_ISO)
        a_null = await insert_atom(storage, "null atom", last_accessed_at=None)
        # Manually set last_accessed_at to NULL (insert_atom defaults to now).
        await storage.execute_write(
            "UPDATE atoms SET last_accessed_at = NULL WHERE id = ?", (a_null,),
        )
        sid = await _insert_dormant_synapse(storage, a_recent, a_null, strength=0.5)

        engine = ConsolidationEngine(
            storage,
            MagicMock(spec=EmbeddingEngine),
            AtomManager(storage, _mock_embeddings()),
            SynapseManager(storage),
        )
        result = ConsolidationResult()
        await engine._prune_dormant_synapses(result)

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (sid,),
        )
        assert len(rows) == 1
        assert rows[0]["strength"] < 0.5, (
            "Synapse with one NULL-accessed endpoint should be decayed"
        )


# -----------------------------------------------------------------------
# C2. dormant_cutoff_days and dormant_multiplier are config-driven
# -----------------------------------------------------------------------


class TestDormantConfigDriven:
    """Verify that dormant decay uses ConsolidationConfig fields."""

    def test_config_has_dormant_fields(self) -> None:
        """ConsolidationConfig exposes dormant_cutoff_days and dormant_multiplier."""
        cfg = ConsolidationConfig()
        assert hasattr(cfg, "dormant_cutoff_days"), (
            "ConsolidationConfig must have dormant_cutoff_days"
        )
        assert hasattr(cfg, "dormant_multiplier"), (
            "ConsolidationConfig must have dormant_multiplier"
        )
        assert cfg.dormant_cutoff_days == 30
        assert cfg.dormant_multiplier == pytest.approx(0.80, abs=1e-6)

    async def test_custom_dormant_multiplier_applied(self, storage) -> None:
        """A custom dormant_multiplier should change the decay factor."""
        a1 = await insert_atom(storage, "atom 1", last_accessed_at=_OLD_ISO)
        a2 = await insert_atom(storage, "atom 2", last_accessed_at=_OLD_ISO)
        sid = await _insert_dormant_synapse(storage, a1, a2, strength=0.5)

        engine = ConsolidationEngine(
            storage,
            MagicMock(spec=EmbeddingEngine),
            AtomManager(storage, _mock_embeddings()),
            SynapseManager(storage),
        )
        # Override config to use a different multiplier.
        engine._cfg = ConsolidationConfig(dormant_multiplier=0.50)

        result = ConsolidationResult()
        await engine._prune_dormant_synapses(result)

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (sid,),
        )
        assert len(rows) == 1
        # With multiplier=0.50, strength 0.5 * 0.50 = 0.25 (above default prune_threshold 0.05)
        assert rows[0]["strength"] == pytest.approx(0.25, abs=0.01), (
            f"Expected strength ~0.25 with multiplier=0.50, got {rows[0]['strength']}"
        )

    async def test_custom_dormant_cutoff_days(self, storage) -> None:
        """A synapse younger than dormant_cutoff_days is NOT decayed."""
        a1 = await insert_atom(storage, "atom 1", last_accessed_at=_OLD_ISO)
        a2 = await insert_atom(storage, "atom 2", last_accessed_at=_OLD_ISO)
        # Created 20 days ago -- younger than default 30-day cutoff
        sid = await _insert_dormant_synapse(storage, a1, a2, strength=0.5, created_days_ago=20)

        engine = ConsolidationEngine(
            storage,
            MagicMock(spec=EmbeddingEngine),
            AtomManager(storage, _mock_embeddings()),
            SynapseManager(storage),
        )
        result = ConsolidationResult()
        await engine._prune_dormant_synapses(result)

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (sid,),
        )
        assert len(rows) == 1
        assert rows[0]["strength"] == pytest.approx(0.5, abs=1e-6), (
            "Synapse younger than dormant_cutoff_days should not be decayed"
        )

        # Now override config with a shorter cutoff (10 days) -- should now decay.
        engine._cfg = ConsolidationConfig(dormant_cutoff_days=10)
        result2 = ConsolidationResult()
        await engine._prune_dormant_synapses(result2)

        rows2 = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (sid,),
        )
        assert len(rows2) == 1
        assert rows2[0]["strength"] < 0.5, (
            "Synapse older than custom dormant_cutoff_days=10 should be decayed"
        )


# -----------------------------------------------------------------------
# D1. _merge_duplicates uses get_batch_without_tracking
# -----------------------------------------------------------------------


class TestMergeDuplicatesBatchFetch:
    """_merge_duplicates must use get_batch_without_tracking instead of
    per-item get_without_tracking for candidate atoms.
    """

    async def test_merge_uses_batch_not_individual(self, storage) -> None:
        """Verify get_batch_without_tracking is called and
        get_without_tracking is NOT called during the inner merge loop.
        """
        # Create two atoms with the same type.
        a1 = await insert_atom(storage, "atom alpha", atom_type="fact")
        a2 = await insert_atom(storage, "atom beta", atom_type="fact")

        me = _mock_embeddings()
        atoms_mgr = AtomManager(storage, me)
        # Return a2 as a similar result for a1.
        me.search_similar = AsyncMock(
            return_value=[(a2, 0.01)]  # distance 0.01 -> similarity ~0.99
        )
        me.distance_to_similarity = EmbeddingEngine.distance_to_similarity

        engine = ConsolidationEngine(
            storage,
            me,
            atoms_mgr,
            SynapseManager(storage),
        )
        engine._cfg = ConsolidationConfig(merge_threshold=0.95)

        # Spy on get_batch_without_tracking and get_without_tracking.
        batch_calls: list[list[int]] = []
        individual_calls: list[int] = []

        original_batch = atoms_mgr.get_batch_without_tracking
        original_individual = atoms_mgr.get_without_tracking

        async def spy_batch(ids):
            batch_calls.append(list(ids))
            return await original_batch(ids)

        async def spy_individual(atom_id):
            individual_calls.append(atom_id)
            return await original_individual(atom_id)

        atoms_mgr.get_batch_without_tracking = spy_batch
        atoms_mgr.get_without_tracking = spy_individual

        result = ConsolidationResult()
        await engine._merge_duplicates(result, "all")

        # get_batch_without_tracking should have been called.
        assert len(batch_calls) >= 1, (
            "Expected get_batch_without_tracking to be called at least once "
            f"during _merge_duplicates; batch_calls={batch_calls}"
        )

        # get_without_tracking should NOT have been called for candidate atoms.
        assert len(individual_calls) == 0, (
            "Expected get_without_tracking NOT to be called during "
            f"_merge_duplicates inner loop; individual_calls={individual_calls}"
        )


# -----------------------------------------------------------------------
# M1. _abstract_experiences dedup guard uses batch fetch
# -----------------------------------------------------------------------


class TestAbstractExperiencesBatchDedup:
    """The dedup guard in _abstract_experiences must use batch fetch
    instead of per-item get_without_tracking for similar-fact checks.
    """

    async def test_dedup_guard_uses_batch(self, storage) -> None:
        """The dedup guard should call get_batch_without_tracking, not
        per-item get_without_tracking, when checking for existing facts.
        """
        # Create enough experience atoms to trigger abstraction.
        min_age_days = 7
        old_date = (
            datetime.now(tz=timezone.utc) - timedelta(days=min_age_days + 5)
        ).isoformat()
        atom_ids = []
        for i in range(6):
            aid = await insert_atom(
                storage,
                f"experience {i} about testing batch dedup",
                atom_type="experience",
                access_count=5,
                created_at=old_date,
            )
            atom_ids.append(aid)

        # Also create an existing fact atom that the dedup guard should find.
        fact_id = await insert_atom(
            storage,
            "existing fact about testing batch dedup",
            atom_type="fact",
            access_count=10,
            created_at=old_date,
        )

        me = _mock_embeddings()
        atoms_mgr = AtomManager(storage, me)

        # search_similar returns the other atoms for clustering plus the fact for dedup.
        async def fake_search_similar(content, k=10):
            # Return all experience atoms + the fact as similar results.
            results = [(aid, 0.05) for aid in atom_ids]  # low distance = high similarity
            results.append((fact_id, 0.02))  # fact is very similar
            return results

        me.search_similar = AsyncMock(side_effect=fake_search_similar)
        me.distance_to_similarity = EmbeddingEngine.distance_to_similarity

        engine = ConsolidationEngine(
            storage,
            me,
            atoms_mgr,
            SynapseManager(storage),
        )

        # Spy on batch vs individual calls.
        batch_calls: list[list[int]] = []
        individual_calls: list[int] = []

        original_batch = atoms_mgr.get_batch_without_tracking
        original_individual = atoms_mgr.get_without_tracking

        async def spy_batch(ids):
            batch_calls.append(list(ids))
            return await original_batch(ids)

        async def spy_individual(atom_id):
            individual_calls.append(atom_id)
            return await original_individual(atom_id)

        atoms_mgr.get_batch_without_tracking = spy_batch
        atoms_mgr.get_without_tracking = spy_individual

        result = ConsolidationResult()
        await engine._abstract_experiences(result, "all")

        # The dedup guard should use batch fetch, not individual fetch.
        # individual_calls should be empty (no per-item get_without_tracking
        # in the dedup guard section).
        assert len(individual_calls) == 0, (
            "Expected get_without_tracking NOT to be called during "
            f"dedup guard; individual_calls={individual_calls}"
        )
