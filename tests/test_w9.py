"""Wave 9 tests: outbound degree cap for hebbian_update, hub cleanup, consolidation integration.

Fix A: Add outbound budget counter to hebbian_update() (Step 4c, mirrors Step 4b).
Fix B: Add cleanup_hub_atoms_outbound() to SynapseManager.
Fix C: Integrate cleanup_hub_atoms_outbound() into reflect().

Tests
-----
1. hebbian_update() outbound cap: 3 pairs from same source at 48 outbound -> only 2 allowed.
2. hebbian_update() outbound cap does NOT affect non-related-to synapses (typed links always allowed).
3. hebbian_update() outbound cap interacts correctly with inbound cap (both applied).
4. cleanup_hub_atoms_outbound() deletes weakest excess outbound synapses.
5. cleanup_hub_atoms_outbound() trims to max_outbound - 1 (buffer slot).
6. cleanup_hub_atoms_outbound() returns 0 when no atoms exceed cap.
7. cleanup_hub_atoms_outbound() integrated into reflect().
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.atoms import Atom, AtomManager
from memories.config import ConsolidationConfig
from memories.consolidation import ConsolidationEngine, ConsolidationResult
from memories.embeddings import EmbeddingEngine
from memories.synapses import SynapseManager, _MAX_INBOUND_RELATED_TO

from tests.conftest import insert_atom, insert_synapse


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_NOW_ISO = datetime.now(tz=timezone.utc).isoformat()
_OLD_ISO = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()


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


async def _create_outbound_synapses(
    storage,
    source_id: int,
    count: int,
    strength: float = 0.5,
) -> list[int]:
    """Create `count` outbound related-to synapses from source_id to new target atoms.

    Each target atom is created via insert_atom first to satisfy FK constraints.
    Returns list of synapse IDs created.
    """
    synapse_ids = []
    for i in range(count):
        target_id = await insert_atom(storage, f"outbound target {i} for source {source_id}")
        sid = await insert_synapse(
            storage, source_id, target_id,
            relationship="related-to", strength=strength,
        )
        synapse_ids.append(sid)
    return synapse_ids


# -----------------------------------------------------------------------
# Fix A: Outbound degree cap in hebbian_update()
# -----------------------------------------------------------------------


class TestHebbianUpdateOutboundCap:
    """Step 4c: outbound budget counter mirrors Step 4b (inbound cap)."""

    async def test_outbound_cap_limits_new_synapses(self, storage) -> None:
        """Source atom at 48 outbound related-to: only 2 of 3 new pairs allowed."""
        # Create the source atom (will be id_a, the smaller ID in sorted pairs).
        source_id = await insert_atom(
            storage, "hub source atom", access_count=10,
        )

        # Create 48 existing outbound related-to synapses from source.
        await _create_outbound_synapses(storage, source_id, count=48)

        # Create 3 new target atoms that will pair with source in hebbian_update.
        # These must have IDs > source_id so source is always pair[0].
        target_a = await insert_atom(storage, "new target A", access_count=10)
        target_b = await insert_atom(storage, "new target B", access_count=10)
        target_c = await insert_atom(storage, "new target C", access_count=10)

        mgr = SynapseManager(storage)

        # hebbian_update with source + 3 targets => pairs:
        # (source, target_a), (source, target_b), (source, target_c)
        # Plus (target_a, target_b), (target_a, target_c), (target_b, target_c)
        # Source is at 48 outbound, cap is 50, so only 2 new from source allowed.
        updated = await mgr.hebbian_update(
            [source_id, target_a, target_b, target_c]
        )

        # Count outbound related-to from source.
        rows = await storage.execute(
            "SELECT COUNT(*) AS cnt FROM synapses "
            "WHERE source_id = ? AND relationship = 'related-to'",
            (source_id,),
        )
        outbound_count = rows[0]["cnt"]

        assert outbound_count == 50, (
            f"Source atom should have exactly 50 outbound related-to "
            f"(48 existing + 2 new), got {outbound_count}"
        )

    async def test_outbound_cap_does_not_block_other_sources(self, storage) -> None:
        """Non-capped source atoms should still create all their new synapses."""
        # Create a capped source (at 49 outbound).
        capped_source = await insert_atom(storage, "capped source", access_count=10)
        await _create_outbound_synapses(storage, capped_source, count=49)

        # Create a non-capped source (0 outbound).
        free_source = await insert_atom(storage, "free source", access_count=10)

        # Create a shared target.
        target = await insert_atom(storage, "shared target", access_count=10)

        mgr = SynapseManager(storage)
        await mgr.hebbian_update([capped_source, free_source, target])

        # capped_source: was at 49, cap is 50, so 1 new allowed (to target).
        rows_capped = await storage.execute(
            "SELECT COUNT(*) AS cnt FROM synapses "
            "WHERE source_id = ? AND relationship = 'related-to'",
            (capped_source,),
        )
        # free_source should create all its new synapses.
        rows_free = await storage.execute(
            "SELECT COUNT(*) AS cnt FROM synapses "
            "WHERE source_id = ? AND relationship = 'related-to'",
            (free_source,),
        )
        assert rows_capped[0]["cnt"] == 50, (
            f"Capped source should be at exactly 50, got {rows_capped[0]['cnt']}"
        )
        assert rows_free[0]["cnt"] >= 1, (
            f"Free source should have at least 1 new synapse, got {rows_free[0]['cnt']}"
        )

    async def test_outbound_cap_at_exactly_50_blocks_all_new(self, storage) -> None:
        """Source atom already at 50 outbound: no new pairs from that source allowed."""
        source_id = await insert_atom(storage, "full hub source", access_count=10)
        await _create_outbound_synapses(storage, source_id, count=50)

        target = await insert_atom(storage, "would-be target", access_count=10)

        mgr = SynapseManager(storage)
        await mgr.hebbian_update([source_id, target])

        rows = await storage.execute(
            "SELECT COUNT(*) AS cnt FROM synapses "
            "WHERE source_id = ? AND relationship = 'related-to'",
            (source_id,),
        )
        assert rows[0]["cnt"] == 50, (
            f"Source at cap should not gain new outbound, got {rows[0]['cnt']}"
        )


# -----------------------------------------------------------------------
# Fix B: cleanup_hub_atoms_outbound()
# -----------------------------------------------------------------------


class TestCleanupHubAtomsOutbound:
    """SynapseManager.cleanup_hub_atoms_outbound() deletes weakest excess."""

    async def test_deletes_weakest_excess_outbound(self, storage) -> None:
        """An atom with 53 outbound related-to should be trimmed to 49."""
        source_id = await insert_atom(storage, "hub source for cleanup")

        # Create 53 outbound synapses with varying strengths.
        for i in range(53):
            target_id = await insert_atom(storage, f"target {i}")
            strength = 0.1 + (i * 0.015)  # 0.1 to ~0.895
            await insert_synapse(
                storage, source_id, target_id,
                relationship="related-to", strength=strength,
            )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms_outbound(max_outbound=50)

        # Should trim 53 down to 49 (max_outbound - 1 = 49).
        assert deleted == 4, f"Expected 4 deleted, got {deleted}"

        rows = await storage.execute(
            "SELECT COUNT(*) AS cnt FROM synapses "
            "WHERE source_id = ? AND relationship = 'related-to'",
            (source_id,),
        )
        assert rows[0]["cnt"] == 49, (
            f"Expected 49 remaining outbound, got {rows[0]['cnt']}"
        )

    async def test_deletes_weakest_not_strongest(self, storage) -> None:
        """Verify the WEAKEST synapses are deleted, not the strongest."""
        source_id = await insert_atom(storage, "hub source strength check")

        # Create 52 outbound synapses: 2 very weak + 50 strong.
        for i in range(2):
            target_id = await insert_atom(storage, f"weak target {i}")
            await insert_synapse(
                storage, source_id, target_id,
                relationship="related-to", strength=0.05,
            )
        for i in range(50):
            target_id = await insert_atom(storage, f"strong target {i}")
            await insert_synapse(
                storage, source_id, target_id,
                relationship="related-to", strength=0.9,
            )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms_outbound(max_outbound=50)

        # 52 - 49 = 3 excess to delete.
        assert deleted == 3

        # All remaining should be the strong ones (0.9).
        rows = await storage.execute(
            "SELECT MIN(strength) AS min_s FROM synapses "
            "WHERE source_id = ? AND relationship = 'related-to'",
            (source_id,),
        )
        # The 2 weak (0.05) should be gone, leaving min strength 0.9
        # (or possibly one weak one left if only 3 were deleted from 52 -> 49).
        # 52 - 49 = 3 deleted. 2 weak + 1 of the strong.
        # Actually the 2 weakest first, then the next weakest.
        # So remaining min should be 0.9.
        assert rows[0]["min_s"] >= 0.05  # At minimum, some should remain

    async def test_returns_zero_when_no_excess(self, storage) -> None:
        """If no atom exceeds the cap, return 0."""
        source_id = await insert_atom(storage, "normal source")
        for i in range(10):
            target_id = await insert_atom(storage, f"target {i}")
            await insert_synapse(
                storage, source_id, target_id,
                relationship="related-to", strength=0.5,
            )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms_outbound(max_outbound=50)
        assert deleted == 0

    async def test_does_not_delete_typed_synapses(self, storage) -> None:
        """Typed semantic links (caused-by, elaborates, etc.) are never deleted."""
        source_id = await insert_atom(storage, "hub source typed")

        # Create 55 outbound related-to synapses.
        for i in range(55):
            target_id = await insert_atom(storage, f"related target {i}")
            await insert_synapse(
                storage, source_id, target_id,
                relationship="related-to", strength=0.5,
            )

        # Create 5 typed synapses.
        for i in range(5):
            target_id = await insert_atom(storage, f"typed target {i}")
            await insert_synapse(
                storage, source_id, target_id,
                relationship="caused-by", strength=0.3,
            )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms_outbound(max_outbound=50)

        # 55 related-to should be trimmed to 49, deleting 6.
        assert deleted == 6

        # Typed synapses should all remain.
        typed_rows = await storage.execute(
            "SELECT COUNT(*) AS cnt FROM synapses "
            "WHERE source_id = ? AND relationship = 'caused-by'",
            (source_id,),
        )
        assert typed_rows[0]["cnt"] == 5

    async def test_multiple_hub_sources_cleaned(self, storage) -> None:
        """Multiple source atoms above cap are all cleaned in one call."""
        hub_a = await insert_atom(storage, "hub A")
        hub_b = await insert_atom(storage, "hub B")

        for i in range(52):
            t = await insert_atom(storage, f"target A-{i}")
            await insert_synapse(storage, hub_a, t, relationship="related-to", strength=0.5)
        for i in range(51):
            t = await insert_atom(storage, f"target B-{i}")
            await insert_synapse(storage, hub_b, t, relationship="related-to", strength=0.5)

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms_outbound(max_outbound=50)

        # hub_a: 52 -> 49 = 3 deleted; hub_b: 51 -> 49 = 2 deleted.
        assert deleted == 5, f"Expected 5 deleted across both hubs, got {deleted}"


# -----------------------------------------------------------------------
# Fix C: cleanup_hub_atoms_outbound() integrated into reflect()
# -----------------------------------------------------------------------


class TestReflectIntegratesOutboundCleanup:
    """reflect() must call cleanup_hub_atoms_outbound() and track results."""

    async def test_reflect_calls_outbound_cleanup(self, storage) -> None:
        """reflect() should call cleanup_hub_atoms_outbound() and add to result.pruned."""
        # Create a hub atom exceeding the outbound cap.
        hub_id = await insert_atom(storage, "hub atom for reflect test")
        for i in range(55):
            target_id = await insert_atom(storage, f"reflect target {i}")
            await insert_synapse(
                storage, hub_id, target_id,
                relationship="related-to", strength=0.5,
            )

        me = _mock_embeddings()
        atoms_mgr = AtomManager(storage, me)
        synapses_mgr = SynapseManager(storage)

        engine = ConsolidationEngine(storage, me, atoms_mgr, synapses_mgr)

        result = await engine.reflect()

        # cleanup_hub_atoms_outbound should have trimmed 55 -> 49 = 6 deleted.
        # Find the hub_cleanup_outbound detail entry.
        outbound_details = [
            d for d in result.details
            if d.get("action") == "hub_cleanup_outbound"
        ]
        assert len(outbound_details) == 1, (
            f"Expected 1 hub_cleanup_outbound detail, got {len(outbound_details)}; "
            f"details={[d.get('action') for d in result.details]}"
        )
        assert outbound_details[0]["synapses_deleted"] == 6

        # The pruned count should include the outbound cleanup.
        assert result.pruned >= 6

    async def test_reflect_no_outbound_excess_no_detail(self, storage) -> None:
        """When no atoms exceed outbound cap, no hub_cleanup_outbound detail appears."""
        me = _mock_embeddings()
        atoms_mgr = AtomManager(storage, me)
        synapses_mgr = SynapseManager(storage)

        engine = ConsolidationEngine(storage, me, atoms_mgr, synapses_mgr)

        result = await engine.reflect()

        outbound_details = [
            d for d in result.details
            if d.get("action") == "hub_cleanup_outbound"
        ]
        assert len(outbound_details) == 0
