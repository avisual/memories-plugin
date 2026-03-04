"""Wave 6-D: cleanup_hub_atoms() window-query optimisation.

After hub detection returns N hub atoms, the old code issued N separate
SELECT queries (one per hub) to find the weakest synapses.  The fix
replaces these with a single SQL window-function query (ROW_NUMBER OVER
PARTITION BY) that identifies all excess synapses across all hubs at once,
followed by a single batched DELETE.

W7-B1 update: HAVING cnt >= ? (was cnt > ?) and excess formula trims to
max_inbound - 1 (one buffer slot below cap).  Atoms at exactly cap now
get cleaned.

W7-B2 update: The CTE is inlined into a single atomic DELETE via
execute_write_returning (no separate read + write).

Tests
-----
1. Correctness: weakest per hub deleted, trimmed to max_inbound - 1.
2. Efficiency: only 1 write SQL call (atomic CTE+DELETE), not 1 read + 1 write.
3. Return value equals the total count of deleted synapses.
4. Multiple hubs in a single pass.
5. Large hub with >500 excess synapses handled in single DELETE.
"""

from __future__ import annotations

import pytest

from memories.synapses import SynapseManager

from tests.conftest import count_synapses, insert_atom, insert_synapse


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


async def _create_hub(
    storage,
    name: str,
    inbound_count: int,
    *,
    strength_start: float = 0.01,
    strength_step: float = 0.01,
) -> int:
    """Create a hub atom with *inbound_count* inbound related-to synapses.

    Strengths are assigned ascending from *strength_start* so that the
    weakest synapses are deterministic: the first ones inserted.
    """
    hub_id = await insert_atom(storage, name, access_count=10)
    for i in range(inbound_count):
        src = await insert_atom(storage, f"{name} src {i}", access_count=1)
        strength = round(strength_start + strength_step * i, 6)
        await insert_synapse(
            storage, src, hub_id, relationship="related-to", strength=strength,
        )
    return hub_id


# -----------------------------------------------------------------------
# 1. Correctness: weakest synapses per hub are deleted
# -----------------------------------------------------------------------


class TestCleanupHubWindowCorrectness:
    """The window-query implementation must produce identical results to the
    old per-hub loop: for each over-cap hub, the weakest ``related-to``
    synapses are deleted, leaving exactly *max_inbound* survivors.
    """

    async def test_single_hub_weakest_deleted(self, storage) -> None:
        """55 inbound related-to on one hub, max_inbound=50 -> 6 weakest gone (to 49 buffer)."""
        hub_id = await _create_hub(storage, "hub-A", inbound_count=55)

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        assert deleted == 6  # W7-B1: trims to 49 (cap-1 buffer)
        remaining = await count_synapses(
            storage, target_id=hub_id, relationship="related-to",
        )
        assert remaining == 49

        # The weakest 6 (strengths 0.01 .. 0.06) should be gone.
        rows = await storage.execute(
            "SELECT strength FROM synapses "
            "WHERE target_id = ? AND relationship = 'related-to' "
            "ORDER BY strength ASC",
            (hub_id,),
        )
        weakest_remaining = rows[0]["strength"]
        assert weakest_remaining == pytest.approx(0.07, abs=1e-4), (
            f"Expected weakest remaining ~0.07, got {weakest_remaining}"
        )

    async def test_multiple_hubs_all_pruned(self, storage) -> None:
        """Three hubs each 11 over target (60 -> 49) -> 33 total deletions."""
        hubs = []
        for i in range(3):
            hid = await _create_hub(
                storage, f"multi-hub-{i}", inbound_count=60,
            )
            hubs.append(hid)

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        assert deleted == 33  # 3 hubs * 11 excess each (60 -> 49 buffer)
        for hid in hubs:
            remaining = await count_synapses(
                storage, target_id=hid, relationship="related-to",
            )
            assert remaining == 49

    async def test_typed_links_not_deleted(self, storage) -> None:
        """Non-related-to synapses must never be touched by hub cleanup."""
        hub_id = await _create_hub(storage, "typed-hub", inbound_count=55)

        # Add some typed semantic links.
        for rel in ("caused-by", "contradicts", "elaborates"):
            src = await insert_atom(storage, f"typed src {rel}", access_count=1)
            await insert_synapse(
                storage, src, hub_id, relationship=rel, strength=0.01,
            )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        # W7-B1: 6 weakest related-to synapses deleted (55 -> 49 buffer).
        assert deleted == 6

        # Typed links survive.
        for rel in ("caused-by", "contradicts", "elaborates"):
            cnt = await count_synapses(
                storage, target_id=hub_id, relationship=rel,
            )
            assert cnt == 1, f"Typed link '{rel}' was unexpectedly deleted"

    async def test_no_op_when_under_cap(self, storage) -> None:
        """Hubs strictly below the cap should not be touched.

        W7-B1: Atoms at exactly cap ARE cleaned (to cap-1 buffer).
        So we test with inbound_count=49 (truly under cap).
        """
        await _create_hub(storage, "under-cap", inbound_count=49)

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        assert deleted == 0


# -----------------------------------------------------------------------
# 2. Efficiency: SQL call count
# -----------------------------------------------------------------------


class TestCleanupHubWindowEfficiency:
    """W7-B2: The atomic implementation uses exactly 1 SQL call total:
    a single execute_write_returning with inline CTE+DELETE, regardless of
    the number of hubs.  No separate read call.
    """

    async def test_sql_call_count_single_hub(
        self, storage, monkeypatch,
    ) -> None:
        """One hub over cap -> exactly 1 write call (atomic CTE+DELETE), 0 reads."""
        await _create_hub(storage, "eff-hub", inbound_count=55)

        read_calls: list[str] = []
        write_calls: list[str] = []
        original_execute = storage.execute
        original_write_returning = storage.execute_write_returning

        async def spy_execute(sql, params=()):
            read_calls.append(sql)
            return await original_execute(sql, params)

        async def spy_write_returning(sql, params=()):
            write_calls.append(sql)
            return await original_write_returning(sql, params)

        monkeypatch.setattr(storage, "execute", spy_execute)
        monkeypatch.setattr(
            storage, "execute_write_returning", spy_write_returning,
        )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        assert deleted == 6  # W7-B1: 55 -> 49

        # W7-B2: No separate read call — CTE is inlined into DELETE.
        assert len(read_calls) == 0, (
            f"Expected 0 read SQL calls (atomic CTE+DELETE), got {len(read_calls)}: {read_calls}"
        )
        # Exactly 1 write call (the atomic CTE+DELETE).
        assert len(write_calls) == 1, (
            f"Expected 1 write SQL call, got {len(write_calls)}: {write_calls}"
        )

    async def test_sql_call_count_many_hubs(
        self, storage, monkeypatch,
    ) -> None:
        """10 hubs over cap -> still exactly 1 write call, 0 reads."""
        for i in range(10):
            await _create_hub(
                storage, f"many-hub-{i}", inbound_count=55,
            )

        read_calls: list[str] = []
        write_calls: list[str] = []
        original_execute = storage.execute
        original_write_returning = storage.execute_write_returning

        async def spy_execute(sql, params=()):
            read_calls.append(sql)
            return await original_execute(sql, params)

        async def spy_write_returning(sql, params=()):
            write_calls.append(sql)
            return await original_write_returning(sql, params)

        monkeypatch.setattr(storage, "execute", spy_execute)
        monkeypatch.setattr(
            storage, "execute_write_returning", spy_write_returning,
        )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        assert deleted == 60  # W7-B1: 10 hubs * 6 excess (55 -> 49)
        assert len(read_calls) == 0, (
            f"Expected 0 read SQL calls, got {len(read_calls)}"
        )
        assert len(write_calls) == 1, (
            f"Expected 1 write SQL call, got {len(write_calls)}"
        )

    async def test_no_write_call_when_nothing_to_delete(
        self, storage, monkeypatch,
    ) -> None:
        """When no hub exceeds the cap, the atomic DELETE returns 0 rows.

        W7-B1: Atoms at exactly cap are cleaned, so use inbound=49 (under cap).
        W7-B2: The atomic CTE+DELETE is always issued (1 write call) but
        returns 0 rows when there's nothing to delete.
        """
        await _create_hub(storage, "no-excess", inbound_count=49)

        write_calls: list[str] = []
        original_write_returning = storage.execute_write_returning

        async def spy_write_returning(sql, params=()):
            write_calls.append(sql)
            return await original_write_returning(sql, params)

        monkeypatch.setattr(
            storage, "execute_write_returning", spy_write_returning,
        )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        assert deleted == 0
        # W7-B2: Atomic CTE+DELETE is always issued as 1 write call.
        assert len(write_calls) == 1, (
            f"Expected 1 write call (atomic CTE+DELETE), got {len(write_calls)}"
        )


# -----------------------------------------------------------------------
# 3. Return value
# -----------------------------------------------------------------------


class TestCleanupHubReturnValue:
    """cleanup_hub_atoms must return the exact count of deleted synapses."""

    async def test_return_value_matches_deletion_count(self, storage) -> None:
        """Two hubs, 8 and 4 over target (49) -> returns 12.

        W7-B1: hub_a: 57 -> 49 = 8, hub_b: 53 -> 49 = 4, total = 12.
        """
        hub_a = await _create_hub(storage, "rv-hub-a", inbound_count=57)
        hub_b = await _create_hub(storage, "rv-hub-b", inbound_count=53)

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        assert deleted == 12

        remaining_a = await count_synapses(
            storage, target_id=hub_a, relationship="related-to",
        )
        remaining_b = await count_synapses(
            storage, target_id=hub_b, relationship="related-to",
        )
        assert remaining_a == 49
        assert remaining_b == 49


# -----------------------------------------------------------------------
# 4. Batched DELETE for large ID sets (>500)
# -----------------------------------------------------------------------


class TestCleanupHubBatchedDelete:
    """W7-B2: The atomic CTE+DELETE handles all excess in a single call.
    No batching needed since the CTE is inlined into the DELETE.
    """

    async def test_large_excess_single_call(self, storage, monkeypatch) -> None:
        """Hub with 601 excess synapses (1100 -> 499) -> 1 atomic DELETE call.

        W7-B1: trims to max_inbound - 1 = 499, so 1100 - 499 = 601.
        W7-B2: single atomic CTE+DELETE, no batching.
        """
        # Create a hub with 1100 inbound related-to, max_inbound=500
        # -> 601 excess (trims to 499 buffer).
        hub_id = await _create_hub(storage, "big-hub", inbound_count=1100)

        write_calls: list[str] = []
        original_write_returning = storage.execute_write_returning

        async def spy_write_returning(sql, params=()):
            write_calls.append(sql)
            return await original_write_returning(sql, params)

        monkeypatch.setattr(
            storage, "execute_write_returning", spy_write_returning,
        )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=500)

        assert deleted == 601  # W7-B1: 1100 -> 499

        remaining = await count_synapses(
            storage, target_id=hub_id, relationship="related-to",
        )
        assert remaining == 499  # W7-B1: cap - 1 buffer

        # W7-B2: single atomic CTE+DELETE call (no batching needed).
        assert len(write_calls) == 1, (
            f"Expected 1 atomic DELETE call, got {len(write_calls)}"
        )
