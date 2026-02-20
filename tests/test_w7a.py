"""Wave 7-A+B tests: hub-violation fixes in hebbian_update and cleanup_hub_atoms.

Fix A1: Within-batch TOCTOU — Python budget counter in hebbian_update()
Fix A2: Cross-session TOCTOU — cap check inside execute_transaction in create()
Fix B1: cleanup_hub_atoms() HAVING off-by-one (>= instead of >)
Fix B2: cleanup_hub_atoms() atomic CTE+DELETE (single execute_write_returning)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from memories.synapses import (
    SynapseManager,
    _MAX_INBOUND_RELATED_TO,
)

from tests.conftest import count_synapses, insert_atom, insert_synapse


# -----------------------------------------------------------------------
# A1: Within-batch TOCTOU — Python budget counter in hebbian_update()
# -----------------------------------------------------------------------


class TestHebbianBatchBudgetCounter:
    """Fix A1: Multiple pairs in one hebbian_update() batch can all target
    the same atom.  The pre-filter checks the DB once, then executes all rows.
    An atom at 48 inbound could end up at 51+ because 3 pairs all pass the
    pre-filter simultaneously.

    The fix adds a Python-side budget counter that accumulates per-target
    within the batch, so only (cap - current) new synapses are allowed.
    """

    async def test_batch_budget_counter_limits_same_target(self, storage) -> None:
        """3 new pairs all targeting same atom (at 48 inbound) must only insert 2.

        Hub must have a HIGH id so it appears as target (second element) in
        sorted pairs: (low_id_atom, high_id_hub).
        """
        # Create source atoms FIRST so they get low IDs.
        existing_sources = []
        for i in range(48):
            src = await insert_atom(storage, f"existing source {i}", access_count=10)
            existing_sources.append(src)

        new_atoms = []
        for i in range(3):
            aid = await insert_atom(storage, f"new atom {i}", access_count=10)
            new_atoms.append(aid)

        # Create hub LAST so it has the highest ID -- it will appear as
        # target (second element) in sorted pairs: (new_atom, hub).
        hub_id = await insert_atom(storage, "hub atom", access_count=10)

        # Fill hub to 48 inbound related-to (2 slots remaining).
        for src in existing_sources:
            await insert_synapse(
                storage, src, hub_id, relationship="related-to", strength=0.5
            )

        mgr = SynapseManager(storage)
        # hebbian_update with hub + 3 new atoms.
        # Sorted pairs targeting hub: (new_0, hub), (new_1, hub), (new_2, hub)
        # Only 2 of the 3 hub-targeting pairs should succeed (cap=50, current=48).
        await mgr.hebbian_update([hub_id] + new_atoms)

        inbound = await count_synapses(
            storage, target_id=hub_id, relationship="related-to"
        )
        assert inbound <= _MAX_INBOUND_RELATED_TO, (
            f"Hub atom exceeded cap: {inbound} inbound related-to synapses "
            f"(cap={_MAX_INBOUND_RELATED_TO}). "
            "Within-batch TOCTOU: budget counter not applied."
        )

    async def test_batch_budget_counter_allows_remaining_slots(self, storage) -> None:
        """Hub at 49 inbound should allow exactly 1 new pair from a batch of 3.

        Hub must have a HIGH id so it appears as target in sorted pairs.
        """
        # Create source atoms FIRST (low IDs).
        existing_sources = []
        for i in range(49):
            src = await insert_atom(storage, f"src49 {i}", access_count=10)
            existing_sources.append(src)

        new_atoms = []
        for i in range(3):
            aid = await insert_atom(storage, f"new49 atom {i}", access_count=10)
            new_atoms.append(aid)

        # Create hub LAST (highest ID = target in sorted pairs).
        hub_id = await insert_atom(storage, "hub atom 49", access_count=10)
        for src in existing_sources:
            await insert_synapse(
                storage, src, hub_id, relationship="related-to", strength=0.5
            )

        mgr = SynapseManager(storage)
        await mgr.hebbian_update([hub_id] + new_atoms)

        inbound = await count_synapses(
            storage, target_id=hub_id, relationship="related-to"
        )
        assert inbound == _MAX_INBOUND_RELATED_TO, (
            f"Expected exactly {_MAX_INBOUND_RELATED_TO} inbound, got {inbound}. "
            "Budget counter should allow exactly 1 slot (49 + 1 = 50)."
        )

    async def test_batch_budget_also_checks_source_side(self, storage) -> None:
        """Budget counter must track targets correctly for bidirectional pairs.

        In hebbian_update, new_pairs are (id_a, id_b, ...) where id_a < id_b.
        The target is id_b (second element). If hub has a high ID, it appears
        as target in the sorted pair. This test ensures the budget tracks the
        correct element.
        """
        # Give hub a high ID by creating it after other atoms.
        filler_atoms = []
        for i in range(5):
            filler_atoms.append(await insert_atom(storage, f"filler {i}", access_count=10))

        hub_id = await insert_atom(storage, "high-id hub", access_count=10)
        # Fill to 48 inbound.
        for i in range(48):
            src = await insert_atom(storage, f"hub fill {i}", access_count=10)
            await insert_synapse(
                storage, src, hub_id, relationship="related-to", strength=0.4
            )

        # The filler atoms (lower IDs) will pair with hub as (filler, hub)
        # so hub is the target (second element) in each pair.
        mgr = SynapseManager(storage)
        await mgr.hebbian_update(filler_atoms + [hub_id])

        inbound = await count_synapses(
            storage, target_id=hub_id, relationship="related-to"
        )
        assert inbound <= _MAX_INBOUND_RELATED_TO, (
            f"Hub exceeded cap: {inbound} inbound (cap={_MAX_INBOUND_RELATED_TO}). "
            "Budget counter tracking wrong pair element."
        )


# -----------------------------------------------------------------------
# A2: Cross-session TOCTOU — cap check inside execute_transaction
# -----------------------------------------------------------------------


class TestCreateAtomicCapCheck:
    """Fix A2: create() currently checks the cap via two reads OUTSIDE the
    write transaction, then writes inside.  Two concurrent calls can both
    read count=49, both proceed, and the atom ends up at 51.

    The fix moves the cap check INSIDE _do_upsert so the count query and
    insert happen atomically within BEGIN IMMEDIATE.
    """

    async def test_create_cap_check_inside_transaction(self, storage) -> None:
        """Cap check must be inside the write transaction for atomicity.

        We verify this by checking that create() uses execute_transaction
        (which wraps in BEGIN IMMEDIATE) and the cap check happens inside
        the transaction callback, not before it.
        """
        hub_id = await insert_atom(storage, "atomic hub", access_count=10)
        # Fill to exactly cap.
        for i in range(_MAX_INBOUND_RELATED_TO):
            src = await insert_atom(storage, f"atomic src {i}", access_count=10)
            await insert_synapse(
                storage, src, hub_id, relationship="related-to", strength=0.5
            )

        newcomer = await insert_atom(storage, "newcomer", access_count=10)

        mgr = SynapseManager(storage)
        result = await mgr.create(
            source_id=newcomer, target_id=hub_id, relationship="related-to"
        )

        # Must be blocked.
        assert result is None, (
            "create() must block new related-to when target is at cap"
        )

        # Verify count did not increase.
        inbound = await count_synapses(
            storage, target_id=hub_id, relationship="related-to"
        )
        assert inbound == _MAX_INBOUND_RELATED_TO

    async def test_create_cap_check_no_separate_read_before_transaction(
        self, storage, monkeypatch
    ) -> None:
        """create() must NOT issue a separate COUNT query outside the transaction.

        After the fix, for new related-to synapses the cap check happens
        inside execute_transaction's callback. We spy on storage.execute()
        to ensure no cap-count query goes through the read path.
        """
        hub_id = await insert_atom(storage, "spy hub", access_count=10)
        # Fill to cap - 1 so the insert would normally succeed.
        for i in range(_MAX_INBOUND_RELATED_TO - 1):
            src = await insert_atom(storage, f"spy src {i}", access_count=10)
            await insert_synapse(
                storage, src, hub_id, relationship="related-to", strength=0.5
            )

        newcomer = await insert_atom(storage, "spy newcomer", access_count=10)

        # Spy on execute (the read-path method).
        read_queries: list[str] = []
        original_execute = storage.execute

        async def spy_execute(sql: str, params=()):
            read_queries.append(sql)
            return await original_execute(sql, params)

        monkeypatch.setattr(storage, "execute", spy_execute)

        mgr = SynapseManager(storage)
        result = await mgr.create(
            source_id=newcomer, target_id=hub_id, relationship="related-to"
        )

        # The synapse should be created (under cap).
        assert result is not None

        # After the fix, create() should NOT issue a separate COUNT query
        # via the read path for the cap check. The _fetch_by_triple read is
        # acceptable, but a "COUNT(*) ... WHERE target_id" cap query should
        # not appear on the read path.
        cap_queries = [
            q for q in read_queries
            if "COUNT(*)" in q and "target_id" in q and "related-to" in q
        ]
        assert len(cap_queries) == 0, (
            f"create() still issues cap-count query via read path: {cap_queries}. "
            "The cap check must be inside execute_transaction for atomicity."
        )


# -----------------------------------------------------------------------
# B1: cleanup_hub_atoms() HAVING off-by-one
# -----------------------------------------------------------------------


class TestCleanupHavingOffByOne:
    """Fix B1: cleanup_hub_atoms() uses `HAVING cnt > ?` which means atoms
    at exactly max_inbound (e.g. 50) are NOT cleaned. They oscillate at
    50-51 each Hebbian cycle.

    The fix changes to `HAVING cnt >= ?` so atoms at exactly cap get
    cleaned. The excess formula also changes to trim down to max_inbound - 1
    (one buffer slot below cap) so atoms land at 49, giving a 1-slot buffer.
    """

    async def test_cleanup_catches_atoms_at_exactly_cap(self, storage) -> None:
        """Hub with exactly 50 inbound related-to must be cleaned (not skipped)."""
        hub_id = await insert_atom(storage, "exactly-at-cap hub", access_count=10)

        for i in range(_MAX_INBOUND_RELATED_TO):
            src = await insert_atom(storage, f"exact src {i}", access_count=10)
            strength = round(0.01 * (i + 1), 3)  # 0.01 .. 0.50
            await insert_synapse(
                storage, src, hub_id, relationship="related-to", strength=strength
            )

        # Verify starting count.
        before = await count_synapses(
            storage, target_id=hub_id, relationship="related-to"
        )
        assert before == _MAX_INBOUND_RELATED_TO

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=_MAX_INBOUND_RELATED_TO)

        # With the off-by-one fix, atoms at exactly cap should be cleaned
        # down to cap - 1 (49), so 1 synapse should be deleted.
        assert deleted >= 1, (
            f"Expected at least 1 deletion for atom at exactly cap={_MAX_INBOUND_RELATED_TO}, "
            f"got {deleted}. HAVING clause likely uses '>' instead of '>='."
        )

        after = await count_synapses(
            storage, target_id=hub_id, relationship="related-to"
        )
        assert after == _MAX_INBOUND_RELATED_TO - 1, (
            f"Expected {_MAX_INBOUND_RELATED_TO - 1} remaining after cleanup, "
            f"got {after}. Excess formula should trim to cap - 1."
        )

    async def test_cleanup_trims_to_one_below_cap(self, storage) -> None:
        """Hub with 55 inbound must be trimmed to cap - 1 (49), not cap (50).

        This ensures the 1-slot buffer: after cleanup, the hub is at 49 so
        the next Hebbian cycle can add 1 without exceeding 50.
        """
        hub_id = await insert_atom(storage, "trim-buffer hub", access_count=10)

        for i in range(55):
            src = await insert_atom(storage, f"trim src {i}", access_count=10)
            strength = round(0.01 * (i + 1), 3)
            await insert_synapse(
                storage, src, hub_id, relationship="related-to", strength=strength
            )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=_MAX_INBOUND_RELATED_TO)

        # Should delete 6 (55 -> 49), not 5 (55 -> 50).
        assert deleted == 6, (
            f"Expected 6 deletions (55 -> 49 for 1-slot buffer), got {deleted}."
        )

        remaining = await count_synapses(
            storage, target_id=hub_id, relationship="related-to"
        )
        assert remaining == _MAX_INBOUND_RELATED_TO - 1, (
            f"Expected {_MAX_INBOUND_RELATED_TO - 1} remaining, got {remaining}."
        )


# -----------------------------------------------------------------------
# B2: Atomic CTE+DELETE in cleanup_hub_atoms()
# -----------------------------------------------------------------------


class TestCleanupAtomicDelete:
    """Fix B2: cleanup_hub_atoms() currently uses execute() (read connection)
    to find IDs via CTE, then execute_write_returning() (write connection)
    to delete them.  Non-atomic: between the read and write, new synapses
    could be added and different synapses deleted.

    The fix inlines the CTE into a single DELETE ... WHERE id IN (SELECT ...)
    executed via execute_write_returning().
    """

    async def test_cleanup_uses_single_write_call(self, storage, monkeypatch) -> None:
        """cleanup_hub_atoms() must NOT call execute() for the CTE read.

        After the fix, the entire CTE+DELETE is a single
        execute_write_returning() call. We spy on execute() to ensure
        no CTE/hub_targets read query goes through the read path.
        """
        hub_id = await insert_atom(storage, "atomic cleanup hub", access_count=10)
        for i in range(55):
            src = await insert_atom(storage, f"ac src {i}", access_count=10)
            strength = round(0.01 * (i + 1), 3)
            await insert_synapse(
                storage, src, hub_id, relationship="related-to", strength=strength
            )

        read_queries: list[str] = []
        original_execute = storage.execute

        async def spy_execute(sql: str, params=()):
            read_queries.append(sql)
            return await original_execute(sql, params)

        monkeypatch.setattr(storage, "execute", spy_execute)

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=_MAX_INBOUND_RELATED_TO)

        assert deleted > 0, "Cleanup should have deleted some synapses"

        # After the fix, NO CTE/hub query should go through the read path.
        cte_reads = [
            q for q in read_queries
            if "hub_targets" in q or ("ranked" in q and "HAVING" in q)
        ]
        assert len(cte_reads) == 0, (
            f"cleanup_hub_atoms() still issues CTE read via execute(): {cte_reads}. "
            "The CTE must be inlined into the DELETE via execute_write_returning()."
        )

    async def test_cleanup_still_returns_correct_count(self, storage) -> None:
        """After refactoring to atomic CTE+DELETE, the return count must be accurate."""
        hub_id = await insert_atom(storage, "count-check hub", access_count=10)
        for i in range(55):
            src = await insert_atom(storage, f"cc src {i}", access_count=10)
            strength = round(0.01 * (i + 1), 3)
            await insert_synapse(
                storage, src, hub_id, relationship="related-to", strength=strength
            )

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=_MAX_INBOUND_RELATED_TO)

        # With B1 fix (>= and cap-1 buffer): 55 -> 49 = 6 deleted.
        assert deleted == 6, (
            f"Expected 6 deletions (55 -> 49), got {deleted}"
        )

        remaining = await count_synapses(
            storage, target_id=hub_id, relationship="related-to"
        )
        assert remaining == _MAX_INBOUND_RELATED_TO - 1
