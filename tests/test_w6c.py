"""Wave 6-C -- pathway() BFS N+1 batch optimization tests.

The BFS loop in pathway() originally calls get_neighbors() once per frontier
atom AND get_without_tracking() once per newly discovered neighbor.  For a
frontier of F atoms each with N neighbors, one BFS level = F + F*N SQL reads.

After the fix, each BFS level uses exactly 2 SQL calls:
    1. get_neighbors_batch(frontier_ids)
    2. get_batch_without_tracking(new_neighbor_ids)

Tests are written BEFORE the implementation (TDD red phase).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from memories.atoms import Atom
from memories.brain import Brain
from memories.synapses import Synapse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atom(
    atom_id: int,
    content: str = "test",
    atom_type: str = "fact",
    region: str = "cortex",
) -> Atom:
    """Build a minimal Atom for testing without hitting the database."""
    return Atom(
        id=atom_id,
        content=content,
        type=atom_type,
        region=region,
        tags=[],
        is_deleted=False,
        access_count=1,
        last_accessed_at="2026-01-01T00:00:00",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )


def _make_synapse(
    source_id: int,
    target_id: int,
    relationship: str = "related_to",
    strength: float = 0.5,
) -> Synapse:
    """Build a minimal Synapse for testing."""
    return Synapse(
        id=1,
        source_id=source_id,
        target_id=target_id,
        relationship=relationship,
        strength=strength,
        bidirectional=True,
        activated_count=0,
        last_activated_at=None,
        created_at="2026-01-01T00:00:00",
    )


def _make_brain_with_mocks() -> Brain:
    """Create a Brain instance with mocked internals for pathway() testing.

    Sets up a small graph:
        atom 1 --related_to--> atom 2
        atom 1 --related_to--> atom 3
        atom 2 --related_to--> atom 4
        atom 3 --related_to--> atom 5
    """
    brain = Brain.__new__(Brain)
    brain._initialized = True
    brain._atoms = AsyncMock()
    brain._synapses = AsyncMock()
    brain._storage = AsyncMock()
    brain._embeddings = AsyncMock()
    brain._retrieval = AsyncMock()
    brain._learning = AsyncMock()
    brain._consolidation = AsyncMock()
    brain._context = AsyncMock()

    # Atoms in the graph
    atoms = {
        1: _make_atom(1, "start atom"),
        2: _make_atom(2, "neighbor A"),
        3: _make_atom(3, "neighbor B"),
        4: _make_atom(4, "level-2 neighbor A"),
        5: _make_atom(5, "level-2 neighbor B"),
    }

    # Synapses
    syn_1_2 = _make_synapse(1, 2)
    syn_1_3 = _make_synapse(1, 3)
    syn_2_4 = _make_synapse(2, 4)
    syn_3_5 = _make_synapse(3, 5)

    # -- get_without_tracking: returns the start atom for initial lookup
    brain._atoms.get_without_tracking = AsyncMock(
        side_effect=lambda aid: atoms.get(aid)
    )

    # -- get_batch_without_tracking: returns dict of requested atoms
    async def _batch_get(ids: list[int]) -> dict[int, Atom]:
        return {aid: atoms[aid] for aid in ids if aid in atoms}

    brain._atoms.get_batch_without_tracking = AsyncMock(side_effect=_batch_get)

    # -- get_neighbors (per-atom, should NOT be called after fix)
    async def _get_neighbors(aid: int, min_strength: float = 0.0):
        graph = {
            1: [(2, syn_1_2), (3, syn_1_3)],
            2: [(4, syn_2_4)],
            3: [(5, syn_3_5)],
        }
        return graph.get(aid, [])

    brain._synapses.get_neighbors = AsyncMock(side_effect=_get_neighbors)

    # -- get_neighbors_batch: returns dict mapping atom_id -> neighbors list
    async def _get_neighbors_batch(
        ids: list[int], min_strength: float = 0.0
    ) -> dict[int, list[tuple[int, Synapse]]]:
        graph = {
            1: [(2, syn_1_2), (3, syn_1_3)],
            2: [(4, syn_2_4)],
            3: [(5, syn_3_5)],
        }
        return {aid: graph.get(aid, []) for aid in ids}

    brain._synapses.get_neighbors_batch = AsyncMock(
        side_effect=_get_neighbors_batch
    )

    return brain


# ---------------------------------------------------------------------------
# C1: pathway() must call get_neighbors_batch, NOT per-atom get_neighbors
# ---------------------------------------------------------------------------


class TestPathwayUsesGetNeighborsBatch:
    """After Wave 6-C, pathway() BFS must use get_neighbors_batch."""

    async def test_get_neighbors_batch_called_instead_of_get_neighbors(
        self,
    ) -> None:
        """pathway() must call get_neighbors_batch (batch), not get_neighbors (per-atom).

        Before fix: get_neighbors called F times per BFS level (F = frontier size).
        After fix: get_neighbors_batch called once per BFS level.
        """
        brain = _make_brain_with_mocks()

        result = await brain.pathway(atom_id=1, depth=2, min_strength=0.1)

        # get_neighbors_batch MUST have been called (once per BFS level)
        assert brain._synapses.get_neighbors_batch.call_count >= 1, (
            "get_neighbors_batch was never called. "
            "pathway() still uses per-atom get_neighbors."
        )

        # get_neighbors (per-atom) must NOT have been called
        # (the start-atom lookup uses get_without_tracking, not get_neighbors)
        brain._synapses.get_neighbors.assert_not_called(), (
            "get_neighbors (per-atom) was called. "
            "pathway() should use get_neighbors_batch instead."
        )

    async def test_get_neighbors_batch_called_once_per_bfs_level(
        self,
    ) -> None:
        """With depth=2, get_neighbors_batch should be called exactly 2 times
        (once per BFS level), not once per frontier atom."""
        brain = _make_brain_with_mocks()

        await brain.pathway(atom_id=1, depth=2, min_strength=0.1)

        # Depth=2 means 2 BFS levels, so exactly 2 batch calls
        assert brain._synapses.get_neighbors_batch.call_count == 2, (
            f"Expected 2 calls to get_neighbors_batch (one per BFS level), "
            f"got {brain._synapses.get_neighbors_batch.call_count}."
        )


# ---------------------------------------------------------------------------
# C2: pathway() must call get_batch_without_tracking, NOT per-atom
# ---------------------------------------------------------------------------


class TestPathwayUsesGetBatchWithoutTracking:
    """After Wave 6-C, pathway() must batch-fetch new neighbor atoms."""

    async def test_get_batch_without_tracking_called(self) -> None:
        """pathway() must call get_batch_without_tracking for neighbor atoms.

        Before fix: get_without_tracking called once per new neighbor.
        After fix: get_batch_without_tracking called once per BFS level.
        """
        brain = _make_brain_with_mocks()

        await brain.pathway(atom_id=1, depth=2, min_strength=0.1)

        # get_batch_without_tracking must have been called at least once
        assert brain._atoms.get_batch_without_tracking.call_count >= 1, (
            "get_batch_without_tracking was never called. "
            "pathway() still uses per-atom get_without_tracking for neighbors."
        )

    async def test_per_atom_get_without_tracking_not_called_for_neighbors(
        self,
    ) -> None:
        """get_without_tracking should only be called once (for the start atom),
        not for BFS-discovered neighbors."""
        brain = _make_brain_with_mocks()

        await brain.pathway(atom_id=1, depth=2, min_strength=0.1)

        # get_without_tracking is called once for the start atom (atom_id=1).
        # It must NOT be called additional times for BFS neighbors.
        assert brain._atoms.get_without_tracking.call_count == 1, (
            f"get_without_tracking called {brain._atoms.get_without_tracking.call_count} "
            f"times; expected exactly 1 (start atom only). "
            f"BFS neighbors should use get_batch_without_tracking."
        )


# ---------------------------------------------------------------------------
# C3: pathway() still returns correct results after batching
# ---------------------------------------------------------------------------


class TestPathwayCorrectness:
    """After batching, pathway() must return the same graph structure."""

    async def test_pathway_returns_all_nodes(self) -> None:
        """All reachable atoms must appear in the result nodes."""
        brain = _make_brain_with_mocks()

        result = await brain.pathway(atom_id=1, depth=2, min_strength=0.1)

        node_ids = {n["id"] for n in result["nodes"]}
        # With depth=2 from atom 1: should reach 1, 2, 3, 4, 5
        assert node_ids == {1, 2, 3, 4, 5}, (
            f"Expected nodes {{1, 2, 3, 4, 5}}, got {node_ids}"
        )

    async def test_pathway_returns_all_edges(self) -> None:
        """All traversed edges must appear in the result."""
        brain = _make_brain_with_mocks()

        result = await brain.pathway(atom_id=1, depth=2, min_strength=0.1)

        edge_pairs = {(e["source"], e["target"]) for e in result["edges"]}
        expected = {(1, 2), (1, 3), (2, 4), (3, 5)}
        assert edge_pairs == expected, (
            f"Expected edges {expected}, got {edge_pairs}"
        )

    async def test_pathway_returns_clusters_by_region(self) -> None:
        """Clusters should group atom IDs by region."""
        brain = _make_brain_with_mocks()

        result = await brain.pathway(atom_id=1, depth=2, min_strength=0.1)

        # All test atoms use region "cortex"
        assert "cortex" in result["clusters"]
        assert set(result["clusters"]["cortex"]) == {1, 2, 3, 4, 5}

    async def test_pathway_depth_1_limits_traversal(self) -> None:
        """With depth=1, only immediate neighbors should be returned."""
        brain = _make_brain_with_mocks()

        result = await brain.pathway(atom_id=1, depth=1, min_strength=0.1)

        node_ids = {n["id"] for n in result["nodes"]}
        # Depth=1: atom 1 + its direct neighbors 2, 3
        assert node_ids == {1, 2, 3}, (
            f"With depth=1, expected nodes {{1, 2, 3}}, got {node_ids}"
        )

    async def test_pathway_edge_deduplication(self) -> None:
        """Edges must be deduplicated by (min(src,tgt), max(src,tgt), rel)."""
        brain = _make_brain_with_mocks()

        result = await brain.pathway(atom_id=1, depth=2, min_strength=0.1)

        # Verify no duplicate edges
        edge_keys = [
            (min(e["source"], e["target"]), max(e["source"], e["target"]))
            for e in result["edges"]
        ]
        assert len(edge_keys) == len(set(edge_keys)), (
            f"Duplicate edges detected: {edge_keys}"
        )

    async def test_pathway_empty_frontier_stops_early(self) -> None:
        """If a BFS level produces no new neighbors, traversal stops."""
        brain = _make_brain_with_mocks()

        # depth=5, but graph only goes 2 levels deep
        result = await brain.pathway(atom_id=1, depth=5, min_strength=0.1)

        node_ids = {n["id"] for n in result["nodes"]}
        assert node_ids == {1, 2, 3, 4, 5}

        # Should have stopped early -- at most 3 batch calls
        # (level 0->1, level 1->2, level 2 has empty frontier -> break)
        assert brain._synapses.get_neighbors_batch.call_count <= 3


# ---------------------------------------------------------------------------
# C4: SQL call count regression test
# ---------------------------------------------------------------------------


class TestPathwaySQLCallCount:
    """Verify the fix reduces SQL calls from O(frontier*fanout) to O(depth)."""

    async def test_batch_calls_scale_with_depth_not_frontier(self) -> None:
        """Total batch calls should be proportional to depth, not frontier size.

        Before fix with depth=2, frontier=1->2->2:
            get_neighbors: 1 + 2 = 3 calls
            get_without_tracking: 2 + 2 = 4 calls
            Total: 7 SQL calls

        After fix with depth=2:
            get_neighbors_batch: 2 calls (one per level)
            get_batch_without_tracking: 2 calls (one per level)
            Total: 4 SQL calls (plus 1 for start atom)
        """
        brain = _make_brain_with_mocks()

        await brain.pathway(atom_id=1, depth=2, min_strength=0.1)

        batch_neighbor_calls = brain._synapses.get_neighbors_batch.call_count
        batch_atom_calls = brain._atoms.get_batch_without_tracking.call_count
        start_atom_call = 1  # get_without_tracking for atom_id=1

        total_sql = batch_neighbor_calls + batch_atom_calls + start_atom_call

        # Must be <= 2*depth + 1 (2 batch calls per level + 1 start atom)
        max_expected = 2 * 2 + 1  # depth=2 -> 5
        assert total_sql <= max_expected, (
            f"Total SQL calls = {total_sql} (batch_neighbors={batch_neighbor_calls}, "
            f"batch_atoms={batch_atom_calls}, start={start_atom_call}), "
            f"expected <= {max_expected} for depth=2."
        )
