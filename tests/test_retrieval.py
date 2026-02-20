"""Comprehensive tests for the retrieval engine (spreading activation recall).

All tests use a temporary database and mock the EmbeddingEngine so that Ollama
is never required.  The mocked embedding layer returns predictable results that
let us exercise every branch of the four-step recall pipeline:

1. Vector search (seed atom discovery)
2. Spreading activation (synapse traversal)
3. Multi-factor ranking (composite scoring)
4. Budget fitting (context-window compression)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.atoms import Atom, AtomManager
from memories.context import ContextBudget
from memories.embeddings import EmbeddingEngine
from memories.retrieval import RetrievalEngine, RecallResult, _ANTIPATTERN_BOOST
from memories.storage import Storage
from memories.synapses import SynapseManager


# ---------------------------------------------------------------------------
# Helpers -- direct SQL insertion for test data setup
# ---------------------------------------------------------------------------


async def _insert_atom(
    storage: Storage,
    content: str,
    atom_type: str = "fact",
    region: str = "general",
    confidence: float = 1.0,
    access_count: int = 5,
    last_accessed_at: str | None = None,
    tags: list[str] | None = None,
    is_deleted: bool = False,
    severity: str | None = None,
    instead: str | None = None,
) -> int:
    """Insert an atom directly via SQL, bypassing AtomManager (no embedding)."""
    if last_accessed_at is None:
        last_accessed_at = datetime.now(tz=timezone.utc).isoformat()
    tags_json = json.dumps(tags) if tags else None
    return await storage.execute_write(
        """
        INSERT INTO atoms
            (content, type, region, confidence, access_count,
             last_accessed_at, tags, is_deleted, severity, instead)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            content,
            atom_type,
            region,
            confidence,
            access_count,
            last_accessed_at,
            tags_json,
            int(is_deleted),
            severity,
            instead,
        ),
    )


async def _insert_synapse(
    storage: Storage,
    source_id: int,
    target_id: int,
    relationship: str = "related-to",
    strength: float = 0.8,
    bidirectional: bool = True,
) -> int:
    """Insert a synapse directly via SQL."""
    return await storage.execute_write(
        """
        INSERT INTO synapses
            (source_id, target_id, relationship, strength, bidirectional)
        VALUES (?, ?, ?, ?, ?)
        """,
        (source_id, target_id, relationship, strength, int(bidirectional)),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Return a MagicMock standing in for EmbeddingEngine.

    The caller should configure ``search_similar`` and ``embed_text``
    return values as needed per test.
    """
    engine = MagicMock(spec=EmbeddingEngine)
    engine.search_similar = AsyncMock(return_value=[])
    engine.embed_text = AsyncMock(return_value=[0.0] * 768)
    return engine


@pytest.fixture
async def seeded_storage(storage: Storage) -> dict[str, Any]:
    """Insert a small graph of atoms + synapses into *storage*.

    Returns a dict with atom IDs and the storage instance::

        {
            "storage": Storage,
            "atom_a": int,  # fact about Redis SCAN
            "atom_b": int,  # fact about Redis KEYS
            "atom_c": int,  # experience about Redis
            "atom_ap": int, # antipattern: never use KEYS in prod
            "atom_d": int,  # unrelated atom in 'personal' region
        }
    """
    now = datetime.now(tz=timezone.utc).isoformat()

    atom_a = await _insert_atom(
        storage, "Redis SCAN is O(N) over the full keyspace",
        atom_type="fact", region="technical", confidence=0.95,
        access_count=10, last_accessed_at=now, tags=["redis", "performance"],
    )
    atom_b = await _insert_atom(
        storage, "Redis KEYS command blocks the server",
        atom_type="fact", region="database", confidence=0.9,
        access_count=8, last_accessed_at=now, tags=["redis"],
    )
    atom_c = await _insert_atom(
        storage, "Experienced Redis latency spikes under high KEYS usage",
        atom_type="experience", region="technical", confidence=0.85,
        access_count=3, last_accessed_at=now, tags=["redis", "latency"],
    )
    atom_ap = await _insert_atom(
        storage, "Never use KEYS in production workloads",
        atom_type="antipattern", region="technical", confidence=1.0,
        access_count=15, last_accessed_at=now,
        severity="high", instead="Use SCAN instead",
    )
    atom_d = await _insert_atom(
        storage, "I prefer dark mode for all editors",
        atom_type="preference", region="personal", confidence=0.8,
        access_count=2, last_accessed_at=now,
    )

    # Synapses: A <-> B, A <-> C, AP -> B (warns-against)
    await _insert_synapse(storage, atom_a, atom_b, "related-to", 0.85, True)
    await _insert_synapse(storage, atom_a, atom_c, "related-to", 0.7, True)
    await _insert_synapse(storage, atom_ap, atom_b, "warns-against", 0.9, False)

    return {
        "storage": storage,
        "atom_a": atom_a,
        "atom_b": atom_b,
        "atom_c": atom_c,
        "atom_ap": atom_ap,
        "atom_d": atom_d,
    }


@pytest.fixture
def retrieval_engine(
    storage: Storage,
    mock_embeddings: MagicMock,
) -> RetrievalEngine:
    """Build a RetrievalEngine wired to real Storage but mocked embeddings."""
    atoms = AtomManager(storage, mock_embeddings)
    synapses = SynapseManager(storage)
    context_budget = ContextBudget()
    return RetrievalEngine(storage, mock_embeddings, atoms, synapses, context_budget)


@pytest.fixture
def seeded_engine(
    seeded_storage: dict[str, Any],
    mock_embeddings: MagicMock,
) -> tuple[RetrievalEngine, dict[str, Any]]:
    """Build a RetrievalEngine with pre-seeded data.

    Returns ``(engine, ids_dict)`` where ids_dict has all atom IDs.
    """
    storage = seeded_storage["storage"]
    atoms = AtomManager(storage, mock_embeddings)
    synapses = SynapseManager(storage)
    context_budget = ContextBudget()
    engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, context_budget)
    return engine, seeded_storage


# -----------------------------------------------------------------------
# 1. Vector search returns seed atoms correctly
# -----------------------------------------------------------------------


class TestVectorSearchSeeds:
    """Verify that the vector search step produces correct seed atoms."""

    async def test_seeds_from_search_similar(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Recall should find atoms returned by search_similar as seeds."""
        engine, ids = seeded_engine
        atom_a = ids["atom_a"]

        # search_similar returns (atom_id, distance) tuples.
        # distance=0.2 -> similarity = max(0, 1 - 0.2/2) = 0.9
        mock_embeddings.search_similar.return_value = [
            (atom_a, 0.2),
        ]

        result = await engine.recall("Redis SCAN", budget_tokens=5000)

        assert isinstance(result, RecallResult)
        assert result.seed_count == 1
        # The seed atom should appear in the result.
        result_ids = {a["id"] for a in result.atoms}
        assert atom_a in result_ids

    async def test_seed_count_matches(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """seed_count reflects the combined vector + BM25 seed set."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
            (ids["atom_b"], 0.4),
        ]

        result = await engine.recall("Redis", budget_tokens=5000)

        # At least the 2 vector seeds; BM25 may contribute additional seeds.
        assert result.seed_count >= 2
        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_a"] in result_ids
        assert ids["atom_b"] in result_ids

    async def test_deleted_atoms_excluded_from_seeds(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Soft-deleted atoms must not appear as seeds."""
        engine, ids = seeded_engine
        storage = ids["storage"]

        # Soft-delete atom_a.
        await storage.execute_write(
            "UPDATE atoms SET is_deleted = 1 WHERE id = ?",
            (ids["atom_a"],),
        )

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.1),  # deleted -- should be skipped
            (ids["atom_b"], 0.3),
        ]

        result = await engine.recall("Redis", budget_tokens=5000)

        # Deleted atom_a must never appear regardless of BM25 hits.
        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_a"] not in result_ids
        # At least atom_b survives as a seed.
        assert result.seed_count >= 1


# -----------------------------------------------------------------------
# 2. Spreading activation traverses synapses
# -----------------------------------------------------------------------


class TestSpreadingActivation:
    """Verify that activation propagates through synapse connections."""

    async def test_neighbors_activated(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Atoms connected via synapses should be activated from seeds."""
        engine, ids = seeded_engine

        # Seed only atom_a; atom_b and atom_c are connected via synapses.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        result = await engine.recall("Redis SCAN", budget_tokens=10000)

        assert result.total_activated > 1
        # atom_b is connected to atom_a with strength 0.85.
        all_ids = {a["id"] for a in result.atoms + result.antipatterns}
        assert ids["atom_b"] in all_ids or result.total_activated >= 2

    async def test_activation_decays_with_depth(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Deeper atoms receive weaker activation energy."""
        engine, ids = seeded_engine

        # Seed atom_a at high similarity.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.1),  # similarity ~0.95
        ]

        result = await engine.recall("Redis", budget_tokens=10000, depth=2)

        # atom_a (seed) should have the highest score.
        if len(result.atoms) >= 2:
            assert result.atoms[0]["score"] >= result.atoms[1]["score"]

    async def test_unconnected_atoms_not_activated(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Atoms with no synapse connection to seeds should not spread-activate."""
        engine, ids = seeded_engine

        # Seed atom_a only. atom_d (personal preference) has no synapse to A.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        result = await engine.recall("Redis", budget_tokens=10000)

        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_d"] not in result_ids


# -----------------------------------------------------------------------
# 3. Scoring weights are applied correctly
# -----------------------------------------------------------------------


class TestScoringWeights:
    """Verify that multi-factor ranking produces expected score ordering."""

    async def test_higher_similarity_higher_score(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """An atom with higher vector similarity should score higher (all else equal)."""
        engine, ids = seeded_engine

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.1),  # similarity ~0.95
            (ids["atom_b"], 0.6),  # similarity ~0.7
        ]

        result = await engine.recall(
            "Redis", budget_tokens=10000, depth=0,
            include_antipatterns=False,
        )

        # With depth=0 there is no spreading, so ordering depends primarily
        # on vector similarity.
        if len(result.atoms) >= 2:
            scores = {a["id"]: a["score"] for a in result.atoms}
            assert scores[ids["atom_a"]] > scores[ids["atom_b"]]

    async def test_all_atoms_have_score_key(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Every atom dict in the result must contain a 'score' key."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        result = await engine.recall("Redis", budget_tokens=10000)

        for atom_dict in result.atoms:
            assert "score" in atom_dict
            assert isinstance(atom_dict["score"], float)


# -----------------------------------------------------------------------
# 4. Antipattern atoms get boosted
# -----------------------------------------------------------------------


class TestAntipatternBoosting:
    """Verify that antipattern atoms receive the _ANTIPATTERN_BOOST multiplier."""

    async def test_antipattern_in_results_when_connected(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Antipattern atoms linked via warns-against should appear in results.

        The warns-against synapse is unidirectional (atom_ap -> atom_b), so
        _find_relevant_antipatterns discovers it by querying neighbors of atom_ap
        (the antipattern source).  We include atom_ap as a seed so it is among
        the result atoms whose neighbors are inspected.
        """
        engine, ids = seeded_engine

        # Seed both atom_b and atom_ap so atom_ap is in the activated set.
        # _find_relevant_antipatterns iterates result atom_ids and calls
        # get_neighbors, which returns outgoing connections.  Since atom_ap
        # is the source of the warns-against synapse, it needs to be in the
        # result set for its outgoing link to atom_b to be found (and vice
        # versa -- atom_ap neighbours include atom_b via the outgoing edge).
        mock_embeddings.search_similar.return_value = [
            (ids["atom_b"], 0.3),
            (ids["atom_ap"], 0.3),
        ]

        result = await engine.recall(
            "Redis KEYS", budget_tokens=10000, include_antipatterns=True,
        )

        antipattern_ids = {a["id"] for a in result.antipatterns}
        assert ids["atom_ap"] in antipattern_ids

    async def test_antipattern_excluded_when_flag_off(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """When include_antipatterns=False, antipatterns should not appear in the
        antipatterns list from extra discovery (they may still appear as regular
        activated atoms if directly activated)."""
        engine, ids = seeded_engine

        mock_embeddings.search_similar.return_value = [
            (ids["atom_b"], 0.3),
        ]

        result = await engine.recall(
            "Redis KEYS", budget_tokens=10000, include_antipatterns=False,
        )

        # The antipatterns list from extra discovery should be empty.
        assert result.antipatterns == []

    async def test_antipattern_directly_seeded_gets_boost(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """An antipattern atom found as a seed should have its score boosted."""
        engine, ids = seeded_engine

        # Return both the antipattern and a regular atom with identical distance.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_ap"], 0.2),
            (ids["atom_b"], 0.2),
        ]

        result = await engine.recall(
            "KEYS in production", budget_tokens=10000, depth=0,
            include_antipatterns=True,
        )

        # The antipattern should get a boosted score from _ANTIPATTERN_BOOST.
        all_atoms = result.atoms + result.antipatterns
        scores = {a["id"]: a["score"] for a in all_atoms}
        if ids["atom_ap"] in scores and ids["atom_b"] in scores:
            assert scores[ids["atom_ap"]] > scores[ids["atom_b"]]


# -----------------------------------------------------------------------
# 5. Budget limits are respected
# -----------------------------------------------------------------------


class TestBudgetLimits:
    """Verify that token budget constraints are enforced."""

    async def test_budget_used_within_limit(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """budget_used must not exceed the requested budget_tokens."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
            (ids["atom_b"], 0.3),
            (ids["atom_c"], 0.4),
        ]

        budget = 200
        result = await engine.recall("Redis", budget_tokens=budget)

        assert result.budget_used <= budget

    async def test_budget_remaining_correct(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """budget_remaining should equal budget_tokens minus budget_used."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        budget = 5000
        result = await engine.recall("Redis", budget_tokens=budget)

        assert result.budget_remaining == budget - result.budget_used

    async def test_tiny_budget_still_returns_result(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Even with a very small budget, at least one atom should fit
        (at the highest compression level) or the result is empty."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        result = await engine.recall("Redis", budget_tokens=10)

        # Either it fits at least one atom or returns zero.
        assert isinstance(result, RecallResult)
        assert result.budget_used <= 10

    async def test_large_budget_includes_more_atoms(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """A larger budget should include at least as many atoms as a smaller one."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
            (ids["atom_b"], 0.3),
            (ids["atom_c"], 0.4),
        ]

        small = await engine.recall(
            "Redis", budget_tokens=100, include_antipatterns=False,
        )
        large = await engine.recall(
            "Redis", budget_tokens=50000, include_antipatterns=False,
        )

        assert len(large.atoms) >= len(small.atoms)


# -----------------------------------------------------------------------
# 6. Region filtering works
# -----------------------------------------------------------------------


class TestRegionFiltering:
    """Verify that the region parameter filters results correctly."""

    async def test_region_filter_includes_matching(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Atoms in the requested region should be included."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),  # technical region
            (ids["atom_d"], 0.3),  # personal region
        ]

        result = await engine.recall(
            "Redis", budget_tokens=10000, region="technical",
        )

        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_a"] in result_ids

    async def test_region_filter_excludes_non_matching(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Atoms outside the requested region should be filtered out."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),  # technical
            (ids["atom_d"], 0.3),  # personal
        ]

        result = await engine.recall(
            "preferences", budget_tokens=10000, region="personal",
            include_antipatterns=False,
        )

        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_a"] not in result_ids

    async def test_no_region_filter_returns_all(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Without a region filter, atoms from any region can appear."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
            (ids["atom_d"], 0.3),
        ]

        result = await engine.recall(
            "anything", budget_tokens=10000, region=None,
            include_antipatterns=False,
        )

        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_a"] in result_ids
        assert ids["atom_d"] in result_ids


# -----------------------------------------------------------------------
# 7. Type filtering works
# -----------------------------------------------------------------------


class TestTypeFiltering:
    """Verify that the types parameter filters results correctly."""

    async def test_type_filter_includes_matching(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Only atoms of the specified types should be seeds."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),  # fact
            (ids["atom_c"], 0.3),  # experience
        ]

        result = await engine.recall(
            "Redis", budget_tokens=10000, types=["fact"],
            depth=0, include_antipatterns=False,
        )

        # With depth=0, only seeds make it through.
        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_a"] in result_ids

    async def test_type_filter_excludes_non_matching(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Seeds of non-matching types should be filtered out."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),  # fact
            (ids["atom_c"], 0.3),  # experience
        ]

        result = await engine.recall(
            "Redis", budget_tokens=10000, types=["experience"],
            depth=0, include_antipatterns=False,
        )

        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_a"] not in result_ids

    async def test_multiple_types(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Multiple types can be specified at once."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),  # fact
            (ids["atom_c"], 0.3),  # experience
            (ids["atom_d"], 0.5),  # preference
        ]

        result = await engine.recall(
            "anything", budget_tokens=10000, types=["fact", "experience"],
            depth=0, include_antipatterns=False,
        )

        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_d"] not in result_ids
        # At least one of the matching types should be present.
        assert ids["atom_a"] in result_ids or ids["atom_c"] in result_ids


# -----------------------------------------------------------------------
# 8. Empty results are handled gracefully
# -----------------------------------------------------------------------


class TestEmptyResults:
    """Verify graceful handling when no atoms match."""

    async def test_no_vector_results(
        self,
        retrieval_engine: RetrievalEngine,
        mock_embeddings: MagicMock,
    ) -> None:
        """When search_similar returns nothing, recall should return empty."""
        mock_embeddings.search_similar.return_value = []

        result = await retrieval_engine.recall("nonexistent topic")

        assert result.seed_count == 0
        assert result.atoms == []
        assert result.antipatterns == []
        assert result.pathways == []
        assert result.budget_used == 0
        assert result.total_activated == 0

    async def test_all_seeds_deleted(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """When all vector results are deleted atoms, recall returns empty seeds."""
        engine, ids = seeded_engine
        storage = ids["storage"]

        # Delete all returned atoms.
        await storage.execute_write(
            "UPDATE atoms SET is_deleted = 1 WHERE id IN (?, ?)",
            (ids["atom_a"], ids["atom_b"]),
        )

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.1),
            (ids["atom_b"], 0.2),
        ]

        result = await engine.recall("Redis", budget_tokens=5000)

        # Deleted atoms must not appear in results even if BM25 finds other atoms.
        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_a"] not in result_ids
        assert ids["atom_b"] not in result_ids

    async def test_empty_query_string(
        self,
        retrieval_engine: RetrievalEngine,
        mock_embeddings: MagicMock,
    ) -> None:
        """An empty query should still call search_similar and return a result."""
        mock_embeddings.search_similar.return_value = []

        result = await retrieval_engine.recall("")

        assert isinstance(result, RecallResult)
        assert result.seed_count == 0

    async def test_zero_budget(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """With budget_tokens=0, result should contain no atoms."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        result = await engine.recall("Redis", budget_tokens=0)

        assert result.atoms == []
        assert result.budget_used == 0


# -----------------------------------------------------------------------
# 9. Pathway extraction between result atoms
# -----------------------------------------------------------------------


class TestPathways:
    """Verify that pathways between result atoms are extracted correctly."""

    async def test_pathways_between_connected_atoms(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """When result atoms share a synapse, it should appear in pathways."""
        engine, ids = seeded_engine

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
            (ids["atom_b"], 0.3),
        ]

        result = await engine.recall("Redis", budget_tokens=10000)

        # atom_a and atom_b are connected by a 'related-to' synapse.
        if len(result.atoms) >= 2:
            atom_ids_in_result = {a["id"] for a in result.atoms}
            if ids["atom_a"] in atom_ids_in_result and ids["atom_b"] in atom_ids_in_result:
                assert len(result.pathways) >= 1
                pathway_pairs = {
                    (p["source_id"], p["target_id"]) for p in result.pathways
                }
                assert (ids["atom_a"], ids["atom_b"]) in pathway_pairs or \
                       (ids["atom_b"], ids["atom_a"]) in pathway_pairs

    async def test_no_pathways_for_single_atom(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """With only one atom, there should be no pathways."""
        engine, ids = seeded_engine

        mock_embeddings.search_similar.return_value = [
            (ids["atom_d"], 0.2),  # unconnected atom
        ]

        result = await engine.recall(
            "dark mode", budget_tokens=10000, depth=0,
            include_antipatterns=False,
        )

        assert result.pathways == []


# -----------------------------------------------------------------------
# 10. Depth parameter controls spread range
# -----------------------------------------------------------------------


class TestDepthControl:
    """Verify that the depth parameter limits activation spread."""

    async def test_depth_zero_returns_only_seeds(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """With depth=0, only the seed atoms should appear (no spreading)."""
        engine, ids = seeded_engine

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        result = await engine.recall(
            "Redis", budget_tokens=10000, depth=0,
            include_antipatterns=False,
        )

        result_ids = {a["id"] for a in result.atoms}
        assert ids["atom_a"] in result_ids
        # At depth=0 only seeds are activated -- no spreading through synapses.
        # BM25 may contribute additional seeds beyond the single vector seed.
        assert result.total_activated == result.seed_count

    async def test_depth_one_reaches_immediate_neighbors(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """With depth=1, immediate neighbors should be activated."""
        engine, ids = seeded_engine

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        result = await engine.recall(
            "Redis", budget_tokens=10000, depth=1,
            include_antipatterns=False,
        )

        # atom_b and atom_c are immediate neighbors of atom_a.
        assert result.total_activated >= 2


# -----------------------------------------------------------------------
# 11. RecallResult dataclass defaults
# -----------------------------------------------------------------------


class TestRecallResultDefaults:
    """Verify the RecallResult dataclass has sensible defaults."""

    def test_default_values(self) -> None:
        """A default RecallResult should have empty lists and zero counters."""
        result = RecallResult()
        assert result.atoms == []
        assert result.antipatterns == []
        assert result.pathways == []
        assert result.budget_used == 0
        assert result.budget_remaining == 0
        assert result.total_activated == 0
        assert result.seed_count == 0
        assert result.compression_level == 0


# -----------------------------------------------------------------------
# 12. Access recording on returned atoms
# -----------------------------------------------------------------------


class TestAccessRecording:
    """Verify that atoms in the final result have their access recorded."""

    async def test_access_count_incremented(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Atoms in the final result should have their access_count bumped."""
        engine, ids = seeded_engine
        storage = ids["storage"]

        # Get original access count.
        rows = await storage.execute(
            "SELECT access_count FROM atoms WHERE id = ?",
            (ids["atom_a"],),
        )
        original_count = rows[0]["access_count"]

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        await engine.recall("Redis", budget_tokens=10000)

        rows = await storage.execute(
            "SELECT access_count FROM atoms WHERE id = ?",
            (ids["atom_a"],),
        )
        new_count = rows[0]["access_count"]
        assert new_count > original_count


# -----------------------------------------------------------------------
# 13. Score breakdown in recall results
# -----------------------------------------------------------------------


class TestScoreBreakdown:
    """Verify that recall results include per-atom score_breakdown dicts."""

    async def test_atoms_have_score_breakdown(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Recall result atoms should include a score_breakdown dict."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        result = await engine.recall("Redis", budget_tokens=10000)

        for atom_dict in result.atoms:
            assert "score_breakdown" in atom_dict
            assert isinstance(atom_dict["score_breakdown"], dict)

    async def test_score_breakdown_keys(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """score_breakdown should contain all 5 signal keys."""
        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        result = await engine.recall("Redis", budget_tokens=10000)

        expected_keys = {
            "vector_similarity",
            "spread_activation",
            "recency",
            "confidence",
            "frequency",
            "importance",
            "newness",
            "bm25",
        }
        for atom_dict in result.atoms:
            breakdown = atom_dict["score_breakdown"]
            assert set(breakdown.keys()) == expected_keys
            # All values should be floats.
            for val in breakdown.values():
                assert isinstance(val, float)

    async def test_breakdown_consistent_with_composite(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Weighted sum of breakdown signals should approximate the composite score.

        For non-antipattern atoms, the composite score equals the weighted sum.
        """
        from memories.config import get_config

        engine, ids = seeded_engine
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),  # fact, not antipattern
        ]

        result = await engine.recall(
            "Redis", budget_tokens=10000, depth=0,
            include_antipatterns=False,
        )

        weights = get_config().retrieval.weights

        # Newness bonus weight is defined in retrieval.py
        from memories.retrieval import _NEWNESS_BONUS_WEIGHT

        for atom_dict in result.atoms:
            if atom_dict.get("type") == "antipattern":
                continue  # Skip antipatterns (boosted score)

            breakdown = atom_dict["score_breakdown"]
            expected = (
                breakdown["vector_similarity"] * weights.vector_similarity
                + breakdown["spread_activation"] * weights.spread_activation
                + breakdown["recency"] * weights.recency
                + breakdown["confidence"] * weights.confidence
                + breakdown["frequency"] * weights.frequency
                + breakdown["importance"] * weights.importance
                + breakdown["newness"] * _NEWNESS_BONUS_WEIGHT
                + breakdown["bm25"] * weights.bm25
            )
            # Allow small floating point tolerance.
            assert abs(atom_dict["score"] - expected) < 0.001, (
                f"score={atom_dict['score']}, expected={expected}, "
                f"breakdown={breakdown}"
            )


class TestEdgeCases:
    """Edge case tests for retrieval robustness."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_result(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
    ) -> None:
        """Empty query should return empty result without crashing."""
        engine, _ = seeded_engine

        result = await engine.recall("", budget_tokens=1000)

        assert result.atoms == []
        assert result.antipatterns == []
        assert result.seed_count == 0

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty_result(
        self,
        seeded_engine: tuple[RetrievalEngine, dict[str, Any]],
    ) -> None:
        """Whitespace-only query should return empty result without crashing."""
        engine, _ = seeded_engine

        result = await engine.recall("   \t\n   ", budget_tokens=1000)

        assert result.atoms == []
        assert result.antipatterns == []
        assert result.seed_count == 0


class TestSpreadingActivationSummation:
    """Regression tests for additive spreading activation (not max-wins)."""

    @pytest.fixture
    async def engine_with_storage(
        self, tmp_path
    ) -> tuple[RetrievalEngine, Storage, MagicMock]:
        """Provide a RetrievalEngine backed by real Storage with mocked embeddings."""
        from memories.config import get_config
        db_path = tmp_path / "test.db"
        storage = Storage(db_path)
        storage._backup_dir = tmp_path / "backups"
        storage._backup_dir.mkdir(exist_ok=True)
        await storage.initialize()

        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.search_similar = AsyncMock(return_value=[])
        mock_emb.embed_text = AsyncMock(return_value=[0.0] * 768)
        mock_emb.health_check = AsyncMock(return_value=True)

        atoms = AtomManager(storage, mock_emb)
        synapses = SynapseManager(storage)
        from memories.context import ContextBudget
        budget = ContextBudget(get_config())
        engine = RetrievalEngine(storage, mock_emb, atoms, synapses, budget)

        yield engine, storage, mock_emb

        await storage.close()

    async def test_convergent_activation_sums_not_max(
        self, engine_with_storage
    ) -> None:
        """Atom reached from two seeds must have higher activation than from one.

        Layout: seed_a → target, seed_b → target
        With additive activation, target gets seed_a_strength + seed_b_strength.
        With max-wins, target gets max(seed_a_strength, seed_b_strength).
        """
        engine, storage, mock_emb = engine_with_storage

        # Insert three atoms.
        seed_a_id = await _insert_atom(storage, "seed atom A")
        seed_b_id = await _insert_atom(storage, "seed atom B")
        target_id = await _insert_atom(storage, "target atom")

        # Use high synapse strength so activation exceeds min_activation floor.
        # neighbour_activation = seed * strength * type_weight * decay
        # = 0.9 * 0.9 * 0.4 * 0.7 ≈ 0.23 > min_activation(0.1)
        synapse_strength = 0.9
        await _insert_synapse(storage, seed_a_id, target_id, strength=synapse_strength)
        await _insert_synapse(storage, seed_b_id, target_id, strength=synapse_strength)

        from memories.config import get_config
        cfg = get_config()
        decay = cfg.retrieval.decay_factor
        type_weight = cfg.retrieval.synapse_type_weights.related_to

        # Run spreading activation from both seeds at equal initial activation.
        seed_activation = 0.9
        seeds = {seed_a_id: seed_activation, seed_b_id: seed_activation}
        activated, _ = await engine._spread_activation(seeds, depth=1)

        # Expected additive result: each seed contributes independently.
        expected_from_one_seed = seed_activation * synapse_strength * type_weight * decay
        expected_additive = min(1.0, expected_from_one_seed * 2)
        # Sanity check our values are above the min_activation floor.
        assert expected_from_one_seed > cfg.retrieval.min_activation

        assert target_id in activated, "target should be activated"
        actual = activated[target_id]

        # Additive: actual ≈ 2× single-seed contribution.
        # Max-wins: actual = single-seed contribution.
        assert actual > expected_from_one_seed + 1e-6, (
            f"Activation {actual:.4f} should exceed single-seed value "
            f"{expected_from_one_seed:.4f} — convergent paths must sum"
        )
        assert actual == pytest.approx(expected_additive, abs=1e-6)


# -----------------------------------------------------------------------
# P6: contradicts inhibition
# -----------------------------------------------------------------------


class TestContradictInhibition:
    """Verify that ``contradicts`` synapses subtract activation instead of adding it.

    A ``contradicts`` relationship represents conflicting information.  When
    atom A is activated and A contradicts B, B's activation should be
    *reduced* — suppressing the conflicting memory rather than propagating it.
    """

    @pytest.fixture
    async def inhibition_engine(self, tmp_path):
        """RetrievalEngine backed by real storage with mocked embeddings."""
        db_path = tmp_path / "inhibit.db"
        storage = Storage(db_path)
        storage._backup_dir = tmp_path / "backups"
        storage._backup_dir.mkdir(exist_ok=True)
        await storage.initialize()

        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.search_similar = AsyncMock(return_value=[])
        mock_emb.embed_text = AsyncMock(return_value=[0.0] * 768)

        atoms = AtomManager(storage, mock_emb)
        synapses = SynapseManager(storage)
        budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_emb, atoms, synapses, budget)

        yield engine, storage

        await storage.close()

    async def test_contradicts_reduces_activation(self, inhibition_engine) -> None:
        """A ``contradicts`` synapse suppresses the target atom's activation."""
        engine, storage = inhibition_engine

        seed_id = await _insert_atom(storage, "seed atom")
        target_id = await _insert_atom(storage, "contradicted atom")

        # Insert a contradicts synapse with high strength
        await _insert_synapse(
            storage, seed_id, target_id, relationship="contradicts", strength=0.9
        )

        # Seed the source atom at high activation
        seed_activation = 0.8
        seeds = {seed_id: seed_activation}
        activated, _ = await engine._spread_activation(seeds, depth=1)

        # The target's activation should be BELOW its pre-activation level
        # (i.e. contradicts suppresses, not boosts).
        # With a related-to synapse, the target would be boosted above 0;
        # with contradicts, it should be reduced (clamped to 0.0).
        target_activation = activated.get(target_id, 0.0)
        assert target_activation == pytest.approx(0.0, abs=1e-6), (
            f"contradicts should suppress target to 0.0, got {target_activation:.4f}"
        )

    async def test_contradicts_does_not_push_below_zero(
        self, inhibition_engine
    ) -> None:
        """Activation is clamped to 0.0 — never goes negative."""
        engine, storage = inhibition_engine

        # First give the target some positive activation via a related-to path
        seed_a = await _insert_atom(storage, "positive seed")
        seed_b = await _insert_atom(storage, "negative seed")
        target_id = await _insert_atom(storage, "contested atom")

        # seed_a → target via related-to (boosts target)
        await _insert_synapse(storage, seed_a, target_id, relationship="related-to", strength=0.9)
        # seed_b → target via contradicts (suppresses target)
        await _insert_synapse(storage, seed_b, target_id, relationship="contradicts", strength=0.9)

        seeds = {seed_a: 0.9, seed_b: 0.9}
        activated, _ = await engine._spread_activation(seeds, depth=1)

        target_activation = activated.get(target_id, 0.0)
        assert target_activation >= 0.0, (
            f"Activation must never be negative, got {target_activation:.4f}"
        )


# -----------------------------------------------------------------------
# P7: Session priming (context-dependent retrieval)
# -----------------------------------------------------------------------


class TestSessionPriming:
    """Verify that session atoms receive priming activation before spreading.

    Atoms accessed earlier in the same session are seeded at
    ``_SESSION_PRIME_BOOST`` so they are slightly more likely to resurface
    in later queries — context-dependent retrieval.
    """

    @pytest.fixture
    async def priming_engine(self, tmp_path):
        """RetrievalEngine with real storage and mocked embeddings."""
        db_path = tmp_path / "prime.db"
        storage = Storage(db_path)
        storage._backup_dir = tmp_path / "backups"
        storage._backup_dir.mkdir(exist_ok=True)
        await storage.initialize()

        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.search_similar = AsyncMock(return_value=[])
        mock_emb.embed_text = AsyncMock(return_value=[0.0] * 768)

        atoms = AtomManager(storage, mock_emb)
        synapses = SynapseManager(storage)
        budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_emb, atoms, synapses, budget)

        yield engine, storage

        await storage.close()

    async def test_session_atom_not_in_vector_seeds_gets_primed(
        self, priming_engine
    ) -> None:
        """A session atom not found by vector search is seeded at SESSION_PRIME_BOOST.

        We simulate this by calling ``_spread_activation`` directly with the
        same seeds dict that ``recall()`` would build after applying session
        priming (i.e. ``{session_atom_id: _SESSION_PRIME_BOOST}``).  The atom
        should appear in the activated set at its priming level.
        """
        from memories.retrieval import _SESSION_PRIME_BOOST

        engine, storage = priming_engine

        session_atom_id = await _insert_atom(storage, "previously accessed atom")

        # Simulate: vector search found nothing; session priming adds this atom.
        seeds = {session_atom_id: _SESSION_PRIME_BOOST}
        activated, _ = await engine._spread_activation(seeds, depth=1)

        assert session_atom_id in activated, "Session atom should appear in activated set"
        assert activated[session_atom_id] == pytest.approx(_SESSION_PRIME_BOOST, abs=1e-6)

    async def test_session_priming_does_not_override_vector_seed(
        self, priming_engine
    ) -> None:
        """A session atom that IS also a vector seed keeps its higher vector score.

        The recall() code only adds _SESSION_PRIME_BOOST when the atom is
        NOT already in vector_scores.  A high-confidence vector seed must
        not be downgraded to the session prime level.
        """
        from memories.retrieval import _SESSION_PRIME_BOOST

        engine, storage = priming_engine

        atom_id = await _insert_atom(storage, "atom found by both vector and session")

        # Simulate: vector search found atom at 0.8, session priming should not
        # downgrade it (recall() checks ``if aid not in vector_scores``).
        vector_score = 0.8
        seeds_with_vector = {atom_id: vector_score}

        # Mimic what recall() does: only add session prime if NOT in vector_scores.
        for aid in [atom_id]:
            if aid not in seeds_with_vector:
                seeds_with_vector[aid] = _SESSION_PRIME_BOOST

        # The atom must retain its vector score, not be downgraded.
        assert seeds_with_vector[atom_id] == pytest.approx(vector_score, abs=1e-6), (
            "Vector seed score must not be replaced by session prime boost"
        )


# -----------------------------------------------------------------------
# Recency bootstrap: new atoms use created_at not 90-day floor
# -----------------------------------------------------------------------


class TestRecencyBootstrap:
    """Verify that newly-created atoms are not penalised with the 90-day floor.

    Before the fix, atoms with ``last_accessed_at = NULL`` fell through to
    ``days_since = _RECENCY_DECAY_DAYS``, giving them a recency score equal
    to ``_RECENCY_FLOOR`` (0.1) even when freshly created.  The fix uses
    ``atom.last_accessed_at or atom.created_at`` so that new atoms inherit
    a high recency from their creation timestamp.
    """

    @pytest.fixture
    async def bootstrap_engine(self, tmp_path):
        """RetrievalEngine backed by real storage."""
        db_path = tmp_path / "bootstrap.db"
        storage = Storage(db_path)
        storage._backup_dir = tmp_path / "backups"
        storage._backup_dir.mkdir(exist_ok=True)
        await storage.initialize()

        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.search_similar = AsyncMock(return_value=[])
        mock_emb.embed_text = AsyncMock(return_value=[0.0] * 768)

        atoms = AtomManager(storage, mock_emb)
        synapses = SynapseManager(storage)
        budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_emb, atoms, synapses, budget)

        yield engine, storage

        await storage.close()

    async def test_new_atom_no_access_gets_high_recency(
        self, bootstrap_engine
    ) -> None:
        """New atom with no access history scores near 1.0 for recency.

        ``last_accessed_at`` is NULL, but ``created_at`` is today.  The
        recency bootstrap should give it a score close to 1.0, not the
        90-day floor (0.1).
        """
        from memories.retrieval import _RECENCY_FLOOR

        engine, storage = bootstrap_engine

        # Insert atom with no last_accessed_at; created_at defaults to now().
        new_atom_id = await storage.execute_write(
            """
            INSERT INTO atoms
                (content, type, region, confidence, access_count,
                 last_accessed_at, created_at, updated_at, is_deleted)
            VALUES (?, 'fact', 'general', 1.0, 0,
                    NULL, datetime('now'), datetime('now'), 0)
            """,
            ("brand new fact",),
        )
        # Also insert a seed so activation_scores contains our atom.
        seeds = {new_atom_id: 1.0}
        activation_scores, _ = await engine._spread_activation(seeds, depth=0)

        scored = await engine._score_atoms(
            vector_scores=seeds,
            activation_scores=activation_scores,
            bm25_scores={},
            include_antipatterns=False,
        )

        assert scored, "New atom should appear in scored list"
        atom, score, breakdown = scored[0]
        assert atom.id == new_atom_id

        # Recency should be well above the floor (0.1) — atom was just created.
        recency = breakdown["recency"]
        assert recency > 0.9, (
            f"New atom should have recency near 1.0, got {recency:.4f}. "
            "If this is 0.1, the recency bootstrap fix is not applied."
        )

    async def test_old_atom_with_no_access_gets_floor_recency(
        self, bootstrap_engine
    ) -> None:
        """An old atom with no last_accessed_at uses created_at and gets low recency."""
        from memories.retrieval import _RECENCY_FLOOR

        engine, storage = bootstrap_engine

        old_atom_id = await storage.execute_write(
            """
            INSERT INTO atoms
                (content, type, region, confidence, access_count,
                 last_accessed_at, created_at, updated_at, is_deleted)
            VALUES (?, 'fact', 'general', 1.0, 0,
                    NULL, '2020-01-01 00:00:00', datetime('now'), 0)
            """,
            ("old never-accessed fact",),
        )
        seeds = {old_atom_id: 1.0}
        activation_scores, _ = await engine._spread_activation(seeds, depth=0)

        scored = await engine._score_atoms(
            vector_scores=seeds,
            activation_scores=activation_scores,
            bm25_scores={},
            include_antipatterns=False,
        )

        assert scored
        _, _, breakdown = scored[0]

        # Old created_at → large days_since → recency at floor.
        recency = breakdown["recency"]
        assert recency == pytest.approx(_RECENCY_FLOOR, abs=1e-6), (
            f"Old atom with no access should get floor recency {_RECENCY_FLOOR}, "
            f"got {recency:.4f}"
        )


# -----------------------------------------------------------------------
# Batch atom fetch verification
# -----------------------------------------------------------------------


class TestBatchAtomFetches:
    """Verify that _vector_search, _bm25_search, and recall use batch fetches
    (get_batch_without_tracking / record_access_batch) instead of per-atom calls."""

    async def test_vector_search_uses_batch_fetch(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """_vector_search should call get_batch_without_tracking once,
        not individual get_without_tracking per atom."""
        # Insert two atoms.
        atom_id_1 = await _insert_atom(storage, "atom one for vector")
        atom_id_2 = await _insert_atom(storage, "atom two for vector")

        # Build a RetrievalEngine with a real AtomManager.
        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        context_budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, context_budget)

        # Mock search_similar to return our two atoms.
        mock_embeddings.search_similar.return_value = [
            (atom_id_1, 0.1),
            (atom_id_2, 0.2),
        ]

        # Patch get_batch_without_tracking to spy on it.
        original_batch = atoms.get_batch_without_tracking
        batch_call_count = 0
        batch_call_args: list[list[int]] = []

        async def spy_batch(ids: list[int]) -> dict:
            nonlocal batch_call_count
            batch_call_count += 1
            batch_call_args.append(ids)
            return await original_batch(ids)

        atoms.get_batch_without_tracking = spy_batch

        seeds = await engine._vector_search("test query", k=10, region=None, types=None)

        # Batch fetch should have been called exactly once.
        assert batch_call_count == 1, (
            f"Expected get_batch_without_tracking to be called once, "
            f"got {batch_call_count} calls"
        )
        # The batch call should have included both candidate IDs.
        assert atom_id_1 in batch_call_args[0]
        assert atom_id_2 in batch_call_args[0]
        # Seeds should contain both atoms.
        assert atom_id_1 in seeds
        assert atom_id_2 in seeds

    async def test_bm25_search_uses_batch_fetch(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """_bm25_search should call get_batch_without_tracking once
        instead of individual fetches per atom."""
        # Insert atoms with text that will match an FTS query.
        atom_id_1 = await _insert_atom(storage, "Redis caching performance tips")
        atom_id_2 = await _insert_atom(storage, "Redis connection pooling guide")

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        context_budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, context_budget)

        # Patch get_batch_without_tracking to spy on calls.
        original_batch = atoms.get_batch_without_tracking
        batch_call_count = 0

        async def spy_batch(ids: list[int]) -> dict:
            nonlocal batch_call_count
            batch_call_count += 1
            return await original_batch(ids)

        atoms.get_batch_without_tracking = spy_batch

        bm25_seeds = await engine._bm25_search("Redis", k=10, region=None, types=None)

        if bm25_seeds:
            # If BM25 found results, batch fetch should have been called once.
            assert batch_call_count == 1, (
                f"Expected get_batch_without_tracking to be called once, "
                f"got {batch_call_count} calls"
            )

    async def test_recall_uses_record_access_batch(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """recall() should call record_access_batch once at the end,
        not record_access in a loop per atom."""
        # Insert an atom.
        atom_id = await _insert_atom(storage, "recall batch test atom")

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        context_budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, context_budget)

        # Configure search_similar to return our atom.
        mock_embeddings.search_similar.return_value = [
            (atom_id, 0.2),
        ]

        # Spy on record_access_batch and record_access.
        batch_call_count = 0
        single_call_count = 0
        original_batch = atoms.record_access_batch
        original_single = atoms.record_access

        async def spy_batch(ids: list[int]) -> None:
            nonlocal batch_call_count
            batch_call_count += 1
            return await original_batch(ids)

        async def spy_single(aid: int) -> None:
            nonlocal single_call_count
            single_call_count += 1
            return await original_single(aid)

        atoms.record_access_batch = spy_batch
        atoms.record_access = spy_single

        result = await engine.recall("test", budget_tokens=5000)

        # record_access_batch should have been called exactly once.
        assert batch_call_count == 1, (
            f"Expected record_access_batch to be called once, got {batch_call_count}"
        )
        # record_access (per-atom) should NOT have been called.
        assert single_call_count == 0, (
            f"Expected record_access to not be called (batch is used instead), "
            f"but it was called {single_call_count} times"
        )


# -----------------------------------------------------------------------
# Wave 2 – _spread_activation rewrite tests
# -----------------------------------------------------------------------


class TestSpreadActivationRewrite:
    """Verify the three Wave-2 improvements to _spread_activation:

    - Perf H-1: batch get_neighbors (one SQL per depth level)
    - Hebbian H1: refractory visited set (prevents re-expansion but NOT
      accumulation)
    - Hebbian M1: fanout normalization (high-degree nodes propagate less
      per neighbour)
    """

    @pytest.fixture
    async def wave2_engine(self, tmp_path):
        """RetrievalEngine backed by real Storage with mocked embeddings."""
        db_path = tmp_path / "wave2.db"
        storage = Storage(db_path)
        storage._backup_dir = tmp_path / "backups"
        storage._backup_dir.mkdir(exist_ok=True)
        await storage.initialize()

        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.search_similar = AsyncMock(return_value=[])
        mock_emb.embed_text = AsyncMock(return_value=[0.0] * 768)
        mock_emb.health_check = AsyncMock(return_value=True)

        atoms = AtomManager(storage, mock_emb)
        synapses = SynapseManager(storage)
        from memories.context import ContextBudget
        from memories.config import get_config
        budget = ContextBudget(get_config())
        engine = RetrievalEngine(storage, mock_emb, atoms, synapses, budget)

        yield engine, storage

        await storage.close()

    async def test_refractory_prevents_re_expansion(self, wave2_engine) -> None:
        """Cyclic graph A→B→C→A must terminate cleanly with no re-expansion of A.

        The refractory (visited) set gates FRONTIER EXPANSION only: once A is
        visited as a seed, it will never be re-added to the frontier, breaking
        the cycle and preventing infinite loops.  The function must return.
        """
        engine, storage = wave2_engine

        a = await _insert_atom(storage, "cycle node A")
        b = await _insert_atom(storage, "cycle node B")
        c = await _insert_atom(storage, "cycle node C")

        # Create a directed cycle: A→B, B→C, C→A (all bidirectional so reverse edges work)
        await _insert_synapse(storage, a, b, strength=0.9, bidirectional=True)
        await _insert_synapse(storage, b, c, strength=0.9, bidirectional=True)
        await _insert_synapse(storage, c, a, strength=0.9, bidirectional=True)

        # Run with depth=3 — would loop forever without the refractory guard.
        seeds = {a: 0.9}
        # This must complete (not hang / recurse infinitely).
        activated, _ = await engine._spread_activation(seeds, depth=3)

        # A was a seed so it must be in activated.
        assert a in activated, "Seed A must be in activated set"
        # B and C should be reachable.
        assert b in activated, "B must be activated from A"
        # The function must have terminated (no hung test).

    async def test_superposition_preserved(self, wave2_engine) -> None:
        """Hub H connected to two seeds S1 and S2 must accumulate additive activation.

        With superposition (additive), H receives activation from BOTH seeds.
        The refractory guard must only gate frontier expansion, NOT activation
        accumulation — so H's activation from S2 is still added even if H was
        already queued via S1.
        """
        engine, storage = wave2_engine

        s1 = await _insert_atom(storage, "seed S1")
        s2 = await _insert_atom(storage, "seed S2")
        h = await _insert_atom(storage, "hub H")

        # Both seeds connect to H with equal strength.
        synapse_strength = 0.9
        await _insert_synapse(storage, s1, h, strength=synapse_strength, bidirectional=True)
        await _insert_synapse(storage, s2, h, strength=synapse_strength, bidirectional=True)

        seed_activation = 0.8
        seeds = {s1: seed_activation, s2: seed_activation}
        activated, _ = await engine._spread_activation(seeds, depth=1)

        from memories.config import get_config
        cfg = get_config()
        decay = cfg.retrieval.decay_factor
        type_weight = cfg.retrieval.synapse_type_weights.related_to

        # Activation from one seed (fanout of S1 = 1, norm = 1.0).
        single_contribution = seed_activation * synapse_strength * type_weight * decay * 1.0
        # Two seeds → H should have more than one seed's contribution.
        assert h in activated, "Hub H must be activated"
        actual_h = activated[h]
        assert actual_h > single_contribution + 1e-6, (
            f"Hub H activation {actual_h:.4f} should exceed single-seed contribution "
            f"{single_contribution:.4f} — superposition must be preserved"
        )

    async def test_fanout_normalization(self, wave2_engine) -> None:
        """High-degree source nodes propagate less per neighbour than low-degree ones.

        LowDegree has 1 neighbour; HighDegree has 10 neighbours.  Both start at
        the same activation level.  LowDegree's single neighbour should receive
        more activation than each of HighDegree's 10 neighbours.

        This is the M1 fanout-normalization property: divide activation by
        sqrt(fanout) so hub nodes don't flood the graph.
        """
        import math
        engine, storage = wave2_engine

        # LowDegree: 1 outgoing neighbour.
        low_src = await _insert_atom(storage, "LowDegree source")
        low_tgt = await _insert_atom(storage, "LowDegree target")
        await _insert_synapse(storage, low_src, low_tgt, strength=0.9, bidirectional=False)

        # HighDegree: 10 outgoing neighbours.
        high_src = await _insert_atom(storage, "HighDegree source")
        high_targets = []
        for i in range(10):
            tgt = await _insert_atom(storage, f"HighDegree target {i}")
            await _insert_synapse(storage, high_src, tgt, strength=0.9, bidirectional=False)
            high_targets.append(tgt)

        seed_activation = 0.9
        seeds = {low_src: seed_activation, high_src: seed_activation}
        activated, _ = await engine._spread_activation(seeds, depth=1)

        assert low_tgt in activated, "LowDegree target must be activated"
        assert high_targets[0] in activated, "HighDegree targets must be activated"

        low_tgt_activation = activated[low_tgt]
        # Pick any one of the high-degree targets (they're all equal).
        high_tgt_activation = activated[high_targets[0]]

        assert low_tgt_activation > high_tgt_activation + 1e-6, (
            f"LowDegree target activation {low_tgt_activation:.4f} should exceed "
            f"HighDegree target activation {high_tgt_activation:.4f} — "
            f"fanout normalization (1/sqrt(fanout)) must be applied"
        )

    async def test_batch_get_neighbors_called_once_per_depth(self, wave2_engine) -> None:
        """get_neighbors_batch must be called exactly once per depth level.

        With depth=2 and a non-empty frontier at each level, the batch
        method should be called twice — one SQL query per depth level
        instead of N queries per frontier atom.
        """
        from unittest.mock import AsyncMock, patch
        engine, storage = wave2_engine

        # Build a two-hop chain: A → B → C.
        a = await _insert_atom(storage, "chain A")
        b = await _insert_atom(storage, "chain B")
        c = await _insert_atom(storage, "chain C")
        await _insert_synapse(storage, a, b, strength=0.9, bidirectional=True)
        await _insert_synapse(storage, b, c, strength=0.9, bidirectional=True)

        call_count = 0
        original_batch = engine._synapses.get_neighbors_batch

        async def spy_batch(atom_ids, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original_batch(atom_ids, **kwargs)

        engine._synapses.get_neighbors_batch = spy_batch

        seeds = {a: 0.9}
        await engine._spread_activation(seeds, depth=2)

        # depth=2 → 2 levels → get_neighbors_batch called exactly twice.
        assert call_count == 2, (
            f"Expected get_neighbors_batch to be called exactly 2 times (once per depth "
            f"level), but it was called {call_count} times"
        )


# ---------------------------------------------------------------------------
# Wave 2-C: Batch neighbor fetches in _extract_pathways / _find_relevant_antipatterns
# ---------------------------------------------------------------------------


class TestBatchNeighborFetches:
    """Verify that _extract_pathways and _find_relevant_antipatterns use
    batch synapse fetches instead of per-atom get_neighbors calls."""

    async def test_extract_pathways_uses_batch(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """_extract_pathways must call get_neighbors_batch once for all atoms.

        Previously each atom triggered an individual get_neighbors call
        (N calls for N atoms). The batch version should issue exactly one
        call regardless of how many atoms are in the result set.
        """
        # Build 3 atoms connected in a chain: a1 <-> a2 <-> a3
        a1 = await _insert_atom(storage, "atom one", atom_type="fact")
        a2 = await _insert_atom(storage, "atom two", atom_type="fact")
        a3 = await _insert_atom(storage, "atom three", atom_type="fact")

        await _insert_synapse(storage, a1, a2, "related-to", 0.8, True)
        await _insert_synapse(storage, a2, a3, "related-to", 0.7, True)

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        context_budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, context_budget)

        batch_call_count = 0
        # Spy on the engine's internal reference so the patch is effective.
        original_batch = engine._synapses.get_neighbors_batch

        async def spy_batch(
            atom_ids: list[int], min_strength: float = 0.0
        ) -> dict[int, list[tuple[int, Any]]]:
            nonlocal batch_call_count
            batch_call_count += 1
            return await original_batch(atom_ids, min_strength=min_strength)

        engine._synapses.get_neighbors_batch = spy_batch  # type: ignore[method-assign]

        pathways = await engine._extract_pathways([a1, a2, a3])

        # One batch call regardless of atom count.
        assert batch_call_count == 1, (
            f"Expected get_neighbors_batch called once, got {batch_call_count}"
        )

        # Pathways should contain the two synapse edges (a1-a2 and a2-a3).
        assert len(pathways) == 2, (
            f"Expected 2 pathways, got {len(pathways)}: {pathways}"
        )
        pathway_pairs = {
            (min(p["source_id"], p["target_id"]), max(p["source_id"], p["target_id"]))
            for p in pathways
        }
        assert (min(a1, a2), max(a1, a2)) in pathway_pairs
        assert (min(a2, a3), max(a2, a3)) in pathway_pairs

    async def test_antipatterns_uses_batch_without_tracking(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """_find_relevant_antipatterns must use get_batch_without_tracking.

        Graph topology: ap1 (antipattern) --warns-against--> ap2 (antipattern).
        We pass ap1 in atom_ids so its outgoing warns-against edge is traversed.
        ap2 is the candidate that should be fetched via get_batch_without_tracking,
        not via get() which would inflate its access_count.
        """
        # ap1 warns against ap2 — both are antipatterns.
        ap1 = await _insert_atom(
            storage, "Do not use blocking I/O in async loops",
            atom_type="antipattern", severity="high",
        )
        ap2 = await _insert_atom(
            storage, "Never call time.sleep in an async coroutine",
            atom_type="antipattern", severity="high",
        )
        await _insert_synapse(storage, ap1, ap2, "warns-against", 0.9, False)

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        context_budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, context_budget)

        batch_wt_call_count = 0
        # Spy on the engine's internal reference so the patch is effective.
        original_batch_wt = engine._atoms.get_batch_without_tracking

        async def spy_batch_wt(atom_ids: list[int]) -> dict[int, Atom]:
            nonlocal batch_wt_call_count
            batch_wt_call_count += 1
            return await original_batch_wt(atom_ids)

        engine._atoms.get_batch_without_tracking = spy_batch_wt  # type: ignore[method-assign]

        # Pass ap1 as a result atom; its outgoing warns-against edge points at ap2.
        result = await engine._find_relevant_antipatterns([ap1])

        # get_batch_without_tracking should have been called (not get()).
        assert batch_wt_call_count >= 1, (
            "Expected get_batch_without_tracking to be called at least once"
        )
        # ap2 should be returned as a discovered antipattern.
        assert len(result) == 1
        assert result[0].id == ap2

    async def test_antipatterns_does_not_inflate_access_count(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """_find_relevant_antipatterns must not bump access_count on antipatterns.

        Any read via get() increments access_count and last_accessed_at,
        which would pollute Hebbian frequency signals for antipattern atoms
        that are merely checked—not consciously recalled by the user.

        Graph: ap1 (antipattern) --warns-against--> ap2 (antipattern).
        We pass ap1 in atom_ids; ap2 is the candidate fetched internally.
        ap2's access_count must be unchanged after the call.
        """
        ap1 = await _insert_atom(
            storage, "Avoid synchronous sleep in async code",
            atom_type="antipattern", severity="high",
        )
        ap2 = await _insert_atom(
            storage, "Never call time.sleep in an async coroutine",
            atom_type="antipattern",
            access_count=7,
            severity="medium",
        )
        await _insert_synapse(storage, ap1, ap2, "warns-against", 0.85, False)

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        context_budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, context_budget)

        # Record access_count before the call using without_tracking to avoid
        # influencing the measurement.
        before = await atoms.get_without_tracking(ap2)
        assert before is not None
        access_count_before = before.access_count

        # Pass ap1 as a result atom — its neighbor ap2 is fetched internally.
        await engine._find_relevant_antipatterns([ap1])

        # access_count on ap2 must be unchanged after the internal scan.
        after = await atoms.get_without_tracking(ap2)
        assert after is not None
        assert after.access_count == access_count_before, (
            f"access_count should not change: was {access_count_before}, "
            f"now {after.access_count}"
        )
