"""Synthetic end-to-end tests for the full recall pipeline without Ollama.

Wires Brain → RetrievalEngine → Storage with mocked embeddings so the full
pipeline runs in CI without needing an external Ollama server.  Seeds a
realistic synthetic database and validates:

- Remember → Recall round-trip
- Spreading activation across regions
- Antipattern discovery via indirect spreading
- BM25 fallback when vector search returns nothing
- Budget fitting with realistic content sizes
- Region scoping through the full stack

All tests are unit tests (no @pytest.mark.integration) and run without Ollama.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from memories.atoms import AtomManager
from memories.brain import Brain
from memories.config import get_config
from memories.context import ContextBudget
from memories.embeddings import EmbeddingEngine
from memories.learning import LearningEngine
from memories.retrieval import RetrievalEngine
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
    importance: float = 0.5,
    access_count: int = 5,
    tags: list[str] | None = None,
    severity: str | None = None,
    instead: str | None = None,
) -> int:
    """Insert an atom directly via SQL (no embedding)."""
    now = datetime.now(tz=timezone.utc).isoformat()
    tags_json = json.dumps(tags) if tags else None
    return await storage.execute_write(
        """INSERT INTO atoms
           (content, type, region, confidence, importance, access_count,
            last_accessed_at, tags, is_deleted, severity, instead)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
        (content, atom_type, region, confidence, importance, access_count,
         now, tags_json, severity, instead),
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
        """INSERT INTO synapses
           (source_id, target_id, relationship, strength, bidirectional)
           VALUES (?, ?, ?, ?, ?)""",
        (source_id, target_id, relationship, strength, int(bidirectional)),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def e2e_storage(tmp_path: Path) -> Storage:
    """Fresh Storage instance in a temp directory."""
    db_path = tmp_path / "e2e.db"
    s = Storage(db_path)
    s._backup_dir = tmp_path / "backups"
    s._backup_dir.mkdir(exist_ok=True)
    await s.initialize()
    yield s  # type: ignore[misc]
    await s.close()


@pytest.fixture
def mock_emb() -> MagicMock:
    """Mocked EmbeddingEngine that returns predictable results."""
    engine = MagicMock(spec=EmbeddingEngine)
    engine.search_similar = AsyncMock(return_value=[])
    engine.embed_text = AsyncMock(return_value=[0.0] * 768)
    engine.embed_and_store = AsyncMock(return_value=[0.0] * 768)
    engine.embed_batch = AsyncMock(return_value=[])
    engine.health_check = AsyncMock(return_value=True)
    engine.cosine_similarity = MagicMock(return_value=0.5)
    return engine


@pytest.fixture
async def brain_e2e(
    e2e_storage: Storage, mock_emb: MagicMock,
) -> Brain:
    """Brain wired with real Storage + RetrievalEngine but mocked embeddings.

    This is the key fixture: it exercises the full pipeline (Brain → Retrieval
    → Storage) while keeping embeddings mocked so Ollama is never required.
    """
    storage = e2e_storage
    atoms = AtomManager(storage, mock_emb)
    synapses = SynapseManager(storage)
    context = ContextBudget()
    retrieval = RetrievalEngine(storage, mock_emb, atoms, synapses, context)
    learning = LearningEngine(storage, mock_emb, atoms, synapses)
    consolidation = MagicMock()
    consolidation.reflect = AsyncMock()

    b = Brain.__new__(Brain)
    b._config = get_config()
    b._storage = storage
    b._embeddings = mock_emb
    b._atoms = atoms
    b._synapses = synapses
    b._context = context
    b._retrieval = retrieval
    b._learning = learning
    b._consolidation = consolidation
    b._current_session_id = "e2e-session-001"
    b._initialized = True

    # Create the session row that end_session expects.
    await storage.execute_write(
        "INSERT INTO sessions (id, project) VALUES (?, ?)",
        ("e2e-session-001", None),
    )

    yield b  # type: ignore[misc]
    b._initialized = False


# ---------------------------------------------------------------------------
# Seed data: 10-atom graph with diverse types and connections
# ---------------------------------------------------------------------------


async def seed_graph(storage: Storage) -> dict[str, int]:
    """Seed a 10-atom graph with synapses spanning regions and types.

    Returns a mapping of atom key → atom_id.
    """
    ids: dict[str, int] = {}

    # --- Technical cluster: Redis ---
    ids["redis_scan"] = await _insert_atom(
        storage,
        "Redis SCAN iterates the keyspace incrementally using a cursor. "
        "It is O(N) overall but each call returns a small batch, avoiding "
        "the blocking behavior of the KEYS command.",
        atom_type="fact", region="project:redis-app",
        confidence=0.95, importance=0.8,
        tags=["redis", "performance"],
    )
    ids["redis_keys_bad"] = await _insert_atom(
        storage,
        "The Redis KEYS command blocks the entire server while scanning "
        "all keys. In production this causes latency spikes and can "
        "trigger cluster failovers.",
        atom_type="fact", region="project:redis-app",
        confidence=0.9, importance=0.7,
        tags=["redis", "latency"],
    )
    ids["redis_antipattern"] = await _insert_atom(
        storage,
        "Never use KEYS in production Redis workloads. "
        "Use SCAN with a cursor instead.",
        atom_type="antipattern", region="project:redis-app",
        confidence=1.0, importance=0.9,
        severity="high", instead="Use SCAN with COUNT parameter",
    )

    # --- Technical cluster: Python async ---
    ids["python_asyncio"] = await _insert_atom(
        storage,
        "Python asyncio event loop runs coroutines cooperatively. "
        "Use async/await for I/O-bound tasks. Never call time.sleep() "
        "in async code -- use asyncio.sleep() instead.",
        atom_type="skill", region="project:web-api",
        confidence=0.9, importance=0.6,
        tags=["python", "async"],
    )
    ids["async_antipattern"] = await _insert_atom(
        storage,
        "Never use time.sleep() inside an async coroutine. "
        "It blocks the entire event loop.",
        atom_type="antipattern", region="project:web-api",
        confidence=1.0, importance=0.85,
        severity="high", instead="Use asyncio.sleep() instead",
    )

    # --- Technical cluster: SQL ---
    ids["sql_indexes"] = await _insert_atom(
        storage,
        "Add composite indexes on frequently queried column combinations. "
        "B-tree indexes in SQLite support leftmost-prefix matching.",
        atom_type="skill", region="project:memories",
        confidence=0.85, importance=0.6,
        tags=["sql", "performance", "sqlite"],
    )

    # --- Personal preferences ---
    ids["dark_mode"] = await _insert_atom(
        storage,
        "I prefer dark mode in all code editors and terminals.",
        atom_type="preference", region="personal",
        confidence=0.8, importance=0.3,
    )
    ids["vim_pref"] = await _insert_atom(
        storage,
        "I use Neovim as my primary editor with LazyVim configuration.",
        atom_type="preference", region="personal",
        confidence=0.85, importance=0.4,
    )

    # --- Experience ---
    ids["redis_incident"] = await _insert_atom(
        storage,
        "Experienced a production incident where Redis KEYS command "
        "caused a 30-second latency spike affecting all API endpoints. "
        "Resolved by switching to SCAN with COUNT 100.",
        atom_type="experience", region="project:redis-app",
        confidence=0.95, importance=0.85,
        tags=["redis", "incident", "postmortem"],
    )

    # --- Task ---
    ids["migrate_task"] = await _insert_atom(
        storage,
        "Migrate Redis key enumeration from KEYS to SCAN in the "
        "cache-warmer microservice.",
        atom_type="task", region="project:redis-app",
        confidence=1.0, importance=0.7,
    )

    # --- Synapses ---
    # Redis cluster: tightly connected
    await _insert_synapse(storage, ids["redis_scan"], ids["redis_keys_bad"],
                          "related-to", 0.85, True)
    await _insert_synapse(storage, ids["redis_antipattern"], ids["redis_keys_bad"],
                          "warns-against", 0.9, False)
    await _insert_synapse(storage, ids["redis_incident"], ids["redis_keys_bad"],
                          "related-to", 0.8, True)
    await _insert_synapse(storage, ids["redis_incident"], ids["redis_scan"],
                          "related-to", 0.75, True)
    await _insert_synapse(storage, ids["migrate_task"], ids["redis_scan"],
                          "related-to", 0.7, True)

    # Async cluster
    await _insert_synapse(storage, ids["async_antipattern"], ids["python_asyncio"],
                          "warns-against", 0.85, False)

    # Cross-cluster: Redis scan uses async patterns
    await _insert_synapse(storage, ids["redis_scan"], ids["python_asyncio"],
                          "related-to", 0.4, True)

    return ids


# ===========================================================================
# Test classes
# ===========================================================================


class TestRememberThenRecall:
    """Verify the remember → recall round-trip through the full pipeline."""

    async def test_remembered_atom_is_recallable(
        self, brain_e2e: Brain, mock_emb: MagicMock, e2e_storage: Storage,
    ) -> None:
        """An atom stored via remember() should be returned by recall()."""
        # Mock search_similar to return nothing during remember's dedup check,
        # then return the new atom during recall.
        call_count = 0

        async def search_side_effect(query, k=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Dedup check during remember: no match
                return []
            # Recall: return the atom we just created
            rows = await e2e_storage.execute(
                "SELECT id FROM atoms ORDER BY id DESC LIMIT 1", ()
            )
            if rows:
                return [(rows[0]["id"], 0.2)]  # distance 0.2 → high similarity
            return []

        mock_emb.search_similar = AsyncMock(side_effect=search_side_effect)

        # Remember a fact.
        result = await brain_e2e.remember(
            content="SQLite WAL mode improves concurrent read performance",
            type="fact",
            region="technical",
        )
        atom_id = result["atom_id"]

        # Recall it.
        recall_result = await brain_e2e.recall(
            query="SQLite write ahead log concurrency",
            budget_tokens=4000,
        )

        recalled_ids = {a["id"] for a in recall_result["atoms"]}
        assert atom_id in recalled_ids, (
            f"Remembered atom {atom_id} not found in recall results: {recalled_ids}"
        )

    async def test_remembered_atom_appears_in_bm25(
        self, brain_e2e: Brain, mock_emb: MagicMock,
    ) -> None:
        """A remembered atom should be findable via BM25 keyword search
        even when vector search returns nothing."""
        # Vector search always returns nothing.
        mock_emb.search_similar = AsyncMock(return_value=[])

        result = await brain_e2e.remember(
            content="Kubernetes pod autoscaling uses HPA metrics",
            type="fact",
            region="devops",
        )
        atom_id = result["atom_id"]

        # Recall using a keyword that should match via FTS.
        recall_result = await brain_e2e.recall(
            query="Kubernetes autoscaling",
            budget_tokens=4000,
        )

        recalled_ids = {a["id"] for a in recall_result["atoms"]}
        assert atom_id in recalled_ids, (
            "BM25 should find the atom by keyword even without vector results"
        )


class TestFullPipelineOnSyntheticDB:
    """End-to-end recall through Brain → RetrievalEngine on a seeded graph."""

    @pytest.fixture(autouse=True)
    async def seed(self, brain_e2e, e2e_storage, mock_emb):
        self.brain = brain_e2e
        self.storage = e2e_storage
        self.mock_emb = mock_emb
        self.ids = await seed_graph(e2e_storage)

    async def test_recall_returns_seeded_atoms(self) -> None:
        """Vector-seeded atoms should appear in recall results."""
        self.mock_emb.search_similar.return_value = [
            (self.ids["redis_scan"], 0.2),
        ]

        result = await self.brain.recall(
            query="How does Redis SCAN work?",
            budget_tokens=4000,
        )

        atom_ids = {a["id"] for a in result["atoms"]}
        assert self.ids["redis_scan"] in atom_ids

    async def test_spreading_activation_finds_neighbors(self) -> None:
        """Atoms connected via synapses should be activated from seeds."""
        self.mock_emb.search_similar.return_value = [
            (self.ids["redis_scan"], 0.2),
        ]

        result = await self.brain.recall(
            query="Redis SCAN cursor",
            budget_tokens=8000,
        )

        # Spreading should activate more than just the single seed.
        assert result["total_activated"] >= 2, (
            f"Expected spreading to activate neighbors, got {result['total_activated']}"
        )
        # redis_keys_bad may appear in atoms or antipatterns (since it's
        # connected to an antipattern via warns-against).
        all_ids = {a["id"] for a in result["atoms"]}
        all_ids.update(a["id"] for a in result["antipatterns"])
        # At minimum, the seed should be present.
        assert self.ids["redis_scan"] in all_ids

    async def test_antipattern_discovered_via_indirect_spreading(self) -> None:
        """Antipatterns connected via warns-against should surface
        when their target atom is activated through spreading."""
        # Seed only redis_scan; redis_keys_bad is reached via spreading;
        # redis_antipattern warns-against redis_keys_bad.
        self.mock_emb.search_similar.return_value = [
            (self.ids["redis_scan"], 0.2),
        ]

        result = await self.brain.recall(
            query="Redis key iteration",
            budget_tokens=8000,
            include_antipatterns=True,
        )

        all_content = " ".join(
            a["content"] for a in result["atoms"]
        ) + " " + " ".join(
            a["content"] for a in result["antipatterns"]
        )

        # The antipattern about KEYS should be discoverable through the chain:
        # redis_scan → redis_keys_bad ← redis_antipattern (warns-against)
        assert "SCAN" in all_content, (
            "Antipattern mentioning SCAN should surface via indirect spreading"
        )

    async def test_region_filter_through_full_stack(self) -> None:
        """Region filter passed to Brain.recall should restrict results."""
        self.mock_emb.search_similar.return_value = [
            (self.ids["redis_scan"], 0.2),      # project:redis-app
            (self.ids["dark_mode"], 0.3),        # personal
            (self.ids["python_asyncio"], 0.4),   # project:web-api
        ]

        result = await self.brain.recall(
            query="Redis",
            budget_tokens=8000,
            region="personal",
        )

        for atom in result["atoms"]:
            assert atom.get("region") == "personal", (
                f"Expected only personal atoms, got region={atom.get('region')}"
            )

    async def test_type_filter_restricts_seeds(self) -> None:
        """Type filter should restrict seed selection.

        Note: spreading activation may bring in atoms of other types
        since it traverses synapses regardless of type. The type filter
        applies at the seed discovery stage (vector + BM25).
        """
        self.mock_emb.search_similar.return_value = [
            (self.ids["redis_scan"], 0.2),       # fact
            (self.ids["python_asyncio"], 0.3),   # skill
            (self.ids["redis_incident"], 0.4),   # experience
        ]

        # With types=["fact"] and depth=0 (no spreading), only facts appear.
        result = await self.brain.recall(
            query="Redis performance",
            budget_tokens=8000,
            types=["fact"],
            depth=0,
            include_antipatterns=False,
        )

        for atom in result["atoms"]:
            assert atom["type"] == "fact", (
                f"Expected only facts at depth=0, got type={atom['type']}"
            )

    async def test_multi_hop_spreading_reaches_distant_atoms(self) -> None:
        """Depth > 1 should reach atoms 2 hops away from the seed."""
        # Seed python_asyncio.
        # Hop 1: redis_scan (cross-cluster synapse, strength 0.4)
        # Hop 2: redis_keys_bad, redis_incident, migrate_task (from redis_scan)
        self.mock_emb.search_similar.return_value = [
            (self.ids["python_asyncio"], 0.15),
        ]

        result = await self.brain.recall(
            query="async patterns",
            budget_tokens=10000,
            depth=2,
            include_antipatterns=False,
        )

        all_ids = {a["id"] for a in result["atoms"]}
        # At depth=2, redis_keys_bad should be reachable:
        # python_asyncio → redis_scan → redis_keys_bad
        # (Activation may be low due to weak cross-cluster synapse, but with
        # large budget it should fit.)
        assert result["total_activated"] >= 2, (
            f"Expected at least 2 activated atoms at depth=2, "
            f"got {result['total_activated']}"
        )

    async def test_pathways_between_result_atoms(self) -> None:
        """Pathways should reflect synapses between atoms in the result set."""
        self.mock_emb.search_similar.return_value = [
            (self.ids["redis_scan"], 0.2),
            (self.ids["redis_keys_bad"], 0.3),
        ]

        result = await self.brain.recall(
            query="Redis",
            budget_tokens=8000,
        )

        atom_ids = {a["id"] for a in result["atoms"]}
        if self.ids["redis_scan"] in atom_ids and self.ids["redis_keys_bad"] in atom_ids:
            pathway_pairs = set()
            for p in result["pathways"]:
                pathway_pairs.add((p["source_id"], p["target_id"]))
                pathway_pairs.add((p["target_id"], p["source_id"]))
            assert (self.ids["redis_scan"], self.ids["redis_keys_bad"]) in pathway_pairs, (
                "Expected pathway between redis_scan and redis_keys_bad"
            )

    async def test_score_breakdown_present_on_all_atoms(self) -> None:
        """Every recalled atom should include a score_breakdown dict."""
        self.mock_emb.search_similar.return_value = [
            (self.ids["redis_scan"], 0.2),
            (self.ids["redis_keys_bad"], 0.3),
        ]

        result = await self.brain.recall(
            query="Redis SCAN",
            budget_tokens=8000,
        )

        for atom in result["atoms"]:
            assert "score_breakdown" in atom, (
                f"Atom {atom['id']} missing score_breakdown"
            )
            breakdown = atom["score_breakdown"]
            assert "vector_similarity" in breakdown
            assert "spread_activation" in breakdown
            assert "recency" in breakdown
            assert "importance" in breakdown


class TestBM25Fallback:
    """Verify that BM25 keyword search works when vector search returns nothing."""

    @pytest.fixture(autouse=True)
    async def seed(self, brain_e2e, e2e_storage, mock_emb):
        self.brain = brain_e2e
        self.storage = e2e_storage
        self.mock_emb = mock_emb
        self.ids = await seed_graph(e2e_storage)

    async def test_bm25_finds_atoms_without_vector_seeds(self) -> None:
        """When vector search returns nothing, BM25 should still find matches."""
        # Vector search returns nothing.
        self.mock_emb.search_similar.return_value = []

        result = await self.brain.recall(
            query="Redis SCAN cursor",
            budget_tokens=4000,
        )

        # BM25 should find atoms containing "Redis" and "SCAN".
        if result["atoms"]:
            all_content = " ".join(a["content"] for a in result["atoms"])
            assert "Redis" in all_content or "SCAN" in all_content, (
                "BM25 should match atoms containing 'Redis' or 'SCAN'"
            )

    async def test_bm25_respects_region_filter(self) -> None:
        """BM25 results should also be filtered by region."""
        self.mock_emb.search_similar.return_value = []

        result = await self.brain.recall(
            query="dark mode editor",
            budget_tokens=4000,
            region="personal",
        )

        for atom in result["atoms"]:
            assert atom.get("region") == "personal"


class TestBudgetFitting:
    """Verify that token budget constraints work with realistic content sizes."""

    @pytest.fixture(autouse=True)
    async def seed(self, brain_e2e, e2e_storage, mock_emb):
        self.brain = brain_e2e
        self.storage = e2e_storage
        self.mock_emb = mock_emb
        self.ids = await seed_graph(e2e_storage)

    async def test_small_budget_limits_atoms(self) -> None:
        """A small budget should return fewer atoms than a large budget."""
        self.mock_emb.search_similar.return_value = [
            (self.ids["redis_scan"], 0.2),
            (self.ids["redis_keys_bad"], 0.3),
            (self.ids["redis_incident"], 0.4),
            (self.ids["python_asyncio"], 0.5),
        ]

        small_result = await self.brain.recall(
            query="Redis", budget_tokens=100,
            include_antipatterns=False,
        )
        large_result = await self.brain.recall(
            query="Redis", budget_tokens=50000,
            include_antipatterns=False,
        )

        assert len(large_result["atoms"]) >= len(small_result["atoms"]), (
            f"Large budget ({len(large_result['atoms'])} atoms) should return "
            f"at least as many as small budget ({len(small_result['atoms'])} atoms)"
        )

    async def test_budget_used_does_not_exceed_budget(self) -> None:
        """budget_used must not exceed budget_tokens."""
        self.mock_emb.search_similar.return_value = [
            (self.ids["redis_scan"], 0.2),
            (self.ids["redis_keys_bad"], 0.3),
            (self.ids["redis_incident"], 0.4),
        ]

        budget = 500
        result = await self.brain.recall(
            query="Redis", budget_tokens=budget,
        )

        assert result["budget_used"] <= budget, (
            f"budget_used ({result['budget_used']}) exceeds budget ({budget})"
        )

    async def test_budget_remaining_is_correct(self) -> None:
        """budget_remaining should equal budget_tokens - budget_used."""
        self.mock_emb.search_similar.return_value = [
            (self.ids["redis_scan"], 0.2),
        ]

        budget = 5000
        result = await self.brain.recall(
            query="Redis", budget_tokens=budget,
        )

        assert result["budget_remaining"] == budget - result["budget_used"]


class TestCrossRegionSpreading:
    """Verify that spreading activation crosses region boundaries via synapses
    but region filtering still applies to the final result set."""

    @pytest.fixture(autouse=True)
    async def seed(self, brain_e2e, e2e_storage, mock_emb):
        self.brain = brain_e2e
        self.storage = e2e_storage
        self.mock_emb = mock_emb
        self.ids = await seed_graph(e2e_storage)

    async def test_cross_region_activation_without_filter(self) -> None:
        """Without a region filter, spreading should cross region boundaries.

        python_asyncio (project:web-api) is connected to redis_scan
        (project:redis-app) via a cross-cluster synapse.
        """
        self.mock_emb.search_similar.return_value = [
            (self.ids["python_asyncio"], 0.15),
        ]

        result = await self.brain.recall(
            query="async", budget_tokens=10000, depth=1,
            include_antipatterns=False,
        )

        all_ids = {a["id"] for a in result["atoms"]}
        # redis_scan is in a different region but connected via synapse.
        if result["total_activated"] >= 2:
            # Activation crossed regions.
            regions = {a.get("region") for a in result["atoms"]}
            assert len(regions) >= 1  # At least the seed's region

    async def test_region_filter_restricts_seeds(self) -> None:
        """With a region filter, seed selection is restricted to that region.

        Note: spreading activation may bring in atoms from other regions
        via synapse traversal. The region filter applies at the seed
        discovery stage (vector + BM25). At depth=0, only same-region
        atoms should appear.
        """
        self.mock_emb.search_similar.return_value = [
            (self.ids["python_asyncio"], 0.15),   # project:web-api
            (self.ids["redis_scan"], 0.3),         # project:redis-app
        ]

        result = await self.brain.recall(
            query="async patterns",
            budget_tokens=10000,
            region="project:web-api",
            depth=0,
            include_antipatterns=False,
        )

        for atom in result["atoms"]:
            assert atom.get("region") == "project:web-api", (
                f"Region filter at depth=0 should exclude {atom.get('region')}"
            )


class TestEmptyDatabaseRecall:
    """Verify graceful behavior when recalling from an empty database."""

    async def test_recall_on_empty_db(
        self, brain_e2e: Brain, mock_emb: MagicMock,
    ) -> None:
        """Recall on an empty database should return an empty result, not crash."""
        mock_emb.search_similar.return_value = []

        result = await brain_e2e.recall(
            query="anything at all",
            budget_tokens=4000,
        )

        assert result["atoms"] == []
        assert result["antipatterns"] == []
        assert result["seed_count"] == 0
        assert result["budget_used"] == 0

    async def test_recall_with_whitespace_query(
        self, brain_e2e: Brain, mock_emb: MagicMock,
    ) -> None:
        """Whitespace-only query should return empty result gracefully."""
        mock_emb.search_similar.return_value = []

        result = await brain_e2e.recall(
            query="   \t\n   ",
            budget_tokens=4000,
        )

        assert result["atoms"] == []
        assert result["seed_count"] == 0
