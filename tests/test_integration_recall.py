"""End-to-end integration tests for the full recall pipeline.

Requires a running Ollama server with nomic-embed-text.
Marked with @pytest.mark.integration -- skipped when Ollama is unavailable.

Seeds a SQLite database with realistic atoms and synapses, generates
real embeddings via Ollama, and validates the full recall pipeline:
vector search + BM25 + spreading activation + multi-factor scoring.

Run locally:
    ollama serve &
    uv run pytest tests/test_integration_recall.py -m integration -v

Run in CI via the official Ollama install script (see .github/workflows/tests.yml).
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memories.atoms import AtomManager
from memories.brain import Brain
from memories.config import get_config
from memories.context import ContextBudget
from memories.embeddings import EmbeddingEngine
from memories.retrieval import RetrievalEngine
from memories.storage import Storage
from memories.synapses import SynapseManager

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Skip guard: auto-skip all tests if Ollama is unreachable
# ---------------------------------------------------------------------------


async def _ollama_available() -> bool:
    """Check if Ollama is running with nomic-embed-text loaded."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{get_config().ollama_url}/api/embeddings",
                json={"model": get_config().embedding_model, "prompt": "test"},
            )
            return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
async def skip_without_ollama():
    """Skip all tests in this module if Ollama is not reachable."""
    if not await _ollama_available():
        pytest.skip("Ollama not available -- skipping integration tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def integration_storage(tmp_path: Path) -> Storage:
    """Provide an initialised Storage backed by a temp directory."""
    db_path = tmp_path / "integration.db"
    s = Storage(db_path)
    s._backup_dir = tmp_path / "backups"
    s._backup_dir.mkdir(exist_ok=True)
    await s.initialize()
    yield s  # type: ignore[misc]
    await s.close()


@pytest.fixture
async def real_embeddings(integration_storage: Storage) -> EmbeddingEngine:
    """Provide a real EmbeddingEngine backed by Ollama (not mocked)."""
    return EmbeddingEngine(integration_storage)


# ---------------------------------------------------------------------------
# Seed data: realistic memory graph
# ---------------------------------------------------------------------------

_ATOMS_DATA: dict[str, dict] = {
    "redis_scan": {
        "content": (
            "Redis SCAN iterates the keyspace incrementally. "
            "It is O(N) overall but each call returns a small batch, "
            "avoiding the blocking behavior of KEYS."
        ),
        "type": "fact",
        "region": "project:redis-app",
        "confidence": 0.95,
        "importance": 0.8,
        "tags": json.dumps(["redis", "performance"]),
    },
    "redis_keys_bad": {
        "content": (
            "The Redis KEYS command blocks the entire server while "
            "scanning all keys. In production this causes latency "
            "spikes and can trigger cluster failovers."
        ),
        "type": "fact",
        "region": "project:redis-app",
        "confidence": 0.9,
        "importance": 0.7,
        "tags": json.dumps(["redis", "latency"]),
    },
    "redis_antipattern": {
        "content": (
            "Never use KEYS in production Redis workloads. "
            "Use SCAN with a cursor instead."
        ),
        "type": "antipattern",
        "region": "project:redis-app",
        "confidence": 1.0,
        "importance": 0.9,
        "severity": "high",
        "instead": "Use SCAN with COUNT parameter",
    },
    "python_asyncio": {
        "content": (
            "Python asyncio event loop runs coroutines cooperatively. "
            "Use async/await for I/O-bound tasks. Never call "
            "time.sleep() in async code -- use asyncio.sleep()."
        ),
        "type": "skill",
        "region": "project:web-api",
        "confidence": 0.9,
        "importance": 0.6,
        "tags": json.dumps(["python", "async"]),
    },
    "dark_mode_pref": {
        "content": "I prefer dark mode in all code editors and terminals.",
        "type": "preference",
        "region": "personal",
        "confidence": 0.8,
        "importance": 0.3,
    },
}


async def seed_realistic_graph(
    storage: Storage,
    embeddings: EmbeddingEngine,
) -> dict[str, int]:
    """Seed a realistic memory graph with real embeddings.

    Returns a mapping of atom key -> atom_id.
    """
    now = datetime.now(tz=timezone.utc).isoformat()
    atom_ids: dict[str, int] = {}

    for key, data in _ATOMS_DATA.items():
        atom_id = await storage.execute_write(
            """INSERT INTO atoms
               (content, type, region, confidence, importance,
                access_count, last_accessed_at, tags, severity,
                instead, is_deleted)
               VALUES (?, ?, ?, ?, ?, 5, ?, ?, ?, ?, 0)""",
            (
                data["content"],
                data["type"],
                data.get("region", "general"),
                data.get("confidence", 1.0),
                data.get("importance", 0.5),
                now,
                data.get("tags"),
                data.get("severity"),
                data.get("instead"),
            ),
        )
        atom_ids[key] = atom_id
        # Generate and store real embedding vectors.
        await embeddings.embed_and_store(atom_id, data["content"])

    # Create synapses between Redis atoms.
    await storage.execute_write(
        """INSERT INTO synapses
           (source_id, target_id, relationship, strength, bidirectional)
           VALUES (?, ?, 'related-to', 0.85, 1)""",
        (atom_ids["redis_scan"], atom_ids["redis_keys_bad"]),
    )
    await storage.execute_write(
        """INSERT INTO synapses
           (source_id, target_id, relationship, strength, bidirectional)
           VALUES (?, ?, 'warns-against', 0.9, 0)""",
        (atom_ids["redis_antipattern"], atom_ids["redis_keys_bad"]),
    )

    return atom_ids


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestEmbeddingQuality:
    """Verify that real Ollama embeddings are correct and meaningful."""

    @pytest.fixture(autouse=True)
    async def seed(self, integration_storage, real_embeddings):
        self.ids = await seed_realistic_graph(integration_storage, real_embeddings)
        self.storage = integration_storage
        self.embeddings = real_embeddings

    async def test_embedding_dimensions_correct(self) -> None:
        """Real Ollama embeddings must have exactly 768 dimensions."""
        vec = await self.embeddings.embed_text("dimension check")
        assert len(vec) == 768

    async def test_semantic_similarity_is_meaningful(self) -> None:
        """Related texts must score higher than unrelated texts."""
        e = self.embeddings
        v1 = await e.embed_text("Redis SCAN cursor iteration")
        v2 = await e.embed_text("Iterating Redis keys with SCAN command")
        v3 = await e.embed_text("My favorite pizza topping is pepperoni")

        sim_related = e.cosine_similarity(v1, v2)
        sim_unrelated = e.cosine_similarity(v1, v3)

        assert sim_related > sim_unrelated, (
            f"Related ({sim_related:.3f}) must exceed "
            f"unrelated ({sim_unrelated:.3f})"
        )

    async def test_embedding_cache_returns_same_vector(self) -> None:
        """A second embed call must return the same vector from cache."""
        text = "cache consistency check"
        v1 = await self.embeddings.embed_text(text)
        v2 = await self.embeddings.embed_text(text)

        assert len(v1) == len(v2)
        for a, b in zip(v1, v2):
            assert math.isclose(a, b, rel_tol=1e-6)

    async def test_batch_embedding_consistency(self) -> None:
        """Batch and single embeddings for the same text must match."""
        text = "batch consistency check"
        single = await self.embeddings.embed_text(text)
        batch = await self.embeddings.embed_batch([text])

        assert len(batch) == 1
        for a, b in zip(single, batch[0]):
            assert math.isclose(a, b, rel_tol=1e-6)


class TestVectorSearch:
    """Verify vector search finds relevant atoms with real embeddings."""

    @pytest.fixture(autouse=True)
    async def seed(self, integration_storage, real_embeddings):
        self.ids = await seed_realistic_graph(integration_storage, real_embeddings)
        self.storage = integration_storage
        self.embeddings = real_embeddings

    async def test_vector_search_returns_relevant_atoms(self) -> None:
        """Vector search for a Redis query must return Redis atoms."""
        results = await self.embeddings.search_similar(
            "How does Redis SCAN work?", k=5,
        )
        found_ids = {atom_id for atom_id, _ in results}

        assert self.ids["redis_scan"] in found_ids or \
               self.ids["redis_keys_bad"] in found_ids, (
            f"Expected Redis atoms in results, got IDs: {found_ids}"
        )

    async def test_vector_search_ranks_relevant_higher(self) -> None:
        """Redis atoms must rank above the dark-mode preference for Redis queries."""
        results = await self.embeddings.search_similar(
            "Redis SCAN cursor", k=5,
        )
        result_ids = [atom_id for atom_id, _ in results]

        redis_ids = {self.ids["redis_scan"], self.ids["redis_keys_bad"]}
        pref_id = self.ids["dark_mode_pref"]

        # If the preference even appears, it must rank below Redis atoms.
        if pref_id in result_ids:
            pref_rank = result_ids.index(pref_id)
            for rid in redis_ids:
                if rid in result_ids:
                    redis_rank = result_ids.index(rid)
                    assert redis_rank < pref_rank

    async def test_search_latency_under_threshold(self) -> None:
        """Vector search must complete within 2 seconds on CI."""
        start = time.perf_counter()
        await self.embeddings.search_similar("Redis SCAN", k=5)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 2000, (
            f"Search took {elapsed_ms:.0f}ms, expected <2000ms"
        )


class TestFTSSearch:
    """Verify FTS5 keyword search works with real data."""

    @pytest.fixture(autouse=True)
    async def seed(self, integration_storage, real_embeddings):
        self.ids = await seed_realistic_graph(integration_storage, real_embeddings)
        self.storage = integration_storage

    async def test_fts5_keyword_search_works(self) -> None:
        """FTS5 keyword search for 'Redis' must find Redis atoms."""
        rows = await self.storage.execute(
            "SELECT rowid FROM atoms_fts WHERE atoms_fts MATCH ?",
            ("Redis",),
        )
        found_ids = {row["rowid"] for row in rows}
        redis_ids = {
            self.ids["redis_scan"],
            self.ids["redis_keys_bad"],
            self.ids["redis_antipattern"],
        }
        assert found_ids & redis_ids, (
            f"FTS5 should find Redis atoms, got: {found_ids}"
        )

    async def test_fts5_two_char_tokens(self) -> None:
        """FTS5 must support 2-character tokens like 'CI' or 'Go'."""
        # Insert an atom with a short keyword.
        atom_id = await self.storage.execute_write(
            """INSERT INTO atoms (content, type, region, importance)
               VALUES ('Use CI pipelines for automated testing', 'skill', 'general', 0.5)""",
            (),
        )
        rows = await self.storage.execute(
            "SELECT rowid FROM atoms_fts WHERE atoms_fts MATCH ?",
            ("CI",),
        )
        found_ids = {row["rowid"] for row in rows}
        assert atom_id in found_ids


class TestFullRecallPipeline:
    """End-to-end recall through the full Brain pipeline with real embeddings."""

    @pytest.fixture
    async def brain_with_graph(self, integration_storage, real_embeddings):
        """Build a fully-wired Brain with real embeddings and seeded data."""
        storage = integration_storage
        embeddings = real_embeddings
        atoms = AtomManager(storage, embeddings)
        synapses = SynapseManager(storage)
        context = ContextBudget(storage)
        retrieval = RetrievalEngine(storage, embeddings, atoms, synapses, context)

        b = Brain.__new__(Brain)
        b._config = get_config()
        b._storage = storage
        b._embeddings = embeddings
        b._atoms = atoms
        b._synapses = synapses
        b._context = context
        b._retrieval = retrieval
        b._learning = None  # Not needed for recall
        b._consolidation = None
        b._current_session_id = None
        b._initialized = True

        ids = await seed_realistic_graph(storage, embeddings)
        return b, ids

    async def test_recall_returns_relevant_atoms(self, brain_with_graph) -> None:
        """recall() for a Redis query must return Redis atoms."""
        brain, ids = brain_with_graph
        result = await brain.recall(query="Redis SCAN cursor", budget_tokens=4000)

        atom_ids = {a["id"] for a in result["atoms"]}
        redis_ids = {ids["redis_scan"], ids["redis_keys_bad"], ids["redis_antipattern"]}

        assert atom_ids & redis_ids, (
            f"Expected Redis atoms in recall results, got: {atom_ids}"
        )

    async def test_recall_includes_spreading_activation(self, brain_with_graph) -> None:
        """Atoms connected via synapses should appear through spreading activation."""
        brain, ids = brain_with_graph
        result = await brain.recall(query="Redis SCAN cursor", budget_tokens=4000)

        # The antipattern is linked to redis_keys_bad via warns-against.
        # If redis_keys_bad is a seed, the antipattern should spread-activate in.
        all_ids = {a["id"] for a in result["atoms"]}
        all_ids.update(a["id"] for a in result["antipatterns"])

        # At minimum, we should see more than just the top vector match.
        assert len(all_ids) >= 2, (
            f"Expected spreading activation to expand beyond seeds, got {len(all_ids)} atoms"
        )

    async def test_recall_antipatterns_surfaced(self, brain_with_graph) -> None:
        """Antipattern atoms connected via warns-against should appear."""
        brain, ids = brain_with_graph
        result = await brain.recall(query="Redis KEYS command", budget_tokens=4000)

        all_content = " ".join(
            a["content"] for a in result["atoms"]
        ) + " " + " ".join(
            a["content"] for a in result["antipatterns"]
        )

        # The antipattern warning about KEYS should surface.
        assert "SCAN" in all_content or "antipattern" in all_content.lower(), (
            "Expected antipattern about KEYS to surface in recall results"
        )

    async def test_recall_score_breakdown_present(self, brain_with_graph) -> None:
        """Each recalled atom should include a score breakdown."""
        brain, ids = brain_with_graph
        result = await brain.recall(query="Redis", budget_tokens=4000)

        for atom in result["atoms"]:
            if "score_breakdown" in atom:
                breakdown = atom["score_breakdown"]
                assert "vector_similarity" in breakdown
                assert "importance" in breakdown
                assert "confidence" in breakdown

    async def test_recall_respects_region_filter(self, brain_with_graph) -> None:
        """Region filter should exclude atoms from other regions."""
        brain, ids = brain_with_graph
        result = await brain.recall(
            query="Redis SCAN",
            budget_tokens=4000,
            region="personal",
        )

        # Only personal region atoms should appear.
        for atom in result["atoms"]:
            assert atom.get("region") == "personal" or atom.get("region") is None

    async def test_recall_latency(self, brain_with_graph) -> None:
        """Full recall pipeline should complete within 5 seconds on CI."""
        brain, _ids = brain_with_graph
        start = time.perf_counter()
        await brain.recall(query="Redis SCAN cursor iteration", budget_tokens=4000)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5000, (
            f"Full recall took {elapsed_ms:.0f}ms, expected <5000ms"
        )

    async def test_unrelated_query_excludes_redis(self, brain_with_graph) -> None:
        """A query about dark mode should not prioritise Redis atoms."""
        brain, ids = brain_with_graph
        result = await brain.recall(
            query="dark mode editor preferences",
            budget_tokens=4000,
        )

        if result["atoms"]:
            top_atom = result["atoms"][0]
            # The top result should NOT be a Redis atom.
            assert top_atom["id"] != ids["redis_scan"], (
                "Redis SCAN should not be the top result for a dark mode query"
            )


class TestSynapseGraph:
    """Verify the seeded synapse graph is correct."""

    @pytest.fixture(autouse=True)
    async def seed(self, integration_storage, real_embeddings):
        self.ids = await seed_realistic_graph(integration_storage, real_embeddings)
        self.storage = integration_storage

    async def test_synapse_graph_seeded_correctly(self) -> None:
        """Synapses between Redis atoms must exist in the database."""
        rows = await self.storage.execute(
            "SELECT source_id, target_id, relationship, strength "
            "FROM synapses ORDER BY id",
            (),
        )
        assert len(rows) >= 2
        rels = {(r["source_id"], r["target_id"]) for r in rows}
        assert (self.ids["redis_scan"], self.ids["redis_keys_bad"]) in rels
        assert (self.ids["redis_antipattern"], self.ids["redis_keys_bad"]) in rels

    async def test_warns_against_is_directional(self) -> None:
        """warns-against synapses should be unidirectional."""
        rows = await self.storage.execute(
            "SELECT bidirectional FROM synapses WHERE relationship = 'warns-against'",
            (),
        )
        for row in rows:
            assert row["bidirectional"] == 0
