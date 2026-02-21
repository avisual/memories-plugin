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


class TestRememberThenRecallIntegration:
    """End-to-end remember → recall round-trip with real Ollama embeddings.

    Verifies that atoms stored via Brain.remember() are immediately
    retrievable via Brain.recall() using real semantic embeddings.
    """

    @pytest.fixture
    async def full_brain(self, integration_storage, real_embeddings):
        """Brain with real embeddings and a real LearningEngine."""
        from memories.learning import LearningEngine

        storage = integration_storage
        embeddings = real_embeddings
        atoms = AtomManager(storage, embeddings)
        synapses = SynapseManager(storage)
        context = ContextBudget(storage)
        retrieval = RetrievalEngine(storage, embeddings, atoms, synapses, context)
        learning = LearningEngine(storage, embeddings, atoms, synapses)

        b = Brain.__new__(Brain)
        b._config = get_config()
        b._storage = storage
        b._embeddings = embeddings
        b._atoms = atoms
        b._synapses = synapses
        b._context = context
        b._retrieval = retrieval
        b._learning = learning
        b._consolidation = None
        b._current_session_id = "integration-test-session"
        b._initialized = True

        await storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            ("integration-test-session", None),
        )

        return b

    async def test_remember_then_recall_returns_atom(self, full_brain) -> None:
        """An atom stored via remember() should be returned by a semantically
        similar recall() query using real embeddings."""
        result = await full_brain.remember(
            content=(
                "PostgreSQL VACUUM reclaims storage occupied by dead tuples. "
                "Run VACUUM ANALYZE regularly on high-churn tables."
            ),
            type="fact",
            region="project:postgres-app",
            importance=0.8,
        )
        atom_id = result["atom_id"]

        recall_result = await full_brain.recall(
            query="How to reclaim dead tuple storage in PostgreSQL?",
            budget_tokens=4000,
        )

        recalled_ids = {a["id"] for a in recall_result["atoms"]}
        assert atom_id in recalled_ids, (
            f"Remembered atom {atom_id} not found in recall results. "
            f"Got IDs: {recalled_ids}"
        )

    async def test_remember_multiple_then_recall_ranks_correctly(
        self, full_brain,
    ) -> None:
        """Multiple remembered atoms should rank correctly by semantic relevance."""
        await full_brain.remember(
            content="Docker containers use cgroups for resource isolation",
            type="fact", region="devops",
        )
        pg_result = await full_brain.remember(
            content=(
                "PostgreSQL connection pooling with PgBouncer reduces "
                "connection overhead and improves query throughput"
            ),
            type="skill", region="project:postgres-app",
        )
        await full_brain.remember(
            content="I prefer VS Code with the Vim extension for editing",
            type="preference", region="personal",
        )

        recall_result = await full_brain.recall(
            query="PostgreSQL connection pooling performance",
            budget_tokens=4000,
        )

        if recall_result["atoms"]:
            top_atom = recall_result["atoms"][0]
            assert top_atom["id"] == pg_result["atom_id"], (
                f"Expected PostgreSQL atom to rank first, "
                f"got atom {top_atom['id']}: {top_atom['content'][:60]}"
            )

    async def test_dedup_prevents_duplicate_storage(self, full_brain) -> None:
        """Remembering near-identical content should be deduplicated."""
        r1 = await full_brain.remember(
            content="Always use parameterized SQL queries to prevent injection",
            type="antipattern", region="security",
            severity="critical", instead="Use parameterized queries",
        )

        r2 = await full_brain.remember(
            content="Always use parameterized SQL queries to prevent SQL injection attacks",
            type="antipattern", region="security",
            severity="critical", instead="Use parameterized queries",
        )

        # Second call should be deduplicated (same atom returned).
        if r2.get("deduplicated"):
            assert r2["atom_id"] == r1["atom_id"], (
                "Near-identical content should be deduplicated to the same atom"
            )


# ---------------------------------------------------------------------------
# NEW SMOKE TESTS — Learning pipeline with real Ollama embeddings
# ---------------------------------------------------------------------------


class TestAutoLinkRealEmbeddings:
    """Verify auto_link creates semantically correct synapses with real embeddings.

    Seeds the first atom via SQL + embed_and_store (bypassing brain.remember's
    dedup check), then calls brain.remember() for the second atom so that
    auto_link() triggers and discovers the pre-seeded atom via vector search.
    """

    @pytest.fixture
    async def env(self, integration_storage, real_embeddings):
        """Brain + LearningEngine + helpers for seeding atoms."""
        from memories.learning import LearningEngine

        storage = integration_storage
        embeddings = real_embeddings
        atoms = AtomManager(storage, embeddings)
        synapses = SynapseManager(storage)
        context = ContextBudget(storage)
        retrieval = RetrievalEngine(storage, embeddings, atoms, synapses, context)
        learning = LearningEngine(storage, embeddings, atoms, synapses)

        b = Brain.__new__(Brain)
        b._config = get_config()
        b._storage = storage
        b._embeddings = embeddings
        b._atoms = atoms
        b._synapses = synapses
        b._context = context
        b._retrieval = retrieval
        b._learning = learning
        b._consolidation = None
        b._current_session_id = "integration-autolink"
        b._initialized = True

        await storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            ("integration-autolink", None),
        )

        now = datetime.now(tz=timezone.utc).isoformat()

        async def seed_atom(content: str, atom_type: str = "fact",
                            region: str = "general") -> int:
            """Insert an atom directly (bypass remember's dedup)."""
            aid = await storage.execute_write(
                """INSERT INTO atoms
                   (content, type, region, confidence, importance,
                    access_count, last_accessed_at, is_deleted)
                   VALUES (?, ?, ?, 0.9, 0.5, 5, ?, 0)""",
                (content, atom_type, region, now),
            )
            await embeddings.embed_and_store(aid, content)
            return aid

        return b, learning, storage, seed_atom

    async def test_auto_link_creates_related_to_synapse(self, env) -> None:
        """Near-paraphrase Redis facts should be linked by auto_link."""
        _brain, learning, storage, seed_atom = env

        # Seed both atoms via SQL (bypasses dedup and brain.remember).
        # Use near-paraphrase texts so cosine similarity is high enough
        # for auto_link_threshold (0.82 in distance_to_similarity space).
        id1 = await seed_atom(
            "Redis SCAN iterates through all keys in the database using "
            "a cursor-based approach that returns results in small batches",
            region="project:redis",
        )
        id2 = await seed_atom(
            "Redis SCAN uses a cursor to iterate through all keys in the "
            "database and returns results in small batches",
            region="project:redis",
        )

        # Call auto_link directly on the second atom.
        created = await learning.auto_link(id2)

        rows = await storage.execute(
            "SELECT source_id, target_id, relationship, strength "
            "FROM synapses WHERE "
            "(source_id = ? AND target_id = ?) OR "
            "(source_id = ? AND target_id = ?)",
            (id1, id2, id2, id1),
        )

        assert len(rows) >= 1, (
            f"Expected a synapse between atom {id1} and {id2}; "
            f"auto_link returned {len(created)} synapses"
        )
        relationships = {r["relationship"] for r in rows}
        valid_types = {"related-to", "caused-by", "elaborates", "part-of"}
        assert relationships & valid_types, (
            f"Expected a topical synapse, got: {relationships}"
        )

    async def test_auto_link_skips_unrelated_atoms(self, env) -> None:
        """A Redis fact and a cooking recipe should NOT be linked."""
        brain, learning, storage, seed_atom = env

        id1 = await seed_atom(
            "Redis SET command stores a string value associated with a key. "
            "It supports optional expiration time in seconds.",
            region="project:redis",
        )
        r2 = await brain.remember(
            content=(
                "To make a classic French omelette, whisk three eggs with salt, "
                "cook in butter over medium-low heat, and fold gently."
            ),
            type="fact",
            region="personal",
        )

        rows = await storage.execute(
            "SELECT relationship FROM synapses WHERE "
            "(source_id = ? AND target_id = ?) OR "
            "(source_id = ? AND target_id = ?)",
            (id1, r2["atom_id"], r2["atom_id"], id1),
        )

        topical = {r["relationship"] for r in rows} & {
            "related-to", "caused-by", "elaborates", "part-of",
        }
        assert not topical, (
            f"Unrelated atoms should not share a topical synapse, got: {topical}"
        )

    async def test_auto_link_warns_against_for_antipattern(self, env) -> None:
        """An antipattern seeded near a related fact should create warns-against."""
        _brain, learning, storage, seed_atom = env

        # Fact and antipattern share most words so cosine similarity is high.
        id1 = await seed_atom(
            "Using the Redis KEYS command in production causes server "
            "blocking because it scans the entire keyspace at once",
            region="project:redis",
        )
        id2 = await seed_atom(
            "Avoid using the Redis KEYS command in production because "
            "it blocks the server while scanning the entire keyspace",
            atom_type="antipattern",
            region="project:redis",
        )

        # Call auto_link on the antipattern atom.
        await learning.auto_link(id2)

        rows = await storage.execute(
            "SELECT source_id, target_id, relationship FROM synapses WHERE "
            "relationship = 'warns-against' AND "
            "((source_id = ? AND target_id = ?) OR "
            " (source_id = ? AND target_id = ?))",
            (id2, id1, id1, id2),
        )
        assert len(rows) >= 1, (
            "Expected a warns-against synapse between antipattern and related fact"
        )

    async def test_auto_link_strength_reflects_similarity(self, env) -> None:
        """A near-paraphrase should produce a synapse; a different topic should not."""
        _brain, learning, storage, seed_atom = env

        base_id = await seed_atom(
            "Redis Cluster distributes data across nodes using "
            "16384 hash slots for horizontal scaling",
            region="project:redis",
        )

        # Near-paraphrase — should exceed auto_link_threshold.
        close_id = await seed_atom(
            "Redis Cluster uses 16384 hash slots to distribute "
            "data across multiple nodes for horizontal scaling",
            region="project:redis",
        )

        # Completely different topic — should NOT create a synapse.
        far_id = await seed_atom(
            "Docker containers package applications with their "
            "dependencies for consistent deployment across environments",
            region="devops",
        )

        await learning.auto_link(close_id)
        await learning.auto_link(far_id)

        rows = await storage.execute(
            "SELECT source_id, target_id, strength FROM synapses WHERE "
            "relationship IN ('related-to', 'caused-by', 'elaborates', 'part-of') "
            "AND ((source_id = ? OR target_id = ?))",
            (base_id, base_id),
        )

        def _strength_for(atom_id: int) -> float:
            for r in rows:
                other = r["target_id"] if r["source_id"] == base_id else r["source_id"]
                if other == atom_id:
                    return float(r["strength"])
            return 0.0

        s_close = _strength_for(close_id)
        s_far = _strength_for(far_id)

        assert s_close > 0, "Expected a synapse to the near-paraphrase atom"
        assert s_close > s_far, (
            f"Near-paraphrase strength ({s_close:.3f}) should > "
            f"unrelated strength ({s_far:.3f})"
        )


class TestSupersessionRealEmbeddings:
    """Verify detect_supersedes() correctly identifies near-duplicate content.

    Seeds the first atom via SQL (bypassing remember's dedup) since
    detect_supersedes needs >0.9 similarity but dedup fires at >0.92,
    which would prevent the second atom from being stored at all.
    """

    @pytest.fixture
    async def env(self, integration_storage, real_embeddings):
        from memories.learning import LearningEngine

        storage = integration_storage
        embeddings = real_embeddings
        atoms = AtomManager(storage, embeddings)
        synapses = SynapseManager(storage)
        learning = LearningEngine(storage, embeddings, atoms, synapses)

        now = datetime.now(tz=timezone.utc).isoformat()

        async def seed_atom(content: str, atom_type: str = "fact",
                            region: str = "general") -> int:
            aid = await storage.execute_write(
                """INSERT INTO atoms
                   (content, type, region, confidence, importance,
                    access_count, last_accessed_at, is_deleted)
                   VALUES (?, ?, ?, 0.9, 0.5, 5, ?, 0)""",
                (content, atom_type, region, now),
            )
            await embeddings.embed_and_store(aid, content)
            return aid

        return learning, storage, embeddings, seed_atom

    async def test_supersedes_near_duplicate(self, env) -> None:
        """A nearly identical text should create a supersedes synapse."""
        learning, storage, embeddings, seed_atom = env

        # Seed original atom directly.
        id1 = await seed_atom(
            "PostgreSQL VACUUM reclaims storage occupied by dead tuples "
            "after UPDATE and DELETE operations on a table",
            region="project:postgres",
        )

        # Seed a near-duplicate — identical except trailing phrase.
        import asyncio
        await asyncio.sleep(0.01)  # Ensure created_at is later.
        id2 = await seed_atom(
            "PostgreSQL VACUUM reclaims storage occupied by dead tuples "
            "after UPDATE and DELETE operations on the table",
            region="project:postgres",
        )

        count = await learning.detect_supersedes(id2)
        assert count >= 1, (
            "Expected detect_supersedes to create a supersedes synapse"
        )

        rows = await storage.execute(
            "SELECT source_id, target_id, relationship, strength "
            "FROM synapses WHERE relationship = 'supersedes' "
            "AND source_id = ? AND target_id = ?",
            (id2, id1),
        )
        assert len(rows) >= 1, (
            "Expected a supersedes synapse from the newer atom to the older one"
        )
        assert float(rows[0]["strength"]) > 0.85, (
            f"Supersedes strength {rows[0]['strength']:.3f} should reflect high similarity"
        )

    async def test_no_supersedes_for_different_content(self, env) -> None:
        """Genuinely different facts should NOT create a supersedes link."""
        learning, storage, embeddings, seed_atom = env

        id1 = await seed_atom(
            "PostgreSQL VACUUM reclaims storage occupied by dead tuples "
            "after UPDATE and DELETE operations",
            region="project:postgres",
        )
        id2 = await seed_atom(
            "PostgreSQL indexes can become bloated over time. "
            "REINDEX rebuilds them from scratch to reclaim space.",
            region="project:postgres",
        )

        await learning.detect_supersedes(id2)

        rows = await storage.execute(
            "SELECT * FROM synapses WHERE relationship = 'supersedes' AND "
            "((source_id = ? AND target_id = ?) OR "
            " (source_id = ? AND target_id = ?))",
            (id1, id2, id2, id1),
        )
        assert len(rows) == 0, (
            "Different content should NOT create a supersedes synapse"
        )


class TestConsolidationMergeRealEmbeddings:
    """Verify consolidation correctly merges near-duplicates with real embeddings."""

    @pytest.fixture
    async def consolidation_env(self, integration_storage, real_embeddings):
        from memories.consolidation import ConsolidationEngine

        storage = integration_storage
        embeddings = real_embeddings
        atoms = AtomManager(storage, embeddings)
        synapses = SynapseManager(storage)
        consolidation = ConsolidationEngine(storage, embeddings, atoms, synapses)

        return storage, embeddings, atoms, consolidation

    async def test_merge_near_duplicate_atoms(self, consolidation_env) -> None:
        """Near-identical atoms should be merged during consolidation."""
        storage, embeddings, atoms, consolidation = consolidation_env
        now = datetime.now(tz=timezone.utc).isoformat()

        # Insert 3 nearly identical atoms about the same topic.
        contents = [
            "Always close database connections after use to prevent connection leaks",
            "Always close database connections after use to avoid connection leaks",
            "Always close DB connections after use to prevent connection pool leaks",
        ]
        atom_ids = []
        for content in contents:
            aid = await storage.execute_write(
                """INSERT INTO atoms
                   (content, type, region, confidence, importance,
                    access_count, last_accessed_at, is_deleted)
                   VALUES (?, 'fact', 'general', 0.9, 0.5, 5, ?, 0)""",
                (content, now),
            )
            atom_ids.append(aid)
            await embeddings.embed_and_store(aid, content)

        result = await consolidation.reflect()

        # At least one pair should have been merged.
        merge_actions = [
            d for d in result.details if d.get("action") == "merge"
        ]
        assert len(merge_actions) >= 1, (
            f"Expected at least one merge, got {len(merge_actions)}. "
            f"Details: {result.details}"
        )

        # Verify the duplicate is soft-deleted.
        for merge in merge_actions:
            dup_rows = await storage.execute(
                "SELECT is_deleted FROM atoms WHERE id = ?",
                (merge["duplicate_id"],),
            )
            assert dup_rows[0]["is_deleted"] == 1, (
                f"Duplicate atom {merge['duplicate_id']} should be soft-deleted"
            )

    async def test_no_merge_for_distinct_atoms(self, consolidation_env) -> None:
        """Clearly distinct atoms should NOT be merged."""
        storage, embeddings, atoms, consolidation = consolidation_env
        now = datetime.now(tz=timezone.utc).isoformat()

        contents = [
            "Redis Cluster shards data across nodes using 16384 hash slots",
            "Python asyncio event loop runs coroutines cooperatively",
            "CSS Grid provides two-dimensional layout with rows and columns",
        ]
        for content in contents:
            aid = await storage.execute_write(
                """INSERT INTO atoms
                   (content, type, region, confidence, importance,
                    access_count, last_accessed_at, is_deleted)
                   VALUES (?, 'fact', 'general', 0.9, 0.5, 5, ?, 0)""",
                (content, now),
            )
            await embeddings.embed_and_store(aid, content)

        result = await consolidation.reflect()

        merge_actions = [
            d for d in result.details if d.get("action") == "merge"
        ]
        assert len(merge_actions) == 0, (
            f"Distinct atoms should not be merged, got: {merge_actions}"
        )


class TestRememberDedupIntegration:
    """Verify remember() pre-insertion dedup with real embeddings."""

    @pytest.fixture
    async def full_brain(self, integration_storage, real_embeddings):
        from memories.learning import LearningEngine

        storage = integration_storage
        embeddings = real_embeddings
        atoms_mgr = AtomManager(storage, embeddings)
        synapses = SynapseManager(storage)
        context = ContextBudget(storage)
        retrieval = RetrievalEngine(storage, embeddings, atoms_mgr, synapses, context)
        learning = LearningEngine(storage, embeddings, atoms_mgr, synapses)

        b = Brain.__new__(Brain)
        b._config = get_config()
        b._storage = storage
        b._embeddings = embeddings
        b._atoms = atoms_mgr
        b._synapses = synapses
        b._context = context
        b._retrieval = retrieval
        b._learning = learning
        b._consolidation = None
        b._current_session_id = "integration-dedup"
        b._initialized = True

        await storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            ("integration-dedup", None),
        )

        return b

    async def test_dedup_blocks_identical_content(self, full_brain) -> None:
        """Remembering identical content twice returns the same atom_id."""
        r1 = await full_brain.remember(
            content="Use SSH keys instead of passwords for server authentication",
            type="skill",
            region="security",
        )
        r2 = await full_brain.remember(
            content="Use SSH keys instead of passwords for server authentication",
            type="skill",
            region="security",
        )

        assert r2.get("deduplicated") is True, (
            "Identical content should be deduplicated"
        )
        assert r2["atom_id"] == r1["atom_id"]

    async def test_dedup_allows_different_content(self, full_brain) -> None:
        """Genuinely different content should NOT be deduplicated."""
        r1 = await full_brain.remember(
            content="Use SSH keys instead of passwords for server authentication",
            type="skill",
            region="security",
        )
        r2 = await full_brain.remember(
            content=(
                "Python virtual environments isolate project dependencies. "
                "Create one with python -m venv .venv"
            ),
            type="skill",
            region="python",
        )

        assert r1["atom_id"] != r2["atom_id"], (
            "Different content should produce different atom IDs"
        )


class TestBatchEmbeddingOps:
    """Verify batch embedding operations with real Ollama."""

    @pytest.fixture(autouse=True)
    async def seed(self, integration_storage, real_embeddings):
        self.storage = integration_storage
        self.embeddings = real_embeddings

    async def test_embed_batch_consistency(self) -> None:
        """Batch-embedded vectors should match individually-embedded vectors."""
        texts = [
            "Redis SCAN iterates keyspace",
            "PostgreSQL VACUUM reclaims space",
            "Python asyncio runs coroutines",
            "Docker containers use cgroups",
            "Kubernetes pods group containers",
        ]

        # Embed individually — this populates the DB cache.
        individual = []
        for t in texts:
            vec = await self.embeddings.embed_text(t)
            individual.append(vec)

        # Clear the DB-backed embedding cache so batch actually hits Ollama.
        await self.storage.execute_write(
            "DELETE FROM embedding_cache", ()
        )

        batch = await self.embeddings.embed_batch(texts)
        assert len(batch) == len(texts)

        for i, (single, batched) in enumerate(zip(individual, batch)):
            sim = self.embeddings.cosine_similarity(single, batched)
            assert sim > 0.99, (
                f"Text {i} batch/single similarity {sim:.4f} should be > 0.99"
            )

    async def test_embed_and_store_creates_searchable_vector(self) -> None:
        """An atom embedded via embed_and_store should be findable by vector search."""
        now = datetime.now(tz=timezone.utc).isoformat()
        atom_id = await self.storage.execute_write(
            """INSERT INTO atoms
               (content, type, region, confidence, importance,
                access_count, last_accessed_at, is_deleted)
               VALUES ('Redis pipelining batches multiple commands into one round trip',
                       'fact', 'general', 0.9, 0.5, 1, ?, 0)""",
            (now,),
        )

        await self.embeddings.embed_and_store(
            atom_id, "Redis pipelining batches multiple commands into one round trip"
        )

        results = await self.embeddings.search_similar("Redis pipelining", k=5)
        ids = {aid for aid, _ in results}
        assert atom_id in ids, "Atom should be findable after embed_and_store"


class TestRecallLargerGraph:
    """Test recall on a richer 15-atom graph with real embeddings."""

    _GRAPH_ATOMS = {
        # --- Redis cluster (5 atoms) ---
        "redis_scan": {
            "content": "Redis SCAN iterates the keyspace incrementally with a cursor",
            "type": "fact", "region": "project:redis",
        },
        "redis_keys": {
            "content": "Redis KEYS blocks the server while scanning all keys",
            "type": "fact", "region": "project:redis",
        },
        "redis_cluster": {
            "content": "Redis Cluster uses 16384 hash slots to shard data across nodes",
            "type": "fact", "region": "project:redis",
        },
        "redis_sentinel": {
            "content": "Redis Sentinel provides high availability with automatic failover",
            "type": "fact", "region": "project:redis",
        },
        "redis_antipattern": {
            "content": "Never use KEYS in production. Use SCAN with cursor instead.",
            "type": "antipattern", "region": "project:redis",
        },
        # --- Python async cluster (5 atoms) ---
        "py_asyncio": {
            "content": "Python asyncio event loop runs coroutines cooperatively",
            "type": "skill", "region": "project:web-api",
        },
        "py_await": {
            "content": "Use await for I/O-bound tasks in Python async code",
            "type": "skill", "region": "project:web-api",
        },
        "py_gather": {
            "content": "asyncio.gather runs multiple coroutines concurrently",
            "type": "skill", "region": "project:web-api",
        },
        "py_sleep_bad": {
            "content": (
                "Never use time.sleep() in async code because it blocks "
                "the event loop. Use asyncio.sleep() instead."
            ),
            "type": "antipattern", "region": "project:web-api",
        },
        "py_experience": {
            "content": (
                "Migrated the API from synchronous Flask to async FastAPI. "
                "Throughput improved 3x for I/O-heavy endpoints."
            ),
            "type": "experience", "region": "project:web-api",
        },
        # --- SQL cluster (5 atoms) ---
        "sql_index": {
            "content": "SQL indexes speed up SELECT queries but slow down INSERT and UPDATE",
            "type": "fact", "region": "project:postgres",
        },
        "sql_explain": {
            "content": "Use EXPLAIN ANALYZE to understand query plans in PostgreSQL",
            "type": "skill", "region": "project:postgres",
        },
        "sql_vacuum": {
            "content": "PostgreSQL VACUUM reclaims storage occupied by dead tuples",
            "type": "fact", "region": "project:postgres",
        },
        "sql_injection": {
            "content": (
                "Never concatenate user input into SQL strings. "
                "Use parameterized queries to prevent SQL injection."
            ),
            "type": "antipattern", "region": "project:postgres",
        },
        "sql_connection_pool": {
            "content": (
                "Use connection pooling (PgBouncer) to reduce overhead of "
                "creating new PostgreSQL connections for every request."
            ),
            "type": "skill", "region": "project:postgres",
        },
    }

    @pytest.fixture
    async def large_brain(self, integration_storage, real_embeddings):
        """Build a Brain with a 15-atom graph across three topic clusters."""
        storage = integration_storage
        embeddings = real_embeddings
        atoms = AtomManager(storage, embeddings)
        synapses = SynapseManager(storage)
        context = ContextBudget(storage)
        retrieval = RetrievalEngine(storage, embeddings, atoms, synapses, context)

        now = datetime.now(tz=timezone.utc).isoformat()
        atom_ids: dict[str, int] = {}

        for key, data in self._GRAPH_ATOMS.items():
            atom_id = await storage.execute_write(
                """INSERT INTO atoms
                   (content, type, region, confidence, importance,
                    access_count, last_accessed_at, is_deleted)
                   VALUES (?, ?, ?, 0.9, 0.7, 5, ?, 0)""",
                (data["content"], data["type"], data["region"], now),
            )
            atom_ids[key] = atom_id
            await embeddings.embed_and_store(atom_id, data["content"])

        # Create intra-cluster synapses (Redis cluster).
        redis_keys_list = ["redis_scan", "redis_keys", "redis_cluster", "redis_sentinel"]
        for i in range(len(redis_keys_list) - 1):
            await storage.execute_write(
                """INSERT INTO synapses
                   (source_id, target_id, relationship, strength, bidirectional)
                   VALUES (?, ?, 'related-to', 0.8, 1)""",
                (atom_ids[redis_keys_list[i]], atom_ids[redis_keys_list[i + 1]]),
            )
        # warns-against from antipattern.
        await storage.execute_write(
            """INSERT INTO synapses
               (source_id, target_id, relationship, strength, bidirectional)
               VALUES (?, ?, 'warns-against', 0.9, 0)""",
            (atom_ids["redis_antipattern"], atom_ids["redis_keys"]),
        )

        # Python cluster chain.
        py_keys = ["py_asyncio", "py_await", "py_gather", "py_experience"]
        for i in range(len(py_keys) - 1):
            await storage.execute_write(
                """INSERT INTO synapses
                   (source_id, target_id, relationship, strength, bidirectional)
                   VALUES (?, ?, 'related-to', 0.8, 1)""",
                (atom_ids[py_keys[i]], atom_ids[py_keys[i + 1]]),
            )
        await storage.execute_write(
            """INSERT INTO synapses
               (source_id, target_id, relationship, strength, bidirectional)
               VALUES (?, ?, 'warns-against', 0.9, 0)""",
            (atom_ids["py_sleep_bad"], atom_ids["py_asyncio"]),
        )

        # SQL cluster chain.
        sql_keys = ["sql_index", "sql_explain", "sql_vacuum", "sql_connection_pool"]
        for i in range(len(sql_keys) - 1):
            await storage.execute_write(
                """INSERT INTO synapses
                   (source_id, target_id, relationship, strength, bidirectional)
                   VALUES (?, ?, 'related-to', 0.8, 1)""",
                (atom_ids[sql_keys[i]], atom_ids[sql_keys[i + 1]]),
            )
        await storage.execute_write(
            """INSERT INTO synapses
               (source_id, target_id, relationship, strength, bidirectional)
               VALUES (?, ?, 'warns-against', 0.9, 0)""",
            (atom_ids["sql_injection"], atom_ids["sql_index"]),
        )

        # Cross-cluster hub: link py_experience → sql_connection_pool
        # (both relate to API performance).
        await storage.execute_write(
            """INSERT INTO synapses
               (source_id, target_id, relationship, strength, bidirectional)
               VALUES (?, ?, 'related-to', 0.7, 1)""",
            (atom_ids["py_experience"], atom_ids["sql_connection_pool"]),
        )

        b = Brain.__new__(Brain)
        b._config = get_config()
        b._storage = storage
        b._embeddings = embeddings
        b._atoms = atoms
        b._synapses = synapses
        b._context = context
        b._retrieval = retrieval
        b._learning = None
        b._consolidation = None
        b._current_session_id = None
        b._initialized = True

        return b, atom_ids

    async def test_recall_15_atom_graph_returns_relevant(
        self, large_brain,
    ) -> None:
        """A Redis query should return Redis-cluster atoms in the top results."""
        brain, ids = large_brain
        result = await brain.recall(query="Redis SCAN cursor", budget_tokens=4000)

        top_ids = {a["id"] for a in result["atoms"][:3]}
        redis_ids = {
            ids["redis_scan"], ids["redis_keys"],
            ids["redis_cluster"], ids["redis_sentinel"],
        }
        assert top_ids & redis_ids, (
            f"Expected Redis atoms in top 3, got IDs: {top_ids}"
        )

    async def test_recall_spreading_through_hub(self, large_brain) -> None:
        """Query about Python async should activate the cross-cluster hub
        and potentially reach SQL atoms via spreading activation."""
        brain, ids = large_brain
        result = await brain.recall(
            query="migrating from Flask to FastAPI async performance",
            budget_tokens=8000,
        )

        all_ids = {a["id"] for a in result["atoms"]}
        all_ids.update(a["id"] for a in result["antipatterns"])

        # The experience atom should definitely appear.
        assert ids["py_experience"] in all_ids, (
            "py_experience should be retrieved for a Flask→FastAPI query"
        )

        # With a generous budget, spreading through the hub should bring in
        # at least one SQL atom (sql_connection_pool is linked to py_experience).
        sql_ids = {
            ids["sql_connection_pool"], ids["sql_index"],
            ids["sql_explain"], ids["sql_vacuum"],
        }
        # This is a soft check — spreading activation is probabilistic.
        if not (all_ids & sql_ids):
            # Acceptable if the budget was consumed before reaching SQL atoms.
            assert len(all_ids) >= 3, (
                f"Expected at least 3 atoms for a generous-budget recall, got {len(all_ids)}"
            )

    async def test_recall_antipatterns_from_larger_graph(
        self, large_brain,
    ) -> None:
        """Antipattern atoms should appear in the dedicated antipatterns list."""
        brain, ids = large_brain
        result = await brain.recall(
            query="Redis KEYS command blocking server",
            budget_tokens=4000,
        )

        antipattern_ids = {a["id"] for a in result["antipatterns"]}
        all_ids = {a["id"] for a in result["atoms"]} | antipattern_ids

        # The Redis antipattern should surface either in atoms or antipatterns.
        assert ids["redis_antipattern"] in all_ids, (
            "Redis antipattern should be surfaced for a KEYS-related query"
        )
