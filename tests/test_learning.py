"""Comprehensive tests for the learning engine (auto-linking, antipatterns, Hebbian).

All tests use a temporary database and mock the EmbeddingEngine so that Ollama
is never required.  The mocked embedding layer returns predictable
``search_similar`` results that let us exercise:

- ``auto_link`` -- creating synapses for semantically similar atoms.
- ``detect_antipattern_links`` -- discovering warns-against connections.
- ``detect_supersedes`` -- handling near-duplicate atoms.
- ``suggest_region`` -- keyword and majority-vote region inference.
- ``assess_novelty`` -- identifying novel vs. duplicate content.
- ``session_end_learning`` -- Hebbian co-activation updates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from memories.atoms import Atom, AtomManager
from memories.embeddings import EmbeddingEngine
from memories.learning import (
    LearningEngine,
    _AUTO_LINK_SEARCH_K,
    _SUPERSEDE_SIMILARITY_THRESHOLD,
    _REGION_KEYWORDS,
)
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
    access_count: int = 0,
    last_accessed_at: str | None = None,
    tags: list[str] | None = None,
    is_deleted: bool = False,
    severity: str | None = None,
    instead: str | None = None,
    created_at: str | None = None,
) -> int:
    """Insert an atom directly via SQL, bypassing AtomManager (no embedding)."""
    now = datetime.now(tz=timezone.utc).isoformat()
    if last_accessed_at is None:
        last_accessed_at = now
    tags_json = json.dumps(tags) if tags else None

    atom_id = await storage.execute_write(
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

    # Override created_at if requested (for supersession testing).
    if created_at is not None:
        await storage.execute_write(
            "UPDATE atoms SET created_at = ? WHERE id = ?",
            (created_at, atom_id),
        )

    return atom_id


async def _insert_synapse(
    storage: Storage,
    source_id: int,
    target_id: int,
    relationship: str = "related-to",
    strength: float = 0.5,
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


async def _count_synapses(
    storage: Storage,
    source_id: int | None = None,
    target_id: int | None = None,
    relationship: str | None = None,
) -> int:
    """Count synapses matching the given filters."""
    clauses: list[str] = []
    params: list[Any] = []

    if source_id is not None:
        clauses.append("source_id = ?")
        params.append(source_id)
    if target_id is not None:
        clauses.append("target_id = ?")
        params.append(target_id)
    if relationship is not None:
        clauses.append("relationship = ?")
        params.append(relationship)

    where = " AND ".join(clauses) if clauses else "1=1"
    rows = await storage.execute(
        f"SELECT COUNT(*) AS cnt FROM synapses WHERE {where}",
        tuple(params),
    )
    return rows[0]["cnt"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Return a MagicMock standing in for EmbeddingEngine."""
    engine = MagicMock(spec=EmbeddingEngine)
    engine.search_similar = AsyncMock(return_value=[])
    engine.embed_text = AsyncMock(return_value=[0.0] * 768)
    return engine


@pytest.fixture
def learning_engine(
    storage: Storage,
    mock_embeddings: MagicMock,
) -> LearningEngine:
    """Build a LearningEngine wired to real Storage but mocked embeddings."""
    atoms = AtomManager(storage, mock_embeddings)
    synapses = SynapseManager(storage)
    return LearningEngine(storage, mock_embeddings, atoms, synapses)


@pytest.fixture
async def seeded_storage(storage: Storage) -> dict[str, Any]:
    """Insert a test graph of atoms into *storage*.

    Returns a dict with atom IDs and the storage instance::

        {
            "storage": Storage,
            "atom_a": int,  # fact: Redis SCAN usage
            "atom_b": int,  # fact: Redis KEYS command
            "atom_c": int,  # experience: Redis latency
            "atom_ap": int, # antipattern: never use KEYS
            "atom_d": int,  # preference: dark mode (unrelated)
        }
    """
    now = datetime.now(tz=timezone.utc)
    old_ts = (now - timedelta(days=10)).isoformat()

    atom_a = await _insert_atom(
        storage, "Redis SCAN is O(N) over the full keyspace",
        atom_type="fact", region="technical", confidence=0.95,
        access_count=10, tags=["redis", "performance"],
        created_at=old_ts,
    )
    atom_b = await _insert_atom(
        storage, "Redis KEYS command blocks the server",
        atom_type="fact", region="technical", confidence=0.9,
        access_count=8, tags=["redis"],
        created_at=old_ts,
    )
    atom_c = await _insert_atom(
        storage, "Experienced Redis latency spikes under high KEYS usage",
        atom_type="experience", region="technical", confidence=0.85,
        access_count=3, tags=["redis", "latency"],
        created_at=old_ts,
    )
    atom_ap = await _insert_atom(
        storage, "Never use KEYS in production workloads",
        atom_type="antipattern", region="technical", confidence=1.0,
        access_count=15, severity="high", instead="Use SCAN instead",
        created_at=old_ts,
    )
    atom_d = await _insert_atom(
        storage, "I prefer dark mode for all editors",
        atom_type="preference", region="personal", confidence=0.8,
        access_count=2,
        created_at=old_ts,
    )

    return {
        "storage": storage,
        "atom_a": atom_a,
        "atom_b": atom_b,
        "atom_c": atom_c,
        "atom_ap": atom_ap,
        "atom_d": atom_d,
    }


@pytest.fixture
def seeded_learning(
    seeded_storage: dict[str, Any],
    mock_embeddings: MagicMock,
) -> tuple[LearningEngine, dict[str, Any]]:
    """Return a LearningEngine backed by the seeded storage."""
    storage = seeded_storage["storage"]
    atoms = AtomManager(storage, mock_embeddings)
    synapses = SynapseManager(storage)
    engine = LearningEngine(storage, mock_embeddings, atoms, synapses)
    return engine, seeded_storage


# -----------------------------------------------------------------------
# 1. auto_link creates synapses for similar atoms
# -----------------------------------------------------------------------


class TestAutoLink:
    """Verify that auto_link discovers similar atoms and creates synapses."""

    async def test_creates_related_to_synapse(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """auto_link should create a 'related-to' synapse for similar atoms."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        # search_similar returns atom_b as similar to atom_a.
        # distance=0.3 -> similarity = 1 - 0.3/2 = 0.85, which > 0.82 threshold.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_b"], 0.3),
        ]

        result = await engine.auto_link(ids["atom_a"])

        assert len(result) >= 1
        relationships = [s["relationship"] for s in result]
        assert "related-to" in relationships

    async def test_skips_self_match(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """auto_link should skip the atom itself from search results."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.0),  # self-match
            (ids["atom_b"], 0.4),  # actual match
        ]

        result = await engine.auto_link(ids["atom_a"])

        # No synapse should link atom_a to itself.
        for synapse_dict in result:
            assert not (
                synapse_dict["source_id"] == ids["atom_a"]
                and synapse_dict["target_id"] == ids["atom_a"]
            )

    async def test_skips_below_threshold(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Candidates below the auto_link_threshold should not create synapses."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        # distance=1.4 -> similarity = 1 - 1.4/2 = 0.3, which < 0.6 threshold.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_b"], 1.4),
        ]

        result = await engine.auto_link(ids["atom_a"])

        assert result == []

    async def test_returns_empty_for_nonexistent_atom(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """auto_link for a non-existent atom ID should return an empty list."""
        result = await learning_engine.auto_link(99999)
        assert result == []

    async def test_creates_warns_against_for_antipattern_candidate(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """When a candidate is an antipattern, a 'warns-against' synapse should be created."""
        engine, ids = seeded_learning

        # atom_a auto-links and finds atom_ap (antipattern) as similar.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_ap"], 0.3),  # similarity = 0.85 > threshold
        ]

        result = await engine.auto_link(ids["atom_a"])

        relationships = [s["relationship"] for s in result]
        assert "warns-against" in relationships

    async def test_creates_warns_against_when_new_atom_is_antipattern(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """When the new atom itself is an antipattern and a candidate is not,
        a 'warns-against' synapse should point from the antipattern to the candidate."""
        engine, ids = seeded_learning

        mock_embeddings.search_similar.return_value = [
            (ids["atom_b"], 0.3),  # similarity = 0.85 > threshold
        ]

        result = await engine.auto_link(ids["atom_ap"])

        # There should be a warns-against from atom_ap -> atom_b.
        warns = [
            s for s in result
            if s["relationship"] == "warns-against"
            and s["source_id"] == ids["atom_ap"]
        ]
        assert len(warns) >= 1

    async def test_no_links_on_empty_search(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """When search_similar returns no results, no synapses are created."""
        engine, ids = seeded_learning
        mock_embeddings.search_similar.return_value = []

        result = await engine.auto_link(ids["atom_a"])

        assert result == []

    async def test_handles_embedding_runtime_error(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """When embedding fails with RuntimeError, auto_link returns empty."""
        engine, ids = seeded_learning
        mock_embeddings.search_similar.side_effect = RuntimeError("Ollama down")

        result = await engine.auto_link(ids["atom_a"])

        assert result == []


# -----------------------------------------------------------------------
# 2. detect_antipattern_links finds and links antipattern atoms
# -----------------------------------------------------------------------


class TestDetectAntipatternLinks:
    """Verify detection of warns-against links for antipattern atoms."""

    async def test_creates_warns_against_for_similar_antipattern(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Should create warns-against synapses when similar antipatterns exist."""
        engine, ids = seeded_learning

        # atom_b searches and finds atom_ap (antipattern) as similar.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_ap"], 0.3),  # similarity = 0.85 > threshold
        ]

        count = await engine.detect_antipattern_links(ids["atom_b"])

        assert count >= 1
        synapse_count = await _count_synapses(
            ids["storage"],
            source_id=ids["atom_ap"],
            target_id=ids["atom_b"],
            relationship="warns-against",
        )
        assert synapse_count >= 1

    async def test_skips_non_antipattern_candidates(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Non-antipattern candidates should not produce warns-against links."""
        engine, ids = seeded_learning

        # atom_a finds atom_b (both are facts, not antipatterns).
        mock_embeddings.search_similar.return_value = [
            (ids["atom_b"], 0.3),  # similarity = 0.85
        ]

        count = await engine.detect_antipattern_links(ids["atom_a"])

        assert count == 0

    async def test_antipattern_atom_skips_self(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """An antipattern atom should not try to link to antipatterns (itself)."""
        engine, ids = seeded_learning

        count = await engine.detect_antipattern_links(ids["atom_ap"])

        assert count == 0

    async def test_returns_zero_for_nonexistent_atom(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Non-existent atom should return 0."""
        count = await learning_engine.detect_antipattern_links(99999)
        assert count == 0

    async def test_handles_embedding_error(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Embedding failure should return 0 without raising."""
        engine, ids = seeded_learning
        mock_embeddings.search_similar.side_effect = RuntimeError("down")

        count = await engine.detect_antipattern_links(ids["atom_b"])

        assert count == 0


# -----------------------------------------------------------------------
# 3. detect_supersedes handles near-duplicate atoms
# -----------------------------------------------------------------------


class TestDetectSupersedes:
    """Verify supersession detection for near-duplicate atoms."""

    async def test_creates_supersedes_synapse(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """A newer atom should supersede an older near-duplicate of the same type."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        # Insert a newer version of atom_a.
        new_atom = await _insert_atom(
            storage, "Redis SCAN is O(N) overall but cursor-based",
            atom_type="fact", region="technical", confidence=1.0,
        )

        # distance=0.1 -> similarity = 0.95 > 0.9 supersede threshold.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.1),
        ]

        count = await engine.detect_supersedes(new_atom)

        assert count >= 1
        synapse_count = await _count_synapses(
            storage,
            source_id=new_atom,
            target_id=ids["atom_a"],
            relationship="supersedes",
        )
        assert synapse_count >= 1

    async def test_reduces_superseded_confidence(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """The superseded atom's confidence should be reduced."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        # Get original confidence.
        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?",
            (ids["atom_a"],),
        )
        original_confidence = rows[0]["confidence"]

        new_atom = await _insert_atom(
            storage, "Redis SCAN is O(N) overall but cursor-based",
            atom_type="fact", region="technical", confidence=1.0,
        )

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.1),  # similarity = 0.95
        ]

        await engine.detect_supersedes(new_atom)

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?",
            (ids["atom_a"],),
        )
        new_confidence = rows[0]["confidence"]
        assert new_confidence < original_confidence

    async def test_skips_different_type(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Atoms of different types should not supersede each other."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        # atom_c is an 'experience', so even if very similar to atom_a ('fact'),
        # it should not supersede.
        new_atom = await _insert_atom(
            storage, "Similar to atom_a but different type",
            atom_type="experience", region="technical",
        )

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.1),  # very similar but different type
        ]

        count = await engine.detect_supersedes(new_atom)

        assert count == 0

    async def test_skips_below_similarity_threshold(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Candidates below the supersede similarity threshold are skipped."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        new_atom = await _insert_atom(
            storage, "Redis something different",
            atom_type="fact", region="technical",
        )

        # distance=0.4 -> similarity = 0.8 < 0.9 threshold.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.4),
        ]

        count = await engine.detect_supersedes(new_atom)

        assert count == 0

    async def test_skips_when_new_atom_is_older(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """A new atom with an older created_at should not supersede."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        # Create an atom that is timestamped in the past.
        old_ts = "2020-01-01T00:00:00+00:00"
        new_atom = await _insert_atom(
            storage, "Old Redis fact",
            atom_type="fact", region="technical",
            created_at=old_ts,
        )

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.1),  # very similar
        ]

        count = await engine.detect_supersedes(new_atom)

        assert count == 0

    async def test_returns_zero_for_nonexistent_atom(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Non-existent atom should return 0."""
        count = await learning_engine.detect_supersedes(99999)
        assert count == 0


# -----------------------------------------------------------------------
# 4. suggest_region returns correct regions based on keywords
# -----------------------------------------------------------------------


class TestSuggestRegion:
    """Verify region suggestion based on keywords and vector neighbours."""

    async def test_project_takes_priority(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """When source_project is given, region should be 'project:<name>'."""
        region = await learning_engine.suggest_region(
            "anything at all", source_project="myapp",
        )
        assert region == "project:myapp"

    async def test_keyword_match_errors(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Content containing error-related keywords should suggest 'errors'."""
        region = await learning_engine.suggest_region(
            "There was an error in the deployment pipeline",
        )
        assert region == "errors"

    async def test_keyword_match_personal(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Content with personal preference keywords should suggest 'personal'."""
        region = await learning_engine.suggest_region(
            "I always use Vim for editing",
        )
        assert region == "personal"

    async def test_keyword_match_decisions(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Content about decisions should suggest 'decisions'."""
        region = await learning_engine.suggest_region(
            "We decided to use PostgreSQL for the architecture",
        )
        assert region == "decisions"

    async def test_keyword_match_workflows(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Content about workflows should suggest 'workflows'."""
        region = await learning_engine.suggest_region(
            "The deployment pipeline steps are documented here",
        )
        assert region == "workflows"

    async def test_majority_vote_from_similar_atoms(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """When no keyword matches, suggest the majority region of neighbours."""
        engine, ids = seeded_learning

        # No keyword match in this content.
        content = "something about Redis performance tuning"

        # Similar atoms are all in 'technical' region.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.3),
            (ids["atom_b"], 0.4),
        ]

        region = await engine.suggest_region(content)

        assert region == "technical"

    async def test_default_to_technical(
        self,
        learning_engine: LearningEngine,
        mock_embeddings: MagicMock,
    ) -> None:
        """When nothing matches, default region should be 'technical'."""
        mock_embeddings.search_similar.return_value = []

        region = await learning_engine.suggest_region(
            "generic content with no keyword matches",
        )

        assert region == "technical"

    async def test_embedding_error_defaults_to_technical(
        self,
        learning_engine: LearningEngine,
        mock_embeddings: MagicMock,
    ) -> None:
        """Embedding failure should default to 'technical'."""
        mock_embeddings.search_similar.side_effect = RuntimeError("down")

        region = await learning_engine.suggest_region(
            "generic content without keywords",
        )

        assert region == "technical"


# -----------------------------------------------------------------------
# 5. assess_novelty correctly identifies novel vs duplicate content
# -----------------------------------------------------------------------


class TestAssessNovelty:
    """Verify that novelty assessment gates redundant content."""

    async def test_novel_when_no_existing_atoms(
        self,
        learning_engine: LearningEngine,
        mock_embeddings: MagicMock,
    ) -> None:
        """Content is novel when there are no existing atoms at all."""
        mock_embeddings.search_similar.return_value = []

        is_novel = await learning_engine.assess_novelty("brand new topic")

        assert is_novel is True

    async def test_novel_when_below_threshold(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Content is novel when the closest atom is below the similarity threshold."""
        engine, ids = seeded_learning

        # distance=0.8 -> similarity = 0.6, which is below default threshold 0.7
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.8),
        ]

        is_novel = await engine.assess_novelty("somewhat different content")

        assert is_novel is True

    async def test_not_novel_when_very_similar_and_confident(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Content is not novel when a very similar, high-confidence atom exists."""
        engine, ids = seeded_learning

        # distance=0.2 -> similarity = 0.9, above threshold 0.7.
        # atom_a has confidence 0.95 > 0.7.
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),
        ]

        is_novel = await engine.assess_novelty("Redis SCAN performance")

        assert is_novel is False

    async def test_novel_when_similar_but_low_confidence(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Content is novel when the closest atom has low confidence (< 0.7)."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        # Lower atom_a's confidence below 0.7.
        await storage.execute_write(
            "UPDATE atoms SET confidence = 0.5 WHERE id = ?",
            (ids["atom_a"],),
        )

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.2),  # similarity = 0.9 > threshold
        ]

        is_novel = await engine.assess_novelty("Redis SCAN info")

        assert is_novel is True

    async def test_empty_content_is_not_novel(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Empty or whitespace-only content should not be considered novel."""
        assert await learning_engine.assess_novelty("") is False
        assert await learning_engine.assess_novelty("   ") is False

    async def test_embedding_error_assumes_novel(
        self,
        learning_engine: LearningEngine,
        mock_embeddings: MagicMock,
    ) -> None:
        """When embedding fails, content should be treated as novel."""
        mock_embeddings.search_similar.side_effect = RuntimeError("down")

        is_novel = await learning_engine.assess_novelty("some content")

        assert is_novel is True

    async def test_custom_threshold(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """The threshold parameter should control the novelty boundary."""
        engine, ids = seeded_learning

        # distance=0.4 -> similarity = 0.8
        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.4),
        ]

        # With threshold=0.9, similarity 0.8 is below -> novel.
        is_novel = await engine.assess_novelty("Redis content", threshold=0.9)
        assert is_novel is True

        # With threshold=0.5, similarity 0.8 is above -> not novel (atom_a confidence = 0.95).
        is_novel = await engine.assess_novelty("Redis content", threshold=0.5)
        assert is_novel is False

    async def test_deleted_closest_atom_is_novel(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """If the closest atom is soft-deleted, content should be novel."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        await storage.execute_write(
            "UPDATE atoms SET is_deleted = 1 WHERE id = ?",
            (ids["atom_a"],),
        )

        mock_embeddings.search_similar.return_value = [
            (ids["atom_a"], 0.1),  # very similar but deleted
        ]

        # get_without_tracking returns None for deleted atoms, so novel.
        is_novel = await engine.assess_novelty("Redis SCAN content")
        assert is_novel is True


# -----------------------------------------------------------------------
# 6. session_end_learning applies Hebbian updates
# -----------------------------------------------------------------------


class TestSessionEndLearning:
    """Verify Hebbian strengthening during session-end learning."""

    async def test_strengthens_existing_synapses(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Co-activated atoms with existing synapses should get strengthened."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        # Create a synapse between atom_a and atom_b.
        await _insert_synapse(
            storage, ids["atom_a"], ids["atom_b"],
            "related-to", 0.5, True,
        )

        # Mark both as recently accessed.
        now = datetime.now(tz=timezone.utc).isoformat()
        await storage.execute_write(
            "UPDATE atoms SET last_accessed_at = ? WHERE id IN (?, ?)",
            (now, ids["atom_a"], ids["atom_b"]),
        )

        # Create a session.
        await storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            ("test-session-1", "test"),
        )

        stats = await engine.session_end_learning(
            "test-session-1", [ids["atom_a"], ids["atom_b"]],
        )

        assert stats["synapses_strengthened"] >= 1

    async def test_empty_atom_list(
        self,
        learning_engine: LearningEngine,
        storage: Storage,
    ) -> None:
        """session_end_learning with no atoms should return zero counts."""
        await storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            ("empty-session", "test"),
        )

        stats = await learning_engine.session_end_learning("empty-session", [])

        assert stats["synapses_strengthened"] == 0
        assert stats["synapses_created"] == 0

    async def test_single_atom_no_pairs(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """A single atom cannot form pairs, so no Hebbian update occurs."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        await storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            ("single-atom-session", "test"),
        )

        stats = await engine.session_end_learning(
            "single-atom-session", [ids["atom_a"]],
        )

        assert stats["synapses_strengthened"] == 0
        assert stats["synapses_created"] == 0

    async def test_records_atoms_on_session_row(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """The session row should have atoms_accessed updated."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        await storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            ("record-session", "test"),
        )

        atom_ids = [ids["atom_a"], ids["atom_b"]]
        await engine.session_end_learning("record-session", atom_ids)

        rows = await storage.execute(
            "SELECT atoms_accessed, ended_at FROM sessions WHERE id = ?",
            ("record-session",),
        )
        assert rows[0]["atoms_accessed"] is not None
        accessed = json.loads(rows[0]["atoms_accessed"])
        assert set(accessed) == set(atom_ids)
        # ended_at should be set.
        assert rows[0]["ended_at"] is not None

    async def test_deduplicates_atom_ids(
        self,
        seeded_learning: tuple[LearningEngine, dict[str, Any]],
        mock_embeddings: MagicMock,
    ) -> None:
        """Duplicate atom IDs in the input list should not cause problems."""
        engine, ids = seeded_learning
        storage = ids["storage"]

        await _insert_synapse(
            storage, ids["atom_a"], ids["atom_b"],
            "related-to", 0.5, True,
        )
        now = datetime.now(tz=timezone.utc).isoformat()
        await storage.execute_write(
            "UPDATE atoms SET last_accessed_at = ? WHERE id IN (?, ?)",
            (now, ids["atom_a"], ids["atom_b"]),
        )
        await storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            ("dedup-session", "test"),
        )

        # Pass duplicates -- should still work.
        stats = await engine.session_end_learning(
            "dedup-session",
            [ids["atom_a"], ids["atom_a"], ids["atom_b"], ids["atom_b"]],
        )

        # Should process only 1 unique pair.
        assert stats["synapses_strengthened"] >= 1


# -----------------------------------------------------------------------
# 7. Contradiction detection (via auto_link)
# -----------------------------------------------------------------------


class TestContradictionDetection:
    """Verify that potential contradictions are detected during auto_link."""

    async def test_detects_negation_contradiction(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """Two same-type atoms with opposing negation words should be flagged."""
        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = LearningEngine(storage, mock_embeddings, atoms, synapses)

        atom_a = await _insert_atom(
            storage, "Always use PostgreSQL for relational data storage needs",
            atom_type="fact", region="technical",
        )
        atom_b = await _insert_atom(
            storage, "Never use PostgreSQL for relational data storage needs",
            atom_type="fact", region="technical",
        )

        # distance=0.2 -> similarity = 0.9 > 0.85 contradiction threshold.
        mock_embeddings.search_similar.return_value = [
            (atom_b, 0.2),
        ]

        result = await engine.auto_link(atom_a)

        relationships = [s["relationship"] for s in result]
        assert "contradicts" in relationships

    async def test_no_contradiction_for_different_types(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """Atoms of different types should not be flagged as contradictions."""
        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = LearningEngine(storage, mock_embeddings, atoms, synapses)

        atom_a = await _insert_atom(
            storage, "Always use PostgreSQL for relational data storage needs",
            atom_type="fact", region="technical",
        )
        atom_b = await _insert_atom(
            storage, "Never use PostgreSQL for relational data storage needs",
            atom_type="experience", region="technical",
        )

        mock_embeddings.search_similar.return_value = [
            (atom_b, 0.2),
        ]

        result = await engine.auto_link(atom_a)

        relationships = [s["relationship"] for s in result]
        assert "contradicts" not in relationships

    def test_contraction_normalisation(self) -> None:
        """don't and dont should be treated the same (both → dont after normalisation)."""
        from memories.atoms import Atom
        from memories.learning import LearningEngine

        def _atom(content: str) -> Atom:
            return Atom(id=1, content=content, type="fact", region="test",
                        confidence=0.8, importance=0.5, access_count=0,
                        created_at="2026-01-01")

        # Both have "dont" once normalised — negation sets are equal → no contradiction.
        a = _atom("don't use this library for production workloads and benchmarking")
        b = _atom("dont use this library for production workloads and benchmarking")
        # similarity doesn't matter here — we test the static method directly
        assert not LearningEngine._is_potential_contradiction(a, b, similarity=0.95)

    def test_antonym_pair_detected(self) -> None:
        """Atoms containing antonym pairs (enable/disable) should be flagged."""
        from memories.atoms import Atom
        from memories.learning import LearningEngine

        def _atom(content: str) -> Atom:
            return Atom(id=1, content=content, type="fact", region="test",
                        confidence=0.8, importance=0.5, access_count=0,
                        created_at="2026-01-01")

        a = _atom("enable the feature flag for the new payment processing system")
        b = _atom("disable the feature flag for the new payment processing system")
        assert LearningEngine._is_potential_contradiction(a, b, similarity=0.95)

    def test_no_false_positive_on_short_content(self) -> None:
        """Short atoms (below min length) are never flagged, even with antonyms."""
        from memories.atoms import Atom
        from memories.learning import LearningEngine

        def _atom(content: str) -> Atom:
            return Atom(id=1, content=content, type="fact", region="test",
                        confidence=0.8, importance=0.5, access_count=0,
                        created_at="2026-01-01")

        a = _atom("enable it")
        b = _atom("disable it")
        assert not LearningEngine._is_potential_contradiction(a, b, similarity=0.95)


# -----------------------------------------------------------------------
# 8. extract_antipattern_fields
# -----------------------------------------------------------------------


class TestExtractAntipatternFields:
    """Verify severity and 'instead' extraction from antipattern content."""

    async def test_extracts_critical_severity(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Content with 'critical' keyword should have critical severity."""
        severity, _ = await learning_engine.extract_antipattern_fields(
            "This is a critical issue that must be addressed"
        )
        assert severity == "critical"

    async def test_extracts_instead_text(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Content with 'instead' should extract the alternative."""
        _, instead = await learning_engine.extract_antipattern_fields(
            "Do not use KEYS. Instead, use SCAN for iteration."
        )
        assert instead is not None
        assert "SCAN" in instead or "use SCAN" in instead

    async def test_extracts_should_pattern(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Content with 'should use X' should extract X."""
        _, instead = await learning_engine.extract_antipattern_fields(
            "You should use parameterized queries."
        )
        assert instead is not None

    async def test_no_severity_when_absent(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Content without severity keywords should return None."""
        severity, _ = await learning_engine.extract_antipattern_fields(
            "This is a generic warning about something"
        )
        # 'generic' is not a severity keyword, so should be None.
        assert severity is None

    async def test_no_instead_when_absent(
        self,
        learning_engine: LearningEngine,
    ) -> None:
        """Content without instead-like patterns should return None."""
        _, instead = await learning_engine.extract_antipattern_fields(
            "The sky is blue on a clear day"
        )
        assert instead is None


# -----------------------------------------------------------------------
# Heuristic relationship type inference
# -----------------------------------------------------------------------


class TestInferRelationshipType:
    """Unit tests for LearningEngine._infer_relationship_type.

    This is a pure static method — no storage or async needed.
    Tests verify that explicit linguistic markers are detected and that
    the length heuristic fires for sufficiently mismatched atom sizes.
    """

    def _atom(self, content: str, atom_type: str = "fact") -> Atom:
        return Atom(id=1, content=content, type=atom_type)

    # --- caused-by ---------------------------------------------------

    def test_because_triggers_caused_by(self) -> None:
        a = self._atom("The deployment failed because the config was missing")
        b = self._atom("Config file was absent from the server")
        rel, bidir = LearningEngine._infer_relationship_type(a, b)
        assert rel == "caused-by"
        assert bidir is False

    def test_due_to_triggers_caused_by(self) -> None:
        a = self._atom("Service latency spiked due to lock contention")
        b = self._atom("Database lock contention statistics")
        rel, bidir = LearningEngine._infer_relationship_type(a, b)
        assert rel == "caused-by"
        assert bidir is False

    def test_causal_signal_in_either_atom(self) -> None:
        """Causal phrase in atom_b should still trigger caused-by."""
        a = self._atom("Redis keyspace notifications")
        b = self._atom("High memory usage resulted in OOM killer")
        rel, _ = LearningEngine._infer_relationship_type(a, b)
        assert rel == "caused-by"

    # --- elaborates --------------------------------------------------

    def test_for_example_triggers_elaborates(self) -> None:
        a = self._atom("Caching strategies can reduce database load")
        b = self._atom("For example, Redis TTL-based caching cuts read queries by 80%")
        rel, bidir = LearningEngine._infer_relationship_type(a, b)
        assert rel == "elaborates"
        assert bidir is False

    def test_more_precisely_triggers_elaborates(self) -> None:
        a = self._atom("SQLite uses WAL mode")
        b = self._atom("More precisely, WAL allows concurrent reads while a write is in progress")
        rel, bidir = LearningEngine._infer_relationship_type(a, b)
        assert rel == "elaborates"
        assert bidir is False

    def test_length_heuristic_triggers_elaborates(self) -> None:
        """Atom 2.5× longer than its pair triggers elaborates via length ratio."""
        short = self._atom("SQLite WAL mode")
        long_content = (
            "SQLite write-ahead logging mode keeps the original database file "
            "unchanged during writes; changes go to a separate WAL file and are "
            "periodically checkpointed back, enabling concurrent readers while "
            "a single writer is active without blocking them."
        )
        long = self._atom(long_content)
        rel, bidir = LearningEngine._infer_relationship_type(short, long)
        assert rel == "elaborates"
        assert bidir is False

    def test_length_heuristic_does_not_fire_for_similar_lengths(self) -> None:
        """Atoms of similar length should not get elaborates from length alone."""
        a = self._atom("Redis SCAN iterates safely over the keyspace")
        b = self._atom("Redis KEYS blocks the server during the scan")
        rel, _ = LearningEngine._infer_relationship_type(a, b)
        # Both are ~45 chars — ratio < 2.5, so should fall back to related-to.
        assert rel == "related-to"

    # --- part-of -----------------------------------------------------

    def test_is_part_of_triggers_partof(self) -> None:
        a = self._atom("The WAL file is part of the SQLite persistence model")
        b = self._atom("SQLite persistence model overview")
        rel, bidir = LearningEngine._infer_relationship_type(a, b)
        assert rel == "part-of"
        assert bidir is False

    def test_is_a_component_triggers_partof(self) -> None:
        a = self._atom("The query planner is a component of the SQLite engine")
        b = self._atom("SQLite engine internals")
        rel, bidir = LearningEngine._infer_relationship_type(a, b)
        assert rel == "part-of"
        assert bidir is False

    def test_belongs_to_triggers_partof(self) -> None:
        a = self._atom("The migration module belongs to the database package")
        b = self._atom("Database package structure")
        rel, bidir = LearningEngine._infer_relationship_type(a, b)
        assert rel == "part-of"
        assert bidir is False

    # --- related-to fallback -----------------------------------------

    def test_generic_content_falls_back_to_related_to(self) -> None:
        a = self._atom("Redis SCAN is O(N) over the full keyspace")
        b = self._atom("Redis KEYS command blocks the server")
        rel, bidir = LearningEngine._infer_relationship_type(a, b)
        assert rel == "related-to"
        assert bidir is True

    # --- priority order ----------------------------------------------

    def test_causal_takes_priority_over_elaborates(self) -> None:
        """caused-by is checked before elaborates; a phrase triggering both
        should yield caused-by."""
        a = self._atom(
            "Service crashed because of memory pressure, for example OOM kills"
        )
        b = self._atom("OOM kill events")
        rel, _ = LearningEngine._infer_relationship_type(a, b)
        assert rel == "caused-by"


# -----------------------------------------------------------------------
# auto_link creates typed synapses via heuristics
# -----------------------------------------------------------------------


class TestAutoLinkTypedSynapses:
    """Verify that auto_link creates typed synapses when heuristics fire.

    Each test inserts two atoms where one contains an explicit linguistic
    marker, configures mock_embeddings to return the second atom as
    similar, and asserts the correct relationship type is created.
    """

    async def test_auto_link_creates_caused_by(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """auto_link should create 'caused-by' for atoms with causal language."""
        atoms_mgr = AtomManager(storage, mock_embeddings)
        synapses_mgr = SynapseManager(storage)
        engine = LearningEngine(storage, mock_embeddings, atoms_mgr, synapses_mgr)

        cause_id = await _insert_atom(
            storage, "High write volume on the database"
        )
        effect_id = await _insert_atom(
            storage, "Connection pool exhausted due to high write volume"
        )

        mock_embeddings.search_similar.return_value = [(cause_id, 0.3)]

        result = await engine.auto_link(effect_id)

        relationships = [s["relationship"] for s in result]
        assert "caused-by" in relationships, (
            f"Expected caused-by synapse, got: {relationships}"
        )

    async def test_auto_link_creates_elaborates_from_phrase(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """auto_link creates 'elaborates' when elaboration language is present."""
        atoms_mgr = AtomManager(storage, mock_embeddings)
        synapses_mgr = SynapseManager(storage)
        engine = LearningEngine(storage, mock_embeddings, atoms_mgr, synapses_mgr)

        general_id = await _insert_atom(storage, "SQLite WAL mode overview")
        detail_id = await _insert_atom(
            storage,
            "In particular, WAL mode keeps the original file unchanged during writes",
        )

        mock_embeddings.search_similar.return_value = [(general_id, 0.3)]

        result = await engine.auto_link(detail_id)

        relationships = [s["relationship"] for s in result]
        assert "elaborates" in relationships, (
            f"Expected elaborates synapse, got: {relationships}"
        )

    async def test_auto_link_creates_part_of(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """auto_link creates 'part-of' when compositional language is present."""
        atoms_mgr = AtomManager(storage, mock_embeddings)
        synapses_mgr = SynapseManager(storage)
        engine = LearningEngine(storage, mock_embeddings, atoms_mgr, synapses_mgr)

        whole_id = await _insert_atom(storage, "The memories storage package")
        part_id = await _insert_atom(
            storage, "The migration module belongs to the storage package"
        )

        mock_embeddings.search_similar.return_value = [(whole_id, 0.3)]

        result = await engine.auto_link(part_id)

        relationships = [s["relationship"] for s in result]
        assert "part-of" in relationships, (
            f"Expected part-of synapse, got: {relationships}"
        )

    async def test_auto_link_still_creates_related_to_for_generic(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """auto_link falls back to 'related-to' when no specific marker found."""
        atoms_mgr = AtomManager(storage, mock_embeddings)
        synapses_mgr = SynapseManager(storage)
        engine = LearningEngine(storage, mock_embeddings, atoms_mgr, synapses_mgr)

        atom_a = await _insert_atom(storage, "Redis SCAN iterates safely")
        atom_b = await _insert_atom(storage, "Redis KEYS blocks the server")

        mock_embeddings.search_similar.return_value = [(atom_a, 0.3)]

        result = await engine.auto_link(atom_b)

        relationships = [s["relationship"] for s in result]
        assert "related-to" in relationships


# -----------------------------------------------------------------------
# W3-A: Batch reads in learning.py
# -----------------------------------------------------------------------


class TestBatchLearningReads:
    """Verify that learning methods use batch reads instead of per-candidate fetches.

    W3-A requires that auto_link, detect_antipattern_links, and detect_supersedes
    all replace their per-candidate get_without_tracking loop with a single
    get_batch_without_tracking call.  Writes (_safe_create_synapse) must remain
    sequential -- no asyncio.gather on the write side.
    """

    async def test_auto_link_batch_read(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """auto_link must call get_batch_without_tracking exactly once for all
        candidates regardless of how many candidates are returned by search."""
        atoms_mgr = AtomManager(storage, mock_embeddings)
        synapses_mgr = SynapseManager(storage)
        engine = LearningEngine(storage, mock_embeddings, atoms_mgr, synapses_mgr)

        # Insert a source atom and three candidates.
        source_id = await _insert_atom(storage, "Redis SCAN iterates safely over the keyspace")
        cand1 = await _insert_atom(storage, "Redis KEYS command blocks the server")
        cand2 = await _insert_atom(storage, "Redis memory usage under KEYS workloads")
        cand3 = await _insert_atom(storage, "Redis SCAN cursor-based iteration")

        # All three candidates pass the similarity threshold.
        # distance=0.3 -> similarity = 0.85, above default threshold 0.82.
        mock_embeddings.search_similar.return_value = [
            (cand1, 0.3),
            (cand2, 0.3),
            (cand3, 0.3),
        ]

        # Spy on get_batch_without_tracking.
        original_batch = atoms_mgr.get_batch_without_tracking
        batch_call_count = 0

        async def spy_batch(atom_ids: list[int]) -> dict:
            nonlocal batch_call_count
            batch_call_count += 1
            return await original_batch(atom_ids)

        atoms_mgr.get_batch_without_tracking = spy_batch  # type: ignore[method-assign]

        await engine.auto_link(source_id)

        # Exactly one batch fetch should have occurred (regardless of candidate count).
        assert batch_call_count == 1, (
            f"Expected 1 batch call, got {batch_call_count}. "
            "auto_link should use get_batch_without_tracking, not a per-candidate loop."
        )

    async def test_auto_link_sequential_writes(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """Synapse creation calls in auto_link must remain sequential.

        The batch read optimisation must NOT accidentally parallelise writes via
        asyncio.gather.  We verify this by tracking the order in which
        _safe_create_synapse is called and confirming it is called sequentially
        (each call completes before the next starts), not gathered.
        """
        atoms_mgr = AtomManager(storage, mock_embeddings)
        synapses_mgr = SynapseManager(storage)
        engine = LearningEngine(storage, mock_embeddings, atoms_mgr, synapses_mgr)

        source_id = await _insert_atom(storage, "Redis SCAN iterates safely over the keyspace")
        cand1 = await _insert_atom(storage, "Redis KEYS command blocks the server")
        cand2 = await _insert_atom(storage, "Redis memory usage under KEYS workloads")

        mock_embeddings.search_similar.return_value = [
            (cand1, 0.3),
            (cand2, 0.3),
        ]

        # Track call order and active concurrency.
        call_order: list[tuple[int, int]] = []
        active_count = 0
        max_concurrent = 0
        original_create = engine._safe_create_synapse

        async def spy_create(source_id: int, target_id: int, **kwargs) -> dict | None:
            nonlocal active_count, max_concurrent
            active_count += 1
            if active_count > max_concurrent:
                max_concurrent = active_count
            call_order.append((source_id, target_id))
            result = await original_create(source_id=source_id, target_id=target_id, **kwargs)
            active_count -= 1
            return result

        engine._safe_create_synapse = spy_create  # type: ignore[method-assign]

        await engine.auto_link(source_id)

        # At most 1 write should be active at once (sequential, not gathered).
        assert max_concurrent <= 1, (
            f"Writes were parallelised: max concurrent _safe_create_synapse calls = "
            f"{max_concurrent}. Writes must remain sequential."
        )

    async def test_detect_antipattern_links_batch_read(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """detect_antipattern_links must call get_batch_without_tracking exactly once."""
        atoms_mgr = AtomManager(storage, mock_embeddings)
        synapses_mgr = SynapseManager(storage)
        engine = LearningEngine(storage, mock_embeddings, atoms_mgr, synapses_mgr)

        source_id = await _insert_atom(storage, "Redis SCAN iterates safely over the keyspace")
        ap1 = await _insert_atom(
            storage, "Never use KEYS in production",
            atom_type="antipattern",
        )
        ap2 = await _insert_atom(
            storage, "Avoid blocking Redis commands",
            atom_type="antipattern",
        )
        non_ap = await _insert_atom(storage, "Redis connection pool settings")

        mock_embeddings.search_similar.return_value = [
            (ap1, 0.3),
            (ap2, 0.3),
            (non_ap, 0.3),
        ]

        original_batch = atoms_mgr.get_batch_without_tracking
        batch_call_count = 0

        async def spy_batch(atom_ids: list[int]) -> dict:
            nonlocal batch_call_count
            batch_call_count += 1
            return await original_batch(atom_ids)

        atoms_mgr.get_batch_without_tracking = spy_batch  # type: ignore[method-assign]

        await engine.detect_antipattern_links(source_id)

        assert batch_call_count == 1, (
            f"Expected 1 batch call, got {batch_call_count}. "
            "detect_antipattern_links should use get_batch_without_tracking."
        )

    async def test_detect_supersedes_batch_read(
        self,
        storage: Storage,
        mock_embeddings: MagicMock,
    ) -> None:
        """detect_supersedes must call get_batch_without_tracking exactly once."""
        atoms_mgr = AtomManager(storage, mock_embeddings)
        synapses_mgr = SynapseManager(storage)
        engine = LearningEngine(storage, mock_embeddings, atoms_mgr, synapses_mgr)

        now = datetime.now(tz=timezone.utc)
        old_ts = (now - timedelta(days=30)).isoformat()

        # Source atom is the newer one.
        source_id = await _insert_atom(
            storage, "Redis SCAN cursor-based iteration is the safe approach",
            atom_type="fact",
        )
        # Older candidates of the same type, very similar (similarity > 0.9).
        old1 = await _insert_atom(
            storage, "Redis SCAN cursor-based iteration approach",
            atom_type="fact", created_at=old_ts,
        )
        old2 = await _insert_atom(
            storage, "Redis SCAN cursor iteration is recommended",
            atom_type="fact", created_at=old_ts,
        )

        # distance=0.1 -> similarity = 0.95, above supersede threshold 0.9.
        mock_embeddings.search_similar.return_value = [
            (old1, 0.1),
            (old2, 0.1),
        ]

        original_batch = atoms_mgr.get_batch_without_tracking
        batch_call_count = 0

        async def spy_batch(atom_ids: list[int]) -> dict:
            nonlocal batch_call_count
            batch_call_count += 1
            return await original_batch(atom_ids)

        atoms_mgr.get_batch_without_tracking = spy_batch  # type: ignore[method-assign]

        await engine.detect_supersedes(source_id)

        assert batch_call_count == 1, (
            f"Expected 1 batch call, got {batch_call_count}. "
            "detect_supersedes should use get_batch_without_tracking."
        )
