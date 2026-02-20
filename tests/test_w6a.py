"""Wave 6-A -- Antipattern auto_link skip + batch suggest_region.

Fix 1: auto_link() must NOT create a generic relationship synapse (e.g.
       "related-to") when EITHER endpoint is an antipattern.  Only the
       "warns-against" synapse should be created.

Fix 2: suggest_region() must use get_batch_without_tracking() instead of
       calling get_without_tracking() per candidate.

All tests are written BEFORE the implementation (TDD red phase).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from memories.atoms import Atom, AtomManager
from memories.config import get_config
from memories.embeddings import EmbeddingEngine
from memories.learning import LearningEngine
from memories.storage import Storage
from memories.synapses import SynapseManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atom(
    atom_id: int,
    content: str = "some content",
    atom_type: str = "fact",
    region: str = "general",
    confidence: float = 1.0,
) -> Atom:
    """Build a minimal Atom dataclass for testing."""
    return Atom(
        id=atom_id,
        content=content,
        type=atom_type,
        region=region,
        confidence=confidence,
        importance=0.5,
        access_count=0,
        last_accessed_at=None,
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
        tags=[],
        is_deleted=False,
        severity=None,
        instead=None,
        source_project=None,
        source_session=None,
        source_file=None,
    )


def _build_engine(
    atoms_map: dict[int, Atom],
    similar_results: list[tuple[int, float]],
    batch_map: dict[int, Atom] | None = None,
) -> LearningEngine:
    """Wire up a LearningEngine with mocked dependencies.

    Parameters
    ----------
    atoms_map:
        Mapping of atom_id -> Atom for get_without_tracking calls.
    similar_results:
        Return value for EmbeddingEngine.search_similar (id, distance).
    batch_map:
        Return value for get_batch_without_tracking.  If None, derived from
        atoms_map using the IDs present in similar_results.
    """
    storage = MagicMock(spec=Storage)
    embeddings = MagicMock(spec=EmbeddingEngine)
    atoms_mgr = MagicMock(spec=AtomManager)
    synapses_mgr = MagicMock(spec=SynapseManager)

    # search_similar returns (candidate_id, distance) tuples.
    embeddings.search_similar = AsyncMock(return_value=similar_results)

    # get_without_tracking: used for the source atom lookup.
    async def _get_single(aid: int) -> Atom | None:
        return atoms_map.get(aid)

    atoms_mgr.get_without_tracking = AsyncMock(side_effect=_get_single)

    # get_batch_without_tracking: used for batch candidate lookup.
    if batch_map is None:
        candidate_ids = {cid for cid, _ in similar_results}
        batch_map = {k: v for k, v in atoms_map.items() if k in candidate_ids}
    atoms_mgr.get_batch_without_tracking = AsyncMock(return_value=batch_map)

    # _safe_create_synapse delegates to synapses.create, which returns a
    # synapse-like object.  We make create return a mock with the right attrs.
    async def _fake_create(**kwargs: Any) -> MagicMock:
        s = MagicMock()
        s.source_id = kwargs["source_id"]
        s.target_id = kwargs["target_id"]
        s.relationship = kwargs["relationship"]
        s.strength = kwargs["strength"]
        return s

    synapses_mgr.create = AsyncMock(side_effect=_fake_create)

    engine = LearningEngine(storage, embeddings, atoms_mgr, synapses_mgr)
    return engine


# ---------------------------------------------------------------------------
# Fix 1: auto_link antipattern skip tests
# ---------------------------------------------------------------------------


class TestAutoLinkSkipsRelatedToForAntipatterns:
    """auto_link must NOT create a generic relationship synapse when either
    endpoint is an antipattern.  Only warns-against should be created."""

    async def test_no_related_to_when_candidate_is_antipattern(self) -> None:
        """When the candidate atom is an antipattern, auto_link should create
        only a warns-against synapse (candidate -> source), not a related-to."""
        source = _make_atom(1, content="Use proper error handling", atom_type="fact")
        candidate = _make_atom(2, content="Bare except clauses", atom_type="antipattern")

        # distance=0.2 => similarity = max(0, 1 - 0.2/2) = 0.9, above default threshold (0.82)
        engine = _build_engine(
            atoms_map={1: source, 2: candidate},
            similar_results=[(2, 0.2)],
        )

        result = await engine.auto_link(1)

        # Should have exactly one synapse: warns-against from antipattern -> source.
        relationships = [s["relationship"] for s in result]
        assert "related-to" not in relationships, (
            "auto_link created a 'related-to' synapse when candidate is an antipattern. "
            "Fix 1 requires skipping the generic relationship for antipattern endpoints."
        )
        assert "warns-against" in relationships, (
            "auto_link must still create 'warns-against' when candidate is an antipattern."
        )

    async def test_warns_against_direction_when_candidate_is_antipattern(self) -> None:
        """warns-against must flow from antipattern -> target (the normal atom)."""
        source = _make_atom(1, content="Use proper error handling", atom_type="fact")
        candidate = _make_atom(2, content="Bare except clauses", atom_type="antipattern")

        engine = _build_engine(
            atoms_map={1: source, 2: candidate},
            similar_results=[(2, 0.2)],
        )

        result = await engine.auto_link(1)
        warns = [s for s in result if s["relationship"] == "warns-against"]
        assert len(warns) == 1
        # Direction: antipattern (2) -> normal atom (1)
        assert warns[0]["source_id"] == 2
        assert warns[0]["target_id"] == 1

    async def test_no_related_to_when_source_is_antipattern(self) -> None:
        """When the NEW atom is an antipattern, auto_link should create only
        warns-against (source -> candidate), not a related-to."""
        source = _make_atom(10, content="Never use eval()", atom_type="antipattern")
        candidate = _make_atom(20, content="Dynamic code execution patterns", atom_type="fact")

        engine = _build_engine(
            atoms_map={10: source, 20: candidate},
            similar_results=[(20, 0.2)],
        )

        result = await engine.auto_link(10)

        relationships = [s["relationship"] for s in result]
        assert "related-to" not in relationships, (
            "auto_link created a 'related-to' synapse when source is an antipattern. "
            "Fix 1 requires skipping the generic relationship for antipattern endpoints."
        )
        assert "warns-against" in relationships, (
            "auto_link must still create 'warns-against' when source is an antipattern."
        )

    async def test_warns_against_direction_when_source_is_antipattern(self) -> None:
        """warns-against must flow from antipattern (source) -> candidate."""
        source = _make_atom(10, content="Never use eval()", atom_type="antipattern")
        candidate = _make_atom(20, content="Dynamic code execution patterns", atom_type="fact")

        engine = _build_engine(
            atoms_map={10: source, 20: candidate},
            similar_results=[(20, 0.2)],
        )

        result = await engine.auto_link(10)
        warns = [s for s in result if s["relationship"] == "warns-against"]
        assert len(warns) == 1
        # Direction: antipattern (10) -> normal atom (20)
        assert warns[0]["source_id"] == 10
        assert warns[0]["target_id"] == 20

    async def test_related_to_still_created_for_non_antipattern_pairs(self) -> None:
        """When neither endpoint is an antipattern, the generic relationship
        synapse must still be created (regression guard)."""
        source = _make_atom(1, content="Python error handling best practices", atom_type="fact")
        candidate = _make_atom(2, content="Try/except usage patterns", atom_type="fact")

        engine = _build_engine(
            atoms_map={1: source, 2: candidate},
            similar_results=[(2, 0.2)],
        )

        result = await engine.auto_link(1)

        relationships = [s["relationship"] for s in result]
        assert "related-to" in relationships, (
            "auto_link must still create generic relationship synapses for "
            "non-antipattern pairs."
        )
        # No warns-against for non-antipatterns
        assert "warns-against" not in relationships

    async def test_both_antipattern_no_related_to(self) -> None:
        """When both source and candidate are antipatterns, no related-to
        and no warns-against should be created (antipattern-to-antipattern
        has no meaningful warns-against direction)."""
        source = _make_atom(1, content="Using eval()", atom_type="antipattern")
        candidate = _make_atom(2, content="Using exec()", atom_type="antipattern")

        engine = _build_engine(
            atoms_map={1: source, 2: candidate},
            similar_results=[(2, 0.2)],
        )

        result = await engine.auto_link(1)

        relationships = [s["relationship"] for s in result]
        assert "related-to" not in relationships, (
            "auto_link must not create related-to when both endpoints are antipatterns."
        )


# ---------------------------------------------------------------------------
# Fix 2: suggest_region uses batch read
# ---------------------------------------------------------------------------


class TestSuggestRegionBatchRead:
    """suggest_region must use get_batch_without_tracking instead of
    calling get_without_tracking per candidate."""

    async def test_suggest_region_uses_batch_fetch(self) -> None:
        """suggest_region must call get_batch_without_tracking once, not
        get_without_tracking N times."""
        atoms = {
            100: _make_atom(100, region="errors"),
            101: _make_atom(101, region="errors"),
            102: _make_atom(102, region="technical"),
        }

        # search_similar returns 3 candidates (id, distance).
        similar = [(100, 0.3), (101, 0.35), (102, 0.5)]

        engine = _build_engine(
            atoms_map=atoms,
            similar_results=similar,
            batch_map=atoms,
        )

        # Content that does NOT match any keyword -> falls through to vote.
        result = await engine.suggest_region("some generic content about things")

        # get_batch_without_tracking must have been called.
        engine._atoms.get_batch_without_tracking.assert_called_once()

        # get_without_tracking must NOT have been called for the candidates.
        # It may be called zero times (ideal) or not at all for per-candidate reads.
        per_item_calls = engine._atoms.get_without_tracking.call_count
        assert per_item_calls == 0, (
            f"suggest_region called get_without_tracking {per_item_calls} times. "
            "Fix 2 requires using get_batch_without_tracking instead."
        )

    async def test_suggest_region_majority_vote_still_works(self) -> None:
        """After batching, the majority vote logic must still return the
        correct region."""
        atoms = {
            100: _make_atom(100, region="errors"),
            101: _make_atom(101, region="errors"),
            102: _make_atom(102, region="technical"),
        }

        similar = [(100, 0.3), (101, 0.35), (102, 0.5)]

        engine = _build_engine(
            atoms_map=atoms,
            similar_results=similar,
            batch_map=atoms,
        )

        result = await engine.suggest_region("some generic content about things")
        assert result == "errors", (
            f"Expected majority vote to return 'errors', got {result!r}."
        )

    async def test_suggest_region_handles_missing_atoms_in_batch(self) -> None:
        """If some atoms are missing from the batch result (deleted between
        search and fetch), they should be skipped gracefully."""
        atoms = {
            100: _make_atom(100, region="errors"),
            # 101 is missing (deleted)
            102: _make_atom(102, region="technical"),
        }

        similar = [(100, 0.3), (101, 0.35), (102, 0.5)]

        engine = _build_engine(
            atoms_map=atoms,
            similar_results=similar,
            batch_map={100: atoms[100], 102: atoms[102]},  # 101 missing
        )

        result = await engine.suggest_region("some generic content about things")
        # 1 vote "errors", 1 vote "technical" -- tie, either is acceptable
        assert result in ("errors", "technical")

    async def test_suggest_region_keyword_priority_unaffected(self) -> None:
        """Keyword matching (priority 2) must still take precedence over
        the vector vote (priority 3) -- regression guard."""
        engine = _build_engine(atoms_map={}, similar_results=[])

        result = await engine.suggest_region("I always prefer this error handling style")
        # "prefer" matches "personal", "error" matches "errors"
        # personal keywords come first in _REGION_KEYWORDS iteration
        assert result == "personal"

    async def test_suggest_region_project_priority_unaffected(self) -> None:
        """source_project (priority 1) must still take precedence -- regression guard."""
        engine = _build_engine(atoms_map={}, similar_results=[])

        result = await engine.suggest_region(
            "some content with error keyword",
            source_project="myapp",
        )
        assert result == "project:myapp"
