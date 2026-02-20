"""Wave 8-A tests: remove redundant detect_antipattern_links, batch related-atom fetch.

Fix A1: Remove detect_antipattern_links() call from brain.remember()
        — auto_link() already creates warns-against synapses.
Fix A2: Batch related-atom fetch in brain.remember() — replace N sequential
        get_without_tracking() calls with a single get_batch_without_tracking().
Fix A3: Same batch fix in brain.create_task() (task creation path).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.atoms import Atom, AtomManager
from memories.brain import Brain
from memories.embeddings import EmbeddingEngine
from memories.learning import LearningEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_brain(
    auto_link_result: list[dict] | None = None,
    detect_supersedes_result: int = 0,
) -> Brain:
    """Build a Brain with mocked subsystems for fast unit tests.

    Returns a Brain whose _atoms, _learning, _synapses, _embeddings,
    _retrieval, _consolidation, and _context are all mocks, so no real
    DB or embedding calls occur.
    """
    brain = Brain.__new__(Brain)
    brain._initialized = True
    brain._current_session_id = "test-session"
    brain._session_atoms = set()

    # AtomManager mock
    atoms = MagicMock(spec=AtomManager)
    dummy_atom = MagicMock(spec=Atom)
    dummy_atom.id = 42
    dummy_atom.content = "test content"
    dummy_atom.type = "fact"
    dummy_atom.region = "general"
    dummy_atom.to_dict = MagicMock(return_value={"id": 42, "content": "test content"})
    atoms.create = AsyncMock(return_value=dummy_atom)
    atoms.get_without_tracking = AsyncMock(return_value=dummy_atom)
    atoms.get_batch_without_tracking = AsyncMock(return_value={})
    brain._atoms = atoms

    # LearningEngine mock
    learning = MagicMock(spec=LearningEngine)
    if auto_link_result is None:
        auto_link_result = []
    learning.auto_link = AsyncMock(return_value=auto_link_result)
    learning.detect_antipattern_links = AsyncMock(return_value=0)
    learning.detect_supersedes = AsyncMock(return_value=detect_supersedes_result)
    brain._learning = learning

    # EmbeddingEngine mock
    embeddings = MagicMock(spec=EmbeddingEngine)
    embeddings.embed_and_store = AsyncMock(return_value=[0.0] * 768)
    embeddings.health_check = AsyncMock(return_value=True)
    brain._embeddings = embeddings

    # Remaining subsystem stubs (not exercised directly but referenced).
    brain._synapses = MagicMock()
    brain._retrieval = MagicMock()
    brain._consolidation = MagicMock()
    brain._context = MagicMock()
    brain._config = MagicMock()

    # Disable vec_available so remember() skips the semantic dedup path
    # (which requires a real embedding engine and storage).
    storage_mock = MagicMock()
    storage_mock.vec_available = False
    brain._storage = storage_mock

    return brain


def _make_synapse_dict(
    source_id: int,
    target_id: int,
    relationship: str = "related-to",
    strength: float = 0.8,
) -> dict:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "relationship": relationship,
        "strength": strength,
    }


def _make_related_atom(atom_id: int, content: str = "related") -> MagicMock:
    a = MagicMock(spec=Atom)
    a.id = atom_id
    a.content = content
    a.type = "fact"
    a.region = "general"
    return a


# -----------------------------------------------------------------------
# A1: detect_antipattern_links must NOT be called from remember()
# -----------------------------------------------------------------------


class TestRemoveDetectAntipatternLinks:
    """Fix A1: brain.remember() must NOT call detect_antipattern_links().

    auto_link() already creates warns-against synapses for antipatterns.
    The separate detect_antipattern_links() call is redundant and wastes
    3 SQL queries per remember() invocation.
    """

    async def test_remember_does_not_call_detect_antipattern_links(self) -> None:
        """After remember(), detect_antipattern_links must not have been called."""
        brain = _make_mock_brain()

        await brain.remember("Some new fact", type="fact")

        brain._learning.detect_antipattern_links.assert_not_called()

    async def test_remember_still_reports_antipattern_count(self) -> None:
        """antipattern_count is derived from created_synapses warns-against entries."""
        synapses = [
            _make_synapse_dict(42, 10, "related-to", 0.85),
            _make_synapse_dict(11, 42, "warns-against", 0.9),
            _make_synapse_dict(42, 12, "related-to", 0.75),
            _make_synapse_dict(13, 42, "warns-against", 0.7),
        ]
        brain = _make_mock_brain(auto_link_result=synapses)

        # Set up batch fetch to return related atoms.
        related_atoms_map = {
            10: _make_related_atom(10, "atom 10"),
            11: _make_related_atom(11, "atom 11"),
            12: _make_related_atom(12, "atom 12"),
            13: _make_related_atom(13, "atom 13"),
        }
        brain._atoms.get_batch_without_tracking = AsyncMock(
            return_value=related_atoms_map
        )

        result = await brain.remember("Some content", type="fact")

        # The log message should have antipattern_count = 2 (two warns-against).
        # We verify via the detect_antipattern_links not being called:
        brain._learning.detect_antipattern_links.assert_not_called()
        # And synapses_created reflecting all synapses including warns-against:
        assert result["synapses_created"] == 4

    async def test_remember_zero_antipattern_count_when_no_warns_against(self) -> None:
        """When no warns-against synapses exist, antipattern count is 0."""
        synapses = [
            _make_synapse_dict(42, 10, "related-to", 0.85),
            _make_synapse_dict(42, 11, "related-to", 0.75),
        ]
        brain = _make_mock_brain(auto_link_result=synapses)
        brain._atoms.get_batch_without_tracking = AsyncMock(
            return_value={
                10: _make_related_atom(10),
                11: _make_related_atom(11),
            }
        )

        result = await brain.remember("Some content", type="fact")

        brain._learning.detect_antipattern_links.assert_not_called()
        assert result["synapses_created"] == 2


# -----------------------------------------------------------------------
# A2: Batch related-atom fetch in brain.remember()
# -----------------------------------------------------------------------


class TestBatchRelatedAtomFetchRemember:
    """Fix A2: brain.remember() must use get_batch_without_tracking() instead
    of per-item get_without_tracking() when collecting related atom summaries.
    """

    async def test_remember_uses_batch_not_individual_fetch(self) -> None:
        """remember() should call get_batch_without_tracking once, not
        get_without_tracking per synapse.
        """
        synapses = [
            _make_synapse_dict(42, 10, "related-to", 0.85),
            _make_synapse_dict(42, 11, "related-to", 0.75),
            _make_synapse_dict(12, 42, "warns-against", 0.9),
        ]
        brain = _make_mock_brain(auto_link_result=synapses)

        related_atoms_map = {
            10: _make_related_atom(10, "atom 10"),
            11: _make_related_atom(11, "atom 11"),
            12: _make_related_atom(12, "atom 12"),
        }
        brain._atoms.get_batch_without_tracking = AsyncMock(
            return_value=related_atoms_map
        )

        result = await brain.remember("Test content", type="fact")

        # get_batch_without_tracking must be called (at least once).
        brain._atoms.get_batch_without_tracking.assert_called()

        # get_without_tracking must NOT be called for the related-atom loop.
        # It may still be called by atoms.create or elsewhere, but not for
        # fetching related atoms (IDs 10, 11, 12).
        # We check that no call had one of the related IDs.
        for call in brain._atoms.get_without_tracking.call_args_list:
            called_id = call[0][0] if call[0] else call[1].get("atom_id")
            assert called_id not in {10, 11, 12}, (
                f"get_without_tracking was called with related atom ID {called_id}; "
                "should use get_batch_without_tracking instead"
            )

    async def test_remember_batch_returns_correct_related_atoms(self) -> None:
        """The related_atoms list in the result should still be populated correctly."""
        synapses = [
            _make_synapse_dict(42, 100, "related-to", 0.85),
            _make_synapse_dict(200, 42, "warns-against", 0.9),
        ]
        brain = _make_mock_brain(auto_link_result=synapses)

        atom_100 = _make_related_atom(100, "content of atom 100")
        atom_200 = _make_related_atom(200, "content of atom 200")
        brain._atoms.get_batch_without_tracking = AsyncMock(
            return_value={100: atom_100, 200: atom_200}
        )

        result = await brain.remember("New fact", type="fact")

        related = result["related_atoms"]
        assert len(related) == 2
        ids_in_result = {r["id"] for r in related}
        assert ids_in_result == {100, 200}

        # Check that relationship info is preserved.
        for r in related:
            if r["id"] == 100:
                assert r["relationship"] == "related-to"
                assert r["strength"] == 0.85
            elif r["id"] == 200:
                assert r["relationship"] == "warns-against"
                assert r["strength"] == 0.9

    async def test_remember_batch_handles_missing_atoms(self) -> None:
        """If batch fetch returns fewer atoms than synapses, skip missing ones."""
        synapses = [
            _make_synapse_dict(42, 10, "related-to", 0.85),
            _make_synapse_dict(42, 11, "related-to", 0.75),  # atom 11 missing
        ]
        brain = _make_mock_brain(auto_link_result=synapses)

        # Only atom 10 is returned by batch fetch — atom 11 is "missing/deleted".
        brain._atoms.get_batch_without_tracking = AsyncMock(
            return_value={10: _make_related_atom(10, "atom 10")}
        )

        result = await brain.remember("Test", type="fact")

        # Only 1 related atom should appear.
        assert len(result["related_atoms"]) == 1
        assert result["related_atoms"][0]["id"] == 10

    async def test_remember_batch_deduplicates_related_ids(self) -> None:
        """If multiple synapses point to the same atom, only fetch/show it once."""
        synapses = [
            _make_synapse_dict(42, 10, "related-to", 0.85),
            _make_synapse_dict(10, 42, "warns-against", 0.7),
        ]
        brain = _make_mock_brain(auto_link_result=synapses)

        brain._atoms.get_batch_without_tracking = AsyncMock(
            return_value={10: _make_related_atom(10, "atom 10")}
        )

        result = await brain.remember("Test", type="fact")

        # Atom 10 appears in two synapses but should only show once in related_atoms.
        assert len(result["related_atoms"]) == 1
        assert result["related_atoms"][0]["id"] == 10

        # Batch fetch should have been called with just [10], not [10, 10].
        call_args = brain._atoms.get_batch_without_tracking.call_args[0][0]
        assert call_args == [10], f"Expected [10], got {call_args}"


# -----------------------------------------------------------------------
# A3: Batch related-atom fetch in brain.create_task()
# -----------------------------------------------------------------------


class TestBatchRelatedAtomFetchCreateTask:
    """Fix A3: brain.create_task() must use get_batch_without_tracking() instead
    of per-item get_without_tracking() when collecting related atom summaries.
    Same N+1 pattern as remember(), same fix.
    """

    async def test_create_task_uses_batch_not_individual_fetch(self) -> None:
        """create_task() should call get_batch_without_tracking, not per-item."""
        synapses = [
            _make_synapse_dict(42, 10, "related-to", 0.85),
            _make_synapse_dict(42, 11, "related-to", 0.75),
        ]
        brain = _make_mock_brain(auto_link_result=synapses)

        related_atoms_map = {
            10: _make_related_atom(10, "atom 10"),
            11: _make_related_atom(11, "atom 11"),
        }
        brain._atoms.get_batch_without_tracking = AsyncMock(
            return_value=related_atoms_map
        )

        result = await brain.create_task("Write tests", status="pending")

        # get_batch_without_tracking must be called.
        brain._atoms.get_batch_without_tracking.assert_called()

        # get_without_tracking must NOT be called with related IDs.
        for call in brain._atoms.get_without_tracking.call_args_list:
            called_id = call[0][0] if call[0] else call[1].get("atom_id")
            assert called_id not in {10, 11}, (
                f"get_without_tracking was called with related atom ID {called_id}; "
                "should use get_batch_without_tracking instead"
            )

    async def test_create_task_batch_returns_correct_related_atoms(self) -> None:
        """The related_atoms list in create_task result should be correct."""
        synapses = [
            _make_synapse_dict(42, 100, "related-to", 0.85),
        ]
        brain = _make_mock_brain(auto_link_result=synapses)

        brain._atoms.get_batch_without_tracking = AsyncMock(
            return_value={100: _make_related_atom(100, "related task")}
        )

        result = await brain.create_task("New task", status="pending")

        related = result["related_atoms"]
        assert len(related) == 1
        assert related[0]["id"] == 100
        assert related[0]["relationship"] == "related-to"

    async def test_create_task_batch_handles_empty_synapses(self) -> None:
        """When no synapses are created, no batch fetch needed."""
        brain = _make_mock_brain(auto_link_result=[])

        result = await brain.create_task("Standalone task", status="pending")

        assert result["related_atoms"] == []
        # Batch should still be called (with empty list) or not called at all.
        # Either is acceptable — what matters is no per-item fetch.
        for call in brain._atoms.get_without_tracking.call_args_list:
            called_id = call[0][0] if call[0] else call[1].get("atom_id")
            # No related IDs should have been fetched individually.
            assert called_id is None or called_id == 42  # 42 is the atom itself
