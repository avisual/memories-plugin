"""Wave 5-D: Remove vestigial _session_atoms dict and track_atom_access method.

After Wave 4, the real session-atom accumulator is the hook_session_atoms DB
table (populated by CLI _save_hook_atoms after each hook call).  The in-memory
_session_atoms dict in Brain is never consumed by anything meaningful -- the
CLI stop hook reads from the DB, not from this dict.

D1 - Brain must not have a _session_atoms attribute after __init__
D2 - track_atom_access must be a no-op (not raise, not mutate state)
D3 - end_session must read from hook_session_atoms DB, not _session_atoms
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from memories.atoms import AtomManager
from memories.brain import Brain
from memories.config import get_config
from memories.retrieval import RecallResult
from memories.synapses import SynapseManager

from tests.conftest import insert_atom


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def brain(storage, mock_embeddings):
    """Brain wired to real Storage + managers, mocked engines."""
    b = Brain.__new__(Brain)
    b._config = get_config()
    b._storage = storage
    b._embeddings = mock_embeddings
    b._atoms = AtomManager(storage, mock_embeddings)
    b._synapses = SynapseManager(storage)
    b._context = MagicMock()

    b._retrieval = MagicMock()
    b._learning = MagicMock()
    b._consolidation = MagicMock()

    b._current_session_id = "test-session-w5d"
    b._initialized = True

    # Learning engine async mocks.
    b._learning.suggest_region = AsyncMock(return_value="technical")
    b._learning.extract_antipattern_fields = AsyncMock(
        return_value=("high", "use X instead")
    )
    b._learning.auto_link = AsyncMock(return_value=[])
    b._learning.detect_antipattern_links = AsyncMock(return_value=0)
    b._learning.detect_supersedes = AsyncMock(return_value=0)
    b._learning.session_end_learning = AsyncMock(
        return_value={"synapses_strengthened": 0, "synapses_created": 0}
    )

    # Retrieval engine async mock (default empty recall).
    b._retrieval.recall = AsyncMock(return_value=RecallResult(seed_count=0))

    return b


# ===========================================================================
# D1 -- Brain must not have _session_atoms attribute
# ===========================================================================


class TestD1NoSessionAtomsAttribute:
    """Brain.__init__ must no longer create a _session_atoms dict."""

    async def test_no_session_atoms_attr_after_init(self):
        """A freshly constructed Brain must not have _session_atoms."""
        b = Brain.__new__(Brain)
        b.__init__()  # type: ignore[misc]
        assert not hasattr(b, "_session_atoms"), (
            "Brain still has _session_atoms attribute -- it should be removed"
        )

    async def test_no_session_atoms_attr_on_fixture(self, brain):
        """The test fixture Brain must also lack _session_atoms."""
        assert not hasattr(brain, "_session_atoms"), (
            "Brain fixture still has _session_atoms -- the fixture or Brain "
            "is still creating it"
        )


# ===========================================================================
# D2 -- track_atom_access is a no-op
# ===========================================================================


class TestD2TrackAtomAccessNoOp:
    """track_atom_access must exist (for compat) but must be a no-op."""

    async def test_track_atom_access_exists(self, brain):
        """Method must still be callable (called from remember/recall)."""
        assert hasattr(brain, "track_atom_access")
        assert callable(brain.track_atom_access)

    async def test_track_atom_access_is_noop(self, brain):
        """Calling track_atom_access must not raise and must not store state."""
        # Should not raise.
        brain.track_atom_access(42)
        brain.track_atom_access(99)

        # Must not have created _session_atoms.
        assert not hasattr(brain, "_session_atoms"), (
            "track_atom_access is still populating _session_atoms"
        )

    async def test_track_atom_access_returns_none(self, brain):
        """Return value must be None (no-op)."""
        result = brain.track_atom_access(1)
        assert result is None


# ===========================================================================
# D3 -- end_session reads from hook_session_atoms DB, not _session_atoms
# ===========================================================================


class TestD3EndSessionUsesDB:
    """end_session must read atom IDs from hook_session_atoms table."""

    async def test_end_session_reads_db_atoms(self, brain, storage):
        """end_session passes atom IDs from hook_session_atoms to learning."""
        # Seed the DB with hook_session_atoms for the current session.
        session_id = brain._current_session_id
        await storage.execute_write(
            "INSERT OR IGNORE INTO hook_session_atoms "
            "(claude_session_id, atom_id) VALUES (?, ?)",
            (session_id, 10),
        )
        await storage.execute_write(
            "INSERT OR IGNORE INTO hook_session_atoms "
            "(claude_session_id, atom_id) VALUES (?, ?)",
            (session_id, 20),
        )

        await brain.end_session()

        # session_end_learning must have been called with atom IDs from DB.
        brain._learning.session_end_learning.assert_awaited_once()
        call_args = brain._learning.session_end_learning.call_args
        passed_session_id = call_args[0][0]
        passed_atoms = call_args[0][1]

        assert passed_session_id == session_id
        assert set(passed_atoms) == {10, 20}

    async def test_end_session_no_atoms_in_db(self, brain, storage):
        """When no atoms in hook_session_atoms, learning is skipped."""
        brain._learning.session_end_learning.reset_mock()

        await brain.end_session()

        brain._learning.session_end_learning.assert_not_awaited()

    async def test_end_session_returns_correct_count(self, brain, storage):
        """Returned atoms_accessed count reflects DB rows."""
        session_id = brain._current_session_id
        for aid in [5, 6, 7]:
            await storage.execute_write(
                "INSERT OR IGNORE INTO hook_session_atoms "
                "(claude_session_id, atom_id) VALUES (?, ?)",
                (session_id, aid),
            )

        result = await brain.end_session()

        assert result["session_id"] == session_id
        assert result["atoms_accessed"] == 3

    async def test_end_session_resets_session_id(self, brain, storage):
        """After end_session, _current_session_id is None."""
        await brain.end_session()

        assert brain._current_session_id is None

    async def test_end_session_updates_session_end_time(self, brain, storage):
        """end_session writes ended_at to the sessions table."""
        session_id = brain._current_session_id
        # Pre-create the session row.
        await storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            (session_id, None),
        )

        await brain.end_session()

        rows = await storage.execute(
            "SELECT ended_at FROM sessions WHERE id = ?", (session_id,)
        )
        assert rows and rows[0]["ended_at"] is not None


# ===========================================================================
# Integration: remember and recall still work without _session_atoms
# ===========================================================================


class TestRememberRecallWithoutSessionAtoms:
    """remember() and recall() must work without _session_atoms."""

    async def test_remember_works(self, brain, storage):
        """remember() completes without error after _session_atoms removal."""
        result = await brain.remember(
            content="Python dicts are ordered since 3.7",
            type="fact",
            region="technical",
        )
        assert "atom_id" in result
        assert isinstance(result["atom_id"], int)

    async def test_recall_works(self, brain, storage):
        """recall() completes without error after _session_atoms removal."""
        result = await brain.recall(query="How does Python work?")
        assert isinstance(result, dict)
        assert "atoms" in result

    async def test_recall_passes_empty_session_atoms(self, brain, storage):
        """recall() passes empty list for session_atom_ids (from DB)."""
        await brain.recall(query="test query")

        call_kwargs = brain._retrieval.recall.call_args.kwargs
        # session_atom_ids should be an empty list (nothing in hook_session_atoms).
        assert call_kwargs.get("session_atom_ids") == [] or \
               call_kwargs.get("session_atom_ids") is None

    async def test_recall_passes_db_session_atoms(self, brain, storage):
        """recall() reads session atoms from hook_session_atoms for priming."""
        session_id = brain._current_session_id
        for aid in [100, 200]:
            await storage.execute_write(
                "INSERT OR IGNORE INTO hook_session_atoms "
                "(claude_session_id, atom_id) VALUES (?, ?)",
                (session_id, aid),
            )

        await brain.recall(query="test priming")

        call_kwargs = brain._retrieval.recall.call_args.kwargs
        passed_ids = call_kwargs.get("session_atom_ids", [])
        assert set(passed_ids) == {100, 200}
