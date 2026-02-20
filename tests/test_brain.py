"""Comprehensive tests for the Brain orchestrator."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from memories.atoms import AtomManager
from memories.brain import Brain
from memories.config import get_config
from memories.consolidation import ConsolidationResult
from memories.retrieval import RecallResult
from memories.synapses import SynapseManager

from tests.conftest import insert_atom, insert_synapse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def brain(storage, mock_embeddings):
    """Provide a Brain wired to real Storage + managers but mocked engines."""
    b = Brain.__new__(Brain)
    b._config = get_config()
    b._storage = storage
    b._embeddings = mock_embeddings
    b._atoms = AtomManager(storage, mock_embeddings)
    b._synapses = SynapseManager(storage)
    b._context = MagicMock()

    # Mock engines that talk to external services or do expensive work.
    b._retrieval = MagicMock()
    b._learning = MagicMock()
    b._consolidation = MagicMock()

    b._current_session_id = "test-session-123"
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

    # Retrieval engine async mock.
    b._retrieval.recall = AsyncMock(return_value=RecallResult())

    # Consolidation engine async mock.
    b._consolidation.reflect = AsyncMock(return_value=ConsolidationResult())

    # Create the session row that brain methods expect to exist.
    await storage.execute_write(
        "INSERT INTO sessions (id, project) VALUES (?, ?)",
        ("test-session-123", None),
    )

    yield b

    # Tear down -- don't call shutdown(); the storage fixture handles close.
    b._initialized = False


# ===========================================================================
# TestBrainInitialization
# ===========================================================================


class TestBrainInitialization:
    """Tests for Brain.__init__, initialize, and _ensure_initialized."""

    async def test_initialized_flag_set_after_init(self, brain):
        """After fixture setup the brain should report initialized."""
        assert brain._initialized is True

    async def test_double_init_is_idempotent(self, brain):
        """Calling initialize when already initialized does nothing."""
        # Mark as initialized (already true), call initialize again.
        # It should return immediately without error.
        await brain.initialize()
        assert brain._initialized is True

    async def test_ensure_initialized_raises_before_init(self):
        """_ensure_initialized raises RuntimeError on a fresh Brain."""
        fresh = Brain.__new__(Brain)
        fresh._initialized = False
        with pytest.raises(RuntimeError, match="Brain not initialized"):
            fresh._ensure_initialized()

    async def test_ensure_initialized_passes_when_ready(self, brain):
        """_ensure_initialized succeeds when the brain is initialized."""
        # Should not raise.
        brain._ensure_initialized()


# ===========================================================================
# TestBrainRemember
# ===========================================================================


class TestBrainRemember:
    """Tests for Brain.remember()."""

    async def test_creates_atom_and_returns_correct_keys(self, brain):
        """remember() creates an atom and returns the expected dict keys."""
        result = await brain.remember(
            content="Redis SCAN is O(N) over the full keyspace",
            type="fact",
            region="technical",
        )

        assert "atom_id" in result
        assert "atom" in result
        assert "synapses_created" in result
        assert "related_atoms" in result
        assert isinstance(result["atom_id"], int)
        assert result["atom"]["content"] == "Redis SCAN is O(N) over the full keyspace"
        assert result["atom"]["type"] == "fact"

    async def test_infers_region_when_none(self, brain):
        """When region is None, the learning engine infers one."""
        result = await brain.remember(
            content="Python uses GIL for thread safety",
            type="fact",
            region=None,
        )

        brain._learning.suggest_region.assert_awaited_once()
        assert result["atom"]["region"] == "technical"

    async def test_skips_region_inference_when_provided(self, brain):
        """When region is explicitly set, suggest_region is NOT called."""
        await brain.remember(
            content="My preferred editor is Neovim",
            type="preference",
            region="personal",
        )

        brain._learning.suggest_region.assert_not_awaited()

    async def test_extracts_antipattern_fields(self, brain):
        """For antipattern type with no severity/instead, extraction runs."""
        result = await brain.remember(
            content="Using SELECT * in production queries is bad",
            type="antipattern",
            region="technical",
        )

        brain._learning.extract_antipattern_fields.assert_awaited_once()
        assert result["atom"]["severity"] == "high"
        assert result["atom"]["instead"] == "use X instead"

    async def test_antipattern_preserves_explicit_fields(self, brain):
        """Explicit severity/instead are NOT overridden by extraction."""
        result = await brain.remember(
            content="Avoid global mutable state",
            type="antipattern",
            region="technical",
            severity="critical",
            instead="use dependency injection",
        )

        brain._learning.extract_antipattern_fields.assert_not_awaited()
        assert result["atom"]["severity"] == "critical"
        assert result["atom"]["instead"] == "use dependency injection"

    async def test_antipattern_partial_extraction(self, brain):
        """When only severity is provided, instead is still extracted."""
        result = await brain.remember(
            content="Never store secrets in code",
            type="antipattern",
            region="technical",
            severity=None,
            instead="use env vars",
        )

        # severity is None so extraction fires, but instead is already set.
        brain._learning.extract_antipattern_fields.assert_awaited_once()
        assert result["atom"]["severity"] == "high"  # extracted
        assert result["atom"]["instead"] == "use env vars"  # kept

    async def test_tracks_session_atom(self, brain):
        """track_atom_access is called for the new atom (no-op after W5-D)."""
        result = await brain.remember(
            content="Go slices are backed by arrays",
            type="fact",
            region="technical",
        )

        # track_atom_access is now a no-op; just verify remember completes.
        assert isinstance(result["atom_id"], int)

    async def test_auto_link_called(self, brain):
        """auto_link is invoked with the new atom's id."""
        result = await brain.remember(
            content="Rust borrow checker prevents data races",
            type="fact",
            region="technical",
        )

        brain._learning.auto_link.assert_awaited_once_with(result["atom_id"])

    async def test_synapses_created_reflects_auto_link_count(self, brain):
        """synapses_created mirrors the number of synapses from auto_link."""
        brain._learning.auto_link = AsyncMock(
            return_value=[
                {"source_id": 1, "target_id": 2, "relationship": "related-to", "strength": 0.5},
                {"source_id": 1, "target_id": 3, "relationship": "elaborates", "strength": 0.4},
            ]
        )

        # Pre-insert atoms so get_without_tracking finds them for related_atoms.
        a2 = await insert_atom(brain._storage, "Related atom 2")
        a3 = await insert_atom(brain._storage, "Related atom 3")

        # Patch auto_link to reference real atom IDs.
        brain._learning.auto_link = AsyncMock(
            return_value=[
                {"source_id": 999, "target_id": a2, "relationship": "related-to", "strength": 0.5},
                {"source_id": 999, "target_id": a3, "relationship": "elaborates", "strength": 0.4},
            ]
        )

        result = await brain.remember(
            content="Test content for synapse count",
            type="fact",
            region="technical",
        )

        # source_id=999 won't match atom.id, so related_id picks 999 for one
        # and atom.id for the other -- but synapses_created count is always 2.
        assert result["synapses_created"] == 2

    async def test_remember_with_tags_and_source(self, brain):
        """Tags and source metadata are stored in the atom."""
        result = await brain.remember(
            content="Use pytest fixtures for test isolation",
            type="skill",
            region="technical",
            tags=["testing", "python"],
            source_project="memories",
            source_file="tests/test_brain.py",
        )

        atom = result["atom"]
        assert atom["tags"] == ["testing", "python"]
        assert atom["source_project"] == "memories"
        assert atom["source_file"] == "tests/test_brain.py"


# ===========================================================================
# TestBrainRecall
# ===========================================================================


class TestBrainRecall:
    """Tests for Brain.recall()."""

    async def test_delegates_to_retrieval_engine(self, brain):
        """recall() passes through to the retrieval engine."""
        await brain.recall(query="How does Redis work?")

        brain._retrieval.recall.assert_awaited_once()
        call_kwargs = brain._retrieval.recall.call_args.kwargs
        assert call_kwargs["query"] == "How does Redis work?"

    async def test_passes_all_parameters(self, brain):
        """All keyword arguments are forwarded to the engine."""
        await brain.recall(
            query="test",
            budget_tokens=500,
            depth=3,
            region="technical",
            types=["fact", "skill"],
            include_antipatterns=False,
        )

        call_kwargs = brain._retrieval.recall.call_args.kwargs
        assert call_kwargs["budget_tokens"] == 500
        assert call_kwargs["depth"] == 3
        assert call_kwargs["region"] == "technical"
        assert call_kwargs["types"] == ["fact", "skill"]
        assert call_kwargs["include_antipatterns"] is False

    async def test_result_has_correct_structure(self, brain):
        """Returned dict contains all RecallResult keys."""
        result = await brain.recall(query="anything")

        expected_keys = {
            "atoms",
            "antipatterns",
            "pathways",
            "budget_used",
            "budget_remaining",
            "total_activated",
            "seed_count",
            "compression_level",
        }
        assert set(result.keys()) == expected_keys

    async def test_recall_completes_with_returned_atoms(self, brain):
        """recall() completes when retrieval returns atoms (W5-D: no _session_atoms)."""
        brain._retrieval.recall = AsyncMock(
            return_value=RecallResult(
                atoms=[{"id": 10, "content": "A"}, {"id": 20, "content": "B"}],
                antipatterns=[{"id": 30, "content": "C"}],
            )
        )

        result = await brain.recall(query="test tracking")

        # Atoms and antipatterns are returned in the result dict.
        assert len(result["atoms"]) == 2
        assert len(result["antipatterns"]) == 1

    async def test_recall_raises_when_not_initialized(self):
        """recall() raises RuntimeError if brain is not initialized."""
        fresh = Brain.__new__(Brain)
        fresh._initialized = False
        with pytest.raises(RuntimeError, match="Brain not initialized"):
            await fresh.recall(query="test")


# ===========================================================================
# TestBrainConnect
# ===========================================================================


class TestBrainConnect:
    """Tests for Brain.connect()."""

    async def test_creates_synapse_between_atoms(self, brain, storage):
        """connect() creates a synapse and returns the expected structure."""
        a1 = await insert_atom(storage, "Source atom content")
        a2 = await insert_atom(storage, "Target atom content")

        result = await brain.connect(
            source_id=a1,
            target_id=a2,
            relationship="related-to",
            strength=0.7,
        )

        assert "synapse_id" in result
        assert "synapse" in result
        assert "source_summary" in result
        assert "target_summary" in result
        assert result["synapse"]["relationship"] == "related-to"
        assert result["synapse"]["strength"] == 0.7

    async def test_raises_for_missing_source_atom(self, brain, storage):
        """connect() raises ValueError when source atom does not exist."""
        a2 = await insert_atom(storage, "Target exists")

        with pytest.raises(ValueError, match="Source atom 99999 not found"):
            await brain.connect(
                source_id=99999,
                target_id=a2,
                relationship="related-to",
            )

    async def test_raises_for_missing_target_atom(self, brain, storage):
        """connect() raises ValueError when target atom does not exist."""
        a1 = await insert_atom(storage, "Source exists")

        with pytest.raises(ValueError, match="Target atom 99999 not found"):
            await brain.connect(
                source_id=a1,
                target_id=99999,
                relationship="related-to",
            )

    async def test_source_summary_is_truncated(self, brain, storage):
        """source_summary is truncated to 120 characters."""
        long_content = "A" * 200
        a1 = await insert_atom(storage, long_content)
        a2 = await insert_atom(storage, "Short target")

        result = await brain.connect(
            source_id=a1, target_id=a2, relationship="elaborates"
        )

        assert len(result["source_summary"]) == 120


# ===========================================================================
# TestBrainForget
# ===========================================================================


class TestBrainForget:
    """Tests for Brain.forget()."""

    async def test_soft_delete_default(self, brain, storage):
        """Default forget() soft-deletes the atom."""
        aid = await insert_atom(storage, "To be soft-deleted")

        result = await brain.forget(atom_id=aid)

        assert result["status"] == "deleted"
        assert result["atom_id"] == aid
        assert result["hard"] is False

    async def test_hard_delete(self, brain, storage):
        """forget(hard=True) permanently removes atom and synapses."""
        a1 = await insert_atom(storage, "To be hard-deleted")
        a2 = await insert_atom(storage, "Linked atom")
        await insert_synapse(storage, a1, a2, relationship="related-to")

        result = await brain.forget(atom_id=a1, hard=True)

        assert result["status"] == "deleted"
        assert result["hard"] is True
        assert result["synapses_affected"] >= 1

    async def test_raises_for_missing_atom(self, brain):
        """forget() raises ValueError when the atom doesn't exist."""
        with pytest.raises(ValueError, match="Atom 99999 not found"):
            await brain.forget(atom_id=99999)

    async def test_soft_delete_reports_synapse_count(self, brain, storage):
        """Soft-delete reports the number of connected synapses."""
        a1 = await insert_atom(storage, "Node with links")
        a2 = await insert_atom(storage, "Neighbor 1")
        a3 = await insert_atom(storage, "Neighbor 2")
        await insert_synapse(storage, a1, a2)
        await insert_synapse(storage, a1, a3)

        result = await brain.forget(atom_id=a1)

        # Two synapses touch a1 (one as source to a2, one as source to a3).
        assert result["synapses_affected"] == 2


# ===========================================================================
# TestBrainAmend
# ===========================================================================


class TestBrainAmend:
    """Tests for Brain.amend()."""

    async def test_updates_content_triggers_relink(self, brain, storage):
        """Changing content triggers auto_link re-run."""
        aid = await insert_atom(storage, "Original content")

        result = await brain.amend(atom_id=aid, content="Updated content")

        brain._learning.auto_link.assert_awaited_with(aid)
        assert result["atom"]["content"] == "Updated content"
        assert "new_synapses" in result
        assert "removed_synapses" in result

    async def test_no_content_change_skips_relink(self, brain, storage):
        """When content is None, auto_link is NOT called."""
        aid = await insert_atom(storage, "Keep this content")

        # Reset mock to track calls precisely.
        brain._learning.auto_link.reset_mock()

        result = await brain.amend(atom_id=aid, confidence=0.8)

        brain._learning.auto_link.assert_not_awaited()
        assert result["new_synapses"] == []
        assert result["removed_synapses"] == []

    async def test_raises_for_missing_atom(self, brain):
        """amend() raises ValueError when the atom doesn't exist."""
        with pytest.raises(ValueError, match="Atom 99999 not found"):
            await brain.amend(atom_id=99999, content="new")

    async def test_updates_type_field(self, brain, storage):
        """Updating just the type field works without content change."""
        aid = await insert_atom(storage, "A fact that is really a skill")

        result = await brain.amend(atom_id=aid, type="skill")

        assert result["atom"]["type"] == "skill"

    async def test_updates_tags(self, brain, storage):
        """Tags can be replaced via amend."""
        aid = await insert_atom(storage, "Tag me", tags=["old"])

        result = await brain.amend(atom_id=aid, tags=["new", "tags"])

        assert result["atom"]["tags"] == ["new", "tags"]


# ===========================================================================
# TestBrainReflect
# ===========================================================================


class TestBrainReflect:
    """Tests for Brain.reflect()."""

    async def test_delegates_to_consolidation(self, brain, storage):
        """reflect() calls consolidation engine and returns its result."""
        storage.optimize = AsyncMock()

        result = await brain.reflect(scope="all", dry_run=False)

        brain._consolidation.reflect.assert_awaited_once_with(
            scope="all", dry_run=False
        )
        assert "merged" in result
        assert "decayed" in result
        assert "pruned" in result
        assert "promoted" in result

    async def test_dry_run_skips_optimize(self, brain, storage):
        """In dry_run mode, storage.optimize is NOT called."""
        storage.optimize = AsyncMock()

        await brain.reflect(scope="all", dry_run=True)

        storage.optimize.assert_not_awaited()

    async def test_real_run_calls_optimize(self, brain, storage):
        """Non-dry-run calls storage.optimize after consolidation."""
        storage.optimize = AsyncMock()

        await brain.reflect(scope="all", dry_run=False)

        storage.optimize.assert_awaited_once()


# ===========================================================================
# TestBrainSession
# ===========================================================================


class TestBrainSession:
    """Tests for session tracking, end_session, and shutdown."""

    async def test_track_atom_access_is_noop(self, brain):
        """track_atom_access is a no-op after W5-D (does not raise)."""
        brain.track_atom_access(1)
        brain.track_atom_access(2)
        brain.track_atom_access(3)

        # No _session_atoms attribute should exist.
        assert not hasattr(brain, "_session_atoms")

    async def test_track_atom_access_no_duplicates(self, brain):
        """Repeated calls to track_atom_access are safe no-ops."""
        brain.track_atom_access(42)
        brain.track_atom_access(42)
        brain.track_atom_access(42)

        # No state is stored.
        assert not hasattr(brain, "_session_atoms")

    async def test_end_session_calls_hebbian_learning(self, brain, storage):
        """end_session invokes session_end_learning with atoms from DB."""
        session_id = brain._current_session_id
        for aid in [10, 20]:
            await storage.execute_write(
                "INSERT OR IGNORE INTO hook_session_atoms "
                "(claude_session_id, atom_id) VALUES (?, ?)",
                (session_id, aid),
            )

        await brain.end_session()

        brain._learning.session_end_learning.assert_awaited_once()
        call_args = brain._learning.session_end_learning.call_args
        assert set(call_args[0][1]) == {10, 20}

    async def test_end_session_resets_state(self, brain, storage):
        """After end_session, session ID is cleared."""
        await brain.end_session()

        assert brain._current_session_id is None

    async def test_end_session_returns_stats(self, brain, storage):
        """end_session returns a dict with the expected keys."""
        session_id = brain._current_session_id
        await storage.execute_write(
            "INSERT OR IGNORE INTO hook_session_atoms "
            "(claude_session_id, atom_id) VALUES (?, ?)",
            (session_id, 1),
        )
        result = await brain.end_session()

        assert result["session_id"] == "test-session-123"
        assert result["atoms_accessed"] == 1
        assert "synapses_strengthened" in result
        assert "synapses_created" in result

    async def test_end_session_skips_learning_when_no_atoms(self, brain, storage):
        """If no atoms in hook_session_atoms, session_end_learning is NOT called."""
        brain._learning.session_end_learning.reset_mock()

        await brain.end_session()

        brain._learning.session_end_learning.assert_not_awaited()

    async def test_shutdown_is_idempotent(self, brain, storage):
        """Calling shutdown twice does not raise."""
        # First shutdown ends the session.
        await brain.shutdown()
        assert brain._initialized is False

        # Second shutdown is a no-op.
        await brain.shutdown()
        assert brain._initialized is False

    async def test_shutdown_ends_session(self, brain, storage):
        """shutdown() triggers end_session before closing."""
        session_id = brain._current_session_id
        await storage.execute_write(
            "INSERT OR IGNORE INTO hook_session_atoms "
            "(claude_session_id, atom_id) VALUES (?, ?)",
            (session_id, 7),
        )

        await brain.shutdown()

        brain._learning.session_end_learning.assert_awaited_once()
        assert brain._initialized is False

    async def test_shutdown_without_session(self, brain, storage):
        """shutdown() is safe when session was already ended."""
        brain._current_session_id = None

        await brain.shutdown()

        assert brain._initialized is False


class TestBrainCreateTask:
    """Tests for Brain.create_task — regression for auto_link unpacking crash."""

    async def test_create_task_returns_dict(self, brain, storage):
        """create_task must return a dict with expected keys without crashing."""
        result = await brain.create_task(content="Fix the login bug", status="pending")

        assert isinstance(result, dict)
        assert "atom_id" in result
        assert isinstance(result["synapses_created"], int)
        assert isinstance(result["related_atoms"], list)

    async def test_create_task_stores_task_atom(self, brain, storage):
        """The created atom must have type='task' and the given status."""
        result = await brain.create_task(content="Deploy to production", status="active")

        atom_id = result["atom_id"]
        rows = await storage.execute(
            "SELECT type, task_status FROM atoms WHERE id = ?", (atom_id,)
        )
        assert rows[0]["type"] == "task"
        assert rows[0]["task_status"] == "active"

    async def test_create_task_related_atoms_are_dicts(self, brain, storage):
        """related_atoms must be a list of dicts (not Atom objects)."""
        # Store a related atom first so auto_link has something to connect to.
        await brain.remember(content="Login bug is in auth.py", type="fact")
        result = await brain.create_task(content="Fix the login bug in auth.py")

        for item in result["related_atoms"]:
            assert isinstance(item, dict)
            assert "id" in item
            assert "content" in item


# ===========================================================================
# TestSessionAtomsDict (W3-C)
# ===========================================================================


class TestSessionAtomsRemoved:
    """Tests for W5-D: _session_atoms dict removed, replaced by hook_session_atoms DB."""

    async def test_no_session_atoms_attribute(self, brain):
        """After Brain setup, _session_atoms must not exist (removed in W5-D)."""
        assert not hasattr(brain, "_session_atoms"), (
            "Brain still has _session_atoms attribute -- should be removed"
        )

    async def test_track_atom_access_is_noop(self, brain):
        """track_atom_access is a no-op and does not create _session_atoms."""
        brain.track_atom_access(99)
        brain.track_atom_access(99)
        brain.track_atom_access(99)

        assert not hasattr(brain, "_session_atoms")

    async def test_session_atoms_in_db_deduplicate(self, brain, storage):
        """hook_session_atoms table enforces PRIMARY KEY deduplication."""
        session_id = brain._current_session_id
        for _ in range(3):
            await storage.execute_write(
                "INSERT OR IGNORE INTO hook_session_atoms "
                "(claude_session_id, atom_id) VALUES (?, ?)",
                (session_id, 99),
            )
        rows = await storage.execute(
            "SELECT atom_id FROM hook_session_atoms WHERE claude_session_id = ?",
            (session_id,),
        )
        assert len(rows) == 1
        assert rows[0]["atom_id"] == 99
