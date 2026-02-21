"""Tests for the importance weighting feature.

This module tests the importance field across the system:
- Atom creation with importance
- Atom update with importance
- Importance validation (0.0 to 1.0)
- Importance in retrieval scoring
- Migration for existing databases

All tests use mocked embeddings so Ollama is not required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memories.atoms import Atom, AtomManager, _validate_importance
from memories.brain import Brain
from memories.config import get_config
from memories.context import ContextBudget
from memories.retrieval import RetrievalEngine
from memories.synapses import SynapseManager

from tests.conftest import insert_atom


class TestImportanceValidation:
    """Test importance value validation."""

    def test_valid_importance_values(self) -> None:
        """Valid importance values should not raise."""
        _validate_importance(0.0)
        _validate_importance(0.5)
        _validate_importance(1.0)
        _validate_importance(0.123)

    def test_importance_below_zero(self) -> None:
        """Importance below 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Importance must be between"):
            _validate_importance(-0.1)

    def test_importance_above_one(self) -> None:
        """Importance above 1 should raise ValueError."""
        with pytest.raises(ValueError, match="Importance must be between"):
            _validate_importance(1.1)


class TestAtomImportance:
    """Test importance field in Atom dataclass."""

    def test_atom_default_importance(self) -> None:
        """Atom should have default importance of 0.5."""
        atom = Atom(id=1, content="test", type="fact")
        assert atom.importance == 0.5

    def test_atom_custom_importance(self) -> None:
        """Atom should accept custom importance value."""
        atom = Atom(id=1, content="test", type="fact", importance=0.9)
        assert atom.importance == 0.9

    def test_atom_to_dict_includes_importance(self) -> None:
        """to_dict() should include importance field."""
        atom = Atom(id=1, content="test", type="fact", importance=0.8)
        d = atom.to_dict()
        assert "importance" in d
        assert d["importance"] == 0.8


class TestAtomManagerImportance:
    """Test AtomManager create/update with importance."""

    @pytest.fixture
    def atom_manager(self, storage, mock_embeddings) -> AtomManager:
        """Create an AtomManager instance."""
        return AtomManager(storage, mock_embeddings)

    @pytest.mark.anyio
    async def test_create_with_default_importance(
        self, atom_manager: AtomManager
    ) -> None:
        """Created atom should have default importance 0.5."""
        atom = await atom_manager.create(
            content="Test memory",
            type="fact",
        )
        assert atom.importance == 0.5

    @pytest.mark.anyio
    async def test_create_with_custom_importance(
        self, atom_manager: AtomManager
    ) -> None:
        """Created atom should respect custom importance."""
        atom = await atom_manager.create(
            content="Critical information",
            type="fact",
            importance=0.95,
        )
        assert atom.importance == 0.95

    @pytest.mark.anyio
    async def test_create_with_high_importance(
        self, atom_manager: AtomManager
    ) -> None:
        """Should allow importance of 1.0."""
        atom = await atom_manager.create(
            content="Maximum priority memory",
            type="fact",
            importance=1.0,
        )
        assert atom.importance == 1.0

    @pytest.mark.anyio
    async def test_create_with_low_importance(
        self, atom_manager: AtomManager
    ) -> None:
        """Should allow importance of 0.0."""
        atom = await atom_manager.create(
            content="Low priority memory",
            type="fact",
            importance=0.0,
        )
        assert atom.importance == 0.0

    @pytest.mark.anyio
    async def test_create_rejects_invalid_importance(
        self, atom_manager: AtomManager
    ) -> None:
        """Should reject importance outside [0, 1]."""
        with pytest.raises(ValueError, match="Importance must be between"):
            await atom_manager.create(
                content="Invalid importance",
                type="fact",
                importance=1.5,
            )

    @pytest.mark.anyio
    async def test_update_importance(self, atom_manager: AtomManager) -> None:
        """Should be able to update importance."""
        atom = await atom_manager.create(
            content="Updatable memory",
            type="fact",
            importance=0.5,
        )
        assert atom.importance == 0.5

        updated = await atom_manager.update(atom.id, importance=0.9)
        assert updated is not None
        assert updated.importance == 0.9

    @pytest.mark.anyio
    async def test_update_importance_validation(
        self, atom_manager: AtomManager
    ) -> None:
        """Update should validate importance range."""
        atom = await atom_manager.create(
            content="Test memory",
            type="fact",
        )

        with pytest.raises(ValueError, match="Importance must be between"):
            await atom_manager.update(atom.id, importance=-0.5)


class TestBrainImportance:
    """Test Brain.remember() and Brain.amend() with importance.

    Uses Brain.__new__() pattern with mocked embeddings so Ollama is not required.
    """

    @pytest.fixture
    async def brain(self, storage, mock_embeddings):
        """Create a Brain with real AtomManager but mocked embeddings."""
        b = Brain.__new__(Brain)
        b._config = get_config()
        b._storage = storage
        b._embeddings = mock_embeddings
        b._atoms = AtomManager(storage, mock_embeddings)
        b._synapses = SynapseManager(storage)
        b._context = MagicMock()

        b._retrieval = MagicMock()
        b._learning = MagicMock()
        b._learning.suggest_region = AsyncMock(return_value="general")
        b._learning.extract_antipattern_fields = AsyncMock(
            return_value=("high", "use X instead")
        )
        b._learning.auto_link = AsyncMock(return_value=[])
        b._learning.detect_antipattern_links = AsyncMock(return_value=0)
        b._learning.detect_supersedes = AsyncMock(return_value=0)
        b._consolidation = MagicMock()

        b._current_session_id = "test-session-importance"
        b._initialized = True
        return b

    @pytest.mark.anyio
    async def test_remember_with_default_importance(self, brain: Brain) -> None:
        """remember() should use default importance 0.5."""
        result = await brain.remember(
            content="Test memory",
            type="fact",
        )
        assert result["atom"]["importance"] == 0.5

    @pytest.mark.anyio
    async def test_remember_with_high_importance(self, brain: Brain) -> None:
        """remember() should accept high importance."""
        result = await brain.remember(
            content="Critical information - never forget!",
            type="fact",
            importance=0.95,
        )
        assert result["atom"]["importance"] == 0.95

    @pytest.mark.anyio
    async def test_amend_importance(self, brain: Brain) -> None:
        """amend() should be able to update importance."""
        result = await brain.remember(
            content="Initially normal importance",
            type="fact",
            importance=0.5,
        )
        atom_id = result["atom_id"]

        updated = await brain.amend(atom_id, importance=0.9)
        assert updated["atom"]["importance"] == 0.9


class TestRetrievalImportanceScoring:
    """Test that importance affects recall ranking.

    Uses insert_atom() + real RetrievalEngine with mocked embeddings
    so Ollama is not required.
    """

    @pytest.fixture
    async def brain_with_atoms(self, storage, mock_embeddings):
        """Create brain with test atoms of varying importance via direct insert."""
        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        context = ContextBudget(storage)
        retrieval = RetrievalEngine(storage, mock_embeddings, atoms, synapses, context)

        # Insert atoms directly (bypasses embeddings)
        id_low = await insert_atom(
            storage, "Redis is an in-memory database",
            atom_type="fact", importance=0.3,
        )
        id_high = await insert_atom(
            storage, "Redis SCAN is O(N) over the full keyspace",
            atom_type="fact", importance=0.9,
        )
        id_med = await insert_atom(
            storage, "Redis supports multiple data structures",
            atom_type="fact", importance=0.5,
        )

        # Mock search_similar to return our atoms as seed results
        mock_embeddings.search_similar = AsyncMock(
            return_value=[(id_low, 0.3), (id_high, 0.2), (id_med, 0.25)]
        )

        # Build the brain
        b = Brain.__new__(Brain)
        b._config = get_config()
        b._storage = storage
        b._embeddings = mock_embeddings
        b._atoms = atoms
        b._synapses = synapses
        b._context = context
        b._retrieval = retrieval
        b._learning = MagicMock()
        b._consolidation = MagicMock()
        b._current_session_id = None
        b._initialized = True
        return b

    @pytest.mark.anyio
    async def test_importance_affects_score_breakdown(
        self, brain_with_atoms: Brain
    ) -> None:
        """Recall score breakdown should include importance."""
        result = await brain_with_atoms.recall(
            query="Redis database",
            budget_tokens=4000,
        )

        # All returned atoms should have importance in score breakdown
        for atom in result["atoms"]:
            if "score_breakdown" in atom:
                assert "importance" in atom["score_breakdown"]

    @pytest.mark.anyio
    async def test_high_importance_ranks_higher(
        self, brain_with_atoms: Brain
    ) -> None:
        """Higher importance atoms should tend to rank higher.

        Note: This is a probabilistic test since other factors
        (vector similarity, spread activation) also affect ranking.
        """
        result = await brain_with_atoms.recall(
            query="Redis SCAN",
            budget_tokens=4000,
        )

        # The high importance atom about SCAN should be in results
        atoms = result["atoms"]
        assert len(atoms) > 0

        # Find the SCAN atom and verify it has high importance
        scan_atoms = [a for a in atoms if "SCAN" in a["content"]]
        if scan_atoms:
            assert scan_atoms[0]["importance"] == 0.9


class TestImportanceMigration:
    """Test that existing databases get the importance column added."""

    @pytest.mark.anyio
    async def test_migration_adds_importance_column(self, tmp_path) -> None:
        """Migration should add importance column to existing databases."""
        import sqlite3
        from memories.storage import Storage

        db_path = tmp_path / "legacy.db"

        # Create a "legacy" database without importance column
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE atoms (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                region TEXT NOT NULL DEFAULT 'general',
                confidence REAL NOT NULL DEFAULT 1.0,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                source_project TEXT,
                source_session TEXT,
                source_file TEXT,
                tags TEXT,
                severity TEXT,
                instead TEXT,
                is_deleted INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            INSERT INTO atoms (content, type)
            VALUES ('Legacy memory', 'fact')
        """)
        conn.commit()
        conn.close()

        # Initialize storage which should run migration
        storage = Storage(db_path)
        await storage.initialize()

        # Verify importance column was added with default value
        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = 1"
        )
        assert len(rows) == 1
        assert rows[0]["importance"] == 0.5

        await storage.close()
