"""Tests for automatic antipattern category classification.

Covers:
- Keyword-based classification (tier 1)
- Similarity voting from existing categorised antipatterns (tier 2)
- Brain integration (category injected on remember)
- Consolidation backfill of uncategorised antipatterns
- CLI formatting with category labels
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.atoms import AtomManager
from memories.brain import Brain
from memories.cli import _format_atom_line
from memories.consolidation import ConsolidationEngine, ConsolidationResult
from memories.embeddings import EmbeddingEngine
from memories.learning import (
    ANTIPATTERN_CATEGORY_PREFIX,
    LearningEngine,
    _CATEGORY_KEYWORDS,
)
from memories.storage import Storage
from memories.synapses import SynapseManager
from tests.conftest import insert_atom


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_learning_engine(
    storage: Storage,
    embeddings: EmbeddingEngine | None = None,
) -> LearningEngine:
    """Build a LearningEngine with a mocked EmbeddingEngine."""
    if embeddings is None:
        embeddings = MagicMock(spec=EmbeddingEngine)
        embeddings.search_similar = AsyncMock(return_value=[])
        embeddings.embed_and_store = AsyncMock()

    atoms = AtomManager(storage, embeddings)
    synapses = SynapseManager(storage)
    return LearningEngine(storage, embeddings, atoms, synapses)


def _make_consolidation_engine(
    storage: Storage,
    embeddings: EmbeddingEngine | None = None,
) -> ConsolidationEngine:
    """Build a ConsolidationEngine with a mocked EmbeddingEngine."""
    if embeddings is None:
        embeddings = MagicMock(spec=EmbeddingEngine)
        embeddings.search_similar = AsyncMock(return_value=[])
        embeddings.embed_and_store = AsyncMock()

    atoms = AtomManager(storage, embeddings)
    synapses = SynapseManager(storage)
    return ConsolidationEngine(storage, embeddings, atoms, synapses)


# -----------------------------------------------------------------------
# 1. Keyword classification
# -----------------------------------------------------------------------


class TestKeywordClassification:
    """Verify tier-1 keyword matching assigns the correct category."""

    async def test_security_keywords(self, storage: Storage) -> None:
        engine = _make_learning_engine(storage)
        result = await engine.suggest_antipattern_category(
            "Never hardcode API tokens or credentials in source code", None
        )
        assert result == "security"

    async def test_data_loss_keywords(self, storage: Storage) -> None:
        engine = _make_learning_engine(storage)
        result = await engine.suggest_antipattern_category(
            "Running rm -rf without confirmation can destroy data", None
        )
        assert result == "data-loss"

    async def test_rate_limit_keywords(self, storage: Storage) -> None:
        engine = _make_learning_engine(storage)
        result = await engine.suggest_antipattern_category(
            "API returns 429 when you exceed the rate limit", None
        )
        assert result == "rate-limit"

    async def test_performance_keywords(self, storage: Storage) -> None:
        engine = _make_learning_engine(storage)
        result = await engine.suggest_antipattern_category(
            "N+1 queries cause slow page loads", None
        )
        assert result == "performance"

    async def test_concurrency_keywords(self, storage: Storage) -> None:
        engine = _make_learning_engine(storage)
        result = await engine.suggest_antipattern_category(
            "Race condition when two threads write to the same file", None
        )
        assert result == "concurrency"

    async def test_case_insensitive(self, storage: Storage) -> None:
        engine = _make_learning_engine(storage)
        result = await engine.suggest_antipattern_category(
            "NEVER expose SECRET keys in logs", None
        )
        assert result == "security"

    async def test_existing_category_tag_respected(self, storage: Storage) -> None:
        """Tier 0: if the caller already tagged a category, return it as-is."""
        engine = _make_learning_engine(storage)
        result = await engine.suggest_antipattern_category(
            "Some generic content",
            ["category:custom-cat"],
        )
        assert result == "custom-cat"

    async def test_no_match_falls_through(self, storage: Storage) -> None:
        """Content with no keyword match and no similar atoms → general."""
        engine = _make_learning_engine(storage)
        result = await engine.suggest_antipattern_category(
            "This is a totally unique situation with no recognized patterns", None
        )
        assert result == "general"


# -----------------------------------------------------------------------
# 2. Similarity voting
# -----------------------------------------------------------------------


class TestSimilarityVoting:
    """Verify tier-2 similarity voting from existing categorised antipatterns."""

    async def test_voting_from_similar_antipatterns(self, storage: Storage) -> None:
        """Majority vote from 3 similar categorised antipatterns wins."""
        # Insert 3 antipatterns with category:security tags.
        for i in range(3):
            await insert_atom(
                storage,
                content=f"Never store passwords in plaintext (variant {i})",
                atom_type="antipattern",
                tags=["category:security"],
            )

        mock_emb = MagicMock(spec=EmbeddingEngine)
        # Return the 3 antipattern IDs as similar results.
        mock_emb.search_similar = AsyncMock(return_value=[
            (1, 0.1), (2, 0.1), (3, 0.1),
        ])
        mock_emb.embed_and_store = AsyncMock()

        engine = _make_learning_engine(storage, mock_emb)
        result = await engine.suggest_antipattern_category(
            "A mysterious problem with no keyword matches", None
        )
        assert result == "security"

    async def test_ignores_non_antipatterns(self, storage: Storage) -> None:
        """Non-antipattern atoms should not contribute votes."""
        # Insert a fact with a category tag (shouldn't count).
        await insert_atom(
            storage,
            content="Redis is a cache",
            atom_type="fact",
            tags=["category:performance"],
        )

        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.search_similar = AsyncMock(return_value=[(1, 0.1)])
        mock_emb.embed_and_store = AsyncMock()

        engine = _make_learning_engine(storage, mock_emb)
        result = await engine.suggest_antipattern_category(
            "Some unique antipattern content", None
        )
        assert result == "general"

    async def test_empty_similar_falls_back(self, storage: Storage) -> None:
        """No similar atoms at all → fallback to general."""
        engine = _make_learning_engine(storage)
        result = await engine.suggest_antipattern_category(
            "Completely novel antipattern", None
        )
        assert result == "general"


# -----------------------------------------------------------------------
# 3. Brain integration
# -----------------------------------------------------------------------


@pytest.mark.integration
class TestBrainIntegration:
    """Verify that Brain.remember() injects category tags for antipatterns."""

    async def test_category_injected_on_remember(self, brain: Brain) -> None:
        """Antipatterns get a category: tag automatically."""
        result = await brain.remember(
            content="Never commit .env files with secrets",
            type="antipattern",
        )
        atom = result["atom"]
        tags = atom.get("tags") or []
        category_tags = [t for t in tags if t.startswith("category:")]
        assert len(category_tags) == 1
        assert category_tags[0] == "category:security"

    async def test_non_antipattern_not_categorised(self, brain: Brain) -> None:
        """Non-antipattern atoms should not get a category tag."""
        result = await brain.remember(
            content="Redis uses an in-memory data structure store",
            type="fact",
        )
        atom = result["atom"]
        tags = atom.get("tags") or []
        category_tags = [t for t in tags if t.startswith("category:")]
        assert len(category_tags) == 0

    async def test_caller_category_not_overwritten(self, brain: Brain) -> None:
        """If the caller provides a category: tag, it must not be replaced."""
        result = await brain.remember(
            content="Never commit .env files",
            type="antipattern",
            tags=["category:custom"],
        )
        atom = result["atom"]
        tags = atom.get("tags") or []
        category_tags = [t for t in tags if t.startswith("category:")]
        assert len(category_tags) == 1
        assert category_tags[0] == "category:custom"


# -----------------------------------------------------------------------
# 4. Consolidation backfill
# -----------------------------------------------------------------------


class TestConsolidationBackfill:
    """Verify _classify_uncategorised_antipatterns backfills categories."""

    async def test_classifies_uncategorised(self, storage: Storage) -> None:
        """Antipatterns without category: get one during consolidation."""
        atom_id = await insert_atom(
            storage,
            content="Never store credentials in plain text",
            atom_type="antipattern",
        )

        engine = _make_consolidation_engine(storage)
        result = ConsolidationResult()
        await engine._classify_uncategorised_antipatterns(result)

        rows = await storage.execute(
            "SELECT tags FROM atoms WHERE id = ?", (atom_id,)
        )
        tags = json.loads(rows[0]["tags"])
        category_tags = [t for t in tags if t.startswith("category:")]
        assert len(category_tags) == 1
        assert category_tags[0] == "category:security"
        assert result.categorised == 1

    async def test_skips_already_classified(self, storage: Storage) -> None:
        """Antipatterns that already have category: are not re-processed."""
        await insert_atom(
            storage,
            content="Never store credentials in plain text",
            atom_type="antipattern",
            tags=["category:security"],
        )

        engine = _make_consolidation_engine(storage)
        result = ConsolidationResult()
        await engine._classify_uncategorised_antipatterns(result)

        assert result.categorised == 0

    async def test_dry_run_no_write(self, storage: Storage) -> None:
        """Dry run should count but not update the database."""
        atom_id = await insert_atom(
            storage,
            content="Race condition in concurrent writes",
            atom_type="antipattern",
        )

        engine = _make_consolidation_engine(storage)
        result = ConsolidationResult(dry_run=True)
        await engine._classify_uncategorised_antipatterns(result)

        assert result.categorised == 1

        # DB should be unchanged.
        rows = await storage.execute(
            "SELECT tags FROM atoms WHERE id = ?", (atom_id,)
        )
        raw = rows[0]["tags"]
        assert raw is None or "category:" not in raw

    async def test_batch_size_respected(self, storage: Storage) -> None:
        """Only backfill_batch_size atoms should be processed per cycle."""
        for i in range(5):
            await insert_atom(
                storage,
                content=f"Avoid hardcoded secrets variant {i}",
                atom_type="antipattern",
            )

        with patch("memories.consolidation.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.antipattern_classification.enabled = True
            cfg.antipattern_classification.similarity_voting_enabled = False
            cfg.antipattern_classification.backfill_batch_size = 2
            mock_cfg.return_value = cfg

            engine = _make_consolidation_engine(storage)
            result = ConsolidationResult()
            await engine._classify_uncategorised_antipatterns(result)

        assert result.categorised == 2


# -----------------------------------------------------------------------
# 5. CLI formatting
# -----------------------------------------------------------------------


class TestFormatAtomLine:
    """Verify _format_atom_line shows category labels."""

    def test_category_shown(self) -> None:
        atom = {
            "content": "Never commit .env files",
            "type": "antipattern",
            "severity": "high",
            "id": 42,
            "confidence": 1.0,
            "tags": ["category:security"],
        }
        line = _format_atom_line(atom)
        assert "[security]" in line
        assert "[KNOWN MISTAKE]" in line

    def test_no_category_no_label(self) -> None:
        atom = {
            "content": "Never commit .env files",
            "type": "antipattern",
            "severity": "high",
            "id": 42,
            "confidence": 1.0,
            "tags": [],
        }
        line = _format_atom_line(atom)
        assert "[security]" not in line
        assert "[KNOWN MISTAKE]" in line

    def test_no_tags_key(self) -> None:
        atom = {
            "content": "Avoid X",
            "type": "antipattern",
            "severity": "low",
            "id": 99,
            "confidence": 1.0,
        }
        line = _format_atom_line(atom)
        assert "[warning]" in line
        # No crash, no category label.
        assert "category" not in line
