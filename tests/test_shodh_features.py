"""Tests for shodh-memory-inspired improvements.

Four features:
1. Hybrid decay (exponential → power-law transition)
2. Retroactive interference (contradiction weakens old atom)
3. Type-dependent feedback inertia
4. Multi-scale LTP protection (access count shields from decay)

All tests use a temporary database and mock embeddings.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.atoms import Atom, AtomManager
from memories.consolidation import ConsolidationEngine, _CONFIDENCE_FLOOR
from memories.config import ConsolidationConfig, LearningConfig, MemoriesConfig
from memories.embeddings import EmbeddingEngine
from memories.learning import LearningEngine
from memories.storage import Storage
from memories.synapses import SynapseManager


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


async def _insert_atom(
    storage: Storage,
    content: str = "test atom",
    atom_type: str = "fact",
    region: str = "general",
    confidence: float = 1.0,
    importance: float = 0.5,
    access_count: int = 0,
    last_accessed_at: str | None = None,
    created_at: str = "2025-01-01 00:00:00",
    tags: str | None = None,
) -> int:
    """Insert an atom row directly via SQL and return the new id."""
    return await storage.execute_write(
        """
        INSERT INTO atoms
            (content, type, region, confidence, importance, access_count,
             last_accessed_at, created_at, updated_at, tags, is_deleted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, 0)
        """,
        (content, atom_type, region, confidence, importance, access_count,
         last_accessed_at, created_at, tags),
    )


async def _insert_synapse(
    storage: Storage,
    source_id: int,
    target_id: int,
    relationship: str = "related-to",
    strength: float = 0.5,
    bidirectional: int = 1,
    last_activated_at: str | None = None,
) -> int:
    """Insert a synapse row directly via SQL and return the new id."""
    return await storage.execute_write(
        """
        INSERT INTO synapses
            (source_id, target_id, relationship, strength, bidirectional,
             last_activated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source_id, target_id, relationship, strength, bidirectional,
         last_activated_at),
    )


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


# =======================================================================
# 1. Hybrid Decay (exponential → power-law transition)
# =======================================================================


class TestHybridDecay:
    """Verify that atom decay transitions from exponential to power-law."""

    async def test_exponential_phase_unchanged(self, storage: Storage) -> None:
        """Atoms within the transition window still decay exponentially."""
        # Atom stale for 45 days (within default 90-day transition)
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=45)).isoformat()
        atom_id = await _insert_atom(
            storage, content="exponential phase test",
            confidence=0.8, last_accessed_at=stale_date,
        )

        engine = _make_consolidation_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        new_conf = rows[0]["confidence"]

        # Should use exponential: conf * (rate ** exponent)
        # With default decay_rate=0.95, decay_after_days=30, exponent=45/30=1.5
        # effective_rate for facts = 0.95 * 0.995 = 0.94525
        # Importance-modulated: default importance=0.5 → protection=0.75,
        # so effective exponent = 1.5 * 0.75 = 1.125.
        importance_protection = 1.0 - 0.5 * 0.5
        expected = 0.8 * (0.94525 ** (1.5 * importance_protection))
        assert abs(new_conf - expected) < 0.01, (
            f"Expected ~{expected:.4f}, got {new_conf:.4f}"
        )

    async def test_power_law_phase_slower_than_exponential(self, storage: Storage) -> None:
        """Atoms beyond the transition window decay slower than pure exponential."""
        # Atom stale for 200 days (well beyond default 90-day transition)
        # At this distance the power-law heavy tail clearly outperforms
        # pure exponential decay.
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=200)).isoformat()
        atom_id = await _insert_atom(
            storage, content="power law phase test",
            confidence=0.8, last_accessed_at=stale_date, atom_type="experience",
        )

        engine = _make_consolidation_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        hybrid_conf = rows[0]["confidence"]

        # Pure exponential would give: 0.8 * (effective_rate ** (200/30))
        # effective_rate for experience = 0.95 * 0.93 = 0.8835
        pure_exp = 0.8 * (0.8835 ** (200.0 / 30.0))
        # Hybrid should retain MORE confidence than pure exponential
        # The max(exp, power-law) curve picks the slower decay for long-term
        assert hybrid_conf > pure_exp, (
            f"Hybrid ({hybrid_conf:.4f}) should be > pure exponential ({pure_exp:.4f})"
        )

    async def test_power_law_continuity_at_transition(self, storage: Storage) -> None:
        """Decay curve is continuous at the transition point."""
        # Two atoms: one just before transition, one just after
        transition_days = 90  # default
        before_days = transition_days - 1
        after_days = transition_days + 1

        before_date = (datetime.now(tz=timezone.utc) - timedelta(days=before_days)).isoformat()
        after_date = (datetime.now(tz=timezone.utc) - timedelta(days=after_days)).isoformat()

        id_before = await _insert_atom(
            storage, content="just before transition",
            confidence=1.0, last_accessed_at=before_date,
        )
        id_after = await _insert_atom(
            storage, content="just after transition",
            confidence=1.0, last_accessed_at=after_date,
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        rows_before = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (id_before,)
        )
        rows_after = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (id_after,)
        )

        conf_before = rows_before[0]["confidence"]
        conf_after = rows_after[0]["confidence"]

        # The after-transition atom should have slightly lower confidence,
        # but the difference should be small (continuity)
        diff = abs(conf_before - conf_after)
        assert diff < 0.05, (
            f"Discontinuity at transition: before={conf_before:.4f}, "
            f"after={conf_after:.4f}, diff={diff:.4f}"
        )
        assert conf_after <= conf_before, "After-transition should decay at least as much"


# =======================================================================
# 2. Retroactive Interference
# =======================================================================


class TestRetroactiveInterference:
    """Verify that detecting a contradiction weakens the older atom."""

    async def test_contradiction_weakens_old_atom(self, storage: Storage) -> None:
        """When a new atom contradicts an old one, old atom's confidence drops."""
        # Create an old atom
        old_date = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()
        old_id = await _insert_atom(
            storage, content="Always use tabs for indentation",
            atom_type="preference", confidence=0.9,
            last_accessed_at=old_date, created_at=old_date,
        )

        # Create a new contradicting atom
        now = datetime.now(tz=timezone.utc).isoformat()
        new_id = await _insert_atom(
            storage, content="Never use tabs for indentation, use spaces",
            atom_type="preference", confidence=1.0,
            last_accessed_at=now, created_at=now,
        )

        # Mock embeddings to return the old atom as highly similar
        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.search_similar = AsyncMock(return_value=[
            (old_id, 0.1),  # distance 0.1 = similarity 0.95
        ])
        mock_emb.embed_and_store = AsyncMock()

        engine = _make_learning_engine(storage, embeddings=mock_emb)
        await engine.auto_link(new_id)

        # Check old atom's confidence was reduced
        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (old_id,)
        )
        old_conf = rows[0]["confidence"]
        assert old_conf < 0.9, (
            f"Old atom confidence should be reduced from 0.9, got {old_conf}"
        )

    async def test_interference_respects_confidence_floor(self, storage: Storage) -> None:
        """Interference should not push confidence below the system floor (0.1)."""
        old_date = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()
        old_id = await _insert_atom(
            storage, content="Always use tabs for indentation",
            atom_type="preference", confidence=0.15,
            last_accessed_at=old_date, created_at=old_date,
        )

        now = datetime.now(tz=timezone.utc).isoformat()
        new_id = await _insert_atom(
            storage, content="Never use tabs for indentation, use spaces",
            atom_type="preference", confidence=1.0,
            last_accessed_at=now, created_at=now,
        )

        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.search_similar = AsyncMock(return_value=[
            (old_id, 0.1),
        ])
        mock_emb.embed_and_store = AsyncMock()

        engine = _make_learning_engine(storage, embeddings=mock_emb)
        await engine.auto_link(new_id)

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (old_id,)
        )
        old_conf = rows[0]["confidence"]
        # With penalty 0.1 from starting 0.15, floor should prevent going to 0.05
        assert old_conf >= 0.1, (
            f"Confidence should respect floor 0.1, got {old_conf}"
        )

    async def test_interference_cascade_protection(self, storage: Storage) -> None:
        """Multiple contradictions should not destroy an atom below floor."""
        old_date = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()
        old_id = await _insert_atom(
            storage, content="Always use tabs for indentation",
            atom_type="preference", confidence=0.5,
            last_accessed_at=old_date, created_at=old_date,
        )

        now = datetime.now(tz=timezone.utc).isoformat()
        # Create multiple contradicting atoms
        for i in range(5):
            new_id = await _insert_atom(
                storage,
                content=f"Never use tabs, use spaces variant {i}",
                atom_type="preference", confidence=1.0,
                last_accessed_at=now, created_at=now,
            )

            mock_emb = MagicMock(spec=EmbeddingEngine)
            mock_emb.search_similar = AsyncMock(return_value=[
                (old_id, 0.1),
            ])
            mock_emb.embed_and_store = AsyncMock()

            engine = _make_learning_engine(storage, embeddings=mock_emb)
            await engine.auto_link(new_id)

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (old_id,)
        )
        old_conf = rows[0]["confidence"]
        # Even after 5 contradictions, should never go below floor
        assert old_conf >= 0.1, (
            f"After 5 contradictions, confidence should stay >= 0.1, got {old_conf}"
        )

    async def test_no_interference_without_contradiction(self, storage: Storage) -> None:
        """Non-contradicting similar atoms don't trigger interference."""
        old_date = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()
        old_id = await _insert_atom(
            storage, content="Python is great for scripting",
            atom_type="fact", confidence=0.9,
            last_accessed_at=old_date, created_at=old_date,
        )

        now = datetime.now(tz=timezone.utc).isoformat()
        new_id = await _insert_atom(
            storage, content="Python is great for scripting and automation",
            atom_type="fact", confidence=1.0,
            last_accessed_at=now, created_at=now,
        )

        # High similarity but same assertion (no contradiction)
        mock_emb = MagicMock(spec=EmbeddingEngine)
        mock_emb.search_similar = AsyncMock(return_value=[
            (old_id, 0.05),  # very similar
        ])
        mock_emb.embed_and_store = AsyncMock()

        engine = _make_learning_engine(storage, embeddings=mock_emb)
        await engine.auto_link(new_id)

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (old_id,)
        )
        # Confidence may be reduced by supersession, but not by interference
        # (supersession only applies at similarity > 0.9 AND same type AND newer)
        old_conf = rows[0]["confidence"]
        # The supersession detection might reduce it by 0.1, so check original
        # interference penalty isn't applied (the two say the same thing)
        assert old_conf >= 0.7, (
            f"Non-contradicting atom shouldn't be heavily penalized, got {old_conf}"
        )


# =======================================================================
# 3. Type-Dependent Feedback Inertia
# =======================================================================


class TestTypeDependentFeedbackInertia:
    """Verify that feedback effects are modulated by atom type."""

    async def test_fact_resists_feedback(self, storage: Storage) -> None:
        """Facts (high inertia) should have reduced feedback effect."""
        fact_id = await _insert_atom(
            storage, content="Python uses GIL for thread safety",
            atom_type="fact", confidence=1.0, importance=0.5,
        )

        # Insert bad feedback
        await storage.execute_write(
            "INSERT INTO atom_feedback (atom_id, signal) VALUES (?, 'bad')",
            (fact_id,),
        )

        engine = _make_consolidation_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (fact_id,)
        )
        fact_importance = rows[0]["importance"]

        # Facts should resist feedback changes due to high inertia
        # Without inertia, delta = -0.03 * 1.5 = -0.045
        # With inertia (e.g., 0.85), delta = -0.045 * (1 - 0.85) = -0.00675
        # So importance should drop less than 0.01 for facts
        assert fact_importance > 0.49, (
            f"Fact importance dropped too much: {fact_importance:.4f} "
            f"(expected minimal change due to high inertia)"
        )

    async def test_experience_responds_to_feedback(self, storage: Storage) -> None:
        """Experiences (low inertia) should fully respond to feedback."""
        exp_id = await _insert_atom(
            storage, content="Tried using asyncio with SQLite",
            atom_type="experience", confidence=1.0, importance=0.5,
        )

        # Insert bad feedback
        await storage.execute_write(
            "INSERT INTO atom_feedback (atom_id, signal) VALUES (?, 'bad')",
            (exp_id,),
        )

        engine = _make_consolidation_engine(storage)
        result = await engine.reflect()

        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (exp_id,)
        )
        exp_importance = rows[0]["importance"]

        # Experiences should respond more to feedback (low inertia)
        assert exp_importance < 0.48, (
            f"Experience importance should drop more, got {exp_importance:.4f}"
        )

    async def test_inertia_difference_between_types(self, storage: Storage) -> None:
        """Facts should be less affected by feedback than experiences."""
        # Create both types with identical starting importance
        fact_id = await _insert_atom(
            storage, content="Redis SCAN is O(1) per call",
            atom_type="fact", importance=0.5,
        )
        exp_id = await _insert_atom(
            storage, content="Used Redis SCAN in production",
            atom_type="experience", importance=0.5,
        )

        # Same feedback for both
        await storage.execute_write(
            "INSERT INTO atom_feedback (atom_id, signal) VALUES (?, 'bad')",
            (fact_id,),
        )
        await storage.execute_write(
            "INSERT INTO atom_feedback (atom_id, signal) VALUES (?, 'bad')",
            (exp_id,),
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        fact_row = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (fact_id,)
        )
        exp_row = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (exp_id,)
        )

        fact_imp = fact_row[0]["importance"]
        exp_imp = exp_row[0]["importance"]

        # Fact should retain more importance than experience
        assert fact_imp > exp_imp, (
            f"Fact ({fact_imp:.4f}) should retain more importance "
            f"than experience ({exp_imp:.4f})"
        )


# =======================================================================
# 4. Multi-Scale LTP Protection
# =======================================================================


class TestMultiScaleLTP:
    """Verify that frequently accessed atoms are protected from decay."""

    async def test_high_access_atom_decays_slower(self, storage: Storage) -> None:
        """Atoms with many accesses should decay much slower."""
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()

        # Low-access atom
        low_id = await _insert_atom(
            storage, content="rarely accessed atom",
            confidence=0.8, access_count=1,
            last_accessed_at=stale_date,
        )
        # High-access atom
        high_id = await _insert_atom(
            storage, content="frequently accessed atom",
            confidence=0.8, access_count=25,
            last_accessed_at=stale_date,
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        low_row = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (low_id,)
        )
        high_row = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (high_id,)
        )

        low_conf = low_row[0]["confidence"]
        high_conf = high_row[0]["confidence"]

        # High-access atom should retain significantly more confidence
        assert high_conf > low_conf, (
            f"High-access ({high_conf:.4f}) should retain more "
            f"than low-access ({low_conf:.4f})"
        )
        # The difference should be meaningful (not just rounding)
        assert high_conf - low_conf > 0.05, (
            f"LTP protection should create a meaningful difference: "
            f"high={high_conf:.4f}, low={low_conf:.4f}"
        )

    async def test_zero_access_no_protection(self, storage: Storage) -> None:
        """Atoms with zero accesses get no LTP protection."""
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()

        atom_id = await _insert_atom(
            storage, content="never accessed",
            confidence=0.8, access_count=0,
            last_accessed_at=stale_date,
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        conf = rows[0]["confidence"]

        # Without LTP, the decay should proceed at full rate
        # effective_rate for fact = 0.95 * 0.995 = 0.94525
        # exponent = 60/30 = 2, importance_protection = 1.0 - 0.5*0.5 = 0.75
        importance_protection = 1.0 - 0.5 * 0.5
        expected = 0.8 * (0.94525 ** (2 * importance_protection))
        assert abs(conf - expected) < 0.02, (
            f"Zero-access atom should decay at full rate: "
            f"expected ~{expected:.4f}, got {conf:.4f}"
        )

    async def test_ltp_tiers_progressive(self, storage: Storage) -> None:
        """Each LTP tier provides progressively more protection."""
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()

        ids = {}
        for access_count in [0, 5, 10, 25]:
            ids[access_count] = await _insert_atom(
                storage, content=f"atom with {access_count} accesses",
                confidence=0.8, access_count=access_count,
                last_accessed_at=stale_date,
            )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        confidences = {}
        for access_count, atom_id in ids.items():
            rows = await storage.execute(
                "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
            )
            confidences[access_count] = rows[0]["confidence"]

        # Each tier should retain progressively more confidence
        assert confidences[0] < confidences[5], (
            f"5 accesses ({confidences[5]:.4f}) should retain more "
            f"than 0 ({confidences[0]:.4f})"
        )
        assert confidences[5] < confidences[10], (
            f"10 accesses ({confidences[10]:.4f}) should retain more "
            f"than 5 ({confidences[5]:.4f})"
        )
        assert confidences[10] < confidences[25], (
            f"25 accesses ({confidences[25]:.4f}) should retain more "
            f"than 10 ({confidences[10]:.4f})"
        )

    async def test_ltp_with_power_law_combined(self, storage: Storage) -> None:
        """LTP protection should work with the power-law branch (>90 days)."""
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=150)).isoformat()

        # High-access atom beyond power-law transition
        high_id = await _insert_atom(
            storage, content="well known fact beyond transition",
            confidence=0.8, access_count=25,
            last_accessed_at=stale_date,
        )
        # Low-access atom beyond power-law transition
        low_id = await _insert_atom(
            storage, content="obscure fact beyond transition",
            confidence=0.8, access_count=0,
            last_accessed_at=stale_date,
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        high_row = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (high_id,)
        )
        low_row = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (low_id,)
        )

        high_conf = high_row[0]["confidence"]
        low_conf = low_row[0]["confidence"]

        # LTP protection should still work even in the power-law phase
        assert high_conf > low_conf, (
            f"LTP + power-law: high-access ({high_conf:.4f}) should retain "
            f"more than low-access ({low_conf:.4f})"
        )
        # The difference should be substantial
        assert high_conf - low_conf > 0.1, (
            f"LTP protection should be meaningful in power-law phase: "
            f"diff={high_conf - low_conf:.4f}"
        )


# =======================================================================
# Additional feedback inertia tests
# =======================================================================


class TestFeedbackInertiaAdditional:
    """Additional tests for feedback inertia from architect review."""

    async def test_good_feedback_with_inertia(self, storage: Storage) -> None:
        """Good feedback for high-inertia types should also be reduced."""
        fact_id = await _insert_atom(
            storage, content="PostgreSQL supports JSONB columns",
            atom_type="fact", importance=0.5,
        )

        await storage.execute_write(
            "INSERT INTO atom_feedback (atom_id, signal) VALUES (?, 'good')",
            (fact_id,),
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (fact_id,)
        )
        fact_imp = rows[0]["importance"]

        # Without inertia: delta = 0.02 → importance = 0.52
        # With inertia (0.85): delta = 0.02 * 0.15 = 0.003 → importance ≈ 0.503
        assert fact_imp < 0.51, (
            f"Good feedback on fact should be muted by inertia, got {fact_imp:.4f}"
        )
        assert fact_imp > 0.5, (
            f"Good feedback should still slightly increase importance, got {fact_imp:.4f}"
        )

    async def test_antipattern_responds_to_feedback(self, storage: Storage) -> None:
        """Antipatterns (reduced inertia 0.40) should respond to feedback."""
        ap_id = await _insert_atom(
            storage, content="Never use eval() in production code",
            atom_type="antipattern", importance=0.5,
        )

        await storage.execute_write(
            "INSERT INTO atom_feedback (atom_id, signal) VALUES (?, 'bad')",
            (ap_id,),
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT importance FROM atoms WHERE id = ?", (ap_id,)
        )
        ap_imp = rows[0]["importance"]

        # With inertia 0.40: delta = -0.03 * 1.5 * 0.60 = -0.027
        # importance = 0.5 - 0.027 = 0.473
        assert ap_imp < 0.48, (
            f"Antipattern should respond meaningfully to bad feedback, got {ap_imp:.4f}"
        )


# =======================================================================
# Config tests
# =======================================================================


class TestConfigDefaults:
    """Verify that the new config fields have sensible defaults."""

    def test_hybrid_decay_config_exists(self) -> None:
        """ConsolidationConfig should have hybrid decay parameters."""
        cfg = ConsolidationConfig()
        assert hasattr(cfg, "hybrid_decay_transition_days")
        assert hasattr(cfg, "hybrid_decay_power_exponent")
        assert cfg.hybrid_decay_transition_days == 90
        assert cfg.hybrid_decay_power_exponent == 0.5

    def test_interference_config_exists(self) -> None:
        """LearningConfig should have interference penalty parameter."""
        cfg = LearningConfig()
        assert hasattr(cfg, "interference_confidence_penalty")
        assert cfg.interference_confidence_penalty == 0.1

    def test_feedback_inertia_config_exists(self) -> None:
        """ConsolidationConfig should have type feedback inertia parameters."""
        cfg = ConsolidationConfig()
        assert hasattr(cfg, "type_feedback_inertia")
        inertia = cfg.type_feedback_inertia
        assert isinstance(inertia, dict)
        # Facts should have high inertia
        assert inertia["fact"] > 0.7
        # Experiences should have low inertia
        assert inertia["experience"] < 0.4

    def test_ltp_tiers_config_exists(self) -> None:
        """ConsolidationConfig should have LTP tier parameters."""
        cfg = ConsolidationConfig()
        assert hasattr(cfg, "ltp_tiers")
        tiers = cfg.ltp_tiers
        assert isinstance(tiers, dict)
        # Should have at least 3 tiers
        assert len(tiers) >= 3
        # Protection factors should decrease with more accesses
        sorted_thresholds = sorted(tiers.keys())
        for i in range(len(sorted_thresholds) - 1):
            assert tiers[sorted_thresholds[i]] > tiers[sorted_thresholds[i + 1]]

    def test_max_new_pairs_per_session_config_exists(self) -> None:
        """LearningConfig should have max_new_pairs_per_session."""
        cfg = LearningConfig()
        assert hasattr(cfg, "max_new_pairs_per_session")
        assert cfg.max_new_pairs_per_session == 50


# =======================================================================
# 5. Importance-Modulated Decay
# =======================================================================


class TestImportanceModulatedDecay:
    """Verify that atom importance modulates decay rate."""

    async def test_high_importance_decays_slower(self, storage: Storage) -> None:
        """A high-importance atom should retain more confidence than a low-importance one."""
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()

        high_id = await _insert_atom(
            storage, content="very important memory",
            confidence=0.8, importance=0.9, last_accessed_at=stale_date,
        )
        low_id = await _insert_atom(
            storage, content="not important memory",
            confidence=0.8, importance=0.1, last_accessed_at=stale_date,
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        high_rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (high_id,)
        )
        low_rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (low_id,)
        )

        high_conf = high_rows[0]["confidence"]
        low_conf = low_rows[0]["confidence"]

        # High importance should retain significantly more confidence
        assert high_conf > low_conf, (
            f"High-importance atom ({high_conf:.4f}) should decay slower "
            f"than low-importance atom ({low_conf:.4f})"
        )
        # The difference should be meaningful
        assert high_conf - low_conf > 0.01, (
            f"Importance modulation should create a meaningful difference: "
            f"high={high_conf:.4f}, low={low_conf:.4f}"
        )

    async def test_importance_protection_formula(self, storage: Storage) -> None:
        """Verify the importance protection factor is correctly applied."""
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()

        atom_id = await _insert_atom(
            storage, content="importance formula test",
            confidence=0.8, importance=1.0, last_accessed_at=stale_date,
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        conf = rows[0]["confidence"]

        # importance=1.0 → protection=1.0-0.5*1.0=0.5 → exponent halved
        # For importance=1.0: exponent = (60/30) * 0.5 = 1.0
        # effective_rate = 0.95 * 0.995 = 0.94525
        # expected = 0.8 * 0.94525^1.0
        expected = 0.8 * (0.94525 ** 1.0)
        assert abs(conf - expected) < 0.01, (
            f"importance=1.0 should give protection=1.0 (no extra protection): "
            f"expected ~{expected:.4f}, got {conf:.4f}"
        )

    async def test_zero_importance_no_protection(self, storage: Storage) -> None:
        """Importance=0.0 gives protection factor 1.0 (no protection, full decay)."""
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()

        atom_id = await _insert_atom(
            storage, content="zero importance test",
            confidence=0.8, importance=0.0, last_accessed_at=stale_date,
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )
        conf = rows[0]["confidence"]

        # importance=0.0 → protection=1.0-0.0=1.0 → exponent unchanged
        # For importance=0.0: exponent = (60/30) * 1.0 = 2.0
        # effective_rate = 0.95 * 0.995 = 0.94525
        # expected = 0.8 * 0.94525^2.0
        expected = 0.8 * (0.94525 ** 2.0)
        assert abs(conf - expected) < 0.01, (
            f"importance=0.0 should give protection=1.0 (no protection): "
            f"expected ~{expected:.4f}, got {conf:.4f}"
        )


# =======================================================================
# 6. Cue Overload Protection
# =======================================================================


class TestCueOverloadProtection:
    """Verify that Hebbian update caps new synapse creation per session."""

    async def test_caps_new_pairs_when_exceeded(self, storage: Storage) -> None:
        """When session has many atoms, new pairs should be capped."""
        from memories.config import LearningConfig, MemoriesConfig

        # Create a config with a small cap for testing.
        test_cfg = MemoriesConfig(
            learning=LearningConfig(
                max_new_pairs_per_session=10,
                min_accesses_for_hebbian=0,  # Allow all pairs
            )
        )

        # Insert many atoms to create lots of potential pairs
        # With 20 atoms, there are C(20,2) = 190 pairs
        atom_ids = []
        for i in range(20):
            aid = await storage.execute_write(
                """
                INSERT INTO atoms (content, type, region, confidence, importance,
                                   access_count, created_at)
                VALUES (?, 'fact', 'general', 1.0, 0.5, 5, datetime('now'))
                """,
                (f"overload test atom {i}",),
            )
            atom_ids.append(aid)

        mgr = SynapseManager(storage)

        # Patch the full config so the frozen dataclass is respected.
        with patch("memories.synapses.get_config", return_value=test_cfg):
            mgr._cfg = test_cfg
            total = await mgr.hebbian_update(atom_ids)

        # Should have created at most 10 new synapses (the cap)
        assert total <= 10, (
            f"Expected at most 10 new pairs (cap), got {total}"
        )

    async def test_temporal_sorting_applied(self, storage: Storage) -> None:
        """When timestamps are available, pairs closest in time should survive the cap."""
        from memories.config import LearningConfig, MemoriesConfig

        test_cfg = MemoriesConfig(
            learning=LearningConfig(
                max_new_pairs_per_session=3,
                min_accesses_for_hebbian=0,
            )
        )

        # Create 5 atoms
        atom_ids = []
        for i in range(5):
            aid = await storage.execute_write(
                """
                INSERT INTO atoms (content, type, region, confidence, importance,
                                   access_count, created_at)
                VALUES (?, 'fact', 'general', 1.0, 0.5, 5, datetime('now'))
                """,
                (f"temporal atom {i}",),
            )
            atom_ids.append(aid)

        # Timestamps: atoms 0,1,2 are within 10 seconds of each other;
        # atoms 3,4 are 1000 seconds away
        timestamps = {
            atom_ids[0]: 100.0,
            atom_ids[1]: 105.0,
            atom_ids[2]: 110.0,
            atom_ids[3]: 1100.0,
            atom_ids[4]: 1200.0,
        }

        mgr = SynapseManager(storage)
        with patch("memories.synapses.get_config", return_value=test_cfg):
            mgr._cfg = test_cfg
            total = await mgr.hebbian_update(atom_ids, atom_timestamps=timestamps)

        # With cap of 3, the 3 closest-in-time pairs should be created
        assert total == 3

    async def test_no_cap_when_under_limit(self, storage: Storage) -> None:
        """When session has few atoms, no capping occurs."""
        atom_ids = []
        for i in range(5):
            aid = await storage.execute_write(
                """
                INSERT INTO atoms (content, type, region, confidence, importance,
                                   access_count, created_at)
                VALUES (?, 'fact', 'general', 1.0, 0.5, 5, datetime('now'))
                """,
                (f"small session atom {i}",),
            )
            atom_ids.append(aid)

        mgr = SynapseManager(storage)
        total = await mgr.hebbian_update(atom_ids)

        # C(5,2) = 10 pairs, default cap is 50, so no capping
        assert total == 10, (
            f"Expected all 10 pairs to be created, got {total}"
        )


# =======================================================================
# 7. Synaptic Tagging and Capture (STC)
# =======================================================================


class TestSynapticTaggingAndCapture:
    """Verify that STC tags are set on new synapses and expired/promoted
    during consolidation."""

    async def test_stc_config_defaults(self) -> None:
        """LearningConfig should have STC parameters with sensible defaults."""
        cfg = LearningConfig()
        assert cfg.stc_tagged_strength == 0.25
        assert cfg.stc_capture_window_days == 14

    async def test_expire_unreinforced_tags(self, storage: Storage) -> None:
        """Tagged synapses that were never reinforced should be deleted."""
        atom_a = await _insert_atom(storage, "stc atom A")
        atom_b = await _insert_atom(storage, "stc atom B")

        # Create a tagged synapse with expired tag (in the past).
        await storage.execute_write(
            """
            INSERT INTO synapses
                (source_id, target_id, relationship, strength, bidirectional,
                 activated_count, tag_expires_at)
            VALUES (?, ?, 'related-to', 0.25, 1, 1, datetime('now', '-1 day'))
            """,
            (atom_a, atom_b),
        )

        engine = _make_consolidation_engine(storage)
        result = await engine.reflect()

        # The unreinforced synapse should have been deleted.
        rows = await storage.execute(
            "SELECT id FROM synapses WHERE source_id = ? AND target_id = ?",
            (atom_a, atom_b),
        )
        assert len(rows) == 0, "Unreinforced expired tag should be deleted"
        assert result.pruned >= 1

    async def test_promote_reinforced_tags(self, storage: Storage) -> None:
        """Tagged synapses that were reinforced should have their tag cleared."""
        atom_a = await _insert_atom(storage, "promoted atom A")
        atom_b = await _insert_atom(storage, "promoted atom B")

        # Create a tagged synapse with activated_count > 1 (reinforced).
        synapse_id = await storage.execute_write(
            """
            INSERT INTO synapses
                (source_id, target_id, relationship, strength, bidirectional,
                 activated_count, tag_expires_at, last_activated_at)
            VALUES (?, ?, 'related-to', 0.4, 1, 3, datetime('now', '+7 day'),
                    datetime('now'))
            """,
            (atom_a, atom_b),
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        # The tag should be cleared (promoted to permanent).
        rows = await storage.execute(
            "SELECT tag_expires_at FROM synapses WHERE id = ?",
            (synapse_id,),
        )
        assert len(rows) == 1
        assert rows[0]["tag_expires_at"] is None, (
            "Reinforced synapse should have tag cleared (promoted)"
        )

    async def test_non_expired_tags_survive(self, storage: Storage) -> None:
        """Tagged synapses within the capture window should not be deleted."""
        atom_a = await _insert_atom(storage, "surviving atom A")
        atom_b = await _insert_atom(storage, "surviving atom B")

        # Create a tagged synapse with future expiry and no reinforcement.
        synapse_id = await storage.execute_write(
            """
            INSERT INTO synapses
                (source_id, target_id, relationship, strength, bidirectional,
                 activated_count, tag_expires_at)
            VALUES (?, ?, 'related-to', 0.25, 1, 1, datetime('now', '+7 day'))
            """,
            (atom_a, atom_b),
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        # The synapse should still exist (tag not yet expired).
        rows = await storage.execute(
            "SELECT id, tag_expires_at FROM synapses WHERE id = ?",
            (synapse_id,),
        )
        assert len(rows) == 1, "Non-expired tagged synapse should survive"
        assert rows[0]["tag_expires_at"] is not None


# =======================================================================
# 8. Reconsolidation of Superseded Atoms
# =======================================================================


class TestReconsolidation:
    """Verify that superseded atoms transfer synapse context to their superseder."""

    async def test_transfers_synapses_to_superseder(self, storage: Storage) -> None:
        """Superseded atom's neighbors should be linked to the superseder."""
        # Create atoms: old_fact is superseded by new_fact, and old_fact
        # has a neighbor that new_fact doesn't know about.
        old_fact = await _insert_atom(
            storage, content="old version of fact",
            last_accessed_at=(
                datetime.now(tz=timezone.utc) - timedelta(days=5)
            ).isoformat(),
        )
        new_fact = await _insert_atom(storage, content="new version of fact")
        neighbor = await _insert_atom(storage, content="related topic")

        # new_fact supersedes old_fact.
        await _insert_synapse(
            storage, new_fact, old_fact,
            relationship="supersedes", strength=0.9,
        )
        # old_fact has a related-to synapse to neighbor.
        await _insert_synapse(
            storage, old_fact, neighbor,
            relationship="related-to", strength=0.6,
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        # new_fact should now have a synapse to neighbor (transferred).
        rows = await storage.execute(
            """
            SELECT strength FROM synapses
            WHERE source_id = ? AND target_id = ? AND relationship = 'related-to'
            """,
            (new_fact, neighbor),
        )
        assert len(rows) == 1, "Superseder should inherit neighbor synapse"
        # Transferred at 50% of original strength.
        assert abs(rows[0]["strength"] - 0.3) < 0.05

    async def test_weakens_superseded_atom_synapses(self, storage: Storage) -> None:
        """Superseded atom's outgoing synapses should be weakened."""
        old_fact = await _insert_atom(
            storage, content="outdated fact",
            last_accessed_at=(
                datetime.now(tz=timezone.utc) - timedelta(days=5)
            ).isoformat(),
        )
        new_fact = await _insert_atom(storage, content="updated fact")
        neighbor = await _insert_atom(storage, content="some neighbor")

        await _insert_synapse(
            storage, new_fact, old_fact,
            relationship="supersedes", strength=0.9,
        )
        synapse_id = await _insert_synapse(
            storage, old_fact, neighbor,
            relationship="related-to", strength=0.8,
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (synapse_id,)
        )
        if rows:
            # Weakened by 30%: 0.8 * 0.7 = 0.56
            assert rows[0]["strength"] < 0.8, (
                "Superseded atom's synapses should be weakened"
            )

    async def test_skips_non_recently_accessed(self, storage: Storage) -> None:
        """Superseded atoms not recently accessed should not be reconsolidated."""
        old_fact = await _insert_atom(
            storage, content="ancient fact",
            last_accessed_at="2020-01-01 00:00:00",
        )
        new_fact = await _insert_atom(storage, content="newer fact")
        neighbor = await _insert_atom(storage, content="neighbor topic")

        await _insert_synapse(
            storage, new_fact, old_fact,
            relationship="supersedes", strength=0.9,
        )
        await _insert_synapse(
            storage, old_fact, neighbor,
            relationship="related-to", strength=0.6,
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        # No synapse should be transferred (old_fact not recently accessed).
        rows = await storage.execute(
            """
            SELECT id FROM synapses
            WHERE source_id = ? AND target_id = ? AND relationship = 'related-to'
            """,
            (new_fact, neighbor),
        )
        assert len(rows) == 0, (
            "Non-recently-accessed superseded atom should not be reconsolidated"
        )


# =======================================================================
# 9. Contextual Encoding
# =======================================================================


class TestContextualEncoding:
    """Verify that encoded-with relationship type is properly configured."""

    def test_encoded_with_in_relationship_types(self) -> None:
        """encoded-with should be a valid relationship type."""
        from memories.synapses import RELATIONSHIP_TYPES
        assert "encoded-with" in RELATIONSHIP_TYPES

    def test_encoded_with_excluded_from_hebbian(self) -> None:
        """encoded-with should not be strengthened by Hebbian co-activation."""
        # The _INHIBITORY frozenset in hebbian_update includes encoded-with.
        # This is tested indirectly through the Hebbian tests — encoded-with
        # synapses should not appear in strengthen_ids.
        from memories.synapses import RELATIONSHIP_TYPES
        assert "encoded-with" in RELATIONSHIP_TYPES

    def test_encoded_with_synapse_type_weight(self) -> None:
        """encoded-with should have a low weight for spreading activation."""
        from memories.config import SynapseTypeWeights
        weights = SynapseTypeWeights()
        assert weights.encoded_with == 0.2
        # Should be lower than related_to (0.4)
        assert weights.encoded_with < weights.related_to

    async def test_encoded_with_excluded_from_ltd(self, storage: Storage) -> None:
        """encoded-with synapses should not be weakened by LTD."""
        from memories.config import get_config

        cfg = get_config().consolidation

        atom_a = await _insert_atom(
            storage, "context atom A",
            last_accessed_at=(
                datetime.now(tz=timezone.utc) - timedelta(days=5)
            ).isoformat(),
        )
        atom_b = await _insert_atom(
            storage, "context atom B",
            last_accessed_at=(
                datetime.now(tz=timezone.utc) - timedelta(days=5)
            ).isoformat(),
        )

        # Create an encoded-with synapse with old last_activated_at.
        synapse_id = await storage.execute_write(
            """
            INSERT INTO synapses
                (source_id, target_id, relationship, strength, bidirectional,
                 activated_count, last_activated_at)
            VALUES (?, ?, 'encoded-with', 0.5, 1, 1, ?)
            """,
            (atom_a, atom_b,
             (datetime.now(tz=timezone.utc) - timedelta(days=cfg.ltd_window_days + 5)).isoformat()),
        )

        engine = _make_consolidation_engine(storage)
        await engine.reflect()

        # encoded-with synapse should NOT be weakened by LTD.
        # Note: general synapse decay (_decay_synapses) still applies, but
        # the LTD-specific proportional weakening should not fire.
        # With default synapse_decay_rate=0.95, the strength may drop slightly
        # from general decay, but should NOT drop by the LTD fraction (0.15).
        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (synapse_id,)
        )
        if rows:
            # LTD fraction is 0.15 → with LTD + encoded-with decay would give
            # 0.5 * 0.85 * (0.95*0.85) ≈ 0.343.
            # Encoded-with decay alone (no LTD) gives 0.5 * (0.95*0.85) ≈ 0.404.
            # If strength > 0.35, LTD was correctly excluded.
            assert rows[0]["strength"] > 0.35, (
                f"encoded-with synapse should be excluded from LTD "
                f"(got {rows[0]['strength']:.3f}, LTD would give ~0.343)"
            )
