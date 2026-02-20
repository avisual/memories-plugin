"""Wave 8-B tests: supersedes decay exemption, idempotent supersedes creation,
outbound degree cap in auto_link, misclassified antipattern reclassification,
and get_stats SQL consolidation.

B1: Exempt supersedes from _decay_synapses
B2: Only create supersedes in _resolve_contradictions if absent
B3: Add outbound degree cap in auto_link()
B4: Expand _reclassify_antipatterns to catch short misclassified atoms
B5: atoms.get_stats() collapse to 2 SQL queries
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.atoms import AtomManager
from memories.consolidation import ConsolidationEngine, ConsolidationResult
from memories.config import get_config
from memories.embeddings import EmbeddingEngine
from memories.learning import LearningEngine
from memories.synapses import SynapseManager

from tests.conftest import count_synapses, insert_atom, insert_synapse


# -----------------------------------------------------------------------
# B1: Exempt supersedes from _decay_synapses
# -----------------------------------------------------------------------


class TestSupersedesDecayExempt:
    """B1: supersedes synapses are provenance pointers and must never decay.
    After the fix, _decay_synapses should leave supersedes strength unchanged.
    """

    async def test_supersedes_not_decayed(self, storage, mock_embeddings) -> None:
        """supersedes synapses should retain their original strength after decay."""
        a1 = await insert_atom(storage, "winner atom")
        a2 = await insert_atom(storage, "loser atom")

        sid = await insert_synapse(
            storage, a1, a2, relationship="supersedes", strength=0.8
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = ConsolidationEngine(storage, mock_embeddings, atoms, synapses)

        result = ConsolidationResult()
        await engine._decay_synapses(result)

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (sid,)
        )
        assert rows[0]["strength"] == pytest.approx(0.8), (
            "supersedes synapse was decayed; it should be exempt from _decay_synapses"
        )

    async def test_related_to_still_decayed(self, storage, mock_embeddings) -> None:
        """related-to synapses should still be decayed normally (sanity check)."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")

        sid = await insert_synapse(
            storage, a1, a2, relationship="related-to", strength=0.8
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = ConsolidationEngine(storage, mock_embeddings, atoms, synapses)

        result = ConsolidationResult()
        await engine._decay_synapses(result)

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (sid,)
        )
        assert rows[0]["strength"] < 0.8, (
            "related-to synapse was NOT decayed; decay should still apply"
        )

    async def test_other_typed_synapses_still_decayed(self, storage, mock_embeddings) -> None:
        """Non-exempt typed synapses (e.g. caused-by) should still decay."""
        a1 = await insert_atom(storage, "cause")
        a2 = await insert_atom(storage, "effect")

        sid = await insert_synapse(
            storage, a1, a2, relationship="caused-by", strength=0.8
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = ConsolidationEngine(storage, mock_embeddings, atoms, synapses)

        result = ConsolidationResult()
        await engine._decay_synapses(result)

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (sid,)
        )
        assert rows[0]["strength"] < 0.8, (
            "caused-by synapse was NOT decayed; only related-to and supersedes "
            "should be exempt"
        )


# -----------------------------------------------------------------------
# B2: Only create supersedes in _resolve_contradictions if absent
# -----------------------------------------------------------------------


class TestIdempotentSupersedesCreation:
    """B2: _resolve_contradictions should not recreate an existing supersedes
    synapse. Repeated cycles should not cause the BCM upsert to re-strengthen
    a supersedes link to ~1.0.
    """

    async def _setup_contradiction(self, storage, mock_embeddings):
        """Create two contradicting atoms old enough for resolution, with
        a clear winner (high access_count + confidence) and loser.
        """
        cfg = get_config().consolidation
        min_age = cfg.contradiction_min_age_days

        a_winner = await insert_atom(
            storage,
            "Use postgres for everything",
            confidence=0.9,
            access_count=100,
            created_at="2020-01-01T00:00:00+00:00",
        )
        a_loser = await insert_atom(
            storage,
            "Use mysql for everything",
            confidence=0.3,
            access_count=2,
            created_at="2020-01-01T00:00:00+00:00",
        )

        # Create contradicts synapse
        await insert_synapse(
            storage, a_winner, a_loser, relationship="contradicts", strength=0.9
        )

        return a_winner, a_loser

    async def test_supersedes_created_when_absent(self, storage, mock_embeddings) -> None:
        """First resolution cycle should create the supersedes synapse."""
        a_winner, a_loser = await self._setup_contradiction(storage, mock_embeddings)

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = ConsolidationEngine(storage, mock_embeddings, atoms, synapses)

        result = ConsolidationResult()
        await engine._resolve_contradictions(result)

        cnt = await count_synapses(
            storage, source_id=a_winner, target_id=a_loser, relationship="supersedes"
        )
        assert cnt == 1, "supersedes synapse should be created on first resolution"

    async def test_supersedes_not_recreated_when_present(self, storage, mock_embeddings) -> None:
        """When a supersedes synapse already exists, _resolve_contradictions
        should skip creation. The existing strength should not be re-upserted.
        """
        a_winner, a_loser = await self._setup_contradiction(storage, mock_embeddings)

        # Pre-create the supersedes synapse with a specific strength
        await insert_synapse(
            storage, a_winner, a_loser, relationship="supersedes", strength=0.6
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = ConsolidationEngine(storage, mock_embeddings, atoms, synapses)

        result = ConsolidationResult()
        await engine._resolve_contradictions(result)

        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE source_id = ? AND target_id = ? "
            "AND relationship = 'supersedes'",
            (a_winner, a_loser),
        )
        assert len(rows) == 1, "should still have exactly one supersedes synapse"
        assert rows[0]["strength"] == pytest.approx(0.6), (
            "supersedes strength was changed; _resolve_contradictions should "
            "skip creation when the synapse already exists"
        )


# -----------------------------------------------------------------------
# B3: Outbound degree cap in auto_link()
# -----------------------------------------------------------------------


class TestOutboundDegreeCap:
    """B3: auto_link() should respect an outbound degree cap of 50 for
    related-to synapses, mirroring the existing inbound cap.
    """

    async def test_outbound_cap_blocks_new_links(self, storage, mock_embeddings) -> None:
        """An atom already at 50 outbound related-to should not get more."""
        # Create the source atom
        source_id = await insert_atom(storage, "hub source atom")

        # Create 50 target atoms and wire outbound related-to
        for i in range(50):
            target = await insert_atom(storage, f"target {i}")
            await insert_synapse(
                storage, source_id, target, relationship="related-to", strength=0.5
            )

        # Create a new candidate that would be linked
        new_target = await insert_atom(storage, "new candidate target")

        # Mock embeddings to return the new_target as similar
        mock_embeddings.search_similar = AsyncMock(
            return_value=[(new_target, 0.1)]  # distance 0.1 -> similarity ~0.9
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        learning = LearningEngine(storage, mock_embeddings, atoms, synapses)

        created = await learning.auto_link(source_id)

        # Count outbound related-to after auto_link
        outbound = await count_synapses(
            storage, source_id=source_id, relationship="related-to"
        )
        assert outbound <= 50, (
            f"Outbound related-to exceeded cap: {outbound} > 50. "
            "auto_link must enforce an outbound degree cap."
        )

    async def test_outbound_cap_allows_under_limit(self, storage, mock_embeddings) -> None:
        """An atom under the cap should still get new related-to links."""
        source_id = await insert_atom(storage, "source atom under cap")

        # Only 10 existing outbound
        for i in range(10):
            target = await insert_atom(storage, f"existing target {i}")
            await insert_synapse(
                storage, source_id, target, relationship="related-to", strength=0.5
            )

        new_target = await insert_atom(storage, "new candidate")

        mock_embeddings.search_similar = AsyncMock(
            return_value=[(new_target, 0.1)]  # similarity ~0.9
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        learning = LearningEngine(storage, mock_embeddings, atoms, synapses)

        created = await learning.auto_link(source_id)

        outbound = await count_synapses(
            storage, source_id=source_id, relationship="related-to"
        )
        assert outbound == 11, (
            f"Expected 11 outbound related-to (10 existing + 1 new), got {outbound}. "
            "auto_link should allow links when under the outbound cap."
        )

    async def test_outbound_cap_does_not_block_other_types(self, storage, mock_embeddings) -> None:
        """The outbound cap only applies to related-to, not warns-against etc."""
        source_id = await insert_atom(storage, "source atom", atom_type="antipattern")

        # Fill to 50 outbound related-to
        for i in range(50):
            target = await insert_atom(storage, f"target {i}")
            await insert_synapse(
                storage, source_id, target, relationship="related-to", strength=0.5
            )

        new_target = await insert_atom(storage, "new candidate for warns-against")

        mock_embeddings.search_similar = AsyncMock(
            return_value=[(new_target, 0.1)]
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        learning = LearningEngine(storage, mock_embeddings, atoms, synapses)

        created = await learning.auto_link(source_id)

        # warns-against should still be created even though related-to is at cap
        warns_count = await count_synapses(
            storage, source_id=source_id, relationship="warns-against"
        )
        assert warns_count >= 1, (
            "warns-against synapse should still be created when outbound "
            "related-to is at cap"
        )


# -----------------------------------------------------------------------
# B4: Expand _reclassify_antipatterns to catch misclassified atoms
# -----------------------------------------------------------------------


class TestReclassifyMisclassifiedAntipatterns:
    """B4: Short antipatterns that lack negative keywords should be
    reclassified to 'experience' and their warns-against links deleted.
    """

    async def test_short_non_negative_antipattern_reclassified(
        self, storage, mock_embeddings
    ) -> None:
        """An antipattern with short, neutral content should become experience."""
        aid = await insert_atom(
            storage,
            "A single line of code was edited",
            atom_type="antipattern",
            confidence=1.0,
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = ConsolidationEngine(storage, mock_embeddings, atoms, synapses)

        result = ConsolidationResult()
        await engine._reclassify_antipatterns(result)

        rows = await storage.execute("SELECT type FROM atoms WHERE id = ?", (aid,))
        assert rows[0]["type"] == "experience", (
            "Short neutral antipattern should be reclassified to experience"
        )

    async def test_warns_against_deleted_on_reclassify(
        self, storage, mock_embeddings
    ) -> None:
        """When reclassified, outbound warns-against synapses should be deleted."""
        aid = await insert_atom(
            storage,
            "The command failed with exit code 1",
            atom_type="antipattern",
            confidence=1.0,
        )
        target = await insert_atom(storage, "some other atom")
        await insert_synapse(
            storage, aid, target, relationship="warns-against", strength=0.7
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = ConsolidationEngine(storage, mock_embeddings, atoms, synapses)

        result = ConsolidationResult()
        await engine._reclassify_antipatterns(result)

        cnt = await count_synapses(
            storage, source_id=aid, relationship="warns-against"
        )
        assert cnt == 0, (
            "warns-against synapses from reclassified atom should be deleted"
        )

    async def test_genuine_antipattern_not_reclassified(
        self, storage, mock_embeddings
    ) -> None:
        """Antipatterns containing negative keywords should NOT be reclassified."""
        test_cases = [
            "You should not use eval() in production",
            "Avoid using global variables",
            "Never commit secrets to git",
            "This is a common mistake in async code",
            "Using raw SQL is a bad practice and dangerous",
            "This is an antipattern that causes bugs",
            "Don't use mutable default args",
            "Do not hardcode passwords",
        ]

        atom_ids = []
        for content in test_cases:
            aid = await insert_atom(
                storage,
                content,
                atom_type="antipattern",
                confidence=1.0,
            )
            atom_ids.append(aid)

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = ConsolidationEngine(storage, mock_embeddings, atoms, synapses)

        result = ConsolidationResult()
        await engine._reclassify_antipatterns(result)

        for aid, content in zip(atom_ids, test_cases):
            rows = await storage.execute(
                "SELECT type FROM atoms WHERE id = ?", (aid,)
            )
            assert rows[0]["type"] == "antipattern", (
                f"Genuine antipattern should NOT be reclassified: {content!r}"
            )

    async def test_long_antipattern_not_reclassified(
        self, storage, mock_embeddings
    ) -> None:
        """Antipatterns with content >= 150 chars should NOT be reclassified
        by the short-content pass (even if they lack negative keywords).
        """
        long_content = "x" * 150  # exactly 150 chars, no negative keywords
        aid = await insert_atom(
            storage,
            long_content,
            atom_type="antipattern",
            confidence=1.0,
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        engine = ConsolidationEngine(storage, mock_embeddings, atoms, synapses)

        result = ConsolidationResult()
        await engine._reclassify_antipatterns(result)

        rows = await storage.execute("SELECT type FROM atoms WHERE id = ?", (aid,))
        assert rows[0]["type"] == "antipattern", (
            "Antipattern with content >= 150 chars should NOT be reclassified "
            "by the short-content pass"
        )


# -----------------------------------------------------------------------
# B5: atoms.get_stats() collapse to 2 SQL queries
# -----------------------------------------------------------------------


class TestGetStatsCollapsed:
    """B5: get_stats() should produce the same results as before but use
    only 2 SQL queries instead of 5.
    """

    async def test_stats_correctness(self, storage, mock_embeddings) -> None:
        """Verify get_stats returns correct totals, type breakdown, region
        breakdown, avg confidence, and deleted count.
        """
        await insert_atom(storage, "fact 1", atom_type="fact", confidence=0.8, region="general")
        await insert_atom(storage, "fact 2", atom_type="fact", confidence=0.6, region="general")
        await insert_atom(storage, "exp 1", atom_type="experience", confidence=1.0, region="errors")
        await insert_atom(storage, "skill 1", atom_type="skill", confidence=0.4, region="errors")
        await insert_atom(storage, "deleted", atom_type="fact", is_deleted=True)

        atoms = AtomManager(storage, mock_embeddings)
        stats = await atoms.get_stats()

        assert stats["total"] == 4
        assert stats["total_deleted"] == 1
        assert stats["by_type"]["fact"] == 2
        assert stats["by_type"]["experience"] == 1
        assert stats["by_type"]["skill"] == 1
        assert stats["by_region"]["general"] == 2
        assert stats["by_region"]["errors"] == 2
        # avg confidence = (0.8 + 0.6 + 1.0 + 0.4) / 4 = 0.7
        assert stats["avg_confidence"] == pytest.approx(0.7, abs=0.01)

    async def test_stats_empty_db(self, storage, mock_embeddings) -> None:
        """get_stats on an empty database should return zeros."""
        atoms = AtomManager(storage, mock_embeddings)
        stats = await atoms.get_stats()

        assert stats["total"] == 0
        assert stats["total_deleted"] == 0
        assert stats["by_type"] == {}
        assert stats["by_region"] == {}
        assert stats["avg_confidence"] == 0.0

    async def test_stats_uses_two_queries(self, storage, mock_embeddings) -> None:
        """Verify get_stats makes exactly 2 SQL calls."""
        await insert_atom(storage, "fact 1", atom_type="fact")

        atoms = AtomManager(storage, mock_embeddings)
        original_execute = storage.execute
        call_count = 0

        async def counting_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original_execute(*args, **kwargs)

        storage.execute = counting_execute
        try:
            await atoms.get_stats()
        finally:
            storage.execute = original_execute

        assert call_count == 2, (
            f"get_stats should use exactly 2 SQL queries, but used {call_count}"
        )
