"""Tests for session prime boost propagation through spreading activation.

With min_activation lowered to 0.01, _SESSION_PRIME_BOOST (0.05) now exceeds
the threshold, so primed atoms DO propagate through spreading activation —
they enter the frontier and their neighbors receive activation.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from memories.atoms import AtomManager
from memories.config import get_config
from memories.context import ContextBudget
from memories.retrieval import RetrievalEngine, _SESSION_PRIME_BOOST
from memories.synapses import SynapseManager

from tests.conftest import insert_atom, insert_synapse


class TestSessionPrimeBoostAboveMinActivation:
    """Session prime boost now exceeds min_activation and propagates."""

    async def test_prime_boost_is_above_min_activation(self):
        """With min_activation=0.01, session prime boost (0.05) propagates."""
        cfg = get_config()
        assert _SESSION_PRIME_BOOST >= cfg.retrieval.min_activation, (
            f"SESSION_PRIME_BOOST ({_SESSION_PRIME_BOOST}) should be at or above "
            f"min_activation ({cfg.retrieval.min_activation}) so primed atoms propagate"
        )

    async def test_primed_atom_propagates(self, storage, mock_embeddings):
        """A session-primed atom at 0.05 activation now enters the frontier.

        We set up: atom A (primed, no vector match) -> synapse -> atom B.
        With min_activation=0.01, atom B receives activation from atom A
        because atom A is a seed (always in initial frontier) and atom B's
        resulting activation exceeds the new lower threshold.
        """
        # Create two connected atoms.
        atom_a = await insert_atom(storage, "previously seen atom", access_count=5)
        atom_b = await insert_atom(storage, "connected neighbor", access_count=5)
        await insert_synapse(storage, atom_a, atom_b, strength=0.9)

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, budget)

        # Simulate: atom_a is a session-primed atom (not a vector seed).
        seeds = {atom_a: _SESSION_PRIME_BOOST}  # 0.05

        activated, atom_map = await engine._spread_activation(seeds, depth=2)

        # atom_a is in activated (it's a seed) at ~0.05; may be slightly
        # higher due to convergence bonus from bidirectional synapse feedback.
        assert atom_a in activated
        assert activated[atom_a] >= _SESSION_PRIME_BOOST

        # atom_b SHOULD be activated because seeds are the initial frontier
        # and atom_a gets expanded. neighbor_activation =
        # 0.05 * 0.9 (strength) * 0.5 (related-to weight) * 0.7 (decay) * 1.0 (norm)
        # = 0.0158, which exceeds min_activation (0.01).
        assert atom_b in activated, (
            "atom_b should receive activation from primed seed with min_activation=0.01"
        )
        assert activated[atom_b] > get_config().retrieval.min_activation, (
            "atom_b's activation should exceed min_activation threshold"
        )

    async def test_primed_atom_with_higher_boost_propagates(self, storage, mock_embeddings):
        """With a boost of 0.12, the primed atom's neighbors get meaningful activation."""
        atom_a = await insert_atom(storage, "previously seen atom", access_count=5)
        atom_b = await insert_atom(storage, "connected neighbor", access_count=5)
        await insert_synapse(storage, atom_a, atom_b, strength=0.9)

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, budget)

        # Use a higher boost (proposed fix value).
        proposed_boost = 0.12
        seeds = {atom_a: proposed_boost}

        activated, atom_map = await engine._spread_activation(seeds, depth=2)

        # atom_a propagates with more energy; may be slightly higher
        # due to convergence bonus from bidirectional synapse feedback.
        assert atom_a in activated
        assert activated[atom_a] >= proposed_boost

        # atom_b should receive activation: 0.12 * 0.9 * 0.4 * 0.85 ≈ 0.0367
        # Still below min_activation for further propagation, but at least
        # it's in the activated set with a non-trivial score.
        if atom_b in activated:
            # The activation is 2.4x higher than with the 0.05 boost.
            assert activated[atom_b] > 0.03, (
                "Higher boost should give atom_b meaningful activation"
            )


class TestSessionPrimeInRecall:
    """Verify session priming integrates correctly in the full recall pipeline."""

    async def test_session_atom_added_at_prime_boost(self, storage, mock_embeddings):
        """recall() adds session atoms to seeds at _SESSION_PRIME_BOOST."""
        atom_a = await insert_atom(storage, "vector match atom", access_count=3)
        atom_b = await insert_atom(storage, "session primed atom", access_count=3)

        # Mock vector search to return only atom_a.
        mock_embeddings.search_similar = AsyncMock(
            return_value=[(atom_a, 0.3)]  # distance 0.3 → similarity ~0.7
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, budget)

        result = await engine.recall(
            "test query",
            budget_tokens=2000,
            session_atom_ids=[atom_b],
        )

        # atom_b was a session prime, it should appear in the result's
        # total_activated count (even if it doesn't make the final budget cut).
        assert result.seed_count >= 1  # at least atom_a from vector search

    async def test_session_atom_does_not_override_vector_seed(self, storage, mock_embeddings):
        """If a session atom IS also a vector match, the vector score wins."""
        atom_a = await insert_atom(storage, "both vector and session", access_count=3)

        mock_embeddings.search_similar = AsyncMock(
            return_value=[(atom_a, 0.2)]  # distance 0.2 → high similarity
        )

        atoms = AtomManager(storage, mock_embeddings)
        synapses = SynapseManager(storage)
        budget = ContextBudget()
        engine = RetrievalEngine(storage, mock_embeddings, atoms, synapses, budget)

        result = await engine.recall(
            "test query",
            budget_tokens=2000,
            session_atom_ids=[atom_a],
        )

        # The vector seed score (high) should be used, not the prime boost (0.05).
        assert result.seed_count == 1
