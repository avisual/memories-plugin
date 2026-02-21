"""Comprehensive tests for the synapses module.

Covers the Synapse dataclass, validation helpers, and every public method
on SynapseManager including create, get_connections, strengthen/weaken,
Hebbian learning, decay, deletion, and statistics.

All tests use a temporary database directory (via ``tmp_path``) so that no
real user data is affected.  Tests do NOT require Ollama or any network
access -- the synapse layer is pure SQLite.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from memories.synapses import (
    RELATIONSHIP_TYPES,
    Synapse,
    SynapseManager,
    _MAX_INBOUND_RELATED_TO,
    _validate_atom_ids,
    _validate_relationship,
    _validate_strength,
)

from tests.conftest import count_synapses, insert_atom, insert_synapse


# -----------------------------------------------------------------------
# 1. Synapse dataclass
# -----------------------------------------------------------------------


class TestSynapseDataclass:
    """Tests for the Synapse dataclass factory and serialisation methods."""

    def test_from_row_maps_all_fields(self) -> None:
        """from_row should populate every field from a dict-like row."""
        row = {
            "id": 42,
            "source_id": 1,
            "target_id": 2,
            "relationship": "related-to",
            "strength": 0.75,
            "bidirectional": 1,
            "activated_count": 3,
            "last_activated_at": "2025-01-01T00:00:00",
            "created_at": "2024-12-01T00:00:00",
        }
        synapse = Synapse.from_row(row)
        assert synapse.id == 42
        assert synapse.source_id == 1
        assert synapse.target_id == 2
        assert synapse.relationship == "related-to"
        assert synapse.strength == 0.75
        assert synapse.activated_count == 3
        assert synapse.last_activated_at == "2025-01-01T00:00:00"
        assert synapse.created_at == "2024-12-01T00:00:00"

    def test_from_row_converts_bidirectional_int_to_bool(self) -> None:
        """Integer 1/0 from SQLite must become True/False."""
        base = {
            "id": 1,
            "source_id": 1,
            "target_id": 2,
            "relationship": "related-to",
            "strength": 0.5,
            "activated_count": 0,
            "last_activated_at": None,
            "created_at": "2025-01-01T00:00:00",
        }
        assert Synapse.from_row({**base, "bidirectional": 1}).bidirectional is True
        assert Synapse.from_row({**base, "bidirectional": 0}).bidirectional is False

    def test_to_dict_returns_all_keys(self) -> None:
        """to_dict must contain every field with JSON-safe types."""
        synapse = Synapse(
            id=1,
            source_id=10,
            target_id=20,
            relationship="elaborates",
            strength=0.9,
            bidirectional=True,
            activated_count=5,
            last_activated_at="2025-06-01T00:00:00",
            created_at="2025-05-01T00:00:00",
        )
        d = synapse.to_dict()
        expected_keys = {
            "id",
            "source_id",
            "target_id",
            "relationship",
            "strength",
            "bidirectional",
            "activated_count",
            "last_activated_at",
            "tag_expires_at",
            "created_at",
        }
        assert set(d.keys()) == expected_keys
        assert d["id"] == 1
        assert d["strength"] == 0.9
        assert d["bidirectional"] is True

    def test_to_dict_none_last_activated_at(self) -> None:
        """to_dict handles None last_activated_at correctly."""
        synapse = Synapse(
            id=1, source_id=1, target_id=2, relationship="related-to",
            strength=0.5, bidirectional=True, activated_count=0,
            last_activated_at=None, created_at="2025-01-01T00:00:00",
        )
        assert synapse.to_dict()["last_activated_at"] is None


# -----------------------------------------------------------------------
# 2. Validation helpers
# -----------------------------------------------------------------------


class TestSynapseValidation:
    """Tests for the module-private validation functions."""

    @pytest.mark.parametrize("rel", RELATIONSHIP_TYPES)
    def test_validate_relationship_accepts_all_valid(self, rel: str) -> None:
        """Every member of RELATIONSHIP_TYPES should pass validation."""
        _validate_relationship(rel)  # must not raise

    @pytest.mark.parametrize("bad", ["friend-of", "", "RELATED-TO", "related_to"])
    def test_validate_relationship_rejects_invalid(self, bad: str) -> None:
        """Unknown relationship strings must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid relationship"):
            _validate_relationship(bad)

    @pytest.mark.parametrize("val", [0.0, 0.5, 1.0, 0.001, 0.999])
    def test_validate_strength_accepts_valid_range(self, val: float) -> None:
        """Strengths within [0.0, 1.0] must be accepted."""
        _validate_strength(val)  # must not raise

    @pytest.mark.parametrize("val", [-0.01, 1.01, -1.0, 2.0, 100.0])
    def test_validate_strength_rejects_out_of_range(self, val: float) -> None:
        """Strengths outside [0.0, 1.0] must raise ValueError."""
        with pytest.raises(ValueError, match="Strength must be between"):
            _validate_strength(val)

    def test_validate_atom_ids_rejects_self_reference(self) -> None:
        """Source and target being the same atom must raise ValueError."""
        with pytest.raises(ValueError, match="self-referencing"):
            _validate_atom_ids(5, 5)

    def test_validate_atom_ids_accepts_different(self) -> None:
        """Different source and target IDs must be accepted."""
        _validate_atom_ids(1, 2)  # must not raise


# -----------------------------------------------------------------------
# 3. SynapseManager.create
# -----------------------------------------------------------------------


class TestSynapseManagerCreate:
    """Tests for synapse creation and duplicate upsert behaviour."""

    async def test_create_new_synapse(self, storage) -> None:
        """Creating a synapse between two atoms returns a valid Synapse."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")

        mgr = SynapseManager(storage)
        synapse = await mgr.create(a1, a2, "related-to", strength=0.6)

        assert synapse.source_id == a1
        assert synapse.target_id == a2
        assert synapse.relationship == "related-to"
        assert synapse.strength == 0.6
        assert synapse.bidirectional is True

    async def test_create_duplicate_upserts_strength(self, storage) -> None:
        """Re-creating the same (source, target, relationship) triple strengthens."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")

        mgr = SynapseManager(storage)
        s1 = await mgr.create(a1, a2, "related-to", strength=0.5)
        original_strength = s1.strength

        s2 = await mgr.create(a1, a2, "related-to", strength=0.5)

        # Strength should have increased (Hebbian increment).
        assert s2.strength > original_strength
        # Should still be the same synapse (same ID).
        assert s2.id == s1.id

    async def test_create_invalid_relationship_raises(self, storage) -> None:
        """An unrecognised relationship must raise ValueError."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")

        mgr = SynapseManager(storage)
        with pytest.raises(ValueError, match="Invalid relationship"):
            await mgr.create(a1, a2, "friend-of")

    async def test_create_self_reference_raises(self, storage) -> None:
        """Creating a synapse from an atom to itself must raise ValueError."""
        a1 = await insert_atom(storage, "atom one")

        mgr = SynapseManager(storage)
        with pytest.raises(ValueError, match="self-referencing"):
            await mgr.create(a1, a1, "related-to")

    async def test_create_invalid_strength_raises(self, storage) -> None:
        """Strength outside [0, 1] must raise ValueError."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")

        mgr = SynapseManager(storage)
        with pytest.raises(ValueError, match="Strength must be between"):
            await mgr.create(a1, a2, "related-to", strength=1.5)


# -----------------------------------------------------------------------
# 4. SynapseManager.get_connections
# -----------------------------------------------------------------------


class TestSynapseManagerGetConnections:
    """Tests for retrieving synapses connected to an atom."""

    async def test_finds_synapse_as_source(self, storage) -> None:
        """Atom appearing as source_id must be found."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2, strength=0.7)

        mgr = SynapseManager(storage)
        connections = await mgr.get_connections(a1)

        assert len(connections) == 1
        assert connections[0].source_id == a1

    async def test_bidirectional_finds_via_target(self, storage) -> None:
        """Bidirectional synapse must be found when querying by target atom."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2, bidirectional=True, strength=0.5)

        mgr = SynapseManager(storage)
        connections = await mgr.get_connections(a2)

        assert len(connections) == 1
        assert connections[0].target_id == a2

    async def test_unidirectional_not_found_via_target(self, storage) -> None:
        """Unidirectional synapse must NOT be found when querying by target."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2, bidirectional=False, strength=0.5)

        mgr = SynapseManager(storage)
        connections = await mgr.get_connections(a2)

        assert len(connections) == 0

    async def test_min_strength_filtering(self, storage) -> None:
        """Synapses below min_strength must be excluded."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        a3 = await insert_atom(storage, "atom three")
        await insert_synapse(storage, a1, a2, strength=0.3)
        await insert_synapse(storage, a1, a3, strength=0.8)

        mgr = SynapseManager(storage)
        connections = await mgr.get_connections(a1, min_strength=0.5)

        assert len(connections) == 1
        assert connections[0].strength == 0.8

    async def test_ordered_by_strength_descending(self, storage) -> None:
        """Connections must be returned strongest-first."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        a3 = await insert_atom(storage, "atom three")
        await insert_synapse(storage, a1, a2, strength=0.3)
        await insert_synapse(storage, a1, a3, strength=0.9)

        mgr = SynapseManager(storage)
        connections = await mgr.get_connections(a1)

        assert connections[0].strength >= connections[1].strength


# -----------------------------------------------------------------------
# 5. Strengthen / Weaken
# -----------------------------------------------------------------------


class TestSynapseManagerStrengthenWeaken:
    """Tests for the strengthen() and weaken() lifecycle methods."""

    async def test_strengthen_increments_strength(self, storage) -> None:
        """strengthen() must increase the synapse strength using BCM formula."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        sid = await insert_synapse(storage, a1, a2, strength=0.5)

        mgr = SynapseManager(storage)
        result = await mgr.strengthen(a1, a2, "related-to", amount=0.1)

        assert result is not None
        # BCM: 0.5 + 0.1 * (1 - 0.5) = 0.55 (not linear 0.6)
        assert result.strength == pytest.approx(0.55, abs=1e-6)

    async def test_strengthen_caps_at_one(self, storage) -> None:
        """Strength must never exceed 1.0, but BCM slows approach to cap."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2, strength=0.95)

        mgr = SynapseManager(storage)
        result = await mgr.strengthen(a1, a2, "related-to", amount=0.2)

        assert result is not None
        # BCM: 0.95 + 0.2 * (1 - 0.95) = 0.95 + 0.01 = 0.96 (capped at 1.0 if > 1.0)
        bcm_expected = min(1.0, 0.95 + 0.2 * (1.0 - 0.95))
        assert result.strength == pytest.approx(bcm_expected, abs=1e-6)

    async def test_strengthen_increments_activated_count(self, storage) -> None:
        """Each strengthen call must bump activated_count by 1."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2, strength=0.5)

        mgr = SynapseManager(storage)
        result = await mgr.strengthen(a1, a2, "related-to")

        assert result is not None
        assert result.activated_count == 1

        result2 = await mgr.strengthen(a1, a2, "related-to")
        assert result2 is not None
        assert result2.activated_count == 2

    async def test_weaken_reduces_strength(self, storage) -> None:
        """weaken() must decrease the synapse strength."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        sid = await insert_synapse(storage, a1, a2, strength=0.5)

        mgr = SynapseManager(storage)
        result = await mgr.weaken(sid, amount=0.1)

        assert result is not None
        assert result.strength == pytest.approx(0.4, abs=1e-6)

    async def test_weaken_prunes_below_threshold(self, storage) -> None:
        """Weakening below the prune threshold must delete the synapse."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        # prune_threshold defaults to 0.05, so strength 0.06 - 0.05 = 0.01 < 0.05
        sid = await insert_synapse(storage, a1, a2, strength=0.06)

        mgr = SynapseManager(storage)
        result = await mgr.weaken(sid, amount=0.05)

        assert result is None
        # Confirm the synapse is actually gone.
        assert await mgr.get(sid) is None

    async def test_weaken_nonexistent_returns_none(self, storage) -> None:
        """Weakening a non-existent synapse must return None without error."""
        mgr = SynapseManager(storage)
        result = await mgr.weaken(99999, amount=0.1)
        assert result is None


# -----------------------------------------------------------------------
# 6. Hebbian update
# -----------------------------------------------------------------------


class TestSynapseManagerHebbian:
    """Tests for Hebbian co-activation strengthening."""

    async def test_strengthens_existing_synapse(self, storage) -> None:
        """Hebbian update must strengthen an existing synapse between co-activated atoms."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        sid = await insert_synapse(storage, a1, a2, strength=0.5)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2])

        assert count == 1
        updated = await mgr.get(sid)
        assert updated is not None
        # Default hebbian_increment is 0.05; BCM formula: 0.5 + 0.05 * (1 - 0.5) = 0.525
        assert updated.strength == pytest.approx(0.525, abs=1e-6)

    async def test_creates_new_synapse_when_co_activated(self, storage) -> None:
        """Hebbian update must create a new synapse when atoms are co-activated within window."""
        now = datetime.now(tz=timezone.utc).isoformat()
        # access_count >= 1 is required for new synapses (min_accesses_for_hebbian config)
        a1 = await insert_atom(storage, "atom one", last_accessed_at=now, access_count=1)
        a2 = await insert_atom(storage, "atom two", last_accessed_at=now, access_count=1)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2])

        assert count == 1
        # Verify a synapse was created.
        total = await count_synapses(storage)
        assert total == 1

    async def test_single_atom_skips(self, storage) -> None:
        """Hebbian update with fewer than 2 unique IDs must return 0."""
        a1 = await insert_atom(storage, "atom one")

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1])
        assert count == 0

    async def test_empty_list_returns_zero(self, storage) -> None:
        """Hebbian update with empty list must return 0."""
        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([])
        assert count == 0

    async def test_deduplicates_atom_ids(self, storage) -> None:
        """Duplicate IDs in the input list must be deduplicated."""
        now = datetime.now(tz=timezone.utc).isoformat()
        # access_count >= 1 is required for new synapses (min_accesses_for_hebbian config)
        a1 = await insert_atom(storage, "atom one", last_accessed_at=now, access_count=1)
        a2 = await insert_atom(storage, "atom two", last_accessed_at=now, access_count=1)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2, a1, a2, a1])

        # Only one unique pair (a1, a2) should be processed.
        assert count == 1

    async def test_no_synapse_when_session_atoms_lack_access_history(
        self, storage
    ) -> None:
        """No new synapse created when session atoms have no prior access history.

        Session membership implies co-activation — the old timestamp-window
        check has been replaced by the ``min_accesses_for_hebbian`` guard.
        Atoms with zero accesses (access_count=0) have not demonstrated
        repeated relevance and do not get linked, even when they appear in
        the same session.
        """
        now = datetime.now(tz=timezone.utc)
        # access_count=0 — below min_accesses_for_hebbian (default=1)
        a1 = await insert_atom(
            storage, "brand new atom one",
            last_accessed_at=now.isoformat(),
            access_count=0,
        )
        a2 = await insert_atom(
            storage, "brand new atom two",
            last_accessed_at=now.isoformat(),
            access_count=0,
        )

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2])

        assert count == 0

    async def test_no_synapse_when_insufficient_accesses(self, storage) -> None:
        """No new synapse created when atoms have insufficient access history.

        This prevents creating spurious connections between brand-new atoms
        that happen to appear together in one session.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        # Both atoms have 0 accesses (new atoms)
        a1 = await insert_atom(storage, "new atom one", last_accessed_at=now, access_count=0)
        a2 = await insert_atom(storage, "new atom two", last_accessed_at=now, access_count=0)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2])

        assert count == 0
        # Verify no synapse was created
        total = await count_synapses(storage)
        assert total == 0

    async def test_strengthens_existing_synapse_regardless_of_accesses(self, storage) -> None:
        """Existing synapses are strengthened even if atoms have low access counts.

        The min_accesses requirement only applies to NEW synapse creation.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        # Atoms with 0 accesses but already connected
        a1 = await insert_atom(storage, "atom one", last_accessed_at=now, access_count=0)
        a2 = await insert_atom(storage, "atom two", last_accessed_at=now, access_count=0)
        sid = await insert_synapse(storage, a1, a2, strength=0.5)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2])

        # Existing synapse should be strengthened
        assert count == 1
        updated = await mgr.get(sid)
        assert updated is not None
        assert updated.strength > 0.5


# -----------------------------------------------------------------------
# 7. Decay
# -----------------------------------------------------------------------


class TestSynapseManagerDecay:
    """Tests for decay_all() and its pruning side-effects."""

    async def test_reduces_all_strengths_by_factor(self, storage) -> None:
        """Multiplicative decay must reduce every synapse's strength."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        a3 = await insert_atom(storage, "atom three")
        s1 = await insert_synapse(storage, a1, a2, strength=0.8)
        s2 = await insert_synapse(storage, a1, a3, strength=0.6)

        mgr = SynapseManager(storage)
        await mgr.decay_all(factor=0.5)

        syn1 = await mgr.get(s1)
        syn2 = await mgr.get(s2)
        assert syn1 is not None
        assert syn1.strength == pytest.approx(0.4, abs=1e-6)
        assert syn2 is not None
        assert syn2.strength == pytest.approx(0.3, abs=1e-6)

    async def test_prunes_weak_synapses_after_decay(self, storage) -> None:
        """Synapses decayed below prune_threshold (0.05) must be removed."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        # Strength 0.04 * 0.5 = 0.02 < 0.05 threshold
        sid = await insert_synapse(storage, a1, a2, strength=0.04)

        mgr = SynapseManager(storage)
        pruned = await mgr.decay_all(factor=0.5)

        assert pruned >= 1
        assert await mgr.get(sid) is None

    async def test_rejects_factor_zero(self, storage) -> None:
        """Factor of 0 must raise ValueError."""
        mgr = SynapseManager(storage)
        with pytest.raises(ValueError, match="Decay factor must be in"):
            await mgr.decay_all(factor=0.0)

    async def test_rejects_factor_greater_than_one(self, storage) -> None:
        """Factor > 1 must raise ValueError."""
        mgr = SynapseManager(storage)
        with pytest.raises(ValueError, match="Decay factor must be in"):
            await mgr.decay_all(factor=1.5)

    async def test_accepts_factor_of_one(self, storage) -> None:
        """Factor of exactly 1.0 is valid (no-op decay)."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        sid = await insert_synapse(storage, a1, a2, strength=0.5)

        mgr = SynapseManager(storage)
        pruned = await mgr.decay_all(factor=1.0)

        assert pruned == 0
        syn = await mgr.get(sid)
        assert syn is not None
        assert syn.strength == pytest.approx(0.5, abs=1e-6)


# -----------------------------------------------------------------------
# 8. Deletion and statistics
# -----------------------------------------------------------------------


class TestSynapseManagerDeletion:
    """Tests for delete_for_atom() and get_stats()."""

    async def test_delete_for_atom_removes_as_source(self, storage) -> None:
        """Synapses where atom is source must be deleted."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2, strength=0.5)

        mgr = SynapseManager(storage)
        deleted = await mgr.delete_for_atom(a1)

        assert deleted == 1
        total = await count_synapses(storage)
        assert total == 0

    async def test_delete_for_atom_removes_as_target(self, storage) -> None:
        """Synapses where atom is target must also be deleted."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2, strength=0.5)

        mgr = SynapseManager(storage)
        deleted = await mgr.delete_for_atom(a2)

        assert deleted == 1

    async def test_delete_for_atom_removes_all_connected(self, storage) -> None:
        """All synapses connected to the atom (both source and target) must go."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        a3 = await insert_atom(storage, "atom three")
        await insert_synapse(storage, a1, a2, strength=0.5)
        await insert_synapse(storage, a3, a1, strength=0.7)

        mgr = SynapseManager(storage)
        deleted = await mgr.delete_for_atom(a1)

        assert deleted == 2
        total = await count_synapses(storage)
        assert total == 0

    async def test_get_stats_returns_correct_totals(self, storage) -> None:
        """get_stats must report correct total, avg, and relationship breakdown."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        a3 = await insert_atom(storage, "atom three")
        await insert_synapse(storage, a1, a2, relationship="related-to", strength=0.4)
        await insert_synapse(storage, a1, a3, relationship="elaborates", strength=0.8)

        mgr = SynapseManager(storage)
        stats = await mgr.get_stats()

        assert stats["total"] == 2
        assert stats["avg_strength"] == pytest.approx(0.6, abs=1e-4)
        assert stats["by_relationship"]["related-to"] == 1
        assert stats["by_relationship"]["elaborates"] == 1
        assert stats["weakest"]["strength"] == 0.4
        assert stats["strongest"]["strength"] == 0.8

    async def test_get_stats_empty_database(self, storage) -> None:
        """get_stats with no synapses must return zeros and None extremes."""
        mgr = SynapseManager(storage)
        stats = await mgr.get_stats()

        assert stats["total"] == 0
        assert stats["avg_strength"] == 0.0
        assert stats["by_relationship"] == {}
        assert stats["weakest"] is None
        assert stats["strongest"] is None


class TestHebbianCoActivationTiming:
    """Regression tests for the co-activation window timing fix."""

    async def test_atoms_accessed_within_window_are_co_activated(
        self, storage
    ) -> None:
        """Atoms whose last_accessed_at differ by less than window_minutes must pass."""
        now = datetime.now(tz=timezone.utc)
        a1 = await insert_atom(storage, "atom one", last_accessed_at=now.isoformat())
        a2 = await insert_atom(
            storage,
            "atom two",
            last_accessed_at=(now - timedelta(minutes=2)).isoformat(),
        )

        mgr = SynapseManager(storage)
        result = await mgr._check_co_activation(a1, a2, window_minutes=3)

        assert result is True

    async def test_atoms_accessed_outside_window_are_not_co_activated(
        self, storage
    ) -> None:
        """Atoms whose access timestamps differ by more than window_minutes must fail."""
        now = datetime.now(tz=timezone.utc)
        a1 = await insert_atom(storage, "atom one", last_accessed_at=now.isoformat())
        a2 = await insert_atom(
            storage,
            "atom two",
            last_accessed_at=(now - timedelta(minutes=10)).isoformat(),
        )

        mgr = SynapseManager(storage)
        result = await mgr._check_co_activation(a1, a2, window_minutes=3)

        assert result is False

    async def test_long_session_atoms_not_co_activated_if_accessed_far_apart(
        self, storage
    ) -> None:
        """Atoms from a long session accessed 60+ minutes apart must not link.

        This is the core regression: the old implementation compared both
        timestamps against 'now', so atoms from any session longer than the
        window would always fail.  The new implementation compares the two
        timestamps against each other.
        """
        # Simulate a 90-minute session: both atoms were accessed, but 60 min apart.
        session_end = datetime.now(tz=timezone.utc) - timedelta(hours=2)
        a1 = await insert_atom(
            storage, "early atom", last_accessed_at=session_end.isoformat()
        )
        a2 = await insert_atom(
            storage,
            "late atom",
            last_accessed_at=(session_end + timedelta(minutes=60)).isoformat(),
        )

        mgr = SynapseManager(storage)
        result = await mgr._check_co_activation(a1, a2, window_minutes=3)

        assert result is False

    async def test_old_session_atoms_co_activated_if_accessed_close_together(
        self, storage
    ) -> None:
        """Atoms from an old session accessed within the window must still link.

        The old implementation would fail here: both timestamps are far from
        'now' so the 'accessed within last N minutes from now' check rejects
        them even though they were accessed within 1 minute of each other.
        """
        # Simulate a session that ended 2 hours ago.
        session_end = datetime.now(tz=timezone.utc) - timedelta(hours=2)
        a1 = await insert_atom(
            storage, "atom one", last_accessed_at=session_end.isoformat()
        )
        a2 = await insert_atom(
            storage,
            "atom two",
            last_accessed_at=(session_end + timedelta(seconds=30)).isoformat(),
        )

        mgr = SynapseManager(storage)
        result = await mgr._check_co_activation(a1, a2, window_minutes=3)

        assert result is True


class TestHebbianTypedSynapseStrengthening:
    """Hebbian update must strengthen all synapse types, not just related-to."""

    async def test_hebbian_strengthens_elaborates_synapse(self, storage) -> None:
        """If an elaborates synapse exists between two co-activated atoms, it must strengthen."""
        now = datetime.now(tz=timezone.utc)
        a1 = await insert_atom(
            storage, "concept A", access_count=5, last_accessed_at=now.isoformat()
        )
        a2 = await insert_atom(
            storage,
            "elaboration of A",
            access_count=5,
            last_accessed_at=(now - timedelta(seconds=10)).isoformat(),
        )
        syn_id = await insert_synapse(
            storage, a1, a2, relationship="elaborates", strength=0.5
        )

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2])

        assert count == 1
        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (syn_id,)
        )
        assert rows[0]["strength"] > 0.5, "elaborates synapse must be strengthened"

    async def test_hebbian_strengthens_caused_by_synapse(self, storage) -> None:
        """If a caused-by synapse exists, it must be strengthened on co-activation."""
        now = datetime.now(tz=timezone.utc)
        a1 = await insert_atom(
            storage, "cause atom", access_count=5, last_accessed_at=now.isoformat()
        )
        a2 = await insert_atom(
            storage,
            "effect atom",
            access_count=5,
            last_accessed_at=(now - timedelta(seconds=30)).isoformat(),
        )
        syn_id = await insert_synapse(
            storage, a2, a1, relationship="caused-by", strength=0.6
        )

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2])

        assert count == 1
        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE id = ?", (syn_id,)
        )
        assert rows[0]["strength"] > 0.6, "caused-by synapse must be strengthened"


# -----------------------------------------------------------------------
# 9. Batched Hebbian update (F1 + F2)
# -----------------------------------------------------------------------


class TestHebbianBatchUpdate:
    """Tests for the batched Hebbian implementation (F1+F2).

    The new ``hebbian_update`` batches all DB operations instead of N**2
    individual queries: one SELECT for existing synapses, one SELECT for
    access counts, one UPDATE for strengthening, and one INSERT for new
    synapses.
    """

    async def test_batch_strengthens_existing_synapses(self, storage) -> None:
        """All 3 existing synapses between 3 atoms must be strengthened in one call."""
        a1 = await insert_atom(storage, "alpha", access_count=10)
        a2 = await insert_atom(storage, "beta", access_count=10)
        a3 = await insert_atom(storage, "gamma", access_count=10)

        s12 = await insert_synapse(storage, a1, a2, strength=0.4)
        s13 = await insert_synapse(storage, a1, a3, strength=0.5)
        s23 = await insert_synapse(storage, a2, a3, strength=0.6)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2, a3])

        assert count == 3

        # Verify each synapse was strengthened by hebbian_increment (0.05).
        for sid, original in [(s12, 0.4), (s13, 0.5), (s23, 0.6)]:
            syn = await mgr.get(sid)
            assert syn is not None
            assert syn.strength > original, (
                f"Synapse {sid} was not strengthened (expected > {original}, got {syn.strength})"
            )

    async def test_batch_creates_new_synapses(self, storage) -> None:
        """3 atoms with no existing synapses must produce 3 new related-to links."""
        a1 = await insert_atom(storage, "atom A", access_count=10)
        a2 = await insert_atom(storage, "atom B", access_count=10)
        a3 = await insert_atom(storage, "atom C", access_count=10)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2, a3])

        # 3 atoms -> C(3,2) = 3 pairs, all new.
        assert count == 3

        # Verify 3 synapses exist in the DB.
        total = await count_synapses(storage, relationship="related-to")
        assert total == 3

    async def test_skips_pairs_with_insufficient_accesses(self, storage) -> None:
        """Atom with access_count=0 must not form new links; the other pair can."""
        a1 = await insert_atom(storage, "well-known A", access_count=10)
        a2 = await insert_atom(storage, "well-known B", access_count=10)
        a3 = await insert_atom(storage, "brand-new C", access_count=0)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2, a3])

        # Only a1<->a2 should form (both have access_count >= min_accesses).
        # a1<->a3 and a2<->a3 skipped because a3 has 0 accesses.
        assert count == 1

        total = await count_synapses(storage)
        assert total == 1

    async def test_session_membership_implies_coactivation(self, storage) -> None:
        """Atoms with last_accessed_at far in the past must still link if in same call."""
        past = (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat()
        a1 = await insert_atom(storage, "old atom X", access_count=10, last_accessed_at=past)
        a2 = await insert_atom(storage, "old atom Y", access_count=10, last_accessed_at=past)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a2])

        # Session membership is sufficient -- no timestamp check needed.
        assert count == 1

        total = await count_synapses(storage)
        assert total == 1

    async def test_returns_zero_for_single_atom(self, storage) -> None:
        """A single atom ID should return 0 without touching the DB."""
        a1 = await insert_atom(storage, "lone atom")

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1])

        assert count == 0

    async def test_deduplicates_input_ids(self, storage) -> None:
        """Duplicate IDs in input must be collapsed; synapse should not double-strengthen."""
        a1 = await insert_atom(storage, "dup A", access_count=10)
        a2 = await insert_atom(storage, "dup B", access_count=10)
        sid = await insert_synapse(storage, a1, a2, strength=0.5)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a1, a1, a2, a2])

        # Should be treated as [a1, a2] -> 1 pair, 1 existing synapse strengthened.
        assert count == 1

        syn = await mgr.get(sid)
        assert syn is not None
        # Strengthened exactly once with BCM formula: 0.5 + 0.05 * (1 - 0.5) = 0.525, not twice.
        assert syn.strength == pytest.approx(0.525, abs=1e-6)


# -----------------------------------------------------------------------
# 10. Hebbian inbound degree cap (F3)
# -----------------------------------------------------------------------


class TestHebbianInboundDegreeCap:
    """Tests for F3: the ``_MAX_INBOUND_RELATED_TO = 50`` cap in SynapseManager.create.

    Hub nodes that accumulate too many inbound ``related-to`` links saturate
    spreading activation.  Once a target atom reaches the cap, new
    ``related-to`` links are silently dropped, but typed semantic links and
    strengthening of existing links are always allowed.
    """

    async def _create_hub_with_50_inbound(self, storage) -> tuple[int, list[int]]:
        """Helper: create a hub atom with exactly 50 inbound related-to synapses.

        Returns (hub_id, list_of_source_atom_ids).
        """
        hub_id = await insert_atom(storage, "hub atom")
        source_ids: list[int] = []
        for i in range(50):
            src = await insert_atom(storage, f"source atom {i}")
            await insert_synapse(storage, src, hub_id, relationship="related-to", strength=0.5)
            source_ids.append(src)
        return hub_id, source_ids

    async def test_cap_blocks_new_related_to_when_at_limit(self, storage) -> None:
        """51st inbound related-to synapse must be silently dropped (returns None)."""
        hub_id, _ = await self._create_hub_with_50_inbound(storage)
        newcomer = await insert_atom(storage, "atom 51")

        mgr = SynapseManager(storage)
        result = await mgr.create(
            source_id=newcomer, target_id=hub_id, relationship="related-to",
        )

        assert result is None

        # Count must still be exactly 50.
        total = await count_synapses(storage, target_id=hub_id, relationship="related-to")
        assert total == 50

    async def test_cap_does_not_block_strengthening_existing(self, storage) -> None:
        """Strengthening an existing synapse at the 50-cap hub must succeed."""
        hub_id, source_ids = await self._create_hub_with_50_inbound(storage)
        existing_source = source_ids[0]

        mgr = SynapseManager(storage)

        # Fetch original strength before re-create.
        rows = await storage.execute(
            "SELECT strength FROM synapses "
            "WHERE source_id = ? AND target_id = ? AND relationship = 'related-to'",
            (existing_source, hub_id),
        )
        original_strength = rows[0]["strength"]

        result = await mgr.create(
            source_id=existing_source, target_id=hub_id, relationship="related-to",
        )

        # Must NOT return None -- the existing synapse is upserted (strengthened).
        assert result is not None
        assert result.strength > original_strength

    async def test_cap_does_not_apply_to_semantic_types(self, storage) -> None:
        """A caused-by synapse must succeed even when related-to is at 50-cap."""
        hub_id, _ = await self._create_hub_with_50_inbound(storage)
        newcomer = await insert_atom(storage, "causal atom")

        mgr = SynapseManager(storage)
        result = await mgr.create(
            source_id=newcomer, target_id=hub_id, relationship="caused-by",
        )

        # Typed semantic links bypass the related-to cap.
        assert result is not None
        assert result.relationship == "caused-by"

    async def test_cap_allows_51st_semantic_link(self, storage) -> None:
        """With 50 related-to inbound, a caused-by link must still be creatable."""
        hub_id, _ = await self._create_hub_with_50_inbound(storage)
        newcomer = await insert_atom(storage, "another causal atom")

        mgr = SynapseManager(storage)

        # Verify related-to is blocked.
        blocked = await mgr.create(
            source_id=newcomer, target_id=hub_id, relationship="related-to",
        )
        assert blocked is None

        # But caused-by succeeds.
        allowed = await mgr.create(
            source_id=newcomer, target_id=hub_id, relationship="caused-by",
        )
        assert allowed is not None
        assert allowed.relationship == "caused-by"
        assert allowed.source_id == newcomer
        assert allowed.target_id == hub_id


# -----------------------------------------------------------------------
# 11. Hebbian corrected learning (Wave 1-A)
# -----------------------------------------------------------------------


class TestHebbianCorrectedLearning:
    """Tests for inhibitory-type exclusion and BCM multiplicative increment.

    C1+C2: Inhibitory synapse types (contradicts, supersedes, warns-against)
    must not be strengthened on co-activation and must not trigger creation
    of a new related-to synapse.

    H2: The Hebbian increment uses a BCM-style multiplicative formula:
        delta = increment * (1 - current_strength)
    so high-strength synapses saturate more slowly than weak ones.
    """

    async def test_strengthens_all_synapses_for_pair(self, storage) -> None:
        """Both synapses between A and B must be strengthened when A,B co-activate."""
        a = await insert_atom(storage, "atom A", access_count=5)
        b = await insert_atom(storage, "atom B", access_count=5)
        sid1 = await insert_synapse(storage, a, b, relationship="related-to", strength=0.4)
        sid2 = await insert_synapse(storage, a, b, relationship="caused-by", strength=0.6)

        mgr = SynapseManager(storage)
        count = await mgr.hebbian_update([a, b])

        assert count >= 2, "Both synapses must be counted as strengthened"

        rows1 = await storage.execute("SELECT strength FROM synapses WHERE id = ?", (sid1,))
        rows2 = await storage.execute("SELECT strength FROM synapses WHERE id = ?", (sid2,))
        assert rows1[0]["strength"] > 0.4, "related-to synapse must be strengthened"
        assert rows2[0]["strength"] > 0.6, "caused-by synapse must be strengthened"

    async def test_does_not_strengthen_contradicts(self, storage) -> None:
        """A contradicts synapse must NOT be strengthened on co-activation."""
        a = await insert_atom(storage, "atom A", access_count=5)
        b = await insert_atom(storage, "atom B", access_count=5)
        sid = await insert_synapse(storage, a, b, relationship="contradicts", strength=0.5)

        mgr = SynapseManager(storage)
        await mgr.hebbian_update([a, b])

        rows = await storage.execute("SELECT strength FROM synapses WHERE id = ?", (sid,))
        assert rows[0]["strength"] == pytest.approx(0.5, abs=1e-6), (
            "contradicts synapse strength must be unchanged after Hebbian update"
        )

    async def test_does_not_strengthen_supersedes(self, storage) -> None:
        """A supersedes synapse must NOT be strengthened on co-activation."""
        a = await insert_atom(storage, "atom A", access_count=5)
        b = await insert_atom(storage, "atom B", access_count=5)
        sid = await insert_synapse(storage, a, b, relationship="supersedes", strength=0.7)

        mgr = SynapseManager(storage)
        await mgr.hebbian_update([a, b])

        rows = await storage.execute("SELECT strength FROM synapses WHERE id = ?", (sid,))
        assert rows[0]["strength"] == pytest.approx(0.7, abs=1e-6), (
            "supersedes synapse strength must be unchanged after Hebbian update"
        )

    async def test_does_not_strengthen_warns_against(self, storage) -> None:
        """A warns-against synapse must NOT be strengthened on co-activation."""
        a = await insert_atom(storage, "atom A", access_count=5)
        b = await insert_atom(storage, "atom B", access_count=5)
        sid = await insert_synapse(storage, a, b, relationship="warns-against", strength=0.6)

        mgr = SynapseManager(storage)
        await mgr.hebbian_update([a, b])

        rows = await storage.execute("SELECT strength FROM synapses WHERE id = ?", (sid,))
        assert rows[0]["strength"] == pytest.approx(0.6, abs=1e-6), (
            "warns-against synapse strength must be unchanged after Hebbian update"
        )

    async def test_inhibitory_only_pair_no_new_related_to(self, storage) -> None:
        """When only a contradicts synapse exists between A and B, no related-to must be created."""
        a = await insert_atom(storage, "atom A", access_count=10)
        b = await insert_atom(storage, "atom B", access_count=10)
        await insert_synapse(storage, a, b, relationship="contradicts", strength=0.5)

        mgr = SynapseManager(storage)
        await mgr.hebbian_update([a, b])

        total = await count_synapses(storage)
        assert total == 1, (
            "Only the original contradicts synapse should exist; no new related-to created"
        )
        rows = await storage.execute(
            "SELECT relationship FROM synapses WHERE source_id = ? AND target_id = ?", (a, b)
        )
        assert rows[0]["relationship"] == "contradicts"

    async def test_bcm_saturation(self, storage) -> None:
        """High-strength synapse gains less than low-strength synapse (BCM property)."""
        # Pair 1: low strength (a <-> b at 0.2)
        a = await insert_atom(storage, "atom A low", access_count=5)
        b = await insert_atom(storage, "atom B low", access_count=5)
        sid_low = await insert_synapse(storage, a, b, relationship="related-to", strength=0.2)

        # Pair 2: high strength (c <-> d at 0.8)
        c = await insert_atom(storage, "atom C high", access_count=5)
        d = await insert_atom(storage, "atom D high", access_count=5)
        sid_high = await insert_synapse(storage, c, d, relationship="related-to", strength=0.8)

        mgr = SynapseManager(storage)
        # Run separate Hebbian updates per pair so increments are independent.
        await mgr.hebbian_update([a, b])
        await mgr.hebbian_update([c, d])

        rows_low = await storage.execute("SELECT strength FROM synapses WHERE id = ?", (sid_low,))
        rows_high = await storage.execute("SELECT strength FROM synapses WHERE id = ?", (sid_high,))

        gain_low = rows_low[0]["strength"] - 0.2
        gain_high = rows_high[0]["strength"] - 0.8

        assert gain_low > gain_high, (
            f"Low-strength synapse (gain={gain_low:.4f}) must gain more than "
            f"high-strength synapse (gain={gain_high:.4f}) — BCM saturation property"
        )

    async def test_bcm_formula_numerically(self, storage) -> None:
        """BCM formula: new_strength == strength + increment * (1 - strength)."""
        initial_strength = 0.5
        a = await insert_atom(storage, "atom A bcm", access_count=5)
        b = await insert_atom(storage, "atom B bcm", access_count=5)
        sid = await insert_synapse(storage, a, b, relationship="related-to", strength=initial_strength)

        mgr = SynapseManager(storage)
        increment = mgr._cfg.learning.hebbian_increment  # default 0.05

        await mgr.hebbian_update([a, b])

        rows = await storage.execute("SELECT strength FROM synapses WHERE id = ?", (sid,))
        actual = rows[0]["strength"]
        expected = initial_strength + increment * (1.0 - initial_strength)

        assert actual == pytest.approx(expected, abs=1e-6), (
            f"Expected BCM result {expected:.6f}, got {actual:.6f}"
        )


# -----------------------------------------------------------------------
# 12. get_neighbors_batch (Wave 2 – W2-PRE)
# -----------------------------------------------------------------------


class TestGetNeighborsBatch:
    """Tests for the new SynapseManager.get_neighbors_batch() bulk-fetch method.

    get_neighbors_batch must produce the same results as calling get_neighbors()
    individually for each atom, while using a single SQL query for the whole set.
    """

    async def test_batch_returns_same_as_sequential(self, storage) -> None:
        """Batch result for [A, B] must equal get_neighbors(A) + get_neighbors(B)."""
        a = await insert_atom(storage, "atom A")
        b = await insert_atom(storage, "atom B")
        c = await insert_atom(storage, "atom C")

        mgr = SynapseManager(storage)
        # A → B (bidirectional=True so B sees A too)
        await insert_synapse(storage, a, b, strength=0.7, bidirectional=True)
        # B → C (bidirectional=True so C sees B too)
        await insert_synapse(storage, b, c, strength=0.6, bidirectional=True)

        # Sequential individual calls.
        seq_a = await mgr.get_neighbors(a)
        seq_b = await mgr.get_neighbors(b)

        # Build comparable sets: {neighbor_id} for each atom.
        seq_neighbors_a = {nid for nid, _ in seq_a}
        seq_neighbors_b = {nid for nid, _ in seq_b}

        # Batch call.
        batch_result = await mgr.get_neighbors_batch([a, b])

        batch_neighbors_a = {nid for nid, _ in batch_result.get(a, [])}
        batch_neighbors_b = {nid for nid, _ in batch_result.get(b, [])}

        assert batch_neighbors_a == seq_neighbors_a, (
            f"Batch neighbors for A {batch_neighbors_a} "
            f"!= sequential {seq_neighbors_a}"
        )
        assert batch_neighbors_b == seq_neighbors_b, (
            f"Batch neighbors for B {batch_neighbors_b} "
            f"!= sequential {seq_neighbors_b}"
        )

    async def test_batch_empty_returns_empty(self, storage) -> None:
        """get_neighbors_batch([]) must return an empty dict without hitting the DB."""
        mgr = SynapseManager(storage)
        result = await mgr.get_neighbors_batch([])
        assert result == {}

    async def test_batch_respects_min_strength(self, storage) -> None:
        """Synapses below min_strength must be excluded from batch results."""
        a = await insert_atom(storage, "atom A")
        b = await insert_atom(storage, "atom B")
        # Insert a weak synapse (strength=0.1).
        await insert_synapse(storage, a, b, strength=0.1, bidirectional=True)

        mgr = SynapseManager(storage)
        # Ask for neighbors with min_strength=0.5 — synapse should be excluded.
        result = await mgr.get_neighbors_batch([a], min_strength=0.5)

        neighbors_a = result.get(a, [])
        neighbor_ids = {nid for nid, _ in neighbors_a}
        assert b not in neighbor_ids, (
            f"Weak synapse (strength=0.1) should be excluded at min_strength=0.5, "
            f"but {b} appeared in neighbors of {a}"
        )

    async def test_batch_bidirectional(self, storage) -> None:
        """A bidirectional synapse A→B must appear in both A's and B's neighbor lists."""
        a = await insert_atom(storage, "atom A")
        b = await insert_atom(storage, "atom B")
        await insert_synapse(storage, a, b, strength=0.8, bidirectional=True)

        mgr = SynapseManager(storage)
        result = await mgr.get_neighbors_batch([a, b])

        # A should see B as a neighbor.
        neighbors_a = {nid for nid, _ in result.get(a, [])}
        assert b in neighbors_a, f"B should be a neighbor of A, got {neighbors_a}"

        # B should see A as a neighbor (bidirectional).
        neighbors_b = {nid for nid, _ in result.get(b, [])}
        assert a in neighbors_b, f"A should be a neighbor of B (bidirectional), got {neighbors_b}"


# -----------------------------------------------------------------------
# 13. delete_for_atom — RETURNING-based count (W5-A1)
# -----------------------------------------------------------------------


class TestDeleteForAtomReturning:
    """A1: delete_for_atom must use RETURNING to get an accurate deleted count.

    The previous implementation ran DELETE via execute_write() then SELECT
    changes() via execute() — those hit different thread-local SQLite
    connections so changes() always returned 0.  The fix uses
    execute_write_returning with RETURNING id so len(rows) gives the true count.
    """

    async def test_count_accurate_not_always_zero(self, storage) -> None:
        """delete_for_atom must return the actual number of synapses deleted."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        a3 = await insert_atom(storage, "atom three")

        # Insert 3 synapses connected to a1.
        await insert_synapse(storage, a1, a2, strength=0.5)
        await insert_synapse(storage, a3, a1, strength=0.6)
        await insert_synapse(storage, a1, a3, relationship="caused-by", strength=0.7)

        mgr = SynapseManager(storage)
        deleted = await mgr.delete_for_atom(a1)

        assert deleted == 3, (
            f"Expected 3 synapses deleted, got {deleted}. "
            "Likely caused by changes() running on a different connection."
        )

    async def test_delete_for_atom_uses_returning(self, storage, monkeypatch) -> None:
        """delete_for_atom must call execute_write_returning, not execute_write + execute."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2, strength=0.5)

        calls: list[str] = []
        original_returning = storage.execute_write_returning

        async def spy_returning(sql: str, params=()) -> list:
            calls.append("execute_write_returning")
            return await original_returning(sql, params)

        monkeypatch.setattr(storage, "execute_write_returning", spy_returning)

        mgr = SynapseManager(storage)
        await mgr.delete_for_atom(a1)

        assert "execute_write_returning" in calls, (
            "delete_for_atom must use execute_write_returning for RETURNING-based count"
        )


# -----------------------------------------------------------------------
# 14. BCM formula in create() ON CONFLICT clause (W5-A2)
# -----------------------------------------------------------------------


class TestBCMCreateConflict:
    """A2: create() ON CONFLICT clause must use multiplicative BCM increment.

    Previously the ON CONFLICT SET used `synapses.strength + ?` (linear).
    The fix is `synapses.strength + ? * (1.0 - synapses.strength)` (BCM).
    """

    async def test_create_on_conflict_bcm(self, storage) -> None:
        """Second create() call must apply BCM increment, not linear."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")

        mgr = SynapseManager(storage)
        increment = mgr._cfg.learning.hebbian_increment

        # First create — sets initial strength.
        s1 = await mgr.create(a1, a2, "related-to", strength=0.5)
        assert s1 is not None
        initial_strength = s1.strength  # may equal 0.5

        # Second create — triggers ON CONFLICT DO UPDATE.
        s2 = await mgr.create(a1, a2, "related-to", strength=0.5)
        assert s2 is not None

        bcm_expected = initial_strength + increment * (1.0 - initial_strength)
        linear_expected = initial_strength + increment

        # BCM and linear differ when initial_strength != 0.
        assert s2.strength == pytest.approx(bcm_expected, abs=1e-6), (
            f"Expected BCM result {bcm_expected:.6f} "
            f"(not linear {linear_expected:.6f}), got {s2.strength:.6f}"
        )
        assert s2.strength != pytest.approx(linear_expected, abs=1e-6), (
            "ON CONFLICT clause must use BCM formula, not linear addition"
        )


# -----------------------------------------------------------------------
# 15. BCM formula in strengthen() method (W5-A3)
# -----------------------------------------------------------------------


class TestBCMStrengthen:
    """A3: strengthen() must use multiplicative BCM increment, not linear.

    Verifies that `strength + amount` is replaced with
    `strength + amount * (1.0 - strength)`.
    """

    async def test_strengthen_bcm_formula(self, storage) -> None:
        """strengthen(amount=0.1) at initial strength=0.5 must give 0.55, not 0.6."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2, strength=0.5)

        mgr = SynapseManager(storage)
        result = await mgr.strengthen(a1, a2, "related-to", amount=0.1)

        assert result is not None
        bcm_expected = 0.5 + 0.1 * (1.0 - 0.5)  # = 0.55
        linear_expected = 0.5 + 0.1               # = 0.60

        assert result.strength == pytest.approx(bcm_expected, abs=1e-6), (
            f"Expected BCM result {bcm_expected:.4f} (not linear {linear_expected:.4f}), "
            f"got {result.strength:.4f}"
        )
        assert result.strength != pytest.approx(linear_expected, abs=1e-6), (
            "strengthen() must use BCM formula, not linear addition"
        )

    async def test_strengthen_high_strength_bcm(self, storage) -> None:
        """BCM: synapse at 0.9 strengthened by 0.1 gives 0.91, not 1.0 (capped)."""
        a1 = await insert_atom(storage, "atom one")
        a2 = await insert_atom(storage, "atom two")
        await insert_synapse(storage, a1, a2, strength=0.9)

        mgr = SynapseManager(storage)
        result = await mgr.strengthen(a1, a2, "related-to", amount=0.1)

        assert result is not None
        # BCM: 0.9 + 0.1 * (1 - 0.9) = 0.9 + 0.01 = 0.91
        bcm_expected = 0.9 + 0.1 * (1.0 - 0.9)
        assert result.strength == pytest.approx(bcm_expected, abs=1e-6), (
            f"Expected BCM result {bcm_expected:.4f}, got {result.strength:.4f}"
        )


# -----------------------------------------------------------------------
# 16. hebbian_update() inbound degree cap pre-filter (W5-A4)
# -----------------------------------------------------------------------


class TestHebbianInboundCap:
    """A4: hebbian_update must not create new related-to synapses targeting atoms at cap.

    The cap is _MAX_INBOUND_RELATED_TO = 50. Before Step 5 (creating new
    synapses), hebbian_update must count inbound related-to synapses per
    target_id and filter out any new_pair whose target is already at cap.
    """

    async def test_hebbian_update_respects_inbound_cap(self, storage) -> None:
        """hebbian_update must not create a synapse to a target already at cap."""
        from datetime import timezone

        # Create the hub atom (target) and fill it to capacity.
        hub_id = await insert_atom(storage, "hub atom", access_count=10)
        sources_for_hub: list[int] = []
        for i in range(_MAX_INBOUND_RELATED_TO):
            src = await insert_atom(storage, f"hub source {i}", access_count=10)
            await insert_synapse(storage, src, hub_id, relationship="related-to", strength=0.4)
            sources_for_hub.append(src)

        # Create the new atom that will try to link to the hub via Hebbian.
        new_atom = await insert_atom(storage, "new atom trying to link", access_count=10)

        mgr = SynapseManager(storage)
        # hebbian_update([new_atom, hub_id]) — new_atom and hub_id are co-activated.
        # This would normally create a new related-to synapse new_atom -> hub_id,
        # but hub_id is at cap so it must be silently filtered.
        await mgr.hebbian_update([new_atom, hub_id])

        # Verify the cap was not exceeded.
        inbound_count = await count_synapses(
            storage, target_id=hub_id, relationship="related-to"
        )
        assert inbound_count == _MAX_INBOUND_RELATED_TO, (
            f"Expected hub to remain at cap {_MAX_INBOUND_RELATED_TO}, "
            f"got {inbound_count} inbound related-to synapses. "
            "hebbian_update is not filtering over-cap targets."
        )

        # Verify no synapse from new_atom to hub_id was created.
        rows = await storage.execute(
            "SELECT id FROM synapses WHERE source_id = ? AND target_id = ? "
            "AND relationship = 'related-to'",
            (new_atom, hub_id),
        )
        assert len(rows) == 0, (
            "hebbian_update must not create a related-to synapse to an over-cap target"
        )


# -----------------------------------------------------------------------
# 17. Hub cleanup migration (W5-A5)
# -----------------------------------------------------------------------


class TestHubCleanup:
    """A5: cleanup_hub_atoms(max_inbound) must prune weakest inbound related-to
    synapses for atoms that exceed the cap, down to max_inbound.
    """

    async def test_cleanup_hub_atoms(self, storage) -> None:
        """Hub with 55 inbound related-to synapses must be pruned to 49 (cap-1 buffer).

        W7-B1: cleanup trims to max_inbound - 1 (one buffer slot below cap)
        so the next Hebbian cycle can add 1 without immediately exceeding.
        """
        hub_id = await insert_atom(storage, "over-cap hub", access_count=10)

        # Insert 55 inbound related-to synapses with varied strengths.
        # The 6 weakest (strength 0.01..0.06) should be deleted (55 -> 49).
        for i in range(55):
            src = await insert_atom(storage, f"source atom {i}", access_count=10)
            strength = round(0.01 * (i + 1), 3)  # 0.01, 0.02, ..., 0.55
            await insert_synapse(storage, src, hub_id, relationship="related-to", strength=strength)

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        assert deleted == 6, (
            f"Expected 6 weakest synapses deleted (55 -> 49 buffer), got {deleted}"
        )

        # Remaining count must be exactly 49 (cap - 1 buffer).
        remaining = await count_synapses(storage, target_id=hub_id, relationship="related-to")
        assert remaining == 49, (
            f"Expected 49 remaining inbound related-to synapses, got {remaining}"
        )

        # The 6 weakest (strength 0.01..0.06) must be gone.
        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE target_id = ? "
            "AND relationship = 'related-to' ORDER BY strength ASC LIMIT 1",
            (hub_id,),
        )
        assert rows[0]["strength"] == pytest.approx(0.07, abs=1e-6), (
            "The 6 weakest synapses (0.01..0.06) must have been deleted; "
            f"weakest remaining is {rows[0]['strength']:.4f}"
        )

    async def test_cleanup_hub_atoms_no_op_when_under_cap(self, storage) -> None:
        """cleanup_hub_atoms must not delete synapses for atoms under the cap."""
        hub_id = await insert_atom(storage, "normal hub", access_count=5)

        for i in range(30):
            src = await insert_atom(storage, f"source {i}", access_count=5)
            await insert_synapse(storage, src, hub_id, relationship="related-to", strength=0.5)

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        assert deleted == 0, (
            f"Expected 0 deletions (hub at 30 < cap 50), got {deleted}"
        )
        remaining = await count_synapses(storage, target_id=hub_id, relationship="related-to")
        assert remaining == 30

    async def test_cleanup_hub_atoms_returns_count(self, storage) -> None:
        """cleanup_hub_atoms must return the total number of deleted synapses.

        W7-B1: Each hub trims to max_inbound - 1 (49), so 55 -> 49 = 6 each.
        """
        # Two hubs, each 6 over target (55 -> 49).
        hub_a = await insert_atom(storage, "hub A", access_count=10)
        hub_b = await insert_atom(storage, "hub B", access_count=10)

        for i in range(55):
            src_a = await insert_atom(storage, f"hub_a src {i}", access_count=10)
            await insert_synapse(storage, src_a, hub_a, relationship="related-to",
                                 strength=round(0.01 * (i + 1), 3))
            src_b = await insert_atom(storage, f"hub_b src {i}", access_count=10)
            await insert_synapse(storage, src_b, hub_b, relationship="related-to",
                                 strength=round(0.01 * (i + 1), 3))

        mgr = SynapseManager(storage)
        deleted = await mgr.cleanup_hub_atoms(max_inbound=50)

        assert deleted == 12, (
            f"Expected 12 total deletions (2 hubs x 6 over target 49), got {deleted}"
        )
