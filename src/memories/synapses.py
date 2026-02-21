"""Synapse CRUD, Hebbian strengthening, and decay for the memories system.

A **synapse** is a weighted, optionally bidirectional connection between two
:class:`~memories.atoms.Atom` instances.  Synapses encode the *relationship*
between atoms and carry a ``strength`` signal in ``[0, 1]`` that evolves over
time via Hebbian learning (co-activation strengthening) and consolidation
decay.

Relationship taxonomy:

- **related-to** -- general topical association.
- **caused-by** -- causal or consequential link.
- **part-of** -- compositional / containment.
- **contradicts** -- conflicting information.
- **supersedes** -- newer knowledge replacing older.
- **elaborates** -- adds detail to an existing atom.
- **warns-against** -- antipattern / cautionary link.

This module provides:

* :class:`Synapse` -- an immutable-ish dataclass that maps 1:1 with a row in
  the ``synapses`` table, with convenience conversion helpers.
* :class:`SynapseManager` -- async CRUD + Hebbian learning + decay operations
  that coordinate the :class:`~memories.storage.Storage` layer.

Usage::

    from memories.storage import Storage
    from memories.synapses import SynapseManager

    store = Storage()
    await store.initialize()

    mgr = SynapseManager(store)
    synapse = await mgr.create(source_id=1, target_id=2, relationship="related-to")
    print(synapse.to_dict())
"""

from __future__ import annotations

import itertools
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from memories.config import get_config
from memories.storage import Storage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELATIONSHIP_TYPES: tuple[str, ...] = (
    "related-to",
    "caused-by",
    "part-of",
    "contradicts",
    "supersedes",
    "elaborates",
    "warns-against",
)
"""Allowed values for the ``synapses.relationship`` column."""

_NEW_SYNAPSE_DEFAULT_STRENGTH: float = 0.3
"""Default strength when Hebbian update creates a brand-new synapse between
co-activated atoms that were not previously connected."""

_MAX_INBOUND_RELATED_TO: int = 50
"""Maximum number of inbound 'related-to' synapses a target atom can have.

Hub nodes that accumulate hundreds of inbound related-to links saturate
spreading activation at 1.0.  New links are silently dropped once this cap
is reached.  Typed semantic links (caused-by, contradicts, etc.) are always
allowed regardless of degree."""

_MAX_OUTBOUND_RELATED_TO: int = 50
"""Maximum number of outbound 'related-to' synapses a source atom can have.

Mirrors :data:`_MAX_INBOUND_RELATED_TO` for the source side.  Hub nodes
that fan out to hundreds of targets dilute spreading activation by
distributing signal too thinly.  New links are silently dropped once this
cap is reached.  Typed semantic links are always allowed regardless of
degree."""


# ---------------------------------------------------------------------------
# Synapse dataclass
# ---------------------------------------------------------------------------


@dataclass
class Synapse:
    """In-memory representation of a single synapse row.

    Fields map directly to the ``synapses`` table schema.

    Parameters
    ----------
    id:
        Auto-incremented primary key.
    source_id:
        Foreign key to the source :class:`~memories.atoms.Atom`.
    target_id:
        Foreign key to the target :class:`~memories.atoms.Atom`.
    relationship:
        One of :data:`RELATIONSHIP_TYPES`.
    strength:
        Connection weight in ``[0, 1]``.  Higher values indicate a
        stronger association.
    bidirectional:
        Whether this synapse is traversable in both directions.
    activated_count:
        How many times this synapse has been co-activated.
    last_activated_at:
        ISO-8601 timestamp of the most recent activation, or ``None``.
    created_at:
        ISO-8601 timestamp when the synapse was first created.
    """

    id: int
    source_id: int
    target_id: int
    relationship: str
    strength: float
    bidirectional: bool
    activated_count: int
    last_activated_at: str | None
    created_at: str

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_row(cls, row: Any) -> Synapse:
        """Create a :class:`Synapse` from a :class:`sqlite3.Row`.

        Handles the integer-to-bool conversion for ``bidirectional``.

        Parameters
        ----------
        row:
            A :class:`sqlite3.Row` (or any mapping supporting key access)
            from the ``synapses`` table.

        Returns
        -------
        Synapse
            A fully populated dataclass instance.
        """
        return cls(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relationship=row["relationship"],
            strength=row["strength"],
            bidirectional=bool(row["bidirectional"]),
            activated_count=row["activated_count"],
            last_activated_at=row["last_activated_at"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the synapse to a plain dict suitable for MCP tool responses.

        Returns
        -------
        dict[str, Any]
            All fields as a flat dictionary with JSON-safe types.
        """
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "strength": self.strength,
            "bidirectional": self.bidirectional,
            "activated_count": self.activated_count,
            "last_activated_at": self.last_activated_at,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Validation helpers (module-private)
# ---------------------------------------------------------------------------


def _validate_relationship(relationship: str) -> None:
    """Raise :class:`ValueError` if *relationship* is not in :data:`RELATIONSHIP_TYPES`."""
    if relationship not in RELATIONSHIP_TYPES:
        raise ValueError(
            f"Invalid relationship {relationship!r}. "
            f"Must be one of: {', '.join(RELATIONSHIP_TYPES)}"
        )


def _validate_strength(strength: float) -> None:
    """Raise :class:`ValueError` if *strength* is outside ``[0, 1]``."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            f"Strength must be between 0.0 and 1.0, got {strength}"
        )


def _validate_atom_ids(source_id: int, target_id: int) -> None:
    """Raise :class:`ValueError` if source and target are the same atom."""
    if source_id == target_id:
        raise ValueError(
            f"Cannot create a self-referencing synapse (atom_id={source_id})"
        )


# ---------------------------------------------------------------------------
# Synapse manager
# ---------------------------------------------------------------------------


class SynapseManager:
    """Async CRUD manager for synapses with Hebbian learning and decay.

    Coordinates writes to the ``synapses`` table and implements the
    biologically inspired strengthening, weakening, and pruning lifecycle.

    Parameters
    ----------
    storage:
        An initialised :class:`~memories.storage.Storage` instance.
    """

    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._cfg = get_config()

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create(
        self,
        source_id: int,
        target_id: int,
        relationship: str,
        strength: float = 0.5,
        bidirectional: bool = True,
    ) -> Synapse | None:
        """Create a synapse between two atoms, or strengthen an existing one.

        If a synapse with the same ``(source_id, target_id, relationship)``
        triple already exists, the existing synapse is strengthened by the
        configured Hebbian increment rather than creating a duplicate.

        For ``related-to`` synapses only: if the target atom already has
        :data:`_MAX_INBOUND_RELATED_TO` inbound ``related-to`` connections and
        no synapse yet exists between this pair, the new link is silently
        dropped to prevent hub-node formation that saturates spreading
        activation.  Typed semantic links (``caused-by``, ``contradicts``,
        etc.) are always allowed regardless of inbound degree.

        Parameters
        ----------
        source_id:
            Foreign key to the source atom.
        target_id:
            Foreign key to the target atom.
        relationship:
            One of :data:`RELATIONSHIP_TYPES`.
        strength:
            Initial connection weight in ``[0, 1]`` (default ``0.5``).
        bidirectional:
            Whether this synapse can be traversed in both directions.

        Returns
        -------
        Synapse | None
            The newly created or strengthened synapse, or ``None`` if the
            inbound ``related-to`` degree cap would be exceeded.

        Raises
        ------
        ValueError
            If validation of *relationship*, *strength*, or atom IDs fails.
        """
        _validate_relationship(relationship)
        _validate_strength(strength)
        _validate_atom_ids(source_id, target_id)

        increment = self._cfg.learning.hebbian_increment

        def _do_upsert(conn: sqlite3.Connection) -> dict[str, Any] | None:
            # W7-A2: Cap check INSIDE the transaction to prevent cross-session
            # TOCTOU.  Previously the cap check was outside the transaction, so
            # two concurrent calls could both read count=49, both proceed, and
            # the atom would end up at 51.  With BEGIN IMMEDIATE, the count
            # query and insert happen atomically.
            if relationship == "related-to":
                existing = conn.execute(
                    "SELECT id FROM synapses "
                    "WHERE source_id = ? AND target_id = ? AND relationship = ?",
                    (source_id, target_id, relationship),
                ).fetchone()
                if existing is None:
                    count = conn.execute(
                        "SELECT COUNT(*) AS cnt FROM synapses "
                        "WHERE target_id = ? AND relationship = 'related-to'",
                        (target_id,),
                    ).fetchone()["cnt"]
                    if count >= _MAX_INBOUND_RELATED_TO:
                        log.debug(
                            "Skipping new related-to synapse: target atom %d at max "
                            "inbound degree (%d)",
                            target_id,
                            _MAX_INBOUND_RELATED_TO,
                        )
                        return None

            conn.execute(
                """
                INSERT INTO synapses
                    (source_id, target_id, relationship, strength, bidirectional)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, relationship) DO UPDATE SET
                    strength = MIN(1.0, synapses.strength + ? * (1.0 - synapses.strength)),
                    activated_count = synapses.activated_count + 1,
                    last_activated_at = datetime('now')
                """,
                (source_id, target_id, relationship, strength, int(bidirectional), increment),
            )
            row = conn.execute(
                "SELECT * FROM synapses WHERE source_id = ? AND target_id = ? AND relationship = ?",
                (source_id, target_id, relationship),
            ).fetchone()
            return dict(row)

        row_dict = await self._storage.execute_transaction(_do_upsert)
        if row_dict is None:
            return None
        synapse = Synapse.from_row(row_dict)

        log.info(
            "Upserted synapse %d: atom %d -[%s]-> atom %d (strength=%.2f)",
            synapse.id,
            source_id,
            relationship,
            target_id,
            synapse.strength,
        )

        return synapse

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, synapse_id: int) -> Synapse | None:
        """Retrieve a synapse by primary key.

        Parameters
        ----------
        synapse_id:
            Primary key of the synapse.

        Returns
        -------
        Synapse | None
            The synapse, or ``None`` if it does not exist.
        """
        rows = await self._storage.execute(
            "SELECT * FROM synapses WHERE id = ?",
            (synapse_id,),
        )
        if not rows:
            return None
        return Synapse.from_row(rows[0])

    async def get_connections(
        self,
        atom_id: int,
        min_strength: float = 0.0,
    ) -> list[Synapse]:
        """Get all synapses connected to an atom.

        For bidirectional synapses, both directions are considered: the
        atom may appear as ``source_id`` or ``target_id``.  For
        unidirectional synapses, only outgoing connections (where the
        atom is the source) are returned.

        Parameters
        ----------
        atom_id:
            The atom whose connections to retrieve.
        min_strength:
            Minimum strength threshold (inclusive).  Synapses weaker
            than this are excluded.

        Returns
        -------
        list[Synapse]
            Connected synapses ordered by strength descending.
        """
        rows = await self._storage.execute(
            """
            SELECT * FROM synapses
            WHERE strength >= ?
              AND (
                  source_id = ?
                  OR (target_id = ? AND bidirectional = 1)
              )
            ORDER BY strength DESC
            """,
            (min_strength, atom_id, atom_id),
        )
        return [Synapse.from_row(r) for r in rows]

    async def get_neighbors(
        self,
        atom_id: int,
        min_strength: float = 0.0,
    ) -> list[tuple[int, Synapse]]:
        """Get neighbor atom IDs with their connecting synapses.

        For each connection, determines which end is the "other" atom
        (the neighbor) and returns it paired with the synapse.

        Parameters
        ----------
        atom_id:
            The atom whose neighbors to discover.
        min_strength:
            Minimum strength threshold (inclusive).

        Returns
        -------
        list[tuple[int, Synapse]]
            Pairs of ``(neighbor_atom_id, synapse)`` ordered by
            synapse strength descending.
        """
        connections = await self.get_connections(atom_id, min_strength)
        neighbors: list[tuple[int, Synapse]] = []

        for synapse in connections:
            if synapse.source_id == atom_id:
                neighbor_id = synapse.target_id
            else:
                neighbor_id = synapse.source_id
            neighbors.append((neighbor_id, synapse))

        return neighbors

    async def get_neighbors_batch(
        self,
        atom_ids: list[int],
        min_strength: float = 0.0,
    ) -> dict[int, list[tuple[int, "Synapse"]]]:
        """Batch-fetch neighbors for multiple atoms in a single SQL query.

        Replicates the per-atom logic of :meth:`get_neighbors` for an entire
        set of atoms at once, avoiding one SQL round-trip per frontier atom
        during spreading activation.

        For each queried atom:

        - **Outgoing** connections (atom is ``source_id``) are always included.
        - **Incoming** connections (atom is ``target_id``) are included only
          when ``bidirectional = 1``, matching :meth:`get_connections`.

        Parameters
        ----------
        atom_ids:
            Atom IDs whose neighbors to fetch.  An empty list returns ``{}``.
        min_strength:
            Minimum synapse strength (inclusive).  Synapses below this are
            excluded.

        Returns
        -------
        dict[int, list[tuple[int, Synapse]]]
            Mapping from each queried atom ID to a list of
            ``(neighbor_id, synapse)`` pairs, ordered by synapse strength
            descending.  Every requested atom ID is present as a key even
            if it has no neighbors.
        """
        if not atom_ids:
            return {}

        placeholders = ",".join("?" for _ in atom_ids)

        # Match get_connections logic:
        # - Include row when queried atom is source_id (outgoing, any directionality).
        # - Include row when queried atom is target_id AND bidirectional = 1.
        rows = await self._storage.execute(
            f"SELECT * FROM synapses "
            f"WHERE strength >= ? "
            f"  AND ("
            f"    source_id IN ({placeholders})"
            f"    OR (target_id IN ({placeholders}) AND bidirectional = 1)"
            f"  ) "
            f"ORDER BY strength DESC",
            (min_strength, *atom_ids, *atom_ids),
        )

        result: dict[int, list[tuple[int, Synapse]]] = {aid: [] for aid in atom_ids}
        atom_id_set = set(atom_ids)

        for row in rows:
            syn = Synapse.from_row(row)
            # Append to source-side entry if source_id is in our queried set.
            if syn.source_id in atom_id_set:
                result[syn.source_id].append((syn.target_id, syn))
            # Append to target-side entry if target_id is in our queried set
            # and the synapse is bidirectional (already guaranteed by WHERE clause,
            # but guard again for clarity when source and target are both queried).
            if syn.bidirectional and syn.target_id in atom_id_set:
                result[syn.target_id].append((syn.source_id, syn))

        return result

    # ------------------------------------------------------------------
    # Strengthen / Weaken
    # ------------------------------------------------------------------

    async def strengthen(
        self,
        source_id: int,
        target_id: int,
        relationship: str,
        amount: float | None = None,
    ) -> Synapse | None:
        """Increase a synapse's strength (Hebbian learning).

        The strength is capped at ``1.0``.  The ``activated_count`` is
        incremented and ``last_activated_at`` is updated to the current
        timestamp.

        Parameters
        ----------
        source_id:
            Source atom foreign key.
        target_id:
            Target atom foreign key.
        relationship:
            The relationship type identifying the synapse.
        amount:
            Strength increment.  Defaults to
            ``config.learning.hebbian_increment``.

        Returns
        -------
        Synapse | None
            The updated synapse, or ``None`` if no matching synapse
            was found.
        """
        if amount is None:
            amount = self._cfg.learning.hebbian_increment

        # A3: BCM multiplicative increment — delta = amount * (1 - strength).
        # High-strength synapses gain less per activation (biological saturation).
        await self._storage.execute_write(
            """
            UPDATE synapses
            SET strength = MIN(1.0, strength + ? * (1.0 - strength)),
                activated_count = activated_count + 1,
                last_activated_at = datetime('now')
            WHERE source_id = ? AND target_id = ? AND relationship = ?
            """,
            (amount, source_id, target_id, relationship),
        )

        return await self._fetch_by_triple(source_id, target_id, relationship)

    async def weaken(
        self,
        synapse_id: int,
        amount: float = 0.05,
    ) -> Synapse | None:
        """Decrease a synapse's strength.

        If the resulting strength falls below the configured prune
        threshold, the synapse is deleted entirely.

        Parameters
        ----------
        synapse_id:
            Primary key of the synapse to weaken.
        amount:
            Strength decrement (positive value).

        Returns
        -------
        Synapse | None
            The weakened synapse, or ``None`` if it was pruned or did
            not exist.
        """
        prune_threshold = self._cfg.consolidation.prune_threshold

        def _do_weaken(conn: sqlite3.Connection) -> dict[str, Any] | None:
            row = conn.execute(
                "SELECT * FROM synapses WHERE id = ?",
                (synapse_id,),
            ).fetchone()
            if row is None:
                return None

            new_strength = row["strength"] - amount

            if new_strength < prune_threshold:
                conn.execute(
                    "DELETE FROM synapses WHERE id = ?",
                    (synapse_id,),
                )
                return {"pruned": True, "strength": new_strength}

            conn.execute(
                "UPDATE synapses SET strength = ? WHERE id = ?",
                (new_strength, synapse_id),
            )
            return {"pruned": False}

        result = await self._storage.execute_transaction(_do_weaken)
        if result is None:
            return None

        if result["pruned"]:
            log.info(
                "Pruned synapse %d (strength %.4f below threshold %.4f)",
                synapse_id,
                result["strength"],
                prune_threshold,
            )
            return None

        return await self.get(synapse_id)

    # ------------------------------------------------------------------
    # Hebbian update
    # ------------------------------------------------------------------

    async def hebbian_update(
        self,
        atom_ids: list[int],
        atom_timestamps: dict[int, float] | None = None,
    ) -> int:
        """Apply Hebbian strengthening for atoms co-activated in a session.

        All atoms in *atom_ids* are considered co-activated by virtue of
        appearing in the same session — the timestamp-window check used
        previously was unreliable because ``last_accessed_at`` is updated by
        the Stop hook at session end, not at the moment of individual access.

        For every unique pair of atom IDs in the provided list:

        1. If a synapse already exists between the pair (in either direction),
           strengthen it by ``config.learning.hebbian_increment``.
        2. If no synapse exists and both atoms have at least
           ``min_accesses_for_hebbian`` prior accesses, create a new
           ``"related-to"`` synapse with an initial strength of
           :data:`_NEW_SYNAPSE_DEFAULT_STRENGTH`.

        All DB interaction is batched: one query pre-fetches existing synapses,
        one query fetches atom access counts, one UPDATE strengthens all
        existing synapses, and one ``execute_many`` creates all new ones.
        This reduces O(n²) individual queries to O(1) bulk operations.

        Parameters
        ----------
        atom_ids:
            Flat list of atom IDs that were accessed during a session.
            Duplicates are ignored.
        atom_timestamps:
            Optional mapping of atom ID to Unix timestamp (float) indicating
            when each atom was accessed.  When provided, new synapse pairs
            whose atoms were accessed within ``learning.temporal_window_seconds``
            of each other receive the full Hebbian increment; pairs accessed
            further apart receive ``increment * 0.5``.  Existing synapses are
            always strengthened at the full rate.  Backward-compatible: when
            ``None``, all pairs use the full increment.

        Returns
        -------
        int
            Total number of synapses updated or created.
        """
        unique_ids = list(set(atom_ids))
        if len(unique_ids) < 2:
            return 0

        increment = self._cfg.learning.hebbian_increment
        min_accesses = self._cfg.learning.min_accesses_for_hebbian
        temporal_window = self._cfg.learning.temporal_window_seconds
        placeholders = ",".join("?" * len(unique_ids))

        # Inhibitory relationship types must never be strengthened on co-activation.
        # Co-activating two atoms that contradict/supersede/warn-against each other
        # should not reinforce those negative associations.
        _INHIBITORY = frozenset({"contradicts", "supersedes", "warns-against"})

        # Step 1: Pre-fetch all existing synapses where both endpoints are in
        # the session atom set (any relationship type, either direction).
        # C1: also fetch relationship so we can exclude inhibitory types.
        existing_rows = await self._storage.execute(
            f"SELECT id, source_id, target_id, relationship FROM synapses "
            f"WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})",
            tuple(unique_ids) * 2,
        )

        # Build canonical-pair → list[synapse-id] map for non-inhibitory synapses.
        # C2: inhibitory synapses are excluded from strengthening entirely.
        existing_by_pair: dict[tuple[int, int], list[int]] = {}
        for row in existing_rows:
            if row["relationship"] in _INHIBITORY:
                continue  # skip — do NOT strengthen on co-activation
            pair = (min(row["source_id"], row["target_id"]), max(row["source_id"], row["target_id"]))
            existing_by_pair.setdefault(pair, []).append(row["id"])

        # Build a set of ALL pairs that have ANY existing synapse (including inhibitory).
        # Used to prevent creating a new related-to when only inhibitory links exist.
        all_existing_pairs: set[tuple[int, int]] = set()
        for row in existing_rows:
            all_existing_pairs.add(
                (min(row["source_id"], row["target_id"]), max(row["source_id"], row["target_id"]))
            )

        # Flatten non-inhibitory synapse IDs to strengthen.
        strengthen_ids: list[int] = [sid for ids in existing_by_pair.values() for sid in ids]

        # Step 2: Pre-fetch access counts for all session atoms.
        access_rows = await self._storage.execute(
            f"SELECT id, access_count FROM atoms "
            f"WHERE id IN ({placeholders}) AND is_deleted = 0",
            tuple(unique_ids),
        )
        atom_access: dict[int, int] = {row["id"]: row["access_count"] for row in access_rows}

        # Step 3: Classify every pair as strengthen-existing or create-new.
        new_pairs: list[tuple[int, int, float, float]] = []  # (src, tgt, strength, increment)

        for id_a, id_b in itertools.combinations(sorted(unique_ids), 2):
            pair = (id_a, id_b)  # already sorted: id_a < id_b
            if pair in existing_by_pair:
                # Already collected in strengthen_ids above — nothing to do here.
                pass
            elif pair in all_existing_pairs:
                # Inhibitory-only pair: do NOT create a new related-to synapse.
                log.debug(
                    "Hebbian skipped new synapse %d <-> %d: "
                    "only inhibitory synapse(s) exist between this pair",
                    id_a, id_b,
                )
            else:
                # Truly new pair — apply min_accesses guard then queue for creation.
                if min_accesses > 0:
                    acc_a = atom_access.get(id_a, 0)
                    acc_b = atom_access.get(id_b, 0)
                    if acc_a < min_accesses or acc_b < min_accesses:
                        log.debug(
                            "Hebbian skipped new synapse %d <-> %d: "
                            "access counts %d/%d < min %d",
                            id_a, id_b, acc_a, acc_b, min_accesses,
                        )
                        continue
                # Temporal weighting: pairs accessed within temporal_window_seconds
                # of each other get the full increment; distant pairs get 0.5×.
                if atom_timestamps:
                    ts_a = atom_timestamps.get(id_a, 0.0)
                    ts_b = atom_timestamps.get(id_b, 0.0)
                    pair_increment = (
                        increment
                        if abs(ts_a - ts_b) <= temporal_window
                        else increment * 0.5
                    )
                else:
                    pair_increment = increment
                new_pairs.append((id_a, id_b, _NEW_SYNAPSE_DEFAULT_STRENGTH, pair_increment))

        # Step 3b: Cue overload protection — cap new synapse creation.
        # Large sessions generate O(n^2) candidate pairs; capping prevents the
        # fan effect where hundreds of weak associations dilute the learning
        # signal.  Pairs are prioritised by temporal proximity when timestamps
        # are available (closer = stronger signal).
        max_new = self._cfg.learning.max_new_pairs_per_session
        if len(new_pairs) > max_new:
            total_candidates = len(new_pairs)
            if atom_timestamps:
                new_pairs.sort(
                    key=lambda p: abs(
                        atom_timestamps.get(p[0], 0.0)
                        - atom_timestamps.get(p[1], 0.0)
                    )
                )
            new_pairs = new_pairs[:max_new]
            log.debug(
                "Cue overload cap: limited new pairs to %d (from %d candidates)",
                max_new,
                total_candidates,
            )

        # Step 4: Batch-strengthen all non-inhibitory existing synapses in one UPDATE.
        # H2: BCM multiplicative increment — delta = increment * (1 - strength) — so
        # already-strong synapses saturate more slowly than weak ones.
        if strengthen_ids:
            id_ph = ",".join("?" * len(strengthen_ids))
            await self._storage.execute_write(
                f"UPDATE synapses "
                f"SET strength = MIN(1.0, strength + ? * (1.0 - strength)), "
                f"    activated_count = activated_count + 1, "
                f"    last_activated_at = datetime('now') "
                f"WHERE id IN ({id_ph})",
                (increment, *strengthen_ids),
            )

        # Step 4b: A4 — Inbound degree cap pre-filter.
        # Before creating new related-to synapses, count existing inbound
        # related-to synapses for each candidate target_id.  Targets already
        # at _MAX_INBOUND_RELATED_TO are excluded from new_pairs to prevent
        # hub formation that saturates spreading activation.
        if new_pairs:
            target_ids = list({tgt for _, tgt, _, _ in new_pairs})
            tgt_placeholders = ",".join("?" * len(target_ids))
            cap_rows = await self._storage.execute(
                f"SELECT target_id, COUNT(*) AS cnt FROM synapses "
                f"WHERE target_id IN ({tgt_placeholders}) "
                f"  AND relationship = 'related-to' "
                f"GROUP BY target_id",
                tuple(target_ids),
            )
            current_inbound: dict[int, int] = {
                row["target_id"]: row["cnt"]
                for row in cap_rows
            }

            # W7-A1: Python-side budget counter to prevent within-batch TOCTOU.
            # The pre-filter query reads counts ONCE, but multiple pairs in this
            # batch can target the same atom.  Without tracking how many we've
            # already queued, an atom at 48 could receive 3+ new synapses in one
            # batch, ending up at 51+.  The budget counter accumulates per-target
            # so only (cap - current) new synapses are allowed per target.
            inbound_budget: dict[int, int] = {}
            filtered_pairs: list[tuple[int, int, float, float]] = []
            for pair in new_pairs:
                _, tgt, _, _ = pair
                already = current_inbound.get(tgt, 0)
                added_this_batch = inbound_budget.get(tgt, 0)
                if already + added_this_batch < _MAX_INBOUND_RELATED_TO:
                    filtered_pairs.append(pair)
                    inbound_budget[tgt] = added_this_batch + 1
            before = len(new_pairs)
            new_pairs = filtered_pairs
            filtered = before - len(new_pairs)
            if filtered:
                log.debug(
                    "Hebbian inbound cap: filtered %d new synapse(s) via budget counter",
                    filtered,
                )

        # Step 4c: Outbound degree cap pre-filter (mirrors Step 4b for source_id).
        # Before creating new related-to synapses, count existing outbound
        # related-to synapses for each candidate source_id.  Sources already
        # at _MAX_OUTBOUND_RELATED_TO are excluded from new_pairs to prevent
        # hub formation that dilutes spreading activation signal.
        if new_pairs:
            source_ids = list({src for src, _, _, _ in new_pairs})
            src_ph = ",".join("?" * len(source_ids))
            outbound_rows = await self._storage.execute(
                f"SELECT source_id, COUNT(*) AS cnt FROM synapses "
                f"WHERE source_id IN ({src_ph}) AND relationship = 'related-to' "
                f"GROUP BY source_id",
                tuple(source_ids),
            )
            current_outbound: dict[int, int] = {
                row["source_id"]: row["cnt"] for row in outbound_rows
            }

            # W9-A: Python-side budget counter to prevent within-batch TOCTOU.
            # Same pattern as the inbound budget counter in Step 4b.
            outbound_budget: dict[int, int] = {}
            filtered_out: list[tuple[int, int, float, float]] = []
            for pair in new_pairs:
                src = pair[0]  # source_id is first element (id_a, the smaller ID)
                already = current_outbound.get(src, 0)
                added = outbound_budget.get(src, 0)
                if already + added < _MAX_OUTBOUND_RELATED_TO:
                    filtered_out.append(pair)
                    outbound_budget[src] = added + 1
            before_out = len(new_pairs)
            new_pairs = filtered_out
            filtered_out_count = before_out - len(new_pairs)
            if filtered_out_count:
                log.debug(
                    "Hebbian outbound cap: filtered %d new synapse(s) via budget counter",
                    filtered_out_count,
                )

        # Step 5: Batch-create all new related-to synapses.
        # H2: ON CONFLICT clause also uses BCM formula for idempotent re-runs.
        if new_pairs:
            await self._storage.execute_many(
                "INSERT INTO synapses "
                "    (source_id, target_id, relationship, strength, "
                "     bidirectional, activated_count, last_activated_at) "
                "VALUES (?, ?, 'related-to', ?, 1, 1, datetime('now')) "
                "ON CONFLICT(source_id, target_id, relationship) DO UPDATE SET "
                "    strength = MIN(1.0, synapses.strength + ? * (1.0 - synapses.strength)), "
                "    activated_count = synapses.activated_count + 1, "
                "    last_activated_at = datetime('now')",
                new_pairs,
            )

        strengthen_count = len(strengthen_ids)
        create_count = len(new_pairs)
        log.info(
            "Hebbian update: %d strengthened, %d created from %d session atoms",
            strengthen_count,
            create_count,
            len(unique_ids),
        )
        return strengthen_count + create_count

    # ------------------------------------------------------------------
    # Decay and pruning
    # ------------------------------------------------------------------

    async def decay_all(self, factor: float = 0.95) -> int:
        """Decay all synapse strengths by a multiplicative factor.

        Each synapse's strength is multiplied by *factor* (e.g. 0.95
        reduces every strength by 5%).  Synapses whose post-decay
        strength falls below the configured prune threshold are deleted.

        Parameters
        ----------
        factor:
            Multiplicative decay factor in ``(0, 1]``.

        Returns
        -------
        int
            Number of synapses pruned (deleted) after decay.
        """
        if not 0.0 < factor <= 1.0:
            raise ValueError(
                f"Decay factor must be in (0, 1], got {factor}"
            )

        # Apply multiplicative decay to all synapses.
        await self._storage.execute_write(
            "UPDATE synapses SET strength = strength * ?",
            (factor,),
        )

        # Prune those that fell below the threshold.
        pruned = await self.prune_weak()

        log.info(
            "Decay applied (factor=%.4f): %d synapses pruned",
            factor,
            pruned,
        )
        return pruned

    async def prune_weak(self, threshold: float | None = None) -> int:
        """Delete synapses whose strength is below the threshold.

        Parameters
        ----------
        threshold:
            Minimum strength to survive.  Defaults to
            ``config.consolidation.prune_threshold``.

        Returns
        -------
        int
            Number of synapses deleted.
        """
        if threshold is None:
            threshold = self._cfg.consolidation.prune_threshold

        # C3: Use a single DELETE and read rowcount instead of COUNT + DELETE.
        def _do_prune(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "DELETE FROM synapses WHERE strength < ?",
                (threshold,),
            )
            return cursor.rowcount

        count = await self._storage.execute_transaction(_do_prune)

        if count > 0:
            log.info(
                "Pruned %d weak synapses (threshold=%.4f)",
                count,
                threshold,
            )

        return count

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    async def delete_for_atom(self, atom_id: int) -> int:
        """Delete all synapses connected to an atom.

        Removes synapses where the atom appears as either ``source_id``
        or ``target_id``, regardless of directionality.

        Parameters
        ----------
        atom_id:
            The atom whose synapses should be removed.

        Returns
        -------
        int
            Number of synapses deleted.
        """
        # A1: Use RETURNING to get an accurate count of deleted rows.
        # execute_write() and execute() hit different thread-local SQLite
        # connections, so changes() returns 0.  RETURNING id gives the true count.
        rows = await self._storage.execute_write_returning(
            "DELETE FROM synapses WHERE source_id = ? OR target_id = ? RETURNING id",
            (atom_id, atom_id),
        )
        count = len(rows)

        if count > 0:
            log.info(
                "Deleted %d synapses connected to atom %d",
                count,
                atom_id,
            )

        return count

    async def cleanup_hub_atoms(self, max_inbound: int = 50) -> int:
        """Delete the weakest inbound ``related-to`` synapses for over-cap hub atoms.

        Hub atoms that have accumulated *max_inbound* or more inbound
        ``related-to`` connections saturate spreading activation at 1.0.
        This idempotent cleanup finds all such atoms and deletes their
        weakest inbound ``related-to`` synapses until each is at
        ``max_inbound - 1`` (one buffer slot below cap, preventing
        oscillation at the boundary every Hebbian cycle).

        The entire operation is atomic: the CTE identifying excess synapses
        and the DELETE removing them run in a single ``execute_write_returning``
        call, preventing phantom-read race conditions.

        Typed semantic links (``caused-by``, ``contradicts``, etc.) are never
        deleted by this method.

        Parameters
        ----------
        max_inbound:
            Hard cap on inbound ``related-to`` synapses per atom.
            Atoms at or above this count are trimmed to ``max_inbound - 1``.
            Defaults to :data:`_MAX_INBOUND_RELATED_TO` (50).

        Returns
        -------
        int
            Total number of synapses deleted across all over-cap atoms.
        """
        # W7-B1+B2: Atomic CTE+DELETE in a single execute_write_returning() call.
        #
        # B1 fix: HAVING cnt >= ? (was cnt > ?) so atoms at exactly max_inbound
        # are cleaned.  Excess formula uses (? - 1) so atoms land at
        # max_inbound - 1, giving a 1-slot buffer below cap.
        #
        # B2 fix: The CTE is inlined into the DELETE statement so the read and
        # write happen atomically on the same connection.  Previously the CTE
        # ran via execute() (read connection) and the DELETE via
        # execute_write_returning() (write connection) -- non-atomic.
        deleted_rows = await self._storage.execute_write_returning(
            """
            DELETE FROM synapses WHERE id IN (
                SELECT id FROM (
                    WITH hub_targets AS (
                        SELECT target_id, COUNT(*) AS cnt
                        FROM synapses
                        WHERE relationship = 'related-to'
                        GROUP BY target_id
                        HAVING cnt >= ?
                    ),
                    ranked AS (
                        SELECT s.id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY s.target_id ORDER BY s.strength ASC
                               ) AS rn,
                               h.cnt - (? - 1) AS excess
                        FROM synapses s
                        JOIN hub_targets h ON h.target_id = s.target_id
                        WHERE s.relationship = 'related-to'
                    )
                    SELECT id FROM ranked WHERE rn <= excess
                )
            ) RETURNING id
            """,
            (max_inbound, max_inbound),
        )

        total_deleted = len(deleted_rows)

        if total_deleted:
            log.info(
                "Hub cleanup: deleted %d over-cap inbound related-to synapses "
                "(max_inbound=%d)",
                total_deleted,
                max_inbound,
            )

        return total_deleted

    async def cleanup_hub_atoms_outbound(self, max_outbound: int = 50) -> int:
        """Delete the weakest outbound ``related-to`` synapses for over-cap hub atoms.

        Mirrors :meth:`cleanup_hub_atoms` for the source side.  Hub atoms
        that fan out to *max_outbound* or more outbound ``related-to``
        connections dilute spreading activation signal.  This idempotent
        cleanup finds all such atoms and deletes their weakest outbound
        ``related-to`` synapses until each is at ``max_outbound - 1``
        (one buffer slot below cap, preventing oscillation at the boundary
        every Hebbian cycle).

        The entire operation is atomic: the CTE identifying excess synapses
        and the DELETE removing them run in a single
        ``execute_write_returning`` call.

        Typed semantic links (``caused-by``, ``contradicts``, etc.) are
        never deleted by this method.

        Parameters
        ----------
        max_outbound:
            Hard cap on outbound ``related-to`` synapses per atom.
            Atoms at or above this count are trimmed to
            ``max_outbound - 1``.  Defaults to
            :data:`_MAX_OUTBOUND_RELATED_TO` (50).

        Returns
        -------
        int
            Total number of synapses deleted across all over-cap atoms.
        """
        deleted_rows = await self._storage.execute_write_returning(
            """
            DELETE FROM synapses WHERE id IN (
                SELECT id FROM (
                    WITH hub_sources AS (
                        SELECT source_id, COUNT(*) AS cnt
                        FROM synapses
                        WHERE relationship = 'related-to'
                        GROUP BY source_id
                        HAVING cnt >= ?
                    ),
                    ranked AS (
                        SELECT s.id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY s.source_id ORDER BY s.strength ASC
                               ) AS rn,
                               h.cnt - (? - 1) AS excess
                        FROM synapses s
                        JOIN hub_sources h ON h.source_id = s.source_id
                        WHERE s.relationship = 'related-to'
                    )
                    SELECT id FROM ranked WHERE rn <= excess
                )
            ) RETURNING id
            """,
            (max_outbound, max_outbound),
        )

        total_deleted = len(deleted_rows)

        if total_deleted:
            log.info(
                "Hub cleanup: deleted %d over-cap outbound related-to synapses "
                "(max_outbound=%d)",
                total_deleted,
                max_outbound,
            )

        return total_deleted

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Compute aggregate statistics about the synapse population.

        Returns
        -------
        dict[str, Any]
            A dictionary containing:

            - ``total`` -- total number of synapses.
            - ``avg_strength`` -- mean strength across all synapses.
            - ``by_relationship`` -- dict mapping each relationship type
              to its count.
            - ``weakest`` -- the synapse with the lowest strength
              (as a dict), or ``None``.
            - ``strongest`` -- the synapse with the highest strength
              (as a dict), or ``None``.
        """
        # Query 1: aggregate stats — total count, average, min and max strength
        # in a single scan, plus the full row for weakest and strongest synapses
        # via subqueries.  Combines 4 of the original 5 queries into 1.
        agg_rows = await self._storage.execute(
            """
            SELECT
                COUNT(*)       AS total,
                AVG(strength)  AS avg_str,
                MIN(strength)  AS min_str,
                MAX(strength)  AS max_str
            FROM synapses
            """
        )
        agg = agg_rows[0] if agg_rows else None
        total: int = agg["total"] if agg else 0
        avg_strength: float = round(float(agg["avg_str"] or 0.0), 4) if agg else 0.0

        weakest: dict | None = None
        strongest: dict | None = None
        if total > 0:
            # Fetch the extreme rows only when synapses exist.
            extremes_rows = await self._storage.execute(
                """
                SELECT * FROM (
                    SELECT * FROM synapses WHERE strength = (SELECT MIN(strength) FROM synapses) LIMIT 1
                )
                UNION ALL
                SELECT * FROM (
                    SELECT * FROM synapses WHERE strength = (SELECT MAX(strength) FROM synapses) LIMIT 1
                )
                """
            )
            if extremes_rows:
                weakest = Synapse.from_row(extremes_rows[0]).to_dict()
            if len(extremes_rows) >= 2:
                strongest = Synapse.from_row(extremes_rows[1]).to_dict()
            elif extremes_rows:
                # Only one row exists (min == max) — it is both weakest and strongest.
                strongest = weakest

        # Query 2: relationship breakdown — GROUP BY is a separate scan.
        rel_rows = await self._storage.execute(
            """
            SELECT relationship, COUNT(*) AS cnt
            FROM synapses
            GROUP BY relationship
            """
        )
        by_relationship: dict[str, int] = {
            row["relationship"]: row["cnt"] for row in rel_rows
        }

        return {
            "total": total,
            "avg_strength": avg_strength,
            "by_relationship": by_relationship,
            "weakest": weakest,
            "strongest": strongest,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_synapse(self, synapse_id: int) -> Synapse:
        """Fetch a single synapse by ID.

        Raises
        ------
        RuntimeError
            If the synapse does not exist (indicates a logic error in the
            calling code).
        """
        rows = await self._storage.execute(
            "SELECT * FROM synapses WHERE id = ?",
            (synapse_id,),
        )
        if not rows:
            raise RuntimeError(
                f"Synapse {synapse_id} not found after insert/update"
            )
        return Synapse.from_row(rows[0])

    async def _fetch_any_between(
        self,
        atom_id_a: int,
        atom_id_b: int,
    ) -> Synapse | None:
        """Return the strongest synapse between two atoms (any type, either direction).

        Used by Hebbian update to strengthen whichever connection already
        exists, regardless of its relationship type.  Returns ``None`` if no
        synapse exists between the pair.
        """
        rows = await self._storage.execute(
            """
            SELECT * FROM synapses
            WHERE (source_id = ? AND target_id = ?)
               OR (source_id = ? AND target_id = ?)
            ORDER BY strength DESC
            LIMIT 1
            """,
            (atom_id_a, atom_id_b, atom_id_b, atom_id_a),
        )
        if not rows:
            return None
        return Synapse.from_row(rows[0])

    async def _fetch_by_triple(
        self,
        source_id: int,
        target_id: int,
        relationship: str,
    ) -> Synapse | None:
        """Fetch a synapse by its unique (source, target, relationship) triple.

        Returns ``None`` when no matching synapse exists.
        """
        rows = await self._storage.execute(
            """
            SELECT * FROM synapses
            WHERE source_id = ? AND target_id = ? AND relationship = ?
            """,
            (source_id, target_id, relationship),
        )
        if not rows:
            return None
        return Synapse.from_row(rows[0])

    async def _check_co_activation(
        self,
        atom_id_a: int,
        atom_id_b: int,
        window_minutes: int,
    ) -> bool:
        """Check whether two atoms were accessed within *window_minutes* of each other.

        Compares the two atoms' ``last_accessed_at`` timestamps against each
        other rather than against the current time.  This correctly handles
        sessions longer than the window — atoms accessed an hour apart in a
        long session are not co-activated, while atoms accessed within
        ``window_minutes`` of each other are, regardless of how long ago that was.

        Parameters
        ----------
        atom_id_a:
            First atom ID.
        atom_id_b:
            Second atom ID.
        window_minutes:
            Maximum gap (in minutes) between the two access times.

        Returns
        -------
        bool
            ``True`` if both atoms have access timestamps and they fall within
            *window_minutes* of each other.
        """
        rows = await self._storage.execute(
            """
            SELECT
                MAX(JULIANDAY(last_accessed_at)) - MIN(JULIANDAY(last_accessed_at))
                    AS diff_days
            FROM atoms
            WHERE id IN (?, ?)
              AND is_deleted = 0
              AND last_accessed_at IS NOT NULL
            """,
            (atom_id_a, atom_id_b),
        )
        if not rows or rows[0]["diff_days"] is None:
            return False
        diff_minutes = rows[0]["diff_days"] * 24 * 60
        return diff_minutes <= window_minutes

    async def _check_min_accesses(
        self,
        atom_id_a: int,
        atom_id_b: int,
        min_accesses: int,
    ) -> bool:
        """Check whether both atoms have sufficient access history.

        This prevents creating spurious Hebbian connections between
        brand-new atoms that happen to appear together in one session.
        By requiring a minimum access count, we ensure that atoms have
        demonstrated relevance through repeated use before linking them.

        Parameters
        ----------
        atom_id_a:
            First atom ID.
        atom_id_b:
            Second atom ID.
        min_accesses:
            Minimum number of prior accesses required for each atom.

        Returns
        -------
        bool
            ``True`` if both atoms have at least ``min_accesses`` accesses.
        """
        if min_accesses <= 0:
            return True

        rows = await self._storage.execute(
            """
            SELECT COUNT(*) AS cnt FROM atoms
            WHERE id IN (?, ?)
              AND is_deleted = 0
              AND access_count >= ?
            """,
            (atom_id_a, atom_id_b, min_accesses),
        )
        # Both atoms must meet the threshold (count == 2).
        return rows[0]["cnt"] == 2 if rows else False
