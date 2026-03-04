"""Atom CRUD operations and type definitions for the memories system.

An **atom** is the fundamental unit of memory -- a single discrete piece of
knowledge stored in the brain-like memory graph.  Each atom has a type that
describes the kind of knowledge it represents:

- **fact** -- objective, verifiable information.
- **experience** -- something observed or learned through practice.
- **skill** -- a procedural capability or technique.
- **preference** -- a subjective choice or style decision.
- **insight** -- a higher-order conclusion synthesised from other atoms.
- **antipattern** -- a known-bad practice, optionally with severity and an
  alternative (``instead``).

This module provides:

* :class:`Atom` -- an immutable-ish dataclass that maps 1:1 with a row in the
  ``atoms`` table, with convenience conversion helpers.
* :class:`AtomManager` -- async CRUD + search operations that coordinate the
  :class:`~memories.storage.Storage` layer and
  :class:`~memories.embeddings.EmbeddingEngine`.

Usage::

    from memories.storage import Storage
    from memories.embeddings import EmbeddingEngine
    from memories.atoms import AtomManager

    store = Storage()
    await store.initialize()
    engine = EmbeddingEngine(store)

    mgr = AtomManager(store, engine)
    atom = await mgr.create("Redis SCAN is O(N) over the full keyspace", type="fact")
    print(atom.to_dict())
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from memories.embeddings import EmbeddingEngine
from memories.storage import Storage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATOM_TYPES: tuple[str, ...] = (
    "fact",
    "experience",
    "skill",
    "preference",
    "insight",
    "antipattern",
    "task",
)
"""Allowed values for the ``atoms.type`` column."""

SEVERITY_LEVELS: tuple[str, ...] = (
    "low",
    "medium",
    "high",
    "critical",
)
"""Allowed severity levels for antipattern atoms."""

TASK_STATUSES: tuple[str, ...] = (
    "pending",
    "active",
    "done",
    "archived",
)
"""Allowed status values for task atoms."""


# ---------------------------------------------------------------------------
# Atom dataclass
# ---------------------------------------------------------------------------


@dataclass
class Atom:
    """In-memory representation of a single atom row.

    Fields map directly to the ``atoms`` table schema.  The ``tags``
    column is stored as a JSON string in SQLite but exposed here as a
    Python :class:`list` of strings for ergonomic access.

    Parameters
    ----------
    id:
        Auto-incremented primary key.
    content:
        The knowledge payload.
    type:
        One of :data:`ATOM_TYPES`.
    region:
        Logical grouping (defaults to ``"general"``).
    confidence:
        Belief strength in ``[0, 1]`` (defaults to ``1.0``).
    importance:
        Priority weight in ``[0, 1]`` (defaults to ``0.5``).
        Higher values cause the atom to surface more prominently
        during recall.  Use for critical memories that should be
        prioritised regardless of recency or frequency.
    access_count:
        How many times this atom has been retrieved by the user.
    last_accessed_at:
        ISO-8601 timestamp of the most recent retrieval, or ``None``.
    created_at:
        ISO-8601 timestamp when the atom was first stored.
    updated_at:
        ISO-8601 timestamp of the last mutation.
    source_project:
        Optional project identifier that produced this atom.
    source_session:
        Optional session identifier.
    source_file:
        Optional originating file path.
    tags:
        Flat list of string tags for categorisation and FTS.
    severity:
        Antipattern-only severity level (one of :data:`SEVERITY_LEVELS`).
    instead:
        Antipattern-only recommended alternative.
    is_deleted:
        Soft-delete flag.  When ``True`` the atom is hidden from normal
        queries but remains in the database for audit purposes.
    """

    id: int
    content: str
    type: str
    region: str = "general"
    confidence: float = 1.0
    importance: float = 0.5
    access_count: int = 0
    last_accessed_at: str | None = None
    created_at: str = ""
    updated_at: str = ""
    source_project: str | None = None
    source_session: str | None = None
    source_file: str | None = None
    tags: list[str] = field(default_factory=list)
    severity: str | None = None
    instead: str | None = None
    is_deleted: bool = False
    task_status: str | None = None  # For task atoms: pending/active/done/archived

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_row(cls, row: Any) -> Atom:
        """Create an :class:`Atom` from a :class:`sqlite3.Row`.

        Handles the JSON-string-to-list conversion for ``tags`` and the
        integer-to-bool conversion for ``is_deleted``.

        Parameters
        ----------
        row:
            A :class:`sqlite3.Row` (or any mapping supporting key access)
            from the ``atoms`` table.

        Returns
        -------
        Atom
            A fully populated dataclass instance.
        """
        raw_tags = row["tags"]
        if raw_tags:
            try:
                tags = json.loads(raw_tags)
                if not isinstance(tags, list):
                    tags = []
            except (json.JSONDecodeError, TypeError):
                tags = []
        else:
            tags = []

        # Handle task_status gracefully for old databases without the column.
        task_status = None
        try:
            task_status = row["task_status"]
        except (KeyError, IndexError):
            pass

        return cls(
            id=row["id"],
            content=row["content"],
            type=row["type"],
            region=row["region"],
            confidence=row["confidence"],
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed_at=row["last_accessed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            source_project=row["source_project"],
            source_session=row["source_session"],
            source_file=row["source_file"],
            tags=tags,
            severity=row["severity"],
            instead=row["instead"],
            is_deleted=bool(row["is_deleted"]),
            task_status=task_status,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the atom to a plain dict suitable for MCP tool responses.

        ``tags`` is kept as a Python list (JSON-serialisable) rather than
        being re-encoded to a JSON string.

        Returns
        -------
        dict[str, Any]
            All fields as a flat dictionary.
        """
        d = asdict(self)
        # Ensure is_deleted is a plain bool for JSON consumers.
        d["is_deleted"] = bool(d["is_deleted"])
        return d


# ---------------------------------------------------------------------------
# Validation helpers (module-private)
# ---------------------------------------------------------------------------


def _validate_atom_type(atom_type: str) -> None:
    """Raise :class:`ValueError` if *atom_type* is not in :data:`ATOM_TYPES`."""
    if atom_type not in ATOM_TYPES:
        raise ValueError(
            f"Invalid atom type {atom_type!r}. "
            f"Must be one of: {', '.join(ATOM_TYPES)}"
        )


def _validate_severity(severity: str | None) -> None:
    """Raise :class:`ValueError` if *severity* is set but not a valid level."""
    if severity is not None and severity not in SEVERITY_LEVELS:
        raise ValueError(
            f"Invalid severity {severity!r}. "
            f"Must be one of: {', '.join(SEVERITY_LEVELS)}"
        )


def _validate_task_status(task_status: str | None) -> None:
    """Raise :class:`ValueError` if *task_status* is set but not a valid status."""
    if task_status is not None and task_status not in TASK_STATUSES:
        raise ValueError(
            f"Invalid task_status {task_status!r}. "
            f"Must be one of: {', '.join(TASK_STATUSES)}"
        )


def _validate_confidence(confidence: float) -> None:
    """Raise :class:`ValueError` if *confidence* is outside ``[0, 1]``."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Confidence must be between 0.0 and 1.0, got {confidence}"
        )


def _validate_importance(importance: float) -> None:
    """Raise :class:`ValueError` if *importance* is outside ``[0, 1]``."""
    if not 0.0 <= importance <= 1.0:
        raise ValueError(
            f"Importance must be between 0.0 and 1.0, got {importance}"
        )


# ---------------------------------------------------------------------------
# Atom manager
# ---------------------------------------------------------------------------


class AtomManager:
    """Async CRUD manager for atoms.

    Coordinates writes to the ``atoms`` table, embedding generation via
    :class:`~memories.embeddings.EmbeddingEngine`, region bookkeeping in
    the ``regions`` table, and full-text search via the ``atoms_fts``
    virtual table.

    Parameters
    ----------
    storage:
        An initialised :class:`~memories.storage.Storage` instance.
    embeddings:
        An initialised :class:`~memories.embeddings.EmbeddingEngine`
        instance.
    """

    def __init__(self, storage: Storage, embeddings: EmbeddingEngine) -> None:
        self._storage = storage
        self._embeddings = embeddings

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create(
        self,
        content: str,
        type: str,
        region: str = "general",
        tags: list[str] | None = None,
        severity: str | None = None,
        instead: str | None = None,
        source_project: str | None = None,
        source_session: str | None = None,
        source_file: str | None = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        task_status: str | None = None,
    ) -> Atom:
        """Create a new atom, embed it, and update the region counter.

        Parameters
        ----------
        content:
            The knowledge payload (required, must not be empty).
        type:
            One of :data:`ATOM_TYPES`.
        region:
            Logical grouping (created automatically if it does not exist).
        tags:
            Optional list of string tags.
        severity:
            Optional severity for antipattern atoms.
        instead:
            Optional alternative for antipattern atoms.
        source_project:
            Project that produced this atom.
        source_session:
            Session identifier.
        source_file:
            Originating file path.
        confidence:
            Initial confidence in ``[0, 1]``.
        importance:
            Priority weight in ``[0, 1]``.  Higher values surface
            the memory more prominently during recall.
        task_status:
            For task atoms only: one of 'pending', 'active', 'done', 'archived'.
            Auto-set to 'pending' for task atoms if not provided.

        Returns
        -------
        Atom
            The newly created atom with all database-generated fields
            populated (``id``, ``created_at``, ``updated_at``).

        Raises
        ------
        ValueError
            If *type*, *severity*, *confidence*, *importance*, or
            *task_status* fail validation, or if *content* is empty.
        """
        # --- Validation ---------------------------------------------------
        if not content or not content.strip():
            raise ValueError("Atom content must not be empty")
        _validate_atom_type(type)
        _validate_severity(severity)
        _validate_confidence(confidence)
        _validate_importance(importance)
        _validate_task_status(task_status)

        # Auto-set task_status for task atoms.
        if type == "task" and task_status is None:
            task_status = "pending"

        tags_json = json.dumps(tags) if tags else None

        # --- Atomic insert + region update --------------------------------
        def _do_create(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                """
                INSERT INTO atoms
                    (content, type, region, confidence, importance, tags,
                     severity, instead, source_project, source_session, source_file,
                     task_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content,
                    type,
                    region,
                    confidence,
                    importance,
                    tags_json,
                    severity,
                    instead,
                    source_project,
                    source_session,
                    source_file,
                    task_status,
                ),
            )
            new_id = cursor.lastrowid or 0
            conn.execute(
                "INSERT OR IGNORE INTO regions (name) VALUES (?)",
                (region,),
            )
            conn.execute(
                "UPDATE regions SET atom_count = atom_count + 1 WHERE name = ?",
                (region,),
            )
            return new_id

        atom_id = await self._storage.execute_transaction(_do_create)

        # --- Embed and store vector (I/O — outside transaction) -----------
        try:
            await self._embeddings.embed_and_store(atom_id, content)
        except Exception:
            log.warning(
                "Failed to embed atom %d; vector search will miss it "
                "until re-embedded",
                atom_id,
            )

        # --- Return the fully populated atom ------------------------------
        return await self._fetch_atom(atom_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, atom_id: int) -> Atom | None:
        """Retrieve an atom by primary key, recording the access.

        This is the standard retrieval path used when a user (or the
        retrieval engine) actively reads an atom.  The access is tracked
        so that the Hebbian learning and decay systems can use frequency
        and recency signals.

        Uses a single ``UPDATE ... RETURNING`` statement so the access
        count increment and the read are one atomic round-trip instead of
        two (previously: SELECT then UPDATE then SELECT again).

        Parameters
        ----------
        atom_id:
            Primary key of the atom.

        Returns
        -------
        Atom | None
            The atom with updated access stats, or ``None`` if it does
            not exist or is soft-deleted.
        """
        rows = await self._storage.execute_write_returning(
            "UPDATE atoms "
            "SET access_count = access_count + 1, "
            "    last_accessed_at = datetime('now'), "
            "    updated_at = datetime('now') "
            "WHERE id = ? AND is_deleted = 0 "
            "RETURNING *",
            (atom_id,),
        )
        return Atom.from_row(rows[0]) if rows else None

    async def get_without_tracking(self, atom_id: int) -> Atom | None:
        """Retrieve an atom by primary key without updating access stats.

        Useful for internal operations (consolidation, synapse
        strengthening, migration) where the read should not influence
        the Hebbian learning or decay signals.

        Parameters
        ----------
        atom_id:
            Primary key of the atom.

        Returns
        -------
        Atom | None
            The atom, or ``None`` if it does not exist or is soft-deleted.
        """
        return await self._fetch_active_atom(atom_id)

    async def get_batch_without_tracking(
        self, atom_ids: list[int]
    ) -> dict[int, Atom]:
        """Retrieve multiple atoms by primary key without updating access stats.

        Efficient batch retrieval that avoids N+1 queries when checking
        multiple atoms (e.g., during spreading activation).

        Parameters
        ----------
        atom_ids:
            List of primary keys to fetch.

        Returns
        -------
        dict[int, Atom]
            Mapping of atom ID to Atom for non-deleted atoms.
            Missing/deleted atoms are omitted from the result.
        """
        if not atom_ids:
            return {}

        # Build parameterized query with placeholders
        placeholders = ",".join("?" * len(atom_ids))
        rows = await self._storage.execute(
            f"SELECT * FROM atoms WHERE id IN ({placeholders}) AND is_deleted = 0",
            tuple(atom_ids),
        )
        return {row["id"]: Atom.from_row(row) for row in rows}

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update(
        self,
        atom_id: int,
        content: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        region: str | None = None,
    ) -> Atom | None:
        """Update one or more mutable fields on an existing atom.

        Only the fields whose arguments are not ``None`` are changed.
        If ``content`` is changed the embedding is regenerated
        automatically.  If ``region`` is changed the old and new region
        counters are adjusted.

        Parameters
        ----------
        atom_id:
            Primary key of the atom to update.
        content:
            New knowledge payload.
        type:
            New atom type (must be in :data:`ATOM_TYPES`).
        tags:
            Replacement tag list (completely replaces the old tags).
        confidence:
            New confidence value in ``[0, 1]``.
        importance:
            New importance value in ``[0, 1]``.
        region:
            New region (created automatically if it does not exist).

        Returns
        -------
        Atom | None
            The updated atom, or ``None`` if the atom does not exist
            or is soft-deleted.

        Raises
        ------
        ValueError
            If any provided value fails validation.
        """
        # --- Validation ---------------------------------------------------
        if type is not None:
            _validate_atom_type(type)
        if confidence is not None:
            _validate_confidence(confidence)
        if importance is not None:
            _validate_importance(importance)

        # --- Build SET clause dynamically ---------------------------------
        set_clauses: list[str] = []
        params: list[Any] = []

        if content is not None:
            set_clauses.append("content = ?")
            params.append(content)
        if type is not None:
            set_clauses.append("type = ?")
            params.append(type)
        if tags is not None:
            set_clauses.append("tags = ?")
            params.append(json.dumps(tags))
        if confidence is not None:
            set_clauses.append("confidence = ?")
            params.append(confidence)
        if importance is not None:
            set_clauses.append("importance = ?")
            params.append(importance)
        if region is not None:
            set_clauses.append("region = ?")
            params.append(region)

        if not set_clauses:
            return await self._fetch_active_atom(atom_id)

        # Always bump updated_at.
        set_clauses.append("updated_at = datetime('now')")

        set_sql = f"UPDATE atoms SET {', '.join(set_clauses)} WHERE id = ? AND is_deleted = 0"
        params.append(atom_id)

        # --- Atomic fetch + update + region adjustment --------------------
        def _do_update(conn: sqlite3.Connection) -> str | None:
            row = conn.execute(
                "SELECT region FROM atoms WHERE id = ? AND is_deleted = 0",
                (atom_id,),
            ).fetchone()
            if row is None:
                return None  # atom doesn't exist or is deleted

            old_region: str = row[0]

            conn.execute(set_sql, tuple(params))

            if region is not None and region != old_region:
                conn.execute(
                    "INSERT OR IGNORE INTO regions (name) VALUES (?)",
                    (region,),
                )
                conn.execute(
                    "UPDATE regions SET atom_count = atom_count + 1 WHERE name = ?",
                    (region,),
                )
                conn.execute(
                    "UPDATE regions SET atom_count = MAX(atom_count - 1, 0) WHERE name = ?",
                    (old_region,),
                )

            return old_region

        old_region = await self._storage.execute_transaction(_do_update)
        if old_region is None:
            return None

        # --- Re-embed if content changed (I/O — outside transaction) ------
        if content is not None:
            try:
                await self._embeddings.embed_and_store(atom_id, content)
            except Exception:
                log.warning(
                    "Failed to re-embed atom %d after content update",
                    atom_id,
                )

        return await self._fetch_atom(atom_id)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def soft_delete(self, atom_id: int) -> bool:
        """Mark an atom as deleted without removing it from the database.

        The atom is hidden from all normal queries (``is_deleted = 0``
        filter) but remains available for audit or recovery.  The
        associated region counter is decremented.

        Parameters
        ----------
        atom_id:
            Primary key of the atom to soft-delete.

        Returns
        -------
        bool
            ``True`` if the atom existed and was soft-deleted,
            ``False`` otherwise.
        """
        def _do_soft_delete(conn: sqlite3.Connection) -> str | None:
            row = conn.execute(
                "SELECT region FROM atoms WHERE id = ? AND is_deleted = 0",
                (atom_id,),
            ).fetchone()
            if row is None:
                return None

            region: str = row[0]
            conn.execute(
                "UPDATE atoms SET is_deleted = 1, updated_at = datetime('now') WHERE id = ?",
                (atom_id,),
            )
            conn.execute(
                "UPDATE regions SET atom_count = MAX(atom_count - 1, 0) WHERE name = ?",
                (region,),
            )
            return region

        region = await self._storage.execute_transaction(_do_soft_delete)
        if region is None:
            return False

        log.info("Soft-deleted atom %d (region=%s)", atom_id, region)
        return True

    async def hard_delete(self, atom_id: int) -> bool:
        """Permanently remove an atom from all tables.

        Deletes the row from ``atoms`` (which cascades to FTS triggers),
        removes the embedding from ``atoms_vec``, and adjusts the region
        counter if the atom was not already soft-deleted.

        Parameters
        ----------
        atom_id:
            Primary key of the atom to destroy.

        Returns
        -------
        bool
            ``True`` if the atom existed and was removed, ``False``
            otherwise.
        """
        vec_available = self._storage.vec_available

        def _do_hard_delete(conn: sqlite3.Connection) -> tuple[bool, str]:
            row = conn.execute(
                "SELECT region, is_deleted FROM atoms WHERE id = ?",
                (atom_id,),
            ).fetchone()
            if row is None:
                return False, ""

            region: str = row[0]
            was_deleted: bool = bool(row[1])

            if vec_available:
                conn.execute(
                    "DELETE FROM atoms_vec WHERE atom_id = ?",
                    (atom_id,),
                )

            conn.execute(
                "DELETE FROM atoms WHERE id = ?",
                (atom_id,),
            )

            if not was_deleted:
                conn.execute(
                    "UPDATE regions SET atom_count = MAX(atom_count - 1, 0) WHERE name = ?",
                    (region,),
                )

            return True, region

        existed, region = await self._storage.execute_transaction(_do_hard_delete)
        if not existed:
            return False

        log.info("Hard-deleted atom %d (region=%s)", atom_id, region)
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_fts(self, query: str, limit: int = 20) -> list[Atom]:
        """Full-text search on atom content and tags.

        Uses the ``atoms_fts`` FTS5 virtual table.  Results are ordered
        by FTS5 rank (most relevant first) and limited to non-deleted
        atoms.

        Parameters
        ----------
        query:
            An FTS5 query string (supports ``AND``, ``OR``, ``NOT``,
            phrase matching with ``"..."``, prefix matching with ``*``).
        limit:
            Maximum number of results.

        Returns
        -------
        list[Atom]
            Matching atoms ordered by relevance.
        """
        if not query or not query.strip():
            return []

        rows = await self._storage.execute(
            """
            SELECT a.*
            FROM atoms_fts f
            JOIN atoms a ON a.id = f.rowid
            WHERE atoms_fts MATCH ?
              AND a.is_deleted = 0
            ORDER BY f.rank
            LIMIT ?
            """,
            (query, limit),
        )
        return [Atom.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Filtered queries
    # ------------------------------------------------------------------

    async def get_by_region(self, region: str, limit: int = 50) -> list[Atom]:
        """Retrieve non-deleted atoms belonging to a specific region.

        Parameters
        ----------
        region:
            The region name to filter on.
        limit:
            Maximum number of results.

        Returns
        -------
        list[Atom]
            Atoms in the region ordered by descending confidence then
            most recently created first.
        """
        rows = await self._storage.execute(
            """
            SELECT * FROM atoms
            WHERE region = ? AND is_deleted = 0
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            """,
            (region, limit),
        )
        return [Atom.from_row(r) for r in rows]

    async def get_stale(self, days: int = 30) -> list[Atom]:
        """Find atoms that have not been accessed in *days* or more.

        An atom is considered stale if ``last_accessed_at`` is older
        than *days* ago, **or** if it has never been accessed
        (``last_accessed_at IS NULL``) and was created more than *days*
        ago.

        Parameters
        ----------
        days:
            Staleness threshold in days.

        Returns
        -------
        list[Atom]
            Stale atoms ordered by ``last_accessed_at`` ascending
            (oldest access first).
        """
        rows = await self._storage.execute(
            """
            SELECT * FROM atoms
            WHERE is_deleted = 0
              AND (
                  (last_accessed_at IS NOT NULL
                   AND last_accessed_at < datetime('now', ?))
                  OR
                  (last_accessed_at IS NULL
                   AND created_at < datetime('now', ?))
              )
            ORDER BY COALESCE(last_accessed_at, created_at) ASC
            """,
            (f"-{days} days", f"-{days} days"),
        )
        return [Atom.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Compute aggregate statistics about the atom population.

        Returns
        -------
        dict[str, Any]
            A dictionary containing:

            - ``total`` -- total number of non-deleted atoms.
            - ``by_type`` -- dict mapping each atom type to its count.
            - ``by_region`` -- dict mapping each region to its count.
            - ``avg_confidence`` -- mean confidence across all non-deleted atoms.
            - ``total_deleted`` -- number of soft-deleted atoms.
        """
        # Query 1: totals + avg confidence in a single aggregate scan.
        agg = await self._storage.execute(
            """
            SELECT
                SUM(CASE WHEN is_deleted=0 THEN 1 ELSE 0 END) AS total,
                SUM(CASE WHEN is_deleted=1 THEN 1 ELSE 0 END) AS total_deleted,
                AVG(CASE WHEN is_deleted=0 THEN confidence END) AS avg_conf
            FROM atoms
            """
        )
        if agg:
            row = agg[0]
            total = row["total"] or 0
            total_deleted = row["total_deleted"] or 0
            avg_confidence = round(row["avg_conf"] or 0.0, 4)
        else:
            total = 0
            total_deleted = 0
            avg_confidence = 0.0

        # Query 2: UNION ALL for type + region breakdown in one round-trip.
        breakdown = await self._storage.execute(
            """
            SELECT 'type' AS grp_kind, type AS grp_val, COUNT(*) AS cnt
            FROM atoms WHERE is_deleted=0 GROUP BY type
            UNION ALL
            SELECT 'region', region, COUNT(*)
            FROM atoms WHERE is_deleted=0 GROUP BY region
            """
        )
        by_type: dict[str, int] = {}
        by_region: dict[str, int] = {}
        for brow in breakdown:
            if brow["grp_kind"] == "type":
                by_type[brow["grp_val"]] = brow["cnt"]
            else:
                by_region[brow["grp_val"]] = brow["cnt"]

        return {
            "total": total,
            "by_type": by_type,
            "by_region": by_region,
            "avg_confidence": avg_confidence,
            "total_deleted": total_deleted,
        }

    # ------------------------------------------------------------------
    # Access tracking
    # ------------------------------------------------------------------

    async def record_access(self, atom_id: int) -> None:
        """Increment ``access_count`` and update ``last_accessed_at``.

        This is the core Hebbian signal -- atoms that are retrieved
        frequently develop stronger "memory traces" which the decay
        and consolidation systems use to decide what to keep, merge,
        or prune.

        Parameters
        ----------
        atom_id:
            Primary key of the atom being accessed.
        """
        await self._storage.execute_write(
            """
            UPDATE atoms
            SET access_count = access_count + 1,
                last_accessed_at = datetime('now')
            WHERE id = ?
            """,
            (atom_id,),
        )

    async def record_access_batch(self, atom_ids: list[int]) -> None:
        """Increment ``access_count`` for multiple atoms in a single UPDATE.

        More efficient than calling :meth:`record_access` in a loop
        because it issues one SQL statement instead of N.

        Parameters
        ----------
        atom_ids:
            Primary keys of the atoms being accessed.
        """
        if not atom_ids:
            return
        placeholders = ",".join("?" * len(atom_ids))
        await self._storage.execute_write(
            f"UPDATE atoms SET access_count = access_count + 1, "
            f"last_accessed_at = datetime('now') WHERE id IN ({placeholders})",
            tuple(atom_ids),
        )

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    async def update_task_status(
        self,
        task_id: int,
        status: str,
    ) -> Atom | None:
        """Update the status of a task atom.

        Parameters
        ----------
        task_id:
            Primary key of the task atom.
        status:
            New status: 'pending', 'active', 'done', or 'archived'.

        Returns
        -------
        Atom | None
            The updated atom, or ``None`` if not found or not a task.

        Raises
        ------
        ValueError
            If *status* is not a valid task status.
        """
        _validate_task_status(status)

        def _do_update(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT type FROM atoms WHERE id = ? AND is_deleted = 0",
                (task_id,),
            ).fetchone()
            if row is None or row[0] != "task":
                return False

            conn.execute(
                """
                UPDATE atoms
                SET task_status = ?, updated_at = datetime('now')
                WHERE id = ?
                """,
                (status, task_id),
            )
            return True

        updated = await self._storage.execute_transaction(_do_update)
        if not updated:
            return None

        log.info("Updated task %d status to %s", task_id, status)
        return await self._fetch_atom(task_id)

    async def get_tasks(
        self,
        status: str | None = None,
        region: str | None = None,
    ) -> list[Atom]:
        """Retrieve task atoms with optional filters.

        Parameters
        ----------
        status:
            Filter by task status (e.g., 'pending', 'active').
        region:
            Filter by region.

        Returns
        -------
        list[Atom]
            Task atoms matching the filters.
        """
        conditions = ["type = 'task'", "is_deleted = 0"]
        params: list[str] = []

        if status is not None:
            conditions.append("task_status = ?")
            params.append(status)
        if region is not None:
            conditions.append("region = ?")
            params.append(region)

        sql = f"SELECT * FROM atoms WHERE {' AND '.join(conditions)} ORDER BY created_at DESC"
        rows = await self._storage.execute(sql, tuple(params))
        return [Atom.from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_atom(self, atom_id: int) -> Atom:
        """Fetch a single atom by ID (does not filter on ``is_deleted``).

        Raises
        ------
        RuntimeError
            If the atom does not exist (indicates a logic error in the
            calling code).
        """
        rows = await self._storage.execute(
            "SELECT * FROM atoms WHERE id = ?",
            (atom_id,),
        )
        if not rows:
            raise RuntimeError(f"Atom {atom_id} not found after insert/update")
        return Atom.from_row(rows[0])

    async def _fetch_active_atom(self, atom_id: int) -> Atom | None:
        """Fetch a non-deleted atom by ID.

        Returns ``None`` when the atom does not exist or is soft-deleted.
        """
        rows = await self._storage.execute(
            "SELECT * FROM atoms WHERE id = ? AND is_deleted = 0",
            (atom_id,),
        )
        if not rows:
            return None
        return Atom.from_row(rows[0])

