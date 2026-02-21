"""Core storage layer for the memories system.

Manages a SQLite database with sqlite-vec for vector search and FTS5 for
keyword search.  All public methods are async-friendly, wrapping synchronous
sqlite3 calls via :func:`anyio.to_thread.run_sync`.

Connection strategy:
    - A single ``threading.Lock`` serialises write operations.
    - Thread-local persistent connections — each thread pool worker keeps one
      long-lived connection open, eliminating per-call open/close overhead.
    - WAL mode enables concurrent readers alongside a single writer.

Usage::

    from memories.storage import Storage

    store = Storage(config.db_path)
    await store.initialize()
    row_id = await store.execute_write("INSERT INTO atoms ...", (...))
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

import anyio
import sqlite_vec

from memories.config import get_config

_T = TypeVar("_T")

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding serialisation helpers
# ---------------------------------------------------------------------------


def serialize_embedding(vec: list[float]) -> bytes:
    """Pack a float vector into a compact binary representation.

    Parameters
    ----------
    vec:
        A list of floats (typically 768-dimensional).

    Returns
    -------
    bytes
        Little-endian packed floats suitable for sqlite-vec queries.
    """
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_embedding(data: bytes) -> list[float]:
    """Unpack binary embedding data back into a list of floats.

    Parameters
    ----------
    data:
        Bytes previously produced by :func:`serialize_embedding`.

    Returns
    -------
    list[float]
        The original float vector.
    """
    count = len(data) // struct.calcsize("f")
    return list(struct.unpack(f"{count}f", data))


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
-- Core memory atoms
CREATE TABLE IF NOT EXISTS atoms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('fact','experience','skill','preference','insight','antipattern','task')),
    region TEXT NOT NULL DEFAULT 'general',
    confidence REAL NOT NULL DEFAULT 1.0,
    importance REAL NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    source_project TEXT,
    source_session TEXT,
    source_file TEXT,
    tags TEXT,
    severity TEXT CHECK(severity IN ('low','medium','high','critical') OR severity IS NULL),
    instead TEXT,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    task_status TEXT CHECK(task_status IN ('pending','active','done','archived') OR task_status IS NULL)
);

-- Full-text search virtual table (external content, synced via triggers)
CREATE VIRTUAL TABLE IF NOT EXISTS atoms_fts USING fts5(
    content, tags, content=atoms, content_rowid=id
);

-- FTS sync triggers
CREATE TRIGGER IF NOT EXISTS atoms_ai AFTER INSERT ON atoms BEGIN
    INSERT INTO atoms_fts(rowid, content, tags)
    VALUES (new.id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS atoms_ad AFTER DELETE ON atoms BEGIN
    INSERT INTO atoms_fts(atoms_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS atoms_au AFTER UPDATE ON atoms BEGIN
    INSERT INTO atoms_fts(atoms_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
    INSERT INTO atoms_fts(rowid, content, tags)
    VALUES (new.id, new.content, new.tags);
END;

-- Connections between atoms
CREATE TABLE IF NOT EXISTS synapses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES atoms(id),
    target_id INTEGER NOT NULL REFERENCES atoms(id),
    relationship TEXT NOT NULL CHECK(relationship IN (
        'related-to','caused-by','part-of','contradicts',
        'supersedes','elaborates','warns-against','encoded-with'
    )),
    strength REAL NOT NULL DEFAULT 0.5,
    bidirectional INTEGER NOT NULL DEFAULT 1,
    activated_count INTEGER NOT NULL DEFAULT 0,
    last_activated_at TEXT,
    tag_expires_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_synapses_source ON synapses(source_id);
CREATE INDEX IF NOT EXISTS idx_synapses_target ON synapses(target_id);
CREATE INDEX IF NOT EXISTS idx_synapses_strength ON synapses(strength DESC);

-- Logical groupings
CREATE TABLE IF NOT EXISTS regions (
    name TEXT PRIMARY KEY,
    description TEXT,
    atom_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Session tracking
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    atoms_accessed TEXT
);

-- Audit log for consolidation actions
CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    details TEXT,
    atoms_affected TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Embedding cache keyed by content hash
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Advisory locks for cross-process mutual exclusion
CREATE TABLE IF NOT EXISTS locks (
    name TEXT PRIMARY KEY,
    holder TEXT,
    acquired_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Hook invocation statistics for observability
CREATE TABLE IF NOT EXISTS hook_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hook_type TEXT NOT NULL CHECK(hook_type IN (
        'session-start','prompt-submit','pre-tool','post-tool',
        'post-response','stop','subagent-stop','pre-compact','post-tool-failure',
        'session-end','subagent-start','permission-request',
        'task-completed','notification'
    )),
    project TEXT,
    query TEXT,
    atoms_returned INTEGER NOT NULL DEFAULT 0,
    atom_ids TEXT,
    avg_score REAL,
    max_score REAL,
    budget_used INTEGER NOT NULL DEFAULT 0,
    budget_total INTEGER NOT NULL DEFAULT 0,
    compression_level INTEGER NOT NULL DEFAULT 0,
    seed_count INTEGER NOT NULL DEFAULT 0,
    total_activated INTEGER NOT NULL DEFAULT 0,
    novelty_result TEXT CHECK(novelty_result IN ('pass','fail') OR novelty_result IS NULL),
    latency_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Active sessions for sub-agent lineage detection.
-- One row per active Claude Code session (parent and sub-agent alike).
CREATE TABLE IF NOT EXISTS active_sessions (
    session_id TEXT PRIMARY KEY,
    project     TEXT,
    started_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Sub-agent → parent session relationships detected by project + recency.
CREATE TABLE IF NOT EXISTS session_lineage (
    child_session_id  TEXT PRIMARY KEY,
    parent_session_id TEXT NOT NULL
);

-- Cross-hook atom accumulation for Hebbian learning.
-- Each hook invocation writes the atom IDs it surfaced/created here,
-- keyed by the Claude Code session_id.  The stop hook reads this table
-- to run session_end_learning with the full cross-hook atom set, then
-- deletes the rows.
CREATE TABLE IF NOT EXISTS hook_session_atoms (
    claude_session_id TEXT NOT NULL,
    atom_id INTEGER NOT NULL,
    accessed_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (claude_session_id, atom_id)
);

-- User feedback on recalled atoms.
-- good → raises confidence; bad → lowers confidence.
CREATE TABLE IF NOT EXISTS atom_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    atom_id INTEGER NOT NULL REFERENCES atoms(id),
    signal TEXT NOT NULL CHECK(signal IN ('good', 'bad')),
    session_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_atom_feedback_atom ON atom_feedback(atom_id);
"""

# Vector table DDL -- only executed when sqlite-vec loads successfully.
_VEC_SCHEMA_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS atoms_vec USING vec0(
    atom_id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);
"""

# Additional indexes created after main schema.
_INDEX_SQL = """\
CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms(type);
CREATE INDEX IF NOT EXISTS idx_atoms_region ON atoms(region);
CREATE INDEX IF NOT EXISTS idx_atoms_confidence ON atoms(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_atoms_is_deleted ON atoms(is_deleted);
CREATE INDEX IF NOT EXISTS idx_hook_stats_type ON hook_stats(hook_type);
CREATE INDEX IF NOT EXISTS idx_hook_stats_created ON hook_stats(created_at);
CREATE INDEX IF NOT EXISTS idx_atoms_last_accessed
    ON atoms(last_accessed_at) WHERE is_deleted = 0;
CREATE INDEX IF NOT EXISTS idx_atoms_access_count
    ON atoms(access_count DESC) WHERE is_deleted = 0;
CREATE INDEX IF NOT EXISTS idx_atoms_created_at
    ON atoms(created_at) WHERE is_deleted = 0;
CREATE INDEX IF NOT EXISTS idx_atoms_type_deleted
    ON atoms(type, is_deleted);
CREATE INDEX IF NOT EXISTS idx_synapses_source_strength
    ON synapses(source_id, strength DESC);
CREATE INDEX IF NOT EXISTS idx_synapses_target_strength
    ON synapses(target_id, strength DESC);
CREATE INDEX IF NOT EXISTS idx_synapses_tag_expires
    ON synapses(tag_expires_at) WHERE tag_expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_synapses_relationship
    ON synapses(relationship);
CREATE INDEX IF NOT EXISTS idx_feedback_atom_processed
    ON atom_feedback(atom_id, processed_at);
"""


# ---------------------------------------------------------------------------
# Storage class
# ---------------------------------------------------------------------------


class Storage:
    """Async-friendly SQLite storage backend for the memories system.

    Parameters
    ----------
    db_path:
        Filesystem path for the SQLite database file.  Parent directories
        are created automatically during :meth:`initialize`.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        cfg = get_config()
        self._db_path: Path = db_path or cfg.db_path
        self._backup_dir: Path = cfg.backup_dir
        self._backup_count: int = cfg.backup_count
        self._write_lock = threading.Lock()
        self._local = threading.local()  # thread-local persistent connections
        self._all_connections: list[sqlite3.Connection] = []  # M-8: track every connection across all threads
        self._connections_lock = threading.Lock()  # guards _all_connections
        self._initialized = False
        self._vec_available = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> Path:
        """Filesystem path of the SQLite database."""
        return self._db_path

    @property
    def vec_available(self) -> bool:
        """Whether the sqlite-vec extension loaded successfully."""
        return self._vec_available

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Prepare the database for use.

        This method is idempotent and safe to call multiple times.  It:

        1. Creates the database directory and backup directory.
        2. Opens a connection with WAL mode and sqlite-vec loaded (if available).
        3. Creates all tables, indexes, triggers and virtual tables.
        4. Runs an automatic backup (pruning old backups).
        """
        await anyio.to_thread.run_sync(self._initialize_sync)
        self._initialized = True
        log.info(
            "Storage initialised at %s (vec=%s)",
            self._db_path,
            self._vec_available,
        )

    def _initialize_sync(self) -> None:
        """Synchronous initialisation run inside a worker thread."""
        # Verify SQLite version supports RETURNING clause (>= 3.35.0).
        sqlite_version = tuple(
            int(x) for x in sqlite3.sqlite_version.split(".")
        )
        if sqlite_version < (3, 35, 0):
            raise RuntimeError(
                f"SQLite {sqlite3.sqlite_version} is too old; memories requires >= 3.35.0 "
                "(needed for RETURNING clause support)"
            )

        # Ensure directories exist.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        # Probe for sqlite-vec support once at init time.
        self._vec_available = self._probe_vec_support()

        # Use a dedicated one-time connection for schema setup (not thread-local).
        conn = self._open_connection()
        try:
            # Core schema (atoms, FTS5, synapses, regions, sessions, etc.).
            conn.executescript(_SCHEMA_SQL)

            # Vector table only when extension is available.
            if self._vec_available:
                conn.executescript(_VEC_SCHEMA_SQL)

            # Migrations for existing databases (must run BEFORE index creation
            # because some indexes reference columns added by migrations).
            self._run_migrations(conn)

            conn.executescript(_INDEX_SQL)

            conn.execute("ANALYZE")
            conn.commit()
        finally:
            conn.close()

        # Create a backup after successful init.
        self._backup_sync()

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply schema migrations for existing databases.

        Each migration checks whether it needs to run before executing.
        """
        cursor = conn.execute("PRAGMA table_info(atoms)")
        columns = {row[1] for row in cursor.fetchall()}

        # Migration 1: Add importance column to atoms table (v1.1.0)
        if "importance" not in columns:
            conn.execute(
                "ALTER TABLE atoms ADD COLUMN importance REAL NOT NULL DEFAULT 0.5"
            )
            log.info("Migration: Added 'importance' column to atoms table")

        # Migration 2: Add task_status column for task atoms (v1.2.0)
        # NOTE: We intentionally omit the CHECK constraint from the ALTER TABLE
        # because SQLite 3.45+ raises a spurious "NOT NULL constraint failed"
        # when ADD COLUMN with CHECK follows an earlier ADD COLUMN with NOT NULL
        # DEFAULT on the same table.  The CHECK is already in _SCHEMA_SQL for
        # newly-created databases; migrated databases rely on application
        # validation.
        if "task_status" not in columns:
            conn.execute(
                "ALTER TABLE atoms ADD COLUMN task_status TEXT"
            )
            log.info("Migration: Added 'task_status' column to atoms table")

        # Migration 3: Add post-response to hook_stats CHECK constraint (v1.3.0)
        # SQLite can't ALTER a CHECK constraint — recreate the table.
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='hook_stats'"
        ).fetchone()
        if row and "post-response" not in row[0]:
            conn.executescript("""
                ALTER TABLE hook_stats RENAME TO hook_stats_old;

                CREATE TABLE hook_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hook_type TEXT NOT NULL CHECK(hook_type IN (
                        'session-start','prompt-submit','pre-tool','post-tool',
                        'post-response','stop','subagent-stop','pre-compact'
                    )),
                    project TEXT,
                    query TEXT,
                    atoms_returned INTEGER NOT NULL DEFAULT 0,
                    atom_ids TEXT,
                    avg_score REAL,
                    max_score REAL,
                    budget_used INTEGER NOT NULL DEFAULT 0,
                    budget_total INTEGER NOT NULL DEFAULT 0,
                    compression_level INTEGER NOT NULL DEFAULT 0,
                    seed_count INTEGER NOT NULL DEFAULT 0,
                    total_activated INTEGER NOT NULL DEFAULT 0,
                    novelty_result TEXT CHECK(novelty_result IN ('pass','fail') OR novelty_result IS NULL),
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                INSERT INTO hook_stats SELECT * FROM hook_stats_old;
                DROP TABLE hook_stats_old;
            """)
            log.info("Migration: Added 'post-response' to hook_stats CHECK constraint")

        # Migration 4: Add processed_at to atom_feedback for idempotent consolidation (v1.4.0)
        cursor = conn.execute("PRAGMA table_info(atom_feedback)")
        fb_columns = {row[1] for row in cursor.fetchall()}
        if "processed_at" not in fb_columns:
            conn.execute(
                "ALTER TABLE atom_feedback ADD COLUMN processed_at TEXT"
            )
            log.info("Migration: Added 'processed_at' column to atom_feedback table")

        # Migration 5: Add singleton retrieval_weights table for auto-tuning (v1.5.0)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='retrieval_weights'"
        ).fetchone()
        if row is None:
            conn.execute("""
                CREATE TABLE retrieval_weights (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    confidence       REAL NOT NULL DEFAULT 0.07,
                    importance       REAL NOT NULL DEFAULT 0.11,
                    frequency        REAL NOT NULL DEFAULT 0.07,
                    recency          REAL NOT NULL DEFAULT 0.10,
                    spread_activation REAL NOT NULL DEFAULT 0.25,
                    updated_at       TEXT NOT NULL
                )
            """)
            log.info("Migration: Created 'retrieval_weights' table")

        # Migration 6: Add spread_activation to retrieval_weights (v1.6.0)
        rw_info = conn.execute("PRAGMA table_info(retrieval_weights)").fetchall()
        rw_columns = {r[1] for r in rw_info}
        if "spread_activation" not in rw_columns:
            conn.execute(
                "ALTER TABLE retrieval_weights "
                "ADD COLUMN spread_activation REAL NOT NULL DEFAULT 0.25"
            )
            log.info("Migration: Added 'spread_activation' column to retrieval_weights")

        # Migration 7: Add accessed_at to hook_session_atoms for temporal Hebbian (v1.7.0)
        # NOTE: SQLite's ALTER TABLE ADD COLUMN with a non-constant DEFAULT expression
        # (datetime('now') is a function) does NOT backfill existing rows — they get NULL.
        # Consumers use atom_timestamps.get(id, 0.0) which defaults to Unix epoch when
        # accessed_at is NULL, causing those pairs to be treated as temporally distant.
        # This is safe — only new synapse creation is affected, never existing strengthening.
        hsa_info = conn.execute("PRAGMA table_info(hook_session_atoms)").fetchall()
        hsa_columns = {r[1] for r in hsa_info}
        if "accessed_at" not in hsa_columns:
            # ALTER TABLE only supports constant defaults — use no default here.
            # New rows get datetime('now') from the CREATE TABLE schema; existing
            # rows get NULL, handled by atom_timestamps.get(id, 0.0) fallback.
            conn.execute(
                "ALTER TABLE hook_session_atoms "
                "ADD COLUMN accessed_at TEXT"
            )
            log.info("Migration: Added 'accessed_at' column to hook_session_atoms")

        # Migration 8: Add post-tool-failure to hook_stats CHECK constraint (v1.8.0)
        # SQLite can't ALTER a CHECK constraint — recreate the table.
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='hook_stats'"
        ).fetchone()
        if row and "post-tool-failure" not in row[0]:
            conn.executescript("""
                ALTER TABLE hook_stats RENAME TO hook_stats_old;

                CREATE TABLE hook_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hook_type TEXT NOT NULL CHECK(hook_type IN (
                        'session-start','prompt-submit','pre-tool','post-tool',
                        'post-response','stop','subagent-stop','pre-compact','post-tool-failure'
                    )),
                    project TEXT,
                    query TEXT,
                    atoms_returned INTEGER NOT NULL DEFAULT 0,
                    atom_ids TEXT,
                    avg_score REAL,
                    max_score REAL,
                    budget_used INTEGER NOT NULL DEFAULT 0,
                    budget_total INTEGER NOT NULL DEFAULT 0,
                    compression_level INTEGER NOT NULL DEFAULT 0,
                    seed_count INTEGER NOT NULL DEFAULT 0,
                    total_activated INTEGER NOT NULL DEFAULT 0,
                    novelty_result TEXT CHECK(novelty_result IN ('pass','fail') OR novelty_result IS NULL),
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                INSERT INTO hook_stats SELECT * FROM hook_stats_old;
                DROP TABLE hook_stats_old;
            """)
            log.info("Migration: Added 'post-tool-failure' to hook_stats CHECK constraint")

        # Migration 9: Add session-end, subagent-start, permission-request to hook_stats (v1.9.0)
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='hook_stats'"
        ).fetchone()
        if row and "session-end" not in row[0]:
            conn.executescript("""
                ALTER TABLE hook_stats RENAME TO hook_stats_old;

                CREATE TABLE hook_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hook_type TEXT NOT NULL CHECK(hook_type IN (
                        'session-start','prompt-submit','pre-tool','post-tool',
                        'post-response','stop','subagent-stop','pre-compact','post-tool-failure',
                        'session-end','subagent-start','permission-request'
                    )),
                    project TEXT,
                    query TEXT,
                    atoms_returned INTEGER NOT NULL DEFAULT 0,
                    atom_ids TEXT,
                    avg_score REAL,
                    max_score REAL,
                    budget_used INTEGER NOT NULL DEFAULT 0,
                    budget_total INTEGER NOT NULL DEFAULT 0,
                    compression_level INTEGER NOT NULL DEFAULT 0,
                    seed_count INTEGER NOT NULL DEFAULT 0,
                    total_activated INTEGER NOT NULL DEFAULT 0,
                    novelty_result TEXT CHECK(novelty_result IN ('pass','fail') OR novelty_result IS NULL),
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                INSERT INTO hook_stats SELECT * FROM hook_stats_old;
                DROP TABLE hook_stats_old;
            """)
            log.info("Migration: Added Wave 13 hook types to hook_stats CHECK constraint")

        # Migration 10: Add task-completed and notification to hook_stats (v1.10.0)
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='hook_stats'"
        ).fetchone()
        if row and "task-completed" not in row[0]:
            conn.executescript("""
                ALTER TABLE hook_stats RENAME TO hook_stats_old;

                CREATE TABLE hook_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hook_type TEXT NOT NULL CHECK(hook_type IN (
                        'session-start','prompt-submit','pre-tool','post-tool',
                        'post-response','stop','subagent-stop','pre-compact','post-tool-failure',
                        'session-end','subagent-start','permission-request',
                        'task-completed','notification'
                    )),
                    project TEXT,
                    query TEXT,
                    atoms_returned INTEGER NOT NULL DEFAULT 0,
                    atom_ids TEXT,
                    avg_score REAL,
                    max_score REAL,
                    budget_used INTEGER NOT NULL DEFAULT 0,
                    budget_total INTEGER NOT NULL DEFAULT 0,
                    compression_level INTEGER NOT NULL DEFAULT 0,
                    seed_count INTEGER NOT NULL DEFAULT 0,
                    total_activated INTEGER NOT NULL DEFAULT 0,
                    novelty_result TEXT CHECK(novelty_result IN ('pass','fail') OR novelty_result IS NULL),
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                INSERT INTO hook_stats SELECT * FROM hook_stats_old;
                DROP TABLE hook_stats_old;
            """)
            log.info("Migration: Added task-completed + notification to hook_stats CHECK constraint")

        # Migration 11: Add tag_expires_at + encoded-with relationship for STC (v1.11.0)
        syn_info = conn.execute("PRAGMA table_info(synapses)").fetchall()
        syn_columns = {col[1] for col in syn_info}
        if "tag_expires_at" not in syn_columns:
            # Need both the new column and the updated CHECK constraint.
            conn.executescript("""
                ALTER TABLE synapses RENAME TO synapses_old;

                CREATE TABLE IF NOT EXISTS synapses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL REFERENCES atoms(id),
                    target_id INTEGER NOT NULL REFERENCES atoms(id),
                    relationship TEXT NOT NULL CHECK(relationship IN (
                        'related-to','caused-by','part-of','contradicts',
                        'supersedes','elaborates','warns-against','encoded-with'
                    )),
                    strength REAL NOT NULL DEFAULT 0.5,
                    bidirectional INTEGER NOT NULL DEFAULT 1,
                    activated_count INTEGER NOT NULL DEFAULT 0,
                    last_activated_at TEXT,
                    tag_expires_at TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(source_id, target_id, relationship)
                );

                INSERT INTO synapses (id, source_id, target_id, relationship,
                    strength, bidirectional, activated_count, last_activated_at,
                    created_at)
                SELECT id, source_id, target_id, relationship,
                    strength, bidirectional, activated_count, last_activated_at,
                    created_at
                FROM synapses_old;

                DROP TABLE synapses_old;

                CREATE INDEX IF NOT EXISTS idx_synapses_source ON synapses(source_id);
                CREATE INDEX IF NOT EXISTS idx_synapses_target ON synapses(target_id);
                CREATE INDEX IF NOT EXISTS idx_synapses_strength ON synapses(strength DESC);
            """)
            log.info(
                "Migration: Recreated synapses table with tag_expires_at "
                "and encoded-with relationship"
            )
        else:
            # tag_expires_at exists but encoded-with may not be in CHECK.
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='synapses'"
            ).fetchone()
            if row and "encoded-with" not in row[0]:
                conn.executescript("""
                    ALTER TABLE synapses RENAME TO synapses_old;

                    CREATE TABLE IF NOT EXISTS synapses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_id INTEGER NOT NULL REFERENCES atoms(id),
                        target_id INTEGER NOT NULL REFERENCES atoms(id),
                        relationship TEXT NOT NULL CHECK(relationship IN (
                            'related-to','caused-by','part-of','contradicts',
                            'supersedes','elaborates','warns-against','encoded-with'
                        )),
                        strength REAL NOT NULL DEFAULT 0.5,
                        bidirectional INTEGER NOT NULL DEFAULT 1,
                        activated_count INTEGER NOT NULL DEFAULT 0,
                        last_activated_at TEXT,
                        tag_expires_at TEXT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        UNIQUE(source_id, target_id, relationship)
                    );

                    INSERT INTO synapses SELECT * FROM synapses_old;
                    DROP TABLE synapses_old;

                    CREATE INDEX IF NOT EXISTS idx_synapses_source ON synapses(source_id);
                    CREATE INDEX IF NOT EXISTS idx_synapses_target ON synapses(target_id);
                    CREATE INDEX IF NOT EXISTS idx_synapses_strength ON synapses(strength DESC);
                """)
                log.info("Migration: Added 'encoded-with' to synapses CHECK constraint")

    def _probe_vec_support(self) -> bool:
        """Check whether sqlite-vec can be loaded in this environment.

        Returns ``True`` if the extension loaded and the vec0 module is
        functional, ``False`` otherwise.  This check is performed once
        during :meth:`initialize` and cached for the lifetime of the
        :class:`Storage` instance.
        """
        conn = sqlite3.connect(str(self._db_path))
        try:
            if not hasattr(conn, "enable_load_extension"):
                log.warning(
                    "sqlite3 module compiled without extension loading support; "
                    "vector search will be unavailable"
                )
                return False

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            log.debug("sqlite-vec extension loaded successfully")
            return True
        except (AttributeError, OSError, sqlite3.OperationalError) as exc:
            log.warning(
                "sqlite-vec extension could not be loaded (%s); "
                "vector search will be unavailable",
                exc,
            )
            return False
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Connection factory
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return the thread-local persistent :class:`sqlite3.Connection`.

        Each thread pool worker gets one long-lived connection so that the
        per-call overhead of opening, pragma-setting, and closing is paid
        only once per thread rather than on every query.  WAL mode allows
        concurrent readers; the write lock serialises writers.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open_connection()
            self._local.conn = conn
            # M-8: register in the global list so close() can shut down all threads.
            with self._connections_lock:
                self._all_connections.append(conn)
        return conn

    def _open_connection(self) -> sqlite3.Connection:
        """Open and configure a new :class:`sqlite3.Connection`.

        Every connection is configured with:

        - WAL journal mode for concurrent reads.
        - Foreign key enforcement.
        - sqlite-vec extension loaded (when available).
        - Row factory set to :class:`sqlite3.Row` for dict-like access.
        """
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row

        # Load sqlite-vec when the runtime supports it.
        if self._vec_available:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

        # Performance and safety pragmas.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA cache_size=-64000")  # ~64 MB
        conn.execute("PRAGMA busy_timeout=5000")

        return conn

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        sql: str,
        params: tuple | dict = (),
    ) -> list[sqlite3.Row]:
        """Execute a read-only query and return all rows.

        Parameters
        ----------
        sql:
            SQL SELECT statement.
        params:
            Bind parameters (positional tuple or named dict).

        Returns
        -------
        list[sqlite3.Row]
            Result rows with dict-like column access.
        """
        return await anyio.to_thread.run_sync(
            lambda: self._execute_sync(sql, params),
        )

    def _execute_sync(
        self,
        sql: str,
        params: tuple | dict = (),
    ) -> list[sqlite3.Row]:
        conn = self._get_connection()
        cursor = conn.execute(sql, params)
        return cursor.fetchall()

    async def execute_write(
        self,
        sql: str,
        params: tuple | dict = (),
    ) -> int:
        """Execute a write query under the write lock.

        Parameters
        ----------
        sql:
            SQL INSERT / UPDATE / DELETE statement.
        params:
            Bind parameters.

        Returns
        -------
        int
            The ``lastrowid`` of the executed statement.
        """
        return await anyio.to_thread.run_sync(
            lambda: self._execute_write_sync(sql, params),
        )

    def _execute_write_sync(
        self,
        sql: str,
        params: tuple | dict = (),
    ) -> int:
        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(sql, params)
                conn.commit()
                return cursor.lastrowid or 0
            except Exception:
                conn.rollback()
                raise

    async def execute_write_returning(
        self,
        sql: str,
        params: tuple | dict = (),
    ) -> list[sqlite3.Row]:
        """Execute a write query with a RETURNING clause under the write lock.

        Use this instead of :meth:`execute_write` when the SQL statement
        includes a ``RETURNING`` clause and the caller needs the returned rows
        (e.g. ``UPDATE ... RETURNING *``).

        Parameters
        ----------
        sql:
            SQL INSERT / UPDATE / DELETE statement with a RETURNING clause.
        params:
            Bind parameters.

        Returns
        -------
        list[sqlite3.Row]
            Rows produced by the RETURNING clause.
        """
        return await anyio.to_thread.run_sync(
            lambda: self._execute_write_returning_sync(sql, params),
        )

    def _execute_write_returning_sync(
        self,
        sql: str,
        params: tuple | dict = (),
    ) -> list[sqlite3.Row]:
        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                conn.commit()
                return rows
            except Exception:
                conn.rollback()
                raise

    async def execute_many(
        self,
        sql: str,
        params_list: list[tuple | dict],
    ) -> None:
        """Execute a statement for each set of parameters under the write lock.

        Parameters
        ----------
        sql:
            SQL statement to repeat.
        params_list:
            A list of parameter tuples / dicts.
        """
        await anyio.to_thread.run_sync(
            lambda: self._execute_many_sync(sql, params_list),
        )

    def _execute_many_sync(
        self,
        sql: str,
        params_list: list[tuple | dict],
    ) -> None:
        with self._write_lock:
            conn = self._get_connection()
            try:
                conn.executemany(sql, params_list)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    async def execute_script(self, sql: str) -> None:
        """Execute a multi-statement SQL script under the write lock.

        Parameters
        ----------
        sql:
            One or more SQL statements separated by semicolons.
        """
        await anyio.to_thread.run_sync(
            lambda: self._execute_script_sync(sql),
        )

    def _execute_script_sync(self, sql: str) -> None:
        with self._write_lock:
            conn = self._get_connection()
            try:
                conn.executescript(sql)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    async def execute_transaction(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        """Execute a callback inside a single ``BEGIN IMMEDIATE`` transaction.

        The write lock is held for the entire duration, and the callback
        receives a raw :class:`sqlite3.Connection` that is already inside
        a ``BEGIN IMMEDIATE`` transaction.  The framework handles commit
        on success, rollback on exception, and connection close in all
        cases.

        ``BEGIN IMMEDIATE`` acquires the SQLite write-lock upfront so
        that reads within the callback see a consistent snapshot.

        Parameters
        ----------
        fn:
            A synchronous callable that receives a
            :class:`sqlite3.Connection` and returns a value of type *T*.
            All SQL should be executed through the provided connection.

        Returns
        -------
        T
            Whatever *fn* returns.
        """
        return await anyio.to_thread.run_sync(
            lambda: self._execute_transaction_sync(fn),
        )

    def _execute_transaction_sync(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        with self._write_lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE")
                result = fn(conn)
                conn.commit()
                return result
            except Exception:
                conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Advisory lock helpers
    # ------------------------------------------------------------------

    @staticmethod
    def try_acquire_lock(conn: sqlite3.Connection, name: str, holder: str | None = None) -> bool:
        """Attempt to acquire a named advisory lock inside a transaction.

        Stale locks older than 10 minutes are automatically cleaned up
        before the acquisition attempt.

        Parameters
        ----------
        conn:
            A connection already inside an active transaction (e.g.
            from :meth:`execute_transaction`).
        name:
            The lock name (primary key in the ``locks`` table).
        holder:
            An identifier for the holder.  Defaults to a random UUID.

        Returns
        -------
        bool
            ``True`` if the lock was acquired, ``False`` if another
            holder already owns it.
        """
        if holder is None:
            holder = uuid.uuid4().hex

        # Clean up stale locks (>10 minutes old).
        conn.execute(
            "DELETE FROM locks WHERE name = ? AND acquired_at < datetime('now', '-10 minutes')",
            (name,),
        )

        try:
            conn.execute(
                "INSERT INTO locks (name, holder) VALUES (?, ?)",
                (name, holder),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    @staticmethod
    def release_lock(conn: sqlite3.Connection, name: str, holder: str | None = None) -> None:
        """Release a named advisory lock inside a transaction.

        Parameters
        ----------
        conn:
            A connection already inside an active transaction.
        name:
            The lock name.
        holder:
            If provided, only releases the lock if it is held by this
            holder.  If ``None``, releases unconditionally.
        """
        if holder is not None:
            conn.execute(
                "DELETE FROM locks WHERE name = ? AND holder = ?",
                (name, holder),
            )
        else:
            conn.execute(
                "DELETE FROM locks WHERE name = ?",
                (name,),
            )

    # ------------------------------------------------------------------
    # Maintenance operations
    # ------------------------------------------------------------------

    async def backup(self) -> Path:
        """Create a timestamped backup of the database.

        Old backups beyond the configured retention count are deleted.

        Returns
        -------
        Path
            Filesystem path of the newly created backup file.
        """
        return await anyio.to_thread.run_sync(self._backup_sync)

    def _backup_sync(self) -> Path:
        """Synchronous backup implementation."""
        if not self._db_path.exists():
            log.debug("No database file to back up yet")
            return self._db_path

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = self._backup_dir / f"memories_{timestamp}.db"

        # Use SQLite's online backup API for a consistent snapshot.
        src = sqlite3.connect(str(self._db_path))
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
            log.info("Backup created: %s", backup_path)
        finally:
            dst.close()
            src.close()

        # Prune old backups.
        self._prune_backups()
        return backup_path

    def _prune_backups(self) -> None:
        """Delete old backups, keeping only the most recent ``backup_count``."""
        backups = sorted(
            self._backup_dir.glob("memories_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in backups[self._backup_count :]:
            try:
                old.unlink()
                log.debug("Pruned old backup: %s", old.name)
            except OSError as exc:
                log.warning("Failed to remove old backup %s: %s", old.name, exc)

    async def optimize(self) -> None:
        """Run periodic maintenance tasks.

        - ``ANALYZE`` to update the query planner statistics.
        - FTS5 ``optimize`` to merge internal b-tree segments.
        - ``integrity_check`` to verify database health.
        """
        await anyio.to_thread.run_sync(self._optimize_sync)

    def _optimize_sync(self) -> None:
        with self._write_lock:
            conn = self._get_connection()
            try:
                conn.execute("ANALYZE")
                log.debug("ANALYZE complete")

                conn.execute(
                    "INSERT INTO atoms_fts(atoms_fts) VALUES ('optimize')"
                )
                log.debug("FTS5 optimize complete")

                rows = conn.execute("PRAGMA integrity_check(1)").fetchall()
                status = rows[0][0] if rows else "unknown"
                if status != "ok":
                    log.warning("Integrity check returned: %s", status)
                else:
                    log.debug("Integrity check: ok")

                conn.commit()
            except Exception:
                conn.rollback()
                raise

    async def vacuum(self) -> None:
        """Run ``VACUUM`` to rebuild the database and reclaim space.

        Note: VACUUM requires exclusive access and cannot run inside a
        transaction, so it is handled specially.
        """
        await anyio.to_thread.run_sync(self._vacuum_sync)

    def _vacuum_sync(self) -> None:
        with self._write_lock:
            conn = self._get_connection()
            # VACUUM cannot run inside a transaction.
            conn.execute("VACUUM")
            log.info("VACUUM complete")

    # ------------------------------------------------------------------
    # Database metadata
    # ------------------------------------------------------------------

    async def get_db_size_mb(self) -> float:
        """Return the database file size in megabytes.

        Returns
        -------
        float
            File size in MB, or ``0.0`` if the file does not exist.
        """
        return await anyio.to_thread.run_sync(self._get_db_size_mb_sync)

    def _get_db_size_mb_sync(self) -> float:
        if not self._db_path.exists():
            return 0.0
        size_bytes = self._db_path.stat().st_size
        # Include WAL file size if present.
        wal_path = self._db_path.with_suffix(".db-wal")
        if wal_path.exists():
            size_bytes += wal_path.stat().st_size
        return round(size_bytes / (1024 * 1024), 2)

    async def table_counts(self) -> dict[str, int]:
        """Return row counts for all core tables.

        Uses a single UNION ALL query to fetch all counts in one round-trip
        instead of issuing one query per table (6 → 1 round-trip).

        Returns
        -------
        dict[str, int]
            Mapping of table name to row count.
        """
        rows = await self.execute(
            """
            SELECT 'atoms'             AS tbl, COUNT(*) AS cnt FROM atoms
            UNION ALL
            SELECT 'synapses',                  COUNT(*)        FROM synapses
            UNION ALL
            SELECT 'regions',                   COUNT(*)        FROM regions
            UNION ALL
            SELECT 'sessions',                  COUNT(*)        FROM sessions
            UNION ALL
            SELECT 'consolidation_log',         COUNT(*)        FROM consolidation_log
            UNION ALL
            SELECT 'embedding_cache',           COUNT(*)        FROM embedding_cache
            """
        )
        return {row["tbl"]: row["cnt"] for row in rows}

    # ------------------------------------------------------------------
    # Retrieval weight auto-tuning helpers
    # ------------------------------------------------------------------

    async def load_retrieval_weights(self) -> dict[str, float] | None:
        """Return stored weight overrides or None if the table is empty."""
        rows = await self.execute(
            "SELECT confidence, importance, frequency, recency, spread_activation "
            "FROM retrieval_weights WHERE id = 1"
        )
        if not rows:
            return None
        row = rows[0]
        return {
            "confidence":        float(row["confidence"]),
            "importance":        float(row["importance"]),
            "frequency":         float(row["frequency"]),
            "recency":           float(row["recency"]),
            "spread_activation": float(row["spread_activation"]),
        }

    async def save_retrieval_weights(self, weights: dict[str, float]) -> None:
        """Upsert the singleton weights row."""
        now = datetime.now(tz=timezone.utc).isoformat()
        await self.execute_write(
            """
            INSERT INTO retrieval_weights
                (id, confidence, importance, frequency, recency, spread_activation, updated_at)
            VALUES (1, :confidence, :importance, :frequency, :recency, :spread_activation, :updated_at)
            ON CONFLICT(id) DO UPDATE SET
                confidence        = excluded.confidence,
                importance        = excluded.importance,
                frequency         = excluded.frequency,
                recency           = excluded.recency,
                spread_activation = excluded.spread_activation,
                updated_at        = excluded.updated_at
            """,
            {**weights, "updated_at": now},
        )

    async def get_feedback_atom_ids_since(
        self, cutoff: str
    ) -> tuple[list[int], list[int]]:
        """Return (good_ids, bad_ids) from atom_feedback since cutoff."""
        rows = await self.execute(
            """
            SELECT atom_id, signal
            FROM atom_feedback
            WHERE created_at >= ?
            """,
            (cutoff,),
        )
        good_ids: list[int] = []
        bad_ids: list[int] = []
        for row in rows:
            if row["signal"] == "good":
                good_ids.append(int(row["atom_id"]))
            else:
                bad_ids.append(int(row["atom_id"]))
        # Deduplicate while preserving order.
        good_ids = list(dict.fromkeys(good_ids))
        bad_ids  = list(dict.fromkeys(bad_ids))
        return good_ids, bad_ids

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all persistent connections opened across all threads.

        M-8: Iterates the global ``_all_connections`` registry and closes every
        connection regardless of which thread opened it, preventing connection
        leaks on shutdown when thread-pool workers have exited without explicit
        cleanup.
        """
        with self._connections_lock:
            conns = list(self._all_connections)
            self._all_connections.clear()

        for conn in conns:
            try:
                conn.close()
            except Exception as exc:
                log.warning("Failed to close connection: %s", exc)

        self._local.conn = None
        log.debug("Storage closed (%d connections released)", len(conns))

    async def __aenter__(self) -> Storage:
        await self.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
