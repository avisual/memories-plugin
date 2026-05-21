"""SQLite-backed atom store with embedding recall.

Single-file, no external services. Embeddings stored as float32 blobs
in the atoms table; recall does a plain Python cosine over normalized
vectors (fine for thousands of atoms; swap in sqlite-vec or hnswlib
once we cross ~100k).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

from lattice.atoms.atom import Atom, AtomType, _now
from lattice.atoms.embedder import Embedder, MiniLMEmbedder


class RecallResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    atom: Atom
    score: float


@runtime_checkable
class AtomStore(Protocol):
    def add(
        self,
        content: str,
        *,
        type: AtomType = AtomType.FACT,
        region: str = "",
        tags: tuple[str, ...] = (),
        importance: float = 0.5,
        confidence: float = 1.0,
    ) -> Atom: ...

    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        region: str | None = None,
        types: tuple[AtomType, ...] | None = None,
    ) -> list[RecallResult]: ...

    def reinforce(self, atom_ids: list[int], delta: float) -> int: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS atoms (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL,
    type            TEXT NOT NULL,
    region          TEXT NOT NULL DEFAULT '',
    tags            TEXT NOT NULL DEFAULT '[]',
    importance      REAL NOT NULL DEFAULT 0.5,
    confidence      REAL NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    last_accessed_at TEXT,
    embedding       BLOB NOT NULL,
    embed_dim       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS atoms_type_idx ON atoms(type);
CREATE INDEX IF NOT EXISTS atoms_region_idx ON atoms(region);
"""


class SQLiteAtomStore:
    """Atom store backed by a single SQLite file + a pluggable Embedder.

    Concurrency: one writer per process; SQLite WAL handles concurrent
    readers fine. Not thread-safe within a single process (use a lock
    if you fan out).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._embedder: Embedder = embedder or MiniLMEmbedder()
        self._dim = self._embedder.dim

    # ----- writes -----

    def add(
        self,
        content: str,
        *,
        type: AtomType = AtomType.FACT,
        region: str = "",
        tags: tuple[str, ...] = (),
        importance: float = 0.5,
        confidence: float = 1.0,
    ) -> Atom:
        if not content.strip():
            raise ValueError("content must be non-empty")
        vec = self._embedder.embed([content])[0]
        created = _now()
        cur = self._conn.execute(
            """
            INSERT INTO atoms(content, type, region, tags, importance, confidence,
                              created_at, embedding, embed_dim)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content,
                type.value,
                region,
                json.dumps(list(tags)),
                importance,
                confidence,
                created,
                vec.astype(np.float32).tobytes(),
                self._dim,
            ),
        )
        atom_id = cur.lastrowid
        return Atom(
            id=atom_id,
            content=content,
            type=type,
            region=region,
            tags=tags,
            importance=importance,
            confidence=confidence,
            created_at=created,
        )

    # ----- reads -----

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()
        return int(row[0])

    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        region: str | None = None,
        types: tuple[AtomType, ...] | None = None,
    ) -> list[RecallResult]:
        if not query.strip() or k <= 0:
            return []
        q_vec = self._embedder.embed([query])[0]
        clauses, params = [], []
        if region is not None:
            clauses.append("region = ?")
            params.append(region)
        if types is not None and len(types) > 0:
            placeholders = ",".join("?" * len(types))
            clauses.append(f"type IN ({placeholders})")
            params.extend(t.value for t in types)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT id, content, type, region, tags, importance, confidence,
                   access_count, created_at, last_accessed_at, embedding
            FROM atoms {where}
        """
        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []

        # Cosine over normalized vectors == dot product. Importance adds
        # a small additive bonus so EVOLVE's boost_recurrent_traces (which
        # raises importance on patterns the system has seen N+ times)
        # actually changes recall order. 0.10 is small enough that strong
        # semantic mismatches still lose, large enough that two atoms with
        # similar cosines get ordered by 'lattice has learned this matters.'
        matrix = np.vstack(
            [np.frombuffer(r[10], dtype=np.float32) for r in rows]
        )
        cosines = matrix @ q_vec
        importances = np.array([r[5] for r in rows], dtype=np.float32)
        scores = cosines + 0.10 * importances
        top_idx = np.argsort(-scores)[:k]

        # Update access stats for the recalled atoms (best-effort; failures
        # don't break the read).
        recalled_ids: list[int] = []
        out: list[RecallResult] = []
        for idx in top_idx:
            r = rows[idx]
            atom = Atom(
                id=r[0],
                content=r[1],
                type=AtomType(r[2]),
                region=r[3],
                tags=tuple(json.loads(r[4])),
                importance=r[5],
                confidence=r[6],
                access_count=r[7],
                created_at=r[8],
                last_accessed_at=r[9],
            )
            out.append(RecallResult(atom=atom, score=float(scores[idx])))
            recalled_ids.append(r[0])

        if recalled_ids:
            self._conn.executemany(
                "UPDATE atoms SET access_count = access_count + 1, "
                "last_accessed_at = ? WHERE id = ?",
                [(_now(), aid) for aid in recalled_ids],
            )
        return out

    def reinforce(self, atom_ids: list[int], delta: float) -> int:
        """Hebbian update — bump importance for atoms that participated
        in a successful step (or decay them when they participated in a
        failed step). Clipped to [0.0, 0.95] so seed atoms with
        importance>=0.95 stay dominant.

        Returns the count of atoms actually updated.
        """
        if not atom_ids or delta == 0.0:
            return 0
        rows = self._conn.execute(
            f"SELECT id, importance FROM atoms WHERE id IN ({','.join('?' * len(atom_ids))})",
            atom_ids,
        ).fetchall()
        updates = [
            (max(0.0, min(0.95, importance + delta)), aid)
            for aid, importance in rows
        ]
        if not updates:
            return 0
        self._conn.executemany(
            "UPDATE atoms SET importance = ? WHERE id = ?",
            updates,
        )
        return len(updates)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteAtomStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
