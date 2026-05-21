"""Tests for the SQLite atom store.

Uses a deterministic hashing embedder so tests don't depend on a real
sentence-transformers model (avoids the ~90MB download). A live
MiniLMEmbedder smoke test runs only when LATTICE_LLM_SMOKE=1, mirroring
the LocalLLMProposer test.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from lattice.atoms import (
    Atom,
    AtomStore,
    AtomType,
    Embedder,
    RecallResult,
    SQLiteAtomStore,
)


class HashEmbedder:
    """Deterministic embedder: stable cosine similarity from text hash."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = [w.lower() for w in text.split() if w.strip()]
            for w in words:
                h = hashlib.sha1(w.encode()).digest()
                for j in range(self._dim):
                    out[i, j] += (h[j % len(h)] / 255.0) - 0.5
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out

    @property
    def dim(self) -> int:
        return self._dim


_: Embedder = HashEmbedder()


# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> SQLiteAtomStore:
    return SQLiteAtomStore(tmp_path / "atoms.db", embedder=HashEmbedder())


def test_protocol_satisfaction(store: SQLiteAtomStore):
    assert isinstance(store, AtomStore)


def test_add_and_count(store: SQLiteAtomStore):
    a = store.add("Redis SCAN is O(N) over the full keyspace.")
    assert isinstance(a, Atom)
    assert a.id is not None
    assert store.count() == 1


def test_add_rejects_empty(store: SQLiteAtomStore):
    with pytest.raises(ValueError):
        store.add("")


def test_recall_returns_most_similar_first(store: SQLiteAtomStore):
    store.add("Stripe charge processing for payments and refunds.")
    store.add("Postgres autovacuum tuning notes.")
    store.add("Redis SCAN vs KEYS performance.")

    results = store.recall("payment processing with stripe", k=3)
    assert len(results) == 3
    assert "stripe" in results[0].atom.content.lower()
    assert results[0].score >= results[1].score >= results[2].score


def test_recall_empty_query(store: SQLiteAtomStore):
    store.add("anything")
    assert store.recall("") == []


def test_recall_empty_store(store: SQLiteAtomStore):
    assert store.recall("anything") == []


def test_recall_filters_by_region(store: SQLiteAtomStore):
    store.add("relevant", region="project:foo")
    store.add("relevant", region="project:bar")
    results = store.recall("relevant", region="project:foo")
    assert len(results) == 1
    assert results[0].atom.region == "project:foo"


def test_recall_filters_by_type(store: SQLiteAtomStore):
    store.add("regular fact", type=AtomType.FACT)
    store.add("watch out", type=AtomType.ANTIPATTERN)
    results = store.recall("watch out", types=(AtomType.ANTIPATTERN,))
    assert len(results) == 1
    assert results[0].atom.type == AtomType.ANTIPATTERN


def test_recall_updates_access(store: SQLiteAtomStore):
    store.add("only atom in store")
    [r] = store.recall("anything", k=1)
    assert r.atom.access_count == 0  # snapshot is pre-increment
    [r2] = store.recall("anything", k=1)
    assert r2.atom.access_count == 1
    assert r2.atom.last_accessed_at is not None


def test_recall_result_shape(store: SQLiteAtomStore):
    store.add("hello world")
    [r] = store.recall("hello", k=1)
    assert isinstance(r, RecallResult)
    assert 0.0 <= r.score <= 1.5  # normalized vectors; cosine in [-1, 1].


def test_persistence_across_reopen(tmp_path: Path):
    db = tmp_path / "atoms.db"
    s1 = SQLiteAtomStore(db, embedder=HashEmbedder())
    s1.add("persisted content")
    s1.close()

    s2 = SQLiteAtomStore(db, embedder=HashEmbedder())
    assert s2.count() == 1
    [r] = s2.recall("persisted", k=1)
    assert r.atom.content == "persisted content"
    s2.close()


def test_tags_and_importance(store: SQLiteAtomStore):
    a = store.add(
        "tagged atom",
        tags=("redis", "performance"),
        importance=0.9,
        type=AtomType.SKILL,
    )
    assert a.tags == ("redis", "performance")
    assert a.importance == 0.9

    [r] = store.recall("tagged", k=1)
    assert r.atom.tags == ("redis", "performance")
    assert r.atom.importance == 0.9
    assert r.atom.type == AtomType.SKILL


# ---------------------------------------------------------------------------
# Live MiniLMEmbedder smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("LATTICE_LLM_SMOKE") != "1", reason="LATTICE_LLM_SMOKE not set"
)
def test_minilm_embedder_smoke(tmp_path: Path):
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    from lattice.atoms import MiniLMEmbedder

    store = SQLiteAtomStore(tmp_path / "atoms.db", embedder=MiniLMEmbedder())
    store.add("Stripe charge processing for payments.")
    store.add("Postgres autovacuum tuning notes.")
    store.add("Redis SCAN vs KEYS performance.")

    [top, *_] = store.recall("how do I take a credit card payment", k=1)
    assert "stripe" in top.atom.content.lower() or "charge" in top.atom.content.lower()
