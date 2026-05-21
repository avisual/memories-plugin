"""Tests for the import/export and stats helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lattice.atoms import (
    AtomType,
    SQLiteAtomStore,
    export_atoms,
    import_atoms,
    seed_store,
    stats,
)


class HashEmbedder:
    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def embed(self, texts):
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for w in text.lower().split():
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


@pytest.fixture
def store(tmp_path: Path) -> SQLiteAtomStore:
    return SQLiteAtomStore(tmp_path / "atoms.db", embedder=HashEmbedder())


def test_import_atoms_from_json(store: SQLiteAtomStore, tmp_path: Path):
    src = tmp_path / "atoms.json"
    src.write_text(
        json.dumps(
            [
                {
                    "content": "Use httpx for async HTTP requests in Python.",
                    "type": "skill",
                    "region": "python:http",
                    "tags": ["http", "httpx"],
                    "importance": 0.8,
                },
                {
                    "content": "Don't store secrets in the repo.",
                    "type": "antipattern",
                    "region": "security",
                    "importance": 0.95,
                },
            ]
        )
    )
    added = import_atoms(store, src)
    assert added == 2
    [r] = store.recall("httpx async http", k=1)
    assert r.atom.type == AtomType.SKILL
    assert r.atom.region == "python:http"


def test_import_requires_content_field(store: SQLiteAtomStore, tmp_path: Path):
    src = tmp_path / "atoms.json"
    src.write_text(json.dumps([{"type": "fact"}]))
    with pytest.raises(ValueError, match="content"):
        import_atoms(store, src)


def test_import_rejects_unknown_type(store: SQLiteAtomStore, tmp_path: Path):
    src = tmp_path / "atoms.json"
    src.write_text(json.dumps([{"content": "x", "type": "nonsense"}]))
    with pytest.raises(ValueError, match="unknown atom type"):
        import_atoms(store, src)


def test_round_trip_export_import(store: SQLiteAtomStore, tmp_path: Path):
    seed_store(store)
    seeded = store.count()
    out = tmp_path / "brain.json"
    n = export_atoms(store, out)
    assert n == seeded

    # Re-import into a fresh store with the same embedder; counts match.
    store2 = SQLiteAtomStore(tmp_path / "atoms2.db", embedder=HashEmbedder())
    re_added = import_atoms(store2, out)
    assert re_added == seeded
    assert store2.count() == seeded


def test_stats_groups_by_region_and_type(store: SQLiteAtomStore):
    seed_store(store)
    layout = stats(store)
    # Seed pack covers >=4 regions, each potentially with multiple types.
    assert len(layout) >= 4
    # python:style is a known seeded region.
    assert "python:style" in layout
    assert sum(sum(t.values()) for t in layout.values()) == store.count()
