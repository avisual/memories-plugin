"""Tests for the DISTILL apprentice (Organ 8 v0)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from lattice.atoms import SQLiteAtomStore, write_trace
from lattice.atoms.evolve import boost_recurrent_traces
from lattice.distill import ApprenticeProposer
from lattice.propose import ObservationContext


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


def test_apprentice_no_store_returns_empty():
    p = ApprenticeProposer(atom_store=None)
    out = p.propose(ObservationContext(task="add an import"))
    assert out == []


def test_apprentice_empty_store_returns_empty(store: SQLiteAtomStore):
    p = ApprenticeProposer(atom_store=store)
    out = p.propose(ObservationContext(task="add an import"))
    assert out == []


def test_apprentice_no_boost_returns_empty(store: SQLiteAtomStore):
    """Traces below min_importance don't fire the apprentice — until
    EVOLVE boosts them, the apprentice stays out of the proposer chain.
    """
    write_trace(
        store=store,
        task="Add an import of os to src/a.py",
        actions=["AddImport"],
        files_touched=["src/a.py"],
    )
    p = ApprenticeProposer(atom_store=store, min_similarity=0.0)
    out = p.propose(ObservationContext(task="Add an import of os to src/a.py"))
    # Importance is 0.55 (default), below the 0.7 min — no candidate.
    assert out == []


def test_apprentice_fires_after_boost(store: SQLiteAtomStore):
    """When EVOLVE has boosted a recurring pattern, the apprentice
    surfaces its action as a candidate for similar new tasks.
    """
    # Three identical-shape traces — EVOLVE will boost them.
    write_trace(
        store=store,
        task="Add an import of os to src/a.py",
        actions=["AddImport"],
        files_touched=["src/a.py"],
    )
    write_trace(
        store=store,
        task="Add an import of json to src/b.py",
        actions=["AddImport"],
        files_touched=["src/b.py"],
    )
    write_trace(
        store=store,
        task="Add an import of sys to src/c.py",
        actions=["AddImport"],
        files_touched=["src/c.py"],
    )
    boost_recurrent_traces(store, min_recurrence=3)

    # Apprentice with permissive similarity so the hash embedder's
    # weakness doesn't block the test.
    p = ApprenticeProposer(
        atom_store=store,
        min_importance=0.7,
        min_similarity=0.0,
    )
    out = p.propose(ObservationContext(task="Add an import of os to src/a.py"))
    assert len(out) == 1
    assert out[0].verb == "AddImport"


def test_apprentice_below_similarity_returns_empty(store: SQLiteAtomStore):
    write_trace(
        store=store,
        task="rename foo to bar in src/x.py",
        actions=["RenameSymbol"],
        files_touched=["src/x.py"],
    )
    boost_recurrent_traces(store, min_recurrence=1)
    p = ApprenticeProposer(
        atom_store=store,
        min_importance=0.6,
        min_similarity=0.99,  # impossibly strict
    )
    out = p.propose(ObservationContext(task="something completely unrelated"))
    assert out == []
