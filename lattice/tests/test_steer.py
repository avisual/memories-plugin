"""Tests for STEER (Organ 4) v0 — importance-aware priming."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from lattice.atoms import AtomType, SQLiteAtomStore, write_trace
from lattice.atoms.evolve import boost_recurrent_traces
from lattice.steer import render_priming_block, top_priming_atoms


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


def test_top_priming_atoms_empty_store(store: SQLiteAtomStore):
    assert top_priming_atoms(store, "anything", k=2) == []


def test_top_priming_atoms_skips_below_importance(store: SQLiteAtomStore):
    # Write three traces; only ONE will be boosted above threshold.
    write_trace(store=store, task="add an import of os to src/a.py", actions=["AddImport"], files_touched=[])
    write_trace(store=store, task="rename foo to bar in src/b.py", actions=["RenameSymbol"], files_touched=[])
    write_trace(store=store, task="add an import of sys to src/c.py", actions=["AddImport"], files_touched=[])

    # Boost the AddImport sequence above the default min_importance=0.6.
    boost_recurrent_traces(store, min_recurrence=2)

    primed = top_priming_atoms(store, "add an import to a file", k=4)
    # Only AddImport traces clear the importance bar (RenameSymbol had only 1 trace).
    assert primed, "expected at least one primed atom"
    for r in primed:
        assert r.atom.importance >= 0.6


def test_top_priming_atoms_respects_k(store: SQLiteAtomStore):
    for task in ("a", "b", "c", "d", "e"):
        write_trace(store=store, task=task, actions=["AddImport"], files_touched=[])
    boost_recurrent_traces(store, min_recurrence=2)
    primed = top_priming_atoms(store, "any task", k=2)
    assert len(primed) <= 2


def test_render_priming_block_format(store: SQLiteAtomStore):
    for task in ("first", "second", "third"):
        write_trace(store=store, task=task, actions=["AddImport"], files_touched=[])
    boost_recurrent_traces(store, min_recurrence=2)
    primed = top_priming_atoms(store, "any", k=2)
    block = render_priming_block(primed)
    assert block.startswith("PRIMING")
    assert "importance" in block
    # JSON trailer stripped from the rendered view.
    assert "JSON:" not in block


def test_render_priming_block_empty_string_when_no_atoms():
    assert render_priming_block([]) == ""


def test_none_store_returns_empty():
    assert top_priming_atoms(None, "anything", k=2) == []
