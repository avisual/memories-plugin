"""Tests for the seed atom pack + Hebbian feedback writer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from lattice.atoms import (
    SEED_ATOMS,
    AtomType,
    SQLiteAtomStore,
    record_antipattern,
    record_experience,
    seed_store,
    summarize_trace_for_experience,
)
from lattice.atoms.atom import Atom


class HashEmbedder:
    """Deterministic test embedder (mirrors test_atoms.py)."""

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


# ---------------------------------------------------------------------------
# Seed pack
# ---------------------------------------------------------------------------


def test_seed_pack_is_non_empty():
    assert len(SEED_ATOMS) >= 20
    types = {a.type for a in SEED_ATOMS}
    assert AtomType.ANTIPATTERN in types
    assert AtomType.PREFERENCE in types
    assert AtomType.SKILL in types
    assert AtomType.FACT in types


def test_seed_pack_contents_are_unique():
    contents = [a.content for a in SEED_ATOMS]
    assert len(contents) == len(set(contents))


def test_seed_store_loads_everything(store: SQLiteAtomStore):
    added = seed_store(store)
    assert added == len(SEED_ATOMS)
    assert store.count() == len(SEED_ATOMS)


def test_seed_then_recall_payments_returns_stripe(store: SQLiteAtomStore):
    seed_store(store)
    results = store.recall("how do I take a credit card payment", k=3)
    assert results
    # Stripe-related atom should surface.
    contents = [r.atom.content.lower() for r in results]
    assert any("stripe" in c for c in contents)


# ---------------------------------------------------------------------------
# Feedback writers
# ---------------------------------------------------------------------------


def test_record_experience_writes_atom(store: SQLiteAtomStore):
    atom_id = record_experience(
        store=store,
        task="add stripe import to charge.py",
        actions_summary=["AddImport", "AddParameter"],
        files_touched=["src/payments/charge.py"],
    )
    assert atom_id is not None

    [r] = store.recall("add stripe", k=1)
    assert r.atom.type == AtomType.EXPERIENCE
    assert "stripe" in r.atom.content.lower()
    assert "AddImport" in r.atom.content


def test_record_antipattern_writes_atom(store: SQLiteAtomStore):
    record_antipattern(
        store=store,
        task="rename frobulate to frobby",
        failure_summary="model keeps emitting the same RenameSymbol",
    )
    [r] = store.recall("rename frobulate", k=1)
    assert r.atom.type == AtomType.ANTIPATTERN
    assert "Reconsider" in r.atom.content


def test_summarize_trace_extracts_verbs_and_files():
    class FakeStep:
        def __init__(self, kind, verb, diff):
            self.kind = kind
            self.verb = verb
            self.diff = diff

    class FakeTrace:
        steps = (
            FakeStep("edit", "AddImport", "+import x\n"),
            FakeStep("edit", "AddImport", ""),  # no-op, should be skipped
            FakeStep("error", "AddImport", ""),  # error, skipped
            FakeStep("edit", "AddParameter", "+ y\n"),
        )
        final_files = {"a.py": "x = 1", "b.py": "y = 2"}

    actions, files = summarize_trace_for_experience(FakeTrace())
    assert actions == ["AddImport", "AddParameter"]
    assert set(files) == {"a.py", "b.py"}


def test_summarize_trace_handles_multi_subtask_shape():
    class FakeStep:
        def __init__(self, kind, verb, diff):
            self.kind = kind
            self.verb = verb
            self.diff = diff

    class FakeTrace:
        steps = (FakeStep("edit", "AddImport", "+import x\n"),)

    class FakeReport:
        traces = (FakeTrace(), FakeTrace())
        final_files = {"a.py": "x = 1"}

    actions, files = summarize_trace_for_experience(FakeReport())
    assert actions == ["AddImport", "AddImport"]
    assert files == ["a.py"]
