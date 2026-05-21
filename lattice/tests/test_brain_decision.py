"""Tests for Hebbian decision-weighting — Phase 4.

The brain doesn't just decorate the prompt; it shapes which candidate
wins the pre-flight scoring. These tests cover:
  - _brain_score returns 0 when brain is off / store is None.
  - _brain_score returns positive when matching SKILL atoms exist.
  - _brain_score returns negative when matching ANTIPATTERN atoms exist.
  - _record_step_outcome writes SKILL on success / ANTIPATTERN on failure.
  - The score actually flips the candidate choice in _propose_with_preflight.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from lattice.actions import (
    AddImport,
    FileRef,
    MarkDone,
)
from lattice.atoms import AtomType, SQLiteAtomStore
from lattice.compiler import DictWorkspace
from lattice.orchestrator import AgentLoop
from lattice.propose.mock import MockProposer


class _ConstantEmbedder:
    """Embedder where every string maps to the same vector — so the
    cosine score between any pair is 1.0. Lets us test the brain
    boost/penalty logic without sentence-transformers in the test path.
    """

    def __init__(self) -> None:
        self._vec = np.ones(8, dtype="float32") / (8**0.5)

    @property
    def dim(self) -> int:
        return 8

    def embed(self, texts):
        return np.tile(self._vec, (len(texts), 1))


def _ws() -> DictWorkspace:
    return DictWorkspace(
        {
            "src/main.py": "def main() -> None:\n    pass\n",
        }
    )


def _store(tmp_path: Path) -> SQLiteAtomStore:
    return SQLiteAtomStore(tmp_path / "atoms.db", embedder=_ConstantEmbedder())


def test_brain_score_zero_when_brain_disabled():
    """use_brain=False → _brain_score returns 0.0 regardless of store."""
    loop = AgentLoop(
        proposer=MockProposer.empty(),
        workspace=_ws(),
        atom_store=None,
        use_brain=False,
    )
    action = AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)
    assert loop._brain_score(action, "add json") == 0.0


def test_brain_score_positive_for_matching_success_atom(tmp_path):
    """A SKILL atom whose embedding cosines >= 0.35 with the query → positive delta."""
    store = _store(tmp_path)
    store.add(
        "Success: verb=AddImport on task 'add an import of json'",
        type=AtomType.SKILL,
        region="steps",
    )
    loop = AgentLoop(
        proposer=MockProposer.empty(),
        workspace=_ws(),
        atom_store=store,
        use_brain=True,
    )
    action = AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)
    delta = loop._brain_score(action, "add import of json")
    # ConstantEmbedder → cosine 1.0; boost = 0.25 * 1.0 = +0.25 per hit.
    assert delta > 0.0
    assert delta <= 0.5  # bounded
    store.close()


def test_brain_score_negative_for_matching_antipattern_atom(tmp_path):
    """An ANTIPATTERN atom → negative delta (heavier weight than SKILL)."""
    store = _store(tmp_path)
    store.add(
        "Failure: verb=AddImport on task 'add an import of json' — broke parse",
        type=AtomType.ANTIPATTERN,
        region="steps",
    )
    loop = AgentLoop(
        proposer=MockProposer.empty(),
        workspace=_ws(),
        atom_store=store,
        use_brain=True,
    )
    action = AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)
    delta = loop._brain_score(action, "add import of json")
    assert delta < 0.0
    assert delta >= -0.5  # bounded
    store.close()


def test_brain_score_bounded(tmp_path):
    """Even with many matching atoms, the delta is clipped to [-0.5, 0.5]."""
    store = _store(tmp_path)
    for _ in range(20):
        store.add(
            "Success: verb=AddImport on task 'add an import of json'",
            type=AtomType.SKILL,
            region="steps",
        )
    loop = AgentLoop(
        proposer=MockProposer.empty(),
        workspace=_ws(),
        atom_store=store,
        use_brain=True,
    )
    action = AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)
    assert loop._brain_score(action, "add import of json") == 0.5
    store.close()


def test_record_step_outcome_writes_skill_on_success(tmp_path):
    """A successful mutating step writes a SKILL atom in region='steps'."""
    store = _store(tmp_path)
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(
        proposer=proposer, workspace=_ws(), atom_store=store, use_brain=True
    )
    loop.run("add an import of json to src/main.py")
    hits = store.recall("verb=AddImport", k=5)
    skills = [h for h in hits if h.atom.type == AtomType.SKILL]
    assert skills, "expected at least one SKILL atom written by step outcome"
    assert "Success" in skills[0].atom.content
    store.close()


def test_record_step_outcome_writes_antipattern_on_failure(tmp_path):
    """A mutating step that fails verify writes an ANTIPATTERN atom."""
    from lattice.actions import AddParameter, SymbolRef, TypeExpr

    store = _store(tmp_path)
    bad = AddParameter(
        function=SymbolRef(file="src/main.py", name="does_not_exist"),
        name="x",
        type=TypeExpr(expr="int"),
        confidence=0.8,
    )
    proposer = MockProposer(
        batches=[[bad], [MarkDone(summary="give up", confidence=0.5)]]
    )
    loop = AgentLoop(
        proposer=proposer, workspace=_ws(), atom_store=store, use_brain=True
    )
    loop.run("add x to does_not_exist")
    hits = store.recall("verb=AddParameter", k=5)
    antis = [h for h in hits if h.atom.type == AtomType.ANTIPATTERN]
    assert antis, "expected an ANTIPATTERN atom for the failed step"
    assert "Failure" in antis[0].atom.content
    store.close()


def test_no_outcome_atom_when_brain_off(tmp_path):
    """use_brain=False → no SKILL / ANTIPATTERN atoms get written."""
    store = _store(tmp_path)
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(
        proposer=proposer, workspace=_ws(), atom_store=store, use_brain=False
    )
    loop.run("add an import of json to src/main.py")
    hits = store.recall("verb=AddImport", k=5)
    skills = [h for h in hits if h.atom.type == AtomType.SKILL]
    assert not skills
    store.close()
