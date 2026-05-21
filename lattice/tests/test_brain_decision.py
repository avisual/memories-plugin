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
from lattice.orchestrator import AgentLoop, StepRecord
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
    """use_brain=False → _brain_score returns (0.0, []) regardless of store."""
    loop = AgentLoop(
        proposer=MockProposer.empty(),
        workspace=_ws(),
        atom_store=None,
        use_brain=False,
    )
    action = AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)
    delta, ids = loop._brain_score(action, "add json")
    assert delta == 0.0 and ids == []


def test_brain_score_positive_for_matching_success_atom(tmp_path):
    """A SKILL atom in region='steps' whose embedding cosines >= 0.35
    with the query → positive delta and IDs returned for reinforcement."""
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
    delta, ids = loop._brain_score(action, "add import of json")
    # ConstantEmbedder → cosine 1.0; boost = 0.25 * 1.0 = +0.25 per hit.
    assert delta > 0.0
    assert delta <= 0.5  # bounded
    assert ids, "matching SKILL atom should be returned as a contributor"
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
    delta, ids = loop._brain_score(action, "add import of json")
    assert delta < 0.0
    assert delta >= -0.5  # bounded
    assert ids
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
    delta, _ids = loop._brain_score(action, "add import of json")
    assert delta == 0.5
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


def test_reinforce_bumps_importance_for_recalled_atoms_on_success(tmp_path):
    """Atoms recalled into the obs of a successful step get importance up.

    Wiring: store has an atom that will be recalled (constant embedder →
    cosine 1.0 with anything). Run a single successful mutating step.
    Read the same atom back — its importance should have moved up by
    +0.05.
    """
    store = _store(tmp_path)
    seeded = store.add(
        "Skill: useful for adding imports cleanly",
        type=AtomType.SKILL,
        region="general",
        importance=0.5,
    )
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(
        proposer=proposer, workspace=_ws(), atom_store=store, use_brain=True
    )
    loop.run("add an import of json")
    # Fetch the atom's current importance by recalling.
    hits = store.recall("useful for adding imports", k=3)
    same_atom = next((h for h in hits if h.atom.id == seeded.id), None)
    assert same_atom is not None, "expected to find the seeded atom"
    # +0.05 from successful step participation. (Allow tiny floor for
    # double-recall on cycle 2.)
    assert same_atom.atom.importance >= 0.5 + 0.04
    store.close()


def test_reinforce_does_not_decay_observation_atoms_on_failure(tmp_path):
    """Observation-recall atoms are NOT decayed on a failed step.

    This is the post-fix semantics: a failed step's
    observation-recall atoms often contain context (or even the FIX
    REQUIRED hint via prior failures) that helps the NEXT step
    succeed. Penalising them buries the very signals that enable
    self-correction. Only atoms in region='steps' that brain_score
    used for the failing candidate get decayed.
    """
    from lattice.actions import AddParameter, SymbolRef, TypeExpr

    store = _store(tmp_path)
    seeded = store.add(
        "skill: misleading guidance",
        type=AtomType.SKILL,
        region="general",  # NOT region='steps' — observation only
        importance=0.6,
    )
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
    loop.run("attempt the bad action")
    hits = store.recall("misleading guidance", k=3)
    same_atom = next((h for h in hits if h.atom.id == seeded.id), None)
    assert same_atom is not None
    # Observation-recall atom unchanged from 0.6.
    assert abs(same_atom.atom.importance - 0.6) < 0.001
    store.close()


def test_reinforce_directly_decays_brain_score_atoms_on_failure(tmp_path):
    """Direct unit test on _reinforce_recalled: a failure record with
    _winning_brain_ids set DOES decay those atoms by -0.03.

    Avoids the full agent loop (where compile-error candidates have
    no brain_score contributors anyway) so we test the reinforce
    semantics in isolation.
    """
    from lattice.actions import AddImport

    store = _store(tmp_path)
    a = store.add("contributor 1", type=AtomType.SKILL, importance=0.6)
    b = store.add("contributor 2", type=AtomType.ANTIPATTERN, importance=0.6)

    loop = AgentLoop(
        proposer=MockProposer.empty(),
        workspace=_ws(),
        atom_store=store,
        use_brain=True,
    )
    # Inject as-if a previous brain_score said "these atoms moved the
    # decision toward this candidate", and the step then failed verify.
    loop._winning_brain_ids = [a.id, b.id]
    loop._cycle_recalled_ids = []
    bad_record = StepRecord(
        step=1,
        action=AddImport(
            file=FileRef(path="src/main.py"), module="json", confidence=0.9
        ),
        kind="error",
        error="some verify failure",
    )
    loop._reinforce_recalled(bad_record)

    # Both contributors decayed by 0.03.
    hits = store.recall("contributor", k=5)
    for h in hits:
        if h.atom.id in (a.id, b.id):
            assert h.atom.importance <= 0.6 - 0.02
    store.close()


def test_reinforce_skipped_when_brain_off(tmp_path):
    """use_brain=False → no importance updates after a step."""
    store = _store(tmp_path)
    seeded = store.add(
        "Skill: useful for adding imports",
        type=AtomType.SKILL,
        importance=0.5,
    )
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(
        proposer=proposer, workspace=_ws(), atom_store=store, use_brain=False
    )
    loop.run("add an import of json")
    hits = store.recall("useful for adding imports", k=3)
    same_atom = next((h for h in hits if h.atom.id == seeded.id), None)
    assert same_atom is not None
    # Importance unchanged from initial 0.5.
    assert abs(same_atom.atom.importance - 0.5) < 0.001
    store.close()


def test_brain_flips_candidate_choice_end_to_end(tmp_path):
    """End-to-end proof: brain ON vs OFF picks a DIFFERENT candidate.

    Setup: pre-flight gets two equally-confident candidates (AddImport
    of 'json' vs AddImport of 'stripe'). The atom store is pre-loaded
    with:
      - a SKILL atom whose content embeds near the 'stripe' candidate
      - an ANTIPATTERN atom whose content embeds near the 'json' candidate
    Brain ON: stripe wins (+SKILL boost + ANTIPATTERN penalty makes the
    delta strongly favor stripe).
    Brain OFF: ties broken by score-on-ties (first in scored list); the
    json candidate (listed first) wins.

    Uses HashEmbedder so embeddings are deterministic and keyword-tied.
    """
    import hashlib

    import numpy as np

    from lattice.atoms.embedder import Embedder

    class _KeywordEmbedder:
        """Per-word hash → vector; strings sharing words have similar vectors."""

        def __init__(self, dim: int = 32) -> None:
            self._dim = dim

        @property
        def dim(self) -> int:
            return self._dim

        def embed(self, texts):
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

    _: Embedder = _KeywordEmbedder()

    store_path = tmp_path / "brain.db"
    store = SQLiteAtomStore(store_path, embedder=_KeywordEmbedder())
    # Seed: positive history on stripe; negative history on json.
    store.add(
        "verb=AddImport file=src/main.py module=stripe task add stripe import",
        type=AtomType.SKILL,
        region="steps",
        importance=0.6,
    )
    store.add(
        "verb=AddImport file=src/main.py module=json task add json import broken",
        type=AtomType.ANTIPATTERN,
        region="steps",
        importance=0.6,
    )

    # Both candidates have identical confidence — the brain_delta is the
    # ONLY thing that should differentiate them.
    candidate_json = AddImport(
        file=FileRef(path="src/main.py"), module="json", confidence=0.5
    )
    candidate_stripe = AddImport(
        file=FileRef(path="src/main.py"), module="stripe", confidence=0.5
    )
    proposer = MockProposer(
        batches=[
            [candidate_json, candidate_stripe],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(
        proposer=proposer, workspace=_ws(), atom_store=store, use_brain=True
    )
    trace = loop.run("add stripe import")
    assert trace.ok
    final = trace.final_files["src/main.py"]
    # Brain steered toward stripe (SKILL match) over json (ANTIPATTERN).
    assert "import stripe" in final, (
        "expected brain decision-weighting to pick the SKILL-matched "
        "candidate (stripe) over the ANTIPATTERN-matched one (json)"
    )
    assert "import json" not in final
    store.close()


def test_brain_off_picks_first_candidate_same_setup(tmp_path):
    """With brain OFF on the same setup, the first candidate (json) wins.

    Proves the choice in the prior test came from the brain, not from
    some other tie-breaker. With brain off, the json candidate (listed
    first) wins because there's no decision-weighting and confidence is
    identical."""
    import hashlib

    import numpy as np

    class _KeywordEmbedder:
        def __init__(self, dim: int = 32) -> None:
            self._dim = dim

        @property
        def dim(self) -> int:
            return self._dim

        def embed(self, texts):
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

    store_path = tmp_path / "brain.db"
    store = SQLiteAtomStore(store_path, embedder=_KeywordEmbedder())
    store.add(
        "verb=AddImport file=src/main.py module=stripe",
        type=AtomType.SKILL,
        region="steps",
    )
    store.add(
        "verb=AddImport file=src/main.py module=json broken",
        type=AtomType.ANTIPATTERN,
        region="steps",
    )

    candidate_json = AddImport(
        file=FileRef(path="src/main.py"), module="json", confidence=0.5
    )
    candidate_stripe = AddImport(
        file=FileRef(path="src/main.py"), module="stripe", confidence=0.5
    )
    proposer = MockProposer(
        batches=[
            [candidate_json, candidate_stripe],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(
        proposer=proposer, workspace=_ws(), atom_store=store, use_brain=False
    )
    trace = loop.run("add an import")
    assert trace.ok
    final = trace.final_files["src/main.py"]
    # With brain off, the brain delta is 0 for both — json (first in
    # list, stable sort) wins.
    assert "import json" in final
    store.close()


def test_winning_brain_contributors_get_reinforced(tmp_path):
    """Atoms recalled by brain_score for the WINNING candidate get the
    same Hebbian update as observation-recall atoms.

    Without this wiring, brain_score atoms only get access_count++
    (which doesn't move importance) — so their score contribution
    would be invisible to future cycles. Correctness: the atom that
    earned the +/- delta IS a participant in the decision and should
    move by outcome like any other.
    """
    store = _store(tmp_path)
    # Seed in region='steps' (only region brain_score consults).
    contributor = store.add(
        "verb=AddImport file=src/main.py module=json :: Success",
        type=AtomType.SKILL,
        region="steps",
        importance=0.5,
    )
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(
        proposer=proposer, workspace=_ws(), atom_store=store, use_brain=True
    )
    loop.run("add import of json")
    # The contributor was the only steps-region atom; it scored the
    # winning candidate so it should have been reinforced by +0.05.
    hits = store.recall(
        "verb=AddImport file=src/main.py module=json",
        k=5,
        region="steps",
    )
    same = next((h for h in hits if h.atom.id == contributor.id), None)
    assert same is not None
    assert same.atom.importance >= 0.55 - 0.001
    store.close()


def test_region_filter_excludes_non_steps_atoms(tmp_path):
    """_brain_score only consults atoms in region='steps'.

    Generic seeded atoms in other regions don't have the (verb, slots,
    task) signature shape that the query expects and would produce
    spurious low-cosine boosts. The region filter prevents that.
    """
    store = _store(tmp_path)
    # Atom that would normally match (constant embedder → cosine 1.0)
    # but is in the wrong region.
    store.add(
        "Success: verb=AddImport on task add import of json",
        type=AtomType.SKILL,
        region="seeded-knowledge",
    )
    loop = AgentLoop(
        proposer=MockProposer.empty(),
        workspace=_ws(),
        atom_store=store,
        use_brain=True,
    )
    action = AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)
    delta, ids = loop._brain_score(action, "add import of json")
    # No matching atom in 'steps' region → delta is 0.
    assert delta == 0.0
    assert ids == []
    store.close()


def test_outcome_atom_embeds_near_brain_score_query(tmp_path):
    """The atom written by _record_step_outcome is recallable by the
    same query that _brain_score will issue next cycle.

    This is the correctness fix that closed the active_task mismatch:
    both sides MUST use _signature_with_task or atoms written for step
    N are in a different part of embedding space than the query for
    step N+1's brain_score, and the brain misses its own data.
    """
    store = _store(tmp_path)
    # Run a successful step so the loop writes an outcome atom.
    proposer = MockProposer(
        batches=[
            [AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)],
            [MarkDone(summary="done", confidence=0.9)],
        ]
    )
    loop = AgentLoop(
        proposer=proposer, workspace=_ws(), atom_store=store, use_brain=True
    )
    loop.run("add an import of json")
    # Now query the brain in the same shape brain_score would.
    action = AddImport(file=FileRef(path="src/main.py"), module="json", confidence=0.9)
    delta, ids = loop._brain_score(action, "add an import of json")
    # The just-written SKILL atom should boost this candidate.
    assert delta > 0.0
    assert ids, "expected to find the atom we just wrote"
    store.close()


def test_pending_added_atom_ids_seeded_into_next_cycle_recall(tmp_path):
    """Atoms added by a non-mutating step (Research) get seeded into
    the NEXT cycle's _cycle_recalled_ids so the post-step reinforcement
    credits them for the outcome of the next edit.

    Without this wiring, research atoms wrote but never received any
    outcome feedback — their importance stayed static even when the
    research clearly enabled the next step's success.
    """
    from lattice.actions import AddImport

    store = _store(tmp_path)
    loop = AgentLoop(
        proposer=MockProposer.empty(),
        workspace=_ws(),
        atom_store=store,
        use_brain=True,
    )
    # Stash a 'pending' ID as if the prior cycle was a Research call.
    research_atom = store.add(
        "Fetched docs about adding imports",
        type=AtomType.FACT,
        region="research",
        importance=0.6,
    )
    loop._pending_added_atom_ids = [research_atom.id]

    # Build observation; should consume the pending ID into _cycle_recalled_ids.
    loop._build_observation("add an import", [])
    assert research_atom.id in loop._cycle_recalled_ids
    # And the pending list got reset so the same atom doesn't double-credit.
    assert loop._pending_added_atom_ids == []

    # Simulate a successful step — _reinforce_recalled should bump the
    # research atom's importance even though it wasn't matched by recall
    # directly this cycle.
    good_record = StepRecord(
        step=1,
        action=AddImport(
            file=FileRef(path="src/main.py"), module="json", confidence=0.9
        ),
        kind="edit",
        diff="+ import json\n",
    )
    loop._reinforce_recalled(good_record)
    hits = store.recall("Fetched docs", k=3)
    same = next((h for h in hits if h.atom.id == research_atom.id), None)
    assert same is not None
    assert same.atom.importance >= 0.6 + 0.04
    store.close()


def test_winning_brain_ids_cleared_at_cycle_start(tmp_path):
    """_winning_brain_ids must be cleared at the start of EVERY cycle.

    Without this, atoms from a previous cycle's winning candidate get
    reinforced for an outcome they didn't influence (proposer raised
    or returned no candidates this cycle). Sets the IDs manually then
    runs a cycle where the proposer returns no candidates; checks
    they're cleared rather than carried forward.
    """
    store = _store(tmp_path)
    proposer = MockProposer(batches=[])  # empty — terminates immediately
    loop = AgentLoop(
        proposer=proposer, workspace=_ws(), atom_store=store, use_brain=True
    )
    loop._winning_brain_ids = [999, 1000]  # stale state
    loop.run("any task")
    # After run, the loop should have cleared stale state before
    # any reinforcement attempt could have used it.
    assert loop._winning_brain_ids == []
    store.close()


def test_store_reinforce_clips_to_zero_and_ninety_five(tmp_path):
    """The reinforce API clips importance to [0.0, 0.95]."""
    store = _store(tmp_path)
    a = store.add("low", importance=0.04)
    b = store.add("high", importance=0.94)
    # Try to push below 0.
    store.reinforce([a.id], delta=-0.10)
    # Try to push above 0.95.
    store.reinforce([b.id], delta=+0.10)
    new_low = next(h for h in store.recall("low", k=2) if h.atom.id == a.id).atom.importance
    new_high = next(h for h in store.recall("high", k=2) if h.atom.id == b.id).atom.importance
    assert new_low == 0.0
    assert new_high == 0.95
    store.close()
