"""Tests for the EVOLVE organ (macro discovery from traces)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from lattice.atoms import (
    AtomType,
    SQLiteAtomStore,
    discover,
    write_trace,
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


def test_write_trace_persists_a_structured_atom(store: SQLiteAtomStore):
    atom_id = write_trace(
        store=store,
        task="Add an import of json to src/main.py",
        actions=["AddImport"],
        files_touched=["src/main.py"],
    )
    assert atom_id is not None
    [r] = store.recall("json import main", k=1)
    assert r.atom.type == AtomType.EXPERIENCE
    assert r.atom.region == "traces"
    assert "AddImport" in r.atom.content
    assert "JSON:" in r.atom.content  # marker for the embedded payload


def test_write_trace_rejects_empty(store: SQLiteAtomStore):
    assert write_trace(store=store, task="", actions=["X"], files_touched=[]) is None
    assert write_trace(store=store, task="t", actions=[], files_touched=[]) is None


def test_discover_empty_store_returns_empty(store: SQLiteAtomStore):
    report = discover(store)
    assert report.candidates == ()
    assert report.total_traces == 0


def test_discover_clusters_by_action_sequence(store: SQLiteAtomStore):
    # Three runs of the SAME action sequence on slightly different tasks.
    for task in (
        "add import of os to src/a.py",
        "add import of sys to src/b.py",
        "add import of json to src/c.py",
    ):
        write_trace(
            store=store,
            task=task,
            actions=["AddImport"],
            files_touched=["src/x.py"],
        )
    # Two runs of a DIFFERENT sequence (won't reach min_recurrence=3).
    for task in (
        "rename foo to bar in src/d.py",
        "rename baz to qux in src/e.py",
    ):
        write_trace(
            store=store,
            task=task,
            actions=["RenameSymbol"],
            files_touched=["src/x.py"],
        )

    report = discover(store, min_recurrence=3)
    assert report.total_traces == 5
    assert len(report.candidates) == 1
    cand = report.candidates[0]
    assert cand.action_sequence == ("AddImport",)
    assert cand.sample_count == 3
    assert len(cand.sample_tasks) == 3


def test_discover_min_recurrence_filter(store: SQLiteAtomStore):
    write_trace(store=store, task="t1", actions=["A", "B"], files_touched=[])
    write_trace(store=store, task="t2", actions=["A", "B"], files_touched=[])
    # min_recurrence=3 → no candidates (only 2 occurrences of [A, B])
    assert discover(store, min_recurrence=3).candidates == ()
    # min_recurrence=2 → one candidate
    report = discover(store, min_recurrence=2)
    assert len(report.candidates) == 1
    assert report.candidates[0].action_sequence == ("A", "B")
    assert report.candidates[0].sample_count == 2


def test_discover_sorts_by_recurrence(store: SQLiteAtomStore):
    # Sequence X appears 5 times, sequence Y appears 3 times.
    for _ in range(5):
        write_trace(store=store, task="t", actions=["X"], files_touched=[])
    for _ in range(3):
        write_trace(store=store, task="t", actions=["Y"], files_touched=[])

    report = discover(store, min_recurrence=2)
    assert [c.action_sequence for c in report.candidates] == [("X",), ("Y",)]
    assert report.candidates[0].sample_count == 5
    assert report.candidates[1].sample_count == 3


def test_record_experience_also_writes_trace(store: SQLiteAtomStore):
    """The agent CLI's record_experience hook should write a parallel trace
    atom so EVOLVE can mine it without changes to callers.
    """
    from lattice.atoms import record_experience

    record_experience(
        store=store,
        task="add an import of os to src/main.py",
        actions_summary=["AddImport"],
        files_touched=["src/main.py"],
    )
    report = discover(store, min_recurrence=1)
    assert report.total_traces == 1
    assert report.candidates[0].action_sequence == ("AddImport",)


def test_boost_recurrent_traces_raises_importance(store: SQLiteAtomStore):
    """Phase 2: when a sequence recurs >= min_recurrence times, boost the
    importance of its trace atoms so they surface higher in recall."""
    from lattice.atoms import boost_recurrent_traces

    # Three same-sequence traces.
    for task in ("a", "b", "c"):
        write_trace(store=store, task=task, actions=["AddImport"], files_touched=[])
    # Two different-sequence traces (below threshold, should NOT boost).
    for task in ("d", "e"):
        write_trace(store=store, task=task, actions=["RenameSymbol"], files_touched=[])

    boosted = boost_recurrent_traces(store, min_recurrence=3)
    assert boosted == 3

    # The three AddImport traces now carry the boosted importance.
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT content, importance FROM atoms WHERE region = 'traces'"
    ).fetchall()
    importances = {tuple(_extract_actions(c)): imp for c, imp in rows}
    assert importances[("AddImport",)] >= 0.85  # 0.55 + 3*0.1 = 0.85
    assert importances[("RenameSymbol",)] < 0.85  # below threshold; not boosted


def _extract_actions(content: str) -> list[str]:
    """Test helper: pull verb names from a trace atom's JSON payload.

    Tolerant of both the legacy verb-string form and the new structured
    dict form. The tuple of strings is used as a dict key, so this must
    return hashables.
    """
    import json as _json

    marker = "\nJSON: "
    idx = content.rfind(marker)
    if idx < 0:
        return []
    raw = _json.loads(content[idx + len(marker):]).get("actions", [])
    out: list[str] = []
    for a in raw:
        if isinstance(a, dict):
            out.append(str(a.get("verb", "?")))
        else:
            out.append(str(a))
    return out


def test_boost_below_threshold_no_op(store: SQLiteAtomStore):
    """Below min_recurrence, no atom gets boosted."""
    from lattice.atoms import boost_recurrent_traces

    for task in ("a", "b"):
        write_trace(store=store, task=task, actions=["AddImport"], files_touched=[])
    assert boost_recurrent_traces(store, min_recurrence=3) == 0
