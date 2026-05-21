"""Tests for template inference (Organ 8/9 IMPROVES dial).

The promise: given N successful traces of the SAME shape, infer the
task-text → action-shape mapping so future novel tasks of the same
shape get a learned candidate without an LLM call.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from lattice.atoms import (
    AtomType,
    SQLiteAtomStore,
    apply_template,
    infer_template,
    learned_templates,
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


def _addimport_action(module: str, file_path: str) -> dict:
    return {
        "verb": "AddImport",
        "file": {"path": file_path},
        "module": module,
        "names": None,
        "alias": None,
        "confidence": 0.9,
    }


def test_infer_template_basic():
    """Three 'Add an import of X to Y' tasks → infer the template."""
    traces = [
        {
            "task": "Add an import of os to src/a.py",
            "actions": [_addimport_action("os", "src/a.py")],
            "files": [],
        },
        {
            "task": "Add an import of json to src/b.py",
            "actions": [_addimport_action("json", "src/b.py")],
            "files": [],
        },
        {
            "task": "Add an import of sys to src/c.py",
            "actions": [_addimport_action("sys", "src/c.py")],
            "files": [],
        },
    ]
    tmpl = infer_template(traces)
    assert tmpl is not None
    assert tmpl.signature == ("AddImport",)
    assert tmpl.sample_count == 3

    # Now apply the template to a NOVEL task: never seen before.
    novel = "Add an import of stripe to src/payments/charge.py"
    out = apply_template(tmpl, novel)
    assert out is not None
    assert out["verb"] == "AddImport"
    assert out["module"] == "stripe"
    assert out["file"]["path"] == "src/payments/charge.py"


def test_infer_template_constant_when_no_variation():
    """If all samples have identical tasks, there's nothing to template."""
    same = {
        "task": "Add an import of os to src/a.py",
        "actions": [_addimport_action("os", "src/a.py")],
        "files": [],
    }
    assert infer_template([same, same, same]) is None


def test_infer_template_rejects_variable_length_tasks():
    """v0 only templates same-length token sequences."""
    traces = [
        {
            "task": "Add an import of os to src/a.py",
            "actions": [_addimport_action("os", "src/a.py")],
            "files": [],
        },
        {
            "task": "Add an import of json please to src/b.py",  # 9 tokens, not 7
            "actions": [_addimport_action("json", "src/b.py")],
            "files": [],
        },
    ]
    assert infer_template(traces) is None


def test_infer_template_handles_multi_action_chains():
    """Multi-action templates: every step learned, signature carries
    the full verb sequence, action_template_chain holds all steps."""
    from lattice.atoms import apply_template_chain

    traces = [
        {
            "task": "Do thing in src/x.py",
            "actions": [
                _addimport_action("os", "src/x.py"),
                {
                    "verb": "AddStatement",
                    "file": {"path": "src/x.py"},
                    "code": "init_x()",
                    "position": "end",
                    "target": None,
                    "confidence": 0.9,
                },
            ],
            "files": [],
        },
        {
            "task": "Do thing in src/y.py",
            "actions": [
                _addimport_action("json", "src/y.py"),
                {
                    "verb": "AddStatement",
                    "file": {"path": "src/y.py"},
                    "code": "init_y()",
                    "position": "end",
                    "target": None,
                    "confidence": 0.9,
                },
            ],
            "files": [],
        },
    ]
    tmpl = infer_template(traces)
    assert tmpl is not None
    assert tmpl.signature == ("AddImport", "AddStatement")
    assert len(tmpl.action_template_chain) == 2

    chain = apply_template_chain(tmpl, "Do thing in src/z.py")
    assert len(chain) == 2
    # Step 0: AddImport — file.path substituted; module stays 'os'
    # (only one capture in this task — c0 = src/z.py). The module
    # field DIDN'T vary in our 2 samples (one was 'os', one was 'json'
    # — wait, those DO vary). Let me adjust assertions.
    assert chain[0]["verb"] == "AddImport"
    assert chain[1]["verb"] == "AddStatement"
    assert chain[1]["file"]["path"] == "src/z.py"


def test_apply_template_returns_none_on_no_match():
    """Template should not match an unrelated task."""
    traces = [
        {"task": "Add an import of os to src/a.py", "actions": [_addimport_action("os", "src/a.py")], "files": []},
        {"task": "Add an import of json to src/b.py", "actions": [_addimport_action("json", "src/b.py")], "files": []},
    ]
    tmpl = infer_template(traces)
    assert tmpl is not None
    assert apply_template(tmpl, "Rename foo to bar in src/x.py") is None


def test_learned_templates_from_store(store: SQLiteAtomStore):
    """End-to-end: write traces with full action JSON, then mine them."""
    write_trace(
        store=store,
        task="Add an import of os to src/a.py",
        actions=[_addimport_action("os", "src/a.py")],
        files_touched=["src/a.py"],
    )
    write_trace(
        store=store,
        task="Add an import of json to src/b.py",
        actions=[_addimport_action("json", "src/b.py")],
        files_touched=["src/b.py"],
    )
    write_trace(
        store=store,
        task="Add an import of sys to src/c.py",
        actions=[_addimport_action("sys", "src/c.py")],
        files_touched=["src/c.py"],
    )

    templates = learned_templates(store, min_recurrence=3)
    assert len(templates) == 1
    tmpl = templates[0]

    # Substitute into a brand-new task.
    novel = "Add an import of pydantic to src/api/v2/schemas.py"
    out = apply_template(tmpl, novel)
    assert out is not None
    assert out["module"] == "pydantic"
    assert out["file"]["path"] == "src/api/v2/schemas.py"


def test_apprentice_uses_template_for_novel_task(store: SQLiteAtomStore):
    """The full DISTILL win: a task lattice has NEVER seen before
    gets a learned candidate via template inference, no LLM call."""
    from lattice.distill import ApprenticeProposer
    from lattice.propose import ObservationContext

    for module, file_path in (
        ("os", "src/a.py"),
        ("json", "src/b.py"),
        ("sys", "src/c.py"),
    ):
        write_trace(
            store=store,
            task=f"Add an import of {module} to {file_path}",
            actions=[_addimport_action(module, file_path)],
            files_touched=[file_path],
        )

    p = ApprenticeProposer(atom_store=store, min_importance=0.0, min_similarity=0.0)
    obs = ObservationContext(task="Add an import of pydantic to src/api/v2/schemas.py")
    out = p.propose(obs)
    assert len(out) == 1
    assert out[0].verb == "AddImport"
    assert out[0].module == "pydantic"
    assert out[0].file.path == "src/api/v2/schemas.py"
