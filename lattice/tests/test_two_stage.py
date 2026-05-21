"""Structural tests for TwoStageProposer.

Live model-driven tests are gated on LATTICE_LLM_SMOKE=1 because
loading transformers + a model is slow. The cheap tests below verify
the verb-classification regex and the per-verb schema table — both
of which can fail silently and would be hard to debug from a live run.
"""

from __future__ import annotations

import os

import pytest

from lattice.propose.two_stage import (
    _VERB_CHOICES,
    _VERB_PARSE_RE,
    _VERB_SCHEMAS,
    TwoStageProposer,
)


class TestVerbClassification:
    def test_every_verb_in_schema_table(self):
        for verb in _VERB_CHOICES:
            assert verb in _VERB_SCHEMAS, f"missing schema for {verb}"

    def test_regex_recognizes_bare_word(self):
        for verb in _VERB_CHOICES:
            m = _VERB_PARSE_RE.search(verb)
            assert m and m.group(1) == verb

    def test_regex_extracts_from_prose(self):
        cases = [
            ("The next verb is AddImport.", "AddImport"),
            ("AddParameter\n", "AddParameter"),
            ("I think we should call MarkDone now.", "MarkDone"),
            ("Research! That's what we need.", "Research"),
        ]
        for text, expected in cases:
            m = _VERB_PARSE_RE.search(text)
            assert m and m.group(1) == expected, f"failed for {text!r}"

    def test_regex_rejects_non_verbs(self):
        for text in ("nothing here", "do something", "??", ""):
            assert _VERB_PARSE_RE.search(text) is None


class TestSchemaShape:
    def test_every_schema_is_a_string(self):
        for verb, schema in _VERB_SCHEMAS.items():
            assert isinstance(schema, str)
            assert verb in schema

    def test_paths_use_dot_py_placeholder(self):
        # Schemas that reference file paths should show the .py form so
        # the executor uses real-style paths.
        for verb in ("AddImport", "RenameSymbol", "AddField", "AddParameter", "AddTest"):
            assert ".py" in _VERB_SCHEMAS[verb]


class TestConstruction:
    def test_construct_without_loading(self):
        p = TwoStageProposer()
        assert p.planner_model
        assert p.executor_model
        # Models are lazy-loaded; constructing doesn't pull weights.
        assert p._planner is None and p._executor is None


_SMOKE = os.environ.get("LATTICE_LLM_SMOKE") == "1"


@pytest.mark.skipif(not _SMOKE, reason="LATTICE_LLM_SMOKE not set")
def test_two_stage_emits_valid_action_live():
    """End-to-end live: pick verb + fill slots produces a valid Action."""
    pytest.importorskip("transformers", reason="transformers not installed")

    from lattice.propose.base import ObservationContext
    from lattice.sense import Symbol, SymbolKind

    obs = ObservationContext(
        task="Add an import of os to src/main.py.",
        symbols=(
            Symbol(file="src/main.py", name="main", kind=SymbolKind.FUNCTION, line=1),
        ),
        hints=("WORKSPACE FILES (use these EXACT paths): src/main.py",),
    )
    proposer = TwoStageProposer()
    out = proposer.propose(obs)
    assert len(out) == 1
    # Don't pin a specific verb (small models drift); just check it parsed.
    assert hasattr(out[0], "verb")
