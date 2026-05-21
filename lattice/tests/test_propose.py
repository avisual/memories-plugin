"""Tests for the Proposer abstraction.

Mock proposer tests are always on. LocalLLMProposer tests are gated:
- JSON-extraction logic is tested directly (no model needed).
- A model-loading + generation smoke test runs only when transformers
  is importable AND the LATTICE_LLM_SMOKE env var is set, since loading
  a model is slow and we don't want it in the default test loop.
"""

from __future__ import annotations

import os

import pytest

from lattice.actions import AddImport, Branch, FileRef
from lattice.propose import (
    MockProposer,
    ObservationContext,
    Proposer,
    ProposerError,
)
from lattice.propose.local import _coerce_loose, _extract_json


# ---------------------------------------------------------------------------
# ObservationContext
# ---------------------------------------------------------------------------


def test_observation_context_minimal():
    obs = ObservationContext(task="add a log line")
    assert obs.task == "add a log line"
    assert obs.symbols == ()
    assert obs.hints == ()


def test_observation_context_rejects_empty_task():
    with pytest.raises(Exception):
        ObservationContext(task="")


# ---------------------------------------------------------------------------
# MockProposer
# ---------------------------------------------------------------------------


class TestMockProposer:
    def test_satisfies_proposer_protocol(self):
        assert isinstance(MockProposer.empty(), Proposer)

    def test_empty_returns_nothing(self):
        p = MockProposer.empty()
        assert p.propose(ObservationContext(task="t")) == []

    def test_returns_pre_canned_batches(self):
        actions_a = [AddImport(file=FileRef(path="a.py"), module="json", confidence=0.9)]
        actions_b = [Branch(rationale="alternative", confidence=0.5)]
        p = MockProposer(batches=[actions_a, actions_b])
        obs = ObservationContext(task="t")

        first = p.propose(obs)
        assert len(first) == 1 and first[0].verb == "AddImport"
        second = p.propose(obs)
        assert len(second) == 1 and second[0].verb == "Branch"
        third = p.propose(obs)
        assert third == []

    def test_n_limits_batch_size(self):
        actions = [
            AddImport(file=FileRef(path="a.py"), module="json", confidence=0.9),
            AddImport(file=FileRef(path="a.py"), module="os", confidence=0.9),
            AddImport(file=FileRef(path="a.py"), module="sys", confidence=0.9),
        ]
        p = MockProposer(batches=[actions])
        out = p.propose(ObservationContext(task="t"), n=2)
        assert len(out) == 2

    def test_call_counter(self):
        p = MockProposer.empty()
        p.propose(ObservationContext(task="t"))
        p.propose(ObservationContext(task="t"))
        assert p.calls == 2


# ---------------------------------------------------------------------------
# JSON extraction (used by LocalLLMProposer)
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_bare_object(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_with_surrounding_text(self):
        out = _extract_json('Sure! Here is the JSON: {"verb":"Branch","rationale":"x","confidence":0.5}.')
        assert out == {"verb": "Branch", "rationale": "x", "confidence": 0.5}

    def test_with_markdown_fence(self):
        text = '```json\n{"verb":"Branch","rationale":"x","confidence":0.5}\n```'
        out = _extract_json(text)
        assert out == {"verb": "Branch", "rationale": "x", "confidence": 0.5}

    def test_nested_object(self):
        text = '{"verb":"AddImport","file":{"path":"a.py"},"module":"json","confidence":0.5}'
        out = _extract_json(text)
        assert out is not None
        assert out["file"] == {"path": "a.py"}

    def test_no_object(self):
        assert _extract_json("no json here, sorry") is None

    def test_malformed_object(self):
        assert _extract_json("{this is not, json}") is None


# ---------------------------------------------------------------------------
# Loose-coercion layer (forgives small-model JSON-shape mistakes)
# ---------------------------------------------------------------------------


class TestCoerceLoose:
    def test_file_as_string_becomes_dict(self):
        out = _coerce_loose({"file": "src/x.py"})
        assert out["file"] == {"path": "src/x.py"}

    def test_strips_leading_slash(self):
        out = _coerce_loose({"file": "/src/x.py"})
        assert out["file"] == {"path": "src/x.py"}

    def test_strips_dot_slash(self):
        out = _coerce_loose({"file": "./src/x.py"})
        assert out["file"] == {"path": "src/x.py"}

    def test_type_as_string_becomes_dict(self):
        out = _coerce_loose({"type": "int"})
        assert out["type"] == {"expr": "int"}

    def test_default_as_string_becomes_dict(self):
        out = _coerce_loose({"default": "0"})
        assert out["default"] == {"code": "0"}

    def test_symbol_double_colon(self):
        out = _coerce_loose({"symbol": "a.py::Foo.bar"})
        assert out["symbol"] == {"file": "a.py", "name": "Foo.bar"}

    def test_symbol_single_colon(self):
        out = _coerce_loose({"function": "a.py:f"})
        assert out["function"] == {"file": "a.py", "name": "f"}

    def test_span_file_normalized(self):
        out = _coerce_loose({"span": {"file": "/a.py", "start_line": 1, "end_line": 2}})
        assert out["span"]["file"] == "a.py"

    def test_already_correct_passed_through(self):
        payload = {"file": {"path": "a.py"}, "module": "json", "confidence": 0.9}
        assert _coerce_loose(payload) == payload

    def test_full_payload_with_addimport_shorthand(self):
        from lattice.actions import parse_action

        raw = {
            "verb": "AddImport",
            "file": "/src/payments/charge.py",
            "module": "stripe",
            "confidence": 0.9,
        }
        action = parse_action(_coerce_loose(raw))
        assert action.verb == "AddImport"
        assert action.file.path == "src/payments/charge.py"


# ---------------------------------------------------------------------------
# Live LocalLLMProposer smoke test — only runs when explicitly opted in.
# ---------------------------------------------------------------------------


_SMOKE_ENABLED = os.environ.get("LATTICE_LLM_SMOKE") == "1"


@pytest.mark.skipif(not _SMOKE_ENABLED, reason="LATTICE_LLM_SMOKE not set")
def test_local_llm_emits_a_valid_action():
    """Loads a real small model and asserts it produces ONE valid Action."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed")

    from lattice.propose.local import LocalLLMProposer
    from lattice.sense import Symbol, SymbolKind

    proposer = LocalLLMProposer()
    obs = ObservationContext(
        task="Add an import of 'stripe' to src/payments/charge.py.",
        symbols=(
            Symbol(file="src/payments/charge.py", name="ChargeProcessor", kind=SymbolKind.CLASS, line=4),
        ),
    )
    try:
        out = proposer.propose(obs)
    except ProposerError as exc:
        pytest.fail(f"LocalLLMProposer raised: {exc}")
    assert len(out) == 1
    # The model is expected to produce SOMETHING valid; we don't pin the verb
    # since instruction following at 0.5B is imperfect. We just assert the
    # output round-trips through the action parser.
    assert out[0].confidence >= 0.0
