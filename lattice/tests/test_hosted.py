"""Structural tests for HostedLLMProposer.

We can't make real Anthropic API calls without an API key, so these
tests verify the tool-spec generation, schema-inlining, and Proposer
contract conformance. The live API path is covered by the hosted
proposer's own runtime behavior when ANTHROPIC_API_KEY is present.
"""

from __future__ import annotations

import json

import pytest

from lattice.propose import ObservationContext, ProposerError


def test_tool_spec_generation_covers_every_verb():
    from lattice.propose.hosted import _VERB_DESCRIPTIONS, _build_tool_specs

    specs = _build_tool_specs()
    assert {s["name"] for s in specs} == set(_VERB_DESCRIPTIONS.keys())


def test_tool_specs_have_required_anthropic_shape():
    from lattice.propose.hosted import _build_tool_specs

    for spec in _build_tool_specs():
        assert "name" in spec
        assert "description" in spec
        assert "input_schema" in spec
        schema = spec["input_schema"]
        # 'verb' is the discriminator — it must NOT appear in the
        # tool's required list since the tool name carries it.
        if "required" in schema:
            assert "verb" not in schema["required"]


def test_tool_specs_inline_all_refs():
    from lattice.propose.hosted import _build_tool_specs

    blob = json.dumps(_build_tool_specs())
    assert "$ref" not in blob, "tool specs should have no remaining $refs"


def test_tool_specs_are_json_serializable():
    from lattice.propose.hosted import _build_tool_specs

    json.dumps(_build_tool_specs())  # must not raise


def test_addimport_tool_spec_fields():
    from lattice.propose.hosted import _build_tool_specs

    addimport = next(s for s in _build_tool_specs() if s["name"] == "AddImport")
    props = addimport["input_schema"]["properties"]
    assert "file" in props
    assert "module" in props
    assert "confidence" in props
    # nested FileRef should be inlined
    assert props["file"]["properties"]["path"]["type"] == "string"


def test_missing_api_key_raises():
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is set; can't test missing-key path here")

    from lattice.propose.hosted import HostedLLMProposer

    with pytest.raises(ProposerError, match="ANTHROPIC_API_KEY"):
        HostedLLMProposer()


def test_observation_renders_for_hosted_proposer():
    from lattice.propose.local import _render_observation

    obs = ObservationContext(
        task="add an import",
        hints=("WORKSPACE FILES: src/main.py", "skill (score 0.5): use httpx"),
    )
    rendered = _render_observation(obs)
    assert "TASK:" in rendered
    assert "HINTS:" in rendered
    assert "WORKSPACE FILES" in rendered
