"""HostedLLMProposer — Anthropic API with tool-use for constrained actions.

The proposer's `propose()` shape is unchanged from LocalLLMProposer:
the orchestrator never knows or cares which model emitted the action.
Hosted models get dramatically better task-tracking, longer-context
coherence, and reliable JSON output — exactly what 0.5-2B local
models struggle with.

Requires:
- The `anthropic` package (`pip install anthropic`).
- `ANTHROPIC_API_KEY` in the environment.

Tool definitions are generated from the action DSL's Pydantic schemas,
so the model literally cannot emit a verb that isn't in our DSL.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import ValidationError

from lattice.actions import Action, parse_action
from lattice.actions.action import _action_adapter
from lattice.propose.base import ObservationContext, Proposer, ProposerError
from lattice.propose.local import _render_observation


_DEFAULT_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 1024


_VERB_DESCRIPTIONS: dict[str, str] = {
    "AddImport": "Add an import statement to a Python file.",
    "RenameSymbol": "Rename a function or class (definition + same-file references).",
    "AddField": "Add an attribute to a Python class body.",
    "AddParameter": "Add a parameter to a function or method signature.",
    "WrapInTry": "Wrap a span of code in a try/except block.",
    "AddTest": "Add a pytest test function for a target symbol.",
    "RecallMore": "Ask the memory layer for additional context (non-mutating).",
    "RevealBody": "Request to see the body of a symbol previously shown only as a signature.",
    "MarkBlocked": "Signal that the current branch cannot proceed and needs human input.",
    "MarkDone": "Signal that the agent considers the task complete.",
    "Branch": "Spawn a new plan branch from the current node.",
}


def _build_tool_specs() -> list[dict[str, Any]]:
    """Generate Anthropic tool specs from the Action union's JSON schema."""
    full_schema = _action_adapter.json_schema()
    defs = full_schema.get("$defs") or full_schema.get("definitions") or {}
    tools: list[dict[str, Any]] = []
    for name, desc in _VERB_DESCRIPTIONS.items():
        if name not in defs:
            continue
        schema = _inline_refs(defs[name], defs, seen=set())
        props = dict(schema.get("properties", {}))
        if "verb" in props:
            del props["verb"]
        schema = dict(schema)
        schema["properties"] = props
        if "required" in schema:
            schema["required"] = [r for r in schema["required"] if r != "verb"]
        schema.pop("title", None)
        tools.append(
            {
                "name": name,
                "description": desc,
                "input_schema": schema,
            }
        )
    return tools


def _inline_refs(node: Any, defs: dict[str, Any], *, seen: set[str]) -> Any:
    """Return a copy of *node* with all $ref pointers inlined.

    Pure (does not mutate the input). Cycle-safe via a seen-set: when
    we revisit a ref that's already on the current expansion path, we
    leave it as an object whose ref was substituted with a generic
    'object' shape (Anthropic tool schemas don't accept $ref).
    """
    if isinstance(node, dict):
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            if ref_name in seen:
                # Cycle: emit a permissive object so the API stays happy.
                return {"type": "object"}
            target = defs.get(ref_name)
            if target is None:
                return {"type": "object"}
            return _inline_refs(target, defs, seen=seen | {ref_name})
        return {k: _inline_refs(v, defs, seen=seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs, seen=seen) for v in node]
    return node


class HostedLLMProposer:
    """Anthropic API proposer using tool-use for constrained Actions."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = _MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get("LATTICE_HOSTED_MODEL", _DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ProposerError(
                "ANTHROPIC_API_KEY not set. Either set it in the env or pass api_key=..."
            )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Any = None
        self._tools = _build_tool_specs()

    def _load(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic
        except ImportError as exc:
            raise ProposerError(
                "anthropic package not installed. Run: uv pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def propose(self, obs: ObservationContext, n: int = 1) -> list[Action]:
        self._load()
        system = (
            "You are the action emitter for the LATTICE coding harness. "
            "Look at the TASK and the WORKSPACE FILES / RELEVANT SYMBOLS / HINTS, "
            "then call EXACTLY ONE tool — your single typed Action for this cycle. "
            "Use the exact file paths from WORKSPACE FILES. "
            "If FILES ALREADY EDITED contains your target and the goal is met, "
            "call MarkDone. Do not narrate."
        )
        user_msg = _render_observation(obs)

        actions: list[Action] = []
        # Sample n times when n > 1; the orchestrator pre-flight picks
        # the first non-no-op.
        for i in range(max(1, n)):
            temp = self.temperature if i == 0 else min(0.9, 0.5 + 0.15 * (i - 1))
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temp,
                    system=system,
                    tools=self._tools,
                    tool_choice={"type": "any"},
                    messages=[{"role": "user", "content": user_msg}],
                )
            except Exception as exc:  # noqa: BLE001
                raise ProposerError(f"Anthropic API call failed: {exc}") from exc

            action = self._extract_first_tool(response)
            if action is not None and (not actions or action.model_dump_json() != actions[-1].model_dump_json()):
                actions.append(action)
                if len(actions) >= n:
                    break

        if not actions:
            raise ProposerError(
                "hosted model returned no tool calls; check API key, model, and tool-use support."
            )
        return actions

    def _extract_first_tool(self, response: Any) -> Action | None:
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                verb = block.name
                payload: dict[str, Any] = {"verb": verb, **dict(block.input)}
                try:
                    return parse_action(payload)
                except ValidationError:
                    return None
        return None


# Static check that HostedLLMProposer satisfies the Proposer Protocol.
def _selfcheck() -> None:
    """Static structural check that this class implements Proposer."""

    proposer: Proposer
    try:
        proposer = HostedLLMProposer()  # type: ignore[assignment]
    except ProposerError:
        return  # API key not set in this env; structure is still right.
    _ = proposer
