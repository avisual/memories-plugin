"""Intents — high-level instructions that expand into typed Actions.

An Intent is a human-scale ask ("add a dry_run parameter to all charge
methods"). The orchestrator expands it into a list of Actions; each
Action is then compiled and verified independently. Expansion is the
v0 stand-in for the LLM's `propose` step — same shape, deterministic.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from lattice.actions import Action, AddParameter, Expr, SymbolRef, TypeExpr
from lattice.compiler.workspace import Workspace
from lattice.sense import SymbolKind, find_symbols, walk_workspace


class _Intent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AddParameterToAllMatching(_Intent):
    """Add a parameter to every function or method whose leaf-name matches.

    The orchestrator finds all matching symbols by walking the workspace,
    then emits one AddParameter action per match.
    """

    kind: Literal["AddParameterToAllMatching"] = "AddParameterToAllMatching"
    function_name: str = Field(min_length=1)
    parameter_name: str = Field(min_length=1)
    parameter_type: str = Field(min_length=1)
    parameter_default: str | None = None
    keyword_only: bool = False
    include_functions: bool = True
    include_methods: bool = True


Intent = Annotated[
    Union[AddParameterToAllMatching],
    Field(discriminator="kind"),
]


def expand_intent(
    intent: Intent, workspace: Workspace, files: list[str]
) -> list[Action]:
    """Deterministically expand *intent* into typed Actions.

    For v0 we route on the intent's `kind`; LLM-driven expansion will
    drop into the same return shape via constrained decoding.
    """
    match intent:
        case AddParameterToAllMatching():
            return _expand_add_param_all(intent, workspace, files)
        case _:
            raise ValueError(f"no expansion implemented for intent {intent!r}")


def _expand_add_param_all(
    intent: AddParameterToAllMatching, workspace: Workspace, files: list[str]
) -> list[Action]:
    kinds: list[SymbolKind] = []
    if intent.include_functions:
        kinds.append(SymbolKind.FUNCTION)
    if intent.include_methods:
        kinds.append(SymbolKind.METHOD)
    if not kinds:
        return []

    symbols = walk_workspace(workspace, files)
    matches = find_symbols(symbols, leaf_name=intent.function_name, kinds=tuple(kinds))

    default = Expr(code=intent.parameter_default) if intent.parameter_default else None
    return [
        AddParameter(
            function=SymbolRef(file=sym.file, name=sym.name),
            name=intent.parameter_name,
            type=TypeExpr(expr=intent.parameter_type),
            default=default,
            keyword_only=intent.keyword_only,
            confidence=intent.confidence,
        )
        for sym in matches
    ]
