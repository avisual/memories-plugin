"""The ten starter verbs of the action DSL.

Each verb is a frozen Pydantic model with a `verb` discriminator literal
and a `confidence ∈ [0, 1]`. New verbs may be added (Organ 9 promotes
macros into first-class verbs). The `WrapInTry.handler_body` field
holds nested actions — that's the compositional point at the verb level.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lattice.actions.refs import Expr, FileRef, IntentTag, SpanRef, SymbolRef, TypeExpr

if TYPE_CHECKING:
    from lattice.actions.action import Action


_DOTTED_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


_BlockedReason = Literal[
    "ambiguous_intent",
    "missing_context",
    "incompatible_types",
    "external_dependency",
    "needs_human",
]


class _Verb(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    confidence: float = Field(ge=0.0, le=1.0)


class AddImport(_Verb):
    verb: Literal["AddImport"] = "AddImport"
    file: FileRef
    module: str = Field(min_length=1)
    names: list[str] | None = None
    alias: str | None = None

    @field_validator("module")
    @classmethod
    def _module_is_dotted_identifier(cls, v: str) -> str:
        if not _DOTTED_IDENTIFIER_RE.match(v):
            raise ValueError(
                f"module must be a dotted Python identifier (e.g. 'os.path'), got {v!r}"
            )
        return v

    @field_validator("alias")
    @classmethod
    def _alias_is_identifier(cls, v: str | None) -> str | None:
        if v is not None and not _IDENTIFIER_RE.match(v):
            raise ValueError(f"alias must be a Python identifier, got {v!r}")
        return v

    @field_validator("names")
    @classmethod
    def _names_are_identifiers(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        for name in v:
            if not _IDENTIFIER_RE.match(name):
                raise ValueError(f"import name must be a Python identifier, got {name!r}")
        return v

    def model_post_init(self, __context) -> None:
        if self.alias is not None:
            if self.names is not None and len(self.names) != 1:
                raise ValueError(
                    "alias is only valid for a bare 'import module' or a single-name import"
                )
        if self.names is not None:
            if len(self.names) == 0:
                raise ValueError("names must be non-empty when provided")
            if len(set(self.names)) != len(self.names):
                raise ValueError("names must be unique")


class RenameSymbol(_Verb):
    verb: Literal["RenameSymbol"] = "RenameSymbol"
    symbol: SymbolRef
    new_name: str = Field(min_length=1)

    @field_validator("new_name")
    @classmethod
    def _new_name_is_identifier(cls, v: str) -> str:
        if not _IDENTIFIER_RE.match(v):
            raise ValueError(f"new_name must be a Python identifier, got {v!r}")
        return v

    def model_post_init(self, __context) -> None:
        if self.new_name == self.symbol.name.split(".")[-1]:
            raise ValueError("new_name is identical to current name")


class AddField(_Verb):
    verb: Literal["AddField"] = "AddField"
    cls: SymbolRef
    name: str = Field(min_length=1)
    type: TypeExpr
    default: Expr | None = None

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        if not _IDENTIFIER_RE.match(v):
            raise ValueError(f"name must be a Python identifier, got {v!r}")
        return v


class AddParameter(_Verb):
    verb: Literal["AddParameter"] = "AddParameter"
    function: SymbolRef
    name: str = Field(min_length=1)
    type: TypeExpr
    default: Expr | None = None
    position: int | None = Field(default=None, ge=0)
    keyword_only: bool = False

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        if not _IDENTIFIER_RE.match(v):
            raise ValueError(f"name must be a Python identifier, got {v!r}")
        return v

    def model_post_init(self, __context) -> None:
        if self.keyword_only and self.position is not None:
            raise ValueError("keyword_only parameters cannot have a positional index")


class WrapInTry(_Verb):
    verb: Literal["WrapInTry"] = "WrapInTry"
    span: SpanRef
    exception_type: TypeExpr
    handler_body: list["Action"] = Field(default_factory=list)
    finally_body: list["Action"] | None = None


class AddTest(_Verb):
    verb: Literal["AddTest"] = "AddTest"
    target: SymbolRef
    test_name: str = Field(min_length=1)
    given: Expr
    when: Expr
    then: Expr

    def model_post_init(self, __context) -> None:
        if not self.test_name.startswith("test_"):
            raise ValueError("test_name must start with 'test_'")


class RecallMore(_Verb):
    """Non-mutating: ask the memory layer for more context."""

    verb: Literal["RecallMore"] = "RecallMore"
    query: str = Field(min_length=1)
    intent: IntentTag | None = None


class RevealBody(_Verb):
    """Non-mutating: request the body of a symbol previously seen as a signature."""

    verb: Literal["RevealBody"] = "RevealBody"
    symbol: SymbolRef


class MarkBlocked(_Verb):
    """Signal that the current plan branch cannot proceed.

    Triggers Organ 6 (population) to deactivate this branch and may seed
    a new branch from a sibling.
    """

    verb: Literal["MarkBlocked"] = "MarkBlocked"
    reason_code: _BlockedReason
    detail: str = Field(min_length=1, max_length=280)


class Branch(_Verb):
    """Spawn a new plan branch from the current plan-DAG node."""

    verb: Literal["Branch"] = "Branch"
    rationale: str = Field(min_length=1, max_length=280)
