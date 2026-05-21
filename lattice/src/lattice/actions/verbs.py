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
    # Default to 0.5 (mid-confidence) so a small model that forgets to
    # emit the field still gets a valid action. The orchestrator can
    # weight low-confidence candidates lower at pre-flight, so the
    # signal is preserved without making confidence a structural gate.
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


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


class MarkDone(_Verb):
    """Signal that the agent considers the task complete.

    The orchestrator stops iterating on this verb. Carries a short
    summary the user (and future recall) can see.
    """

    verb: Literal["MarkDone"] = "MarkDone"
    summary: str = Field(min_length=1, max_length=280)


_HTTP_URL_RE = re.compile(r"^https?://[^\s<>\"']+$", re.IGNORECASE)


_INSERT_POSITION = Literal["end", "top_after_imports"]


class AddStatement(_Verb):
    """Insert a module-level statement at a chosen position in a file.

    Closes the gap between AddImport and 'actually wire it up' — e.g.
    after AddImport of flask_cors, AddStatement(`cors = CORS(app)`)
    completes the integration.

    `code` is parsed with libcst.parse_module before insertion, so
    invalid Python is rejected at compile time. Multiple statements
    are allowed (each becomes its own SimpleStatementLine).
    """

    verb: Literal["AddStatement"] = "AddStatement"
    file: FileRef
    code: str = Field(min_length=1, max_length=4000)
    position: _INSERT_POSITION = "end"


class AddFunction(_Verb):
    """Insert a complete function definition at module level.

    `source` is the full source of the function (def or async def),
    including any decorators. libcst.parse_statement validates it
    before insertion so the file is guaranteed parseable.

    Examples of `source`:
        def health() -> dict:
            return {"ok": True}

        @app.route("/health")
        def health():
            return {"ok": True}
    """

    verb: Literal["AddFunction"] = "AddFunction"
    file: FileRef
    source: str = Field(min_length=4, max_length=8000)
    position: _INSERT_POSITION = "end"


class Research(_Verb):
    """Fetch a URL into the brain as an atom — the LLM learns at runtime.

    The harness fetches the URL with browser-impersonating HTTP, strips
    HTML to a readable extract, and stores it as a fact atom tagged
    with the URL so future recall surfaces it. The next cycle sees the
    fetched knowledge in the observation context.

    Use this when the task references a library/API/pattern the brain
    doesn't already know about. SSRF-guarded: localhost / private
    networks are rejected at fetch time.
    """

    verb: Literal["Research"] = "Research"
    url: str = Field(min_length=8, max_length=2048)
    reason: str = Field(
        min_length=1,
        max_length=200,
        description="One short sentence: why does the agent need this URL?",
    )

    @field_validator("url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        if not _HTTP_URL_RE.match(v):
            raise ValueError("url must start with http:// or https://")
        return v
