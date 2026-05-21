"""Typed references — the values verbs operate on.

A reference identifies a thing in the world (a symbol, a file, a span,
a type, an expression) by structural data, not by free text. The world
model and the compiler validate that a reference resolves to something
real; this module only enforces shape.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FileRef(_Frozen):
    path: str = Field(min_length=1, description="Repository-relative file path.")

    @field_validator("path")
    @classmethod
    def _no_traversal(cls, v: str) -> str:
        if ".." in v.split("/"):
            raise ValueError("path may not contain '..' segments")
        if v.startswith("/"):
            raise ValueError("path must be repository-relative, not absolute")
        return v


class SymbolRef(_Frozen):
    file: str = Field(min_length=1)
    name: str = Field(
        min_length=1,
        description="Dotted name, e.g. 'ChargeProcessor.refund' or 'charge_customer'.",
    )

    @field_validator("file")
    @classmethod
    def _no_traversal(cls, v: str) -> str:
        if ".." in v.split("/"):
            raise ValueError("file may not contain '..' segments")
        if v.startswith("/"):
            raise ValueError("file must be repository-relative, not absolute")
        return v


class SpanRef(_Frozen):
    file: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @field_validator("end_line")
    @classmethod
    def _end_after_start(cls, v: int, info) -> int:
        start = info.data.get("start_line")
        if start is not None and v < start:
            raise ValueError("end_line must be >= start_line")
        return v


class TypeExpr(_Frozen):
    """A type expression as it would appear in the target language.

    Not parsed here — the world model validates resolvability against the
    project's type lattice. This wrapper only ensures non-empty content.
    """

    expr: str = Field(min_length=1)


class Expr(_Frozen):
    """A code expression as a string in the target language.

    Not parsed here — the compiler validates syntactically when it
    materializes the action.
    """

    code: str = Field(min_length=1)


class IntentTag(_Frozen):
    """A short, structured label for the intent behind an action.

    Used for clustering traces during macro extraction (Organ 9).
    """

    label: str = Field(min_length=1, max_length=64)
