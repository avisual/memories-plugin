"""ACT — the typed action DSL.

The LLM's only code-modifying output channel. Every verb is a Pydantic
model; every action that passes validation is structurally well-formed
by construction. Symbol resolution and syntactic-validity guarantees
come from later organs (store, world_model, compiler).
"""

from lattice.actions.action import Action, parse_action
from lattice.actions.refs import Expr, FileRef, IntentTag, SpanRef, SymbolRef, TypeExpr
from lattice.actions.verbs import (
    AddDecorator,
    AddField,
    AddFunction,
    AddImport,
    AddParameter,
    AddStatement,
    AddTest,
    Branch,
    ChangeReturnType,
    DeleteSymbol,
    MarkBlocked,
    MarkDone,
    ModifyDocstring,
    MoveSymbol,
    RecallMore,
    RenameSymbol,
    ReplaceBody,
    Research,
    RevealBody,
    WrapInTry,
)

__all__ = [
    "Action",
    "AddDecorator",
    "AddField",
    "AddFunction",
    "AddImport",
    "AddParameter",
    "AddStatement",
    "AddTest",
    "Branch",
    "ChangeReturnType",
    "DeleteSymbol",
    "Expr",
    "FileRef",
    "IntentTag",
    "MarkBlocked",
    "MarkDone",
    "ModifyDocstring",
    "MoveSymbol",
    "RecallMore",
    "RenameSymbol",
    "ReplaceBody",
    "Research",
    "RevealBody",
    "SpanRef",
    "SymbolRef",
    "TypeExpr",
    "WrapInTry",
    "parse_action",
]
