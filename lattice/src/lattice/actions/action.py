"""The Action discriminated union and parser entry point."""

from __future__ import annotations

from typing import Annotated, Any, Union

from pydantic import Field, TypeAdapter, ValidationError

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

Action = Annotated[
    Union[
        AddImport,
        RenameSymbol,
        AddField,
        AddParameter,
        WrapInTry,
        AddTest,
        AddStatement,
        AddFunction,
        AddDecorator,
        DeleteSymbol,
        ChangeReturnType,
        ModifyDocstring,
        MoveSymbol,
        ReplaceBody,
        RecallMore,
        RevealBody,
        MarkBlocked,
        MarkDone,
        Branch,
        Research,
    ],
    Field(discriminator="verb"),
]


_action_adapter: TypeAdapter[Action] = TypeAdapter(Action)
"""Public so HostedLLMProposer can emit tool-spec JSON schemas from it."""

WrapInTry.model_rebuild()


def parse_action(data: dict[str, Any]) -> Action:
    """Parse a dict into a typed Action.

    Raises pydantic.ValidationError if the data does not match exactly one
    verb's schema. Constrained-decoding wrappers (Outlines, structured-
    output APIs) consume the JSON schema of this union to ensure the LLM
    can only emit values that pass this parser.
    """
    return _action_adapter.validate_python(data)


def action_json_schema() -> dict[str, Any]:
    """Return the JSON schema of the Action union.

    Used by constrained-decoding stacks to restrict the LLM's output
    space.
    """
    return _action_adapter.json_schema()


__all__ = ["Action", "ValidationError", "action_json_schema", "parse_action"]
