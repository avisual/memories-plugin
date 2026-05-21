"""Syntactic verification — does the compiler output parse?

A CompiledAction may contain multiple FileChanges. This gate parses
every changed file's `after` content with ast.parse. If any fails, the
outcome is failure with a categorical error string per file.
"""

from __future__ import annotations

import ast

from pydantic import BaseModel, ConfigDict

from lattice.compiler.types import CompiledAction


class SyntacticOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    ok: bool
    errors: tuple[tuple[str, str], ...] = ()  # (path, message) pairs

    def __bool__(self) -> bool:
        return self.ok


def verify_syntactic(compiled: CompiledAction) -> SyntacticOutcome:
    errors: list[tuple[str, str]] = []
    for change in compiled.file_changes:
        if change.is_noop:
            continue
        try:
            ast.parse(change.after, filename=change.path)
        except SyntaxError as exc:
            errors.append((change.path, f"SyntaxError: {exc.msg} at line {exc.lineno}"))
    return SyntacticOutcome(ok=not errors, errors=tuple(errors))
