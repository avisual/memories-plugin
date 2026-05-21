"""Errors the compiler can raise.

Compilation errors are categorical so the orchestrator can route them
correctly: SymbolNotFound triggers a recall+retry, WorkspaceError marks
the branch blocked, UnsupportedAction is an invariant violation.
"""

from __future__ import annotations


class CompileError(Exception):
    """Base class for all compiler errors."""


class WorkspaceError(CompileError):
    """The workspace could not produce a file the action references."""


class SymbolNotFound(CompileError):
    """A SymbolRef did not resolve in the target file."""


class NonMutatingAction(CompileError):
    """Action does not produce file changes; the orchestrator should handle it."""


class UnsupportedAction(CompileError):
    """The compiler does not yet implement this verb."""
