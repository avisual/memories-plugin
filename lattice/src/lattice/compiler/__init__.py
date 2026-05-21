"""Action compiler — turns typed Actions into file changes + diffs.

Deterministic, pure: given an action and current workspace content,
produces a CompiledAction containing one or more FileChange records.
Each FileChange has the prior content, the new content, and a unified
diff. No filesystem mutation happens here — that's the verify gate's
job, and only after the action has been simulated by the world model.

Currently supports a subset of mutating verbs (Python only). Non-
mutating verbs (RecallMore, RevealBody, MarkBlocked, Branch) raise
NonMutatingAction since they're handled by the orchestrator, not the
compiler.
"""

from lattice.compiler.errors import (
    CompileError,
    NonMutatingAction,
    SymbolNotFound,
    UnsupportedAction,
    WorkspaceError,
)
from lattice.compiler.overlay import OverlayWorkspace
from lattice.compiler.python import compile_action
from lattice.compiler.types import CompiledAction, FileChange
from lattice.compiler.workspace import DictWorkspace, FilesystemWorkspace, Workspace

__all__ = [
    "CompileError",
    "CompiledAction",
    "DictWorkspace",
    "FileChange",
    "FilesystemWorkspace",
    "NonMutatingAction",
    "OverlayWorkspace",
    "SymbolNotFound",
    "UnsupportedAction",
    "Workspace",
    "WorkspaceError",
    "compile_action",
]
