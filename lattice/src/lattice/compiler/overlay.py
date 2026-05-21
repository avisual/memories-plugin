"""Overlay workspace — chained edits on top of a base workspace.

Used by the orchestrator so each action sees the cumulative state of
prior actions in the same plan. Read-through to the base; writes land
only in the in-memory overlay. The base is never mutated.
"""

from __future__ import annotations

from lattice.compiler.workspace import Workspace, WorkspaceError


class OverlayWorkspace:
    """Read-through, in-memory write-overlay on top of a base Workspace."""

    def __init__(self, base: Workspace) -> None:
        self._base = base
        self._overlay: dict[str, str] = {}

    def read(self, path: str) -> str:
        if path in self._overlay:
            return self._overlay[path]
        return self._base.read(path)

    def exists(self, path: str) -> bool:
        if path in self._overlay:
            return True
        return self._base.exists(path)

    def iter_files(self, *, suffix: str | None = None) -> list[str]:
        base_files = self._base.iter_files(suffix=suffix)
        combined = set(base_files) | {
            p for p in self._overlay if suffix is None or p.endswith(suffix)
        }
        return sorted(combined)

    def update(self, path: str, content: str) -> None:
        """Stage *content* as the new value of *path* in the overlay."""
        self._overlay[path] = content

    def overlay_paths(self) -> tuple[str, ...]:
        """Paths that have been edited through this overlay."""
        return tuple(sorted(self._overlay))

    def base_content(self, path: str) -> str:
        """Read the *original* (pre-overlay) content of *path*.

        Useful for computing a final consolidated diff against the
        baseline after a sequence of chained actions.
        """
        try:
            return self._base.read(path)
        except WorkspaceError:
            return ""
