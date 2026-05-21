"""Workspace abstraction — the compiler reads from one of these.

Pure: the compiler never writes through a Workspace. That's verify's
job, working from the CompiledAction's FileChange records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from lattice.compiler.errors import WorkspaceError


@runtime_checkable
class Workspace(Protocol):
    def read(self, path: str) -> str:
        """Return the current text content of *path*.

        Raises WorkspaceError if the path doesn't exist or can't be read.
        """
        ...

    def exists(self, path: str) -> bool: ...


class DictWorkspace:
    """In-memory workspace; for tests and the world model's imagination."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self._files: dict[str, str] = dict(files or {})

    def read(self, path: str) -> str:
        try:
            return self._files[path]
        except KeyError as exc:
            raise WorkspaceError(f"no such file in workspace: {path!r}") from exc

    def exists(self, path: str) -> bool:
        return path in self._files


class FilesystemWorkspace:
    """Reads files from a real filesystem root, repository-relative."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {self._root}")

    def _resolve(self, path: str) -> Path:
        if path.startswith("/"):
            raise WorkspaceError(f"path must be repository-relative, got {path!r}")
        full = (self._root / path).resolve()
        try:
            full.relative_to(self._root)
        except ValueError as exc:
            raise WorkspaceError(f"path escapes workspace root: {path!r}") from exc
        return full

    def read(self, path: str) -> str:
        full = self._resolve(path)
        try:
            return full.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise WorkspaceError(f"no such file: {path!r}") from exc
        except OSError as exc:
            raise WorkspaceError(f"could not read {path!r}: {exc}") from exc

    def exists(self, path: str) -> bool:
        try:
            return self._resolve(path).is_file()
        except WorkspaceError:
            return False
