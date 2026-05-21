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

    def iter_files(self, *, suffix: str | None = None) -> list[str]:
        """List repository-relative paths in the workspace.

        If *suffix* is given (e.g. '.py'), only return paths ending with
        it. Skips conventional non-source directories like .git, .venv,
        __pycache__, node_modules.
        """
        ...


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

    def iter_files(self, *, suffix: str | None = None) -> list[str]:
        if suffix is None:
            return sorted(self._files.keys())
        return sorted(p for p in self._files if p.endswith(suffix))


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

    def iter_files(self, *, suffix: str | None = None) -> list[str]:
        skip_dirs = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", "dist", "build"}
        out: list[str] = []
        for full in self._root.rglob("*"):
            if not full.is_file():
                continue
            rel = full.relative_to(self._root)
            if any(part in skip_dirs for part in rel.parts):
                continue
            path = rel.as_posix()
            if suffix is not None and not path.endswith(suffix):
                continue
            out.append(path)
        return sorted(out)
