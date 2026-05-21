"""Apply-to-disk helpers.

Writes verified file changes to the real filesystem. Path-safety
guards mirror FilesystemWorkspace's: repository-relative paths only,
no traversal, can't escape the root.
"""

from __future__ import annotations

from pathlib import Path

from lattice.compiler.errors import WorkspaceError
from lattice.compiler.types import CompiledAction


def _resolve(root: Path, rel: str) -> Path:
    if rel.startswith("/"):
        raise WorkspaceError(f"path must be repository-relative, got {rel!r}")
    if ".." in rel.split("/"):
        raise WorkspaceError(f"path may not contain '..' segments: {rel!r}")
    full = (root / rel).resolve()
    try:
        full.relative_to(root.resolve())
    except ValueError as exc:
        raise WorkspaceError(f"path escapes workspace root: {rel!r}") from exc
    return full


def write_compiled(compiled: CompiledAction, *, root: str | Path) -> list[str]:
    """Apply a single CompiledAction's file changes to disk.

    Returns the list of repository-relative paths actually written
    (no-op file_changes are skipped). Atomic per-file via write-then-
    rename so a crash mid-write leaves the original intact.
    """
    root_path = Path(root)
    written: list[str] = []
    for change in compiled.file_changes:
        if change.is_noop:
            continue
        target = _resolve(root_path, change.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".lattice.tmp")
        tmp.write_text(change.after, encoding="utf-8")
        tmp.replace(target)
        written.append(change.path)
    return written


def write_final(report: object, *, root: str | Path) -> list[str]:
    """Apply an ExecutionReport's `final_files` map to disk.

    Accepts ExecutionReport-shaped objects (anything with a
    `final_files: dict[str, str]` attribute) to avoid an import cycle
    with the orchestrator module.
    """
    final_files: dict[str, str] = getattr(report, "final_files", {})
    if not final_files:
        return []
    root_path = Path(root)
    written: list[str] = []
    for rel_path, content in final_files.items():
        target = _resolve(root_path, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".lattice.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(target)
        written.append(rel_path)
    return written


__all__ = ["write_compiled", "write_final"]
