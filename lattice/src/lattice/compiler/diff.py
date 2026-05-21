"""Unified-diff generation."""

from __future__ import annotations

import difflib


def unified_diff(*, path: str, before: str, after: str, context: int = 3) -> str:
    """Return a git-style unified diff between *before* and *after*.

    Empty string when before == after.
    """
    if before == after:
        return ""
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
    )
    return "".join(lines)
