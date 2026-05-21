"""Tests for OverlayWorkspace.fork / snapshot / restore.

These primitives exist for beam-search (Organ 6 v1) — each alive
branch needs its own overlay so it can commit different edits
without leaking back to siblings. The fork is shallow-copy of the
overlay dict (str values are immutable; the base workspace is
shared read-only by contract).
"""

from __future__ import annotations

from lattice.compiler import DictWorkspace, OverlayWorkspace


def test_fork_produces_independent_overlay():
    """Edits on the fork don't leak back to the parent."""
    base = DictWorkspace({"src/main.py": "original\n"})
    parent = OverlayWorkspace(base)
    parent.update("src/main.py", "parent edit\n")

    forked = parent.fork()
    assert forked.read("src/main.py") == "parent edit\n"

    forked.update("src/main.py", "fork edit\n")
    assert forked.read("src/main.py") == "fork edit\n"
    # Parent still sees its own state.
    assert parent.read("src/main.py") == "parent edit\n"


def test_fork_shares_base_workspace():
    """The base Workspace is shared (read-through). Reads to a
    file that's neither in parent nor fork overlay come from the
    same base.
    """
    base = DictWorkspace({"src/main.py": "untouched\n"})
    parent = OverlayWorkspace(base)
    parent.update("src/other.py", "parent only\n")

    forked = parent.fork()
    # Base file is visible to both.
    assert forked.read("src/main.py") == "untouched\n"
    # Parent's overlay entries copy into the fork.
    assert forked.read("src/other.py") == "parent only\n"


def test_snapshot_and_restore_roundtrip():
    """snapshot() captures current overlay state; restore() rolls back."""
    base = DictWorkspace({"src/main.py": "v0\n"})
    ws = OverlayWorkspace(base)
    ws.update("src/main.py", "v1\n")
    snap = ws.snapshot()

    ws.update("src/main.py", "v2\n")
    assert ws.read("src/main.py") == "v2\n"

    ws.restore(snap)
    assert ws.read("src/main.py") == "v1\n"


def test_snapshot_isolated_from_overlay_mutation():
    """A captured snapshot is its own dict — overlay edits after capture
    don't show up in the snapshot."""
    base = DictWorkspace({"src/main.py": "v0\n"})
    ws = OverlayWorkspace(base)
    ws.update("src/main.py", "v1\n")
    snap = ws.snapshot()

    ws.update("src/main.py", "v2\n")
    assert snap["src/main.py"] == "v1\n"
