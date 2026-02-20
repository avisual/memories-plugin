"""Entry point for ``python -m memories``.

Dispatches to CLI commands (hook, migrate, health) or starts the MCP
server over stdio transport if no CLI command is given.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Dispatch CLI commands or run the MCP server."""
    args = sys.argv[1:]

    # Check if this is a CLI command (hook, migrate, health).
    if args and args[0] in (
        "hook", "migrate", "health", "setup", "stats", "diagnose",
        "backfill", "relink", "normalise", "normalize", "reatomise", "reatomize",
        "eval", "feedback",
    ):
        from memories.cli import dispatch
        dispatch(args)
        return

    # Run the MCP server.
    # Multiple instances are safe: SQLite WAL + per-process write lock
    # handle concurrent access. Each Claude Code window gets its own process.
    from memories.server import mcp
    mcp.run()


if __name__ == "__main__":
    main()
