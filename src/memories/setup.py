"""Automated setup and uninstall for the memories system.

Provides idempotent ``run_setup()`` and ``run_uninstall()`` functions that
configure a fresh machine for brain-like memory:

1. Create ``~/.memories/`` and initialize the database.
2. Pull the Ollama embedding model (gracefully handles Ollama being absent).
3. Register the MCP server in ``~/.claude.json``.
4. Configure lifecycle hooks in ``~/.claude/settings.json``.
5. Run a health check and print next-steps.

Usage::

    python -m memories setup
    python -m memories setup --uninstall
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CLAUDE_JSON = Path("~/.claude.json").expanduser()
_CLAUDE_SETTINGS = Path("~/.claude/settings.json").expanduser()
_MEMORIES_DIR = Path("~/.memories").expanduser()

# ---------------------------------------------------------------------------
# MCP entry
# ---------------------------------------------------------------------------


def _mcp_entry() -> dict[str, Any]:
    """Build the MCP server registration entry for memories."""
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "memories"],
    }


# ---------------------------------------------------------------------------
# Hook entries
# ---------------------------------------------------------------------------

_HOOK_ENTRIES: dict[str, list[dict[str, Any]]] = {
    "SessionStart": [
        {
            "matcher": "startup|clear|compact",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook session-start",
                    "timeout": 30,
                },
            ],
        },
    ],
    "UserPromptSubmit": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook prompt-submit",
                    "timeout": 15,
                },
            ],
        },
    ],
    "PreCompact": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook pre-compact",
                    "timeout": 10,
                },
            ],
        },
    ],
    "PostToolUse": [
        {
            "matcher": "Bash",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook post-tool",
                    "timeout": 10,
                },
            ],
        },
    ],
    "Stop": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook stop",
                    "timeout": 30,
                },
            ],
        },
    ],
    "PostToolUseFailure": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook post-tool-failure",
                    "timeout": 10,
                },
            ],
        },
    ],
    "SessionEnd": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook session-end",
                    "timeout": 15,
                },
            ],
        },
    ],
    "SubagentStart": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook subagent-start",
                    "timeout": 10,
                },
            ],
        },
    ],
    "PermissionRequest": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook permission-request",
                    "timeout": 10,
                },
            ],
        },
    ],
    "TaskCompleted": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook task-completed",
                    "timeout": 10,
                },
            ],
        },
    ],
    "Notification": [
        {
            "matcher": "elicitation_dialog",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m memories hook notification",
                    "timeout": 10,
                },
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# JSON file helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file, returning empty dict if missing."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with backup of the original file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------


def _setup_database() -> list[str]:
    """Create ~/.memories/ and initialize the database."""
    _MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [ok] Database directory: {_MEMORIES_DIR}")
    return []


def _setup_ollama(interactive: bool = True) -> list[str]:
    """Pull the Ollama embedding model if Ollama is available."""
    import asyncio
    from memories.ollama_manager import OllamaManager

    next_steps: list[str] = []

    if not shutil.which("ollama"):
        print("  [skip] Ollama not found on PATH")
        next_steps.append("Install Ollama: https://ollama.com/download")
        next_steps.append("Then run: ollama pull nomic-embed-text")
        return next_steps

    # Use OllamaManager for automatic setup
    manager = OllamaManager()

    async def _async_setup():
        success, message = await manager.ensure_ready()
        if success:
            print("  [ok] Ollama model: nomic-embed-text")
        else:
            print(f"  [warn] {message}")
            next_steps.append("Run manually: ollama pull nomic-embed-text")

    try:
        asyncio.run(_async_setup())
    except Exception as exc:
        print(f"  [warn] Ollama setup failed: {exc}")
        next_steps.append("Run manually: ollama pull nomic-embed-text")

    return next_steps


def _setup_mcp() -> list[str]:
    """Register the MCP server in ~/.claude.json."""
    data = _read_json(_CLAUDE_JSON)
    if "mcpServers" not in data:
        data["mcpServers"] = {}

    data["mcpServers"]["memories"] = _mcp_entry()
    _write_json(_CLAUDE_JSON, data)
    print(f"  [ok] MCP server registered in {_CLAUDE_JSON}")
    return []


def _setup_hooks() -> list[str]:
    """Configure lifecycle hooks in ~/.claude/settings.json."""
    data = _read_json(_CLAUDE_SETTINGS)
    if "hooks" not in data:
        data["hooks"] = {}

    for event_name, entries in _HOOK_ENTRIES.items():
        existing = data["hooks"].get(event_name, [])
        # Filter out any existing memories entries to avoid duplicates.
        existing = [
            e for e in existing
            if not (isinstance(e, dict) and any(
                "memories" in h.get("command", "")
                for h in e.get("hooks", [])
            ))
        ]
        existing.extend(entries)
        data["hooks"][event_name] = existing

    _write_json(_CLAUDE_SETTINGS, data)
    print(f"  [ok] Hooks configured in {_CLAUDE_SETTINGS}")
    return []


def _run_health_check() -> list[str]:
    """Run the health check and report status."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "memories", "health"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print(f"  [ok] Health check passed")
            for line in result.stdout.strip().splitlines():
                print(f"        {line}")
        else:
            print(f"  [warn] Health check issues: {result.stderr.strip()}")
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"  [warn] Health check failed: {exc}")

    return []


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def _ask_user(prompt: str, default: bool = True) -> bool:
    """Ask user for yes/no confirmation (interactive mode only).

    Parameters
    ----------
    prompt:
        The question to ask the user.
    default:
        Default value if user just presses Enter.

    Returns
    -------
    bool
        ``True`` if user confirms, ``False`` otherwise.
    """
    default_str = "[Y/n]" if default else "[y/N]"
    try:
        response = input(f"{prompt} {default_str}: ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


async def setup_interactive() -> bool:
    """Interactive setup wizard with explicit user consent.

    Returns
    -------
    bool
        ``True`` if setup completed successfully, ``False`` otherwise.
    """
    print("🧠 Memories Setup Wizard\n")
    print("This wizard will guide you through setup.")
    print("You'll be asked before any changes are made.\n")

    # Step 1: Check prerequisites (READ-ONLY)
    print("━━━ Step 1: Checking Prerequisites ━━━\n")

    from memories.ollama_manager import OllamaManager
    manager = OllamaManager()

    ollama_installed = manager.is_installed()
    print(f"  Ollama installed: {'✓' if ollama_installed else '✗'}")

    if not ollama_installed:
        print("\n  ⚠️  Ollama is required but not installed.")
        print("  Please install it manually:\n")
        print("    macOS:  brew install ollama")
        print("    Linux:  curl -fsSL https://ollama.com/install.sh | sh")
        print("    Other:  https://ollama.com/download\n")
        print("  Then run this setup again.")
        return False

    # Check if daemon running
    daemon_running = await manager.is_daemon_running()
    print(f"  Ollama daemon running: {'✓' if daemon_running else '✗'}")

    if not daemon_running:
        print("\n  Ollama daemon is not running.")
        response = input("  Start Ollama now? [y/N]: ").strip().lower()
        if response == 'y':
            print("  Starting Ollama...")
            started = await manager.try_start_daemon()
            if not started:
                print("  ✗ Failed to start. Please run manually: ollama serve")
                return False
            print("  ✓ Ollama started")
        else:
            print("  Please start Ollama manually: ollama serve")
            return False

    # Check model
    model_exists = await manager.has_model("nomic-embed-text")
    print(f"  Model nomic-embed-text: {'✓' if model_exists else '✗'}")

    if not model_exists:
        print("\n  The embedding model needs to be downloaded (~274MB).")
        response = input("  Download now? [y/N]: ").strip().lower()
        if response == 'y':
            print("  Pulling model (this may take a minute)...")
            success = await manager.pull_model("nomic-embed-text")
            if not success:
                print("  ✗ Failed. Please run manually: ollama pull nomic-embed-text")
                return False
            print("  ✓ Model downloaded")
        else:
            print("  Please download manually: ollama pull nomic-embed-text")
            return False

    # Step 2: Create database directory
    print("\n━━━ Step 2: Database Directory ━━━\n")
    db_path = _MEMORIES_DIR
    if db_path.exists():
        print(f"  ✓ {db_path} already exists")
    else:
        response = input(f"  Create {db_path}? [Y/n]: ").strip().lower()
        if response != 'n':
            db_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created {db_path}")
        else:
            print("  Skipped. You'll need to create it manually.")

    # Step 3: MCP server registration
    print("\n━━━ Step 3: MCP Server Registration ━━━\n")
    # Show what would be added
    venv_python = sys.executable
    print("  Add this to your ~/.claude.json under 'mcpServers':\n")
    print('  "memories": {')
    print('    "type": "stdio",')
    print(f'    "command": "{venv_python}",')
    print('    "args": ["-m", "memories"]')
    print('  }\n')

    response = input("  Auto-register in ~/.claude.json? [y/N]: ").strip().lower()
    if response == 'y':
        success = await _register_mcp_server()
        if success:
            print("  ✓ Registered in ~/.claude.json")
        else:
            print("  ✗ Failed. Please add manually.")
    else:
        print("  Please add the config manually.")

    # Step 4: Health check
    print("\n━━━ Step 4: Health Check ━━━\n")
    try:
        from memories.brain import Brain
        brain = Brain()
        await brain.initialize()
        status = await brain.status()
        await brain.shutdown()

        print("  ✓ Brain initialized")
        print(f"    Atoms: {status.get('total_atoms', 0)}")
        print(f"    Ollama: {'healthy' if status.get('ollama_healthy') else 'unhealthy'}")
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        return False

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    print("✅ Setup complete!")
    print("   Restart Claude Code to activate memories.\n")
    return True


async def _register_mcp_server() -> bool:
    """Register memories MCP server in Claude Code config.

    Returns
    -------
    bool
        ``True`` if registration succeeded, ``False`` otherwise.
    """
    try:
        config = _read_json(_CLAUDE_JSON)

        if "mcpServers" not in config:
            config["mcpServers"] = {}

        config["mcpServers"]["memories"] = _mcp_entry()
        _write_json(_CLAUDE_JSON, config)
        return True
    except Exception:
        return False


def run_setup(args: list[str] | None = None) -> None:
    """Run the full setup sequence with user consent prompts.

    Parameters
    ----------
    args:
        CLI arguments after ``setup`` (supports ``--uninstall``, ``--non-interactive``, ``--interactive``).
    """
    if args and "--uninstall" in args:
        run_uninstall()
        return

    # Check for explicit --interactive flag (enhanced wizard mode)
    if args and "--interactive" in args:
        import asyncio
        success = asyncio.run(setup_interactive())
        sys.exit(0 if success else 1)

    interactive = not (args and "--non-interactive" in args)

    print("memories setup")
    print("=" * 40)
    print("\nThis will configure the memories system for Claude Code.\n")

    if interactive:
        print("The setup will:")
        print("  1. Create ~/.memories/ directory for the database")
        print("  2. Pull the Ollama embedding model (nomic-embed-text)")
        print("  3. Register the MCP server in ~/.claude.json")
        print("  4. Configure lifecycle hooks in ~/.claude/settings.json")
        print()

        if not _ask_user("Continue with setup?"):
            print("Setup cancelled.")
            return

    next_steps: list[str] = []

    # Step 1: Database (always run, minimal impact)
    print("\nDatabase:")
    next_steps.extend(_setup_database())

    # Step 2: Ollama model (ask permission)
    if interactive:
        if _ask_user("\nPull Ollama embedding model? (required for embeddings)"):
            print("Ollama model:")
            next_steps.extend(_setup_ollama(interactive=True))
        else:
            print("  [skip] Skipped by user")
            next_steps.append("Pull model manually: ollama pull nomic-embed-text")
    else:
        print("\nOllama model:")
        next_steps.extend(_setup_ollama(interactive=False))

    # Step 3: MCP server (ask permission)
    if interactive:
        if _ask_user("\nRegister MCP server in ~/.claude.json?"):
            print("MCP server:")
            next_steps.extend(_setup_mcp())
        else:
            print("  [skip] Skipped by user")
            next_steps.append("Register MCP server manually in ~/.claude.json")
    else:
        print("\nMCP server:")
        next_steps.extend(_setup_mcp())

    # Step 4: Hooks (ask permission)
    if interactive:
        if _ask_user("\nConfigure lifecycle hooks in ~/.claude/settings.json?"):
            print("Hooks:")
            next_steps.extend(_setup_hooks())
        else:
            print("  [skip] Skipped by user")
            next_steps.append("Configure hooks manually in ~/.claude/settings.json")
    else:
        print("\nHooks:")
        next_steps.extend(_setup_hooks())

    # Step 5: Health check (always run)
    print("\nHealth check:")
    next_steps.extend(_run_health_check())

    print("\n" + "=" * 40)
    if next_steps:
        print("\nNext steps:")
        for step in next_steps:
            print(f"  - {step}")
    else:
        print("\nSetup complete! Start a new Claude Code session to activate.")


def run_uninstall() -> None:
    """Remove memories configuration from Claude Code.

    Does NOT delete ``~/.memories/`` (user data).
    """
    print("memories uninstall")
    print("=" * 40)

    # Remove MCP server entry.
    data = _read_json(_CLAUDE_JSON)
    if "mcpServers" in data and "memories" in data["mcpServers"]:
        del data["mcpServers"]["memories"]
        _write_json(_CLAUDE_JSON, data)
        print(f"  [ok] Removed MCP server from {_CLAUDE_JSON}")
    else:
        print(f"  [skip] No MCP entry found in {_CLAUDE_JSON}")

    # Remove hook entries.
    data = _read_json(_CLAUDE_SETTINGS)
    if "hooks" in data:
        changed = False
        for event_name in list(data["hooks"].keys()):
            original = data["hooks"][event_name]
            filtered = [
                e for e in original
                if not (isinstance(e, dict) and "memories" in e.get("command", ""))
            ]
            if len(filtered) != len(original):
                data["hooks"][event_name] = filtered
                changed = True
                # Remove empty event keys.
                if not filtered:
                    del data["hooks"][event_name]

        if changed:
            _write_json(_CLAUDE_SETTINGS, data)
            print(f"  [ok] Removed hooks from {_CLAUDE_SETTINGS}")
        else:
            print(f"  [skip] No memories hooks found in {_CLAUDE_SETTINGS}")
    else:
        print(f"  [skip] No hooks section in {_CLAUDE_SETTINGS}")

    print(f"\n  Note: ~/.memories/ directory was NOT deleted.")
    print(f"  To remove all data: rm -rf {_MEMORIES_DIR}")
    print("\nUninstall complete.")
