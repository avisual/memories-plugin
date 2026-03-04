"""Ollama daemon and model management with auto-setup capabilities.

Provides health checking, daemon startup, and model pulling for the Ollama
embedding service. Detects common configuration issues and provides helpful
error messages with installation and troubleshooting instructions.

Usage::

    from memories.ollama_manager import OllamaManager

    manager = OllamaManager()
    success, message = await manager.ensure_ready()
    if not success:
        print(f"Ollama setup failed: {message}")
"""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess

import httpx


class OllamaManager:
    """Manages Ollama daemon and models with automatic setup.

    Handles installation detection, daemon health checking, automatic daemon
    startup, and model downloading. Provides platform-specific installation
    instructions when Ollama is not available.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        required_model: str = "nomic-embed-text",
    ):
        """Initialize OllamaManager.

        Parameters
        ----------
        base_url:
            The URL of the Ollama API server. Defaults to the standard
            local instance at ``http://localhost:11434``.
        required_model:
            The embedding model that must be available. Defaults to
            ``nomic-embed-text``.
        """
        self.base_url = base_url
        self.required_model = required_model

    async def ensure_ready(self) -> tuple[bool, str]:
        """Ensure Ollama is ready for use. Returns (success, message).

        Performs a complete health check sequence:

        1. Check if Ollama is installed on the system
        2. Check if the daemon is running
        3. Attempt to start the daemon if it's not running
        4. Check if the required embedding model is available
        5. Attempt to pull the model if it's missing

        Returns
        -------
        tuple[bool, str]
            A tuple of ``(success, message)`` where *success* is ``True``
            if Ollama is fully ready, and *message* contains either a
            success confirmation or helpful error/instruction text.
        """
        # Step 1: Is Ollama installed?
        if not self.is_installed():
            return False, self._install_instructions()

        # Step 2: Is daemon running?
        if not await self.is_daemon_running():
            print("  [info] Ollama daemon not running, attempting to start...")
            started = await self.try_start_daemon()
            if not started:
                return False, self._daemon_instructions()

        # Step 3: Is model available?
        if not await self.has_model(self.required_model):
            print(f"  [info] Pulling {self.required_model} model...")
            success = await self.pull_model(self.required_model)
            if not success:
                return False, f"Failed to pull model {self.required_model}"

        return True, "Ollama ready"

    def is_installed(self) -> bool:
        """Check if Ollama is installed on the system.

        Returns
        -------
        bool
            ``True`` if the ``ollama`` command is available on PATH,
            ``False`` otherwise.
        """
        return shutil.which("ollama") is not None

    async def is_daemon_running(self) -> bool:
        """Check if the Ollama daemon is running and responding.

        Returns
        -------
        bool
            ``True`` if the daemon responds to an API health check,
            ``False`` otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def has_model(self, model: str) -> bool:
        """Check if a specific model is available.

        Parameters
        ----------
        model:
            The model name to check (e.g. "nomic-embed-text").

        Returns
        -------
        bool
            ``True`` if the model is available, ``False`` otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    return any(model in m for m in models)
        except Exception:
            pass
        return False

    async def pull_model(self, model: str) -> bool:
        """Pull a model from the Ollama registry.

        Parameters
        ----------
        model:
            The model name to pull (e.g. "nomic-embed-text").

        Returns
        -------
        bool
            ``True`` if the pull succeeded, ``False`` otherwise.
        """
        try:
            proc = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                timeout=300,
            )
            return proc.returncode == 0
        except Exception:
            return False

    async def try_start_daemon(self) -> bool:
        """Attempt to start the Ollama daemon in the background.

        Spawns ``ollama serve`` as a detached background process and waits
        up to 10 seconds for it to become responsive.

        Returns
        -------
        bool
            ``True`` if the daemon started successfully and is responding,
            ``False`` otherwise.
        """
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for daemon to become responsive
            for _ in range(10):
                await asyncio.sleep(1)
                if await self.is_daemon_running():
                    return True
        except Exception:
            pass
        return False

    def _install_instructions(self) -> str:
        """Get platform-specific Ollama installation instructions.

        Returns
        -------
        str
            Installation instructions appropriate for the current platform.
        """
        system = platform.system()
        if system == "Darwin":
            return (
                "Ollama is not installed.\n\n"
                "Install via Homebrew:\n"
                "  brew install ollama\n\n"
                "Or download from:\n"
                "  https://ollama.com/download"
            )
        elif system == "Linux":
            return (
                "Ollama is not installed.\n\n"
                "Install via:\n"
                "  curl -fsSL https://ollama.com/install.sh | sh"
            )
        else:
            return (
                "Ollama is not installed.\n\n"
                "Download from:\n"
                "  https://ollama.com/download"
            )

    def _daemon_instructions(self) -> str:
        """Get instructions for starting the Ollama daemon.

        Returns
        -------
        str
            Instructions for starting the daemon manually or as a service.
        """
        system = platform.system()
        if system == "Darwin":
            return (
                "Ollama daemon is not running.\n\n"
                "Start manually:\n"
                "  ollama serve\n\n"
                "Or run as a service:\n"
                "  brew services start ollama"
            )
        else:
            return (
                "Ollama daemon is not running.\n\n"
                "Start manually:\n"
                "  ollama serve"
            )
