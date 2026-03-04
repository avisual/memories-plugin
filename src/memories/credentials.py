"""Optional credential retrieval via *pass* (password-store) with env-var fallback.

Currently all backends are local (Ollama), so no secrets are required.
This module future-proofs the system for remote embedding providers or
authenticated storage backends.

Usage::

    from memories.credentials import get_secret

    api_key = get_secret("openai_api_key")
"""

from __future__ import annotations

import logging
import shutil
import subprocess

log = logging.getLogger(__name__)


def has_pass() -> bool:
    """Return ``True`` if the ``pass`` CLI is installed and on ``$PATH``."""
    return shutil.which("pass") is not None


def _read_pass(name: str) -> str | None:
    """Try to read ``memories/{name}`` from the password store.

    Returns ``None`` on any failure (missing entry, gpg error, etc.) so that
    the caller can fall through to env-var lookup.
    """
    if not has_pass():
        return None

    entry = f"memories/{name}"
    try:
        result = subprocess.run(
            ["pass", "show", entry],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            value = result.stdout.strip()
            if value:
                log.debug("Loaded secret '%s' from pass", entry)
                return value
    except (subprocess.TimeoutExpired, FileNotFoundError):
        log.debug("pass lookup for '%s' failed or timed out", entry)

    return None


def get_secret(name: str) -> str | None:
    """Retrieve a secret by *name*.

    Resolution order:

    1. ``pass show memories/{name}``
    2. Environment variable ``MEMORIES_{NAME}`` (upper-cased, hyphens to
       underscores)

    Returns ``None`` if the secret cannot be found through either mechanism.

    Parameters
    ----------
    name:
        Logical secret name, e.g. ``"openai_api_key"``.
    """
    import os

    # 1. Try password store.
    value = _read_pass(name)
    if value is not None:
        return value

    # 2. Fall back to environment variable.
    env_key = f"MEMORIES_{name}".upper().replace("-", "_")
    value = os.environ.get(env_key)
    if value:
        log.debug("Loaded secret '%s' from env var %s", name, env_key)
        return value

    log.debug("Secret '%s' not found in pass or environment", name)
    return None
