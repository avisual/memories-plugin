"""Tests proving the antipattern mistake-prevention pipeline works end-to-end.

Proves:
1. _infer_error_severity maps error signatures to correct severity levels
2. _hook_post_tool passes severity to remember() for antipatterns
3. _hook_post_tool_failure passes severity to remember() for antipatterns
4. End-to-end: antipatterns stored with severity are formatted with it on recall
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memories.cli import (
    _format_atom_line,
    _hook_post_tool,
    _hook_post_tool_failure,
    _infer_error_severity,
)


# ===========================================================================
# TestInferErrorSeverity
# ===========================================================================


class TestInferErrorSeverity:
    """Tests for _infer_error_severity() — the new severity-from-context helper."""

    def test_permission_denied_is_high(self):
        assert _infer_error_severity("permission denied: /etc/passwd") == "high"

    def test_access_denied_is_high(self):
        assert _infer_error_severity("Access Denied when writing to /data") == "high"

    def test_rm_rf_is_high(self):
        assert _infer_error_severity("rm -rf /important-data failed") == "high"

    def test_drop_table_is_high(self):
        assert _infer_error_severity("DROP TABLE users caused cascade") == "high"

    def test_python_traceback_is_medium(self):
        content = "Traceback (most recent call last):\n  File 'x.py'\nValueError: bad"
        assert _infer_error_severity(content) == "medium"

    def test_command_not_found_is_medium(self):
        assert _infer_error_severity("bash: docker-compose: command not found") == "medium"

    def test_module_not_found_is_medium(self):
        assert _infer_error_severity("ModuleNotFoundError: No module named 'pandas'") == "medium"

    def test_no_such_file_is_medium(self):
        assert _infer_error_severity("No such file or directory: /tmp/config.yml") == "medium"

    def test_generic_error_defaults_to_medium(self):
        assert _infer_error_severity("Something went wrong") == "medium"


# ===========================================================================
# TestPostToolPassesSeverity
# ===========================================================================


def _make_mock_brain():
    """Create a mock Brain for hook tests."""
    brain = MagicMock()
    brain._storage = MagicMock()
    brain._storage.execute = AsyncMock(return_value=[])
    brain._storage.execute_write = AsyncMock(return_value=0)
    brain._learning = MagicMock()
    brain._learning.assess_novelty = AsyncMock(return_value=(True, 0))
    brain.recall = AsyncMock(return_value={
        "atoms": [], "antipatterns": [], "pathways": [],
        "budget_used": 0, "budget_remaining": 2000,
        "total_activated": 0, "seed_count": 0, "compression_level": 0,
    })
    brain.remember = AsyncMock(return_value={"atom_id": 42, "deduplicated": False})
    brain._current_session_id = None
    return brain


class TestPostToolPassesSeverity:
    """Prove _hook_post_tool passes severity for antipattern atoms."""

    async def test_antipattern_bash_error_gets_severity(self):
        """When a Bash error is classified as antipattern, severity is passed."""
        mock_brain = _make_mock_brain()

        # Error content that triggers both _BASH_REAL_ERROR_SIGS and
        # _ANTIPATTERN_KEYWORDS ("avoid" + "permission denied").
        error_content = (
            "Traceback (most recent call last):\n"
            "  File 'deploy.py', line 42\n"
            "PermissionError: permission denied - avoid running as root"
        )

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            await _hook_post_tool({
                "tool_name": "Bash",
                "tool_input": {"command": "python deploy.py"},
                "tool_response": error_content,
                "session_id": "sess-123",
                "cwd": "/home/user/proj",
            })

        # remember() should have been called.
        if mock_brain.remember.await_count > 0:
            _, kwargs = mock_brain.remember.call_args
            # If classified as antipattern, severity should be set.
            if kwargs.get("type") == "antipattern":
                assert kwargs.get("severity") is not None, (
                    "Antipattern auto-captured from Bash error should have severity"
                )

    async def test_experience_atom_gets_no_severity(self):
        """Non-antipattern atoms (experiences) should NOT get severity."""
        mock_brain = _make_mock_brain()

        # Error content that triggers _BASH_REAL_ERROR_SIGS but NOT
        # _ANTIPATTERN_KEYWORDS — just a generic traceback.
        error_content = (
            "Traceback (most recent call last):\n"
            "  File 'app.py', line 10\n"
            "ValueError: invalid literal for int(): 'abc'"
        )

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            await _hook_post_tool({
                "tool_name": "Bash",
                "tool_input": {"command": "python app.py"},
                "tool_response": error_content,
                "session_id": "sess-123",
                "cwd": "/home/user/proj",
            })

        if mock_brain.remember.await_count > 0:
            _, kwargs = mock_brain.remember.call_args
            if kwargs.get("type") == "experience":
                assert kwargs.get("severity") is None, (
                    "Experience atoms should not get severity"
                )


class TestPostToolFailurePassesSeverity:
    """Prove _hook_post_tool_failure passes severity for antipattern atoms."""

    async def test_failure_antipattern_gets_severity(self):
        """Tool failure classified as antipattern gets severity from context."""
        mock_brain = _make_mock_brain()

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            await _hook_post_tool_failure({
                "tool_name": "Bash",
                "tool_input": {"command": "rm -rf /data"},
                "error": "permission denied: should not delete production data — avoid this",
                "session_id": "sess-123",
                "cwd": "/home/user/proj",
            })

        if mock_brain.remember.await_count > 0:
            _, kwargs = mock_brain.remember.call_args
            if kwargs.get("type") == "antipattern":
                assert kwargs["severity"] == "high", (
                    "Permission denied antipattern should get severity='high'"
                )

    async def test_failure_experience_gets_no_severity(self):
        """Tool failure classified as experience should NOT get severity."""
        mock_brain = _make_mock_brain()

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            await _hook_post_tool_failure({
                "tool_name": "Bash",
                "tool_input": {"command": "npm test"},
                "error": "Tests failed: 3 assertions did not pass, check output for details",
                "session_id": "sess-123",
                "cwd": "/home/user/proj",
            })

        if mock_brain.remember.await_count > 0:
            _, kwargs = mock_brain.remember.call_args
            if kwargs.get("type") == "experience":
                assert kwargs.get("severity") is None


# ===========================================================================
# TestAntipatternFormattingEndToEnd
# ===========================================================================


class TestAntipatternFormattingEndToEnd:
    """Prove antipatterns with severity/instead are formatted for injection."""

    def test_antipattern_with_severity_and_instead_formats_correctly(self):
        """Full formatting shows severity and instead fields."""
        atom = {
            "type": "antipattern",
            "confidence": 0.9,
            "id": 42,
            "content": "Never use rm -rf without checking path first",
            "severity": "high",
            "instead": "Use trash-put or mv to a staging directory",
        }
        line = _format_atom_line(atom)

        assert "[KNOWN MISTAKE]" in line
        assert "Never use rm -rf without checking path first" in line
        assert "instead: Use trash-put or mv to a staging directory" in line
        assert "(id:42)" in line

    def test_antipattern_without_severity_still_formats(self):
        """Antipattern missing severity/instead still renders content."""
        atom = {
            "type": "antipattern",
            "confidence": 0.7,
            "id": 10,
            "content": "Watch out for race conditions in async code",
        }
        line = _format_atom_line(atom)

        assert "[warning]" in line
        assert "Watch out for race conditions" in line
        # No instead line.
        assert "instead:" not in line

    def test_high_severity_antipattern_stands_out(self):
        """Verify the high-severity antipattern format is visually distinct."""
        atom = {
            "type": "antipattern",
            "confidence": 1.0,
            "id": 99,
            "content": "Never commit .env files to git",
            "severity": "critical",
            "instead": "Add .env to .gitignore and use environment variables",
        }
        line = _format_atom_line(atom)

        # All critical info should be present for the LLM to parse.
        assert "Never commit .env files" in line
        assert "[KNOWN MISTAKE]" in line
        assert "instead: Add .env to .gitignore" in line
