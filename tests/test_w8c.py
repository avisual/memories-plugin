"""Wave 8-C — Antipattern classification guard.

Atoms should only be classified as ``antipattern`` when the content contains
genuine negative-assertion vocabulary (e.g. "avoid", "never use", "bad practice").
Content that merely mentions "error" or "failed" (like command output or status
reports) should be classified as ``experience`` instead.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Tests for _content_looks_like_antipattern guard
# ---------------------------------------------------------------------------


class TestContentLooksLikeAntipattern:
    """Unit tests for the keyword guard function."""

    def test_true_antipattern_content_avoid(self):
        from memories.cli import _content_looks_like_antipattern

        assert _content_looks_like_antipattern(
            "You should avoid using mutable default arguments in Python"
        )

    def test_true_antipattern_content_never_use(self):
        from memories.cli import _content_looks_like_antipattern

        assert _content_looks_like_antipattern(
            "Never use eval() on untrusted input"
        )

    def test_true_antipattern_content_bad_practice(self):
        from memories.cli import _content_looks_like_antipattern

        assert _content_looks_like_antipattern(
            "Storing passwords in plaintext is a bad practice"
        )

    def test_true_antipattern_content_should_not(self):
        from memories.cli import _content_looks_like_antipattern

        assert _content_looks_like_antipattern(
            "You should not commit .env files to version control"
        )

    def test_true_antipattern_content_pitfall(self):
        from memories.cli import _content_looks_like_antipattern

        assert _content_looks_like_antipattern(
            "A common pitfall with async/await is forgetting to await coroutines"
        )

    def test_true_antipattern_content_footgun(self):
        from memories.cli import _content_looks_like_antipattern

        assert _content_looks_like_antipattern(
            "This API is a footgun because it silently truncates data"
        )

    def test_true_antipattern_content_mistake(self):
        from memories.cli import _content_looks_like_antipattern

        assert _content_looks_like_antipattern(
            "A common mistake is to use == instead of is for None checks"
        )

    def test_true_antipattern_content_dont(self):
        from memories.cli import _content_looks_like_antipattern

        assert _content_looks_like_antipattern(
            "Don't use wildcard imports in production code"
        )

    def test_true_antipattern_content_bug(self):
        from memories.cli import _content_looks_like_antipattern

        assert _content_looks_like_antipattern(
            "This creates a subtle bug where the list is shared between calls"
        )

    def test_false_for_simple_error_output(self):
        from memories.cli import _content_looks_like_antipattern

        assert not _content_looks_like_antipattern(
            "Command `pytest tests/` produced error: ModuleNotFoundError: No module named 'foo'"
        )

    def test_false_for_edit_summary(self):
        from memories.cli import _content_looks_like_antipattern

        assert not _content_looks_like_antipattern(
            "A single line of code was edited"
        )

    def test_false_for_command_failure_report(self):
        from memories.cli import _content_looks_like_antipattern

        assert not _content_looks_like_antipattern(
            "The command failed due to a syntax error in the script"
        )

    def test_false_for_file_update_summary(self):
        from memories.cli import _content_looks_like_antipattern

        assert not _content_looks_like_antipattern(
            "The file has been updated with the new configuration"
        )

    def test_false_for_traceback_output(self):
        from memories.cli import _content_looks_like_antipattern

        assert not _content_looks_like_antipattern(
            "Traceback (most recent call last): File 'main.py', line 5"
        )

    def test_false_for_status_report(self):
        from memories.cli import _content_looks_like_antipattern

        assert not _content_looks_like_antipattern(
            "The build process completed with 3 warnings and 0 errors"
        )

    def test_case_insensitive(self):
        from memories.cli import _content_looks_like_antipattern

        assert _content_looks_like_antipattern(
            "You SHOULD NEVER commit secrets to Git"
        )

    def test_empty_string(self):
        from memories.cli import _content_looks_like_antipattern

        assert not _content_looks_like_antipattern("")


# ---------------------------------------------------------------------------
# Tests for _infer_atom_type with guard applied
# ---------------------------------------------------------------------------


class TestInferAtomTypeGuard:
    """The guard should prevent _infer_atom_type from returning 'antipattern'
    when content lacks negative-assertion vocabulary."""

    def test_bash_error_without_antipattern_vocab_returns_experience(self):
        from memories.cli import _infer_atom_type

        # A command that produced an error, but the content is just a status report
        result = _infer_atom_type(
            tool_name="Bash",
            tool_input={"command": "python main.py"},
            tool_response="ModuleNotFoundError: No module named 'requests'",
        )
        assert result == "experience", (
            f"Expected 'experience' but got '{result}' — "
            "error output without negative-assertion vocab should not be antipattern"
        )

    def test_bash_error_with_antipattern_vocab_returns_antipattern(self):
        from memories.cli import _infer_atom_type

        result = _infer_atom_type(
            tool_name="Bash",
            tool_input={"command": "python main.py"},
            tool_response="Error: you should never use eval() on user input — this is dangerous",
        )
        assert result == "antipattern"

    def test_bash_traceback_without_antipattern_vocab_returns_experience(self):
        from memories.cli import _infer_atom_type

        result = _infer_atom_type(
            tool_name="Bash",
            tool_input={"command": "pytest"},
            tool_response="Traceback (most recent call last):\n  File 'test.py', line 3",
        )
        assert result == "experience"

    def test_edit_tool_in_skip_tools(self):
        """Edit tools are in _SKIP_TOOLS — _infer_atom_type is never called for them."""
        from memories.cli import _SKIP_TOOLS

        assert "Edit" in _SKIP_TOOLS
        assert "Write" in _SKIP_TOOLS

    def test_bash_success_returns_fact(self):
        """Successful Bash output without errors should still be 'fact'."""
        from memories.cli import _infer_atom_type

        result = _infer_atom_type(
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_response="file1.py\nfile2.py",
        )
        assert result == "fact"


# ---------------------------------------------------------------------------
# Tests for _extract_response_learnings with guard applied
# ---------------------------------------------------------------------------


class TestExtractResponseLearningsGuard:
    """The guard should prevent _extract_response_learnings from classifying
    sentences as antipattern when they lack negative-assertion vocabulary."""

    def test_error_mention_without_vocab_becomes_experience(self):
        from memories.cli import _extract_response_learnings

        # "error:" triggers error_indicators — but no antipattern vocab
        response = (
            "error: the configuration file was missing from the expected directory location on the server"
        )
        learnings = _extract_response_learnings(response)
        assert len(learnings) >= 1
        content, atom_type = learnings[0]
        assert atom_type == "experience", (
            f"Expected 'experience' but got '{atom_type}' — "
            "simple error mention without negative vocabulary should not be antipattern"
        )

    def test_error_mention_with_vocab_stays_antipattern(self):
        from memories.cli import _extract_response_learnings

        response = (
            "Error: you should never use rm -rf / without checking the path first, this is a dangerous mistake"
        )
        learnings = _extract_response_learnings(response)
        assert len(learnings) >= 1
        content, atom_type = learnings[0]
        assert atom_type == "antipattern"

    def test_problem_mention_without_vocab_becomes_experience(self):
        from memories.cli import _extract_response_learnings

        # "problem:" triggers error_indicators but no antipattern vocab
        response = (
            "problem: the database connection pool was exhausted after too many concurrent requests to the server"
        )
        learnings = _extract_response_learnings(response)
        assert len(learnings) >= 1
        content, atom_type = learnings[0]
        assert atom_type == "experience"

    def test_issue_mention_without_vocab_becomes_experience(self):
        from memories.cli import _extract_response_learnings

        # "issue:" triggers error_indicators, but content has no antipattern vocab
        response = (
            "issue: the container image was built with an outdated base image, causing library incompatibilities at runtime"
        )
        learnings = _extract_response_learnings(response)
        assert len(learnings) >= 1
        content, atom_type = learnings[0]
        assert atom_type == "experience"
