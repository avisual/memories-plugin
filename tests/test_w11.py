"""Wave 11 — Bash error tightening + dead-branch / _SKIP_TOOLS cleanup.

Changes validated:
- HIGH: Bash false-positive rate eliminated — only genuine error signatures
  (Python traceback, command not found, permission denied, no such file) trigger
  atom creation.  Broad "error"/"failed" keyword matches suppressed.
- LOW L1: Dead Edit/Write/MultiEdit/NotebookEdit branch removed from _infer_atom_type.
- LOW L3: Edit/Write/MultiEdit/NotebookEdit moved into _SKIP_TOOLS (single skip list).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# HIGH: Bash error false-positive tightening
# ---------------------------------------------------------------------------


class TestBashRealErrors:
    """Genuine runtime errors must still be captured."""

    def test_python_traceback_captured(self):
        from memories.cli import _extract_tool_content

        resp = (
            "Traceback (most recent call last):\n"
            "  File 'x.py', line 5, in <module>\n"
            "ValueError: invalid literal for int() with base 10: 'abc'"
        )
        result = _extract_tool_content("Bash", {"command": "python x.py"}, resp)
        assert result is not None
        assert "produced error" in result

    def test_command_not_found_captured(self):
        from memories.cli import _extract_tool_content

        resp = "bash: foobar: command not found"
        result = _extract_tool_content("Bash", {"command": "foobar"}, resp)
        assert result is not None
        assert "produced error" in result

    def test_permission_denied_captured(self):
        from memories.cli import _extract_tool_content

        resp = "cat: /etc/shadow: Permission denied"
        result = _extract_tool_content("Bash", {"command": "cat /etc/shadow"}, resp)
        assert result is not None

    def test_no_such_file_captured(self):
        from memories.cli import _extract_tool_content

        resp = "cat: missing.txt: No such file or directory"
        result = _extract_tool_content("Bash", {"command": "cat missing.txt"}, resp)
        assert result is not None

    def test_traceback_case_insensitive(self):
        from memories.cli import _extract_tool_content

        resp = "TRACEBACK (MOST RECENT CALL LAST):\n  RuntimeError: oops"
        result = _extract_tool_content("Bash", {"command": "python bad.py"}, resp)
        assert result is not None


class TestBashFalsePositivesSuppressed:
    """Outputs that merely *contain* the word 'error' must not be captured."""

    def test_readme_containing_word_error_not_captured(self):
        from memories.cli import _extract_tool_content

        resp = "# memories\n\nThis module handles error cases gracefully.\n"
        result = _extract_tool_content("Bash", {"command": "cat README.md"}, resp)
        assert result is None

    def test_grep_output_with_error_not_captured(self):
        from memories.cli import _extract_tool_content

        resp = "14: error handling code\n32: error recovery path\n"
        result = _extract_tool_content("Bash", {"command": "grep error src/"}, resp)
        assert result is None

    def test_passed_tests_with_zero_failed_not_captured(self):
        from memories.cli import _extract_tool_content

        resp = "12 passed, 0 failed in 2.34s"
        result = _extract_tool_content("Bash", {"command": "pytest"}, resp)
        assert result is None

    def test_stderr_word_not_captured(self):
        from memories.cli import _extract_tool_content

        # "stderr" contains "error" as a substring — old code captured this
        resp = "Output written to stderr\n"
        result = _extract_tool_content("Bash", {"command": "make 2>&1"}, resp)
        assert result is None

    def test_word_failed_in_log_not_captured(self):
        from memories.cli import _extract_tool_content

        resp = "Build step 'compile' failed. Check logs above."
        result = _extract_tool_content("Bash", {"command": "make"}, resp)
        assert result is None

    def test_word_error_in_json_output_not_captured(self):
        from memories.cli import _extract_tool_content

        resp = '{"status": "ok", "last_error": null}'
        result = _extract_tool_content("Bash", {"command": "curl api"}, resp)
        assert result is None

    def test_empty_response_not_captured(self):
        from memories.cli import _extract_tool_content

        result = _extract_tool_content("Bash", {"command": "echo hi"}, "")
        assert result is None

    def test_successful_ls_not_captured(self):
        from memories.cli import _extract_tool_content

        result = _extract_tool_content("Bash", {"command": "ls"}, "cli.py\nconfig.py\n")
        assert result is None


# ---------------------------------------------------------------------------
# LOW L3: Edit/Write tools must be in _SKIP_TOOLS
# ---------------------------------------------------------------------------


class TestEditWriteInSkipTools:
    """Edit/Write/MultiEdit/NotebookEdit must live in _SKIP_TOOLS (single skip list)."""

    def test_edit_in_skip_tools(self):
        from memories.cli import _SKIP_TOOLS

        assert "Edit" in _SKIP_TOOLS

    def test_write_in_skip_tools(self):
        from memories.cli import _SKIP_TOOLS

        assert "Write" in _SKIP_TOOLS

    def test_multiedit_in_skip_tools(self):
        from memories.cli import _SKIP_TOOLS

        assert "MultiEdit" in _SKIP_TOOLS

    def test_notebookedit_in_skip_tools(self):
        from memories.cli import _SKIP_TOOLS

        assert "NotebookEdit" in _SKIP_TOOLS

    def test_extract_tool_content_edit_still_returns_none(self):
        """Redundant branch in _extract_tool_content removed, but None still returned
        via _SKIP_TOOLS early-exit in _hook_post_tool.  Direct call returns None."""
        from memories.cli import _extract_tool_content

        result = _extract_tool_content("Edit", {"file_path": "x.py"}, "")
        assert result is None

    def test_extract_tool_content_write_still_returns_none(self):
        from memories.cli import _extract_tool_content

        result = _extract_tool_content("Write", {"file_path": "x.py", "content": "x"}, "")
        assert result is None


# ---------------------------------------------------------------------------
# LOW L1: Dead branch removed from _infer_atom_type
# ---------------------------------------------------------------------------


class TestInferAtomTypeDeadBranchRemoved:
    """Edit/Write/MultiEdit/NotebookEdit must not appear in _infer_atom_type source."""

    def test_infer_atom_type_has_no_edit_write_branch(self):
        import inspect

        from memories.cli import _infer_atom_type

        src = inspect.getsource(_infer_atom_type)
        # The dead branch referencing edit tools should be gone
        assert '"Write", "Edit", "MultiEdit", "NotebookEdit"' not in src
        assert "Write', 'Edit', 'MultiEdit', 'NotebookEdit'" not in src
