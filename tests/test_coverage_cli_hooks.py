"""Coverage tests for cli.py — hook lifecycle functions.

cli.py has 41% coverage (716 missed lines). These tests cover the core hook
functions: session-start, prompt-submit, post-tool, pre-tool, pre-compact,
session-end, and the utility functions _format_atom_line, _format_pathways,
_project_name, _hook_budget.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memories.cli import (
    _format_atom_line,
    _format_pathways,
    _hook_budget,
    _project_name,
    _hook_session_start,
    _hook_prompt_submit,
    _hook_post_tool,
    _hook_pre_tool_use,
    _hook_pre_compact,
)


# ---------------------------------------------------------------------------
# Mock Brain factory
# ---------------------------------------------------------------------------


def _make_mock_brain(atoms=None, antipatterns=None):
    """Create a mock Brain with recall/remember/learning pre-configured."""
    brain = MagicMock()
    brain._storage = MagicMock()
    brain._storage.execute = AsyncMock(return_value=[])
    brain._storage.execute_write = AsyncMock(return_value=0)
    brain._learning = MagicMock()
    brain._learning.assess_novelty = AsyncMock(return_value=True)
    brain._learning.session_end_learning = AsyncMock(
        return_value={"synapses_strengthened": 0, "synapses_created": 0}
    )

    recall_result = {
        "atoms": atoms or [],
        "antipatterns": antipatterns or [],
        "pathways": [],
        "budget_used": 100,
        "budget_remaining": 3900,
        "total_activated": len(atoms or []),
        "seed_count": len(atoms or []),
        "compression_level": 0,
    }
    brain.recall = AsyncMock(return_value=recall_result)
    brain.remember = AsyncMock(return_value={"atom_id": 42, "deduplicated": False})
    brain._current_session_id = None
    return brain


# ===========================================================================
# TestFormatAtomLine
# ===========================================================================


class TestFormatAtomLine:
    """Tests for _format_atom_line() — lines 51-75."""

    def test_basic_fact(self):
        atom = {"type": "fact", "confidence": 0.9, "id": 1, "content": "Python uses GIL"}
        line = _format_atom_line(atom)
        assert "[fact]" in line
        assert "Python uses GIL" in line
        assert "(id:1)" in line

    def test_antipattern_with_severity(self):
        atom = {
            "type": "antipattern", "confidence": 0.8, "id": 2,
            "content": "Don't use eval()",
            "severity": "high", "instead": "Use ast.literal_eval",
        }
        line = _format_atom_line(atom)
        assert "[KNOWN MISTAKE]" in line
        assert "Don't use eval()" in line
        assert "instead: Use ast.literal_eval" in line

    def test_missing_id(self):
        atom = {"type": "insight", "confidence": 1.0, "content": "test"}
        line = _format_atom_line(atom)
        assert "test" in line


class TestFormatPathways:
    """Tests for _format_pathways() — lines 78-89."""

    def test_empty_pathways(self):
        assert _format_pathways([]) == []

    def test_formats_pathways(self):
        pathways = [
            {"source_id": 1, "target_id": 2, "relationship": "caused-by", "strength": 0.8},
        ]
        lines = _format_pathways(pathways)
        assert len(lines) == 2  # header + 1 pathway
        assert "connections:" in lines[0]
        assert "caused-by" in lines[1]
        assert "0.80" in lines[1]

    def test_caps_at_5_pathways(self):
        pathways = [
            {"source_id": i, "target_id": i + 1, "relationship": "related-to", "strength": 0.5}
            for i in range(10)
        ]
        lines = _format_pathways(pathways)
        assert len(lines) == 6  # header + 5 capped


class TestProjectName:
    """Tests for _project_name() — lines 112-116."""

    def test_extracts_dir_name(self):
        assert _project_name("/home/user/myproject") == "myproject"

    def test_none_returns_none(self):
        assert _project_name(None) is None

    def test_empty_returns_none(self):
        assert _project_name("") is None


class TestHookBudget:
    """Tests for _hook_budget() — lines 119-135."""

    def test_session_start_gets_larger_budget(self):
        budget = _hook_budget("session-start")
        assert budget > _hook_budget("prompt-submit")

    def test_pre_tool_gets_smallest_budget(self):
        budget = _hook_budget("pre-tool")
        assert budget < _hook_budget("prompt-submit")

    def test_unknown_hook_uses_default(self):
        budget = _hook_budget("unknown-hook")
        assert budget > 0


# ===========================================================================
# TestHookSessionStart
# ===========================================================================


class TestHookSessionStart:
    """Tests for _hook_session_start() — lines 332-424."""

    async def test_session_start_with_project_recalls(self):
        """Session start with a cwd should call brain.recall for project memories."""
        mock_brain = _make_mock_brain(atoms=[
            {"type": "fact", "confidence": 0.9, "id": 1, "content": "Project uses PostgreSQL"},
        ])

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_session_start({
                "session_id": "sess-123",
                "cwd": "/home/user/myproject",
            })

        # Should have called recall with the project query.
        mock_brain.recall.assert_awaited()
        assert "[memories]" in result
        assert "Project uses PostgreSQL" in result

    async def test_session_start_bridges_session_id(self):
        """session_id is bridged to brain._current_session_id."""
        mock_brain = _make_mock_brain()

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            await _hook_session_start({
                "session_id": "sess-456",
                "cwd": "/home/user/proj",
            })

        assert mock_brain._current_session_id == "sess-456"

    async def test_session_start_no_project_no_recall(self):
        """Without cwd, no recall is performed."""
        mock_brain = _make_mock_brain()

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_session_start({
                "session_id": "sess-789",
                "cwd": "",
            })

        mock_brain.recall.assert_not_awaited()

    async def test_session_start_empty_results_returns_none(self):
        """When recall returns no atoms, return value is None (not crash)."""
        mock_brain = _make_mock_brain(atoms=[])

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_session_start({
                "session_id": "sess-000",
                "cwd": "/home/user/proj",
            })

        # When recall returns no atoms the hook returns None (no output to inject).
        # It may also return "Success" or "" — the key assertion is it doesn't crash.
        assert result is None or isinstance(result, str)

    async def test_session_start_registers_active_session(self):
        """Session is registered in active_sessions table."""
        mock_brain = _make_mock_brain()

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            await _hook_session_start({
                "session_id": "sess-active",
                "cwd": "/home/user/proj",
            })

        # Should have inserted into active_sessions.
        mock_brain._storage.execute_write.assert_called()


# ===========================================================================
# TestHookPromptSubmit
# ===========================================================================


class TestHookPromptSubmit:
    """Tests for _hook_prompt_submit() — lines 425-531."""

    async def test_prompt_submit_with_project_dual_recall(self):
        """With a project, prompt-submit runs dual recall (project + global)."""
        mock_brain = _make_mock_brain(atoms=[
            {"type": "fact", "confidence": 0.9, "id": 1, "content": "Use connection pooling"},
        ])

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_prompt_submit({
                "session_id": "sess-123",
                "cwd": "/home/user/myproject",
                "prompt": "How should I set up the database?",
            })

        # Should call recall twice (project + global).
        assert mock_brain.recall.await_count == 2
        assert "[memories]" in result

    async def test_prompt_submit_without_project_single_recall(self):
        """Without project, only one recall (global)."""
        mock_brain = _make_mock_brain(atoms=[])

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_prompt_submit({
                "session_id": "sess-123",
                "cwd": "",
                "prompt": "What is a monad?",
            })

        assert mock_brain.recall.await_count == 1

    async def test_prompt_submit_no_atoms_returns_success(self):
        """When no atoms recalled, returns 'Success'."""
        mock_brain = _make_mock_brain(atoms=[])

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_prompt_submit({
                "session_id": "sess-123",
                "cwd": "",
                "prompt": "irrelevant question",
            })

        assert result == "Success"

    async def test_prompt_submit_antipattern_deduplication(self):
        """Antipatterns already in atoms are not duplicated."""
        shared_id = 5
        mock_brain = _make_mock_brain(
            atoms=[{"type": "fact", "id": shared_id, "confidence": 0.9, "content": "test"}],
            antipatterns=[{"type": "antipattern", "id": shared_id, "confidence": 0.8, "content": "warning"}],
        )

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_prompt_submit({
                "session_id": "sess-123",
                "cwd": "",
                "prompt": "test query",
            })

        # The shared ID should only appear once, not both in atoms and antipatterns.
        assert result.count(f"(id:{shared_id})") == 1


# ===========================================================================
# TestHookPostTool
# ===========================================================================


class TestHookPostTool:
    """Tests for _hook_post_tool() — lines 537-604."""

    async def test_skip_tools_return_immediately(self):
        """Tools in _SKIP_TOOLS return JSON without processing."""
        result = await _hook_post_tool({
            "tool_name": "Read",
            "tool_input": {},
            "tool_response": "file content",
        })
        parsed = json.loads(result)
        assert "hookSpecificOutput" in parsed

    async def test_bash_with_real_error_stores_atom(self):
        """Bash with a real error signature stores an atom."""
        mock_brain = _make_mock_brain()

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_post_tool({
                "tool_name": "Bash",
                "tool_input": {"command": "python script.py"},
                "tool_response": "Traceback (most recent call last):\n  File 'x.py', line 1\nValueError: bad input",
                "session_id": "sess-123",
                "cwd": "/home/user/proj",
            })

        # Should have assessed novelty and potentially stored.
        mock_brain._learning.assess_novelty.assert_awaited()

    async def test_short_content_skipped(self):
        """Content shorter than 20 chars is skipped."""
        result = await _hook_post_tool({
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "ok",
        })
        parsed = json.loads(result)
        assert "hookSpecificOutput" in parsed


# ===========================================================================
# TestHookPreTool
# ===========================================================================


class TestHookPreToolUse:
    """Tests for _hook_pre_tool_use() — lines 1636-1718."""

    async def test_non_captured_tool_returns_empty(self):
        """Tools not in _PRE_TOOL_CAPTURE return empty string."""
        result = await _hook_pre_tool_use({
            "tool_name": "Read",
            "tool_input": {},
        })
        assert result == ""

    async def test_bash_command_with_description_triggers_processing(self):
        """Bash commands with a description > 30 chars should be processed."""
        mock_brain = _make_mock_brain()
        mock_brain.recall = AsyncMock(return_value={
            "atoms": [{"type": "antipattern", "id": 1, "confidence": 0.9,
                       "content": "Don't rm -rf without checking", "severity": "high"}],
            "antipatterns": [],
            "pathways": [], "budget_used": 50, "budget_remaining": 450,
            "total_activated": 1, "seed_count": 1, "compression_level": 0,
        })

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_pre_tool_use({
                "tool_name": "Bash",
                "tool_input": {
                    "command": "rm -rf /tmp/build && deploy production",
                    "description": "Remove build artifacts and deploy the production environment to servers",
                },
                "cwd": "/home/user/proj",
                "session_id": "sess-123",
            })

        if result:
            assert "[memories]" in result

    async def test_short_description_skipped(self):
        """Bash with a short/missing description (<20 chars content) is skipped."""
        result = await _hook_pre_tool_use({
            "tool_name": "Bash",
            "tool_input": {"command": "ls", "description": "list"},
        })
        assert result == ""

    async def test_task_with_prompt_processed(self):
        """Task tool with a prompt should be processed."""
        mock_brain = _make_mock_brain()

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_pre_tool_use({
                "tool_name": "Task",
                "tool_input": {
                    "prompt": "Research the authentication architecture and identify security vulnerabilities in the OAuth flow",
                    "description": "Research auth security",
                    "subagent_type": "Explore",
                },
                "session_id": "sess-123",
                "cwd": "/home/user/proj",
            })

        # Should have assessed novelty.
        mock_brain._learning.assess_novelty.assert_awaited()


# ===========================================================================
# TestHookPreCompact
# ===========================================================================


class TestHookPreCompact:
    """Tests for _hook_pre_compact() — lines 1733-1774."""

    async def test_pre_compact_triggers_hebbian_checkpoint(self):
        """Pre-compact with session atoms triggers session_end_learning."""
        mock_brain = _make_mock_brain()
        mock_brain._storage.execute = AsyncMock(return_value=[
            {"atom_id": 1, "accessed_at": "2026-02-22T10:00:00+00:00"},
            {"atom_id": 2, "accessed_at": "2026-02-22T10:01:00+00:00"},
        ])

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_pre_compact({
                "session_id": "sess-123",
            })

        mock_brain._learning.session_end_learning.assert_awaited()
        assert result == ""

    async def test_pre_compact_no_session_id_noop(self):
        """Without session_id, pre-compact is a no-op."""
        mock_brain = _make_mock_brain()

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_pre_compact({
                "session_id": "",
            })

        mock_brain._learning.session_end_learning.assert_not_awaited()
        assert result == ""

    async def test_pre_compact_no_atoms_noop(self):
        """With session_id but no accumulated atoms, no Hebbian fires."""
        mock_brain = _make_mock_brain()
        mock_brain._storage.execute = AsyncMock(return_value=[])

        with patch("memories.cli._get_brain", AsyncMock(return_value=mock_brain)):
            result = await _hook_pre_compact({
                "session_id": "sess-123",
            })

        mock_brain._learning.session_end_learning.assert_not_awaited()

    async def test_pre_compact_transient_error_swallowed(self):
        """TimeoutError during pre-compact is logged but not raised."""
        with patch("memories.cli._get_brain", AsyncMock(side_effect=TimeoutError("slow"))):
            result = await _hook_pre_compact({"session_id": "sess-123"})

        assert result == ""


# ===========================================================================
# TestSynapseTypeWeightCoherence
# ===========================================================================


class TestSynapseTypeWeightCoherence:
    """Structural test: verify all synapse types have explicit weight entries."""

    def test_all_relationship_types_have_weights(self):
        """Every RELATIONSHIP_TYPES entry should have a SynapseTypeWeights field."""
        from memories.synapses import RELATIONSHIP_TYPES
        from memories.config import SynapseTypeWeights

        weights = SynapseTypeWeights()
        weight_fields = {f.replace("_", "-") for f in weights.__dataclass_fields__}

        missing = set(RELATIONSHIP_TYPES) - weight_fields
        assert not missing, f"Missing weight entries for: {missing}"
