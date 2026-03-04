"""Tests for activity-aware rule surfacing in the pre-tool hook.

Covers:
- _detect_activity: pure function, no I/O
- _format_rules_reminder: pure function, no I/O
- _hook_pre_tool_use: integration with mocked Brain (activity recall path)
- _seed_project_rules: integration with real Brain (in-memory SQLite)
- run_seed_rules: CLI argument parsing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.cli import (
    _detect_activity,
    _format_rules_reminder,
    _hook_pre_tool_use,
    _seed_project_rules,
    _PROJECT_RULES,
    run_seed_rules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_brain(
    atoms: list | None = None,
    antipatterns: list | None = None,
) -> MagicMock:
    """Return a minimal mock Brain suitable for pre-tool hook tests."""
    brain = MagicMock()
    brain._storage = MagicMock()
    brain._storage.execute = AsyncMock(return_value=[])
    brain._storage.execute_write = AsyncMock(return_value=0)
    brain._storage.execute_many = AsyncMock(return_value=None)
    brain._learning = MagicMock()
    brain._learning.assess_novelty = AsyncMock(return_value=(True, 0))
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
# TestDetectActivity
# ===========================================================================


class TestDetectActivity:
    """Unit tests for _detect_activity() — pure function, no I/O."""

    # --- PR creation ---

    def test_bb_pr_sh_detected(self):
        assert _detect_activity("Bash", {"command": "~/.agent-messaging/bb-pr.sh 'My PR'"}) == "pr-creation"

    def test_git_push_detected_as_pr_creation(self):
        assert _detect_activity("Bash", {"command": "git push origin feature/my-branch"}) == "pr-creation"

    def test_bb_pr_sh_uppercase_ignored(self):
        # Pattern matching is case-insensitive on the command.
        assert _detect_activity("Bash", {"command": "BB-PR.SH create"}) == "pr-creation"

    # --- Commit ---

    def test_git_commit_detected(self):
        assert _detect_activity("Bash", {"command": "git commit -m 'feat: add widget'"}) == "commit"

    def test_git_commit_amend_detected(self):
        assert _detect_activity("Bash", {"command": "git commit --amend --no-edit"}) == "commit"

    # --- Deploy ---

    def test_deploy_keyword_detected(self):
        assert _detect_activity("Bash", {"command": "./scripts/deploy.sh production"}) == "deploy"

    def test_rsync_detected_as_deploy(self):
        assert _detect_activity("Bash", {"command": "rsync -avz dist/ user@host:/var/www/"}) == "deploy"

    # --- No match ---

    def test_read_file_not_an_activity(self):
        assert _detect_activity("Bash", {"command": "cat README.md"}) is None

    def test_go_test_not_an_activity(self):
        assert _detect_activity("Bash", {"command": "go test -race ./..."}) is None

    def test_empty_command_returns_none(self):
        assert _detect_activity("Bash", {"command": ""}) is None

    def test_non_bash_tool_returns_none(self):
        # Task, Edit, etc. should never produce an activity.
        assert _detect_activity("Task", {"prompt": "bb-pr.sh create"}) is None
        assert _detect_activity("Edit", {"command": "bb-pr.sh create"}) is None
        assert _detect_activity("Read", {}) is None

    # --- Input type handling ---

    def test_json_string_input_parsed(self):
        payload = json.dumps({"command": "git commit -m 'chore: update deps'"})
        assert _detect_activity("Bash", payload) == "commit"

    def test_invalid_json_string_returns_none(self):
        assert _detect_activity("Bash", "not valid json {{{") is None

    def test_non_dict_input_returns_none(self):
        assert _detect_activity("Bash", ["list", "not", "dict"]) is None

    def test_missing_command_key_returns_none(self):
        assert _detect_activity("Bash", {"description": "something"}) is None

    def test_none_command_value_returns_none(self):
        assert _detect_activity("Bash", {"command": None}) is None

    # --- Priority: first pattern wins ---

    def test_git_push_wins_over_deploy_if_both_present(self):
        # "git push" appears before "deploy" in _ACTIVITY_PATTERNS, so
        # a command containing both should return "pr-creation".
        cmd = "git push && deploy.sh"
        result = _detect_activity("Bash", {"command": cmd})
        assert result == "pr-creation"


# ===========================================================================
# TestFormatRulesReminder
# ===========================================================================


class TestFormatRulesReminder:
    """Unit tests for _format_rules_reminder() — pure function, no I/O."""

    def test_basic_output_structure(self):
        rules = [
            {"content": "Run tests before PR", "tags": ["rule", "pr-creation", "example"]},
        ]
        lines = _format_rules_reminder("example", "pr-creation", rules)
        text = "\n".join(lines)
        assert "<memories>" in text
        assert "</memories>" in text
        assert "PR CREATION" in text
        assert "example" in text
        assert "Run tests before PR" in text
        assert "MUST:" in text

    def test_empty_rules_returns_empty_list(self):
        assert _format_rules_reminder("example", "commit", []) == []

    def test_multiple_rules_all_rendered(self):
        rules = [
            {"content": "Rule one", "tags": ["rule"]},
            {"content": "Rule two", "tags": ["rule"]},
            {"content": "Rule three", "tags": ["rule"]},
        ]
        lines = _format_rules_reminder("myproject", "commit", rules)
        text = "\n".join(lines)
        assert "Rule one" in text
        assert "Rule two" in text
        assert "Rule three" in text

    def test_xml_special_chars_escaped(self):
        rules = [{"content": "Use <defer> & cancel()", "tags": ["rule"]}]
        lines = _format_rules_reminder("proj", "deploy", rules)
        text = "\n".join(lines)
        # Raw < and & must not appear in the XML block.
        assert "<defer>" not in text
        assert "&amp;" in text or "&lt;" in text

    def test_activity_uppercased_in_header(self):
        rules = [{"content": "Some rule", "tags": ["rule"]}]
        lines = _format_rules_reminder("proj", "pr-creation", rules)
        text = "\n".join(lines)
        assert "PR CREATION" in text

    def test_deploy_activity_uppercased(self):
        rules = [{"content": "Check env vars", "tags": ["rule"]}]
        lines = _format_rules_reminder("proj", "deploy", rules)
        text = "\n".join(lines)
        assert "DEPLOY" in text


# ===========================================================================
# TestHookPreToolUseActivityRules
# ===========================================================================


class TestHookPreToolUseActivityRules:
    """Integration tests for _hook_pre_tool_use() — activity-aware rule surfacing."""

    async def _invoke(self, data: dict, brain: MagicMock) -> str:
        """Invoke _hook_pre_tool_use with brain patched in."""
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)), \
             patch("memories.cli._record_hook_stat", AsyncMock()), \
             patch("memories.cli._save_hook_atoms", AsyncMock()):
            return await _hook_pre_tool_use(data)

    async def test_pr_creation_triggers_rule_recall(self):
        """When bb-pr.sh is detected, a rule-focused recall fires."""
        rule_atom = {
            "id": 10,
            "content": "MUST run tests before PR",
            "type": "skill",
            "tags": ["rule", "pr-creation", "example"],
            "confidence": 1.0,
        }
        # First recall (rules): returns the rule atom.
        # Second recall (antipatterns): returns nothing.
        brain = _make_mock_brain()
        rule_result = {
            "atoms": [rule_atom],
            "antipatterns": [],
            "pathways": [],
            "budget_used": 100,
            "budget_remaining": 700,
            "total_activated": 1,
            "seed_count": 1,
            "compression_level": 0,
        }
        empty_result = {
            "atoms": [],
            "antipatterns": [],
            "pathways": [],
            "budget_used": 0,
            "budget_remaining": 500,
            "total_activated": 0,
            "seed_count": 0,
            "compression_level": 0,
        }
        brain.recall = AsyncMock(side_effect=[rule_result, empty_result])

        data = {
            "tool_name": "Bash",
            "tool_input": {
                "command": "~/.agent-messaging/bb-pr.sh create --repo myapp --title 'feat: add widget'",
                "description": "Create a pull request for the feature branch",
            },
            "cwd": "/home/user/git/myapp",
            "session_id": "sess-001",
        }
        output = await self._invoke(data, brain)
        assert "MUST run tests before PR" in output
        assert "PR CREATION" in output

    async def test_rule_atoms_without_rule_tag_excluded(self):
        """Skill atoms missing the 'rule' tag are not surfaced as rules."""
        atom_no_rule_tag = {
            "id": 20,
            "content": "This is a general tip",
            "type": "skill",
            "tags": ["pr-creation", "example"],  # No "rule" tag
            "confidence": 1.0,
        }
        brain = _make_mock_brain()
        recall_with_untagged = {
            "atoms": [atom_no_rule_tag],
            "antipatterns": [],
            "pathways": [],
            "budget_used": 100,
            "budget_remaining": 700,
            "total_activated": 1,
            "seed_count": 1,
            "compression_level": 0,
        }
        empty_result = {
            "atoms": [],
            "antipatterns": [],
            "pathways": [],
            "budget_used": 0,
            "budget_remaining": 500,
            "total_activated": 0,
            "seed_count": 0,
            "compression_level": 0,
        }
        brain.recall = AsyncMock(side_effect=[recall_with_untagged, empty_result])

        data = {
            "tool_name": "Bash",
            "tool_input": {
                "command": "~/.agent-messaging/bb-pr.sh create --repo myapp --title 'feat: add widget'",
                "description": "Create a pull request for the feature branch",
            },
            "cwd": "/home/user/git/myapp",
            "session_id": "sess-002",
        }
        output = await self._invoke(data, brain)
        # The header should NOT appear because no atoms passed the tag filter.
        assert "PR CREATION" not in output

    async def test_no_activity_skips_rule_recall(self):
        """A plain Bash command (no activity pattern) does not fire rule recall."""
        brain = _make_mock_brain()
        brain.recall = AsyncMock(return_value={
            "atoms": [],
            "antipatterns": [],
            "pathways": [],
            "budget_used": 0,
            "budget_remaining": 500,
            "total_activated": 0,
            "seed_count": 0,
            "compression_level": 0,
        })

        data = {
            "tool_name": "Bash",
            "tool_input": {
                "command": "go test -race ./...",
                "description": "Run the full Go test suite with race detector enabled",
            },
            "cwd": "/home/user/git/myapp",
            "session_id": "sess-003",
        }
        await self._invoke(data, brain)
        # Only one recall should fire (the antipattern recall), not two.
        assert brain.recall.call_count == 1

    async def test_commit_activity_uses_correct_query(self):
        """Commit activity fires a recall with 'commit rules checklist' in the query."""
        brain = _make_mock_brain()
        empty_result = {
            "atoms": [],
            "antipatterns": [],
            "pathways": [],
            "budget_used": 0,
            "budget_remaining": 500,
            "total_activated": 0,
            "seed_count": 0,
            "compression_level": 0,
        }
        brain.recall = AsyncMock(return_value=empty_result)

        data = {
            "tool_name": "Bash",
            "tool_input": {
                "command": "git commit -m 'feat: implement widgets'",
                "description": "Commit the implementation changes with a descriptive message",
            },
            "cwd": "/home/user/git/myapp",
            "session_id": "sess-004",
        }
        await self._invoke(data, brain)

        # First call should be the rule recall for "commit".
        first_call_kwargs = brain.recall.call_args_list[0].kwargs
        assert "commit" in first_call_kwargs.get("query", "")
        assert "rules" in first_call_kwargs.get("query", "")
        assert first_call_kwargs.get("types") == ["skill"]

    async def test_non_bash_tool_skips_activity_detection(self):
        """Task tool is not subject to activity detection."""
        brain = _make_mock_brain()

        data = {
            "tool_name": "Task",
            "tool_input": {
                "prompt": "~/.agent-messaging/bb-pr.sh create --repo myapp --title 'feat: add widget'",
                "description": "Run a sub-agent to create a pull request",
                "subagent_type": "general-purpose",
            },
            "cwd": "/home/user/git/myapp",
            "session_id": "sess-005",
        }
        await self._invoke(data, brain)
        # No recall should fire for Task (no antipattern or rule recalls).
        assert brain.recall.call_count == 0

    async def test_no_cwd_skips_rule_recall(self):
        """Without cwd we can't determine a project — rule recall is skipped."""
        brain = _make_mock_brain()
        empty_result = {
            "atoms": [],
            "antipatterns": [],
            "pathways": [],
            "budget_used": 0,
            "budget_remaining": 500,
            "total_activated": 0,
            "seed_count": 0,
            "compression_level": 0,
        }
        brain.recall = AsyncMock(return_value=empty_result)

        data = {
            "tool_name": "Bash",
            "tool_input": {
                "command": "~/.agent-messaging/bb-pr.sh create --repo myapp --title 'fix: patch'",
                "description": "Create a pull request using GitHub CLI with a descriptive title",
            },
            "cwd": "",  # No cwd — project is None
            "session_id": "sess-006",
        }
        await self._invoke(data, brain)
        # Only one recall fires: the antipattern recall. Rule recall is skipped
        # because project is None.
        assert brain.recall.call_count == 1

    async def test_returns_empty_string_for_task_with_short_content(self):
        """Hook returns '' immediately when content is too short (< 20 chars)."""
        brain = _make_mock_brain()

        data = {
            "tool_name": "Bash",
            "tool_input": {"command": "ls", "description": "List files"},
            "cwd": "/home/user/git/myapp",
            "session_id": "sess-007",
        }
        output = await self._invoke(data, brain)
        # No recall should fire — content is too short.
        assert brain.recall.call_count == 0
        assert output == ""


# ===========================================================================
# TestSeedProjectRules
# ===========================================================================


@pytest.mark.integration
class TestSeedProjectRules:
    """Integration tests for _seed_project_rules() using a real in-memory Brain."""

    async def test_seed_fog_stores_all_rules(self, brain):
        """Seeding 'example' stores all defined rules across all activities."""
        total_expected = sum(
            len(rules) for rules in _PROJECT_RULES["example"].values()
        )

        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            result = await _seed_project_rules("example")

        assert result["project"] == "example"
        assert result["stored"] == total_expected
        assert result["skipped"] == 0
        assert "error" not in result

    async def test_seed_fog_atoms_have_correct_tags(self, brain):
        """Each stored rule atom carries 'rule', the activity, and project tags."""
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _seed_project_rules("example")

        # Query atoms directly from storage.
        rows = await brain._storage.execute(
            "SELECT tags FROM atoms WHERE region = 'project:example' AND type = 'skill'"
        )
        assert rows, "Expected skill atoms in project:example region"

        import json as _json
        for row in rows:
            tags = _json.loads(row["tags"] or "[]")
            assert "rule" in tags, f"'rule' tag missing from {tags}"
            assert "example" in tags, f"'example' tag missing from {tags}"

    async def test_seed_fog_atoms_have_high_importance(self, brain):
        """Rule atoms are stored with importance=0.95 so they survive decay."""
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _seed_project_rules("example")

        rows = await brain._storage.execute(
            "SELECT importance FROM atoms WHERE region = 'project:example' AND type = 'skill'"
        )
        for row in rows:
            assert row["importance"] >= 0.94, (
                f"Expected importance >= 0.95, got {row['importance']}"
            )

    async def test_seed_twice_skips_duplicates(self, brain):
        """Running seed-rules twice on the same project skips existing atoms."""
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            first = await _seed_project_rules("example")
            second = await _seed_project_rules("example")

        assert first["stored"] > 0
        assert second["stored"] == 0
        assert second["skipped"] == first["stored"]

    async def test_unknown_project_returns_error(self, brain):
        """Seeding an unknown project returns an error dict, not an exception."""
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            result = await _seed_project_rules("nonexistent_project")

        assert "error" in result
        assert result["stored"] == 0
        assert "nonexistent_project" in result["error"]

    async def test_seed_correct_region(self, brain):
        """Rules are stored in the project-scoped region."""
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _seed_project_rules("example")

        rows = await brain._storage.execute(
            "SELECT COUNT(*) AS cnt FROM atoms WHERE region = 'project:example'"
        )
        assert rows[0]["cnt"] > 0

    async def test_pr_creation_rules_tagged_correctly(self, brain):
        """PR-creation rules carry the 'pr-creation' activity tag."""
        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _seed_project_rules("example")

        rows = await brain._storage.execute(
            "SELECT tags FROM atoms WHERE region = 'project:example' AND type = 'skill'"
        )
        import json as _json
        pr_rules = [
            row for row in rows
            if "pr-creation" in _json.loads(row["tags"] or "[]")
        ]
        expected_pr_count = len(_PROJECT_RULES["example"]["pr-creation"])
        assert len(pr_rules) == expected_pr_count


# ===========================================================================
# TestRunSeedRulesCLI
# ===========================================================================


class TestRunSeedRulesCLI:
    """Unit tests for run_seed_rules() argument parsing."""

    def test_list_flag_prints_projects(self, capsys):
        """--list prints available projects without seeding."""
        run_seed_rules(["--list"])
        captured = capsys.readouterr()
        assert "example" in captured.out

    def test_missing_project_flag_exits(self):
        """Missing --project causes sys.exit(1)."""
        with pytest.raises(SystemExit) as exc_info:
            run_seed_rules([])
        assert exc_info.value.code == 1

    def test_unknown_project_exits(self):
        """Unknown project causes sys.exit(1) after calling _seed_project_rules."""
        async def _fake_seed(project: str):
            return {"project": project, "stored": 0, "skipped": 0, "error": "No rules for 'bad'"}

        with patch("memories.cli._seed_project_rules", _fake_seed), \
             pytest.raises(SystemExit) as exc_info:
            run_seed_rules(["--project", "bad"])
        assert exc_info.value.code == 1

    def test_successful_seed_prints_summary(self, capsys):
        """A successful seed prints stored/skipped counts."""
        async def _fake_seed(project: str):
            return {"project": project, "stored": 7, "skipped": 0}

        with patch("memories.cli._seed_project_rules", _fake_seed):
            run_seed_rules(["--project", "example"])

        captured = capsys.readouterr()
        assert "stored=7" in captured.out
        assert "skipped=0" in captured.out
