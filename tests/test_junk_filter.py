"""Tests for the _is_junk() quality gate in the recording pipeline."""

from __future__ import annotations

import pytest

from memories.cli import _is_junk, _JUNK_PATTERNS


class TestIsJunkStaticPatterns:
    """Test static junk pattern matching."""

    @pytest.mark.parametrize(
        "content,expected_reason",
        [
            # Tool narration / self-talk
            ("Delegated to Explore sub-agent for codebase search", "tool_narration"),
            ("Let me now check the git status", "self_talk"),
            ("Agent team teammate 'researcher' went idle: checking files", "team_noise"),
            ("I have verified the changes are correct", "self_talk"),
            ("I've verified that all tests pass", "self_talk"),
            ("The plan is ready for execution", "self_talk"),
            ("Please submit it for your approval", "self_talk"),
            ("Let me check if the file exists", "self_talk"),
            ("I will now implement the changes", "self_talk"),
            ("I'll now run the test suite", "self_talk"),
            # Meta-noise
            ("Note: I was only able to extract 2 key insights from the data", "meta_noise"),
            ("The atomic fact was extracted from reasoning", "meta_noise"),
            ("This is an atomic process that runs daily", "meta_noise"),
            ("Note that there is only one instance of this", "meta_noise"),
            # Generic truisms
            ("A generic framework is necessary for this kind of work", "truism"),
            ("Seamless integration of various components is important", "truism"),
            ("This approach works without compromising the user experience", "truism"),
            # Edit descriptions
            ("The file has been edited to include the new function", "edit_description"),
            ("The line of code was edited to fix the bug", "edit_description"),
            ("A single line of code was changed in the module", "edit_description"),
            # Stdout narration
            ("The stdout value is 0 which indicates success", "stdout_narration"),
            # Agent task prompts
            ("[Explore] Extract src/ core interfaces: Explore the directory comprehensively", "agent_task_prompt"),
            ("[general-purpose] Query MongoDB presence data: I need to investigate tokens", "agent_task_prompt"),
            # Monitoring noise
            ("Wait 3 minutes then check progress - check 12 with long description here", "monitoring_noise"),
            # Messaging noise
            ("Send message via agent-messaging/memories-to-team.sh to coordinate work", "messaging_noise"),
            # Extended self-talk
            ("Let me analyze this plan and break it into components for implementation", "self_talk"),
            ("Now let me read the remaining call sites I need for the implementation", "self_talk"),
            ("Now I need to read more of cli.py to find the brain.remember call sites", "self_talk"),
        ],
    )
    def test_static_patterns_detected(self, content: str, expected_reason: str):
        is_junk, reason = _is_junk(content, "experience")
        assert is_junk, f"Expected junk: {content!r}"
        assert reason == expected_reason

    def test_all_static_patterns_have_reason_codes(self):
        """Every static pattern must have a non-empty reason code."""
        for pattern, reason in _JUNK_PATTERNS:
            assert pattern, "Empty pattern found"
            assert reason, f"Empty reason for pattern: {pattern!r}"


class TestIsJunkTooShort:
    """Test the short-content gate for fact/insight types."""

    def test_short_fact_rejected(self):
        is_junk, reason = _is_junk("Short fact", "fact")
        assert is_junk
        assert reason == "too_short"

    def test_short_insight_rejected(self):
        is_junk, reason = _is_junk("Brief insight here.", "insight")
        assert is_junk
        assert reason == "too_short"

    def test_very_short_experience_rejected_by_length(self):
        """Experience atoms under 30 chars are rejected as too_short."""
        content = "Short exp"
        is_junk, reason = _is_junk(content, "experience")
        assert is_junk
        assert reason == "too_short"

    def test_moderate_experience_not_rejected_by_length(self):
        """Experience atoms over 30 chars pass the length gate."""
        content = "Completed: Fix data pipeline routing"
        is_junk, reason = _is_junk(content, "experience")
        # 36 chars — above the 30-char floor, not rejected for length
        assert not is_junk or reason != "too_short"

    def test_long_fact_not_rejected_for_length(self):
        content = "The database schema uses composite primary keys on the hook_session_atoms table for deduplication."
        is_junk, reason = _is_junk(content, "fact")
        assert not is_junk


class TestIsJunkGenericOpener:
    """Test the generic opener regex pattern."""

    def test_generic_system_opener(self):
        # Use "experience" type to avoid the too_short check for facts
        content = "The system processes incoming data in batches from the queue every five minutes on schedule."
        is_junk, reason = _is_junk(content, "experience")
        assert is_junk
        assert reason == "generic_opener"

    def test_generic_process_opener(self):
        content = "A process runs on the server handling the incoming webhook requests periodically."
        is_junk, reason = _is_junk(content, "experience")
        assert is_junk
        assert reason == "generic_opener"

    def test_generic_function_opener(self):
        content = "The function returns a boolean value indicating whether the operation completed."
        is_junk, reason = _is_junk(content, "experience")
        assert is_junk
        assert reason == "generic_opener"

    def test_bash_command_description(self):
        content = 'Check for batch send success/failure messages (`ssh -i ~/.ssh/key pi@host "grep pattern"`)'
        is_junk, reason = _is_junk(content, "experience")
        assert is_junk
        assert reason == "bash_command_description"

    def test_bash_command_find(self):
        content = 'Find all files in project package (`find /Users/dev/git/myapp -name "*.go" | head -20`)'
        is_junk, reason = _is_junk(content, "experience")
        assert is_junk
        assert reason == "bash_command_description"

    def test_bash_command_with_tilde(self):
        content = 'Check log last modification time (`stat -c "%Y" /tmp/scan.log && date +%s`)'
        is_junk, reason = _is_junk(content, "experience")
        assert is_junk
        assert reason == "bash_command_description"

    def test_content_with_backticks_in_middle_not_rejected(self):
        """Content mentioning commands inline but not as a description should pass."""
        content = "The `git log --oneline` command shows 5 commits since the rebase was applied to main branch."
        is_junk, reason = _is_junk(content, "experience")
        assert not is_junk

    def test_generic_opener_long_content_allowed(self):
        """Generic opener pattern only triggers for short content (<120 chars)."""
        content = (
            "The system uses SQLite with WAL mode for concurrent reads alongside "
            "a single writer, with thread-local connections and a write lock for serialization."
        )
        assert len(content) >= 120
        is_junk, reason = _is_junk(content, "fact")
        assert not is_junk

    def test_non_generic_opener(self):
        content = "JWT tokens must include an expiry claim to prevent indefinite session persistence."
        is_junk, reason = _is_junk(content, "insight")
        assert not is_junk


class TestIsJunkGoodContent:
    """Verify that legitimate content passes through the filter."""

    @pytest.mark.parametrize(
        "content,atom_type",
        [
            (
                "N+1 queries in _score_atoms cause O(n) database round-trips per recall; "
                "pass atom_map from _spread_activation instead.",
                "antipattern",
            ),
            (
                "SQLite WAL mode allows concurrent readers alongside a single writer, "
                "eliminating per-call open/close overhead with persistent connections.",
                "fact",
            ),
            (
                "The Hebbian learning engine uses temporal co-activation within a sliding "
                "window to strengthen synapse connections between co-accessed atoms.",
                "insight",
            ),
            (
                "Command `git log --oneline -20` produced error: fatal: bad default revision 'HEAD'",
                "experience",
            ),
            (
                "Avoid using `Edit` without prerequisite — old_string not found in file",
                "antipattern",
            ),
            (
                "Permission requested for `Bash`: uv run pytest tests/test_storage.py -x",
                "experience",
            ),
        ],
    )
    def test_good_content_passes(self, content: str, atom_type: str):
        is_junk, reason = _is_junk(content, atom_type)
        assert not is_junk, f"False positive: {content!r} rejected as {reason}"


class TestIsJunkEdgeCases:
    """Edge cases for the junk filter."""

    def test_empty_content(self):
        is_junk, reason = _is_junk("", "experience")
        assert is_junk
        assert reason == "empty"

    def test_none_like_empty(self):
        """Whitespace-only content."""
        is_junk, reason = _is_junk("   ", "fact")
        assert is_junk  # too_short for fact type

    def test_pattern_case_insensitive(self):
        is_junk, reason = _is_junk("DELEGATED TO the sub-agent for review", "experience")
        assert is_junk
        assert reason == "tool_narration"
