"""Wave 16a tests: context injection improvements.

1. XML structural delimiter around hook injection
2. Primacy-recency sandwich ordering (antipatterns first/last)
3. Dynamic ContextBudget priority tiers by prompt length
4. Pathway rendering with content previews (not raw IDs)
5. Stop hook does NOT call reflect() or consolidation
6. Session-start predictive pre-activation from CLAUDE.md
7. "No results" signal with XML wrapper
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atom(
    atom_id: int,
    content: str,
    atom_type: str = "fact",
    score: float = 0.5,
    confidence: float = 1.0,
    severity: str | None = None,
    instead: str | None = None,
) -> dict[str, Any]:
    """Build a minimal atom dict for testing."""
    atom: dict[str, Any] = {
        "id": atom_id,
        "content": content,
        "type": atom_type,
        "score": score,
        "confidence": confidence,
    }
    if severity is not None:
        atom["severity"] = severity
    if instead is not None:
        atom["instead"] = instead
    return atom


def _make_recall_result(
    atoms: list[dict[str, Any]] | None = None,
    antipatterns: list[dict[str, Any]] | None = None,
    pathways: list[dict[str, Any]] | None = None,
    budget_used: int = 100,
    budget_remaining: int = 900,
) -> dict[str, Any]:
    """Build a recall result dict."""
    return {
        "atoms": atoms or [],
        "antipatterns": antipatterns or [],
        "pathways": pathways or [],
        "budget_used": budget_used,
        "budget_remaining": budget_remaining,
        "seed_count": 1,
        "total_activated": len(atoms or []) + len(antipatterns or []),
    }


def _mock_brain(recall_result: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock Brain with a recall method that returns the given result."""
    brain = MagicMock()
    brain._storage = MagicMock()
    brain._storage.execute = AsyncMock(return_value=[])
    brain._storage.execute_write = AsyncMock(return_value=0)
    brain._storage.execute_many = AsyncMock()
    brain._learning = MagicMock()
    brain._learning.session_end_learning = AsyncMock()
    brain._learning.assess_novelty = AsyncMock(return_value=False)
    brain.recall = AsyncMock(return_value=recall_result or _make_recall_result())
    brain.end_session = AsyncMock()
    brain.remember = AsyncMock(return_value={"atom_id": 999})
    brain._current_session_id = None
    return brain


# ---------------------------------------------------------------------------
# 1. XML tags present and well-formed in hook output
# ---------------------------------------------------------------------------


class TestXMLDelimiters:
    """W16a-1: hook injection is wrapped in <memories>...</memories>."""

    async def test_prompt_submit_xml_tags(self) -> None:
        """prompt-submit output has opening and closing <memories> tags."""
        atoms = [_make_atom(1, "Redis SCAN is O(N)", score=0.9)]
        result = _make_recall_result(atoms=atoms)
        brain = _mock_brain(result)

        from memories.cli import _hook_prompt_submit, _reset_brain_singleton

        await _reset_brain_singleton()

        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            output = await _hook_prompt_submit({
                "prompt": "How do I scan keys in Redis?",
                "cwd": "/tmp/testproject",
                "session_id": "sess-xml-1",
            })

        assert output.startswith("<memories>"), f"Missing opening tag: {output[:80]}"
        assert output.rstrip().endswith("</memories>"), f"Missing closing tag: {output[-80:]}"

    async def test_session_start_xml_tags(self) -> None:
        """session-start output has opening and closing <memories> tags."""
        atoms = [_make_atom(1, "Project uses FastAPI", score=0.8)]
        result = _make_recall_result(atoms=atoms)
        brain = _mock_brain(result)

        from memories.cli import _hook_session_start, _reset_brain_singleton

        await _reset_brain_singleton()

        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            output = await _hook_session_start({
                "session_id": "sess-xml-2",
                "cwd": "/tmp/testproject",
            })

        assert output.startswith("<memories>"), f"Missing opening tag: {output[:80]}"
        assert output.rstrip().endswith("</memories>"), f"Missing closing tag: {output[-80:]}"


# ---------------------------------------------------------------------------
# 2. Primacy-recency sandwich ordering
# ---------------------------------------------------------------------------


class TestPrimacyRecencyOrdering:
    """W16a-2: highest-severity antipattern is first, remaining at end."""

    async def test_highest_severity_antipattern_is_first_atom(self) -> None:
        """The single highest-severity antipattern appears as the first atom line."""
        atoms = [
            _make_atom(10, "Some regular fact", score=0.9),
            _make_atom(11, "Another regular fact", score=0.8),
        ]
        antipatterns = [
            _make_atom(20, "Low severity warning", atom_type="antipattern",
                       severity="low", score=0.7),
            _make_atom(21, "CRITICAL DANGER", atom_type="antipattern",
                       severity="critical", score=0.6),
            _make_atom(22, "High severity issue", atom_type="antipattern",
                       severity="high", score=0.5),
        ]
        result = _make_recall_result(atoms=atoms, antipatterns=antipatterns)
        brain = _mock_brain(result)

        from memories.cli import _hook_prompt_submit, _reset_brain_singleton

        await _reset_brain_singleton()

        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            output = await _hook_prompt_submit({
                "prompt": "How do I handle this dangerous operation?" * 5,
                "cwd": "/tmp/testproject",
                "session_id": "sess-primacy-1",
            })

        lines = output.strip().split("\n")
        # Atom lines start with "  [" (indented), not "[memories]" header lines.
        atom_lines = [l for l in lines if l.startswith("  [")]
        assert len(atom_lines) >= 3, f"Expected at least 3 atom lines, got {len(atom_lines)}"
        # The first atom line should contain the critical antipattern.
        assert "CRITICAL DANGER" in atom_lines[0], (
            f"Expected critical antipattern first, got: {atom_lines[0]}"
        )

    async def test_antipatterns_appear_at_end(self) -> None:
        """Remaining antipatterns (non-top) appear after regular atoms at the end."""
        atoms = [_make_atom(10, "Regular fact", score=0.9)]
        antipatterns = [
            _make_atom(20, "Medium warning A", atom_type="antipattern",
                       severity="medium", score=0.7),
            _make_atom(21, "Medium warning B", atom_type="antipattern",
                       severity="medium", score=0.6),
        ]
        result = _make_recall_result(atoms=atoms, antipatterns=antipatterns)
        brain = _mock_brain(result)

        from memories.cli import _hook_prompt_submit, _reset_brain_singleton

        await _reset_brain_singleton()

        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            output = await _hook_prompt_submit({
                "prompt": "Implementing a new feature with database changes" * 3,
                "cwd": "/tmp/testproject",
                "session_id": "sess-primacy-2",
            })

        # The "known pitfalls" section (remaining antipatterns) should
        # appear after the regular atoms and before </memories>.
        assert "known pitfalls" in output, "Missing 'known pitfalls' section for remaining antipatterns"
        pitfalls_pos = output.index("known pitfalls")
        closing_pos = output.index("</memories>")
        assert pitfalls_pos < closing_pos, "Pitfalls section should appear before closing tag"

        # The regular fact should appear before the pitfalls section.
        regular_pos = output.index("Regular fact")
        assert regular_pos < pitfalls_pos, "Regular atoms should appear before pitfalls section"


# ---------------------------------------------------------------------------
# 3. Budget tier: dynamic selection by prompt length
# ---------------------------------------------------------------------------


class TestDynamicBudgetTiers:
    """W16a-3: budget tier selected by prompt length and hook type."""

    def test_budget_tier_ordering(self) -> None:
        """Budget tiers are ordered: critical > background > minimal."""
        from memories.cli import _hook_budget

        critical = _hook_budget("prompt-submit", prompt_length=600)
        background = _hook_budget("prompt-submit", prompt_length=300)
        minimal = _hook_budget("prompt-submit", prompt_length=50)
        assert critical > background > minimal
        # Sanity: critical should never be below 4000 (catastrophic misconfiguration).
        assert critical >= 4000, f"Critical budget dangerously low: {critical}"

    def test_session_start_always_background(self) -> None:
        """session-start hook always uses background tier regardless of prompt."""
        from memories.cli import _hook_budget

        budget = _hook_budget("session-start", prompt_length=0)
        assert budget > 0

    def test_pre_tool_always_minimal(self) -> None:
        """pre-tool hook always uses minimal tier."""
        from memories.cli import _hook_budget

        pre_tool_budget = _hook_budget("pre-tool", prompt_length=1000)
        critical_budget = _hook_budget("prompt-submit", prompt_length=600)
        assert pre_tool_budget < critical_budget, (
            f"pre-tool ({pre_tool_budget}) should be less than critical ({critical_budget})"
        )


# ---------------------------------------------------------------------------
# 4. Pathway rendering uses content previews
# ---------------------------------------------------------------------------


class TestPathwayContentPreviews:
    """W16a-4: pathway rendering shows atom content, not raw IDs."""

    def test_pathway_renders_content_not_ids(self) -> None:
        """Pathway lines show quoted content previews instead of integer IDs."""
        from memories.cli import _format_pathways

        atoms = [
            _make_atom(1, "Redis SCAN is O(N) over full keyspace, not per-call."),
            _make_atom(2, "Never use KEYS in production — it blocks the event loop."),
        ]
        pathways = [
            {
                "source_id": 1,
                "target_id": 2,
                "relationship": "warns-against",
                "strength": 0.9,
            },
        ]
        result = _make_recall_result(atoms=atoms, pathways=pathways)

        lines = _format_pathways(pathways, result)

        # Should contain content previews.
        joined = "\n".join(lines)
        assert "Redis SCAN" in joined, f"Expected content preview for source, got: {joined}"
        assert "Never use KEYS" in joined, f"Expected content preview for target, got: {joined}"
        # Should NOT contain raw integer IDs as standalone tokens.
        assert ' 1 ' not in joined.replace("connections:", ""), (
            f"Raw source ID found in pathway line: {joined}"
        )

    def test_pathway_truncates_long_content(self) -> None:
        """Content previews longer than 60 chars are truncated with ellipsis."""
        from memories.cli import _format_pathways

        long_content = "A" * 100
        atoms = [
            _make_atom(1, long_content),
            _make_atom(2, "Short target"),
        ]
        pathways = [
            {"source_id": 1, "target_id": 2, "relationship": "related-to", "strength": 0.5},
        ]
        result = _make_recall_result(atoms=atoms, pathways=pathways)

        lines = _format_pathways(pathways, result)
        joined = "\n".join(lines)

        # The truncated content should end with ...
        assert '..."' in joined, f"Expected truncated content with '...', got: {joined}"
        # Should not contain the full 100-char string.
        assert long_content not in joined

    def test_pathway_falls_back_to_id_for_unknown_atoms(self) -> None:
        """When atom content is not in result, fall back to raw ID."""
        from memories.cli import _format_pathways

        pathways = [
            {"source_id": 999, "target_id": 888, "relationship": "related-to", "strength": 0.5},
        ]
        result = _make_recall_result(atoms=[], pathways=pathways)

        lines = _format_pathways(pathways, result)
        joined = "\n".join(lines)

        assert "999" in joined, f"Expected fallback to ID 999, got: {joined}"
        assert "888" in joined, f"Expected fallback to ID 888, got: {joined}"


# ---------------------------------------------------------------------------
# 5. Stop hook does NOT call reflect() or consolidation
# ---------------------------------------------------------------------------


class TestStopHookNoConsolidation:
    """W16a-5: stop hook runs Hebbian learning but NOT consolidation/reflect."""

    async def test_stop_hook_does_not_call_reflect(self) -> None:
        """_hook_stop must NOT invoke brain.reflect() or consolidation."""
        brain = _mock_brain()
        brain.reflect = AsyncMock()
        # Simulate accumulated atoms for Hebbian learning.
        brain._storage.execute = AsyncMock(side_effect=[
            [{"atom_id": 1, "accessed_at": "2026-01-01T00:00:00"}],  # hook_session_atoms
            [],  # lineage query
        ])

        from memories.cli import _hook_stop, _reset_brain_singleton

        await _reset_brain_singleton()

        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            await _hook_stop({
                "session_id": "sess-stop-1",
                "cwd": "/tmp/testproject",
            })

        brain.reflect.assert_not_called()

    async def test_stop_hook_still_runs_hebbian(self) -> None:
        """_hook_stop should still call session_end_learning for Hebbian."""
        brain = _mock_brain()
        brain._storage.execute = AsyncMock(return_value=[
            {"atom_id": 1, "accessed_at": "2026-01-01T00:00:00"},
            {"atom_id": 2, "accessed_at": "2026-01-01T00:00:01"},
        ])

        from memories.cli import _hook_stop, _reset_brain_singleton

        await _reset_brain_singleton()

        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            # Patch _store_transcript_insights to avoid filesystem access.
            with patch("memories.cli._store_transcript_insights", AsyncMock(return_value=[])):
                await _hook_stop({
                    "session_id": "sess-stop-2",
                    "cwd": "/tmp/testproject",
                })

        brain._learning.session_end_learning.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Session-start predictive pre-activation from CLAUDE.md
# ---------------------------------------------------------------------------


class TestSessionStartPreActivation:
    """W16a-6: session-start reads CLAUDE.md and pre-seeds session atoms."""

    async def test_preseed_reads_claude_md(self, tmp_path: Path) -> None:
        """_preseed_from_claude_md reads .claude/CLAUDE.md and calls recall."""
        # Create a fake CLAUDE.md.
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        claude_md = claude_dir / "CLAUDE.md"
        claude_md.write_text("# My Project\n\nThis project uses FastAPI and PostgreSQL.")

        preseed_result = _make_recall_result(
            atoms=[_make_atom(42, "FastAPI project pattern", score=0.7)]
        )
        brain = _mock_brain()
        brain.recall = AsyncMock(return_value=preseed_result)

        from memories.cli import _preseed_from_claude_md

        await _preseed_from_claude_md(brain, "sess-preseed-1", str(tmp_path))

        # recall should have been called with the first 200 chars.
        brain.recall.assert_called_once()
        call_kwargs = brain.recall.call_args
        query = call_kwargs.kwargs.get("query") or call_kwargs.args[0] if call_kwargs.args else call_kwargs[1].get("query", "")
        # Handle both positional and keyword args.
        if not query and call_kwargs.kwargs:
            query = call_kwargs.kwargs.get("query", "")
        assert "My Project" in query or "FastAPI" in query, f"Unexpected query: {query}"

        # Atom IDs should have been inserted into hook_session_atoms.
        brain._storage.execute_many.assert_called_once()

    async def test_preseed_skips_missing_claude_md(self, tmp_path: Path) -> None:
        """_preseed_from_claude_md is a no-op if CLAUDE.md does not exist."""
        brain = _mock_brain()

        from memories.cli import _preseed_from_claude_md

        await _preseed_from_claude_md(brain, "sess-preseed-2", str(tmp_path))

        brain.recall.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Empty recall produces XML-wrapped "no results" message
# ---------------------------------------------------------------------------


class TestNoResultsSignal:
    """W16a-7: empty recall returns XML-wrapped message, not empty string."""

    async def test_empty_recall_returns_xml_no_results(self) -> None:
        """When recall returns 0 atoms, inject the 'no relevant' message."""
        result = _make_recall_result(atoms=[], antipatterns=[])
        brain = _mock_brain(result)

        from memories.cli import _hook_prompt_submit, _reset_brain_singleton

        await _reset_brain_singleton()

        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            output = await _hook_prompt_submit({
                "prompt": "Tell me about quantum computing applications in finance",
                "cwd": "/tmp/testproject",
                "session_id": "sess-empty-1",
            })

        assert "<memories>" in output, f"Missing opening tag: {output}"
        assert "</memories>" in output, f"Missing closing tag: {output}"
        assert "No relevant prior learnings" in output, f"Missing no-results message: {output}"


# ---------------------------------------------------------------------------
# 8. XML injection in atom content is escaped
# ---------------------------------------------------------------------------


class TestXMLInjectionEscaping:
    """P0: atom content with XML special characters must be escaped."""

    def test_xml_injection_in_atom_content_is_escaped(self) -> None:
        """Atom content containing </memories> must be escaped."""
        from memories.cli import _format_atom_line

        atom = _make_atom(1, 'Do not use </memories> in content')
        output = _format_atom_line(atom)

        assert "&lt;/memories&gt;" in output, f"Expected escaped tag, got: {output}"
        assert output.count("</memories>") == 0, (
            f"Unescaped closing tag found in output: {output}"
        )

    def test_xml_injection_in_instead_field_is_escaped(self) -> None:
        """Antipattern 'instead' field with XML is escaped."""
        from memories.cli import _format_atom_line

        atom = _make_atom(
            2, "Bad pattern", atom_type="antipattern", severity="high",
            instead='Use <safe> approach & avoid "quotes"',
        )
        output = _format_atom_line(atom)

        assert "&lt;safe&gt;" in output, f"Expected escaped tag in instead, got: {output}"
        assert "&amp;" in output, f"Expected escaped ampersand, got: {output}"

    def test_xml_injection_in_pathway_preview_is_escaped(self) -> None:
        """Pathway content preview with XML is escaped."""
        from memories.cli import _format_pathways

        atoms = [_make_atom(1, "Use <memories> tag carefully")]
        pathways = [{"source_id": 1, "target_id": 2, "relationship": "related-to"}]
        result = _make_recall_result(atoms=atoms, pathways=pathways)

        lines = _format_pathways(pathways, result)
        joined = "\n".join(lines)

        assert "&lt;memories&gt;" in joined, f"Expected escaped tag in preview, got: {joined}"


# ---------------------------------------------------------------------------
# 9. Session-start empty recall returns XML wrapper
# ---------------------------------------------------------------------------


class TestSessionStartEmptyRecall:
    """P2: session-start with empty recall returns XML-wrapped message."""

    async def test_session_start_empty_recall_returns_xml(self) -> None:
        """When session-start recall returns 0 atoms and 0 antipatterns,
        the output must be wrapped in <memories>...</memories>."""
        result = _make_recall_result(atoms=[], antipatterns=[])
        brain = _mock_brain(result)

        from memories.cli import _hook_session_start, _reset_brain_singleton

        await _reset_brain_singleton()

        with patch("memories.cli._get_brain", AsyncMock(return_value=brain)):
            output = await _hook_session_start({
                "session_id": "sess-empty-start-1",
                "cwd": "/tmp/testproject",
            })

        assert output.startswith("<memories>"), f"Missing opening tag: {output}"
        assert output.rstrip().endswith("</memories>"), f"Missing closing tag: {output}"
        assert "No prior learnings" in output, f"Missing empty message: {output}"
        # Ensure exactly one opening and one closing tag.
        assert output.count("<memories>") == 1
        assert output.count("</memories>") == 1
