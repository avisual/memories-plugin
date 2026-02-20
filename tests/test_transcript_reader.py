"""Tests for the transcript reader feature in memories.cli.

Covers:
- _derive_transcript_path: path formula, missing files, edge cases
- _read_transcript_insights: JSONL parsing, filtering, deduplication, caps
- _store_transcript_insights: novelty gating, atom-type mapping, return count
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memories.cli import (
    _derive_transcript_path,
    _read_transcript_insights,
    _store_transcript_insights,
)


# ---------------------------------------------------------------------------
# JSONL helper
# ---------------------------------------------------------------------------


def make_jsonl_line(entry_type: str, content_blocks: list[dict[str, Any]]) -> str:
    """Build a JSONL transcript line.

    Parameters
    ----------
    entry_type:
        Outer ``"type"`` field, e.g. ``"assistant"``, ``"user"``, or
        ``"file-history-snapshot"``.
    content_blocks:
        List of content-block dicts placed in ``message.content``.
    """
    record: dict[str, Any] = {
        "type": entry_type,
        "sessionId": "test-session-abc123",
        "message": {
            "role": entry_type if entry_type in ("user", "assistant") else entry_type,
            "content": content_blocks,
        },
    }
    return json.dumps(record)


def thinking_block(text: str) -> dict[str, Any]:
    """Return an assistant content block of type ``"thinking"``."""
    return {"type": "thinking", "thinking": text, "signature": "sig-xyz"}


def text_block(text: str) -> dict[str, Any]:
    """Return an assistant content block of type ``"text"``."""
    return {"type": "text", "text": text}


def _long(prefix: str, length: int = 60) -> str:
    """Return a string of ``length`` chars starting with ``prefix``."""
    return (prefix + "x" * length)[:length]


# ---------------------------------------------------------------------------
# Mock brain factory
# ---------------------------------------------------------------------------


def _mock_brain(
    is_novel: bool = True,
    deduplicated: bool = False,
) -> MagicMock:
    """Return a minimal mock brain for _store_transcript_insights tests."""
    brain = MagicMock()
    brain._storage = MagicMock()
    brain._storage.execute_many = AsyncMock()
    brain._learning = MagicMock()
    brain._learning.assess_novelty = AsyncMock(return_value=is_novel)
    brain.remember = AsyncMock(
        return_value={"atom_id": 1, "deduplicated": deduplicated}
    )
    return brain


# ===========================================================================
# TestDeriveTranscriptPath
# ===========================================================================


class TestDeriveTranscriptPath:
    def test_path_formula(self, tmp_path: Path) -> None:
        """CWD /Users/foo/bar maps to ~/.claude/projects/-Users-foo-bar/<id>.jsonl."""
        session_id = "abc123"
        cwd = "/Users/foo/bar"
        expected_dir = Path.home() / ".claude" / "projects" / "-Users-foo-bar"
        expected_file = expected_dir / f"{session_id}.jsonl"

        # Create the file so the function can confirm it exists.
        expected_dir.mkdir(parents=True, exist_ok=True)
        expected_file.touch()

        try:
            result = _derive_transcript_path(session_id, cwd)
            assert result == expected_file
        finally:
            expected_file.unlink(missing_ok=True)

    def test_returns_none_when_missing(self) -> None:
        """A session that has no transcript file returns None."""
        result = _derive_transcript_path(
            "nonexistent-session-zzz", "/no/such/project"
        )
        assert result is None

    def test_empty_session_id_returns_none(self) -> None:
        """Empty session_id → no file will exist → None."""
        result = _derive_transcript_path("", "/some/cwd")
        assert result is None

    def test_empty_cwd_returns_none(self) -> None:
        """Empty cwd → the computed path won't exist → None."""
        result = _derive_transcript_path("some-session-id", "")
        assert result is None


# ===========================================================================
# TestReadTranscriptInsights
# ===========================================================================


class TestReadTranscriptInsights:
    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _write_transcript(self, tmp_path: Path, lines: list[str]) -> Path:
        """Write JSONL lines to a transcript file; patch path derivation to return it."""
        f = tmp_path / "session.jsonl"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f

    # -----------------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------------

    def test_extracts_thinking_blocks(self, tmp_path: Path) -> None:
        """Two long thinking blocks are returned as ('thinking', ...) tuples."""
        t1 = _long("First reasoning block: ", 80)
        t2 = _long("Second reasoning block: ", 80)

        lines = [
            make_jsonl_line("assistant", [thinking_block(t1), thinking_block(t2)])
        ]
        transcript = self._write_transcript(tmp_path, lines)

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        thinking_results = [(bt, tx) for bt, tx in results if bt == "thinking"]
        assert len(thinking_results) == 2
        contents = [tx for _, tx in thinking_results]
        assert any(t1[:50] in c for c in contents)
        assert any(t2[:50] in c for c in contents)

    def test_extracts_last_text_block_only(self, tmp_path: Path) -> None:
        """When multiple text blocks are present only the last one is returned."""
        blocks = [
            text_block(_long("First response paragraph ", 60)),
            text_block(_long("Second response paragraph ", 60)),
            text_block(_long("Third response paragraph FINAL ", 60)),
        ]
        lines = [make_jsonl_line("assistant", blocks)]
        transcript = self._write_transcript(tmp_path, lines)

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        text_results = [(bt, tx) for bt, tx in results if bt == "text"]
        assert len(text_results) == 1
        assert "FINAL" in text_results[0][1]

    def test_short_blocks_skipped(self, tmp_path: Path) -> None:
        """Blocks shorter than 50 characters are not returned."""
        short_thinking = "Too short"          # 9 chars
        short_text = "Also tiny"             # 9 chars
        long_thinking = _long("Long enough thinking: ", 60)

        lines = [
            make_jsonl_line(
                "assistant",
                [
                    thinking_block(short_thinking),
                    thinking_block(long_thinking),
                    text_block(short_text),
                ],
            )
        ]
        transcript = self._write_transcript(tmp_path, lines)

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        block_types = [bt for bt, _ in results]
        assert "text" not in block_types          # short text skipped
        assert block_types.count("thinking") == 1  # only the long one

    def test_non_assistant_entries_ignored(self, tmp_path: Path) -> None:
        """user and file-history-snapshot entries are not processed."""
        user_line = make_jsonl_line(
            "user",
            [{"type": "text", "text": _long("User says something important: ", 80)}],
        )
        snapshot_line = json.dumps(
            {"type": "file-history-snapshot", "files": ["/some/file.py"]}
        )
        assistant_line = make_jsonl_line(
            "assistant",
            [thinking_block(_long("Assistant reasoning here: ", 80))],
        )

        lines = [user_line, snapshot_line, assistant_line]
        transcript = self._write_transcript(tmp_path, lines)

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        # Only the assistant thinking block should appear.
        assert len(results) == 1
        assert results[0][0] == "thinking"

    def test_caps_at_five_thinking_blocks(self, tmp_path: Path) -> None:
        """More than 5 unique thinking blocks → only 5 are returned."""
        blocks = [
            thinking_block(_long(f"Unique reasoning block number {i}: ", 90))
            for i in range(8)
        ]
        lines = [make_jsonl_line("assistant", blocks)]
        transcript = self._write_transcript(tmp_path, lines)

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        thinking_results = [bt for bt, _ in results if bt == "thinking"]
        assert len(thinking_results) == 5

    def test_deduplicates_thinking_blocks(self, tmp_path: Path) -> None:
        """Two thinking blocks sharing the same first-80-char prefix → only one kept."""
        shared_prefix = "A" * 80
        t1 = shared_prefix + " first continuation that is longer"
        t2 = shared_prefix + " second continuation that differs"

        lines = [
            make_jsonl_line("assistant", [thinking_block(t1), thinking_block(t2)])
        ]
        transcript = self._write_transcript(tmp_path, lines)

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        thinking_results = [r for r in results if r[0] == "thinking"]
        assert len(thinking_results) == 1

    def test_missing_file_returns_empty(self) -> None:
        """When the transcript file doesn't exist the result is an empty list."""
        with patch("memories.cli._derive_transcript_path", return_value=None):
            results = _read_transcript_insights("s", "/cwd")

        assert results == []

    def test_malformed_json_lines_skipped(self, tmp_path: Path) -> None:
        """Invalid JSON lines in the transcript don't crash the reader."""
        valid_line = make_jsonl_line(
            "assistant",
            [thinking_block(_long("Valid thinking block: ", 70))],
        )
        lines = [
            "this is not json at all{{{",
            valid_line,
            '{"incomplete":',
        ]
        transcript = self._write_transcript(tmp_path, lines)

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        # Only the valid assistant thinking block should survive.
        assert len(results) == 1
        assert results[0][0] == "thinking"

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        """An empty transcript file returns an empty list."""
        transcript = tmp_path / "empty.jsonl"
        transcript.write_text("", encoding="utf-8")

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        assert results == []

    def test_both_thinking_and_text_returned(self, tmp_path: Path) -> None:
        """A file with both block types returns thinking and text tuples."""
        lines = [
            make_jsonl_line(
                "assistant",
                [
                    thinking_block(_long("Reasoning: decided to refactor by ", 70)),
                    text_block(_long("Here is the implementation plan for ", 70)),
                ],
            )
        ]
        transcript = self._write_transcript(tmp_path, lines)

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        block_types = {bt for bt, _ in results}
        assert "thinking" in block_types
        assert "text" in block_types

    def test_thinking_truncated_to_300_chars(self, tmp_path: Path) -> None:
        """Thinking blocks longer than 300 chars are truncated to 300."""
        long_thinking = "T" * 500
        lines = [make_jsonl_line("assistant", [thinking_block(long_thinking)])]
        transcript = self._write_transcript(tmp_path, lines)

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        thinking_results = [tx for bt, tx in results if bt == "thinking"]
        assert len(thinking_results) == 1
        assert len(thinking_results[0]) <= 300

    def test_text_truncated_to_800_chars(self, tmp_path: Path) -> None:
        """Text blocks longer than 800 chars are truncated to 800."""
        long_text = "R" * 1200
        lines = [make_jsonl_line("assistant", [text_block(long_text)])]
        transcript = self._write_transcript(tmp_path, lines)

        with patch(
            "memories.cli._derive_transcript_path", return_value=transcript
        ):
            results = _read_transcript_insights("s", "/cwd")

        text_results = [tx for bt, tx in results if bt == "text"]
        assert len(text_results) == 1
        assert len(text_results[0]) <= 800


# ===========================================================================
# TestStoreTranscriptInsights
# ===========================================================================


class TestStoreTranscriptInsights:
    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _patch_insights(
        self, insights: list[tuple[str, str]]
    ):
        """Return a context manager patching _read_transcript_insights."""
        return patch(
            "memories.cli._read_transcript_insights", return_value=insights
        )

    # -----------------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------------

    async def test_novel_thinking_stored_as_insight(self) -> None:
        """A novel thinking block is stored with type='insight'."""
        brain = _mock_brain(is_novel=True, deduplicated=False)
        insights = [("thinking", _long("Important reasoning that Claude had: ", 80))]

        with self._patch_insights(insights):
            result = await _store_transcript_insights(
                brain, "sess-1", "/cwd", "myproject"
            )

        assert len(result) == 1
        brain.remember.assert_awaited_once()
        call_kwargs = brain.remember.call_args.kwargs
        assert call_kwargs.get("type") == "insight"

    async def test_novel_text_stored_as_experience(self) -> None:
        """A novel text block is stored with type='experience'."""
        brain = _mock_brain(is_novel=True, deduplicated=False)
        insights = [("text", _long("Detailed implementation plan was written ", 80))]

        with self._patch_insights(insights):
            result = await _store_transcript_insights(
                brain, "sess-2", "/cwd", "myproject"
            )

        assert len(result) == 1
        call_kwargs = brain.remember.call_args.kwargs
        assert call_kwargs.get("type") == "experience"

    async def test_non_novel_not_stored(self) -> None:
        """When assess_novelty returns False, brain.remember is never called."""
        brain = _mock_brain(is_novel=False)
        insights = [("thinking", _long("Redundant reasoning the brain already knows ", 60))]

        with self._patch_insights(insights):
            result = await _store_transcript_insights(
                brain, "sess-3", "/cwd", "myproject"
            )

        brain.remember.assert_not_awaited()
        assert result == []

    async def test_deduplicated_atom_not_counted(self) -> None:
        """An atom that brain.remember marks as deduplicated does not increment count."""
        brain = _mock_brain(is_novel=True, deduplicated=True)
        insights = [("thinking", _long("Reasoning that is already stored verbatim ", 60))]

        with self._patch_insights(insights):
            result = await _store_transcript_insights(
                brain, "sess-4", "/cwd", "myproject"
            )

        assert result == []

    async def test_returns_count_of_stored(self) -> None:
        """Three novel, non-deduplicated blocks → three atom IDs returned."""
        brain = _mock_brain(is_novel=True, deduplicated=False)
        brain.remember.return_value = {"atom_id": 1, "deduplicated": False}
        insights = [
            ("thinking", _long("First reasoning block here now ", 60)),
            ("thinking", _long("Second reasoning block here now ", 60)),
            ("text", _long("Final response paragraph here now ", 60)),
        ]

        with self._patch_insights(insights):
            result = await _store_transcript_insights(
                brain, "sess-5", "/cwd", "myproject"
            )

        assert len(result) == 3

    async def test_no_transcript_returns_empty(self) -> None:
        """When no transcript file exists, _read_transcript_insights returns [] → []."""
        brain = _mock_brain(is_novel=True, deduplicated=False)

        with self._patch_insights([]):
            result = await _store_transcript_insights(
                brain, "sess-6", "/no/such/cwd", None
            )

        assert result == []
        brain.remember.assert_not_awaited()

    async def test_mixed_novel_and_non_novel(self) -> None:
        """Only novel atoms are in the returned ID list; non-novel ones are skipped."""
        call_count = 0

        async def alternating_novelty(content: str) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count % 2 == 1  # odd calls are novel

        brain = _mock_brain()
        brain._learning.assess_novelty = alternating_novelty  # type: ignore[assignment]

        insights = [
            ("thinking", _long("Novel reasoning block A here: ", 60)),    # novel
            ("thinking", _long("Stale reasoning block B here: ", 60)),    # not novel
            ("text", _long("Novel response text C here: ", 60)),           # novel
        ]

        with self._patch_insights(insights):
            result = await _store_transcript_insights(
                brain, "sess-7", "/cwd", "myproject"
            )

        assert len(result) == 2

    async def test_source_project_passed_to_remember(self) -> None:
        """brain.remember receives the source_project kwarg."""
        brain = _mock_brain(is_novel=True, deduplicated=False)
        insights = [("thinking", _long("Reasoning with project context here: ", 70))]

        with self._patch_insights(insights):
            await _store_transcript_insights(
                brain, "sess-8", "/cwd", "special-project"
            )

        call_kwargs = brain.remember.call_args.kwargs
        assert call_kwargs.get("source_project") == "special-project"

    async def test_none_project_passed_through(self) -> None:
        """A None project value is passed through to brain.remember unchanged."""
        brain = _mock_brain(is_novel=True, deduplicated=False)
        insights = [("thinking", _long("Reasoning without a project context: ", 70))]

        with self._patch_insights(insights):
            await _store_transcript_insights(brain, "sess-9", "/cwd", None)

        call_kwargs = brain.remember.call_args.kwargs
        assert call_kwargs.get("source_project") is None


# ===========================================================================
# TestExtractAtomicFacts
# ===========================================================================


class TestExtractAtomicFacts:
    """Tests for _extract_atomic_facts — the local-LLM multi-fact extractor."""

    async def test_returns_list_of_facts(self) -> None:
        """When Ollama succeeds a list of atomic fact strings is returned."""
        from memories.cli import _extract_atomic_facts

        mock_response = MagicMock()
        mock_response.response = (
            "- BM25 weight should be 0.05 to avoid dominating vector signal.\n"
            "- FTS5 rank values are negative; normalise by dividing by max absolute rank.\n"
            "- Use asyncio.gather to parallelise vector and BM25 searches."
        )
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient", return_value=mock_client):
            result = await _extract_atomic_facts(
                "long reasoning...", "http://localhost:11434", "llama3.2:3b"
            )

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(f, str) for f in result)

    async def test_strips_bullet_prefix(self) -> None:
        """Bullet '- ' prefixes are stripped from each returned fact."""
        from memories.cli import _extract_atomic_facts

        mock_response = MagicMock()
        mock_response.response = "- The key insight is that X implies Y."
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient", return_value=mock_client):
            result = await _extract_atomic_facts(
                "some text", "http://localhost:11434", "llama3.2:3b"
            )

        assert result[0].startswith("The key insight")

    async def test_fallback_on_exception(self) -> None:
        """When Ollama raises, returns [text[:_THINKING_TRUNCATE]]."""
        from memories.cli import _extract_atomic_facts, _THINKING_TRUNCATE

        original = "X" * 600
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=Exception("connection refused"))

        with patch("ollama.AsyncClient", return_value=mock_client):
            result = await _extract_atomic_facts(
                original, "http://localhost:11434", "llama3.2:3b"
            )

        assert result == [original[:_THINKING_TRUNCATE]]

    async def test_fallback_on_empty_response(self) -> None:
        """An empty model response falls back to [truncated original]."""
        from memories.cli import _extract_atomic_facts, _THINKING_TRUNCATE

        mock_response = MagicMock()
        mock_response.response = ""
        original = "Y" * 500
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient", return_value=mock_client):
            result = await _extract_atomic_facts(
                original, "http://localhost:11434", "llama3.2:3b"
            )

        assert result == [original[:_THINKING_TRUNCATE]]

    async def test_caps_at_five_facts(self) -> None:
        """At most 5 facts are returned even if the model outputs more."""
        from memories.cli import _extract_atomic_facts

        lines = "\n".join(f"- Fact number {i} that is long enough to keep." for i in range(8))
        mock_response = MagicMock()
        mock_response.response = lines
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient", return_value=mock_client):
            result = await _extract_atomic_facts(
                "text", "http://localhost:11434", "llama3.2:3b"
            )

        assert len(result) <= 5

    async def test_extraction_called_when_flag_set(self) -> None:
        """When distill_thinking=True, _extract_atomic_facts is called per thinking block."""
        from memories.cli import _store_transcript_insights

        brain = _mock_brain(is_novel=True, deduplicated=False)
        facts = ["First atomic fact.", "Second atomic fact."]

        with patch("memories.cli._read_transcript_insights",
                   return_value=[("thinking", _long("Long reasoning: ", 80))]), \
             patch("memories.cli._extract_atomic_facts",
                   new=AsyncMock(return_value=facts)) as mock_extract, \
             patch("memories.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.distill_thinking = True
            cfg.ollama_url = "http://localhost:11434"
            cfg.distill_model = "llama3.2:3b"
            mock_cfg.return_value = cfg

            result = await _store_transcript_insights(brain, "sid", "/cwd", "proj")

        mock_extract.assert_awaited_once()
        # Each fact stored as a separate atom.
        assert brain.remember.await_count == 2
        assert len(result) == 2

    async def test_extraction_skipped_when_flag_off(self) -> None:
        """When distill_thinking=False (default), _extract_atomic_facts is not called."""
        from memories.cli import _store_transcript_insights

        brain = _mock_brain(is_novel=True, deduplicated=False)

        with patch("memories.cli._read_transcript_insights",
                   return_value=[("thinking", _long("Reasoning: ", 80))]), \
             patch("memories.cli._extract_atomic_facts",
                   new=AsyncMock()) as mock_extract, \
             patch("memories.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.distill_thinking = False
            mock_cfg.return_value = cfg

            await _store_transcript_insights(brain, "sid", "/cwd", "proj")

        mock_extract.assert_not_awaited()

    async def test_extraction_not_applied_to_text_blocks(self) -> None:
        """Text (conclusion) blocks are never decomposed — only thinking blocks are."""
        from memories.cli import _store_transcript_insights

        brain = _mock_brain(is_novel=True, deduplicated=False)

        with patch("memories.cli._read_transcript_insights",
                   return_value=[("text", _long("Final text response: ", 80))]), \
             patch("memories.cli._extract_atomic_facts",
                   new=AsyncMock()) as mock_extract, \
             patch("memories.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.distill_thinking = True
            cfg.ollama_url = "http://localhost:11434"
            cfg.distill_model = "llama3.2:3b"
            mock_cfg.return_value = cfg

            await _store_transcript_insights(brain, "sid", "/cwd", "proj")

        mock_extract.assert_not_awaited()
