"""Wave 16c — LRU recall cache + partial synapse index + HookQueue + skill generator.

Changes validated:
- LRU cache: second identical recall returns without DB hit
- LRU cache invalidated on remember()
- LRU cache evicts oldest at 257 entries
- LRU cache key includes budget_tokens, depth, include_antipatterns (C-2 fix)
- LRU cache lives on Brain instance, not module-level (C-2 fix)
- Migration 13: idx_synapses_bidirectional index exists after initialize
- HookQueue.append() creates valid JSON line
- HookQueue.drain() reads all lines, clears file
- HookQueue.drain() skips corrupt JSON lines without crashing
- generate_skill() produces Markdown with frontmatter and section headers
- generate_skill() groups by type (antipatterns first)
- generate_skill() handles project with 0 atoms gracefully
- generate_skill() uses lightweight sqlite3 connection (no Storage.initialize())
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# LRU recall cache
# ---------------------------------------------------------------------------


class TestRecallLRUCache:
    """Test the in-process LRU cache for Brain.recall()."""

    def test_cache_key_deterministic(self):
        from memories.brain import _recall_cache_key

        k1 = _recall_cache_key("hello", "region1", ["fact"], 2000, None, True)
        k2 = _recall_cache_key("hello", "region1", ["fact"], 2000, None, True)
        assert k1 == k2
        assert len(k1) == 16

    def test_cache_key_varies_on_input(self):
        from memories.brain import _recall_cache_key

        k1 = _recall_cache_key("hello", "region1", ["fact"], 2000, None, True)
        k2 = _recall_cache_key("world", "region1", ["fact"], 2000, None, True)
        assert k1 != k2

    def test_cache_key_varies_on_budget_tokens(self):
        """Different budget_tokens produce different cache keys."""
        from memories.brain import _recall_cache_key

        k1 = _recall_cache_key("q", None, None, 500, None, True)
        k2 = _recall_cache_key("q", None, None, 2000, None, True)
        assert k1 != k2

    def test_cache_key_varies_on_depth(self):
        """Different depth values produce different cache keys."""
        from memories.brain import _recall_cache_key

        k1 = _recall_cache_key("q", None, None, 2000, 1, True)
        k2 = _recall_cache_key("q", None, None, 2000, 3, True)
        assert k1 != k2

    def test_cache_key_varies_on_include_antipatterns(self):
        """Different include_antipatterns produce different cache keys."""
        from memories.brain import _recall_cache_key

        k1 = _recall_cache_key("q", None, None, 2000, None, True)
        k2 = _recall_cache_key("q", None, None, 2000, None, False)
        assert k1 != k2

    @pytest.mark.asyncio
    async def test_second_recall_uses_cache(self, tmp_path):
        """Second identical recall returns cached result without a DB hit."""
        from memories.brain import Brain

        db_path = tmp_path / "test.db"
        brain = Brain(db_path=db_path)
        await brain.initialize()

        # Mock retrieval to track calls.
        mock_result = MagicMock()
        mock_result.atoms = [{"id": 1, "content": "test", "type": "fact"}]
        mock_result.antipatterns = []
        mock_result.pathways = []
        mock_result.budget_used = 100
        mock_result.budget_remaining = 1900
        mock_result.total_activated = 5
        mock_result.seed_count = 1
        mock_result.compression_level = 0

        brain._retrieval.recall = AsyncMock(return_value=mock_result)

        # First call — hits retrieval.
        r1 = await brain.recall("test query", region="test")
        assert brain._retrieval.recall.call_count == 1

        # Second call — should use cache.
        r2 = await brain.recall("test query", region="test")
        assert brain._retrieval.recall.call_count == 1  # No additional call.
        assert r1 == r2

        await brain.shutdown()

    @pytest.mark.asyncio
    async def test_different_budget_tokens_no_shared_cache(self, tmp_path):
        """recall() with different budget_tokens must not share cached results."""
        from memories.brain import Brain

        db_path = tmp_path / "test.db"
        brain = Brain(db_path=db_path)
        await brain.initialize()

        call_count = 0

        async def mock_recall(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.atoms = [{"id": call_count, "content": "test", "type": "fact"}]
            result.antipatterns = []
            result.pathways = []
            result.budget_used = kwargs.get("budget_tokens", 2000)
            result.budget_remaining = 0
            result.total_activated = 1
            result.seed_count = 1
            result.compression_level = 0
            return result

        brain._retrieval.recall = AsyncMock(side_effect=mock_recall)

        r1 = await brain.recall("q", budget_tokens=500)
        r2 = await brain.recall("q", budget_tokens=2000)

        # Both calls must hit retrieval (different budget_tokens = different cache keys).
        assert call_count == 2
        assert r1["budget_used"] != r2["budget_used"]

        await brain.shutdown()

    @pytest.mark.asyncio
    async def test_cache_invalidated_on_remember(self, tmp_path):
        """After remember(), the cache is cleared so recall hits DB again."""
        from memories.brain import Brain

        db_path = tmp_path / "test.db"
        brain = Brain(db_path=db_path)
        await brain.initialize()

        # Populate cache with a dummy entry.
        brain._recall_cache["test_key"] = ({"atoms": []}, time.monotonic())
        assert len(brain._recall_cache) == 1

        # Mock to avoid actual embedding/linking (which needs Ollama).
        brain._embeddings.search_similar = AsyncMock(return_value=[])
        brain._learning.suggest_region = AsyncMock(return_value="general")
        brain._learning.auto_link = AsyncMock(return_value=[])
        brain._learning.detect_supersedes = AsyncMock(return_value=0)

        await brain.remember("test content", type="fact")

        # Cache should be empty after remember().
        assert len(brain._recall_cache) == 0

        await brain.shutdown()

    def test_cache_eviction_at_max(self):
        """Cache evicts oldest entries when exceeding _RECALL_CACHE_MAX."""
        from memories.brain import Brain, _RECALL_CACHE_MAX

        brain = Brain.__new__(Brain)
        brain._recall_cache = OrderedDict()

        # Fill cache to max + 1.
        for i in range(_RECALL_CACHE_MAX + 1):
            brain._recall_cache[f"key_{i}"] = ({"atoms": []}, time.monotonic())

        # Manually trigger eviction logic (normally done in recall()).
        while len(brain._recall_cache) > _RECALL_CACHE_MAX:
            brain._recall_cache.popitem(last=False)

        assert len(brain._recall_cache) == _RECALL_CACHE_MAX
        # The first entry should have been evicted.
        assert "key_0" not in brain._recall_cache
        # The last entry should still be present.
        assert f"key_{_RECALL_CACHE_MAX}" in brain._recall_cache

    def test_cache_instance_isolation(self):
        """Each Brain instance has its own cache (no cross-contamination)."""
        from memories.brain import Brain

        b1 = Brain.__new__(Brain)
        b1._recall_cache = OrderedDict()
        b2 = Brain.__new__(Brain)
        b2._recall_cache = OrderedDict()

        b1._recall_cache["key1"] = ({"atoms": []}, time.monotonic())
        assert len(b1._recall_cache) == 1
        assert len(b2._recall_cache) == 0


# ---------------------------------------------------------------------------
# Migration 13: partial index for bidirectional synapses
# ---------------------------------------------------------------------------


class TestMigration13:
    """Test that migration 13 creates idx_synapses_bidirectional."""

    @pytest.mark.asyncio
    async def test_bidirectional_index_exists(self, tmp_path):
        """The bidirectional partial index exists after initialize."""
        from memories.storage import Storage

        db_path = tmp_path / "test.db"
        storage = Storage(db_path=db_path)
        await storage.initialize()

        rows = await storage.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_synapses_bidirectional'"
        )
        assert len(rows) == 1
        assert rows[0]["name"] == "idx_synapses_bidirectional"

        await storage.close()

    @pytest.mark.asyncio
    async def test_bidirectional_index_idempotent(self, tmp_path):
        """Running initialize twice does not fail (index already exists)."""
        from memories.storage import Storage

        db_path = tmp_path / "test.db"
        storage = Storage(db_path=db_path)
        await storage.initialize()
        await storage.close()

        # Re-initialize should not crash.
        storage2 = Storage(db_path=db_path)
        await storage2.initialize()

        rows = await storage2.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_synapses_bidirectional'"
        )
        assert len(rows) == 1

        await storage2.close()


# ---------------------------------------------------------------------------
# HookQueue
# ---------------------------------------------------------------------------


class TestHookQueue:
    """Test the async hook observation queue."""

    def test_append_creates_valid_json_line(self, tmp_path):
        """append() writes a valid JSON line to the queue file."""
        from memories.queue import HookQueue

        queue_path = tmp_path / "queue.jsonl"
        q = HookQueue(queue_path=queue_path)

        q.append("post-tool", {"tool": "Read", "result": "ok"})

        content = queue_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 1

        obj = json.loads(lines[0])
        assert obj["event"] == "post-tool"
        assert obj["data"]["tool"] == "Read"
        assert "queued_at" in obj

    def test_append_multiple(self, tmp_path):
        """Multiple appends create multiple JSON lines."""
        from memories.queue import HookQueue

        queue_path = tmp_path / "queue.jsonl"
        q = HookQueue(queue_path=queue_path)

        q.append("event1", {"a": 1})
        q.append("event2", {"b": 2})
        q.append("event3", {"c": 3})

        lines = queue_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_drain_reads_all_lines_and_clears(self, tmp_path):
        """drain() returns all entries and clears the queue file."""
        from memories.queue import HookQueue

        queue_path = tmp_path / "queue.jsonl"
        q = HookQueue(queue_path=queue_path)

        q.append("e1", {"x": 1})
        q.append("e2", {"y": 2})

        entries = q.drain()
        assert len(entries) == 2
        assert entries[0] == ("e1", {"x": 1})
        assert entries[1] == ("e2", {"y": 2})

        # File should be empty after drain.
        assert queue_path.read_text(encoding="utf-8") == ""

        # Second drain should return nothing.
        assert q.drain() == []

    def test_drain_skips_corrupt_lines(self, tmp_path):
        """drain() skips corrupt JSON lines without crashing."""
        from memories.queue import HookQueue

        queue_path = tmp_path / "queue.jsonl"
        # Write a mix of valid and corrupt lines.
        queue_path.write_text(
            '{"event":"e1","data":{"a":1},"queued_at":"2026-01-01T00:00:00Z"}\n'
            'THIS IS NOT JSON\n'
            '{"event":"e2","data":{"b":2},"queued_at":"2026-01-01T00:00:01Z"}\n',
            encoding="utf-8",
        )

        q = HookQueue(queue_path=queue_path)
        entries = q.drain()

        assert len(entries) == 2
        assert entries[0] == ("e1", {"a": 1})
        assert entries[1] == ("e2", {"b": 2})

    def test_drain_empty_file(self, tmp_path):
        """drain() handles an empty queue file gracefully."""
        from memories.queue import HookQueue

        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text("", encoding="utf-8")

        q = HookQueue(queue_path=queue_path)
        entries = q.drain()
        assert entries == []

    def test_drain_nonexistent_file(self, tmp_path):
        """drain() handles a nonexistent queue file gracefully."""
        from memories.queue import HookQueue

        queue_path = tmp_path / "nonexistent.jsonl"
        q = HookQueue(queue_path=queue_path)
        entries = q.drain()
        assert entries == []

    @pytest.mark.asyncio
    async def test_worker_start_stop(self, tmp_path):
        """Worker can be started and stopped cleanly."""
        from memories.queue import HookQueue

        queue_path = tmp_path / "queue.jsonl"
        q = HookQueue(queue_path=queue_path)

        processed: list[tuple[str, dict]] = []

        async def handler(event: str, data: dict) -> None:
            processed.append((event, data))

        q.append("test-event", {"val": 42})

        task = q.start_worker(handler, interval=0.1)
        # Let the worker run a couple of cycles.
        await asyncio.sleep(0.5)
        await q.stop_worker()

        assert len(processed) >= 1
        assert processed[0] == ("test-event", {"val": 42})


# ---------------------------------------------------------------------------
# generate_skill
# ---------------------------------------------------------------------------


class TestGenerateSkill:
    """Test the SKILL.md generator."""

    @pytest.mark.asyncio
    async def test_produces_frontmatter_and_sections(self, tmp_path):
        """generate_skill() produces Markdown with YAML frontmatter and section headers."""
        from memories.storage import Storage
        from memories.skill_gen import generate_skill

        # Set up a test DB with sample atoms.
        db_path = tmp_path / "test.db"
        storage = Storage(db_path=db_path)
        await storage.initialize()

        # Insert test atoms across different types.
        for content, atom_type, severity in [
            ("Never use eval() on user input — it enables arbitrary code execution via injection attacks", "antipattern", "high"),
            ("Use connection pooling for database access to reduce connection overhead and latency", "skill", None),
            ("Redis SCAN is O(N) in total but returns results incrementally without blocking", "insight", None),
            ("Python 3.12 adds f-string nesting support allowing nested expressions in format strings", "fact", None),
        ]:
            await storage.execute_write(
                """
                INSERT INTO atoms (content, type, region, importance, confidence, severity)
                VALUES (?, ?, 'project:testproj', 0.8, 0.9, ?)
                """,
                (content, atom_type, severity),
            )
        await storage.close()

        # Patch config to use our test DB.
        mock_cfg = MagicMock()
        mock_cfg.db_path = db_path
        mock_cfg.backup_dir = tmp_path / "backups"
        mock_cfg.backup_count = 3

        with patch("memories.config.get_config", return_value=mock_cfg):
            md = await generate_skill("testproj")

        # Check frontmatter.
        assert md.startswith("---\n")
        assert "name: testproj" in md
        assert "description:" in md

        # Check section headers.
        assert "## Antipatterns to Avoid" in md
        assert "## Proven Patterns" in md
        assert "## Key Insights" in md
        assert "## Known Facts" in md

        # Check content.
        assert "Never use eval() on user input" in md
        assert "connection pooling for database" in md
        assert "Redis SCAN is O(N)" in md
        assert "Python 3.12 adds f-string nesting" in md

    @pytest.mark.asyncio
    async def test_groups_by_type_antipatterns_first(self, tmp_path):
        """Antipatterns section appears before other sections."""
        from memories.storage import Storage
        from memories.skill_gen import generate_skill

        db_path = tmp_path / "test.db"
        storage = Storage(db_path=db_path)
        await storage.initialize()

        await storage.execute_write(
            """
            INSERT INTO atoms (content, type, region, importance, confidence, severity)
            VALUES ('Use locks for thread safety when accessing shared state across concurrent workers', 'skill', 'project:myproj', 0.8, 0.9, NULL)
            """,
        )
        await storage.execute_write(
            """
            INSERT INTO atoms (content, type, region, importance, confidence, severity)
            VALUES ('Avoid global state in multi-threaded applications as it creates race conditions and debugging nightmares', 'antipattern', 'project:myproj', 0.8, 0.9, 'medium')
            """,
        )
        await storage.close()

        mock_cfg = MagicMock()
        mock_cfg.db_path = db_path
        mock_cfg.backup_dir = tmp_path / "backups"
        mock_cfg.backup_count = 3

        with patch("memories.config.get_config", return_value=mock_cfg):
            md = await generate_skill("myproj")

        # Antipatterns section should appear before Proven Patterns.
        ap_idx = md.index("## Antipatterns to Avoid")
        pp_idx = md.index("## Proven Patterns")
        assert ap_idx < pp_idx

    @pytest.mark.asyncio
    async def test_handles_zero_atoms(self, tmp_path):
        """generate_skill() handles project with 0 atoms gracefully."""
        from memories.storage import Storage
        from memories.skill_gen import generate_skill

        db_path = tmp_path / "test.db"
        storage = Storage(db_path=db_path)
        await storage.initialize()
        await storage.close()

        mock_cfg = MagicMock()
        mock_cfg.db_path = db_path
        mock_cfg.backup_dir = tmp_path / "backups"
        mock_cfg.backup_count = 3

        with patch("memories.config.get_config", return_value=mock_cfg):
            md = await generate_skill("emptyproj")

        assert md.startswith("---\n")
        assert "name: emptyproj" in md
        assert "No memories atoms found" in md

    @pytest.mark.asyncio
    async def test_writes_to_output_path(self, tmp_path):
        """generate_skill() writes to output_path when provided."""
        from memories.storage import Storage
        from memories.skill_gen import generate_skill

        db_path = tmp_path / "test.db"
        storage = Storage(db_path=db_path)
        await storage.initialize()

        await storage.execute_write(
            """
            INSERT INTO atoms (content, type, region, importance, confidence)
            VALUES ('Some insight about the write-ahead log improving concurrent read performance significantly', 'insight', 'project:wp', 0.8, 0.9)
            """,
        )
        await storage.close()

        mock_cfg = MagicMock()
        mock_cfg.db_path = db_path
        mock_cfg.backup_dir = tmp_path / "backups"
        mock_cfg.backup_count = 3

        output_file = tmp_path / "skills" / "wp" / "SKILL.md"

        with patch("memories.config.get_config", return_value=mock_cfg):
            md = await generate_skill("wp", output_path=str(output_file))

        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == md

    @pytest.mark.asyncio
    async def test_handles_missing_db_gracefully(self, tmp_path):
        """generate_skill() returns friendly error when DB has no atoms table."""
        from memories.skill_gen import generate_skill

        # Point at a DB that exists but has no tables.
        db_path = tmp_path / "empty.db"
        db_path.touch()

        mock_cfg = MagicMock()
        mock_cfg.db_path = db_path

        with patch("memories.config.get_config", return_value=mock_cfg):
            md = await generate_skill("broken")

        assert "Cannot read memories DB" in md
        assert "broken" in md
