"""Tests for skill_gen quality improvements: dedup, noise filter, truncation, supersedes."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memories.skill_gen import (
    _deduplicate,
    _filter_research_noise,
    _truncate,
    _tech_ids,
    _word_set,
    generate_skill,
)


# ---------------------------------------------------------------------------
# Helper: create a mock row (sqlite3.Row-like dict)
# ---------------------------------------------------------------------------

def _row(content: str, type_: str = "insight", importance: float = 0.8,
         confidence: float = 0.9, severity: str | None = None) -> dict:
    return {
        "content": content,
        "type": type_,
        "importance": importance,
        "confidence": confidence,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# _filter_research_noise
# ---------------------------------------------------------------------------

class TestFilterResearchNoise:
    def test_removes_gepa_atoms(self):
        rows = [
            _row("GEPA adapter interface: GEPAAdapter protocol with 3 core methods"),
            _row("Real project insight about synapses"),
        ]
        result = _filter_research_noise(rows)
        assert len(result) == 1
        assert "synapses" in result[0]["content"]

    def test_removes_gskill_atoms(self):
        rows = [
            _row("gskill pipeline: 5-stage flow"),
            _row("Important implementation detail"),
        ]
        result = _filter_research_noise(rows)
        assert len(result) == 1

    def test_removes_pareto_atoms(self):
        rows = [
            _row("Pareto Front Selection: EvaluationBatch.objective_scores tracks"),
            _row("Real finding"),
        ]
        result = _filter_research_noise(rows)
        assert len(result) == 1

    def test_removes_swe_smith(self):
        rows = [
            _row("SWE-smith Task Generation: Mines GitHub repos for commits"),
            _row("Keep this"),
        ]
        result = _filter_research_noise(rows)
        assert len(result) == 1
        assert result[0]["content"] == "Keep this"

    def test_case_insensitive(self):
        rows = [_row("The GEPA ADAPTER has methods")]
        result = _filter_research_noise(rows)
        assert len(result) == 0

    def test_keeps_non_noise(self):
        rows = [
            _row("_decay_synapses applies full daily decay_rate"),
            _row("hebbian_update bypasses inbound degree cap"),
            _row("Graph health 2026-02-20: 143,305 synapses"),
        ]
        result = _filter_research_noise(rows)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _word_set / _tech_ids
# ---------------------------------------------------------------------------

class TestWordSet:
    def test_basic(self):
        words = _word_set("Hello World 123")
        assert "hello" in words
        assert "world" in words
        assert "123" in words

    def test_underscored(self):
        words = _word_set("_decay_synapses function")
        assert "_decay_synapses" in words or "decay_synapses" in words


class TestTechIds:
    def test_function_names(self):
        ids = _tech_ids("The _decay_synapses method in _apply_ltd")
        assert "_decay_synapses" in ids
        assert "_apply_ltd" in ids

    def test_file_refs(self):
        ids = _tech_ids("Check consolidation.py and synapses.py")
        assert "consolidation.py" in ids
        assert "synapses.py" in ids

    def test_no_false_positives(self):
        ids = _tech_ids("A simple sentence with no code")
        assert len(ids) == 0


# ---------------------------------------------------------------------------
# _deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_exact_duplicates(self):
        rows = [
            _row("The _decay_synapses function is broken in consolidation.py"),
            _row("The _decay_synapses function is broken in consolidation.py"),
        ]
        result = _deduplicate(rows)
        assert len(result) == 1

    def test_high_overlap(self):
        rows = [
            _row("The _decay_synapses function applies decay incorrectly"),
            _row("The _decay_synapses function applies decay wrong every run"),
        ]
        result = _deduplicate(rows)
        assert len(result) == 1

    def test_tech_id_dedup(self):
        """Two atoms sharing 2+ tech IDs with moderate overlap are deduped."""
        rows = [
            _row("Check _decay_synapses in consolidation.py for the decay rate bug"),
            _row("Fix _decay_synapses in consolidation.py to use time-proportional approach"),
        ]
        result = _deduplicate(rows)
        assert len(result) == 1

    def test_different_topics_kept(self):
        rows = [
            _row("The _decay_synapses function applies decay incorrectly"),
            _row("The hebbian_update function bypasses degree caps in synapses.py"),
        ]
        result = _deduplicate(rows)
        assert len(result) == 2

    def test_keeps_first_highest_importance(self):
        """First row (highest importance) is kept in a duplicate cluster."""
        rows = [
            _row("The _decay_synapses function is broken", importance=0.9),
            _row("The _decay_synapses function is broken badly", importance=0.7),
        ]
        result = _deduplicate(rows)
        assert len(result) == 1
        assert result[0]["importance"] == 0.9

    def test_very_short_atoms_dropped(self):
        rows = [_row("ab")]
        result = _deduplicate(rows)
        assert len(result) == 0

    def test_no_false_dedup_on_shared_file(self):
        """Two atoms referencing the same file but different functions aren't deduped."""
        rows = [
            _row("In consolidation.py, _decay_synapses applies wrong rate with daily multiplier"),
            _row("In consolidation.py, _promote_strong_atoms incorrectly promotes low-confidence atoms"),
        ]
        result = _deduplicate(rows)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------

class TestTruncate:
    def test_short_content_unchanged(self):
        assert _truncate("Hello world", 300) == "Hello world"

    def test_truncates_at_sentence_boundary(self):
        text = "First sentence. Second sentence that goes beyond the limit."
        result = _truncate(text, 30)
        assert result == "First sentence."

    def test_truncates_with_ellipsis(self):
        text = "A" * 400
        result = _truncate(text, 300)
        assert result.endswith("...")
        assert len(result) <= 303  # 300 + "..."

    def test_default_max_len(self):
        text = "A" * 400
        result = _truncate(text)
        assert len(result) <= 303


# ---------------------------------------------------------------------------
# generate_skill (integration with temp DB)
# ---------------------------------------------------------------------------

class TestGenerateSkillIntegration:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        """Create a temporary SQLite database with test atoms and synapses."""
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE atoms (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                region TEXT,
                confidence REAL DEFAULT 1.0,
                importance REAL DEFAULT 0.8,
                is_deleted INTEGER DEFAULT 0,
                severity TEXT,
                access_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE synapses (
                id INTEGER PRIMARY KEY,
                source_id INTEGER,
                target_id INTEGER,
                relationship TEXT,
                strength REAL DEFAULT 0.5,
                bidirectional INTEGER DEFAULT 0,
                activated_count INTEGER DEFAULT 0
            )
        """)

        # Insert test atoms (content must be >= 60 chars for quality filter).
        atoms = [
            (1, "Active antipattern: _foo_bar breaks in production when called with empty arguments in batch mode", "antipattern", "project:test", 1.0, 0.9, 0, "high"),
            (2, "Resolved antipattern: _baz_qux was slow but has been fixed now using batch processing approach", "antipattern", "project:test", 1.0, 0.85, 0, "medium"),
            (3, "Fix for _baz_qux: implemented the batch approach which resolved the slow processing issue completely", "insight", "project:test", 1.0, 0.8, 0, None),
            (4, "GEPA adapter interface: GEPAAdapter protocol with 3 core methods for evaluation and reflection", "insight", "project:test", 1.0, 0.7, 0, None),
            (5, "Duplicate of first: _foo_bar breaks in production badly when called with empty arguments in batch", "antipattern", "project:test", 0.9, 0.7, 0, "high"),
            (6, "Valid insight about the database schema and migrations that affect the consolidation pipeline", "insight", "project:test", 1.0, 0.75, 0, None),
        ]
        conn.executemany(
            "INSERT INTO atoms (id, content, type, region, confidence, importance, is_deleted, severity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            atoms,
        )

        # Atom 3 supersedes atom 2 (marks it as resolved).
        conn.execute(
            "INSERT INTO synapses (source_id, target_id, relationship, strength) VALUES (3, 2, 'supersedes', 1.0)"
        )

        conn.commit()
        conn.close()
        return db

    @pytest.mark.asyncio
    async def test_excludes_superseded_antipatterns(self, db_path: Path):
        md = await generate_skill("test", db_path=db_path)
        assert "_baz_qux was slow" not in md  # Superseded atom excluded
        assert "_foo_bar breaks" in md  # Active atom included

    @pytest.mark.asyncio
    async def test_excludes_gepa_noise(self, db_path: Path):
        md = await generate_skill("test", db_path=db_path)
        assert "GEPA adapter" not in md
        assert "GEPAAdapter" not in md

    @pytest.mark.asyncio
    async def test_deduplicates(self, db_path: Path):
        md = await generate_skill("test", db_path=db_path)
        # Only one of the _foo_bar atoms should appear.
        assert md.count("_foo_bar") == 1

    @pytest.mark.asyncio
    async def test_keeps_valid_content(self, db_path: Path):
        md = await generate_skill("test", db_path=db_path)
        assert "database schema" in md

    @pytest.mark.asyncio
    async def test_truncates_long_content(self, db_path: Path):
        # Add a very long atom.
        conn = sqlite3.connect(str(db_path))
        long_content = "A" * 500
        conn.execute(
            "INSERT INTO atoms (id, content, type, region, confidence, importance, is_deleted, severity) VALUES (7, ?, 'fact', 'project:test', 1.0, 0.7, 0, NULL)",
            (long_content,),
        )
        conn.commit()
        conn.close()

        md = await generate_skill("test", db_path=db_path)
        # No single line should have 500+ A's.
        for line in md.split("\n"):
            assert "A" * 400 not in line

    @pytest.mark.asyncio
    async def test_writes_to_output_path(self, db_path: Path, tmp_path: Path):
        out = tmp_path / "out" / "SKILL.md"
        await generate_skill("test", output_path=str(out), db_path=db_path)
        assert out.exists()
        content = out.read_text()
        assert "test — Learned Knowledge" in content

    @pytest.mark.asyncio
    async def test_empty_project(self, db_path: Path):
        md = await generate_skill("nonexistent", db_path=db_path)
        assert "No memories atoms found" in md
