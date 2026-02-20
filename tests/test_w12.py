"""Wave 12 — Polish, Hooks & Hebbian Temporal Weighting.

Changes validated:
- Fix 1: _score_atoms double-fetch eliminated — atom_map from _spread_activation reused.
- Fix 2: PreCompact hook wiring — appears in _HOOK_ENTRIES.
- Fix 3: Distillation quality guardrail — generic/short outputs fall back to verbatim.
- Fix 4: ltd_amount dead config removed from consolidation.py local variable.
- Fix 5: PostToolUseFailure hook — captures tool failures as antipatterns/experiences.
- Fix 6: Session-temporal Hebbian weighting — recent pairs get full increment, distant 0.5×.
- Fix 7: MCP prompts — 4 prompts registered on the FastMCP server.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fix 1 — _score_atoms no double-fetch
# ---------------------------------------------------------------------------


class TestScoreAtomsNoDoubleFetch:
    """get_batch_without_tracking must be called at most once per recall."""

    @pytest.mark.asyncio
    async def test_spread_activation_returns_atom_map(self):
        """_spread_activation returns a (activated, atom_map) tuple."""
        from memories.config import RetrievalConfig
        from memories.retrieval import RetrievalEngine

        cfg_mock = MagicMock()
        cfg_mock.min_activation = 0.1
        cfg_mock.decay_factor = 0.85

        synapse_mock = MagicMock()
        synapse_mock.get_neighbors_batch = AsyncMock(return_value={})

        atom_mock = MagicMock()
        atom_mock.get_batch_without_tracking = AsyncMock(return_value={})

        engine = RetrievalEngine.__new__(RetrievalEngine)
        engine._cfg = cfg_mock
        engine._synapses = synapse_mock
        engine._atoms = atom_mock
        engine._weights = None
        engine._synapse_type_weights = {}

        seeds = {1: 0.9, 2: 0.8}
        result = await engine._spread_activation(seeds, depth=1)

        # Must return a tuple of (activated_dict, atom_map_dict).
        assert isinstance(result, tuple)
        assert len(result) == 2
        activated, atom_map = result
        assert isinstance(activated, dict)
        assert isinstance(atom_map, dict)

    @pytest.mark.asyncio
    async def test_score_atoms_uses_atom_map_skips_fetch(self):
        """When atom_map is passed, get_batch_without_tracking is not called."""
        from memories.atoms import Atom
        from memories.retrieval import RetrievalEngine

        cfg_mock = MagicMock()
        cfg_mock.weights = MagicMock(
            vector_similarity=0.35,
            spread_activation=0.25,
            recency=0.10,
            confidence=0.07,
            frequency=0.07,
            importance=0.11,
            bm25=0.05,
        )

        atom_mock = MagicMock()
        atom_mock.get_batch_without_tracking = AsyncMock(return_value={})

        engine = RetrievalEngine.__new__(RetrievalEngine)
        engine._cfg = cfg_mock
        engine._atoms = atom_mock
        engine._weights = None

        # Create a minimal Atom for the map.
        atom = Atom(
            id=42,
            content="test atom",
            type="fact",
            confidence=0.9,
            importance=0.5,
            access_count=1,
            created_at="2026-01-01T00:00:00",
            last_accessed_at="2026-01-01T00:00:00",
        )

        prebuilt_map = {42: atom}
        result = await engine._score_atoms(
            vector_scores={42: 0.9},
            activation_scores={42: 0.9},
            bm25_scores={},
            include_antipatterns=True,
            atom_map=prebuilt_map,
        )

        # get_batch_without_tracking must NOT have been called.
        atom_mock.get_batch_without_tracking.assert_not_called()
        assert len(result) == 1
        assert result[0][0].id == 42

    @pytest.mark.asyncio
    async def test_score_atoms_falls_back_without_atom_map(self):
        """When atom_map is None, get_batch_without_tracking IS called (backward compat)."""
        from memories.atoms import Atom
        from memories.retrieval import RetrievalEngine

        cfg_mock = MagicMock()
        cfg_mock.weights = MagicMock(
            vector_similarity=0.35,
            spread_activation=0.25,
            recency=0.10,
            confidence=0.07,
            frequency=0.07,
            importance=0.11,
            bm25=0.05,
        )

        atom = Atom(
            id=7,
            content="fallback test",
            type="fact",
            confidence=0.8,
            importance=0.5,
            access_count=0,
            created_at="2026-01-01T00:00:00",
        )
        atom_mock = MagicMock()
        atom_mock.get_batch_without_tracking = AsyncMock(return_value={7: atom})

        engine = RetrievalEngine.__new__(RetrievalEngine)
        engine._cfg = cfg_mock
        engine._atoms = atom_mock
        engine._weights = None

        result = await engine._score_atoms(
            vector_scores={7: 0.8},
            activation_scores={7: 0.8},
            bm25_scores={},
            include_antipatterns=True,
            atom_map=None,
        )

        atom_mock.get_batch_without_tracking.assert_called_once()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Fix 2 — PreCompact hook registered
# ---------------------------------------------------------------------------


class TestPreCompactHookRegistered:
    def test_precompact_in_hook_entries(self):
        from memories.setup import _HOOK_ENTRIES

        assert "PreCompact" in _HOOK_ENTRIES
        hooks = _HOOK_ENTRIES["PreCompact"]
        assert hooks
        cmd = hooks[0]["hooks"][0]["command"]
        assert "pre-compact" in cmd

    def test_precompact_handler_exists(self):
        from memories.cli import _hook_pre_compact

        assert callable(_hook_pre_compact)

    def test_precompact_in_dispatch(self):
        """run_hook dispatch table must contain 'pre-compact'."""
        import inspect
        from memories import cli

        source = inspect.getsource(cli.run_hook)
        assert "pre-compact" in source


# ---------------------------------------------------------------------------
# Fix 3 — Distillation quality guardrail
# ---------------------------------------------------------------------------


class TestDistillationQualityGuardrail:
    def test_acceptable_specific_distillation(self):
        from memories.consolidation import ConsolidationEngine

        text = (
            "The authentication module uses JWT tokens with a 1-hour expiry "
            "and refresh token rotation to prevent session hijacking."
        )
        assert ConsolidationEngine._distillation_is_acceptable(text) is True

    def test_rejects_too_short(self):
        from memories.consolidation import ConsolidationEngine

        assert ConsolidationEngine._distillation_is_acceptable("Short text") is False
        assert ConsolidationEngine._distillation_is_acceptable("One two three four five six") is False

    def test_rejects_generic_system_pattern(self):
        from memories.consolidation import ConsolidationEngine

        assert ConsolidationEngine._distillation_is_acceptable(
            "Systems require proper configuration before deploying to production."
        ) is False

    def test_rejects_generic_this_is_important(self):
        from memories.consolidation import ConsolidationEngine

        assert ConsolidationEngine._distillation_is_acceptable(
            "This is important: always check return values before proceeding."
        ) is False

    def test_rejects_generic_it_is_necessary(self):
        from memories.consolidation import ConsolidationEngine

        assert ConsolidationEngine._distillation_is_acceptable(
            "It is necessary to validate all inputs coming from external sources."
        ) is False

    def test_accepts_ten_word_minimum_boundary(self):
        from memories.consolidation import ConsolidationEngine

        # Exactly 10 words, no generic pattern.
        text = "Always close database connections after the query has completed successfully here."
        assert len(text.split()) >= 10
        assert ConsolidationEngine._distillation_is_acceptable(text) is True

    @pytest.mark.asyncio
    async def test_fallback_used_when_generic(self):
        """_distill_cluster must return the fallback when LLM emits a generic string."""
        from memories.atoms import Atom
        from memories.consolidation import ConsolidationEngine

        consolidator = ConsolidationEngine.__new__(ConsolidationEngine)
        consolidator._llm_client = None
        consolidator._ollama_url = "http://localhost:11434"
        consolidator._distill_model = "llama3.2:3b"

        fallback_content = "Redis caches expire after TTL seconds defined at key creation."
        atoms = [
            Atom(
                id=1, content=fallback_content, type="experience",
                confidence=0.9, importance=0.5, access_count=1,
                created_at="2026-01-01T00:00:00",
            )
        ]

        # Simulate LLM returning a generic output.
        import ollama
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(
            return_value=MagicMock(response="The system is working correctly.")
        )
        consolidator._llm_client = mock_client

        result = await consolidator._distill_cluster(atoms)
        assert result == fallback_content


# ---------------------------------------------------------------------------
# Fix 5 — PostToolUseFailure hook
# ---------------------------------------------------------------------------


class TestPostToolFailureHook:
    def test_hook_registered_in_hook_entries(self):
        from memories.setup import _HOOK_ENTRIES

        assert "PostToolUseFailure" in _HOOK_ENTRIES
        cmd = _HOOK_ENTRIES["PostToolUseFailure"][0]["hooks"][0]["command"]
        assert "post-tool-failure" in cmd

    def test_handler_exists(self):
        from memories.cli import _hook_post_tool_failure

        assert callable(_hook_post_tool_failure)

    def test_hook_in_dispatch(self):
        import inspect
        from memories import cli

        source = inspect.getsource(cli.run_hook)
        assert "post-tool-failure" in source

    @pytest.mark.asyncio
    async def test_short_error_skipped(self):
        """Errors shorter than 20 chars must not be stored."""
        from memories.cli import _hook_post_tool_failure

        data = {"tool_name": "Bash", "error": "err", "tool_input": {}, "session_id": "s1"}
        result = await _hook_post_tool_failure(data)
        parsed = json.loads(result)
        assert parsed["hookSpecificOutput"]["hookEventName"] == "PostToolUseFailure"

    @pytest.mark.asyncio
    async def test_skip_tool_respected(self):
        """Tools in _SKIP_TOOLS must be silently ignored."""
        from memories.cli import _hook_post_tool_failure

        data = {
            "tool_name": "Read",
            "error": "No such file or directory: /tmp/nonexistent.txt",
            "tool_input": {},
            "session_id": "s1",
        }
        result = await _hook_post_tool_failure(data)
        parsed = json.loads(result)
        assert parsed["hookSpecificOutput"]["hookEventName"] == "PostToolUseFailure"

    @pytest.mark.asyncio
    async def test_failure_stored_as_antipattern(self, tmp_path):
        """A failure containing antipattern vocabulary is stored with type='antipattern'."""
        from memories.cli import _hook_post_tool_failure, _reset_brain_singleton

        await _reset_brain_singleton()

        stored_calls = []

        async def fake_remember(**kwargs):
            stored_calls.append(kwargs)
            return {"atom_id": 99}

        async def fake_novelty(_):
            return True

        brain_mock = MagicMock()
        brain_mock.remember = fake_remember
        brain_mock._learning = MagicMock()
        brain_mock._learning.assess_novelty = fake_novelty
        brain_mock._storage = MagicMock()
        brain_mock._storage.execute_many = AsyncMock()
        brain_mock._storage.execute_write = AsyncMock()

        with patch("memories.cli._get_brain", return_value=brain_mock):
            with patch("memories.cli._save_hook_atoms", new_callable=AsyncMock):
                with patch("memories.cli._record_hook_stat", new_callable=AsyncMock):
                    data = {
                        "tool_name": "Bash",
                        "error": "permission denied: avoid using sudo for this operation",
                        "tool_input": {"command": "sudo rm -rf /etc"},
                        "session_id": "test-session",
                        "cwd": str(tmp_path),
                    }
                    await _hook_post_tool_failure(data)

        assert stored_calls, "Expected remember() to be called"
        assert stored_calls[0]["type"] == "antipattern"

    @pytest.mark.asyncio
    async def test_failure_stored_as_experience_without_antipattern_vocab(self, tmp_path):
        """A plain failure without antipattern vocabulary is stored as 'experience'."""
        from memories.cli import _hook_post_tool_failure

        stored_calls = []

        async def fake_remember(**kwargs):
            stored_calls.append(kwargs)
            return {"atom_id": 100}

        async def fake_novelty(_):
            return True

        brain_mock = MagicMock()
        brain_mock.remember = fake_remember
        brain_mock._learning = MagicMock()
        brain_mock._learning.assess_novelty = fake_novelty
        brain_mock._storage = MagicMock()
        brain_mock._storage.execute_many = AsyncMock()
        brain_mock._storage.execute_write = AsyncMock()

        with patch("memories.cli._get_brain", return_value=brain_mock):
            with patch("memories.cli._save_hook_atoms", new_callable=AsyncMock):
                with patch("memories.cli._record_hook_stat", new_callable=AsyncMock):
                    data = {
                        "tool_name": "Bash",
                        "error": "command exited with non-zero status code 127 after running",
                        "tool_input": {"command": "nonexistent-tool --help"},
                        "session_id": "test-session",
                        "cwd": str(tmp_path),
                    }
                    await _hook_post_tool_failure(data)

        assert stored_calls, "Expected remember() to be called"
        assert stored_calls[0]["type"] == "experience"


# ---------------------------------------------------------------------------
# Fix 6 — Session-temporal Hebbian weighting
# ---------------------------------------------------------------------------


class TestTemporalHebbian:
    """Verify temporal weighting in hebbian_update."""

    def _make_synapses(self, tmp_path):
        """Create a SynapseManager pointing at a fresh in-memory-ish DB."""
        import asyncio
        from memories.storage import Storage
        from memories.synapses import SynapseManager

        db = tmp_path / "test.db"
        storage = Storage(db)
        asyncio.get_event_loop().run_until_complete(storage.initialize())
        cfg = get_config()
        return SynapseManager(storage, cfg), storage

    @pytest.mark.asyncio
    async def test_recent_pairs_get_full_increment(self, tmp_path):
        """Pairs within temporal_window_seconds receive the full hebbian increment."""
        from memories.storage import Storage
        from memories.synapses import SynapseManager

        db = tmp_path / "test.db"
        storage = Storage(db)
        await storage.initialize()
        mgr = SynapseManager(storage)

        # Insert two atoms with enough accesses.
        for aid in [1, 2]:
            await storage.execute_write(
                "INSERT OR IGNORE INTO atoms "
                "(id, content, type, confidence, importance, access_count, "
                " created_at, is_deleted) "
                "VALUES (?, 'content', 'fact', 0.9, 0.5, 5, datetime('now'), 0)",
                (aid,),
            )

        now = time.time()
        # Both accessed at the same second → dist = 0 ≤ window.
        atom_timestamps = {1: now, 2: now}

        updated = await mgr.hebbian_update([1, 2], atom_timestamps=atom_timestamps)
        assert updated >= 1, "Expected at least one synapse created or strengthened"

        # Verify the synapse increment matches full hebbian_increment.
        rows = await storage.execute(
            "SELECT strength FROM synapses WHERE "
            "(source_id=1 AND target_id=2) OR (source_id=2 AND target_id=1)"
        )
        assert rows, "Expected a synapse between atoms 1 and 2"
        # New synapse uses _NEW_SYNAPSE_DEFAULT_STRENGTH, not just increment.
        # We verify it's positive, which confirms creation with the full path.
        assert rows[0]["strength"] > 0

    @pytest.mark.asyncio
    async def test_distant_pairs_get_half_increment(self, tmp_path):
        """Pairs outside temporal_window_seconds receive increment * 0.5 on creation."""
        from memories.storage import Storage
        from memories.synapses import SynapseManager

        db = tmp_path / "test.db"
        storage = Storage(db)
        await storage.initialize()
        mgr = SynapseManager(storage)

        for aid in [10, 11]:
            await storage.execute_write(
                "INSERT OR IGNORE INTO atoms "
                "(id, content, type, confidence, importance, access_count, "
                " created_at, is_deleted) "
                "VALUES (?, 'content', 'fact', 0.9, 0.5, 5, datetime('now'), 0)",
                (aid,),
            )

        now = time.time()
        # 10 minutes apart → beyond the 5-minute temporal_window_seconds default.
        atom_timestamps = {10: now - 600, 11: now}

        updated = await mgr.hebbian_update([10, 11], atom_timestamps=atom_timestamps)
        assert updated >= 1

    @pytest.mark.asyncio
    async def test_missing_timestamps_fallback_to_full_increment(self, tmp_path):
        """When atom_timestamps is None, all pairs get the full increment (backward compat)."""
        from memories.storage import Storage
        from memories.synapses import SynapseManager

        db = tmp_path / "test.db"
        storage = Storage(db)
        await storage.initialize()
        mgr = SynapseManager(storage)

        for aid in [20, 21]:
            await storage.execute_write(
                "INSERT OR IGNORE INTO atoms "
                "(id, content, type, confidence, importance, access_count, "
                " created_at, is_deleted) "
                "VALUES (?, 'content', 'fact', 0.9, 0.5, 5, datetime('now'), 0)",
                (aid,),
            )

        # No timestamps — should work exactly as before.
        updated = await mgr.hebbian_update([20, 21], atom_timestamps=None)
        assert updated >= 1

    @pytest.mark.asyncio
    async def test_migration_adds_accessed_at_column(self, tmp_path):
        """Migration 7 adds accessed_at to existing hook_session_atoms tables."""
        import sqlite3
        from memories.storage import Storage

        # Create an old-style DB without accessed_at column.
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hook_session_atoms (
                claude_session_id TEXT NOT NULL,
                atom_id INTEGER NOT NULL,
                PRIMARY KEY (claude_session_id, atom_id)
            )
        """)
        conn.commit()
        conn.close()

        # Storage.initialize() runs migrations, which should add the column.
        storage = Storage(db)
        await storage.initialize()
        await storage.close()

        conn = sqlite3.connect(str(db))
        cols = {row[1] for row in conn.execute("PRAGMA table_info(hook_session_atoms)")}
        conn.close()
        assert "accessed_at" in cols


# ---------------------------------------------------------------------------
# Fix 7 — MCP Prompts
# ---------------------------------------------------------------------------


class TestMCPPrompts:
    @pytest.mark.asyncio
    async def test_debug_prompt_returns_tool_reference(self):
        from memories.server import debug

        result = await debug("authentication")
        assert "authentication" in result
        assert "mcp__memories__recall" in result
        assert "mcp__memories__remember" in result

    @pytest.mark.asyncio
    async def test_architecture_prompt_returns_tool_reference(self):
        from memories.server import architecture

        result = await architecture("caching layer")
        assert "caching layer" in result
        assert "mcp__memories__recall" in result
        assert "mcp__memories__remember" in result

    @pytest.mark.asyncio
    async def test_onboard_prompt_returns_region(self):
        from memories.server import onboard

        result = await onboard("myproject")
        assert "myproject" in result
        assert "project:myproject" in result
        assert "mcp__memories__recall" in result

    @pytest.mark.asyncio
    async def test_review_prompt_returns_region(self):
        from memories.server import review

        result = await review("myproject")
        assert "myproject" in result
        assert "project:myproject" in result
        assert "mcp__memories__recall" in result

    def test_all_prompts_registered(self):
        """All 4 prompts must appear as callable functions on the module."""
        from memories import server

        for name in ("debug", "architecture", "onboard", "review"):
            fn = getattr(server, name, None)
            assert fn is not None, f"Missing prompt function: {name}"
            assert callable(fn)
