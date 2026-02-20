"""Comprehensive tests for the memories.server MCP tool layer.

Tests cover:
- Parameter normalization (empty strings → None)
- _error_response structured error formatting
- All 8 tool endpoints delegating to Brain correctly
- Error handling in every tool (returns error dict, never raises)
- Lazy brain initialization via _ensure_brain
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memories.brain import Brain
import memories.server as server_module
from memories.server import (
    remember,
    recall,
    connect,
    forget,
    amend,
    reflect,
    status,
    pathway,
    _ensure_brain,
    _error_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_brain():
    brain = MagicMock(spec=Brain)
    brain._initialized = True
    brain.initialize = AsyncMock()
    brain.remember = AsyncMock(
        return_value={
            "atom_id": 1,
            "atom": {},
            "synapses_created": 0,
            "related_atoms": [],
        }
    )
    brain.recall = AsyncMock(
        return_value={
            "atoms": [],
            "antipatterns": [],
            "pathways": [],
            "budget_used": 0,
            "budget_remaining": 2000,
            "total_activated": 0,
            "seed_count": 0,
            "compression_level": 0,
        }
    )
    brain.connect = AsyncMock(
        return_value={
            "synapse_id": 1,
            "synapse": {},
            "source_summary": "",
            "target_summary": "",
        }
    )
    brain.forget = AsyncMock(
        return_value={
            "status": "deleted",
            "atom_id": 1,
            "hard": False,
            "synapses_affected": 0,
        }
    )
    brain.amend = AsyncMock(
        return_value={
            "atom": {},
            "new_synapses": [],
            "removed_synapses": [],
        }
    )
    brain.reflect = AsyncMock(
        return_value={
            "merged": 0,
            "decayed": 0,
            "pruned": 0,
            "promoted": 0,
            "compressed": 0,
            "dry_run": False,
            "details": [],
        }
    )
    brain.status = AsyncMock(
        return_value={
            "total_atoms": 0,
            "total_synapses": 0,
            "regions": [],
            "avg_confidence": 0,
            "stale_atoms": 0,
            "orphan_atoms": 0,
            "db_size_mb": 0.0,
            "embedding_model": "nomic-embed-text",
            "current_session_id": "test",
            "ollama_healthy": True,
        }
    )
    brain.pathway = AsyncMock(
        return_value={
            "nodes": [],
            "edges": [],
            "clusters": {},
        }
    )
    return brain


@pytest.fixture(autouse=True)
def patch_brain(mock_brain):
    with patch.object(server_module, "_brain", mock_brain):
        yield mock_brain


# ===================================================================
# TestParameterNormalization
# ===================================================================


class TestParameterNormalization:
    """Verify that empty strings sent by MCP clients are converted to None."""

    async def test_remember_empty_region_becomes_none(self, mock_brain):
        await remember(content="test", type="fact", region="")
        _, kwargs = mock_brain.remember.call_args
        assert kwargs["region"] is None

    async def test_remember_empty_severity_becomes_none(self, mock_brain):
        await remember(content="test", type="fact", severity="")
        _, kwargs = mock_brain.remember.call_args
        assert kwargs["severity"] is None

    async def test_remember_empty_instead_becomes_none(self, mock_brain):
        await remember(content="test", type="fact", instead="")
        _, kwargs = mock_brain.remember.call_args
        assert kwargs["instead"] is None

    async def test_remember_empty_source_project_becomes_none(self, mock_brain):
        await remember(content="test", type="fact", source_project="")
        _, kwargs = mock_brain.remember.call_args
        assert kwargs["source_project"] is None

    async def test_remember_empty_source_file_becomes_none(self, mock_brain):
        await remember(content="test", type="fact", source_file="")
        _, kwargs = mock_brain.remember.call_args
        assert kwargs["source_file"] is None

    async def test_remember_all_empty_strings_become_none(self, mock_brain):
        await remember(
            content="test",
            type="fact",
            region="",
            severity="",
            instead="",
            source_project="",
            source_file="",
        )
        _, kwargs = mock_brain.remember.call_args
        assert kwargs["region"] is None
        assert kwargs["severity"] is None
        assert kwargs["instead"] is None
        assert kwargs["source_project"] is None
        assert kwargs["source_file"] is None

    async def test_remember_nonempty_values_preserved(self, mock_brain):
        await remember(
            content="test",
            type="fact",
            region="technical",
            severity="high",
            instead="use X",
            source_project="myapp",
            source_file="src/db.py",
        )
        _, kwargs = mock_brain.remember.call_args
        assert kwargs["region"] == "technical"
        assert kwargs["severity"] == "high"
        assert kwargs["instead"] == "use X"
        assert kwargs["source_project"] == "myapp"
        assert kwargs["source_file"] == "src/db.py"

    async def test_recall_empty_region_becomes_none(self, mock_brain):
        await recall(query="test query", region="")
        _, kwargs = mock_brain.recall.call_args
        assert kwargs["region"] is None

    async def test_recall_nonempty_region_preserved(self, mock_brain):
        await recall(query="test query", region="technical")
        _, kwargs = mock_brain.recall.call_args
        assert kwargs["region"] == "technical"

    async def test_amend_empty_content_becomes_none(self, mock_brain):
        await amend(atom_id=1, content="")
        _, kwargs = mock_brain.amend.call_args
        assert kwargs["content"] is None

    async def test_amend_empty_type_becomes_none(self, mock_brain):
        await amend(atom_id=1, type="")
        _, kwargs = mock_brain.amend.call_args
        assert kwargs["type"] is None

    async def test_amend_negative_confidence_becomes_none(self, mock_brain):
        await amend(atom_id=1, confidence=-1.0)
        _, kwargs = mock_brain.amend.call_args
        assert kwargs["confidence"] is None

    async def test_amend_valid_confidence_preserved(self, mock_brain):
        await amend(atom_id=1, confidence=0.8)
        _, kwargs = mock_brain.amend.call_args
        assert kwargs["confidence"] == 0.8

    async def test_amend_zero_confidence_preserved(self, mock_brain):
        """Confidence of 0.0 is a valid value and should not be turned to None."""
        await amend(atom_id=1, confidence=0.0)
        _, kwargs = mock_brain.amend.call_args
        assert kwargs["confidence"] == 0.0


# ===================================================================
# TestErrorResponse
# ===================================================================


class TestErrorResponse:
    """Verify _error_response produces the correct structured dict."""

    def test_error_response_keys(self):
        result = _error_response(ValueError("bad value"))
        assert set(result.keys()) == {"error", "detail", "traceback"}

    def test_error_response_uses_class_name(self):
        result = _error_response(TypeError("wrong type"))
        assert result["error"] == "TypeError"

    def test_error_response_detail_is_message(self):
        result = _error_response(RuntimeError("something broke"))
        assert result["detail"] == "something broke"

    def test_error_response_traceback_is_string(self):
        result = _error_response(KeyError("missing"))
        assert isinstance(result["traceback"], str)
        assert "KeyError" in result["traceback"]

    def test_error_response_custom_exception(self):
        class BrainMeltdown(Exception):
            pass

        result = _error_response(BrainMeltdown("overload"))
        assert result["error"] == "BrainMeltdown"
        assert result["detail"] == "overload"


# ===================================================================
# TestToolEndpoints
# ===================================================================


class TestToolEndpoints:
    """Each tool delegates to the brain and returns its result (or error dict)."""

    async def test_remember_delegates(self, mock_brain):
        result = await remember(content="hello world", type="fact")
        mock_brain.remember.assert_awaited_once()
        assert result["atom_id"] == 1

    async def test_remember_returns_error_on_exception(self, mock_brain):
        mock_brain.remember = AsyncMock(side_effect=RuntimeError("db fail"))
        result = await remember(content="x", type="fact")
        assert "error" in result
        assert result["error"] == "RuntimeError"
        assert result["detail"] == "db fail"

    async def test_recall_delegates(self, mock_brain):
        result = await recall(query="redis caching")
        mock_brain.recall.assert_awaited_once_with(
            query="redis caching",
            budget_tokens=2000,
            depth=2,
            region=None,
            types=None,
            include_antipatterns=True,
        )
        assert "atoms" in result

    async def test_recall_returns_error_on_exception(self, mock_brain):
        mock_brain.recall = AsyncMock(side_effect=ValueError("bad query"))
        result = await recall(query="test")
        assert result["error"] == "ValueError"

    async def test_connect_delegates(self, mock_brain):
        result = await connect(source_id=1, target_id=2, relationship="related-to")
        mock_brain.connect.assert_awaited_once_with(
            source_id=1,
            target_id=2,
            relationship="related-to",
            strength=0.5,
        )
        assert result["synapse_id"] == 1

    async def test_connect_returns_error_on_exception(self, mock_brain):
        mock_brain.connect = AsyncMock(side_effect=KeyError("no such atom"))
        result = await connect(source_id=99, target_id=100, relationship="part-of")
        assert result["error"] == "KeyError"

    async def test_forget_delegates(self, mock_brain):
        result = await forget(atom_id=5, hard=True)
        mock_brain.forget.assert_awaited_once_with(atom_id=5, hard=True)
        assert result["status"] == "deleted"

    async def test_forget_returns_error_on_exception(self, mock_brain):
        mock_brain.forget = AsyncMock(side_effect=RuntimeError("locked"))
        result = await forget(atom_id=1)
        assert result["error"] == "RuntimeError"

    async def test_amend_delegates(self, mock_brain):
        result = await amend(atom_id=3, content="updated", type="skill")
        mock_brain.amend.assert_awaited_once_with(
            atom_id=3,
            content="updated",
            type="skill",
            tags=None,
            confidence=None,
        )
        assert "atom" in result

    async def test_amend_returns_error_on_exception(self, mock_brain):
        mock_brain.amend = AsyncMock(side_effect=TypeError("bad type"))
        result = await amend(atom_id=1)
        assert result["error"] == "TypeError"

    async def test_reflect_delegates(self, mock_brain):
        result = await reflect(scope="technical", dry_run=True)
        mock_brain.reflect.assert_awaited_once_with(
            scope="technical",
            dry_run=True,
        )
        assert result["dry_run"] is False  # from mock return value

    async def test_reflect_returns_error_on_exception(self, mock_brain):
        mock_brain.reflect = AsyncMock(side_effect=OSError("disk full"))
        result = await reflect()
        assert result["error"] == "OSError"

    async def test_status_delegates(self, mock_brain):
        result = await status()
        mock_brain.status.assert_awaited_once_with()
        assert result["embedding_model"] == "nomic-embed-text"

    async def test_status_returns_error_on_exception(self, mock_brain):
        mock_brain.status = AsyncMock(side_effect=ConnectionError("no db"))
        result = await status()
        assert result["error"] == "ConnectionError"

    async def test_pathway_delegates(self, mock_brain):
        result = await pathway(atom_id=7, depth=3, min_strength=0.2)
        mock_brain.pathway.assert_awaited_once_with(
            atom_id=7,
            depth=3,
            min_strength=0.2,
        )
        assert "nodes" in result
        assert "edges" in result
        assert "clusters" in result

    async def test_pathway_returns_error_on_exception(self, mock_brain):
        mock_brain.pathway = AsyncMock(side_effect=IndexError("out of range"))
        result = await pathway(atom_id=999)
        assert result["error"] == "IndexError"


# ===================================================================
# TestEnsureBrain
# ===================================================================


class TestEnsureBrain:
    """Lazy initialization: _ensure_brain calls initialize() only when needed."""

    async def test_ensure_brain_initializes_when_not_ready(self, mock_brain):
        mock_brain._initialized = False
        await _ensure_brain()
        mock_brain.initialize.assert_awaited_once()

    async def test_ensure_brain_noop_when_already_initialized(self, mock_brain):
        mock_brain._initialized = True
        await _ensure_brain()
        mock_brain.initialize.assert_not_awaited()

    async def test_ensure_brain_called_by_every_tool(self, mock_brain):
        """Each tool invocation triggers _ensure_brain internally.

        We verify by setting _initialized=False and checking initialize()
        is called once per tool invocation.
        """
        mock_brain._initialized = False

        await remember(content="x", type="fact")
        assert mock_brain.initialize.await_count == 1

        # Reset for next tool
        mock_brain.initialize.reset_mock()
        mock_brain._initialized = False

        await recall(query="test")
        assert mock_brain.initialize.await_count == 1

        mock_brain.initialize.reset_mock()
        mock_brain._initialized = False

        await connect(source_id=1, target_id=2, relationship="related-to")
        assert mock_brain.initialize.await_count == 1

        mock_brain.initialize.reset_mock()
        mock_brain._initialized = False

        await forget(atom_id=1)
        assert mock_brain.initialize.await_count == 1

        mock_brain.initialize.reset_mock()
        mock_brain._initialized = False

        await amend(atom_id=1)
        assert mock_brain.initialize.await_count == 1

        mock_brain.initialize.reset_mock()
        mock_brain._initialized = False

        await reflect()
        assert mock_brain.initialize.await_count == 1

        mock_brain.initialize.reset_mock()
        mock_brain._initialized = False

        await status()
        assert mock_brain.initialize.await_count == 1

        mock_brain.initialize.reset_mock()
        mock_brain._initialized = False

        await pathway(atom_id=1)
        assert mock_brain.initialize.await_count == 1
