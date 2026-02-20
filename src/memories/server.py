"""MCP server exposing the brain's capabilities as tools via stdio transport.

This module is the sole interface between the outside world (Claude Code or
any other MCP client) and the :class:`~memories.brain.Brain`.  Each public
method on the Brain is mapped 1-to-1 to an MCP tool with a descriptive
docstring that helps the calling model choose the right tool.

The ``mcp`` object is imported by :mod:`memories.__main__` and launched with
``mcp.run()`` over stdio.

Architecture notes
------------------
* A single global :pydata:`_brain` instance is lazily initialised on the first
  tool call via :func:`_ensure_brain`.
* Empty-string parameters from MCP (which lacks first-class optionals) are
  normalised to ``None`` before forwarding to the Brain.
* All tools catch exceptions and return structured error dicts so the MCP
  server never crashes on a bad request.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from mcp.server.fastmcp import FastMCP

from memories.brain import Brain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server and Brain instances
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "memories",
    instructions="Brain-like memory system with spreading activation",
)

_brain = Brain()


async def _ensure_brain() -> None:
    """Lazily initialise the brain on the first tool call.

    The Brain's :meth:`~memories.brain.Brain.initialize` method is
    idempotent, but we still guard with the ``_initialized`` flag to
    avoid repeated log noise.
    """
    if not _brain._initialized:
        await _brain.initialize()


def _error_response(err: Exception) -> dict[str, Any]:
    """Create a structured error dict for MCP tool responses.

    Instead of letting exceptions propagate and crash the MCP server,
    every tool catches broadly and returns a dict with ``error`` and
    ``detail`` keys so the calling model can understand what went wrong.
    """
    return {
        "error": type(err).__name__,
        "detail": str(err),
        "traceback": traceback.format_exception_only(type(err), err)[-1].strip(),
    }


# ===================================================================
# MCP Tools
# ===================================================================


@mcp.tool()
async def remember(
    content: str,
    type: str,
    region: str = "",
    tags: list[str] | None = None,
    severity: str = "",
    instead: str = "",
    source_project: str = "",
    source_file: str = "",
    importance: float = 0.5,
) -> dict[str, Any]:
    """Store a new memory atom in the brain. Auto-creates synaptic pathways to related existing memories.

    Use this to persist a discrete piece of knowledge. Keep content atomic
    (roughly 50-200 tokens). The brain will automatically find and link
    related memories via embedding similarity.

    Args:
        content: The memory content to store. Keep it atomic and self-contained
            (one concept per atom, roughly 50-200 tokens).
        type: The kind of memory. Must be one of:
            - "fact"         -- A verified piece of knowledge
            - "experience"   -- Something learned from practice
            - "skill"        -- A how-to or technique
            - "preference"   -- A personal or project preference
            - "insight"      -- A derived observation or conclusion
            - "antipattern"  -- A known mistake to avoid (pair with severity/instead)
        region: Brain region for logical grouping. Examples: 'technical',
            'project:myapp', 'personal', 'decisions', 'errors'. Leave empty
            to let the brain auto-infer an appropriate region.
        tags: Optional list of keyword tags for categorization and filtering.
        severity: For antipatterns only. One of: 'low', 'medium', 'high', 'critical'.
            Leave empty for non-antipattern types or to auto-extract from content.
        instead: For antipatterns only. Describes the recommended alternative.
            Leave empty for non-antipattern types or to auto-extract from content.
        source_project: The project this memory originated from (e.g. 'myapp').
        source_file: The file this memory relates to (e.g. 'src/db.py').
        importance: Priority weight in [0, 1] (default 0.5). Higher values make
            the memory surface more prominently during recall. Use 0.7-1.0 for
            critical learnings, antipatterns, or key decisions.

    Returns:
        A dict with keys:
        - atom_id: The ID of the newly created memory atom
        - atom: Full atom details
        - synapses_created: Number of new pathways created to related memories
        - related_atoms: List of related memories that were linked
    """
    try:
        await _ensure_brain()
        return await _brain.remember(
            content=content,
            type=type,
            region=region or None,
            tags=tags,
            severity=severity or None,
            instead=instead or None,
            source_project=source_project or None,
            source_file=source_file or None,
            importance=importance,
        )
    except Exception as exc:
        logger.exception("remember failed")
        return _error_response(exc)


@mcp.tool()
async def recall(
    query: str,
    budget_tokens: int = 2000,
    depth: int = 2,
    region: str = "",
    types: list[str] | None = None,
    include_antipatterns: bool = True,
) -> dict[str, Any]:
    """Search memories using semantic similarity with spreading activation.

    This is the primary retrieval tool. Given a natural-language query, it
    finds the most relevant memories, then follows synaptic connections
    outward (spreading activation) to surface contextually related knowledge
    and any relevant antipatterns as warnings.

    Results are budget-aware: the brain fits as many relevant atoms as
    possible within the token budget and compresses if necessary.

    Args:
        query: Natural language description of what you want to recall.
            Be specific for best results (e.g. "how does Redis handle key
            expiry?" rather than just "Redis").
        budget_tokens: Maximum number of tokens to spend on the response.
            Higher budgets return more memories. Default 2000.
        depth: How many hops of spreading activation to follow from seed
            memories. 1 = direct matches only, 2 = also neighbors, etc.
            Default 2.
        region: Restrict results to a specific brain region (e.g. 'technical',
            'project:myapp'). Leave empty to search all regions.
        types: Restrict results to specific atom types. Pass a list like
            ["fact", "skill"]. Leave empty to include all types.
        include_antipatterns: When true (default), any antipattern atoms
            connected via 'warns-against' synapses are always surfaced
            as warnings alongside the main results.

    Returns:
        A dict with keys:
        - atoms: List of matching memory atoms, most relevant first
        - antipatterns: List of relevant antipattern warnings
        - pathways: Activation paths showing how memories are connected
        - budget_used: Tokens consumed by the response
        - budget_remaining: Tokens still available in the budget
        - total_activated: Total atoms considered during spreading activation
        - seed_count: Number of initial semantic matches
        - compression_level: 0 = no compression, higher = more compressed
    """
    try:
        await _ensure_brain()
        return await _brain.recall(
            query=query,
            budget_tokens=budget_tokens,
            depth=depth,
            region=region or None,
            types=types,
            include_antipatterns=include_antipatterns,
        )
    except Exception as exc:
        logger.exception("recall failed")
        return _error_response(exc)


@mcp.tool()
async def connect(
    source_id: int,
    target_id: int,
    relationship: str,
    strength: float = 0.5,
) -> dict[str, Any]:
    """Create or strengthen a synaptic connection between two memory atoms.

    Use this to manually link two memories that the brain did not
    automatically connect, or to reinforce an existing connection.
    Connections are directional (source -> target) and typed.

    Args:
        source_id: The ID of the source memory atom.
        target_id: The ID of the target memory atom.
        relationship: The type of connection. Must be one of:
            - "related-to"     -- General semantic relationship
            - "caused-by"      -- Target caused/led to source
            - "part-of"        -- Source is a component of target
            - "contradicts"    -- Source contradicts target
            - "supersedes"     -- Source replaces/updates target
            - "elaborates"     -- Source expands on target
            - "warns-against"  -- Source is an antipattern warning for target
        strength: Connection weight from 0.0 (very weak) to 1.0 (very strong).
            Default 0.5. Stronger connections are followed preferentially
            during spreading activation.

    Returns:
        A dict with keys:
        - synapse_id: The ID of the created/updated synapse
        - synapse: Full synapse details
        - source_summary: Preview of the source atom's content
        - target_summary: Preview of the target atom's content
    """
    try:
        await _ensure_brain()
        return await _brain.connect(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            strength=strength,
        )
    except Exception as exc:
        logger.exception("connect failed")
        return _error_response(exc)


@mcp.tool()
async def forget(
    atom_id: int,
    hard: bool = False,
) -> dict[str, Any]:
    """Delete a memory atom from the brain.

    By default performs a soft-delete (the atom is hidden but recoverable).
    Pass hard=True to permanently erase the atom and all its synaptic
    connections. Use with caution -- hard deletes are irreversible.

    Args:
        atom_id: The ID of the memory atom to delete.
        hard: If false (default), soft-delete the atom (can be recovered).
            If true, permanently destroy the atom and all its connections.

    Returns:
        A dict with keys:
        - status: "deleted" on success, "failed" on failure
        - atom_id: The ID of the affected atom
        - hard: Whether this was a hard delete
        - synapses_affected: Number of connections impacted
    """
    try:
        await _ensure_brain()
        return await _brain.forget(
            atom_id=atom_id,
            hard=hard,
        )
    except Exception as exc:
        logger.exception("forget failed")
        return _error_response(exc)


@mcp.tool()
async def amend(
    atom_id: int,
    content: str = "",
    type: str = "",
    tags: list[str] | None = None,
    confidence: float = -1.0,
) -> dict[str, Any]:
    """Update an existing memory atom. Re-embeds and re-links if content changes.

    Use this to correct, refine, or reclassify an existing memory. When
    the content changes, the brain automatically re-computes embeddings
    and creates new synaptic connections to reflect the updated meaning.

    Args:
        atom_id: The ID of the memory atom to update.
        content: New content to replace the existing text. Leave empty to
            keep the current content unchanged.
        type: New atom type. Leave empty to keep the current type. Must be
            one of: fact, experience, skill, preference, insight, antipattern.
        tags: New list of tags to replace the existing tags. Pass null/None
            to keep existing tags. Pass an empty list to clear all tags.
        confidence: New confidence score from 0.0 to 1.0. Pass -1 (default)
            to keep the current confidence unchanged.

    Returns:
        A dict with keys:
        - atom: The updated atom details
        - new_synapses: Number of new connections created after re-linking
        - removed_synapses: Number of stale connections removed
    """
    try:
        await _ensure_brain()
        return await _brain.amend(
            atom_id=atom_id,
            content=content or None,
            type=type or None,
            tags=tags,
            confidence=confidence if confidence >= 0.0 else None,
        )
    except Exception as exc:
        logger.exception("amend failed")
        return _error_response(exc)


@mcp.tool()
async def reflect(
    scope: str = "all",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Trigger memory consolidation -- like sleep for the brain.

    Runs the full consolidation cycle: decay unused connections, prune
    weak synapses, merge near-duplicate atoms, and promote frequently
    accessed memories. Call this periodically to keep the memory system
    healthy and efficient.

    Use dry_run=True to preview what would happen without making changes.

    Args:
        scope: "all" to consolidate every atom, or a specific region name
            (e.g. "technical", "project:myapp") to limit consolidation
            to atoms in that region only.
        dry_run: If true, preview consolidation actions without applying
            any changes. Useful for seeing what would be cleaned up.

    Returns:
        A dict with keys:
        - merged: Number of near-duplicate atoms merged
        - decayed: Number of connections whose strength was reduced
        - pruned: Number of weak connections removed
        - promoted: Number of frequently-accessed atoms promoted
        - compressed: Number of atoms whose content was compressed
        - dry_run: Whether this was a preview-only run
        - details: List of detailed descriptions of each action taken
    """
    try:
        await _ensure_brain()
        return await _brain.reflect(
            scope=scope,
            dry_run=dry_run,
        )
    except Exception as exc:
        logger.exception("reflect failed")
        return _error_response(exc)


@mcp.tool()
async def status() -> dict[str, Any]:
    """Get memory system health and statistics.

    Returns a comprehensive overview of the brain's current state including
    total memories, connections, region distribution, and system health
    indicators. Use this to understand the state of the memory system and
    identify potential issues like orphaned atoms or stale memories.

    Returns:
        A dict with keys:
        - total_atoms: Total number of active memory atoms
        - total_synapses: Total number of synaptic connections
        - regions: List of brain regions with atom counts
        - avg_confidence: Average confidence score across all atoms
        - stale_atoms: Number of atoms not accessed recently
        - orphan_atoms: Number of atoms with no connections
        - db_size_mb: Database file size in megabytes
        - embedding_model: Name of the embedding model in use
        - current_session_id: UUID of the active session
        - ollama_healthy: Whether the embedding service is reachable
    """
    try:
        await _ensure_brain()
        return await _brain.status()
    except Exception as exc:
        logger.exception("status failed")
        return _error_response(exc)


@mcp.tool()
async def pathway(
    atom_id: int,
    depth: int = 2,
    min_strength: float = 0.1,
) -> dict[str, Any]:
    """Visualize the connection graph radiating from a specific memory atom.

    Performs a breadth-first traversal from the given atom, following
    synaptic connections outward up to the specified depth. Returns a
    graph structure (nodes + edges) suitable for understanding how
    memories are interconnected and which clusters they belong to.

    Args:
        atom_id: The ID of the starting memory atom to explore from.
        depth: Maximum number of hops to traverse from the starting atom.
            1 = immediate neighbors only, 2 = neighbors of neighbors, etc.
            Default 2.
        min_strength: Minimum synapse strength to follow during traversal.
            Connections weaker than this threshold are ignored. Range 0.0
            to 1.0, default 0.1.

    Returns:
        A dict with keys:
        - nodes: List of atoms in the subgraph, each with id, content,
            type, and region
        - edges: List of connections, each with source, target,
            relationship, and strength
        - clusters: Dict mapping region names to lists of atom IDs,
            showing how the subgraph groups by region
    """
    try:
        await _ensure_brain()
        return await _brain.pathway(
            atom_id=atom_id,
            depth=depth,
            min_strength=min_strength,
        )
    except Exception as exc:
        logger.exception("pathway failed")
        return _error_response(exc)


@mcp.tool()
async def stats() -> dict[str, Any]:
    """Get hook invocation statistics and memory effectiveness metrics.

    Returns observability data about how the memory system performs across
    Claude Code hook invocations: how often hooks fire, which atoms are
    recalled most, relevance scores, budget utilisation, novelty filter
    pass rates, and latency breakdowns.

    Use this to introspect the memory system's effectiveness mid-session
    and identify tuning opportunities.

    Returns:
        A dict with keys:
        - counts_7d: Hook invocation counts by type (last 7 days)
        - counts_30d: Hook invocation counts by type (last 30 days)
        - counts_all: Hook invocation counts by type (all time)
        - avg_atoms_returned: Average atoms returned per hook type
        - avg_relevance_score: Mean relevance score across all recalls
        - budget_utilisation_pct: Average budget usage percentage
        - unique_atoms_surfaced: Count of distinct atoms ever returned
        - total_atoms: Total active atoms in the system
        - total_impressions: Total atom appearances across all hooks
        - top_atoms: Top 10 most frequently recalled atoms
        - novelty_stats: Pass/fail counts for the novelty filter
        - latency: Average and max latency per hook type (ms)
    """
    try:
        await _ensure_brain()
        return await _brain.get_hook_stats()
    except Exception as exc:
        logger.exception("stats failed")
        return _error_response(exc)


# ===================================================================
# Task Management Tools
# ===================================================================


@mcp.tool()
async def create_task(
    content: str,
    region: str = "",
    tags: list[str] | None = None,
    importance: float = 0.5,
    status: str = "pending",
) -> dict[str, Any]:
    """Create a new task atom with lifecycle tracking.

    Tasks are special memory atoms that can be tracked through a workflow:
    pending → active → done → archived. When tasks complete, their linked
    memories can be automatically flagged as potentially stale.

    Use tasks to:
    - Track work items and their related knowledge
    - Automatically identify outdated memories when work completes
    - Keep the memory graph current and relevant

    Args:
        content: The task description. Be specific about what needs to be done.
        region: Logical grouping for the task. Defaults to 'tasks'.
        tags: Optional list of tags for categorization.
        importance: Priority weight in [0, 1]. Higher = more prominent in recall.
        status: Initial status. One of: 'pending' (default), 'active',
            'done', 'archived'.

    Returns:
        A dict with keys:
        - atom_id: The ID of the created task
        - atom: Full task details including task_status
        - synapses_created: Number of connections to related memories
        - related_atoms: List of memories linked to this task
    """
    try:
        await _ensure_brain()
        return await _brain.create_task(
            content=content,
            region=region or None,
            tags=tags,
            importance=importance,
            status=status,
        )
    except Exception as exc:
        logger.exception("create_task failed")
        return _error_response(exc)


@mcp.tool()
async def update_task(
    task_id: int,
    status: str,
    flag_linked_memories: bool = True,
) -> dict[str, Any]:
    """Update task status and optionally flag linked memories as stale.

    When you complete or archive a task, memories linked to it may become
    outdated. By default, completing a task reduces the confidence of all
    linked memories by 0.1, making them candidates for review during
    consolidation.

    Args:
        task_id: The ID of the task atom to update.
        status: New status. One of: 'pending', 'active', 'done', 'archived'.
        flag_linked_memories: When true (default) and status is 'done' or
            'archived', reduce confidence of linked memories by 0.1.
            This signals they may be stale.

    Returns:
        A dict with keys:
        - task: The updated task details
        - linked_memories_flagged: Number of memories whose confidence was reduced
    """
    try:
        await _ensure_brain()
        return await _brain.update_task(
            task_id=task_id,
            status=status,
            flag_linked_memories=flag_linked_memories,
        )
    except Exception as exc:
        logger.exception("update_task failed")
        return _error_response(exc)


@mcp.tool()
async def list_tasks(
    status: str = "",
    region: str = "",
) -> dict[str, Any]:
    """List task atoms with optional filters.

    Use this to see all tasks or filter by status or region.

    Args:
        status: Filter by status ('pending', 'active', 'done', 'archived').
            Leave empty to return all tasks.
        region: Filter by region. Leave empty to search all regions.

    Returns:
        A dict with keys:
        - tasks: List of task atoms
        - count: Total number of tasks matching filters
    """
    try:
        await _ensure_brain()
        return await _brain.get_tasks(
            status=status or None,
            region=region or None,
        )
    except Exception as exc:
        logger.exception("list_tasks failed")
        return _error_response(exc)


@mcp.tool()
async def stale_memories(
    min_completed_tasks: int = 1,
) -> dict[str, Any]:
    """Find memories linked to completed tasks that may be outdated.

    When tasks complete, the knowledge associated with them may become
    stale. This tool identifies memories connected to completed/archived
    tasks, helping you review and clean up outdated information.

    Use this periodically to:
    - Find knowledge that may need updating
    - Identify candidates for archival or deletion
    - Keep the memory graph current and relevant

    Args:
        min_completed_tasks: Minimum number of completed tasks a memory
            must be linked to in order to be flagged as stale. Default 1.

    Returns:
        A dict with keys:
        - memories: List of potentially stale memory atoms
        - count: Number of stale memories found
        - linked_tasks: Mapping of memory ID to list of completed task IDs
    """
    try:
        await _ensure_brain()
        return await _brain.get_stale_memories(
            min_completed_tasks=min_completed_tasks,
        )
    except Exception as exc:
        logger.exception("stale_memories failed")
        return _error_response(exc)


# ---------------------------------------------------------------------------
# MCP Prompts — reusable instruction templates for common workflows
# ---------------------------------------------------------------------------


@mcp.prompt()
async def debug(topic: str) -> str:
    """Use before debugging — recalls known antipatterns and error patterns for a topic."""
    return (
        f"Before starting to debug '{topic}', call mcp__memories__recall with the query "
        f"'{topic} error antipattern bug' to surface known failure modes and solutions. "
        f"After resolving the issue, store the root cause and fix as an antipattern atom "
        f"using mcp__memories__remember so it is not investigated again."
    )


@mcp.prompt()
async def architecture(topic: str) -> str:
    """Use before architectural decisions — recalls prior decisions and patterns."""
    return (
        f"Before making architectural decisions about '{topic}', call "
        f"mcp__memories__recall with '{topic} architecture decision pattern' to retrieve "
        f"prior findings. Build on what is already known. After deciding, store the "
        f"rationale as an insight atom using mcp__memories__remember."
    )


@mcp.prompt()
async def onboard(project: str) -> str:
    """Use when starting work on a project — recalls everything known about it."""
    return (
        f"To orient yourself on '{project}', call mcp__memories__recall with "
        f"region='project:{project}' and queries for 'architecture', 'known issues', "
        f"and 'conventions'. Review the returned atoms before writing any code."
    )


@mcp.prompt()
async def review(project: str) -> str:
    """Use before code review — recalls project conventions, antipatterns and decisions."""
    return (
        f"Before reviewing code in '{project}', call mcp__memories__recall with "
        f"region='project:{project}' and query 'antipattern convention decision' to "
        f"surface known issues to check for. After review, store any new findings."
    )
