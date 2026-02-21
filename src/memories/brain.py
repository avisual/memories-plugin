"""Central orchestrator for the brain-like memory system.

The :class:`Brain` coordinates all subsystems -- storage, embeddings, atoms,
synapses, retrieval, learning, consolidation, and context budgeting -- into a
single high-level API surface that the MCP server calls.

There is **one Brain per MCP server process**.  All public methods return plain
dicts (not dataclasses) because their output is JSON-serialised for MCP tool
responses.

Session management tracks which atoms are accessed together within a single
conversation, enabling Hebbian co-activation learning at session end.

Usage::

    from memories.brain import Brain

    brain = Brain()
    await brain.initialize()

    result = await brain.remember("Redis SCAN is O(N)", type="fact")
    memories = await brain.recall("How does Redis SCAN work?")
    await brain.shutdown()
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from memories.atoms import AtomManager
from memories.config import get_config
from memories.consolidation import ConsolidationEngine, ConsolidationResult
from memories.context import ContextBudget
from memories.embeddings import EmbeddingEngine
from memories.learning import LearningEngine
from memories.retrieval import RecallResult, RetrievalEngine
from memories.storage import Storage
from memories.synapses import SynapseManager

logger = logging.getLogger(__name__)


class Brain:
    """The central orchestrator.  One brain per process.

    All components are lazily initialised via :meth:`initialize` and cleaned
    up via :meth:`shutdown`.  Between those two calls the brain maintains a
    session that tracks atom co-activations for Hebbian learning.

    Typical lifecycle::

        brain = Brain()
        await brain.initialize()    # sets up DB, engines, session
        ...                         # MCP tool calls
        await brain.shutdown()      # Hebbian learning, close DB
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._config = get_config()
        self._db_path_override = db_path
        self._storage: Storage | None = None
        self._embeddings: EmbeddingEngine | None = None
        self._atoms: AtomManager | None = None
        self._synapses: SynapseManager | None = None
        self._retrieval: RetrievalEngine | None = None
        self._learning: LearningEngine | None = None
        self._consolidation: ConsolidationEngine | None = None
        self._context: ContextBudget | None = None
        self._current_session_id: str | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize all components.  Must be called before any operations.

        This method is idempotent -- calling it multiple times is safe and
        only the first call performs actual work.

        Startup sequence:

        1. Create :class:`Storage` and initialise (creates DB, tables, backup).
        2. Create :class:`EmbeddingEngine`.
        3. Create :class:`AtomManager`.
        4. Create :class:`SynapseManager`.
        5. Create :class:`RetrievalEngine`.
        6. Create :class:`LearningEngine`.
        7. Create :class:`ConsolidationEngine`.
        8. Create :class:`ContextBudget`.
        9. Start a new session.
        """
        if self._initialized:
            return

        self._storage = Storage(self._db_path_override or Path(self._config.db_path))
        await self._storage.initialize()

        self._embeddings = EmbeddingEngine(self._storage)
        self._atoms = AtomManager(self._storage, self._embeddings)
        self._synapses = SynapseManager(self._storage)
        self._context = ContextBudget()

        stored_w = await self._storage.load_retrieval_weights()
        weight_override = None
        if stored_w:
            from memories.config import RetrievalWeights
            weight_override = RetrievalWeights(**stored_w)

        self._retrieval = RetrievalEngine(
            self._storage,
            self._embeddings,
            self._atoms,
            self._synapses,
            self._context,
            weight_override=weight_override,
        )
        self._learning = LearningEngine(
            self._storage,
            self._embeddings,
            self._atoms,
            self._synapses,
        )
        self._consolidation = ConsolidationEngine(
            self._storage,
            self._embeddings,
            self._atoms,
            self._synapses,
        )

        self._initialized = True

        await self._start_session()
        logger.info("Brain initialized. DB: %s", self._config.db_path)

    def _ensure_initialized(self) -> None:
        """Guard that raises if the brain has not been initialised yet."""
        if not self._initialized:
            raise RuntimeError(
                "Brain not initialized. Call await brain.initialize() first."
            )

    # ==================================================================
    # MCP Tool Methods
    # ==================================================================

    async def remember(
        self,
        content: str,
        type: str,
        region: str | None = None,
        tags: list[str] | None = None,
        severity: str | None = None,
        instead: str | None = None,
        source_project: str | None = None,
        source_file: str | None = None,
        importance: float = 0.5,
    ) -> dict[str, Any]:
        """Store a new memory atom.  Auto-creates pathways to related atoms.

        Parameters
        ----------
        content:
            The knowledge payload to store.
        type:
            Atom type (one of :data:`~memories.atoms.ATOM_TYPES`).
        region:
            Logical grouping.  When ``None`` the learning engine
            infers an appropriate region from the content.
        tags:
            Optional list of string tags.
        severity:
            Antipattern-only severity level (auto-extracted if not given).
        instead:
            Antipattern-only recommended alternative (auto-extracted if not
            given).
        source_project:
            Project that produced this memory.
        source_file:
            File that the memory originated from.
        importance:
            Priority weight in ``[0, 1]`` (default ``0.5``).  Higher values
            surface the memory more prominently during recall.  Use for
            critical memories that should be prioritised.

        Returns
        -------
        dict
            Keys: ``atom_id``, ``atom``, ``synapses_created``,
            ``related_atoms``.
        """
        self._ensure_initialized()
        assert self._learning is not None
        assert self._atoms is not None
        assert self._embeddings is not None

        # 0. Pre-insertion semantic dedup: skip if a near-identical atom exists.
        if self._storage and self._storage.vec_available:
            candidates = await self._embeddings.search_similar(content, k=1)
            if candidates:
                existing_id, distance = candidates[0]
                similarity = EmbeddingEngine.distance_to_similarity(distance)
                if similarity >= self._config.dedup_threshold:
                    existing = await self._atoms.get_without_tracking(existing_id)
                    if existing is not None and not existing.is_deleted:
                        logger.debug(
                            "Dedup: content similar to atom %d (score=%.3f), skipping",
                            existing_id,
                            similarity,
                        )
                        self.track_atom_access(existing_id)
                        return {
                            "atom_id": existing_id,
                            "atom": existing.to_dict(),
                            "synapses_created": 0,
                            "related_atoms": [],
                            "deduplicated": True,
                        }

        # 1. Infer region if not specified.
        if region is None:
            region = await self._learning.suggest_region(
                content, source_project=source_project
            )

        # 2. For antipatterns, auto-extract severity/instead if not provided.
        if type == "antipattern":
            if severity is None or instead is None:
                extracted_severity, extracted_instead = (
                    await self._learning.extract_antipattern_fields(content)
                )
                if severity is None:
                    severity = extracted_severity
                if instead is None:
                    instead = extracted_instead

        # 3. Create the atom.
        atom = await self._atoms.create(
            content=content,
            type=type,
            region=region,
            tags=tags,
            severity=severity,
            instead=instead,
            source_project=source_project,
            source_session=self._current_session_id,
            source_file=source_file,
            importance=importance,
        )

        # 4. Auto-link to related atoms.
        # auto_link() already creates warns-against synapses for antipatterns,
        # so a separate detect_antipattern_links() call is unnecessary.
        created_synapses = await self._learning.auto_link(atom.id)

        # 5. Detect supersession.
        supersede_count = await self._learning.detect_supersedes(atom.id)

        # 5b. Contextual encoding: create lightweight encoded-with synapses
        # to atoms co-active in the current session.  Implements encoding
        # specificity (Tulving & Thomson 1973) — memories encoded in a
        # context are better recalled when that context reappears.
        context_links = 0
        if self._current_session_id and self._storage:
            try:
                context_rows = await self._storage.execute(
                    "SELECT atom_id FROM hook_session_atoms WHERE claude_session_id = ?",
                    (self._current_session_id,),
                )
                context_ids = [r["atom_id"] for r in context_rows if r["atom_id"] != atom.id]
                # Cap to prevent explosion in large sessions.
                _MAX_CONTEXT_LINKS = 10
                for ctx_id in context_ids[:_MAX_CONTEXT_LINKS]:
                    try:
                        syn = await self._synapses.create(
                            source_id=atom.id,
                            target_id=ctx_id,
                            relationship="encoded-with",
                            strength=0.15,
                            bidirectional=True,
                        )
                        if syn is not None:
                            context_links += 1
                    except (ValueError, RuntimeError):
                        pass
                if context_links:
                    logger.debug(
                        "Contextual encoding: created %d encoded-with links for atom %d",
                        context_links,
                        atom.id,
                    )
            except Exception:
                # Contextual encoding is best-effort; never block remember().
                pass

        # Derive antipattern count from the synapses auto_link() created.
        antipattern_count = sum(
            1 for s in created_synapses if s.get("relationship") == "warns-against"
        )

        # 6. Track atom in session.
        self.track_atom_access(atom.id)

        # 7. Collect related atom summaries from the created synapses.
        # Batch-fetch all unique related atom IDs in a single query.
        related_ids: list[int] = []
        seen_ids: set[int] = set()
        for syn_info in created_synapses:
            related_id = (
                syn_info["target_id"]
                if syn_info["source_id"] == atom.id
                else syn_info["source_id"]
            )
            if related_id not in seen_ids:
                seen_ids.add(related_id)
                related_ids.append(related_id)

        related_map = await self._atoms.get_batch_without_tracking(related_ids)

        related_atoms: list[dict[str, Any]] = []
        seen_for_output: set[int] = set()
        for syn_info in created_synapses:
            related_id = (
                syn_info["target_id"]
                if syn_info["source_id"] == atom.id
                else syn_info["source_id"]
            )
            if related_id in seen_for_output:
                continue
            seen_for_output.add(related_id)

            related = related_map.get(related_id)
            if related is not None:
                related_atoms.append({
                    "id": related.id,
                    "content": related.content[:120],
                    "type": related.type,
                    "region": related.region,
                    "relationship": syn_info["relationship"],
                    "strength": syn_info["strength"],
                })

        logger.info(
            "Remembered atom %d (%s, region=%s): %d synapses, "
            "%d antipattern links, %d supersedes",
            atom.id,
            type,
            region,
            len(created_synapses),
            antipattern_count,
            supersede_count,
        )

        return {
            "atom_id": atom.id,
            "atom": atom.to_dict(),
            "synapses_created": len(created_synapses),
            "related_atoms": related_atoms,
        }

    async def recall(
        self,
        query: str,
        budget_tokens: int = 2000,
        depth: int | None = None,
        region: str | None = None,
        types: list[str] | None = None,
        include_antipatterns: bool = True,
    ) -> dict[str, Any]:
        """Semantic search with spreading activation.

        Delegates to :meth:`RetrievalEngine.recall` and tracks all accessed
        atoms in the session for Hebbian learning.

        Parameters
        ----------
        query:
            Natural language query to search for.
        budget_tokens:
            Maximum number of tokens to spend on the result.
        depth:
            Spreading activation depth (hops from seed atoms).  Defaults to
            the configured ``retrieval.spread_depth``.
        region:
            If set, only return atoms belonging to this region.
        types:
            If set, only return atoms whose type is in this list.
        include_antipatterns:
            When True, antipattern atoms connected via ``warns-against``
            synapses are always surfaced.

        Returns
        -------
        dict
            RecallResult serialised as a dict with keys: ``atoms``,
            ``antipatterns``, ``pathways``, ``budget_used``,
            ``budget_remaining``, ``total_activated``, ``seed_count``,
            ``compression_level``.
        """
        self._ensure_initialized()
        assert self._retrieval is not None
        assert self._storage is not None

        # Read session atom IDs from the hook_session_atoms DB table
        # (populated by CLI hooks via _save_hook_atoms).
        session_atom_ids: list[int] = []
        if self._current_session_id:
            rows = await self._storage.execute(
                "SELECT atom_id FROM hook_session_atoms WHERE claude_session_id = ?",
                (self._current_session_id,),
            )
            session_atom_ids = [row["atom_id"] for row in rows]

        result: RecallResult = await self._retrieval.recall(
            query=query,
            budget_tokens=budget_tokens,
            depth=depth,
            region=region,
            types=types,
            include_antipatterns=include_antipatterns,
            session_atom_ids=session_atom_ids,
        )

        return {
            "atoms": result.atoms,
            "antipatterns": result.antipatterns,
            "pathways": result.pathways,
            "budget_used": result.budget_used,
            "budget_remaining": result.budget_remaining,
            "total_activated": result.total_activated,
            "seed_count": result.seed_count,
            "compression_level": result.compression_level,
        }

    async def connect(
        self,
        source_id: int,
        target_id: int,
        relationship: str,
        strength: float = 0.5,
    ) -> dict[str, Any]:
        """Manually create or strengthen a synapse between two atoms.

        Parameters
        ----------
        source_id:
            Foreign key to the source atom.
        target_id:
            Foreign key to the target atom.
        relationship:
            One of :data:`~memories.synapses.RELATIONSHIP_TYPES`.
        strength:
            Initial connection weight in ``[0, 1]``.

        Returns
        -------
        dict
            Keys: ``synapse_id``, ``synapse``, ``source_summary``,
            ``target_summary``.

        Raises
        ------
        ValueError
            If either atom does not exist or the relationship is invalid.
        """
        self._ensure_initialized()
        assert self._atoms is not None
        assert self._synapses is not None

        # Validate both atoms exist.
        source_atom = await self._atoms.get_without_tracking(source_id)
        if source_atom is None:
            raise ValueError(f"Source atom {source_id} not found")

        target_atom = await self._atoms.get_without_tracking(target_id)
        if target_atom is None:
            raise ValueError(f"Target atom {target_id} not found")

        synapse = await self._synapses.create(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            strength=strength,
        )

        return {
            "synapse_id": synapse.id,
            "synapse": synapse.to_dict(),
            "source_summary": source_atom.content[:120],
            "target_summary": target_atom.content[:120],
        }

    async def forget(self, atom_id: int, hard: bool = False) -> dict[str, Any]:
        """Soft-delete or hard-delete an atom.

        Parameters
        ----------
        atom_id:
            Primary key of the atom to forget.
        hard:
            If ``True``, permanently remove the atom and all its synapses
            from the database.  If ``False`` (default), the atom is
            soft-deleted and can be recovered.

        Returns
        -------
        dict
            Keys: ``status``, ``atom_id``, ``hard``, ``synapses_affected``.

        Raises
        ------
        ValueError
            If the atom does not exist.
        """
        self._ensure_initialized()
        assert self._atoms is not None
        assert self._synapses is not None

        # Confirm the atom exists.
        atom = await self._atoms.get_without_tracking(atom_id)
        if atom is None:
            raise ValueError(f"Atom {atom_id} not found")

        synapses_affected = 0

        if hard:
            synapses_affected = await self._synapses.delete_for_atom(atom_id)
            success = await self._atoms.hard_delete(atom_id)
        else:
            # Count synapses before soft-delete for reporting.
            connections = await self._synapses.get_connections(atom_id)
            synapses_affected = len(connections)
            success = await self._atoms.soft_delete(atom_id)

        status = "deleted" if success else "failed"
        logger.info(
            "Forgot atom %d (hard=%s): %s, %d synapses affected",
            atom_id,
            hard,
            status,
            synapses_affected,
        )

        return {
            "status": status,
            "atom_id": atom_id,
            "hard": hard,
            "synapses_affected": synapses_affected,
        }

    async def amend(
        self,
        atom_id: int,
        content: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
        importance: float | None = None,
    ) -> dict[str, Any]:
        """Update an existing atom.  Re-embeds and re-links if content changed.

        Parameters
        ----------
        atom_id:
            Primary key of the atom to update.
        content:
            New knowledge payload.
        type:
            New atom type.
        tags:
            Replacement tag list.
        confidence:
            New confidence value in ``[0, 1]``.
        importance:
            New importance value in ``[0, 1]``.

        Returns
        -------
        dict
            Keys: ``atom``, ``new_synapses``, ``removed_synapses``.

        Raises
        ------
        ValueError
            If the atom does not exist.
        """
        self._ensure_initialized()
        assert self._atoms is not None
        assert self._learning is not None
        assert self._synapses is not None

        # Capture existing synapses before the update.
        existing_connections = await self._synapses.get_connections(atom_id)
        existing_synapse_ids = {s.id for s in existing_connections}

        updated_atom = await self._atoms.update(
            atom_id,
            content=content,
            type=type,
            tags=tags,
            confidence=confidence,
            importance=importance,
        )
        if updated_atom is None:
            raise ValueError(f"Atom {atom_id} not found")

        new_synapses: list[dict[str, Any]] = []
        removed_synapses: list[int] = []

        # If content changed, re-run auto-linking.
        if content is not None:
            new_synapses = await self._learning.auto_link(atom_id)

            # Identify synapses that were removed (existed before but no
            # longer present after re-linking).  In practice auto_link only
            # adds -- it does not remove -- so removed will typically be
            # empty.  This provides forward-compatibility if the learning
            # engine evolves to prune stale links on update.
            post_connections = await self._synapses.get_connections(atom_id)
            post_synapse_ids = {s.id for s in post_connections}
            removed_synapses = sorted(existing_synapse_ids - post_synapse_ids)

        logger.info(
            "Amended atom %d: %d new synapses, %d removed",
            atom_id,
            len(new_synapses),
            len(removed_synapses),
        )

        return {
            "atom": updated_atom.to_dict(),
            "new_synapses": new_synapses,
            "removed_synapses": removed_synapses,
        }

    async def reflect(
        self,
        scope: str = "all",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Trigger consolidation -- like sleep for the brain.

        Runs the full consolidation cycle (decay, prune, merge, promote)
        and then optimises the database if this is not a dry run.

        Parameters
        ----------
        scope:
            ``"all"`` to process every atom, or a region name to limit
            consolidation to atoms in that region.
        dry_run:
            If ``True`` the engine previews what *would* happen without
            making any database mutations.

        Returns
        -------
        dict
            ConsolidationResult serialised as a dict with keys: ``merged``,
            ``decayed``, ``pruned``, ``promoted``, ``compressed``,
            ``dry_run``, ``details``.
        """
        self._ensure_initialized()
        assert self._consolidation is not None
        assert self._storage is not None

        result: ConsolidationResult = await self._consolidation.reflect(
            scope=scope,
            dry_run=dry_run,
        )

        # Optimise the database after a real consolidation cycle.
        if not dry_run:
            await self._storage.optimize()

        logger.info(
            "Reflect complete (dry_run=%s): merged=%d, decayed=%d, "
            "pruned=%d, promoted=%d",
            dry_run,
            result.merged,
            result.decayed,
            result.pruned,
            result.promoted,
        )

        return result.to_dict()

    async def status(self) -> dict[str, Any]:
        """Memory system health and stats.

        Returns
        -------
        dict
            Keys: ``total_atoms``, ``total_synapses``, ``regions``,
            ``avg_confidence``, ``stale_atoms``, ``orphan_atoms``,
            ``db_size_mb``, ``embedding_model``, ``current_session_id``,
            ``ollama_healthy``.
        """
        self._ensure_initialized()
        assert self._storage is not None
        assert self._atoms is not None
        assert self._synapses is not None
        assert self._embeddings is not None

        # Gather atom statistics.
        atom_stats = await self._atoms.get_stats()

        # Gather synapse statistics.
        synapse_stats = await self._synapses.get_stats()

        # Regions with counts.
        region_rows = await self._storage.execute(
            "SELECT name, atom_count FROM regions ORDER BY atom_count DESC"
        )
        regions = [
            {"name": row["name"], "atom_count": row["atom_count"]}
            for row in region_rows
        ]

        # Stale atoms (not accessed in 30 days).
        stale_atoms = await self._atoms.get_stale(
            self._config.consolidation.decay_after_days
        )

        # Orphan atoms (no synapses connected).
        orphan_rows = await self._storage.execute(
            """
            SELECT COUNT(*) AS cnt FROM atoms a
            WHERE a.is_deleted = 0
              AND NOT EXISTS (
                  SELECT 1 FROM synapses s
                  WHERE s.source_id = a.id OR s.target_id = a.id
              )
            """
        )
        orphan_count = orphan_rows[0]["cnt"] if orphan_rows else 0

        # Database size.
        db_size_mb = await self._storage.get_db_size_mb()

        # Ollama health check.
        ollama_healthy = await self._embeddings.health_check()

        return {
            "total_atoms": atom_stats["total"],
            "total_synapses": synapse_stats["total"],
            "regions": regions,
            "avg_confidence": atom_stats["avg_confidence"],
            "stale_atoms": len(stale_atoms),
            "orphan_atoms": orphan_count,
            "db_size_mb": db_size_mb,
            "embedding_model": self._config.embedding_model,
            "current_session_id": self._current_session_id,
            "ollama_healthy": ollama_healthy,
        }

    async def pathway(
        self,
        atom_id: int,
        depth: int = 2,
        min_strength: float = 0.1,
    ) -> dict[str, Any]:
        """Visualize the connection graph from a specific atom.

        Performs a BFS traversal from *atom_id* up to *depth* hops,
        collecting nodes and edges to form a subgraph suitable for
        visualization.

        Parameters
        ----------
        atom_id:
            The starting atom for the traversal.
        depth:
            Maximum number of hops to traverse from the start atom.
        min_strength:
            Minimum synapse strength to follow during traversal.

        Returns
        -------
        dict
            Keys:

            - ``nodes`` -- list of ``{id, content, type, region}`` dicts.
            - ``edges`` -- list of ``{source, target, relationship, strength}``
              dicts.
            - ``clusters`` -- dict mapping region names to lists of atom IDs.

        Raises
        ------
        ValueError
            If the starting atom does not exist.
        """
        self._ensure_initialized()
        assert self._atoms is not None
        assert self._synapses is not None

        # Validate the starting atom.
        start_atom = await self._atoms.get_without_tracking(atom_id)
        if start_atom is None:
            raise ValueError(f"Atom {atom_id} not found")

        # BFS traversal.
        nodes: dict[int, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[int, int, str]] = set()

        # Seed the BFS with the starting atom.
        nodes[start_atom.id] = {
            "id": start_atom.id,
            "content": start_atom.content[:200],
            "type": start_atom.type,
            "region": start_atom.region,
        }

        frontier: set[int] = {atom_id}

        for _level in range(depth):
            next_frontier: set[int] = set()
            frontier_list = list(frontier)

            # 1 SQL: batch neighbor fetch for all frontier atoms.
            batch_neighbors = await self._synapses.get_neighbors_batch(
                frontier_list, min_strength=min_strength
            )

            # Collect all new neighbor IDs that need atom data.
            new_ids: list[int] = []
            seen_new: set[int] = set()
            for current_id in frontier_list:
                for neighbor_id, _synapse in batch_neighbors.get(current_id, []):
                    if neighbor_id not in nodes and neighbor_id not in seen_new:
                        new_ids.append(neighbor_id)
                        seen_new.add(neighbor_id)

            # 1 SQL: batch atom fetch for all new neighbors.
            new_atom_map = await self._atoms.get_batch_without_tracking(new_ids)

            # Process the batch results: record edges and add nodes.
            for current_id in frontier_list:
                for neighbor_id, synapse in batch_neighbors.get(current_id, []):
                    # Record the edge (deduplicated).
                    edge_key = (
                        min(synapse.source_id, synapse.target_id),
                        max(synapse.source_id, synapse.target_id),
                        synapse.relationship,
                    )
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append({
                            "source": synapse.source_id,
                            "target": synapse.target_id,
                            "relationship": synapse.relationship,
                            "strength": synapse.strength,
                        })

                    # Add the neighbor node if not already visited.
                    if neighbor_id not in nodes:
                        neighbor_atom = new_atom_map.get(neighbor_id)
                        if neighbor_atom is not None and not neighbor_atom.is_deleted:
                            nodes[neighbor_id] = {
                                "id": neighbor_atom.id,
                                "content": neighbor_atom.content[:200],
                                "type": neighbor_atom.type,
                                "region": neighbor_atom.region,
                            }
                            next_frontier.add(neighbor_id)

            frontier = next_frontier

            if not frontier:
                break

        # Build clusters by region.
        clusters: dict[str, list[int]] = {}
        for node in nodes.values():
            region_name = node["region"]
            if region_name not in clusters:
                clusters[region_name] = []
            clusters[region_name].append(node["id"])

        logger.info(
            "Pathway from atom %d: %d nodes, %d edges, depth=%d",
            atom_id,
            len(nodes),
            len(edges),
            depth,
        )

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "clusters": clusters,
        }

    # ==================================================================
    # Hook Stats
    # ==================================================================

    async def get_hook_stats(self) -> dict[str, Any]:
        """Query hook_stats table and return a structured dashboard dict.

        Returns
        -------
        dict
            Keys: ``counts_7d``, ``counts_30d``, ``counts_all``,
            ``avg_atoms_returned``, ``avg_relevance_score``,
            ``budget_utilisation_pct``, ``unique_atoms_surfaced``,
            ``total_atoms``, ``total_impressions``, ``top_atoms``,
            ``novelty_stats``, ``latency``.
        """
        self._ensure_initialized()
        assert self._storage is not None

        result: dict[str, Any] = {}

        # Hook counts by time period.
        for label, days in [("counts_7d", 7), ("counts_30d", 30)]:
            rows = await self._storage.execute(
                """
                SELECT hook_type, COUNT(*) AS cnt FROM hook_stats
                WHERE created_at >= datetime('now', ?)
                GROUP BY hook_type
                """,
                (f"-{days} days",),
            )
            result[label] = {r["hook_type"]: r["cnt"] for r in rows}

        rows = await self._storage.execute(
            "SELECT hook_type, COUNT(*) AS cnt FROM hook_stats GROUP BY hook_type"
        )
        result["counts_all"] = {r["hook_type"]: r["cnt"] for r in rows}

        # Avg atoms returned per hook type.
        rows = await self._storage.execute(
            """
            SELECT hook_type, AVG(atoms_returned) AS avg_atoms
            FROM hook_stats
            WHERE atoms_returned > 0
            GROUP BY hook_type
            """
        )
        result["avg_atoms_returned"] = {
            r["hook_type"]: round(r["avg_atoms"], 1) for r in rows
        }

        # Avg relevance score (across all hooks that returned atoms).
        rows = await self._storage.execute(
            "SELECT AVG(avg_score) AS avg FROM hook_stats WHERE avg_score IS NOT NULL"
        )
        result["avg_relevance_score"] = (
            round(rows[0]["avg"], 4) if rows and rows[0]["avg"] is not None else None
        )

        # Budget utilisation.
        rows = await self._storage.execute(
            """
            SELECT AVG(CASE WHEN budget_total > 0
                        THEN CAST(budget_used AS REAL) / budget_total * 100
                        ELSE NULL END) AS pct
            FROM hook_stats
            WHERE budget_total > 0
            """
        )
        result["budget_utilisation_pct"] = (
            round(rows[0]["pct"], 1) if rows and rows[0]["pct"] is not None else None
        )

        # Unique atoms surfaced + total impressions.
        # W3-F: Push JSON aggregation into SQL via json_each() to avoid loading
        # every atom_ids blob into Python memory (O(total_impressions) → O(1)).
        agg_rows = await self._storage.execute(
            """
            SELECT
                COUNT(DISTINCT CAST(j.value AS INTEGER)) AS unique_atoms,
                COUNT(j.value)                            AS total_impressions
            FROM hook_stats h, json_each(h.atom_ids) AS j
            WHERE h.atom_ids IS NOT NULL
            """
        )
        if agg_rows and agg_rows[0]["unique_atoms"] is not None:
            result["unique_atoms_surfaced"] = agg_rows[0]["unique_atoms"]
            result["total_impressions"] = agg_rows[0]["total_impressions"]
        else:
            result["unique_atoms_surfaced"] = 0
            result["total_impressions"] = 0

        # Total atoms in the system.
        atom_rows = await self._storage.execute(
            "SELECT COUNT(*) AS cnt FROM atoms WHERE is_deleted = 0"
        )
        result["total_atoms"] = atom_rows[0]["cnt"] if atom_rows else 0

        # Top 10 most recalled atoms — computed in SQL with a JOIN to atoms table
        # so we avoid N+1 per-atom queries (W3-F: one query instead of up to 10+1).
        top_rows = await self._storage.execute(
            """
            SELECT
                CAST(j.value AS INTEGER) AS atom_id,
                COUNT(*)                 AS cnt,
                a.content                AS content,
                a.type                   AS type
            FROM hook_stats h, json_each(h.atom_ids) AS j
            JOIN atoms a ON a.id = CAST(j.value AS INTEGER)
            WHERE h.atom_ids IS NOT NULL
            GROUP BY atom_id
            ORDER BY cnt DESC
            LIMIT 10
            """
        )
        result["top_atoms"] = [
            {
                "id": r["atom_id"],
                "count": r["cnt"],
                "type": r["type"],
                "content": r["content"],
            }
            for r in top_rows
        ]

        # Novelty pass rate.
        rows = await self._storage.execute(
            """
            SELECT novelty_result, COUNT(*) AS cnt FROM hook_stats
            WHERE novelty_result IS NOT NULL
            GROUP BY novelty_result
            """
        )
        novelty: dict[str, int] = {"pass": 0, "fail": 0, "total": 0}
        for r in rows:
            novelty[r["novelty_result"]] = r["cnt"]
            novelty["total"] += r["cnt"]
        result["novelty_stats"] = novelty

        # Latency by hook type.
        rows = await self._storage.execute(
            """
            SELECT hook_type,
                   AVG(latency_ms) AS avg_ms,
                   MAX(latency_ms) AS max_ms
            FROM hook_stats
            GROUP BY hook_type
            """
        )
        result["latency"] = {
            r["hook_type"]: {
                "avg": round(r["avg_ms"], 1),
                "max": r["max_ms"],
            }
            for r in rows
        }

        # Weekly relevance trend (last 8 weeks, prompt-submit only).
        rows = await self._storage.execute(
            """
            SELECT strftime('%Y-W%W', created_at) AS week,
                   AVG(avg_score)                 AS relevance,
                   COUNT(*)                       AS invocations,
                   AVG(atoms_returned)            AS avg_atoms
            FROM hook_stats
            WHERE hook_type = 'prompt-submit' AND avg_score IS NOT NULL
            GROUP BY week
            ORDER BY week DESC
            LIMIT 8
            """
        )
        result["relevance_trend"] = [
            {
                "week": r["week"],
                "relevance": round(r["relevance"], 4),
                "invocations": r["invocations"],
                "avg_atoms": round(r["avg_atoms"], 1),
            }
            for r in rows
        ]

        # Feedback summary.
        rows = await self._storage.execute(
            """
            SELECT signal, COUNT(*) AS cnt FROM atom_feedback GROUP BY signal
            """
        )
        feedback: dict[str, int] = {"good": 0, "bad": 0, "total": 0}
        for r in rows:
            feedback[r["signal"]] = r["cnt"]
            feedback["total"] += r["cnt"]
        result["feedback"] = feedback

        return result

    # ==================================================================
    # Task Management
    # ==================================================================

    async def create_task(
        self,
        content: str,
        region: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
        status: str = "pending",
    ) -> dict[str, Any]:
        """Create a new task atom.

        Tasks are a special atom type that can be tracked through a
        lifecycle (pending → active → done → archived).  When tasks
        are completed, their linked memories can be flagged as stale.

        Parameters
        ----------
        content:
            The task description.
        region:
            Logical grouping (defaults to 'tasks').
        tags:
            Optional list of tags.
        importance:
            Priority weight in [0, 1].
        status:
            Initial status: 'pending' (default), 'active', 'done', 'archived'.

        Returns
        -------
        dict
            Keys: ``atom_id``, ``atom``, ``synapses_created``, ``related_atoms``.
        """
        self._ensure_initialized()
        assert self._atoms is not None
        assert self._learning is not None

        # Default region for tasks.
        if region is None:
            region = "tasks"

        atom = await self._atoms.create(
            content=content,
            type="task",
            region=region,
            tags=tags,
            importance=importance,
            task_status=status,
        )

        # Auto-link to related memories.
        created_synapses = await self._learning.auto_link(atom.id)

        # Collect related atom summaries from the created synapses.
        # Batch-fetch all unique related atom IDs in a single query.
        related_ids: list[int] = []
        seen_ids: set[int] = set()
        for syn_info in created_synapses:
            related_id = (
                syn_info["target_id"]
                if syn_info["source_id"] == atom.id
                else syn_info["source_id"]
            )
            if related_id not in seen_ids:
                seen_ids.add(related_id)
                related_ids.append(related_id)

        related_map = await self._atoms.get_batch_without_tracking(related_ids)

        related_atoms: list[dict[str, Any]] = []
        seen_for_output: set[int] = set()
        for syn_info in created_synapses:
            related_id = (
                syn_info["target_id"]
                if syn_info["source_id"] == atom.id
                else syn_info["source_id"]
            )
            if related_id in seen_for_output:
                continue
            seen_for_output.add(related_id)

            related = related_map.get(related_id)
            if related is not None:
                related_atoms.append({
                    "id": related.id,
                    "content": related.content[:120],
                    "type": related.type,
                    "region": related.region,
                    "relationship": syn_info["relationship"],
                    "strength": syn_info["strength"],
                })

        logger.info(
            "Created task %d (%s) with %d connections",
            atom.id,
            status,
            len(created_synapses),
        )

        return {
            "atom_id": atom.id,
            "atom": atom.to_dict(),
            "synapses_created": len(created_synapses),
            "related_atoms": related_atoms,
        }

    async def update_task(
        self,
        task_id: int,
        status: str,
        flag_linked_memories: bool = True,
    ) -> dict[str, Any]:
        """Update task status and optionally flag linked memories.

        When a task is marked 'done' or 'archived' with ``flag_linked_memories=True``,
        all memories connected to this task have their confidence reduced slightly,
        signalling they may be stale and candidates for consolidation.

        Parameters
        ----------
        task_id:
            The task atom ID.
        status:
            New status: 'pending', 'active', 'done', 'archived'.
        flag_linked_memories:
            If True and status is 'done' or 'archived', reduce confidence
            of linked memories by 0.1 (to a minimum of 0.1).

        Returns
        -------
        dict
            Keys: ``task``, ``linked_memories_flagged``.

        Raises
        ------
        ValueError
            If task not found or not a task type.
        """
        self._ensure_initialized()
        assert self._atoms is not None
        assert self._synapses is not None
        assert self._storage is not None

        atom = await self._atoms.update_task_status(task_id, status)
        if atom is None:
            raise ValueError(f"Task {task_id} not found or not a task")

        flagged_count = 0

        # Flag linked memories when task completes.
        if flag_linked_memories and status in ("done", "archived"):
            # Get all synapses connected to this task.
            synapses = await self._synapses.get_connections(task_id)

            linked_ids: set[int] = set()
            for syn in synapses:
                if syn.source_id != task_id:
                    linked_ids.add(syn.source_id)
                if syn.target_id != task_id:
                    linked_ids.add(syn.target_id)

            # Reduce confidence of linked atoms.
            for linked_id in linked_ids:
                await self._storage.execute_write(
                    """
                    UPDATE atoms
                    SET confidence = MAX(confidence - 0.1, 0.1),
                        updated_at = datetime('now')
                    WHERE id = ? AND is_deleted = 0 AND type != 'task'
                    """,
                    (linked_id,),
                )
                flagged_count += 1

            logger.info(
                "Task %d marked %s, flagged %d linked memories",
                task_id,
                status,
                flagged_count,
            )

        return {
            "task": atom.to_dict(),
            "linked_memories_flagged": flagged_count,
        }

    async def get_tasks(
        self,
        status: str | None = None,
        region: str | None = None,
    ) -> dict[str, Any]:
        """List task atoms with optional filters.

        Parameters
        ----------
        status:
            Filter by status ('pending', 'active', 'done', 'archived').
        region:
            Filter by region.

        Returns
        -------
        dict
            Keys: ``tasks`` (list of task atoms), ``count``.
        """
        self._ensure_initialized()
        assert self._atoms is not None

        tasks = await self._atoms.get_tasks(status=status, region=region)

        return {
            "tasks": [t.to_dict() for t in tasks],
            "count": len(tasks),
        }

    async def get_stale_memories(
        self,
        min_completed_tasks: int = 1,
    ) -> dict[str, Any]:
        """Find memories linked to completed tasks.

        These memories may be stale because the tasks they relate to
        are done.  They're candidates for review, update, or archival.

        Parameters
        ----------
        min_completed_tasks:
            Minimum number of completed tasks a memory must be linked
            to in order to be considered stale.

        Returns
        -------
        dict
            Keys: ``memories`` (list of potentially stale atoms),
            ``count``, ``linked_tasks`` (mapping of memory ID to task IDs).
        """
        self._ensure_initialized()
        assert self._storage is not None

        # Find all completed task IDs.
        completed_rows = await self._storage.execute(
            """
            SELECT id FROM atoms
            WHERE type = 'task'
              AND task_status IN ('done', 'archived')
              AND is_deleted = 0
            """
        )
        completed_ids = {row["id"] for row in completed_rows}

        if not completed_ids:
            return {"memories": [], "count": 0, "linked_tasks": {}}

        # Find all synapses connected to completed tasks.
        placeholders = ",".join("?" * len(completed_ids))
        synapse_rows = await self._storage.execute(
            f"""
            SELECT source_id, target_id FROM synapses
            WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
            """,
            tuple(completed_ids) + tuple(completed_ids),
        )

        # Build mapping of memory → linked completed tasks.
        memory_to_tasks: dict[int, set[int]] = {}
        for row in synapse_rows:
            src, tgt = row["source_id"], row["target_id"]
            if src in completed_ids and tgt not in completed_ids:
                memory_to_tasks.setdefault(tgt, set()).add(src)
            if tgt in completed_ids and src not in completed_ids:
                memory_to_tasks.setdefault(src, set()).add(tgt)

        # Filter by min_completed_tasks.
        stale_ids = [
            mid for mid, tasks in memory_to_tasks.items()
            if len(tasks) >= min_completed_tasks
        ]

        if not stale_ids:
            return {"memories": [], "count": 0, "linked_tasks": {}}

        # Fetch the stale memories.
        placeholders = ",".join("?" * len(stale_ids))
        memory_rows = await self._storage.execute(
            f"""
            SELECT * FROM atoms
            WHERE id IN ({placeholders})
              AND is_deleted = 0
              AND type != 'task'
            ORDER BY confidence ASC
            """,
            tuple(stale_ids),
        )

        from memories.atoms import Atom
        memories = [Atom.from_row(row).to_dict() for row in memory_rows]

        # Convert sets to lists for JSON serialization.
        linked_tasks = {
            str(mid): list(tasks) for mid, tasks in memory_to_tasks.items()
            if mid in stale_ids
        }

        return {
            "memories": memories,
            "count": len(memories),
            "linked_tasks": linked_tasks,
        }

    # ==================================================================
    # Session Management
    # ==================================================================

    async def _start_session(self, project: str | None = None) -> str:
        """Start a new session.  Store in the sessions table.

        Parameters
        ----------
        project:
            Optional project identifier for the session.

        Returns
        -------
        str
            The UUID of the newly created session.
        """
        assert self._storage is not None

        self._current_session_id = str(uuid.uuid4())
        await self._storage.execute_write(
            "INSERT INTO sessions (id, project) VALUES (?, ?)",
            (self._current_session_id, project),
        )
        logger.info("Session started: %s", self._current_session_id)
        return self._current_session_id

    async def end_session(self) -> dict[str, Any]:
        """End the current session.  Apply Hebbian learning.

        Strengthens synapses between atoms that were co-activated during
        the session, updates the session end time, and returns session
        statistics.

        Returns
        -------
        dict
            Keys: ``session_id``, ``atoms_accessed``,
            ``synapses_strengthened``, ``synapses_created``.
        """
        self._ensure_initialized()
        assert self._learning is not None
        assert self._storage is not None

        session_id = self._current_session_id

        # Read session atom IDs from the hook_session_atoms DB table
        # (populated by CLI hooks via _save_hook_atoms).
        atoms_accessed: list[int] = []
        if session_id:
            rows = await self._storage.execute(
                "SELECT atom_id FROM hook_session_atoms WHERE claude_session_id = ?",
                (session_id,),
            )
            atoms_accessed = [row["atom_id"] for row in rows]

        learning_stats: dict[str, int] = {"synapses_strengthened": 0, "synapses_created": 0}

        if session_id and atoms_accessed:
            learning_stats = await self._learning.session_end_learning(
                session_id, atoms_accessed
            )

        # Update session end time.
        if session_id:
            await self._storage.execute_write(
                "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?",
                (session_id,),
            )

        logger.info(
            "Session ended: %s (%d atoms accessed, %d synapses updated)",
            session_id,
            len(atoms_accessed),
            learning_stats.get("synapses_strengthened", 0),
        )

        # Reset session state.
        self._current_session_id = None

        return {
            "session_id": session_id,
            "atoms_accessed": len(atoms_accessed),
            "synapses_strengthened": learning_stats.get("synapses_strengthened", 0),
            "synapses_created": learning_stats.get("synapses_created", 0),
        }

    def track_atom_access(self, atom_id: int) -> None:
        """No-op. Retained for API compatibility.

        Session atom tracking is now handled by the ``hook_session_atoms``
        DB table, populated by CLI hooks via ``_save_hook_atoms()``.

        .. deprecated::
            This method no longer stores state.  It will be removed in a
            future release.
        """

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Clean shutdown: end session, close storage.

        Safe to call even if the brain was never initialised or the session
        has already ended.
        """
        if self._current_session_id:
            try:
                await self.end_session()
            except Exception:
                logger.exception("Error ending session during shutdown")

        if self._storage:
            await self._storage.close()

        self._initialized = False
        logger.info("Brain shut down")
