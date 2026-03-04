"""Automatic linking, pathway building, and antipattern detection for the memories system.

The **learning engine** is responsible for the organic growth of the memory
graph after new atoms are stored.  Every time :meth:`LearningEngine.auto_link`
is called (typically right after ``remember()``), it finds semantically related
atoms via vector search and wires them together with appropriately typed
synapses:

- ``related-to`` -- general topical association (strength = similarity score).
- ``warns-against`` -- inhibitory link from an antipattern to a related atom.
- ``contradicts`` -- detected when two similar atoms of the same type appear to
  make conflicting assertions.
- ``supersedes`` -- newer atom replaces an older near-duplicate.

Additional responsibilities:

- **Novelty assessment** -- gatekeeper that prevents storing information
  already well-represented in the graph.
- **Region suggestion** -- infers an appropriate region for a new atom based
  on content keywords and the neighbourhood of similar existing atoms.
- **Session-end Hebbian learning** -- strengthens synapses between atoms that
  were co-activated during a session.

Usage::

    from memories.learning import LearningEngine

    engine = LearningEngine(storage, embeddings, atoms, synapses)
    links = await engine.auto_link(new_atom.id)
    novelty = await engine.assess_novelty("some candidate content")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from typing import Any

from memories.atoms import ATOM_TYPES, Atom, AtomManager
from memories.config import get_config
from memories.embeddings import EmbeddingEngine
from memories.storage import Storage
from memories.synapses import (
    RELATIONSHIP_TYPES,
    SynapseManager,
    _MAX_INBOUND_RELATED_TO,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM relationship classifier cache (module-level, max 1000 entries)
# ---------------------------------------------------------------------------

_LLM_CLASSIFY_CACHE: OrderedDict[tuple[str, str], str | None] = OrderedDict()
_LLM_CLASSIFY_CACHE_MAX = 1000

_VALID_LLM_RELATIONSHIPS = frozenset({
    "related-to", "elaborates", "caused-by", "part-of", "contradicts",
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AUTO_LINK_SEARCH_K: int = 10
"""Number of vector-search candidates to retrieve when auto-linking.
We fetch more than needed to account for self-matches and filtering."""

_SUPERSEDE_SIMILARITY_THRESHOLD: float = 0.9
"""Similarity above which a new atom is considered a near-duplicate of an
older atom and may supersede it."""

_SUPERSEDE_CONFIDENCE_REDUCTION: float = 0.1
"""How much to reduce the confidence of the older atom when it is superseded."""

_CONTRADICTION_SIMILARITY_THRESHOLD: float = 0.85
"""Minimum similarity for two same-type atoms to be considered contradiction
candidates.  The actual contradiction detection also requires content
divergence (different assertion tokens)."""

_CONTRADICTION_MIN_LENGTH: int = 20
"""Minimum content length for contradiction analysis to be meaningful."""

# ---------------------------------------------------------------------------
# Region keyword map
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Antipattern category classification
# ---------------------------------------------------------------------------

ANTIPATTERN_CATEGORY_PREFIX = "category:"

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "security": [
        "credential", "auth", "token", "key", "secret", "password",
        "permission", "injection", "xss",
    ],
    "data-loss": [
        "rm -rf", "delete", "drop table", "truncate", "overwrite",
        "corrupt", "wipe",
    ],
    "rate-limit": [
        "rate limit", "429", "throttl", "batch limit", "quota",
        "too many requests",
    ],
    "configuration": [
        "env var", "config", "default", "setting", "misconfigur",
        ".env", "hardcoded",
    ],
    "compatibility": [
        "version", "deprecated", "platform", "incompatible",
        "breaking change",
    ],
    "performance": [
        "slow", "memory leak", "n+1", "timeout", "latency", "blocking",
        "unbounded",
    ],
    "deployment": [
        "docker", "ci/cd", "pipeline", "build", "container",
        "kubernetes", "deploy",
    ],
    "concurrency": [
        "race condition", "deadlock", "thread", "async", "concurrent",
        "mutex", "lock",
    ],
    "general": [],
}

_REGION_KEYWORDS: dict[str, list[str]] = {
    "personal": ["prefer", "always use", "never use", "i like", "i dislike", "my style"],
    "errors": ["error", "bug", "fix", "mistake", "crash", "exception", "failure"],
    "decisions": ["decided", "chose", "architecture", "decision", "trade-off", "tradeoff"],
    "workflows": ["workflow", "process", "steps", "pipeline", "procedure", "automation"],
}
"""Keyword lists that map content phrases to region names."""


# ---------------------------------------------------------------------------
# Learning engine
# ---------------------------------------------------------------------------


class LearningEngine:
    """Automatic learning engine that builds the memory graph organically.

    Responsible for post-insertion linking, antipattern detection, supersession
    detection, region inference, novelty gating, and session-end Hebbian
    learning.

    Parameters
    ----------
    storage:
        An initialised :class:`~memories.storage.Storage` instance.
    embeddings:
        An initialised :class:`~memories.embeddings.EmbeddingEngine` for
        vector similarity operations.
    atoms:
        An :class:`~memories.atoms.AtomManager` for atom retrieval.
    synapses:
        A :class:`~memories.synapses.SynapseManager` for synapse CRUD and
        Hebbian learning.
    """

    def __init__(
        self,
        storage: Storage,
        embeddings: EmbeddingEngine,
        atoms: AtomManager,
        synapses: SynapseManager,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._atoms = atoms
        self._synapses = synapses
        self._cfg = get_config()

    # ------------------------------------------------------------------
    # Auto-link (main entry point after remember())
    # ------------------------------------------------------------------

    async def auto_link(self, atom_id: int) -> list[dict]:
        """Find related existing atoms and create synapses to a new atom.

        This is the primary graph-building method, called after every
        ``remember()`` operation.  It performs a vector similarity search to
        discover semantically related atoms and creates appropriately typed
        synapses for each strong match.

        Algorithm
        ---------
        1. Retrieve the new atom and embed its content (or use the stored
           embedding).
        2. Vector search for the top candidates, excluding the atom itself.
        3. For each candidate with similarity above
           ``config.learning.auto_link_threshold``:

           a. Create a ``related-to`` synapse with strength equal to the
              similarity score.
           b. If the candidate is an antipattern, also create a
              ``warns-against`` synapse.
           c. If the candidate has the same type and very high similarity
              but divergent content tokens, create a ``contradicts`` synapse.

        4. Return a list of summary dicts describing the created synapses.

        Parameters
        ----------
        atom_id:
            Primary key of the newly created atom.

        Returns
        -------
        list[dict]
            Each dict contains ``source_id``, ``target_id``,
            ``relationship``, and ``strength`` for every synapse created.
        """
        atom = await self._atoms.get_without_tracking(atom_id)
        if atom is None:
            logger.warning("auto_link called for non-existent atom %d", atom_id)
            return []

        # Embed the atom content for vector search.
        try:
            similar = await self._embeddings.search_similar(
                atom.content, k=_AUTO_LINK_SEARCH_K
            )
        except RuntimeError:
            logger.warning(
                "Embedding unavailable for auto_link of atom %d; skipping",
                atom_id,
            )
            return []

        if not similar:
            logger.debug("No similar atoms found for atom %d", atom_id)
            return []

        threshold = self._cfg.learning.auto_link_threshold
        stc_floor = self._cfg.learning.stc_tagged_strength
        stc_sim_scale = self._cfg.learning.stc_similarity_scale
        stc_window = self._cfg.learning.stc_capture_window_days
        created_synapses: list[dict] = []
        # H3: Collect STC-tagged synapses for batch tag expiry setting.
        stc_tagged_triples: list[tuple[int, int, int, str]] = []

        # Collect all candidate IDs that pass the basic filters (not self, above
        # threshold) before fetching atoms, so we can do a single batch read.
        filtered: list[tuple[int, float]] = []
        for candidate_id, distance in similar:
            if candidate_id == atom_id:
                continue
            similarity = EmbeddingEngine.distance_to_similarity(distance)
            if similarity < threshold:
                continue
            filtered.append((candidate_id, similarity))

        # Batch-fetch all candidates in one query — avoids N+1 reads.
        candidate_ids = [cid for cid, _ in filtered]
        atoms_map = await self._atoms.get_batch_without_tracking(candidate_ids)

        # B3: Outbound degree cap for related-to — mirrors _MAX_INBOUND_RELATED_TO.
        _MAX_OUTBOUND_RELATED_TO = 50

        try:
            outbound_rows = await self._storage.execute(
                "SELECT COUNT(*) AS cnt FROM synapses "
                "WHERE source_id = ? AND relationship = 'related-to'",
                (atom_id,),
            )
            outbound_count = int(outbound_rows[0]["cnt"]) if outbound_rows else 0
        except (TypeError, KeyError, IndexError):
            outbound_count = 0
        outbound_budget = max(0, _MAX_OUTBOUND_RELATED_TO - outbound_count)

        # --- Phase 1: Collect candidates needing LLM classification ------
        # Run heuristic inference for all candidates first, then batch LLM
        # calls in parallel so one slow/failing call doesn't block the rest.
        pairs_needing_llm: list[tuple[int, float, "Atom"]] = []
        # Store heuristic results keyed by candidate_id for the synapse-
        # creation pass below.
        heuristic_results: dict[int, tuple[str, bool]] = {}

        for candidate_id, similarity in filtered:
            candidate = atoms_map.get(candidate_id)
            if candidate is None:
                continue
            is_antipattern_pair = (
                atom.type == "antipattern" or candidate.type == "antipattern"
            )
            if not is_antipattern_pair:
                relationship, bidirectional = self._infer_relationship_type(
                    atom, candidate
                )
                heuristic_results[candidate_id] = (relationship, bidirectional)
                if (
                    relationship == "related-to"
                    and similarity > 0.87
                    and self._cfg.distill_thinking
                ):
                    pairs_needing_llm.append((candidate_id, similarity, candidate))

        # Run all LLM classifications in parallel.
        llm_results: dict[int, str | None] = {}
        if pairs_needing_llm:
            classify_tasks = [
                self._classify_relationship_llm(atom, candidate)
                for _, _, candidate in pairs_needing_llm
            ]
            classified_results = await asyncio.gather(
                *classify_tasks, return_exceptions=True
            )
            for (cand_id, _, _), result in zip(
                pairs_needing_llm, classified_results
            ):
                if isinstance(result, BaseException):
                    logger.debug(
                        "LLM classification failed for candidate %d: %s",
                        cand_id, result,
                    )
                    llm_results[cand_id] = None
                else:
                    llm_results[cand_id] = result

        # --- Phase 2: Collect synapse specs for batch creation ----------
        # Instead of N sequential _safe_create_synapse calls (each acquiring
        # the write lock), collect all specs and batch-insert in one
        # execute_many call.  Reduces N thread-dispatches + N commits to 1.
        synapse_specs: list[tuple[int, int, str, float, int]] = []
        # Track retroactive interference targets to apply after batch.
        interference_targets: list[tuple[int, float]] = []

        for candidate_id, similarity in filtered:
            candidate = atoms_map.get(candidate_id)
            if candidate is None:
                continue

            # --- (a) primary synapse with inferred relationship type ------
            is_antipattern_pair = (
                atom.type == "antipattern" or candidate.type == "antipattern"
            )

            if not is_antipattern_pair:
                relationship, bidirectional = heuristic_results.get(
                    candidate_id, ("related-to", True)
                )

                # Apply LLM classification override if available.
                if candidate_id in llm_results and llm_results[candidate_id]:
                    relationship = llm_results[candidate_id]
                    bidirectional = False  # typed links are directional

                # B3: Enforce outbound degree cap for related-to synapses.
                if relationship == "related-to":
                    if outbound_budget <= 0:
                        continue
                    outbound_budget -= 1

                # STC: related-to synapses use similarity-proportional initial
                # strength.  Typed semantic links use full similarity.
                use_tagged = relationship == "related-to"
                initial_strength = (
                    max(stc_floor, similarity * stc_sim_scale) if use_tagged
                    else similarity
                )

                synapse_specs.append((
                    atom_id, candidate_id, relationship,
                    min(1.0, max(0.0, initial_strength)), int(bidirectional),
                ))
                if use_tagged:
                    stc_tagged_triples.append((
                        stc_window, atom_id, candidate_id, relationship,
                    ))

            # --- (b) warns-against for antipatterns -----------------------
            if candidate.type == "antipattern":
                synapse_specs.append((
                    candidate_id, atom_id, "warns-against",
                    min(1.0, max(0.0, similarity)), 0,
                ))

            if atom.type == "antipattern" and candidate.type != "antipattern":
                synapse_specs.append((
                    atom_id, candidate_id, "warns-against",
                    min(1.0, max(0.0, similarity)), 0,
                ))

            # --- (c) contradiction detection + retroactive interference ---
            if self._is_potential_contradiction(atom, candidate, similarity):
                synapse_specs.append((
                    atom_id, candidate_id, "contradicts",
                    min(1.0, max(0.0, similarity)), 1,
                ))
                penalty = self._cfg.learning.interference_confidence_penalty
                if penalty > 0 and candidate.created_at <= atom.created_at:
                    interference_targets.append((candidate_id, penalty))

        # --- Phase 3: Batch inbound cap check for related-to targets ---
        # Pre-check _MAX_INBOUND_RELATED_TO for all related-to targets in
        # one query instead of per-synapse checks inside create().
        related_to_targets = list(set(
            t for _, t, r, _, _ in synapse_specs if r == "related-to"
        ))
        over_cap: set[int] = set()
        if related_to_targets:
            placeholders = ",".join("?" * len(related_to_targets))
            cap_rows = await self._storage.execute(
                f"SELECT target_id, COUNT(*) AS cnt FROM synapses "
                f"WHERE target_id IN ({placeholders}) AND relationship = 'related-to' "
                f"GROUP BY target_id",
                tuple(related_to_targets),
            )
            over_cap = {
                row["target_id"] for row in cap_rows
                if row["cnt"] >= _MAX_INBOUND_RELATED_TO
            }
            if over_cap:
                synapse_specs = [
                    spec for spec in synapse_specs
                    if not (spec[2] == "related-to" and spec[1] in over_cap)
                ]
                stc_tagged_triples = [
                    t for t in stc_tagged_triples if t[2] not in over_cap
                ]

        # --- Phase 4: Batch create all synapses -----------------------
        # Use execute_many for the common case (1 commit instead of N).
        # On failure, fall back to individual _safe_create_synapse calls
        # to preserve error isolation — a single FK violation should not
        # lose all synapses for this atom.
        if synapse_specs:
            increment = self._cfg.learning.hebbian_increment
            try:
                await self._storage.execute_many(
                    """
                    INSERT INTO synapses
                        (source_id, target_id, relationship, strength, bidirectional)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, relationship) DO UPDATE SET
                        strength = MIN(1.0, synapses.strength + ? * (1.0 - synapses.strength)),
                        activated_count = synapses.activated_count + 1,
                        last_activated_at = datetime('now')
                    """,
                    [(s, t, r, st, bi, increment) for s, t, r, st, bi in synapse_specs],
                )
            except Exception:
                logger.warning(
                    "Batch synapse insert failed for atom %d; falling back to "
                    "individual creates (%d specs)",
                    atom_id, len(synapse_specs), exc_info=True,
                )
                synapse_specs = [
                    spec for spec in synapse_specs
                    if await self._safe_create_synapse(
                        source_id=spec[0], target_id=spec[1],
                        relationship=spec[2], strength=spec[3],
                        bidirectional=bool(spec[4]),
                    ) is not None
                ]

        created_synapses = [
            {"source_id": s, "target_id": t, "relationship": r, "strength": st}
            for s, t, r, st, _ in synapse_specs
        ]

        # Apply retroactive interference for contradiction targets.
        _INTERFERENCE_FLOOR = 0.1
        for target_id, penalty in interference_targets:
            await self._storage.execute_write(
                """
                UPDATE atoms
                SET confidence = MAX(?, confidence - ?),
                    updated_at = datetime('now')
                WHERE id = ? AND confidence > ?
                """,
                (_INTERFERENCE_FLOOR, penalty, target_id, _INTERFERENCE_FLOOR),
            )
            logger.info(
                "Retroactive interference: atom %d "
                "confidence reduced by %.2f "
                "(contradicted by new atom %d)",
                target_id, penalty, atom_id,
            )

        # H3: Batch-set STC tag expiry for all tagged synapses at once.
        if stc_tagged_triples:
            await self._storage.execute_many(
                """
                UPDATE synapses
                SET tag_expires_at = datetime('now', '+' || ? || ' days')
                WHERE source_id = ? AND target_id = ? AND relationship = ?
                  AND tag_expires_at IS NULL
                  AND activated_count <= 1
                """,
                stc_tagged_triples,
            )

        logger.info(
            "auto_link created %d synapses for atom %d",
            len(created_synapses),
            atom_id,
        )
        return created_synapses

    # ------------------------------------------------------------------
    # Antipattern link detection
    # ------------------------------------------------------------------

    async def detect_antipattern_links(self, atom_id: int) -> int:
        """Check if an atom should be linked to antipatterns via ``warns-against``.

        Searches for antipattern atoms that are topically similar to the given
        atom and creates ``warns-against`` synapses for matches above the
        configured auto-link threshold.

        Parameters
        ----------
        atom_id:
            Primary key of the atom to check.

        Returns
        -------
        int
            Number of new ``warns-against`` synapses created.
        """
        atom = await self._atoms.get_without_tracking(atom_id)
        if atom is None:
            return 0

        # Do not link antipatterns to themselves.
        if atom.type == "antipattern":
            return 0

        try:
            similar = await self._embeddings.search_similar(
                atom.content, k=_AUTO_LINK_SEARCH_K
            )
        except RuntimeError:
            logger.warning(
                "Embedding unavailable for antipattern detection on atom %d",
                atom_id,
            )
            return 0

        threshold = self._cfg.learning.auto_link_threshold
        count = 0

        # Collect candidates that pass basic filters before fetching atoms.
        filtered: list[tuple[int, float]] = []
        for candidate_id, distance in similar:
            if candidate_id == atom_id:
                continue
            similarity = EmbeddingEngine.distance_to_similarity(distance)
            if similarity < threshold:
                continue
            filtered.append((candidate_id, similarity))

        # Batch-fetch all candidates in one query — avoids N+1 reads.
        candidate_ids = [cid for cid, _ in filtered]
        atoms_map = await self._atoms.get_batch_without_tracking(candidate_ids)

        for candidate_id, similarity in filtered:
            candidate = atoms_map.get(candidate_id)
            if candidate is None or candidate.type != "antipattern":
                continue

            synapse = await self._safe_create_synapse(
                source_id=candidate_id,
                target_id=atom_id,
                relationship="warns-against",
                strength=similarity,
                bidirectional=False,
            )
            if synapse is not None:
                count += 1

        logger.info(
            "detect_antipattern_links found %d warns-against links for atom %d",
            count,
            atom_id,
        )
        return count

    # ------------------------------------------------------------------
    # Supersession detection
    # ------------------------------------------------------------------

    async def detect_supersedes(self, atom_id: int) -> int:
        """Check if a new atom supersedes an older near-duplicate.

        If there is a very similar atom (similarity > 0.9) of the same type
        and the new atom was created more recently, a ``supersedes`` synapse is
        created.  The older atom's confidence is reduced slightly to signal
        that newer information has arrived.

        Parameters
        ----------
        atom_id:
            Primary key of the newly created atom.

        Returns
        -------
        int
            Number of ``supersedes`` synapses created.
        """
        atom = await self._atoms.get_without_tracking(atom_id)
        if atom is None:
            return 0

        try:
            similar = await self._embeddings.search_similar(
                atom.content, k=_AUTO_LINK_SEARCH_K
            )
        except RuntimeError:
            logger.warning(
                "Embedding unavailable for supersession detection on atom %d",
                atom_id,
            )
            return 0

        count = 0

        # Collect candidates above the supersede threshold before fetching atoms.
        filtered: list[tuple[int, float]] = []
        for candidate_id, distance in similar:
            if candidate_id == atom_id:
                continue
            similarity = EmbeddingEngine.distance_to_similarity(distance)
            if similarity < _SUPERSEDE_SIMILARITY_THRESHOLD:
                continue
            filtered.append((candidate_id, similarity))

        # Batch-fetch all candidates in one query — avoids N+1 reads.
        candidate_ids = [cid for cid, _ in filtered]
        atoms_map = await self._atoms.get_batch_without_tracking(candidate_ids)

        for candidate_id, similarity in filtered:
            candidate = atoms_map.get(candidate_id)
            if candidate is None:
                continue

            # Must be the same type to supersede.
            if candidate.type != atom.type:
                continue

            # The new atom must be more recent.
            if atom.created_at <= candidate.created_at:
                continue

            synapse = await self._safe_create_synapse(
                source_id=atom_id,
                target_id=candidate_id,
                relationship="supersedes",
                strength=similarity,
                bidirectional=False,
            )
            if synapse is not None:
                count += 1

                # Reduce confidence of the superseded atom.
                new_confidence = max(0.0, candidate.confidence - _SUPERSEDE_CONFIDENCE_REDUCTION)
                await self._atoms.update(candidate_id, confidence=new_confidence)
                logger.info(
                    "Atom %d supersedes atom %d (similarity=%.3f); "
                    "reduced confidence %.2f -> %.2f",
                    atom_id,
                    candidate_id,
                    similarity,
                    candidate.confidence,
                    new_confidence,
                )

        return count

    # ------------------------------------------------------------------
    # Region suggestion
    # ------------------------------------------------------------------

    async def suggest_region(
        self,
        content: str,
        source_project: str | None = None,
    ) -> str:
        """Suggest a region for an atom based on content and context.

        Applies a priority-ordered heuristic:

        1. If *source_project* is provided, return ``"project:<name>"``.
        2. Scan *content* for keyword matches against known region patterns.
        3. Fall back to the majority region among the most similar existing
           atoms.
        4. Default to ``"technical"`` if no signal is available.

        Parameters
        ----------
        content:
            The atom content text.
        source_project:
            Optional project identifier that produced the atom.

        Returns
        -------
        str
            The suggested region name.
        """
        # Priority 1: Project-scoped region.
        if source_project:
            return f"project:{source_project}"

        # Priority 2: Content keyword matching.
        content_lower = content.lower()
        for region, keywords in _REGION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    logger.debug(
                        "Region suggestion: matched keyword %r -> %s",
                        keyword,
                        region,
                    )
                    return region

        # Priority 3: Majority vote from similar atoms.
        try:
            similar = await self._embeddings.search_similar(content, k=5)
        except RuntimeError:
            logger.debug("Embedding unavailable for region suggestion; defaulting")
            return "technical"

        if similar:
            region_votes: dict[str, int] = {}
            candidate_ids = [cid for cid, _ in similar]
            atoms_map = await self._atoms.get_batch_without_tracking(candidate_ids)
            for candidate_id, distance in similar:
                candidate = atoms_map.get(candidate_id)
                if candidate is not None:
                    region_votes[candidate.region] = (
                        region_votes.get(candidate.region, 0) + 1
                    )

            if region_votes:
                majority_region = max(region_votes, key=region_votes.get)  # type: ignore[arg-type]
                logger.debug(
                    "Region suggestion: majority vote -> %s (votes: %s)",
                    majority_region,
                    region_votes,
                )
                return majority_region

        # Priority 4: Default.
        return "technical"

    # ------------------------------------------------------------------
    # Antipattern field extraction
    # ------------------------------------------------------------------

    async def extract_antipattern_fields(
        self,
        content: str,
    ) -> tuple[str | None, str | None]:
        """Extract severity and alternative from antipattern content.

        Applies simple heuristic parsing to identify severity keywords and
        "instead" / "should" / "use X instead" patterns in the content text.

        Parameters
        ----------
        content:
            The antipattern content text.

        Returns
        -------
        tuple[str | None, str | None]
            ``(severity, instead_text)`` where either or both may be ``None``
            if not detected.
        """
        content_lower = content.lower()

        # --- Severity detection -------------------------------------------
        severity: str | None = None
        severity_keywords = [
            ("critical", "critical"),
            ("severe", "critical"),
            ("dangerous", "critical"),
            ("high", "high"),
            ("important", "high"),
            ("significant", "high"),
            ("medium", "medium"),
            ("moderate", "medium"),
            ("low", "low"),
            ("minor", "low"),
            ("trivial", "low"),
        ]
        for keyword, level in severity_keywords:
            if keyword in content_lower:
                severity = level
                break

        # --- "Instead" extraction -----------------------------------------
        instead_text: str | None = None

        # Pattern: "instead, ..." or "instead of ..., use ..."
        instead_match = re.search(
            r"instead[,:]?\s+(.+?)(?:\.|$)",
            content,
            re.IGNORECASE,
        )
        if instead_match:
            instead_text = instead_match.group(1).strip()

        # Pattern: "should ... " or "use ... instead"
        if instead_text is None:
            should_match = re.search(
                r"(?:should|use)\s+(.+?)(?:\s+instead|\.|$)",
                content,
                re.IGNORECASE,
            )
            if should_match:
                candidate = should_match.group(1).strip()
                # Avoid capturing overly long fragments.
                if len(candidate) <= 200:
                    instead_text = candidate

        # Pattern: "prefer X over Y" or "prefer X to Y"
        if instead_text is None:
            prefer_match = re.search(
                r"prefer\s+(.+?)(?:\s+over\s+|\s+to\s+)",
                content,
                re.IGNORECASE,
            )
            if prefer_match:
                instead_text = prefer_match.group(1).strip()

        return severity, instead_text

    # ------------------------------------------------------------------
    # Antipattern category classification
    # ------------------------------------------------------------------

    async def suggest_antipattern_category(
        self,
        content: str,
        existing_tags: list[str] | None = None,
    ) -> str:
        """Suggest a category for an antipattern atom.

        Three-tier classification (mirrors :meth:`suggest_region`):

        1. If *existing_tags* already contains a ``category:`` tag, return it.
        2. Keyword scan against :data:`_CATEGORY_KEYWORDS`.
        3. Similarity voting from existing categorised antipatterns
           (self-learning: more categorised atoms → better votes).
        4. Fallback to ``"general"``.

        Parameters
        ----------
        content:
            The antipattern content text.
        existing_tags:
            Tags already assigned to the atom.

        Returns
        -------
        str
            The category name (without the ``category:`` prefix).
        """
        # Tier 0: Respect caller-provided category.
        for tag in existing_tags or []:
            if tag.startswith(ANTIPATTERN_CATEGORY_PREFIX):
                return tag[len(ANTIPATTERN_CATEGORY_PREFIX):]

        # Tier 1: Keyword matching.
        content_lower = content.lower()
        for category, keywords in _CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    logger.debug(
                        "Antipattern category: keyword %r -> %s",
                        keyword,
                        category,
                    )
                    return category

        # Tier 2: Similarity voting from existing categorised antipatterns.
        cfg = get_config()
        if cfg.antipattern_classification.similarity_voting_enabled:
            try:
                similar = await self._embeddings.search_similar(content, k=5)
            except RuntimeError:
                logger.debug("Embedding unavailable for category voting; defaulting")
                return "general"

            if similar:
                candidate_ids = [cid for cid, _ in similar]
                atoms_map = await self._atoms.get_batch_without_tracking(candidate_ids)

                category_votes: dict[str, int] = {}
                for cid, _ in similar:
                    candidate = atoms_map.get(cid)
                    if candidate is None or candidate.type != "antipattern":
                        continue
                    for tag in candidate.tags or []:
                        if tag.startswith(ANTIPATTERN_CATEGORY_PREFIX):
                            cat = tag[len(ANTIPATTERN_CATEGORY_PREFIX):]
                            category_votes[cat] = category_votes.get(cat, 0) + 1

                if category_votes:
                    winner = max(category_votes, key=category_votes.get)  # type: ignore[arg-type]
                    logger.debug(
                        "Antipattern category: similarity vote -> %s (votes: %s)",
                        winner,
                        category_votes,
                    )
                    return winner

        # Tier 3: Fallback.
        return "general"

    # ------------------------------------------------------------------
    # Session-end learning
    # ------------------------------------------------------------------

    async def session_end_learning(
        self,
        session_id: str,
        atom_ids: list[int],
        atom_timestamps: dict[int, float] | None = None,
    ) -> dict[str, int]:
        """Apply Hebbian learning at the end of a session.

        Strengthens synapses between atoms that were co-activated during the
        session, and records the accessed atoms on the session row.

        Parameters
        ----------
        session_id:
            Identifier of the session that just ended.
        atom_ids:
            Flat list of atom IDs that were accessed during the session.
        atom_timestamps:
            Optional mapping of atom ID to Unix timestamp (float) indicating
            when each atom was first accessed in this session.  When provided,
            pairs accessed within ``learning.temporal_window_seconds`` receive
            the full Hebbian increment; distant pairs receive ``increment * 0.5``.

        Returns
        -------
        dict[str, int]
            Statistics with keys ``synapses_strengthened`` and
            ``synapses_created``.
        """
        if not atom_ids:
            logger.debug(
                "session_end_learning: no atoms for session %s", session_id
            )
            return {"synapses_strengthened": 0, "synapses_created": 0}

        # L-4: Serial position effect (Murdock 1962).  Atoms accessed at the
        # start (primacy) and end (recency) of a session form stronger Hebbian
        # associations than mid-session atoms.  Weight follows a U-curve:
        # edges ≈1.0, middle ≈0.5.
        pos_weights: dict[int, float] | None = None
        if atom_timestamps and len(atom_ids) >= 4:
            sorted_by_time = sorted(
                atom_ids, key=lambda a: atom_timestamps.get(a, 0.0)
            )
            n = len(sorted_by_time)
            primacy_depth = max(1, int(n * 0.3))
            pos_weights = {}
            for i, aid in enumerate(sorted_by_time):
                dist = min(i, n - 1 - i)
                pos_weights[aid] = max(0.5, 1.0 - dist / primacy_depth)

        # Step 1: Hebbian update for co-activated atoms.
        # L-1: hebbian_update returns (strengthened, created) separately.
        strengthened, created = await self._synapses.hebbian_update(
            atom_ids,
            atom_timestamps=atom_timestamps,
            position_weights=pos_weights,
        )

        # Step 2: Record the atoms accessed in the session row.
        atoms_json = json.dumps(atom_ids)
        await self._storage.execute_write(
            """
            UPDATE sessions
            SET atoms_accessed = ?,
                ended_at = datetime('now')
            WHERE id = ?
            """,
            (atoms_json, session_id),
        )

        logger.info(
            "Session %s learning complete: %d strengthened, %d created "
            "from %d atoms",
            session_id,
            strengthened,
            created,
            len(atom_ids),
        )

        return {
            "synapses_strengthened": strengthened,
            "synapses_created": created,
        }

    # ------------------------------------------------------------------
    # Novelty assessment
    # ------------------------------------------------------------------

    async def assess_novelty(
        self,
        content: str,
        threshold: float = 0.7,
    ) -> tuple[bool, int]:
        """Check if content is novel relative to existing atoms.

        Used by auto-capture hooks to avoid storing redundant information.
        Content is considered *not novel* if the most similar existing atom
        exceeds both the similarity threshold and has a confidence above 0.7
        (indicating the existing atom is well-established).

        Parameters
        ----------
        content:
            The candidate text to assess.
        threshold:
            Similarity threshold above which content is considered redundant.

        Returns
        -------
        tuple[bool, int]
            A tuple of ``(is_novel, similar_count)`` where *is_novel* is
            ``True`` if the content is sufficiently different from existing
            atoms (i.e. it should be stored), and *similar_count* is the
            number of existing atoms above the threshold.

        Note: similar_count is capped at the search_similar K value (typically 10),
        even if more than 10 similar atoms exist in the graph.
        """
        if not content or not content.strip():
            return False, 0

        try:
            similar = await self._embeddings.search_similar(content, k=_AUTO_LINK_SEARCH_K)
        except RuntimeError:
            # If embedding is unavailable, err on the side of storing.
            logger.warning(
                "Embedding unavailable for novelty assessment; assuming novel"
            )
            return True, 0

        if not similar:
            # No existing atoms at all -- definitely novel.
            return True, 0

        # Count how many existing atoms are above the threshold.
        similar_count = 0
        for _, distance in similar:
            sim = EmbeddingEngine.distance_to_similarity(distance)
            if sim > threshold:
                similar_count += 1

        closest_id, distance = similar[0]
        similarity = EmbeddingEngine.distance_to_similarity(distance)

        if similarity <= threshold:
            # Not similar enough to be redundant.
            return True, similar_count

        # Check the confidence of the existing atom.
        existing = await self._atoms.get_without_tracking(closest_id)
        if existing is None:
            return True, similar_count

        if existing.confidence > 0.7:
            logger.debug(
                "Content not novel: %.3f similarity to atom %d "
                "(confidence=%.2f)",
                similarity,
                closest_id,
                existing.confidence,
            )
            return False, similar_count

        # Similar but low-confidence existing atom -- the new content may
        # be a correction or update, so treat it as novel.
        return True, similar_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_relationship_type(
        atom_a: Atom,
        atom_b: Atom,
    ) -> tuple[str, bool]:
        """Infer the most specific relationship type between two atoms.

        Checks the combined content of both atoms for explicit linguistic
        markers indicating a causal, elaborative, or compositional
        relationship.  Falls back to ``"related-to"`` when no specific
        marker is found.

        The check scans both atoms so that either one can supply the signal.
        For example, if atom_b says "due to X" and atom_a is about X, the
        pair is classified as ``caused-by`` regardless of which is new.

        Parameters
        ----------
        atom_a:
            The first atom (typically the newly stored one).
        atom_b:
            The second atom (the existing candidate).

        Returns
        -------
        tuple[str, bool]
            ``(relationship_type, bidirectional)`` where ``bidirectional``
            is ``True`` only for the generic ``"related-to"`` fallback.
        """
        combined = (atom_a.content + " " + atom_b.content).lower()

        # caused-by: explicit causal connectors in either atom's content.
        _CAUSAL = (
            "because ", "due to ", "as a result of", "triggered by",
            "led to ", "caused by", "caused the ", "resulted in",
            "resulting from", "owing to", "the reason for",
            "the root cause", "this is why", "as a consequence",
            "stems from", "is attributed to", "originates from",
            "arises from", "is driven by",
        )
        if any(p in combined for p in _CAUSAL):
            return "caused-by", False

        # elaborates: explicit expansion / detail language.
        _ELABORATION = (
            "specifically,", "more precisely", "in particular",
            "for example", "for instance", " i.e.", "namely,",
            "in other words", "to be specific", "that is,",
            "this means that", "to clarify", "to elaborate",
            "expanding on", "building on this", "more specifically",
            "going deeper", "details:", "context:",
        )
        if any(p in combined for p in _ELABORATION):
            return "elaborates", False

        # part-of: compositional / membership language.
        _PARTOF = (
            "is part of", "is a component", "belongs to",
            "is a module", "is a subclass", "is a type of",
            "is a subset", "is included in", "is a member of",
            "is contained in", "is one of the", "is a layer of",
            "is a section of", "is a feature of", "is an element of",
        )
        if any(p in combined for p in _PARTOF):
            return "part-of", False

        # Length heuristic: if one atom is ≥ 2.5× longer, it likely
        # provides more detail about the shorter one → elaborates.
        # M-3: Only fire for same-type atoms — a short task paired with a
        # long fact would be a coincidental length difference, not elaboration.
        # M-4: Require the shorter atom to be at least 10 chars to avoid
        # false positives from trivially short content (e.g. single words).
        len_a = len(atom_a.content)
        len_b = len(atom_b.content)
        if len_a > 0 and len_b > 0 and atom_a.type == atom_b.type:
            shorter = min(len_a, len_b)
            longer = max(len_a, len_b)
            if shorter >= 10 and longer / shorter >= 2.5:
                return "elaborates", False

        # Generic fallback — bidirectional topical association.
        return "related-to", True

    async def _safe_create_synapse(
        self,
        source_id: int,
        target_id: int,
        relationship: str,
        strength: float,
        bidirectional: bool,
    ) -> dict[str, Any] | None:
        """Create a synapse, returning a summary dict or ``None`` on failure.

        Wraps :meth:`SynapseManager.create` with error handling so that
        individual synapse creation failures do not abort the entire
        auto-linking process.

        Parameters
        ----------
        source_id:
            Source atom ID.
        target_id:
            Target atom ID.
        relationship:
            Relationship type (one of :data:`~memories.synapses.RELATIONSHIP_TYPES`).
        strength:
            Initial synapse strength in ``[0, 1]``.
        bidirectional:
            Whether the synapse is traversable in both directions.

        Returns
        -------
        dict[str, Any] | None
            Summary dict with ``source_id``, ``target_id``, ``relationship``,
            and ``strength``; or ``None`` if creation failed.
        """
        try:
            synapse = await self._synapses.create(
                source_id=source_id,
                target_id=target_id,
                relationship=relationship,
                strength=min(1.0, max(0.0, strength)),
                bidirectional=bidirectional,
            )
            return {
                "source_id": synapse.source_id,
                "target_id": synapse.target_id,
                "relationship": synapse.relationship,
                "strength": synapse.strength,
            }
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "Failed to create %s synapse %d -> %d: %s",
                relationship,
                source_id,
                target_id,
                exc,
            )
            return None

    @staticmethod
    def _is_potential_contradiction(
        atom_a: Atom,
        atom_b: Atom,
        similarity: float,
    ) -> bool:
        """Heuristic check for whether two atoms might contradict each other.

        Two atoms are considered potential contradictions when:

        - They share the same type (two facts, two preferences, etc.).
        - Their similarity is very high (> 0.85), meaning they discuss the
          same topic.
        - Their content is long enough for meaningful comparison.
        - They contain divergent assertion tokens (negation words, different
          values after key phrases like ``"should"``, ``"prefer"``, ``"use"``).

        This is a deliberately conservative heuristic -- false negatives are
        acceptable, but false positives would clutter the graph with spurious
        contradiction links.

        Parameters
        ----------
        atom_a:
            The first atom (typically the newly created one).
        atom_b:
            The second atom (the existing candidate).
        similarity:
            The vector similarity between the two atoms.

        Returns
        -------
        bool
            ``True`` if the atoms appear to contradict each other.
        """
        # Must be the same type.
        if atom_a.type != atom_b.type:
            return False

        # Must be highly similar (same topic).
        if similarity < _CONTRADICTION_SIMILARITY_THRESHOLD:
            return False

        # Content must be long enough.
        if (
            len(atom_a.content) < _CONTRADICTION_MIN_LENGTH
            or len(atom_b.content) < _CONTRADICTION_MIN_LENGTH
        ):
            return False

        content_a = atom_a.content.lower()
        content_b = atom_b.content.lower()

        # Normalize contractions so "don't"/"dont" are treated identically.
        _contractions = str.maketrans({"'": ""})
        norm_a = content_a.translate(_contractions)
        norm_b = content_b.translate(_contractions)

        # Check for negation divergence: one has a negation word the other lacks.
        negation_words = {"not", "never", "dont", "avoid", "shouldnt", "wont", "cant"}
        tokens_a = set(norm_a.split())
        tokens_b = set(norm_b.split())

        negations_a = tokens_a & negation_words
        negations_b = tokens_b & negation_words

        # M-5: Require both sides to have unique negation words the other lacks.
        # Simple set inequality produces false positives when one atom is a
        # superset of the other (e.g. "never X" vs "never X and don't Y").
        negations_only_in_a = negations_a - negations_b
        negations_only_in_b = negations_b - negations_a
        if negations_only_in_a and negations_only_in_b:
            return True

        # Check for antonym pairs: if one atom contains a word and the other
        # contains its direct opposite, they likely describe conflicting states.
        _ANTONYM_PAIRS = [
            ("always", "never"),
            ("enable", "disable"),
            ("enabled", "disabled"),
            ("add", "remove"),
            ("start", "stop"),
            ("true", "false"),
            ("increase", "decrease"),
            ("faster", "slower"),
            ("better", "worse"),
            ("safe", "unsafe"),
            ("allowed", "forbidden"),
            ("on", "off"),
            ("yes", "no"),
        ]
        for word_x, word_y in _ANTONYM_PAIRS:
            if (word_x in tokens_a and word_y in tokens_b) or (
                word_y in tokens_a and word_x in tokens_b
            ):
                return True

        # Check for opposing value assertions after key phrases.
        # e.g. "use postgres" vs "use mysql"
        value_patterns = [
            r"(?:use|prefer|choose|recommend)\s+(\S+)",
            r"(?:should|always|must)\s+(\S+)",
        ]
        for pattern in value_patterns:
            match_a = re.search(pattern, content_a)
            match_b = re.search(pattern, content_b)
            if match_a and match_b and match_a.group(1) != match_b.group(1):
                return True

        return False

    async def _classify_relationship_llm(
        self,
        atom_a: Atom,
        atom_b: Atom,
    ) -> str | None:
        """Use a local Ollama model to classify the relationship between two atoms.

        Called when heuristic classification returns ``"related-to"`` and the
        similarity between the atoms is above 0.87, suggesting a more specific
        relationship type may exist.

        Results are cached in a module-level dict keyed by ``(sha256(a), sha256(b))``,
        capped at 1000 entries.

        Parameters
        ----------
        atom_a:
            The first atom (typically the newly stored one).
        atom_b:
            The second atom (the existing candidate).

        Returns
        -------
        str | None
            One of the valid relationship types if the LLM classification
            succeeds, or ``None`` on failure (caller falls back to heuristic).
        """
        global _LLM_CLASSIFY_CACHE

        # Build cache key from content hashes.
        hash_a = hashlib.sha256(atom_a.content.encode()).hexdigest()
        hash_b = hashlib.sha256(atom_b.content.encode()).hexdigest()
        cache_key = (hash_a, hash_b)

        if cache_key in _LLM_CLASSIFY_CACHE:
            _LLM_CLASSIFY_CACHE.move_to_end(cache_key)
            return _LLM_CLASSIFY_CACHE[cache_key]

        prompt = (
            "Given these two knowledge atoms, what is their relationship? "
            "Choose ONE from: related-to, elaborates, caused-by, part-of, contradicts. "
            "Reply with only the relationship type.\n\n"
            f"Atom A: {atom_a.content[:300]}\n"
            f"Atom B: {atom_b.content[:300]}"
        )

        try:
            import ollama as ollama_mod

            cfg = self._cfg
            client = ollama_mod.AsyncClient(host=cfg.ollama_url)
            response = await asyncio.wait_for(
                client.generate(model=cfg.distill_model, prompt=prompt),
                timeout=10.0,
            )
            raw = (response.response or "").strip().lower().strip(".")
            # Validate the response is one of the allowed types.
            if raw in _VALID_LLM_RELATIONSHIPS:
                result: str | None = raw
            else:
                logger.debug(
                    "LLM classifier returned invalid relationship %r; ignoring",
                    raw,
                )
                result = None
        except Exception as exc:
            logger.debug("LLM relationship classification failed: %s", exc)
            result = None

        # Cache the result (evict oldest if at capacity).
        _LLM_CLASSIFY_CACHE[cache_key] = result
        if len(_LLM_CLASSIFY_CACHE) > _LLM_CLASSIFY_CACHE_MAX:
            _LLM_CLASSIFY_CACHE.popitem(last=False)

        return result
