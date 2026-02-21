"""Spreading activation retrieval engine for the memories system.

Implements the core recall algorithm that combines vector similarity search
with brain-inspired spreading activation over the synapse graph, then applies
multi-factor ranking and context-window budgeting to produce a compact,
high-relevance set of memories.

The algorithm proceeds in four steps:

1. **Vector search** -- embed the query and find seed atoms via sqlite-vec
   approximate nearest-neighbour search.
2. **Spreading activation** -- propagate activation energy outward through
   synapse connections, decaying with distance and synapse strength.
3. **Multi-factor ranking** -- combine vector similarity, spread activation,
   recency, confidence, and access frequency into a single composite score.
4. **Budget fitting** -- compress the ranked results to fit within the
   caller's token budget using progressive compression levels.

Usage::

    from memories.retrieval import RetrievalEngine

    engine = RetrievalEngine(storage, embeddings, atoms, synapses, context_budget)
    result = await engine.recall("How does Redis SCAN work?", budget_tokens=2000)

    for atom in result.atoms:
        print(f"[{atom['type']}] {atom['content'][:80]}  (score={atom['score']:.3f})")
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from memories.atoms import Atom, AtomManager
from memories.config import RetrievalWeights, get_config
from memories.context import BudgetResult, ContextBudget
from memories.embeddings import EmbeddingEngine
from memories.storage import Storage
from memories.synapses import Synapse, SynapseManager

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recency decay constant
# ---------------------------------------------------------------------------

_RECENCY_DECAY_DAYS = 90
"""Number of days over which the recency score decays from 1.0 to 0.1."""

_RECENCY_FLOOR = 0.1
"""Minimum recency score for atoms that have not been accessed recently."""

# ---------------------------------------------------------------------------
# Frequency normalisation constant
# ---------------------------------------------------------------------------

_FREQUENCY_CAP_ACCESSES = 100
"""Access count at which the frequency signal saturates to 1.0 (log scale)."""

# ---------------------------------------------------------------------------
# Newness bonus: counteract old-memory bias
# ---------------------------------------------------------------------------

_NEWNESS_BONUS_DAYS = 7
"""Days during which a newly created atom receives a score bonus."""

_NEWNESS_BONUS_WEIGHT = 0.15
"""Additional score weight for atoms created within the newness window."""

# ---------------------------------------------------------------------------
# Antipattern boost multiplier
# ---------------------------------------------------------------------------

_ANTIPATTERN_BOOST = 1.5

_SESSION_PRIME_BOOST = 0.05
"""Initial activation added to atoms already seen in the current session.

Implements context-dependent retrieval: atoms primed by earlier queries in the
same session are slightly more likely to be re-activated by later queries,
creating coherent working-memory-like behaviour within a session.
"""
"""Score multiplier for antipattern atoms connected to result atoms."""


def _sanitize_fts_query(query: str) -> str:
    """Convert a natural language query into a safe FTS5 MATCH expression.

    Extracts alphanumeric tokens of three or more characters (avoiding
    FTS5 operator keywords and punctuation that would cause syntax errors)
    and joins them with implicit AND.  Returns an empty string when no
    usable tokens are found.
    """
    words = re.findall(r"[a-zA-Z0-9_]{3,}", query)[:20]
    return " ".join(words) if words else ""


# ---------------------------------------------------------------------------
# Recall result
# ---------------------------------------------------------------------------


@dataclass
class RecallResult:
    """Container for the output of a spreading activation recall.

    Attributes
    ----------
    atoms:
        Atom dicts with a ``score`` key added, ordered by descending score.
        These are the memories that fit within the token budget.
    antipatterns:
        Antipattern atoms that are connected to result atoms via
        ``warns-against`` synapses.  Always included regardless of
        activation status when ``include_antipatterns`` is True.
    pathways:
        Key synapse connections between result atoms, showing the
        graph structure that links the returned memories.
    budget_used:
        Number of tokens consumed by the formatted result.
    budget_remaining:
        Tokens still available in the budget after recall.
    total_activated:
        Total number of atoms that received activation energy (before
        budget pruning).  Useful for diagnostics.
    seed_count:
        Number of seed atoms returned by the initial vector search.
    compression_level:
        The compression level applied by the context budget manager
        (0 = full detail, 3 = minimal bullets).
    """

    atoms: list[dict] = field(default_factory=list)
    antipatterns: list[dict] = field(default_factory=list)
    pathways: list[dict] = field(default_factory=list)
    budget_used: int = 0
    budget_remaining: int = 0
    total_activated: int = 0
    seed_count: int = 0
    compression_level: int = 0


# ---------------------------------------------------------------------------
# Retrieval engine
# ---------------------------------------------------------------------------


class RetrievalEngine:
    """Brain-inspired retrieval engine combining vector search with spreading activation.

    Orchestrates the four-step recall pipeline: vector seed search, spreading
    activation through the synapse graph, multi-factor ranking, and context-
    window budget fitting.

    Parameters
    ----------
    storage:
        An initialised :class:`~memories.storage.Storage` instance.
    embeddings:
        An initialised :class:`~memories.embeddings.EmbeddingEngine` for
        vector similarity search.
    atoms:
        An :class:`~memories.atoms.AtomManager` for atom retrieval and
        access tracking.
    synapses:
        A :class:`~memories.synapses.SynapseManager` for traversing the
        synapse graph.
    context_budget:
        A :class:`~memories.context.ContextBudget` for fitting results
        into the caller's token budget.
    """

    def __init__(
        self,
        storage: Storage,
        embeddings: EmbeddingEngine,
        atoms: AtomManager,
        synapses: SynapseManager,
        context_budget: ContextBudget,
        weight_override: RetrievalWeights | None = None,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._atoms = atoms
        self._synapses = synapses
        self._budget = context_budget
        self._cfg = get_config().retrieval
        self._weights = weight_override  # None → fall through to self._cfg.weights

        # G2: Pre-compute synapse type weight lookup — avoids repeated
        # attribute lookups and string replacements inside the hot activation
        # inner loop.
        _type_weights = self._cfg.synapse_type_weights
        self._synapse_type_weights: dict[str, float] = {
            rel: getattr(_type_weights, rel.replace("-", "_"), 0.5)
            for rel in (
                "related-to", "caused-by", "part-of", "contradicts",
                "supersedes", "elaborates", "warns-against", "encoded-with",
            )
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        budget_tokens: int = 2000,
        depth: int | None = None,
        region: str | None = None,
        types: list[str] | None = None,
        include_antipatterns: bool = True,
        session_atom_ids: list[int] | None = None,
    ) -> RecallResult:
        """Execute the full spreading activation recall pipeline.

        Parameters
        ----------
        query:
            Natural language query to search for.
        budget_tokens:
            Maximum number of tokens to spend on the result.  The context
            budget manager will progressively compress atoms to fit.
        depth:
            Spreading activation depth (number of hops from seed atoms).
            Defaults to ``config.retrieval.spread_depth``.
        region:
            If set, only return atoms belonging to this region.
        types:
            If set, only return atoms whose type is in this list.
        include_antipatterns:
            When True, antipattern atoms connected via ``warns-against``
            synapses are always surfaced in results (with a score boost).
        session_atom_ids:
            IDs of atoms already accessed in the current session.  These
            receive a small priming boost before spreading activation so
            that contextually relevant atoms from earlier queries are more
            likely to resurface — implements context-dependent retrieval.

        Returns
        -------
        RecallResult
            The recall results including atoms, antipatterns, pathways,
            and budget metadata.
        """
        # Validate query - empty/whitespace-only queries can't be embedded.
        if not query or not query.strip():
            log.debug("Empty query provided, returning empty result")
            return RecallResult(seed_count=0)

        if depth is None:
            depth = self._cfg.spread_depth

        # Step 1: Vector search + BM25 in parallel -- find seed atoms.
        vector_scores, bm25_scores = await asyncio.gather(
            self._vector_search(
                query, k=self._cfg.seed_count, region=region, types=types
            ),
            self._bm25_search(
                query, k=self._cfg.seed_count, region=region, types=types
            ),
        )

        # Merge BM25 seeds that vector search missed.
        for atom_id, bm25_score in bm25_scores.items():
            if atom_id not in vector_scores:
                # BM25-only seeds enter spreading activation at their BM25 score.
                vector_scores[atom_id] = bm25_score

        # Session priming: atoms accessed earlier in this session get a small
        # baseline activation so they propagate alongside real seeds.  They do
        # not override actual vector/BM25 seeds (which have higher scores).
        if session_atom_ids:
            for aid in session_atom_ids:
                if aid not in vector_scores:
                    vector_scores[aid] = _SESSION_PRIME_BOOST

        seed_count = len(vector_scores)

        if not vector_scores:
            log.debug("No seed atoms found for query: %s", query[:100])
            return RecallResult(seed_count=0)

        # Step 2: Spreading activation -- traverse the synapse graph.
        activation_scores, spread_atom_map = await self._spread_activation(
            vector_scores, depth
        )
        total_activated = len(activation_scores)

        # Step 3: Multi-factor ranking -- score all activated atoms.
        # Pass the atom_map collected during spreading to avoid a second fetch.
        scored_atoms = await self._score_atoms(
            vector_scores,
            activation_scores,
            bm25_scores,
            include_antipatterns,
            atom_map=spread_atom_map,
        )

        if not scored_atoms:
            log.debug("No scoreable atoms after activation for query: %s", query[:100])
            return RecallResult(
                seed_count=seed_count,
                total_activated=total_activated,
            )

        # Apply region diversity cap: at most N atoms per region.
        cfg = get_config()
        diversity_cap = cfg.region_diversity_cap
        if diversity_cap > 0:
            seen_regions: dict[str, int] = {}
            diverse: list[tuple] = []
            for triple in scored_atoms:
                atom_region = triple[0].region
                count = seen_regions.get(atom_region, 0)
                if count < diversity_cap:
                    diverse.append(triple)
                    seen_regions[atom_region] = count + 1
            scored_atoms = diverse

        # Apply minimum composite score floor — drop weakly-activated atoms
        # that would dilute the context window with low-relevance content.
        # Antipatterns bypass the floor so warnings are never silently dropped.
        min_score = cfg.retrieval.min_score
        if min_score > 0:
            scored_atoms = [
                (atom, score, bd)
                for atom, score, bd in scored_atoms
                if score >= min_score or atom.type == "antipattern"
            ]

        # Separate antipattern atoms for dedicated surfacing.
        regular_atoms: list[tuple[Atom, float, dict[str, float] | None]] = []
        antipattern_atoms: list[tuple[Atom, float, dict[str, float] | None]] = []

        for atom, score, breakdown in scored_atoms:
            if atom.type == "antipattern":
                antipattern_atoms.append((atom, score, breakdown))
            else:
                regular_atoms.append((atom, score, breakdown))

        # Find additional antipatterns connected via warns-against that
        # may not have been activated.
        result_atom_ids = [atom.id for atom, _, _ in scored_atoms]
        if include_antipatterns:
            extra_antipatterns = await self._find_relevant_antipatterns(
                result_atom_ids,
            )
            # Merge extra antipatterns that were not already scored.
            existing_antipattern_ids = {atom.id for atom, _, _ in antipattern_atoms}
            for ap_atom in extra_antipatterns:
                if ap_atom.id not in existing_antipattern_ids:
                    # Assign a baseline score for unsolicited antipatterns.
                    antipattern_atoms.append((ap_atom, _ANTIPATTERN_BOOST * 0.3, None))

        # Convert to dicts with score for the budget compressor.
        regular_dicts = [
            self._atom_to_scored_dict(atom, score, breakdown)
            for atom, score, breakdown in regular_atoms
        ]
        antipattern_dicts = [
            self._atom_to_scored_dict(atom, score, breakdown)
            for atom, score, breakdown in antipattern_atoms
        ]

        # Step 4: Budget fitting -- compress regular atoms to fit within
        # the token limit.  Antipatterns are handled separately below with
        # a strict cap so the budget_tokens parameter accurately reflects
        # the actual context cost.
        budget_result: BudgetResult = self._budget.compress_to_budget(
            regular_dicts,
            budget_tokens,
        )

        # Determine which atoms actually made the cut.
        final_atom_ids_set = {a["id"] for a in budget_result.atoms}

        # D2: Collect all IDs to record access for in one batch instead of
        # one execute_write per atom (up to 11 sequential commits per recall).
        ids_to_record: list[int] = [a["id"] for a in budget_result.atoms]

        # Extract pathways between result atoms.
        final_atom_ids = [a["id"] for a in budget_result.atoms]
        pathways = await self._extract_pathways(final_atom_ids)

        final_regular = list(budget_result.atoms)

        # Append up to 3 antipatterns, sorted by score.  These are kept
        # outside the main budget to avoid crowding out regular atoms,
        # but strictly capped to prevent context bloat.
        _MAX_ANTIPATTERNS = 3
        final_antipatterns: list[dict] = []
        if include_antipatterns and antipattern_dicts:
            antipattern_dicts.sort(
                key=lambda a: a.get("score", 0), reverse=True,
            )
            for ap_dict in antipattern_dicts[:_MAX_ANTIPATTERNS]:
                final_antipatterns.append(ap_dict)
                ids_to_record.append(ap_dict["id"])

        # Single batch UPDATE for all accessed atoms.
        await self._atoms.record_access_batch(ids_to_record)

        budget_used = budget_result.budget_used
        budget_remaining = budget_tokens - budget_used

        log.info(
            "Recall complete: %d seeds, %d activated, %d returned "
            "(budget: %d/%d tokens, compression=%d)",
            seed_count,
            total_activated,
            len(budget_result.atoms),
            budget_used,
            budget_tokens,
            budget_result.compression_level,
        )

        return RecallResult(
            atoms=final_regular,
            antipatterns=final_antipatterns,
            pathways=pathways,
            budget_used=budget_used,
            budget_remaining=budget_remaining,
            total_activated=total_activated,
            seed_count=seed_count,
            compression_level=budget_result.compression_level,
        )

    # ------------------------------------------------------------------
    # Step 1: Vector search
    # ------------------------------------------------------------------

    async def _vector_search(
        self,
        query: str,
        k: int,
        region: str | None,
        types: list[str] | None,
    ) -> dict[int, float]:
        """Find seed atoms via vector similarity search.

        Embeds the query text, performs an approximate nearest-neighbour
        search via sqlite-vec, then filters results by region, type, and
        deletion status.

        Parameters
        ----------
        query:
            Natural language query text.
        k:
            Number of candidate results to request from the vector index.
            The actual number of seeds may be smaller after filtering.
        region:
            If set, only keep atoms in this region.
        types:
            If set, only keep atoms whose type is in this list.

        Returns
        -------
        dict[int, float]
            Mapping of ``atom_id`` to similarity score in ``[0, 1]``.
        """
        # Request more candidates than k to account for filtering.
        fetch_k = k * 3

        raw_results = await self._embeddings.search_similar(query, k=fetch_k)

        if not raw_results:
            return {}

        # D1: Batch-fetch all candidate atoms in one query instead of N
        # individual get_without_tracking calls.
        candidate_ids = [atom_id for atom_id, _ in raw_results]
        atoms_map = await self._atoms.get_batch_without_tracking(candidate_ids)

        seeds: dict[int, float] = {}
        for atom_id, distance in raw_results:
            atom = atoms_map.get(atom_id)
            if atom is None or atom.is_deleted:
                continue
            if region is not None and atom.region != region:
                continue
            if types is not None and atom.type not in types:
                continue

            similarity = EmbeddingEngine.distance_to_similarity(distance)
            seeds[atom_id] = similarity

            # Stop once we have enough seeds.
            if len(seeds) >= k:
                break

        log.debug(
            "Vector search found %d seeds (from %d candidates) for query: %s",
            len(seeds),
            len(raw_results),
            query[:80],
        )
        return seeds

    # ------------------------------------------------------------------
    # Step 1b: BM25 full-text search
    # ------------------------------------------------------------------

    async def _bm25_search(
        self,
        query: str,
        k: int,
        region: str | None,
        types: list[str] | None,
    ) -> dict[int, float]:
        """Find seed atoms via BM25 full-text search on the FTS5 index.

        Uses the ``atoms_fts`` virtual table's built-in BM25 rank signal
        (``f.rank``, negative -- more negative means more relevant).
        Results are normalised to ``[0, 1]`` before being returned.

        Parameters
        ----------
        query:
            Natural language query text.
        k:
            Target number of seed atoms to return.
        region:
            If set, only keep atoms in this region.
        types:
            If set, only keep atoms whose type is in this list.

        Returns
        -------
        dict[int, float]
            Mapping of ``atom_id`` to normalised BM25 score in ``[0, 1]``.
        """
        fts_query = _sanitize_fts_query(query)
        if not fts_query:
            return {}

        try:
            rows = await self._storage.execute(
                """
                SELECT a.id AS atom_id, -f.rank AS raw_score
                FROM atoms_fts f
                JOIN atoms a ON a.id = f.rowid
                WHERE atoms_fts MATCH ?
                  AND a.is_deleted = 0
                ORDER BY f.rank
                LIMIT ?
                """,
                (fts_query, k * 3),
            )
        except Exception as exc:
            log.debug("BM25 search failed (query=%r): %s", fts_query, exc)
            return {}

        if not rows:
            return {}

        # D1: Batch-fetch all candidate atoms in one query.
        candidate_ids = [row["atom_id"] for row in rows]
        atoms_map = await self._atoms.get_batch_without_tracking(candidate_ids)

        seeds: dict[int, float] = {}
        for row in rows:
            if len(seeds) >= k:
                break
            atom_id = row["atom_id"]
            atom = atoms_map.get(atom_id)
            if atom is None or atom.is_deleted:
                continue
            if region is not None and atom.region != region:
                continue
            if types is not None and atom.type not in types:
                continue
            seeds[atom_id] = row["raw_score"]

        if not seeds:
            return {}

        # Normalise raw BM25 scores to [0, 1].
        max_score = max(seeds.values())
        if max_score > 0:
            return {aid: score / max_score for aid, score in seeds.items()}
        return {aid: 0.0 for aid in seeds}

    # ------------------------------------------------------------------
    # Step 2: Spreading activation
    # ------------------------------------------------------------------

    def _get_synapse_type_weight(self, relationship: str) -> float:
        """Get the activation multiplier for a synapse type.

        Uses the pre-computed :attr:`_synapse_type_weights` dict for O(1)
        lookup instead of per-call attribute lookup and string replacement.

        Parameters
        ----------
        relationship:
            The synapse relationship type (e.g., "related-to", "caused-by").

        Returns
        -------
        float
            The weight multiplier for this synapse type (0.0 to 1.0).
        """
        return self._synapse_type_weights.get(relationship, 0.5)

    async def _spread_activation(
        self,
        seeds: dict[int, float],
        depth: int,
    ) -> tuple[dict[int, float], dict[int, Any]]:
        """Propagate activation energy through the synapse graph.

        Starting from the seed atoms (with their vector similarity as
        initial activation), traverses synapse connections up to *depth*
        hops.  At each hop the activation is attenuated by the synapse
        strength, the synapse type weight, and the configured decay factor.

        Synapse type weighting ensures that semantically meaningful connections
        (caused-by, elaborates, warns-against) propagate more activation than
        generic embedding-similarity connections (related-to).

        Parameters
        ----------
        seeds:
            Mapping of seed atom IDs to their initial activation levels
            (vector similarity scores).
        depth:
            Maximum number of hops to traverse from any seed atom.

        Returns
        -------
        tuple[dict[int, float], dict[int, Any]]
            A pair of ``(activated, atom_map)`` where *activated* maps all
            activated atom IDs to their final activation levels (including
            the original seed atoms) and *atom_map* contains every
            :class:`~memories.atoms.Atom` object fetched during spreading
            (keyed by atom ID), enabling callers to skip a redundant
            ``get_batch_without_tracking`` round-trip.
        """
        min_activation = self._cfg.min_activation

        # Initialise with seed activations.
        activated: dict[int, float] = dict(seeds)

        # Pre-fetch seed atoms so they appear in atom_map.  Seeds are not
        # fetched during the neighbour traversal below (since they are the
        # *source* of expansion, not the *target*), so we prime the map here.
        atom_map: dict[int, Any] = {}
        if seeds:
            seed_atoms = await self._atoms.get_batch_without_tracking(list(seeds.keys()))
            atom_map.update(seed_atoms)

        # Refractory visited set: gates FRONTIER EXPANSION only.
        # Seeds are pre-visited so they are never re-expanded into the frontier,
        # which breaks cycles and prevents infinite loops (Hebbian H1).
        visited: set[int] = set(seeds.keys())

        # The frontier is the set of atoms whose neighbours we will expand
        # at the next depth level.
        frontier: set[int] = set(seeds.keys())

        for level in range(1, depth + 1):
            next_frontier: set[int] = set()

            if not frontier:
                break

            # Perf H-1: batch-fetch all neighbours for the whole frontier in
            # one SQL query instead of one query per frontier atom.
            all_neighbor_map = await self._synapses.get_neighbors_batch(
                list(frontier), min_strength=min_activation
            )

            # Batch-fetch deletion status for all neighbour candidates.
            neighbor_ids = {
                nid
                for nbs in all_neighbor_map.values()
                for nid, _ in nbs
            }
            active_atoms = (
                await self._atoms.get_batch_without_tracking(list(neighbor_ids))
                if neighbor_ids
                else {}
            )
            # Accumulate non-deleted atoms into the persistent atom_map.
            atom_map.update(active_atoms)

            for atom_id in frontier:
                neighbors = all_neighbor_map.get(atom_id, [])

                # M1: Fanout normalization — compute degree BEFORE filtering so
                # high-degree hub nodes are always penalised, even if some
                # neighbours are skipped later (deleted / below threshold).
                fanout = len(neighbors)
                norm = 1.0 / math.sqrt(fanout) if fanout > 0 else 1.0

                current_activation = activated.get(atom_id, 0.0)

                for neighbor_id, synapse in neighbors:
                    # Skip soft-deleted atoms.
                    if active_atoms.get(neighbor_id) is None:
                        continue

                    type_weight = self._get_synapse_type_weight(synapse.relationship)
                    neighbor_activation = (
                        current_activation
                        * synapse.strength
                        * type_weight
                        * self._cfg.decay_factor
                        * norm
                    )

                    # Inhibitory: contradicts suppresses activation (subtract).
                    if synapse.relationship == "contradicts":
                        neighbor_activation = -neighbor_activation

                    # Superposition: update activation unconditionally so that
                    # convergent paths from multiple seeds both contribute
                    # (H1 refractory only gates expansion, not accumulation).
                    prev = activated.get(neighbor_id, 0.0)
                    new_val = max(0.0, min(1.0, prev + neighbor_activation))
                    activated[neighbor_id] = new_val

                    # H1 Refractory: only queue neighbours into the next
                    # frontier if they have not been visited before.
                    if neighbor_id not in visited and new_val >= min_activation:
                        next_frontier.add(neighbor_id)

            visited.update(next_frontier)
            frontier = next_frontier

            if not frontier:
                log.debug(
                    "Spreading activation exhausted at depth %d/%d",
                    level,
                    depth,
                )
                break

        log.debug(
            "Spreading activation: %d seeds expanded to %d atoms over %d levels",
            len(seeds),
            len(activated),
            depth,
        )
        return activated, atom_map

    # ------------------------------------------------------------------
    # Step 3: Multi-factor ranking
    # ------------------------------------------------------------------

    async def _score_atoms(
        self,
        vector_scores: dict[int, float],
        activation_scores: dict[int, float],
        bm25_scores: dict[int, float],
        include_antipatterns: bool,
        atom_map: dict[int, Any] | None = None,
    ) -> list[tuple[Atom, float, dict[str, float]]]:
        """Score all activated atoms using the multi-factor ranking function.

        Combines five signals -- vector similarity, spreading activation,
        recency, confidence, and access frequency -- into a single
        composite score.  Antipattern atoms receive a multiplicative boost
        to ensure warnings are surfaced.

        Parameters
        ----------
        vector_scores:
            Mapping of atom IDs to their vector similarity scores (only
            seed atoms will have entries here).
        activation_scores:
            Mapping of atom IDs to their activation levels from spreading
            activation (includes seeds and expanded atoms).
        include_antipatterns:
            Whether to apply the antipattern score boost.
        atom_map:
            Optional pre-fetched mapping of atom ID to :class:`~memories.atoms.Atom`
            collected during ``_spread_activation``.  When provided the
            redundant ``get_batch_without_tracking`` call is skipped entirely,
            reducing recall to a single SQL round-trip.

        Returns
        -------
        list[tuple[Atom, float, dict[str, float]]]
            Triples of ``(atom, composite_score, score_breakdown)`` sorted
            by descending score.  The breakdown dict contains the individual
            signal values before weighting.
        """
        weights = self._weights if self._weights is not None else self._cfg.weights
        now = datetime.now(tz=timezone.utc)

        scored: list[tuple[Atom, float, dict[str, float]]] = []

        _atoms_by_id = atom_map or await self._atoms.get_batch_without_tracking(
            list(activation_scores.keys())
        )

        for atom_id, spread_activation in activation_scores.items():
            atom = _atoms_by_id.get(atom_id)
            if atom is None:
                continue

            # Vector similarity: only seeds have this signal.
            vector_similarity = vector_scores.get(atom_id, 0.0)

            # BM25 keyword match score (0 for atoms not in FTS results).
            bm25_score = bm25_scores.get(atom_id, 0.0)

            # Recency score: 1.0 for today, decays linearly over 90 days.
            # New atoms with no access history fall back to created_at so they
            # aren't penalised with the 90-day floor on their first recall.
            ref_time_str = atom.last_accessed_at or atom.created_at
            if ref_time_str:
                try:
                    last_accessed = datetime.fromisoformat(ref_time_str)
                    # Ensure timezone-aware comparison.
                    if last_accessed.tzinfo is None:
                        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                    days_since = max(0, (now - last_accessed).days)
                except (ValueError, TypeError):
                    days_since = _RECENCY_DECAY_DAYS
            else:
                days_since = _RECENCY_DECAY_DAYS

            recency = max(
                _RECENCY_FLOOR,
                1.0 - (days_since / _RECENCY_DECAY_DAYS),
            )

            # Frequency score: logarithmic scale, capped at 1.0.
            # Use square root dampening to reduce dominance of old high-access atoms
            # Cap access_count at 100 to prevent old memories from dominating recall
            frequency = min(
                1.0,
                math.sqrt(math.log1p(min(atom.access_count, 100)) / math.log1p(_FREQUENCY_CAP_ACCESSES)),
            )

            # Newness bonus: boost recently created atoms to counteract old-memory bias.
            # This helps surface new learnings that haven't accumulated access counts yet.
            newness = 0.0
            if atom.created_at:
                try:
                    created = datetime.fromisoformat(atom.created_at)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    days_since_creation = max(0, (now - created).days)
                    if days_since_creation < _NEWNESS_BONUS_DAYS:
                        # Linear decay from 1.0 to 0.0 over the newness window
                        newness = 1.0 - (days_since_creation / _NEWNESS_BONUS_DAYS)
                except (ValueError, TypeError):
                    pass

            # Composite score.
            score = (
                vector_similarity * weights.vector_similarity
                + spread_activation * weights.spread_activation
                + recency * weights.recency
                + atom.confidence * weights.confidence
                + frequency * weights.frequency
                + atom.importance * weights.importance
                + newness * _NEWNESS_BONUS_WEIGHT
                + bm25_score * weights.bm25
            )

            # Antipattern boost: surface warnings more prominently.
            if atom.type == "antipattern" and include_antipatterns:
                score *= _ANTIPATTERN_BOOST

            breakdown = {
                "vector_similarity": round(vector_similarity, 4),
                "spread_activation": round(spread_activation, 4),
                "recency": round(recency, 4),
                "confidence": round(atom.confidence, 4),
                "frequency": round(frequency, 4),
                "importance": round(atom.importance, 4),
                "newness": round(newness, 4),
                "bm25": round(bm25_score, 4),
            }

            scored.append((atom, score, breakdown))

        # Sort by descending score.
        scored.sort(key=lambda triple: triple[1], reverse=True)

        return scored

    # ------------------------------------------------------------------
    # Pathway extraction
    # ------------------------------------------------------------------

    async def _extract_pathways(
        self,
        atom_ids: list[int],
    ) -> list[dict[str, Any]]:
        """Extract synapses that connect result atoms to each other.

        Returns the synapse graph structure between the atoms in the
        result set, which helps the consumer understand how the returned
        memories relate to one another.

        Parameters
        ----------
        atom_ids:
            IDs of the atoms in the final result set.

        Returns
        -------
        list[dict[str, Any]]
            Synapse dicts with keys: ``source_id``, ``target_id``,
            ``relationship``, ``strength``.
        """
        if len(atom_ids) < 2:
            return []

        atom_id_set = set(atom_ids)
        pathways: list[dict[str, Any]] = []
        seen_pairs: set[tuple[int, int]] = set()

        # Single batch query instead of N per-atom get_neighbors calls.
        neighbor_map = await self._synapses.get_neighbors_batch(
            atom_ids, min_strength=0.0
        )

        for atom_id in atom_ids:
            for neighbor_id, synapse in neighbor_map.get(atom_id, []):
                if neighbor_id not in atom_id_set:
                    continue

                # Deduplicate: use a canonical pair ordering.
                pair = (min(atom_id, neighbor_id), max(atom_id, neighbor_id))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                pathways.append({
                    "source_id": synapse.source_id,
                    "target_id": synapse.target_id,
                    "relationship": synapse.relationship,
                    "strength": synapse.strength,
                })

        # Sort by strength descending so the strongest connections appear first.
        pathways.sort(key=lambda p: p["strength"], reverse=True)

        return pathways

    # ------------------------------------------------------------------
    # Antipattern discovery
    # ------------------------------------------------------------------

    async def _find_relevant_antipatterns(
        self,
        atom_ids: list[int],
    ) -> list[Atom]:
        """Find antipattern atoms connected to result atoms via warns-against.

        Scans the synapse graph for ``warns-against`` relationships where
        the antipattern atom is the source and the target is in the result
        set.  These atoms are always surfaced in results to ensure the
        user sees relevant warnings.

        Parameters
        ----------
        atom_ids:
            IDs of the atoms in the current result set.

        Returns
        -------
        list[Atom]
            Antipattern atoms that warn against one or more result atoms.
        """
        if not atom_ids:
            return []

        antipatterns: dict[int, Atom] = {}

        # Batch fetch all neighbors in a single SQL round-trip.
        neighbor_map = await self._synapses.get_neighbors_batch(
            atom_ids, min_strength=0.0
        )

        # Collect candidate antipattern IDs from warns-against edges.
        candidate_ids: list[int] = []
        for atom_id in atom_ids:
            for neighbor_id, synapse in neighbor_map.get(atom_id, []):
                if (
                    synapse.relationship == "warns-against"
                    and neighbor_id not in antipatterns
                    and neighbor_id not in candidate_ids
                ):
                    candidate_ids.append(neighbor_id)

        # Batch fetch candidate atoms without inflating their access_count.
        # Using get() here would write phantom Hebbian signals for atoms that
        # are merely inspected by the retrieval engine, not recalled by the user.
        if candidate_ids:
            candidates = await self._atoms.get_batch_without_tracking(candidate_ids)
            for cid, atom in candidates.items():
                if atom and atom.type == "antipattern" and not atom.is_deleted:
                    antipatterns[cid] = atom

        return list(antipatterns.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _atom_to_scored_dict(
        atom: Atom,
        score: float,
        breakdown: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Convert an Atom and its score to a dict for the budget compressor.

        Parameters
        ----------
        atom:
            The atom to serialise.
        score:
            The composite retrieval score.
        breakdown:
            Optional per-signal score breakdown dict.

        Returns
        -------
        dict[str, Any]
            Atom dict with added ``score`` and optional ``score_breakdown``
            keys.
        """
        d = atom.to_dict()
        d["score"] = round(score, 6)
        if breakdown is not None:
            d["score_breakdown"] = breakdown
        return d
