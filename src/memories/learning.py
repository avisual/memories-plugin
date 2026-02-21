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

import json
import logging
import re
from typing import Any

from memories.atoms import ATOM_TYPES, Atom, AtomManager
from memories.config import get_config
from memories.embeddings import EmbeddingEngine
from memories.storage import Storage
from memories.synapses import RELATIONSHIP_TYPES, SynapseManager

logger = logging.getLogger(__name__)

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
        stc_strength = self._cfg.learning.stc_tagged_strength
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

        for candidate_id, similarity in filtered:
            candidate = atoms_map.get(candidate_id)
            if candidate is None:
                continue

            # --- (a) primary synapse with inferred relationship type ------
            # Skip the generic relationship when either endpoint is an
            # antipattern — the warns-against synapse in step (b) is the
            # only conceptual link needed.  Creating both a related-to AND
            # a warns-against doubles the write cost and inflates the
            # warns-against share in the synapse distribution.
            is_antipattern_pair = (
                atom.type == "antipattern" or candidate.type == "antipattern"
            )

            if not is_antipattern_pair:
                # Heuristic: detect causal, elaboration, or compositional
                # language in either atom and use the most specific type.
                # LLM hook (future): for high-similarity pairs where heuristics
                # return "related-to", an LLM classifier could be called here:
                #   if relationship == "related-to" and similarity > 0.87:
                #       relationship = await self._classify_relationship_llm(
                #           atom, candidate
                #       )
                relationship, bidirectional = self._infer_relationship_type(
                    atom, candidate
                )

                # B3: Enforce outbound degree cap for related-to synapses.
                if relationship == "related-to":
                    if outbound_budget <= 0:
                        continue
                    outbound_budget -= 1

                # STC: related-to synapses start at tagged strength and must
                # be reinforced within the capture window.  Typed semantic
                # links (caused-by, elaborates, part-of) use full similarity
                # since they carry explicit evidence.
                use_tagged = relationship == "related-to"
                initial_strength = stc_strength if use_tagged else similarity

                synapse = await self._safe_create_synapse(
                    source_id=atom_id,
                    target_id=candidate_id,
                    relationship=relationship,
                    strength=initial_strength,
                    bidirectional=bidirectional,
                )
                if synapse is not None:
                    created_synapses.append(synapse)
                    # Collect for batch tag expiry setting.
                    if use_tagged:
                        stc_tagged_triples.append((
                            stc_window,
                            synapse["source_id"],
                            synapse["target_id"],
                            synapse["relationship"],
                        ))

            # --- (b) warns-against for antipatterns -----------------------
            if candidate.type == "antipattern":
                ap_synapse = await self._safe_create_synapse(
                    source_id=candidate_id,
                    target_id=atom_id,
                    relationship="warns-against",
                    strength=similarity,
                    bidirectional=False,
                )
                if ap_synapse is not None:
                    created_synapses.append(ap_synapse)

            # If the *new* atom is the antipattern, warn against the candidate.
            if atom.type == "antipattern" and candidate.type != "antipattern":
                ap_synapse = await self._safe_create_synapse(
                    source_id=atom_id,
                    target_id=candidate_id,
                    relationship="warns-against",
                    strength=similarity,
                    bidirectional=False,
                )
                if ap_synapse is not None:
                    created_synapses.append(ap_synapse)

            # --- (c) contradiction detection + retroactive interference -----
            if self._is_potential_contradiction(atom, candidate, similarity):
                contra_synapse = await self._safe_create_synapse(
                    source_id=atom_id,
                    target_id=candidate_id,
                    relationship="contradicts",
                    strength=similarity,
                    bidirectional=True,
                )
                if contra_synapse is not None:
                    created_synapses.append(contra_synapse)
                    # Retroactive interference: weaken the older atom's
                    # confidence immediately.  New competing memories
                    # interfere with older conflicting ones upon detection,
                    # rather than waiting for consolidation-time resolution.
                    #
                    # Safeguards:
                    # - Confidence floor (0.1) prevents auto-destruction
                    # - Atomic SQL prevents lost-update races
                    # - Floor guard prevents stacking from multiple calls
                    _INTERFERENCE_FLOOR = 0.1
                    penalty = self._cfg.learning.interference_confidence_penalty
                    if penalty > 0 and candidate.created_at <= atom.created_at:
                        await self._storage.execute_write(
                            """
                            UPDATE atoms
                            SET confidence = MAX(?, confidence - ?),
                                updated_at = datetime('now')
                            WHERE id = ? AND confidence > ?
                            """,
                            (_INTERFERENCE_FLOOR, penalty,
                             candidate_id, _INTERFERENCE_FLOOR),
                        )
                        logger.info(
                            "Retroactive interference: atom %d "
                            "confidence reduced by %.2f "
                            "(contradicted by new atom %d)",
                            candidate_id,
                            penalty,
                            atom_id,
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

        # Step 1: Hebbian update for co-activated atoms.
        total_updated = await self._synapses.hebbian_update(
            atom_ids, atom_timestamps=atom_timestamps
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
            "Session %s learning complete: %d synapses updated/created "
            "from %d atoms",
            session_id,
            total_updated,
            len(atom_ids),
        )

        # The hebbian_update method handles both strengthening existing
        # synapses and creating new ones, returning the combined count.
        # We report the total under both keys for downstream consumers
        # that may distinguish the two in future.
        return {
            "synapses_strengthened": total_updated,
            "synapses_created": total_updated,
        }

    # ------------------------------------------------------------------
    # Novelty assessment
    # ------------------------------------------------------------------

    async def assess_novelty(
        self,
        content: str,
        threshold: float = 0.7,
    ) -> bool:
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
        bool
            ``True`` if the content is sufficiently different from existing
            atoms (i.e. it is novel and should be stored).  ``False`` if a
            highly similar, confident atom already exists.
        """
        if not content or not content.strip():
            return False

        try:
            similar = await self._embeddings.search_similar(content, k=1)
        except RuntimeError:
            # If embedding is unavailable, err on the side of storing.
            logger.warning(
                "Embedding unavailable for novelty assessment; assuming novel"
            )
            return True

        if not similar:
            # No existing atoms at all -- definitely novel.
            return True

        closest_id, distance = similar[0]
        similarity = EmbeddingEngine.distance_to_similarity(distance)

        if similarity <= threshold:
            # Not similar enough to be redundant.
            return True

        # Check the confidence of the existing atom.
        existing = await self._atoms.get_without_tracking(closest_id)
        if existing is None:
            return True

        if existing.confidence > 0.7:
            logger.debug(
                "Content not novel: %.3f similarity to atom %d "
                "(confidence=%.2f)",
                similarity,
                closest_id,
                existing.confidence,
            )
            return False

        # Similar but low-confidence existing atom -- the new content may
        # be a correction or update, so treat it as novel.
        return True

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
            "resulting from", "owing to",
        )
        if any(p in combined for p in _CAUSAL):
            return "caused-by", False

        # elaborates: explicit expansion / detail language.
        _ELABORATION = (
            "specifically,", "more precisely", "in particular",
            "for example", "for instance", " i.e.", "namely,",
            "in other words", "to be specific", "that is,",
        )
        if any(p in combined for p in _ELABORATION):
            return "elaborates", False

        # part-of: compositional / membership language.
        _PARTOF = (
            "is part of", "is a component", "belongs to",
            "is a module", "is a subclass", "is a type of",
            "is a subset", "is included in",
        )
        if any(p in combined for p in _PARTOF):
            return "part-of", False

        # Length heuristic: if one atom is ≥ 2.5× longer, it likely
        # provides more detail about the shorter one → elaborates.
        len_a = len(atom_a.content)
        len_b = len(atom_b.content)
        if len_a > 0 and len_b > 0:
            shorter = min(len_a, len_b)
            longer = max(len_a, len_b)
            if longer / shorter >= 2.5:
                return "elaborates", False

        # Generic fallback — bidirectional topical association.
        return "related-to", True

    async def _set_tag_expiry(
        self,
        source_id: int,
        target_id: int,
        relationship: str,
        window_days: int,
    ) -> None:
        """Set the STC tag expiry timestamp on a newly created synapse.

        Only sets the tag if the synapse does not already have one (i.e. it
        is truly new, not an existing synapse that was strengthened via upsert).
        """
        await self._storage.execute_write(
            """
            UPDATE synapses
            SET tag_expires_at = datetime('now', '+' || ? || ' days')
            WHERE source_id = ? AND target_id = ? AND relationship = ?
              AND tag_expires_at IS NULL
              AND activated_count <= 1
            """,
            (window_days, source_id, target_id, relationship),
        )

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

        if negations_a != negations_b:
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
