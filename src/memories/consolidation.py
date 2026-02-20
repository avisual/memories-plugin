"""Memory consolidation: decay, pruning, merging, and promotion.

The **consolidation engine** implements the brain-like "sleep" maintenance
cycle for the memory graph.  When :meth:`ConsolidationEngine.reflect` is
called it runs a sequence of operations that mirror biological memory
consolidation:

1. **Decay** -- atoms not accessed within a configurable window have their
   confidence reduced (analogous to forgetting curves).  Synapse strengths
   are similarly decayed by a multiplicative factor.
2. **Prune** -- synapses whose post-decay strength falls below a threshold
   are deleted, removing dead pathways from the graph.
3. **Merge** -- near-duplicate atoms (very high embedding similarity, same
   type) are unified into a single canonical atom.  Tags are merged, synapses
   are redirected, and a ``supersedes`` link records provenance.
4. **Promote** -- frequently accessed atoms receive a confidence boost,
   cementing them as core memories.

All mutations can be previewed via the ``dry_run`` parameter, and every
significant action is logged to the ``consolidation_log`` table for
auditability.

Usage::

    from memories.consolidation import ConsolidationEngine

    engine = ConsolidationEngine(storage, embeddings, atoms, synapses)
    result = await engine.reflect(dry_run=True)
    print(result)

    # Actually apply
    result = await engine.reflect()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from memories.atoms import Atom, AtomManager
from memories.config import get_config
from memories.embeddings import EmbeddingEngine
from memories.storage import Storage
from memories.synapses import SynapseManager


def _content_signature(content: str) -> str:
    """Generate a normalized signature for duplicate detection.

    Strips whitespace, lowercases, and hashes the content to quickly
    identify exact or near-exact duplicates before expensive embedding
    comparisons.
    """
    normalized = " ".join(content.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence floor -- atoms never decay below this value automatically.
# Explicit forget (soft_delete) can still set confidence to 0.
# ---------------------------------------------------------------------------

_CONFIDENCE_FLOOR: float = 0.1
"""Minimum confidence that automatic decay will reduce an atom to.

This ensures that atoms are never silently obliterated by the background
decay process alone.  A human or explicit ``forget()`` call is required
to fully remove a memory."""


# ---------------------------------------------------------------------------
# ConsolidationResult
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationResult:
    """Summary of a single consolidation cycle.

    Attributes
    ----------
    merged:
        Number of atom pairs that were merged (duplicates removed).
    decayed:
        Number of atoms whose confidence was reduced.
    pruned:
        Number of synapses pruned (deleted) because their strength fell
        below the configured threshold.
    promoted:
        Number of atoms that received a confidence boost.
    compressed:
        Reserved for future content compression (currently unused).
    details:
        List of per-action detail dicts for logging and debugging.
    dry_run:
        Whether this was a preview run (no database mutations).
    """

    merged: int = 0
    decayed: int = 0
    pruned: int = 0
    promoted: int = 0
    compressed: int = 0
    reclassified: int = 0
    ltd: int = 0
    feedback_adjusted: int = 0
    abstracted: int = 0
    resolved: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dict for MCP tool responses.

        Returns
        -------
        dict[str, Any]
            All counters plus the detail log.
        """
        return {
            "merged": self.merged,
            "decayed": self.decayed,
            "pruned": self.pruned,
            "promoted": self.promoted,
            "compressed": self.compressed,
            "reclassified": self.reclassified,
            "ltd": self.ltd,
            "feedback_adjusted": self.feedback_adjusted,
            "abstracted": self.abstracted,
            "resolved": self.resolved,
            "dry_run": self.dry_run,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# ConsolidationEngine
# ---------------------------------------------------------------------------


class ConsolidationEngine:
    """Brain-like consolidation engine for the memory graph.

    Runs periodic maintenance that decays stale memories, prunes weak
    connections, merges near-duplicates, and promotes frequently accessed
    atoms.

    Parameters
    ----------
    storage:
        An initialised :class:`~memories.storage.Storage` instance.
    embeddings:
        An initialised :class:`~memories.embeddings.EmbeddingEngine` for
        vector similarity operations.
    atoms:
        An :class:`~memories.atoms.AtomManager` for atom CRUD.
    synapses:
        A :class:`~memories.synapses.SynapseManager` for synapse CRUD,
        decay, and pruning.
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
        self._cfg = get_config().consolidation
        cfg = get_config()
        self._ollama_url = cfg.ollama_url
        self._distill_model = cfg.distill_model
        self._distill_enabled = cfg.distill_thinking
        self._llm_client = None  # lazily created in _distill_cluster()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reflect(
        self,
        scope: str = "all",
        dry_run: bool = False,
    ) -> ConsolidationResult:
        """Run a full consolidation cycle.  Like sleep for the brain.

        Operations are executed in a deliberate order:

        1. **Decay atoms** -- reduce confidence of stale atoms.
        2. **Decay synapses** -- multiplicative strength reduction.
        3. **Prune synapses** -- remove weak connections.
        4. **Merge duplicates** -- unify near-identical atoms.
        5. **Promote strong** -- boost frequently accessed atoms.
        6. **Log** -- write a summary to ``consolidation_log``.

        Parameters
        ----------
        scope:
            ``"all"`` to process every atom, or a region name to limit
            processing to atoms in that region.
        dry_run:
            If ``True`` the engine computes what *would* happen without
            writing any mutations to the database.  Useful for previewing
            the impact of a consolidation cycle.

        Returns
        -------
        ConsolidationResult
            Aggregate counts and per-action detail log.
        """
        result = ConsolidationResult(dry_run=dry_run)
        holder = uuid.uuid4().hex

        # Acquire advisory lock to prevent concurrent consolidation.
        def _try_lock(conn: sqlite3.Connection) -> bool:
            return Storage.try_acquire_lock(conn, "consolidation", holder)

        acquired = await self._storage.execute_transaction(_try_lock)
        if not acquired:
            logger.warning("Consolidation already in progress; skipping")
            return result

        try:
            await self._tune_retrieval_weights(result)
            await self._reclassify_antipatterns(result)
            await self._apply_feedback_signals(result)
            await self._resolve_contradictions(result)
            # Decay before abstraction so stale atoms don't become cluster templates.
            await self._decay_atoms(result, scope)
            await self._decay_synapses(result)
            await self._prune_synapses(result)
            await self._prune_stale_warns_against(result)
            await self._apply_ltd(result)
            await self._prune_dormant_synapses(result)
            # Integrate hub atom cleanup to cap over-connected nodes.
            if not dry_run:
                hub_cleaned = await self._synapses.cleanup_hub_atoms()
                if hub_cleaned:
                    result.pruned += hub_cleaned
                    result.details.append({"action": "hub_cleanup", "synapses_deleted": hub_cleaned})
                outbound_cleaned = await self._synapses.cleanup_hub_atoms_outbound()
                if outbound_cleaned:
                    result.pruned += outbound_cleaned
                    result.details.append({"action": "hub_cleanup_outbound", "synapses_deleted": outbound_cleaned})
            await self._abstract_experiences(result, scope)
            # First pass: merge exact duplicates (fast, hash-based)
            await self._merge_exact_duplicates(result, scope)
            # Second pass: merge near-duplicates (slower, embedding-based)
            await self._merge_duplicates(result, scope)
            await self._promote_strong(result, scope)

            if not dry_run:
                await self._log_consolidation(result)
        finally:
            def _release(conn: sqlite3.Connection) -> None:
                Storage.release_lock(conn, "consolidation", holder)

            await self._storage.execute_transaction(_release)

        logger.info(
            "Consolidation %s complete: "
            "reclassified=%d  merged=%d  abstracted=%d  resolved=%d  "
            "decayed=%d  pruned=%d  ltd=%d  promoted=%d  feedback=%d",
            "(dry-run)" if dry_run else "",
            result.reclassified,
            result.merged,
            result.abstracted,
            result.resolved,
            result.decayed,
            result.pruned,
            result.ltd,
            result.promoted,
            result.feedback_adjusted,
        )

        return result

    async def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve recent entries from the consolidation log.

        Parameters
        ----------
        limit:
            Maximum number of log entries to return.

        Returns
        -------
        list[dict[str, Any]]
            Log entries ordered by most recent first, each containing
            ``id``, ``action``, ``details``, ``atoms_affected``, and
            ``created_at``.
        """
        rows = await self._storage.execute(
            """
            SELECT id, action, details, atoms_affected, created_at
            FROM consolidation_log
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        entries: list[dict[str, Any]] = []
        for row in rows:
            entry: dict[str, Any] = {
                "id": row["id"],
                "action": row["action"],
                "created_at": row["created_at"],
            }
            # Parse JSON columns back into Python objects.
            if row["details"]:
                try:
                    entry["details"] = json.loads(row["details"])
                except (json.JSONDecodeError, TypeError):
                    entry["details"] = row["details"]
            else:
                entry["details"] = None

            if row["atoms_affected"]:
                try:
                    entry["atoms_affected"] = json.loads(row["atoms_affected"])
                except (json.JSONDecodeError, TypeError):
                    entry["atoms_affected"] = row["atoms_affected"]
            else:
                entry["atoms_affected"] = None

            entries.append(entry)

        return entries

    # ------------------------------------------------------------------
    # Step 0: Reclassify misclassified antipatterns
    # ------------------------------------------------------------------

    async def _reclassify_antipatterns(self, result: ConsolidationResult) -> None:
        """Fix atoms wrongly stored as antipatterns by the post-tool hook.

        Before the _infer_atom_type fix, file edits whose old/new content
        contained the word "error" were stored as ``antipattern`` instead of
        ``experience``.  Command outputs with error text but no actual mistake
        were also misclassified.  This pass corrects them automatically so
        each reflect cycle cleans up any remaining noise.

        Patterns reclassified to ``experience``:
        - ``Edited <file>: changed ...``  (Edit/Write/MultiEdit summaries)
        - ``Command `...` produced ...``  (Bash output summaries)
        - ``Ran `...``` / ``Running ...``  (other tool-output summaries)
        """
        patterns = [
            "Edited %:%",
            "Command `%` produced%",
            "Ran `%`%",
            "Running `%`%",
        ]
        like_clauses = " OR ".join("content LIKE ?" for _ in patterns)
        rows = await self._storage.execute(
            f"""
            SELECT id, content FROM atoms
            WHERE type = 'antipattern'
              AND is_deleted = 0
              AND ({like_clauses})
            """,
            tuple(patterns),
        )
        if rows:
            if not result.dry_run:
                ids = [r["id"] for r in rows]
                placeholders = ",".join("?" * len(ids))
                await self._storage.execute_write(
                    f"UPDATE atoms SET type = 'experience' WHERE id IN ({placeholders})",
                    tuple(ids),
                )

            result.reclassified += len(rows)
            for row in rows:
                result.details.append({
                    "action": "reclassify",
                    "atom_id": row["id"],
                    "reason": "edit/command summary misclassified as antipattern",
                })
            logger.info("Reclassified %d misclassified antipatterns", len(rows))

        # --- Pass 2: short antipatterns lacking negative keywords -----------
        # 1,548+ atoms are misclassified as antipattern with content like
        # "A single line of code was edited" — they should be experience.
        # They have confidence=1.0 so the stale-pruning pass ignores them.
        await self._reclassify_short_antipatterns(result)

    async def _reclassify_short_antipatterns(
        self, result: ConsolidationResult
    ) -> None:
        """Reclassify short antipatterns that lack genuine negative keywords.

        Atoms typed ``antipattern`` with ``LENGTH(content) < 150`` and no
        recognized negative keywords (should not, avoid, never, wrong, bug,
        mistake, dangerous, risk, incorrect, don't, do not, problem, pitfall,
        antipattern) are reclassified to ``experience``.  Their outbound
        ``warns-against`` synapses are deleted in the same pass.
        """
        misclassified_rows = await self._storage.execute(
            """
            SELECT id FROM atoms
            WHERE type = 'antipattern'
              AND is_deleted = 0
              AND LENGTH(content) < 150
              AND LOWER(content) NOT LIKE '%should not%'
              AND LOWER(content) NOT LIKE '%avoid%'
              AND LOWER(content) NOT LIKE '%never%'
              AND LOWER(content) NOT LIKE '%wrong%'
              AND LOWER(content) NOT LIKE '%bug%'
              AND LOWER(content) NOT LIKE '%mistake%'
              AND LOWER(content) NOT LIKE '%dangerous%'
              AND LOWER(content) NOT LIKE '%risk%'
              AND LOWER(content) NOT LIKE '%incorrect%'
              AND LOWER(content) NOT LIKE '%don''t%'
              AND LOWER(content) NOT LIKE '%do not%'
              AND LOWER(content) NOT LIKE '%problem%'
              AND LOWER(content) NOT LIKE '%pitfall%'
              AND LOWER(content) NOT LIKE '%antipattern%'
              AND LOWER(content) NOT LIKE '%bad practice%'
              AND LOWER(content) NOT LIKE '%failure%'
              AND LOWER(content) NOT LIKE '%issue%'
              AND LOWER(content) NOT LIKE '%error%'
            """,
        )
        if not misclassified_rows:
            return

        ids = [r["id"] for r in misclassified_rows]

        if not result.dry_run:
            # Process in batches of 500 to respect SQLite placeholder limits.
            batch_size = 500
            for start in range(0, len(ids), batch_size):
                batch = ids[start : start + batch_size]
                id_ph = ",".join("?" * len(batch))
                await self._storage.execute_write(
                    f"UPDATE atoms SET type = 'experience' WHERE id IN ({id_ph})",
                    tuple(batch),
                )
                await self._storage.execute_write(
                    f"DELETE FROM synapses WHERE source_id IN ({id_ph}) "
                    f"AND relationship = 'warns-against'",
                    tuple(batch),
                )

        result.reclassified += len(ids)
        for atom_id in ids:
            result.details.append({
                "action": "reclassify",
                "atom_id": atom_id,
                "reason": "short antipattern lacking negative keywords",
            })
        logger.info(
            "Reclassified %d short misclassified antipatterns to experience",
            len(ids),
        )

    # ------------------------------------------------------------------
    # Step 0b: Long-Term Depression (anti-Hebbian synapse weakening)
    # ------------------------------------------------------------------

    async def _apply_ltd(self, result: ConsolidationResult) -> None:
        """Weaken synapses between atoms that consistently fire apart.

        Implements the anti-Hebbian rule: "neurons that fire apart wire
        apart."  A synapse qualifies for LTD when **both** of these hold:

        1. **Both endpoint atoms are individually active** — each was
           accessed within ``decay_after_days``, confirming they are still
           being used in the current context.
        2. **The synapse itself has not been Hebbian-reinforced** within
           ``ltd_window_days`` — the two atoms keep being accessed in
           separate sessions without co-occurring.

        This differs from passive synapse decay (which applies to all
        synapses regardless of atom activity) by targeting relationships
        that are actively diverging in usage context, not just fading from
        disuse.

        Each qualifying synapse is weakened by ``ltd_amount`` per cycle.
        If weakening pushes strength below ``prune_threshold``, the synapse
        is deleted by :meth:`~memories.synapses.SynapseManager.weaken`.
        """
        ltd_window_days = self._cfg.ltd_window_days
        activity_window = self._cfg.decay_after_days
        prune_threshold = self._cfg.prune_threshold

        rows = await self._storage.execute(
            """
            SELECT s.id, s.strength, s.source_id, s.target_id, s.relationship
            FROM synapses s
            JOIN atoms a1 ON a1.id = s.source_id
            JOIN atoms a2 ON a2.id = s.target_id
            WHERE s.strength > ?
              AND (
                  s.last_activated_at IS NULL
                  OR s.last_activated_at < datetime('now', ?)
              )
              AND s.relationship NOT IN ('contradicts', 'supersedes', 'warns-against')
              AND a1.is_deleted = 0
              AND a2.is_deleted = 0
              AND a1.last_accessed_at IS NOT NULL
              AND a2.last_accessed_at IS NOT NULL
              AND a1.last_accessed_at > datetime('now', ?)
              AND a2.last_accessed_at > datetime('now', ?)
            """,
            (prune_threshold, f"-{ltd_window_days} days", f"-{activity_window} days", f"-{activity_window} days"),
        )

        if not rows:
            return

        for row in rows:
            result.details.append({
                "action": "ltd",
                "synapse_id": row["id"],
                "relationship": row["relationship"],
                "old_strength": round(row["strength"], 4),
                "source_id": row["source_id"],
                "target_id": row["target_id"],
            })

        if not result.dry_run:
            ltd_fraction = self._cfg.ltd_fraction
            ltd_floor = self._cfg.ltd_min_floor
            multiplier = 1.0 - ltd_fraction

            synapse_ids = [row["id"] for row in rows]
            id_ph = ",".join("?" for _ in synapse_ids)

            # Batch proportional LTD: strength = MAX(floor, strength * multiplier).
            # A single UPDATE replaces the old per-row weaken() loop.
            await self._storage.execute_write(
                f"UPDATE synapses "
                f"SET strength = MAX(?, strength * ?) "
                f"WHERE id IN ({id_ph})",
                (ltd_floor, multiplier, *synapse_ids),
            )

            # Batch prune: delete any synapses that fell at or below prune_threshold.
            await self._storage.execute_write(
                f"DELETE FROM synapses "
                f"WHERE id IN ({id_ph}) AND strength <= ?",
                (*synapse_ids, prune_threshold),
            )

        result.ltd += len(rows)
        logger.info(
            "LTD applied to %d synapses (window=%d days, fraction=%.3f)",
            len(rows),
            ltd_window_days,
            self._cfg.ltd_fraction,
        )

    # ------------------------------------------------------------------
    # Step 0b2: Accelerated decay for dormant (never-activated) synapses
    # ------------------------------------------------------------------

    async def _prune_dormant_synapses(self, result: ConsolidationResult) -> None:
        """Accelerated decay for synapses that have never been activated.

        Synapses created but never traversed during spreading activation
        decay at ``dormant_multiplier`` rate per cycle.  Those already below
        ``prune_threshold`` are deleted immediately.

        Synapses where **both** endpoint atoms are recently active (accessed
        within ``decay_after_days``) are excluded -- those are already
        handled by :meth:`_apply_ltd` and applying both would cause a
        double-decay bug (0.85 * 0.80 = 0.68x per cycle).
        """
        dormant_cutoff_days = self._cfg.dormant_cutoff_days
        dormant_multiplier = self._cfg.dormant_multiplier

        rows = await self._storage.execute(
            """
            SELECT s.id, s.strength FROM synapses s
            JOIN atoms a1 ON a1.id = s.source_id
            JOIN atoms a2 ON a2.id = s.target_id
            WHERE s.last_activated_at IS NULL
              AND s.created_at < datetime('now', ?)
              AND s.strength > ?
              AND a1.is_deleted = 0
              AND a2.is_deleted = 0
              -- Exclude synapses already covered by _apply_ltd
              -- (_apply_ltd only fires when BOTH atoms are recently active)
              AND NOT (
                  a1.last_accessed_at IS NOT NULL
                  AND a1.last_accessed_at > datetime('now', ?)
                  AND a2.last_accessed_at IS NOT NULL
                  AND a2.last_accessed_at > datetime('now', ?)
              )
            """,
            (
                f"-{dormant_cutoff_days} days",
                self._cfg.prune_threshold,
                f"-{self._cfg.decay_after_days} days",
                f"-{self._cfg.decay_after_days} days",
            ),
        )
        if not rows:
            return

        synapse_ids = [row["id"] for row in rows]

        if result.dry_run:
            result.details.append({"action": "dormant_decay", "would_affect": len(synapse_ids)})
            return

        # Process in batches of 500 to stay within SQLite placeholder limits.
        batch_size = 500
        total_pruned = 0
        for start in range(0, len(synapse_ids), batch_size):
            batch = synapse_ids[start : start + batch_size]
            id_ph = ",".join("?" for _ in batch)

            await self._storage.execute_write(
                f"UPDATE synapses SET strength = MAX(?, strength * ?) WHERE id IN ({id_ph})",
                (self._cfg.prune_threshold, dormant_multiplier, *batch),
            )
            deleted = await self._storage.execute_write_returning(
                f"DELETE FROM synapses WHERE id IN ({id_ph}) AND strength <= ? RETURNING id",
                (*batch, self._cfg.prune_threshold),
            )
            total_pruned += len(deleted)

        result.pruned += total_pruned
        if total_pruned:
            logger.info(
                "Dormant synapse pruning: decayed %d, deleted %d (cutoff=%d days, multiplier=%.2f)",
                len(synapse_ids),
                total_pruned,
                dormant_cutoff_days,
                dormant_multiplier,
            )

    # ------------------------------------------------------------------
    # Step 0b3: Prune warns-against from low-confidence antipatterns
    # ------------------------------------------------------------------

    async def _prune_stale_warns_against(self, result: ConsolidationResult) -> None:
        """Delete warns-against synapses from decayed antipattern atoms.

        When an antipattern atom's confidence decays below 0.3, its
        ``warns-against`` synapses with strength < 0.4 are stale noise
        that should be removed to keep recall clean.
        """
        if result.dry_run:
            rows = await self._storage.execute(
                """
                SELECT COUNT(*) AS cnt FROM synapses
                WHERE relationship = 'warns-against'
                  AND strength < 0.4
                  AND source_id IN (
                      SELECT id FROM atoms
                      WHERE type = 'antipattern'
                        AND confidence < 0.3
                        AND is_deleted = 0
                  )
                """,
            )
            would_prune = rows[0]["cnt"] if rows else 0
            if would_prune:
                result.details.append({
                    "action": "prune_stale_warns_against",
                    "would_prune": would_prune,
                })
            return

        deleted = await self._storage.execute_write_returning(
            """
            DELETE FROM synapses
            WHERE relationship = 'warns-against'
              AND strength < 0.4
              AND source_id IN (
                  SELECT id FROM atoms
                  WHERE type = 'antipattern'
                    AND confidence < 0.3
                    AND is_deleted = 0
              )
            RETURNING id
            """,
        )
        if deleted:
            result.pruned += len(deleted)
            result.details.append({
                "action": "prune_stale_warns_against",
                "pruned": len(deleted),
            })
            logger.info(
                "Pruned %d stale warns-against synapses from low-confidence antipatterns",
                len(deleted),
            )

    # ------------------------------------------------------------------
    # Step 0c: Feedback signals → importance adjustment
    # ------------------------------------------------------------------

    async def _tune_retrieval_weights(self, result: ConsolidationResult) -> None:
        """Nudge retrieval weights toward atom properties that predict good recall.

        Uses accumulated ``atom_feedback`` signals from the last 30 days to
        compare the mean property values of positively-rated vs negatively-rated
        atoms.  When a property is consistently higher in good atoms it gets a
        small weight increase; the reverse triggers a decrease.  Changes are
        clamped to ±``weight_tuning_max_drift`` of the factory defaults.
        """
        cfg = get_config().consolidation
        lr = cfg.weight_tuning_learning_rate
        max_drift = cfg.weight_tuning_max_drift
        min_samples = cfg.weight_tuning_min_samples

        # Factory defaults used for clamping.
        DEFAULTS: dict[str, float] = {
            "confidence":        0.07,
            "importance":        0.11,
            "frequency":         0.07,
            "recency":           0.10,
            "spread_activation": 0.25,
        }

        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat()
        good_ids, bad_ids = await self._storage.get_feedback_atom_ids_since(cutoff)

        if len(good_ids) < min_samples or len(bad_ids) < min_samples:
            return  # not enough signal to tune

        good_atoms = list((await self._atoms.get_batch_without_tracking(good_ids)).values())
        bad_atoms  = list((await self._atoms.get_batch_without_tracking(bad_ids)).values())

        if not good_atoms or not bad_atoms:
            return

        def sig(a: Atom, name: str) -> float:
            if name == "confidence":
                return a.confidence
            if name == "importance":
                return a.importance
            if name == "frequency":
                return min(1.0, math.log1p(a.access_count) / math.log1p(100))
            # recency: 1.0 for today, 0.0 for 90+ days ago
            ref = a.last_accessed_at or a.created_at or ""
            try:
                days = max(
                    0,
                    (datetime.now(tz=timezone.utc) - datetime.fromisoformat(ref)).days,
                )
            except (ValueError, TypeError):
                days = 90
            return max(0.0, 1.0 - days / 90)

        stored = await self._storage.load_retrieval_weights() or dict(DEFAULTS)

        changed = False
        for signal, default in DEFAULTS.items():
            # spread_activation is a per-query signal, not a per-atom property;
            # skip feedback-driven tuning for it (kept at default in stored weights).
            if signal == "spread_activation":
                continue
            good_mean = sum(sig(a, signal) for a in good_atoms) / len(good_atoms)
            bad_mean  = sum(sig(a, signal) for a in bad_atoms)  / len(bad_atoms)
            diff = good_mean - bad_mean
            if abs(diff) < 0.05:
                continue  # no meaningful difference — skip
            direction = 1 if diff > 0 else -1
            lo = default * (1 - max_drift)
            hi = default * (1 + max_drift)
            stored[signal] = max(lo, min(hi, stored[signal] + direction * lr))
            changed = True

        if changed and not result.dry_run:
            await self._storage.save_retrieval_weights(stored)
            result.details.append({"action": "weights_tuned", "weights": stored})

    async def _apply_feedback_signals(self, result: ConsolidationResult) -> None:
        """Adjust atom importance based on accumulated user feedback.

        Each unprocessed ``good`` or ``bad`` signal in ``atom_feedback``
        nudges the atom's ``importance`` score.  Good feedback increases
        importance so the atom surfaces more prominently in future recalls;
        bad feedback decreases it.  Bad signals are weighted 1.5× because
        negative feedback is a stronger signal (the agent explicitly rated
        the atom as unhelpful, not merely absent).

        Records are marked ``processed_at = now`` after application so
        each feedback event is counted exactly once across consolidation
        cycles.

        The increments are controlled by :attr:`~ConsolidationConfig.feedback_good_increment`
        and :attr:`~ConsolidationConfig.feedback_bad_decrement`.
        """
        good_inc = self._cfg.feedback_good_increment
        bad_dec = self._cfg.feedback_bad_decrement

        rows = await self._storage.execute(
            """
            SELECT atom_id,
                   SUM(CASE WHEN signal = 'good' THEN 1 ELSE 0 END) AS good_count,
                   SUM(CASE WHEN signal = 'bad'  THEN 1 ELSE 0 END) AS bad_count,
                   GROUP_CONCAT(id) AS feedback_ids
            FROM atom_feedback
            WHERE processed_at IS NULL
            GROUP BY atom_id
            """
        )
        if not rows:
            return

        # E2: Batch-fetch all atom importances in one query instead of N queries.
        atom_ids_in_feedback = [int(r["atom_id"]) for r in rows]
        atoms_map = await self._atoms.get_batch_without_tracking(atom_ids_in_feedback)

        importance_updates: list[tuple[float, int]] = []
        all_feedback_ids: list[str] = []  # comma-separated id strings per group
        adjusted = 0

        for row in rows:
            atom_id = int(row["atom_id"])
            good = int(row["good_count"] or 0)
            bad = int(row["bad_count"] or 0)
            feedback_ids_str: str = row["feedback_ids"]
            all_feedback_ids.append(feedback_ids_str)

            # Net delta: bad is weighted 1.5× as a stronger signal.
            delta = (good * good_inc) - (bad * bad_dec * 1.5)

            atom = atoms_map.get(atom_id)
            if atom is None:
                # A4: atom deleted — will be marked processed below; skip delta.
                continue

            if abs(delta) < 1e-6:
                continue  # no net change; still mark processed below

            new_importance = max(0.0, min(1.0, atom.importance + delta))
            importance_updates.append((new_importance, atom_id))

            result.details.append({
                "action": "feedback_importance",
                "atom_id": atom_id,
                "good": good,
                "bad": bad,
                "delta": round(delta, 4),
                "old_importance": round(atom.importance, 4),
                "new_importance": round(new_importance, 4),
            })
            adjusted += 1

        if not result.dry_run:
            # Batch all importance updates in one execute_many.
            if importance_updates:
                await self._storage.execute_many(
                    "UPDATE atoms SET importance = ?, updated_at = datetime('now') WHERE id = ?",
                    importance_updates,
                )
            # Mark all feedback rows (including deleted-atom and no-change rows) processed.
            if all_feedback_ids:
                # Expand the comma-separated id strings into individual int ids
                # so we can build a proper parameterised query.
                flat_ids: list[int] = []
                for group in all_feedback_ids:
                    flat_ids.extend(int(i) for i in group.split(","))
                placeholders = ",".join("?" for _ in flat_ids)
                await self._storage.execute_write(
                    f"UPDATE atom_feedback SET processed_at = datetime('now') "
                    f"WHERE id IN ({placeholders})",
                    tuple(flat_ids),
                )

        result.feedback_adjusted = adjusted
        if adjusted:
            logger.info(
                "Feedback signals applied to %d atoms (good_inc=%.3f, bad_dec=%.3f)",
                adjusted,
                good_inc,
                bad_dec,
            )

    # ------------------------------------------------------------------
    # Step 0d: Contradiction resolution
    # ------------------------------------------------------------------

    async def _resolve_contradictions(self, result: ConsolidationResult) -> None:
        """Settle long-standing contradictions in favour of the stronger atom.

        Two atoms linked by a ``contradicts`` synapse inhibit each other during
        spreading activation indefinitely.  Once one side has accumulated
        substantially more evidence (higher confidence × usage), continuing
        to suppress the winner is counter-productive.

        Resolution criteria
        -------------------
        - Both atoms must be older than ``contradiction_min_age_days`` (14) so
          fresh contradictions from a single session are not auto-resolved.
        - ``winner_score >= loser_score × threshold`` (default 2.0) — the
          winner must be at least twice as strong.  This is deliberately
          conservative; close calls are left for the user to resolve.
        - Score formula: ``confidence × log1p(access_count)``.  The log scale
          prevents a single heavily-accessed atom from trivially winning.

        When resolution fires the loser's confidence is decremented by
        ``contradiction_resolution_decay`` (0.15) and a ``supersedes``
        synapse is created from winner to loser so the provenance is
        preserved.  The loser is not deleted — it may still hold unique
        context — but its reduced confidence will cause it to surface less.
        """
        threshold = self._cfg.contradiction_resolution_threshold
        decay_amount = self._cfg.contradiction_resolution_decay
        min_age_days = self._cfg.contradiction_min_age_days

        # E3: Join atom data directly into the synapse query to eliminate
        # per-pair get_without_tracking calls.
        rows = await self._storage.execute(
            """
            SELECT s.id AS synapse_id, s.source_id, s.target_id,
                   a1.confidence AS conf_a, a1.access_count AS acc_a,
                   a2.confidence AS conf_b, a2.access_count AS acc_b
            FROM synapses s
            JOIN atoms a1 ON a1.id = s.source_id AND a1.is_deleted = 0
            JOIN atoms a2 ON a2.id = s.target_id AND a2.is_deleted = 0
            WHERE s.relationship = 'contradicts'
              AND s.strength > 0
              AND a1.created_at < datetime('now', ?)
              AND a2.created_at < datetime('now', ?)
            """,
            (f"-{min_age_days} days", f"-{min_age_days} days"),
        )
        if not rows:
            return

        seen_pairs: set[frozenset[int]] = set()

        for row in rows:
            pair: frozenset[int] = frozenset([row["source_id"], row["target_id"]])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            score_a = row["conf_a"] * math.log1p(row["acc_a"])
            score_b = row["conf_b"] * math.log1p(row["acc_b"])

            if score_a >= score_b * threshold:
                winner_id = row["source_id"]
                loser_id = row["target_id"]
                loser_confidence = float(row["conf_b"])
                winner_score, loser_score = score_a, score_b
            elif score_b >= score_a * threshold:
                winner_id = row["target_id"]
                loser_id = row["source_id"]
                loser_confidence = float(row["conf_a"])
                winner_score, loser_score = score_b, score_a
            else:
                continue  # Too close to call — leave the contradiction active.

            new_confidence = max(0.0, loser_confidence - decay_amount)

            result.details.append({
                "action": "resolve_contradiction",
                "winner_id": winner_id,
                "loser_id": loser_id,
                "winner_score": round(winner_score, 4),
                "loser_score": round(loser_score, 4),
                "old_confidence": round(loser_confidence, 4),
                "new_confidence": round(new_confidence, 4),
            })

            if not result.dry_run:
                await self._atoms.update(loser_id, confidence=new_confidence)
                # Only create the supersedes synapse if one doesn't already
                # exist. Repeated cycles would otherwise BCM-upsert the
                # strength back to ~1.0, defeating decay.
                existing = await self._synapses._fetch_by_triple(
                    winner_id, loser_id, "supersedes"
                )
                if existing is None:
                    await self._synapses.create(
                        source_id=winner_id,
                        target_id=loser_id,
                        relationship="supersedes",
                        strength=0.8,
                        bidirectional=False,
                    )

            result.resolved += 1

        if result.resolved:
            logger.info(
                "Resolved %d contradiction(s) (threshold=%.1f, decay=%.3f)",
                result.resolved,
                threshold,
                decay_amount,
            )

    # ------------------------------------------------------------------
    # LLM distillation helpers (W3-D)
    # ------------------------------------------------------------------

    _GENERIC_PATTERNS: tuple[str, ...] = (
        "systems require proper",
        "the system is",
        "this is important",
        "it is necessary",
        "things can be",
        "processes should be",
    )

    @staticmethod
    def _distillation_is_acceptable(text: str) -> bool:
        """Return True if *text* is a specific, useful distillation.

        Rejects outputs that are too short (< 10 words) or match known
        generic abstraction patterns that dilute recall precision.
        Falls back to verbatim first-atom content when rejected.
        """
        if len(text.split()) < 10:
            return False
        lower = text.lower()
        return not any(p in lower for p in ConsolidationEngine._GENERIC_PATTERNS)

    async def _distill_cluster(self, cluster: list[Atom]) -> str:
        """Distil a cluster of experience atoms into a single insight string.

        Uses the local LLM (via :attr:`_llm_client`) to summarise the cluster
        into a concise, generalised statement.  Falls back to the verbatim
        content of the first atom when:

        * the LLM call exceeds the 15-second timeout, or
        * the cluster is empty, or
        * any other exception is raised (e.g. Ollama unreachable).

        Parameters
        ----------
        cluster:
            List of :class:`~memories.atoms.Atom` instances forming a
            similarity cluster.  Must be non-empty for a real LLM call.

        Returns
        -------
        str
            Distilled insight string, or verbatim first-atom content on
            fallback, or ``""`` for an empty cluster.
        """
        if not cluster:
            return ""

        fallback = cluster[0].content

        try:
            # Lazy import and client construction — only when actually needed.
            if self._llm_client is None:
                import ollama

                self._llm_client = ollama.AsyncClient(host=self._ollama_url)

            snippets = "\n".join(
                f"- {a.content[:200]}" for a in cluster[:10]
            )
            prompt = (
                "You are a memory consolidation assistant. "
                "Summarise the following related experiences into a single, "
                "concise general fact (1-2 sentences). "
                "Output only the fact, no preamble:\n\n"
                f"{snippets}"
            )

            response = await asyncio.wait_for(
                self._llm_client.generate(
                    model=self._distill_model,
                    prompt=prompt,
                ),
                timeout=15.0,
            )
            distilled: str = response.response.strip()
            if not distilled:
                return fallback
            if not self._distillation_is_acceptable(distilled):
                logger.debug(
                    "Distillation rejected (generic/too short): %r — using fallback",
                    distilled[:100],
                )
                return fallback
            return distilled
        except Exception as exc:
            logger.warning("Distillation failed for cluster: %s", exc, exc_info=True)
            return fallback

    async def _distill_clusters_concurrent(
        self, clusters: list[list[Atom]]
    ) -> list[str]:
        """Distil multiple clusters concurrently using :func:`asyncio.gather`.

        All clusters are submitted simultaneously rather than sequentially,
        reducing total wall-clock time when Ollama can handle parallel
        requests.  Each individual call still has a 15-second timeout (see
        :meth:`_distill_cluster`) and falls back to verbatim on error.

        Parameters
        ----------
        clusters:
            List of atom clusters, one per pending abstraction.

        Returns
        -------
        list[str]
            Distilled strings in the same order as *clusters*.  Exceptions
            from individual calls are caught inside :meth:`_distill_cluster`
            so every entry is always a string.
        """
        tasks = [self._distill_cluster(cluster) for cluster in clusters]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Ensure any unexpected exceptions (beyond what _distill_cluster
        # catches) still produce a valid fallback string.
        out: list[str] = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                out.append(clusters[i][0].content if clusters[i] else "")
            else:
                out.append(res)  # type: ignore[arg-type]
        return out

    # ------------------------------------------------------------------
    # Step 0e: Episodic-to-semantic abstraction
    # ------------------------------------------------------------------

    async def _abstract_experiences(
        self,
        result: ConsolidationResult,
        scope: str,
    ) -> None:
        """Promote clusters of similar experiences into generalised facts.

        When the same experience recurs across multiple sessions the system
        should stop treating it as a one-off anecdote and recognise it as a
        general truth.  This method:

        1. Finds ``experience`` atoms that are old enough to have had time to
           be corroborated (``abstraction_min_age_days``).
        2. For each candidate, searches for similar experiences using vector
           search (similarity >= ``abstraction_similarity``).
        3. Skips candidates already linked to a fact via a ``part-of`` synapse
           (they were abstracted in a previous cycle).
        4. If the cluster has >= ``abstraction_min_cluster`` members, creates
           a new ``fact`` atom whose content is taken from the most-accessed
           experience in the cluster (the one the system has found most useful
           as evidence).
        5. Confidence of the new fact scales with cluster size, capped at 0.85.
        6. Links all cluster members to the fact with ``part-of`` synapses so
           future cycles skip them.
        7. At most ``abstraction_max_per_cycle`` facts are created per run to
           keep consolidation fast.

        Parameters
        ----------
        result:
            The running consolidation result to update in place.
        scope:
            Region filter — ``"all"`` or a specific region name.
        """
        min_similarity = self._cfg.abstraction_similarity
        min_cluster = self._cfg.abstraction_min_cluster
        min_age_days = self._cfg.abstraction_min_age_days
        max_per_cycle = self._cfg.abstraction_max_per_cycle

        # Fetch eligible experience atoms.
        if scope == "all":
            rows = await self._storage.execute(
                """
                SELECT * FROM atoms
                WHERE type = 'experience'
                  AND is_deleted = 0
                  AND created_at < datetime('now', ?)
                ORDER BY access_count DESC
                """,
                (f"-{min_age_days} days",),
            )
        else:
            rows = await self._storage.execute(
                """
                SELECT * FROM atoms
                WHERE type = 'experience'
                  AND is_deleted = 0
                  AND region = ?
                  AND created_at < datetime('now', ?)
                ORDER BY access_count DESC
                """,
                (scope, f"-{min_age_days} days"),
            )

        candidates = [Atom.from_row(r) for r in rows]
        if len(candidates) < min_cluster:
            return

        # Pre-fetch IDs already abstracted (have a part-of synapse to a fact).
        abstracted_rows = await self._storage.execute(
            """
            SELECT DISTINCT s.source_id
            FROM synapses s
            JOIN atoms a ON a.id = s.target_id
            WHERE s.relationship = 'part-of'
              AND s.strength > 0
              AND a.type = 'fact'
              AND a.is_deleted = 0
            """
        )
        already_abstracted: set[int] = {r["source_id"] for r in abstracted_rows}

        processed_ids: set[int] = set()

        # ------------------------------------------------------------------
        # Phase 1: Build clusters — separate reuse vs. new-fact candidates.
        # We collect all clusters before writing so concurrent LLM distillation
        # (W3-D) can fire over all pending new-fact clusters in one gather call.
        # ------------------------------------------------------------------

        reuse_items: list[tuple[list[Atom], Atom]] = []
        new_items: list[tuple[list[Atom], Atom, float, float]] = []

        for seed in candidates:
            if len(reuse_items) + len(new_items) >= max_per_cycle:
                break
            if seed.id in already_abstracted or seed.id in processed_ids:
                continue

            try:
                similar_raw = await self._embeddings.search_similar(
                    seed.content, k=min_cluster * 4
                )
            except RuntimeError:
                logger.warning(
                    "Embedding unavailable for abstraction scan of atom %d; skipping",
                    seed.id,
                )
                continue

            # Build the cluster: seed + similar experiences above threshold.
            # Collect eligible candidate IDs first, then batch-fetch atoms.
            candidate_ids = [
                cid
                for cid, dist in similar_raw
                if cid != seed.id
                and cid not in already_abstracted
                and cid not in processed_ids
                and EmbeddingEngine.distance_to_similarity(dist) >= min_similarity
            ]
            id_to_atom = await self._atoms.get_batch_without_tracking(candidate_ids)

            cluster: list[Atom] = [seed]
            for candidate_id in candidate_ids:
                member = id_to_atom.get(candidate_id)
                if member is None or member.is_deleted or member.type != "experience":
                    continue
                if scope != "all" and member.region != scope:
                    continue
                cluster.append(member)

            if len(cluster) < min_cluster:
                continue

            # The most-accessed experience is the best template for the fact.
            template = max(cluster, key=lambda a: a.access_count)
            fact_confidence = min(0.85, 0.5 + len(cluster) * 0.10)
            fact_importance = max(a.importance for a in cluster)

            # Dedup guard: reuse an existing similar fact instead of creating
            # a near-duplicate.  This prevents repeated consolidation cycles
            # from accumulating facts that cover the same ground.
            existing_fact: Atom | None = None
            try:
                existing_raw = await self._embeddings.search_similar(
                    template.content, k=5
                )
                # M1: Batch-fetch all candidate IDs instead of per-item lookups.
                eligible_eids = [
                    eid
                    for eid, edist in existing_raw
                    if EmbeddingEngine.distance_to_similarity(edist) >= self._cfg.merge_threshold
                ]
                if eligible_eids:
                    eid_map = await self._atoms.get_batch_without_tracking(eligible_eids)
                    existing_fact = next(
                        (
                            eid_map[eid]
                            for eid in eligible_eids
                            if eid_map.get(eid)
                            and not eid_map[eid].is_deleted
                            and eid_map[eid].type == "fact"
                        ),
                        None,
                    )
            except RuntimeError:
                pass  # Vector search unavailable; proceed without dedup check.

            if existing_fact is not None:
                reuse_items.append((cluster, existing_fact))
            else:
                new_items.append((cluster, template, fact_confidence, fact_importance))

            processed_ids.update(a.id for a in cluster)

        # ------------------------------------------------------------------
        # Phase 2 (W3-D): Concurrently distil all new-fact clusters via LLM.
        # When distillation is disabled or fails, verbatim template content
        # is used as the fallback (no exception is ever raised to the caller).
        # ------------------------------------------------------------------

        if self._distill_enabled and new_items:
            distilled_contents = await self._distill_clusters_concurrent(
                [cluster for cluster, _, _, _ in new_items]
            )
        else:
            # Distillation disabled: use verbatim template content.
            distilled_contents = [template.content for _, template, _, _ in new_items]

        # ------------------------------------------------------------------
        # Phase 3: Write results to the database.
        # ------------------------------------------------------------------

        # Link cluster members to existing facts (reuse path).
        for cluster, existing_fact in reuse_items:
            cluster_template = max(cluster, key=lambda a: a.access_count)
            result.details.append({
                "action": "abstract_reuse",
                "existing_fact_id": existing_fact.id,
                "cluster_size": len(cluster),
                "cluster_ids": [a.id for a in cluster],
                "content_preview": cluster_template.content[:80],
            })
            if not result.dry_run:
                for exp in cluster:
                    await self._synapses.create(
                        source_id=exp.id,
                        target_id=existing_fact.id,
                        relationship="part-of",
                        strength=0.8,
                        bidirectional=False,
                    )

        # Create new facts using distilled (or verbatim) content.
        for (cluster, template, fact_confidence, fact_importance), fact_content in zip(
            new_items, distilled_contents
        ):
            result.details.append({
                "action": "abstract",
                "cluster_size": len(cluster),
                "cluster_ids": [a.id for a in cluster],
                "template_id": template.id,
                "fact_confidence": round(fact_confidence, 2),
                "content_preview": fact_content[:80],
            })

            if not result.dry_run:
                fact = await self._atoms.create(
                    content=fact_content,
                    type="fact",
                    region=template.region,
                    confidence=fact_confidence,
                    importance=fact_importance,
                    tags=template.tags,
                    source_project=template.source_project,
                    source_session=template.source_session,
                )
                # Link each experience → new fact so future cycles skip them.
                for exp in cluster:
                    await self._synapses.create(
                        source_id=exp.id,
                        target_id=fact.id,
                        relationship="part-of",
                        strength=0.8,
                        bidirectional=False,
                    )

            result.abstracted += 1

        if result.abstracted:
            logger.info(
                "Abstracted %d experience cluster(s) into facts "
                "(min_cluster=%d, similarity=%.2f)",
                result.abstracted,
                min_cluster,
                min_similarity,
            )

    # ------------------------------------------------------------------
    # Step 1: Decay atom confidence
    # ------------------------------------------------------------------

    async def _decay_atoms(
        self,
        result: ConsolidationResult,
        scope: str,
    ) -> None:
        """Decay confidence for atoms not accessed within the staleness window.

        For each stale atom the confidence is multiplied by the configured
        ``decay_rate`` (default 0.95), but never reduced below
        :data:`_CONFIDENCE_FLOOR` (0.1).

        Parameters
        ----------
        result:
            The running consolidation result to update in place.
        scope:
            Region filter -- ``"all"`` or a specific region name.
        """
        stale_atoms = await self._atoms.get_stale(self._cfg.decay_after_days)

        if scope != "all":
            stale_atoms = [a for a in stale_atoms if a.region == scope]

        if not stale_atoms:
            logger.debug("No stale atoms to decay")
            return

        decay_rate = self._cfg.decay_rate
        decay_after_days = self._cfg.decay_after_days
        type_decay = self._cfg.type_decay_multipliers
        affected_ids: list[int] = []
        confidence_updates: list[tuple[float, int]] = []
        now = datetime.now(tz=timezone.utc)

        for atom in stale_atoms:
            # Time-aware decay: older atoms decay more per cycle.
            # exponent = days_since_access / decay_after_days, so a just-stale
            # atom (exponent=1) decays by decay_rate^1, a twice-as-old atom
            # decays by decay_rate^2, etc.  Preserves backward compatibility
            # for newly-stale atoms while punishing truly abandoned memories.
            if atom.last_accessed_at:
                try:
                    last_accessed = datetime.fromisoformat(atom.last_accessed_at)
                    if last_accessed.tzinfo is None:
                        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                    days_since = max(decay_after_days, (now - last_accessed).days)
                except (ValueError, TypeError):
                    days_since = decay_after_days
            else:
                days_since = decay_after_days

            exponent = days_since / decay_after_days
            # B1: Per-type decay multiplier — skills/facts decay slowly,
            # experiences decay faster.
            effective_rate = decay_rate * type_decay.get(atom.type, 1.0)
            new_confidence = atom.confidence * (effective_rate ** exponent)

            # Enforce confidence floor.
            new_confidence = max(_CONFIDENCE_FLOOR, new_confidence)

            # Skip if the atom is already at or below the floor.
            if atom.confidence <= _CONFIDENCE_FLOOR:
                continue

            affected_ids.append(atom.id)
            confidence_updates.append((new_confidence, atom.id))

            result.details.append({
                "action": "decay",
                "atom_id": atom.id,
                "old_confidence": round(atom.confidence, 4),
                "new_confidence": round(new_confidence, 4),
                "days_since_access": days_since,
                "exponent": round(exponent, 2),
            })

        result.decayed = len(affected_ids)

        # E1: Batch all confidence updates in one execute_many instead of N updates.
        if not result.dry_run and confidence_updates:
            await self._storage.execute_many(
                "UPDATE atoms SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
                confidence_updates,
            )

        if affected_ids:
            logger.info(
                "Decayed %d stale atoms (rate=%.2f, floor=%.2f)",
                len(affected_ids),
                decay_rate,
                _CONFIDENCE_FLOOR,
            )

    # ------------------------------------------------------------------
    # Step 2: Decay synapse strengths (relationship-aware)
    # ------------------------------------------------------------------

    async def _decay_synapses(self, result: ConsolidationResult) -> None:
        """Apply multiplicative decay to all synapse strengths.

        Uses relationship-aware decay: generic "related-to" synapses decay
        faster than semantically typed connections (caused-by, elaborates, etc.).
        This addresses the 99.7% "related-to" problem by gradually pruning
        weak embedding-similarity connections while preserving meaningful ones.

        Decay rates:
        - related-to: decay_rate * 0.9 (faster decay)
        - All other types: decay_rate (normal decay)

        In dry-run mode the count of synapses that *would be* affected is
        estimated from the total synapse population.
        """
        if result.dry_run:
            # In dry-run mode we cannot call decay_all (it mutates).
            # Report the total synapse count as a rough upper bound.
            stats = await self._synapses.get_stats()
            total = stats.get("total", 0)
            related_to_count = stats.get("by_relationship", {}).get("related-to", 0)
            result.details.append({
                "action": "decay_synapses",
                "note": f"Would decay {total} synapses (related-to: {related_to_count} at accelerated rate)",
                "total_synapses": total,
                "related_to_count": related_to_count,
            })
            return

        base_decay = self._cfg.decay_rate
        related_to_decay = base_decay * 0.9  # 10% extra decay for generic connections

        # First, apply extra decay to "related-to" synapses
        await self._storage.execute_write(
            """
            UPDATE synapses
            SET strength = strength * ?
            WHERE relationship = 'related-to'
            """,
            (related_to_decay,),
        )

        # Then apply normal decay to all other synapses (exempt supersedes —
        # provenance pointers that must never decay).
        await self._storage.execute_write(
            """
            UPDATE synapses
            SET strength = strength * ?
            WHERE relationship NOT IN ('related-to', 'supersedes')
            """,
            (base_decay,),
        )

        # Now prune weak synapses
        pruned_by_decay = await self._synapses.prune_weak()

        result.details.append({
            "action": "decay_synapses",
            "base_factor": base_decay,
            "related_to_factor": related_to_decay,
            "synapses_pruned_by_decay": pruned_by_decay,
        })

        # Synapses pruned during decay are counted towards the prune total.
        result.pruned += pruned_by_decay

        logger.info(
            "Synapse decay applied (base=%.2f, related-to=%.2f): %d pruned",
            base_decay,
            related_to_decay,
            pruned_by_decay,
        )

    # ------------------------------------------------------------------
    # Step 3: Prune weak synapses
    # ------------------------------------------------------------------

    async def _prune_synapses(self, result: ConsolidationResult) -> None:
        """Remove synapses whose strength is below the prune threshold.

        Uses :meth:`SynapseManager.prune_weak`.  This is an explicit pass
        that catches any stragglers not already removed during synapse decay.
        """
        threshold = self._cfg.prune_threshold

        if result.dry_run:
            # Count how many would be pruned without actually deleting.
            rows = await self._storage.execute(
                "SELECT COUNT(*) AS cnt FROM synapses WHERE strength < ?",
                (threshold,),
            )
            would_prune = rows[0]["cnt"] if rows else 0
            result.details.append({
                "action": "prune_synapses",
                "note": f"Would prune {would_prune} synapses below threshold {threshold}",
                "would_prune": would_prune,
            })
            result.pruned += would_prune
            return

        pruned = await self._synapses.prune_weak(threshold)
        result.pruned += pruned

        result.details.append({
            "action": "prune_synapses",
            "threshold": threshold,
            "pruned": pruned,
        })

        if pruned > 0:
            logger.info(
                "Pruned %d weak synapses (threshold=%.4f)",
                pruned,
                threshold,
            )

    # ------------------------------------------------------------------
    # Step 4a: Merge exact duplicates (fast, hash-based)
    # ------------------------------------------------------------------

    async def _merge_exact_duplicates(
        self,
        result: ConsolidationResult,
        scope: str,
    ) -> None:
        """Merge atoms with identical normalized content (fast, O(n) pass).

        Uses content hashing to quickly identify exact duplicates before
        the more expensive embedding-based merge pass. This catches cases
        where the same memory was stored multiple times verbatim.

        Parameters
        ----------
        result:
            The running consolidation result to update in place.
        scope:
            Region filter -- ``"all"`` or a specific region name.
        """
        # Fetch all active atoms
        if scope == "all":
            rows = await self._storage.execute(
                "SELECT * FROM atoms WHERE is_deleted = 0 "
                "ORDER BY confidence DESC, created_at DESC"
            )
        else:
            rows = await self._storage.execute(
                "SELECT * FROM atoms WHERE is_deleted = 0 AND region = ? "
                "ORDER BY confidence DESC, created_at DESC",
                (scope,),
            )

        atoms_list = [Atom.from_row(r) for r in rows]

        if not atoms_list:
            return

        # Group atoms by content signature
        signature_groups: dict[str, list[Atom]] = {}
        for atom in atoms_list:
            sig = _content_signature(atom.content)
            if sig not in signature_groups:
                signature_groups[sig] = []
            signature_groups[sig].append(atom)

        # Find groups with duplicates (more than one atom)
        merged_count = 0
        for sig, group in signature_groups.items():
            if len(group) < 2:
                continue

            # Must be same type to merge
            type_groups: dict[str, list[Atom]] = {}
            for atom in group:
                if atom.type not in type_groups:
                    type_groups[atom.type] = []
                type_groups[atom.type].append(atom)

            for atom_type, same_type_group in type_groups.items():
                if len(same_type_group) < 2:
                    continue

                # Sort by confidence (desc) then creation time (desc)
                same_type_group.sort(
                    key=lambda a: (a.confidence, a.created_at),
                    reverse=True,
                )

                survivor = same_type_group[0]
                for duplicate in same_type_group[1:]:
                    result.details.append({
                        "action": "merge_exact",
                        "survivor_id": survivor.id,
                        "duplicate_id": duplicate.id,
                        "content_signature": sig,
                        "type": atom_type,
                    })

                    if not result.dry_run:
                        await self._execute_merge(survivor, duplicate)

                    merged_count += 1
                    logger.info(
                        "Merged exact duplicate: atom %d into %d (type=%s)",
                        duplicate.id,
                        survivor.id,
                        atom_type,
                    )

        result.merged += merged_count

        if merged_count > 0:
            logger.info("Merged %d exact duplicate atoms", merged_count)

    # ------------------------------------------------------------------
    # Step 4b: Merge near-duplicate atoms (embedding-based)
    # ------------------------------------------------------------------

    async def _merge_duplicates(
        self,
        result: ConsolidationResult,
        scope: str,
    ) -> None:
        """Find and merge near-duplicate atom pairs.

        Algorithm
        ---------
        1. Fetch all active (non-deleted) atoms in scope.
        2. For each atom, perform a vector similarity search.
        3. If a pair has similarity above ``merge_threshold`` (0.95) and
           shares the same type:

           a. **Keep** the atom with higher confidence (or the more recently
              created one if confidences are equal).
           b. Merge tags from both atoms onto the survivor.
           c. Redirect all synapses from the duplicate to the survivor.
           d. Create a ``supersedes`` synapse from survivor to duplicate.
           e. Soft-delete the duplicate.

        4. Already-processed pairs are tracked to avoid re-processing
           (since ``A -> B`` and ``B -> A`` would both be discovered).

        Parameters
        ----------
        result:
            The running consolidation result to update in place.
        scope:
            Region filter -- ``"all"`` or a specific region name.
        """
        # Fetch all active atoms, optionally filtered by region.
        if scope == "all":
            rows = await self._storage.execute(
                "SELECT * FROM atoms WHERE is_deleted = 0 "
                "ORDER BY confidence DESC, created_at DESC"
            )
        else:
            rows = await self._storage.execute(
                "SELECT * FROM atoms WHERE is_deleted = 0 AND region = ? "
                "ORDER BY confidence DESC, created_at DESC",
                (scope,),
            )

        atoms_list = [Atom.from_row(r) for r in rows]

        if not atoms_list:
            logger.debug("No active atoms to check for duplicates")
            return

        # Track pairs we have already considered to avoid double processing.
        merged_ids: set[int] = set()

        for atom in atoms_list:
            # Skip atoms that have already been consumed by a merge.
            if atom.id in merged_ids:
                continue

            try:
                similar = await self._embeddings.search_similar(
                    atom.content, k=10
                )
            except RuntimeError:
                logger.warning(
                    "Embedding unavailable for merge scan of atom %d; skipping",
                    atom.id,
                )
                continue

            # D1: Filter candidates above threshold, then batch-fetch atoms
            # instead of issuing per-candidate get_without_tracking calls.
            filtered_candidates = [
                (candidate_id, distance)
                for candidate_id, distance in similar
                if candidate_id != atom.id
                and candidate_id not in merged_ids
                and EmbeddingEngine.distance_to_similarity(distance) >= self._cfg.merge_threshold
            ]
            if not filtered_candidates:
                continue

            candidate_ids = [cid for cid, _ in filtered_candidates]
            candidates_map = await self._atoms.get_batch_without_tracking(candidate_ids)

            for candidate_id, distance in filtered_candidates:
                candidate = candidates_map.get(candidate_id)
                if candidate is None or candidate.is_deleted:
                    continue

                # Must be the same type to merge.
                if candidate.type != atom.type:
                    continue

                # Apply scope filter to the candidate as well.
                if scope != "all" and candidate.region != scope:
                    continue

                similarity = EmbeddingEngine.distance_to_similarity(distance)

                # Determine which atom survives.
                survivor, duplicate = self._pick_survivor(atom, candidate)

                result.details.append({
                    "action": "merge",
                    "survivor_id": survivor.id,
                    "duplicate_id": duplicate.id,
                    "similarity": round(similarity, 4),
                    "survivor_confidence": round(survivor.confidence, 4),
                    "duplicate_confidence": round(duplicate.confidence, 4),
                })

                if not result.dry_run:
                    await self._execute_merge(survivor, duplicate)

                merged_ids.add(duplicate.id)
                result.merged += 1

                logger.info(
                    "Merged atom %d into %d (similarity=%.4f, type=%s)",
                    duplicate.id,
                    survivor.id,
                    similarity,
                    atom.type,
                )

    @staticmethod
    def _pick_survivor(atom_a: Atom, atom_b: Atom) -> tuple[Atom, Atom]:
        """Decide which atom survives a merge.

        The atom with higher confidence wins.  On a tie the more recently
        created atom is preferred (it likely contains fresher information).

        Parameters
        ----------
        atom_a:
            First candidate.
        atom_b:
            Second candidate.

        Returns
        -------
        tuple[Atom, Atom]
            ``(survivor, duplicate)`` -- the first element is kept, the
            second is soft-deleted.
        """
        if atom_a.confidence > atom_b.confidence:
            return atom_a, atom_b
        if atom_b.confidence > atom_a.confidence:
            return atom_b, atom_a

        # Equal confidence -- prefer the more recent atom.
        if atom_a.created_at >= atom_b.created_at:
            return atom_a, atom_b
        return atom_b, atom_a

    async def _execute_merge(self, survivor: Atom, duplicate: Atom) -> None:
        """Carry out the actual merge of *duplicate* into *survivor*.

        The entire merge is executed inside a single transaction to
        guarantee crash-safety.  Steps:

        1. Merge tags from both atoms (union, deduplicated).
        2. Redirect all synapses from the duplicate to the survivor.
        3. Create a ``supersedes`` synapse from survivor to duplicate.
        4. Soft-delete the duplicate atom and decrement its region counter.

        Parameters
        ----------
        survivor:
            The atom that will remain active.
        duplicate:
            The atom that will be absorbed and soft-deleted.
        """
        merged_tags = list(dict.fromkeys(survivor.tags + duplicate.tags))
        tags_json = json.dumps(merged_tags) if merged_tags else None

        def _do_merge(conn: sqlite3.Connection) -> int:
            # 1. Merge tags onto survivor.
            conn.execute(
                "UPDATE atoms SET tags = ?, updated_at = datetime('now') WHERE id = ?",
                (tags_json, survivor.id),
            )

            # 2. Redirect synapses inline.
            rows = conn.execute(
                "SELECT id, source_id, target_id, relationship FROM synapses "
                "WHERE source_id = ? OR target_id = ?",
                (duplicate.id, duplicate.id),
            ).fetchall()

            redirected = 0
            for row in rows:
                sid = row["id"]
                source = row["source_id"]
                target = row["target_id"]
                rel = row["relationship"]

                new_source = survivor.id if source == duplicate.id else source
                new_target = survivor.id if target == duplicate.id else target

                if new_source == new_target:
                    conn.execute("DELETE FROM synapses WHERE id = ?", (sid,))
                    continue

                conflict = conn.execute(
                    "SELECT id FROM synapses "
                    "WHERE source_id = ? AND target_id = ? AND relationship = ?",
                    (new_source, new_target, rel),
                ).fetchone()

                if conflict:
                    conn.execute("DELETE FROM synapses WHERE id = ?", (sid,))
                else:
                    conn.execute(
                        "UPDATE synapses SET source_id = ?, target_id = ? WHERE id = ?",
                        (new_source, new_target, sid),
                    )
                    redirected += 1

            # 3. Create supersedes synapse (ignore if self-reference).
            if survivor.id != duplicate.id:
                conn.execute(
                    """
                    INSERT INTO synapses
                        (source_id, target_id, relationship, strength, bidirectional)
                    VALUES (?, ?, 'supersedes', 1.0, 0)
                    ON CONFLICT(source_id, target_id, relationship) DO UPDATE SET
                        strength = 1.0,
                        activated_count = synapses.activated_count + 1,
                        last_activated_at = datetime('now')
                    """,
                    (survivor.id, duplicate.id),
                )

            # 4. Soft-delete the duplicate and adjust region counter.
            conn.execute(
                "UPDATE atoms SET is_deleted = 1, updated_at = datetime('now') WHERE id = ?",
                (duplicate.id,),
            )
            conn.execute(
                "UPDATE regions SET atom_count = MAX(atom_count - 1, 0) WHERE name = ?",
                (duplicate.region,),
            )

            return redirected

        redirected = await self._storage.execute_transaction(_do_merge)
        logger.debug(
            "Merged atom %d into %d (redirected %d synapses)",
            duplicate.id,
            survivor.id,
            redirected,
        )

    # ------------------------------------------------------------------
    # Step 5: Promote frequently accessed atoms
    # ------------------------------------------------------------------

    async def _promote_strong(
        self,
        result: ConsolidationResult,
        scope: str,
    ) -> None:
        """Boost confidence for frequently accessed atoms.

        Atoms with ``access_count >= promote_access_count`` (default 20)
        receive a tiered confidence boost: each full multiple of
        ``promote_access_count`` earns +0.05, capped at 4 tiers (+0.20 max).
        More frequently accessed atoms earn larger boosts, rewarding core
        memories proportional to how heavily the system relies on them.

        Parameters
        ----------
        result:
            The running consolidation result to update in place.
        scope:
            Region filter -- ``"all"`` or a specific region name.
        """
        min_access = self._cfg.promote_access_count

        if scope == "all":
            rows = await self._storage.execute(
                """
                SELECT * FROM atoms
                WHERE is_deleted = 0
                  AND access_count >= ?
                  AND confidence < 1.0
                ORDER BY access_count DESC
                """,
                (min_access,),
            )
        else:
            rows = await self._storage.execute(
                """
                SELECT * FROM atoms
                WHERE is_deleted = 0
                  AND access_count >= ?
                  AND confidence < 1.0
                  AND region = ?
                ORDER BY access_count DESC
                """,
                (min_access, scope),
            )

        candidates = [Atom.from_row(r) for r in rows]

        if not candidates:
            logger.debug("No atoms eligible for promotion")
            return

        # E1: Collect all confidence updates and batch into one execute_many.
        confidence_updates: list[tuple[float, int]] = []

        for atom in candidates:
            # Tiered boost: each full multiple of promote_access_count earns +0.05,
            # capped at 4 tiers (+0.20 max).  More-accessed atoms get more credit.
            tiers = min(4, atom.access_count // min_access)
            boost = tiers * 0.05
            new_confidence = min(1.0, atom.confidence + boost)

            result.details.append({
                "action": "promote",
                "atom_id": atom.id,
                "access_count": atom.access_count,
                "old_confidence": round(atom.confidence, 4),
                "new_confidence": round(new_confidence, 4),
            })
            confidence_updates.append((new_confidence, atom.id))
            result.promoted += 1

        if not result.dry_run and confidence_updates:
            await self._storage.execute_many(
                "UPDATE atoms SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
                confidence_updates,
            )

        logger.info(
            "Promoted %d frequently accessed atoms (min_access=%d)",
            result.promoted,
            min_access,
        )

    # ------------------------------------------------------------------
    # Consolidation logging
    # ------------------------------------------------------------------

    async def _log_consolidation(self, result: ConsolidationResult) -> None:
        """Write a summary record to the ``consolidation_log`` table.

        Encodes the detail log and affected atom IDs as JSON strings for
        the ``details`` and ``atoms_affected`` columns respectively.

        Parameters
        ----------
        result:
            The completed consolidation result.
        """
        # Extract all unique atom IDs mentioned in the details.
        affected_ids: set[int] = set()
        for detail in result.details:
            for key in ("atom_id", "survivor_id", "duplicate_id"):
                if key in detail:
                    affected_ids.add(detail[key])

        summary = {
            "merged": result.merged,
            "decayed": result.decayed,
            "pruned": result.pruned,
            "promoted": result.promoted,
        }

        # Write one summary row per consolidation cycle.
        await self._storage.execute_write(
            """
            INSERT INTO consolidation_log (action, details, atoms_affected)
            VALUES (?, ?, ?)
            """,
            (
                "reflect",
                json.dumps(summary),
                json.dumps(sorted(affected_ids)),
            ),
        )

        # Optionally write per-action rows for fine-grained auditing.
        action_types = {"merge", "decay", "promote"}
        for action_type in action_types:
            action_details = [
                d for d in result.details if d.get("action") == action_type
            ]
            if not action_details:
                continue

            action_atom_ids: set[int] = set()
            for detail in action_details:
                for key in ("atom_id", "survivor_id", "duplicate_id"):
                    if key in detail:
                        action_atom_ids.add(detail[key])

            await self._storage.execute_write(
                """
                INSERT INTO consolidation_log (action, details, atoms_affected)
                VALUES (?, ?, ?)
                """,
                (
                    action_type,
                    json.dumps(action_details),
                    json.dumps(sorted(action_atom_ids)),
                ),
            )

        logger.debug("Consolidation actions logged to consolidation_log table")
