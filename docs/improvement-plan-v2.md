# Memories Improvement Plan v2
# Synthesised from: Performance Architect + Hebbian Architect cross-review
# Date: 2026-02-20
# Max iterations: 5

---

## Process

For each wave:
1. Write failing tests
2. Run tests — confirm they fail
3. Implement changes
4. Run tests — confirm they pass
5. Commit

After all waves: spawn fresh Opus architects (with `mcp__memories__recall` first) for round-2 review.

---

## Wave 1 — Correctness & SQL Safety (fully independent, can parallelise)

### W1-A: `hebbian_update` rewrite (`synapses.py:556-675`)

Three Hebbian fixes composed into one atomic edit:

**C1 — strengthen ALL synapses per pair (not just first)**
- Change Step 1 SELECT to also fetch `relationship`: `SELECT id, source_id, target_id, relationship FROM synapses WHERE ...`
- Change `existing_by_pair` type from `dict[tuple, int]` → `dict[tuple, list[tuple[int, str]]]` (stores `[(synapse_id, relationship), ...]`)
- Use `setdefault(pair, []).append((row["id"], row["relationship"]))`

**C2 — exclude inhibitory types from strengthening**
- Filter during C1 list-building (before adding to `strengthen_ids`)
- `INHIBITORY = frozenset({"contradicts", "supersedes", "warns-against"})`
- Skip any `(synapse_id, rel)` where `rel in INHIBITORY`
- CRITICAL: inhibitory-only pairs must NOT fall through to the "create new related-to" branch.
  The `if pair in existing_by_pair` check remains true (pair HAS synapses, just all inhibitory),
  so no new related-to is created. Verify this in tests.

**H2 — BCM multiplicative increment**
- Change batch UPDATE from `strength + ?` to `strength + ? * (1.0 - strength)`
- SQL: `SET strength = MIN(1.0, strength + ? * (1.0 - strength))`
- Single `?` param still works — SQLite evaluates per-row using each row's `strength`
- Same applies to ON CONFLICT clause for new pairs

```python
INHIBITORY = frozenset({"contradicts", "supersedes", "warns-against"})

existing_by_pair: dict[tuple[int, int], list[int]] = {}
for row in existing_rows:
    if row["relationship"] in INHIBITORY:
        continue  # C2: skip inhibitory
    pair = (min(row["source_id"], row["target_id"]), max(row["source_id"], row["target_id"]))
    existing_by_pair.setdefault(pair, []).append(row["id"])

strengthen_ids = [sid for ids in existing_by_pair.values() for sid in ids]

if strengthen_ids:
    id_ph = ",".join("?" * len(strengthen_ids))
    await self._storage.execute_write(
        f"UPDATE synapses "
        f"SET strength = MIN(1.0, strength + ? * (1.0 - strength)), "
        f"    activated_count = activated_count + 1, "
        f"    last_activated_at = datetime('now') "
        f"WHERE id IN ({id_ph})",
        (increment, *strengthen_ids),
    )
```

### W1-B: SQL injection hardening (`consolidation.py`)

**Perf C1 — `_apply_feedback_signals` (lines 618-623)**
```python
# BEFORE (injection risk):
combined = ",".join(all_feedback_ids)
f"WHERE id IN ({combined})"

# AFTER:
placeholders = ",".join("?" for _ in all_feedback_ids)
await self._storage.execute_write(
    f"UPDATE atom_feedback SET processed_at = datetime('now') WHERE id IN ({placeholders})",
    tuple(all_feedback_ids),
)
```

**Perf C2 — `_reclassify_antipatterns` (lines 349-358)**
```python
like_clauses = " OR ".join("content LIKE ?" for _ in patterns)
rows = await self._storage.execute(
    f"SELECT id, content FROM atoms WHERE type = 'antipattern' AND is_deleted = 0 AND ({like_clauses})",
    tuple(patterns),
)
```

**Perf C3 — datetime f-strings (5 locations)**
Replace `f"datetime('now', '-{N} days')"` with `f"datetime('now', ?)"` and pass `f"-{N} days"` as param.
Locations: `_apply_ltd` (×2), `_resolve_contradictions` (×1), `_abstract_experiences` (×2).
Ensure Python-side UTC if any cutoffs computed in Python.

**Perf M9 — `_normalise_regions` uses `execute()` for writes (`cli.py:1023,1032,1041`)**
Change `execute` → `execute_write` for all UPDATE/INSERT in `_normalise_regions`.

---

## Wave 2 — Hot Path (one dependency: add `get_neighbors_batch` first)

### W2-PRE: Add `Synapses.get_neighbors_batch()` (`synapses.py`)

Prerequisite for W2-A and W2-C. Add as new method — no existing code modified.

```python
async def get_neighbors_batch(
    self, atom_ids: list[int], min_strength: float = 0.0
) -> dict[int, list[tuple[int, "Synapse"]]]:
    if not atom_ids:
        return {}
    placeholders = ",".join("?" for _ in atom_ids)
    rows = await self._storage.execute(
        f"SELECT * FROM synapses "
        f"WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders})) "
        f"  AND strength >= ? "
        f"ORDER BY strength DESC",
        (*atom_ids, *atom_ids, min_strength),
    )
    result: dict[int, list[tuple[int, Synapse]]] = {aid: [] for aid in atom_ids}
    for row in rows:
        syn = Synapse.from_row(row)
        if syn.source_id in result:
            result[syn.source_id].append((syn.target_id, syn))
        if syn.target_id in result:
            result[syn.target_id].append((syn.source_id, syn))
    return result
```

### W2-A: `_spread_activation` combined rewrite (`retrieval.py:610-714`)

Three components applied in one edit (Perf H-1 + Hebbian H1 refractory + Hebbian M1 fanout norm):

**Implementation order within the loop:**
1. Batch-fetch all frontier neighbors (H-1): one SQL call per depth level
2. Compute pre-filter fanout degree: `fanout = len(neighbors)` — BEFORE refractory filter
3. Apply refractory/visited filter (H1): `if neighbor_id in visited: continue`
   - Visited set gates FRONTIER EXPANSION only, NOT activation accumulation
   - Activated nodes can still receive additional activation from convergent paths (superposition preserved)
   - Correct pattern: update `activated[neighbor_id]` before checking `if neighbor_id in visited`
4. Compute fanout-normalised activation (M1): `activation *= 1.0 / sqrt(fanout)` if fanout > 0

```python
visited: set[int] = set(seeds.keys())

for level in range(1, depth + 1):
    next_frontier: set[int] = set()
    if not frontier:
        break

    # H-1: Batch fetch all neighbors for this depth level
    all_neighbor_map = await self._synapses.get_neighbors_batch(
        list(frontier), min_strength=min_activation
    )
    neighbor_ids = {nid for nbs in all_neighbor_map.values() for nid, _ in nbs}
    active_atoms = await self._atoms.get_batch_without_tracking(list(neighbor_ids))

    for atom_id in frontier:
        neighbors = all_neighbor_map.get(atom_id, [])
        fanout = len(neighbors)  # M1: pre-filter degree
        norm = 1.0 / math.sqrt(fanout) if fanout > 0 else 1.0

        current_activation = activated[atom_id]

        for neighbor_id, synapse in neighbors:
            if active_atoms.get(neighbor_id) is None:
                continue

            type_weight = self._get_synapse_type_weight(synapse.relationship)
            decay_factor = self._cfg.spread_activation ** level
            neighbor_activation = current_activation * synapse.strength * type_weight * decay_factor * norm

            if synapse.relationship == "contradicts":
                neighbor_activation = -neighbor_activation

            # Update activation (superposition: additive, allow convergent paths)
            prev = activated.get(neighbor_id, 0.0)
            new_val = max(0.0, min(1.0, prev + neighbor_activation))
            activated[neighbor_id] = new_val

            # H1: refractory — only add to frontier if not yet visited
            if neighbor_id not in visited and new_val >= min_activation:
                next_frontier.add(neighbor_id)

    visited.update(next_frontier)
    frontier = next_frontier
```

### W2-B: `_apply_ltd` combined rewrite (`consolidation.py:384-454`)

Three changes in one edit (Perf C3 datetime params + Hebbian H3 proportional LTD + batch weaken loop):

```python
# Parameterised datetime (C3)
rows = await self._storage.execute(
    """
    SELECT s.id, s.strength FROM synapses s
    JOIN atoms a1 ON a1.id = s.source_id
    JOIN atoms a2 ON a2.id = s.target_id
    WHERE s.strength > ?
      AND (s.last_activated_at IS NULL OR s.last_activated_at < datetime('now', ?))
      AND s.relationship NOT IN ('contradicts', 'supersedes', 'warns-against')
      AND a1.is_deleted = 0 AND a2.is_deleted = 0
      AND a1.last_accessed_at > datetime('now', ?)
      AND a2.last_accessed_at > datetime('now', ?)
    """,
    (prune_threshold,
     f"-{ltd_window_days} days",
     f"-{activity_window} days",
     f"-{activity_window} days"),
)

if rows and not result.dry_run:
    # H3: proportional LTD — `strength * (1 - ltd_fraction)`
    # Floor prevents immortal weak synapses: max(min_ltd_amount, strength * factor)
    # For batch, use multiplicative: strength * multiplier
    # Synapses that fall below prune_threshold get deleted
    multiplier = 1.0 - ltd_fraction  # e.g. 0.85 for 15% proportional LTD
    synapse_ids = [row["id"] for row in rows]
    id_ph = ",".join("?" * len(synapse_ids))

    # Batch UPDATE: proportional weakening
    await self._storage.execute_write(
        f"UPDATE synapses SET strength = MAX(?, strength * ?) WHERE id IN ({id_ph})",
        (min_ltd_floor, multiplier, *synapse_ids),
    )
    # Batch DELETE: prune those at or below threshold
    await self._storage.execute_write(
        f"DELETE FROM synapses WHERE id IN ({id_ph}) AND strength <= ?",
        (*synapse_ids, prune_threshold),
    )
```

Note: add `ltd_fraction` and `min_ltd_floor` to `ConsolidationConfig`.

### W2-C: Batch `_extract_pathways` and `_find_relevant_antipatterns`

- Both use `get_neighbors_batch` (W2-PRE)
- `_find_relevant_antipatterns`: use `get_batch_without_tracking` for candidate atom fetches
  (MUST be `without_tracking` — access count inflation would distort Hebbian/decay signals)

---

## Wave 3 — Independent Performance (can fully parallelise)

### W3-A: Batch reads in `learning.py` (Perf H-4)
- `auto_link`, `detect_antipattern_links`, `detect_supersedes`
- Batch `get_without_tracking` for all vector-search candidates
- **Batch reads only — keep sequential writes** (`_safe_create_synapse` loop stays sequential)

### W3-B: `atoms.get()` — `UPDATE...RETURNING` (Perf H-9)
```python
async def get(self, atom_id: int) -> Atom | None:
    rows = await self._storage.execute_write(
        "UPDATE atoms SET access_count = access_count + 1, "
        "last_accessed_at = datetime('now'), updated_at = datetime('now') "
        "WHERE id = ? AND is_deleted = 0 RETURNING *",
        (atom_id,),
    )
    return Atom.from_row(rows[0]) if rows else None
```

### W3-C: `_session_atoms` O(1) set (Perf M-7)
- Change `list` → `dict[int, None]` (preserves insertion order, O(1) lookup)
- Sort before JSON: `sorted(self._session_atoms.keys())`
- Note: moot if W4 (Brain singleton) proceeds and eliminates `_session_atoms`

### W3-D: LLM distillation in `_abstract_experiences` (Hebbian M3)
- Use `AsyncClient` (already non-blocking — do NOT use `anyio.to_thread.run_sync`)
- Fire concurrent calls: `asyncio.gather(*[_distill(c) for c in clusters], return_exceptions=True)`
- 15-second timeout per call
- Fallback to verbatim copy on exception/timeout
- Respect Ollama concurrency (semaphore shared with or separate from embeddings)

### W3-E: Remaining perf fixes
- **M-3**: `table_counts` — single UNION ALL query (6 → 1 round-trips)
- **M-4**: `synapses.get_stats` — combine into 2-3 queries (5 → 2)
- **M-6**: `cosine_similarity` — remove `async` keyword (pure CPU, no I/O)
- **M-8**: `Storage.close()` — track all thread connections in list, close all on shutdown
- **M-10**: `synapses.delete_for_atom` — DELETE first, use `changes()` for count

### W3-F: `get_hook_stats` memory fix (Perf H-8)
- Push JSON aggregation into SQL via `json_each()`
- Replace O(total_impressions) Python list with SQL COUNT/GROUP BY

---

## Wave 4 — Brain Singleton (highest risk, implement last)

### W4: Brain singleton for CLI hooks (Perf H-6)

**WARNING**: Must eliminate `_session_atoms` / `end_session()` Hebbian path BEFORE creating singleton.
Otherwise: double Hebbian firing (singleton `end_session()` + stop-hook `session_end_learning()`).

**Required steps:**
1. Remove `_session_atoms` list from Brain (vestigial — `hook_session_atoms` DB table is the real accumulator)
2. Remove Hebbian learning call from `end_session()` / `shutdown()`
3. Ensure stop-hook's `_run_session_stop_logic()` remains the sole trigger for `session_end_learning()`
4. Implement process-level singleton with `asyncio.Lock` guard
5. Guard `initialize()` (already idempotent — line 99)
6. Separate `_start_session` from `initialize` — must NOT create a new Brain session on every hook
7. Register `atexit` handler for `shutdown()` instead of calling per-hook

```python
_brain_instance: Brain | None = None
_brain_lock = asyncio.Lock()

async def _get_brain():
    global _brain_instance
    if _brain_instance is not None:
        return _brain_instance
    async with _brain_lock:
        if _brain_instance is None:
            brain = Brain()
            await brain.initialize()
            import atexit, asyncio
            atexit.register(lambda: asyncio.run(brain.shutdown()))
            _brain_instance = brain
    return _brain_instance
```

---

## Key Architectural Decisions (do not revisit without justification)

| Decision | Rationale |
|----------|-----------|
| Refractory set gates frontier expansion, NOT activation accumulation | Preserves superposition: convergent paths from multiple seeds should additively boost nodes |
| Fanout normalization uses pre-filter degree | Biological: normalization based on total synaptic connections, not surviving ones |
| BCM multiplicative as single-param SQL | `strength + ? * (1.0 - strength)` — SQLite evaluates per-row, single bind suffices |
| LTD proportional with floor | `MAX(min_floor, strength * multiplier)` prevents immortal weak synapses |
| LTD batch: UPDATE + DELETE (not `weaken()` loop) | 2 queries instead of 2N — correctness same, N+1 eliminated |
| Inhibitory-only pairs don't trigger new related-to | Atoms with only `contradicts` must stay contradicting, not acquire new generic links |
| `get_batch_without_tracking` in pathways/antipatterns | `get()` inflates access_count — phantom signals distort Hebbian and decay |
| Brain singleton requires eliminating `_session_atoms` | Without removal: double Hebbian firing on same atom set |
| LLM distillation uses AsyncClient directly | Already non-blocking HTTP; `anyio.to_thread` would be incorrect |
| Outbound degree cap in caller (auto_link) | 1 COUNT per auto_link call vs 1 COUNT per create() call (10-20× reduction) |
| Max 5 review-implement iterations | Agreed with user |

---

## Test Specification (write before implementing each wave)

### Wave 1 tests

**W1-A hebbian_update:**
- `test_strengthens_all_synapses_for_pair` — pair with related-to + caused-by: both get strengthened
- `test_does_not_strengthen_contradicts` — pair with contradicts: NOT strengthened
- `test_does_not_strengthen_supersedes` — pair with supersedes: NOT strengthened
- `test_does_not_strengthen_warns_against` — pair with warns-against: NOT strengthened
- `test_inhibitory_only_pair_no_new_related_to` — pair with only contradicts: no new related-to created
- `test_bcm_saturation` — high-strength synapse gets smaller increment than low-strength (BCM property)
- `test_bcm_formula_correct` — verify `new = old + increment * (1 - old)` numerically

**W1-B SQL hardening:**
- `test_feedback_ids_parameterised` — mock storage.execute_write, verify no f-string in SQL
- `test_like_patterns_parameterised` — same for LIKE clauses
- `test_datetime_parameterised` — same for datetime expressions

### Wave 2 tests

**W2-PRE get_neighbors_batch:**
- `test_batch_returns_same_as_sequential` — compare batch vs per-atom results for 5 atoms
- `test_batch_empty_list` — returns empty dict
- `test_batch_respects_min_strength` — filters weak synapses

**W2-A _spread_activation:**
- `test_refractory_prevents_re_expansion` — node activated at depth 1 does NOT appear in frontier at depth 2
- `test_superposition_preserved` — node reachable from 2 seeds gets additive activation (not just first path)
- `test_fanout_normalization_reduces_hub_dominance` — high-degree node's neighbors get less activation than low-degree
- `test_batch_neighbors_called_once_per_depth` — spy confirms single SQL per depth level

**W2-B _apply_ltd:**
- `test_proportional_ltd_weaker_synapse_loses_less` — 0.1 strength loses less than 0.5 strength (proportional)
- `test_proportional_ltd_floor` — very weak synapse does not get immortal (floor applies)
- `test_ltd_batch_not_per_row` — spy confirms 2 execute_write calls, not N

### Wave 3 tests

**W3-A batch learning.py:**
- `test_auto_link_batch_read` — spy confirms single get_batch_without_tracking call
- `test_auto_link_sequential_writes` — synapse creation calls are sequential (not parallel)

**W3-B atoms.get():**
- `test_get_single_query` — spy confirms one execute_write call, not two selects + update
- `test_get_returns_updated_access_count` — returned atom has access_count incremented

**W3-D LLM distillation:**
- `test_distillation_fallback_on_timeout` — mock Ollama timeout → falls back to verbatim copy
- `test_distillation_concurrent` — spy confirms asyncio.gather used, not sequential awaits

### Wave 4 tests

**W4 Brain singleton:**
- `test_singleton_same_instance` — two calls to `_get_brain()` return same object
- `test_no_double_hebbian_firing` — stop hook fires `session_end_learning` once, not twice
- `test_initialize_idempotent` — calling `initialize()` twice does not double-migrate

---

## Iteration Tracking

| Iteration | Status | Notes |
|-----------|--------|-------|
| 1 | IN PROGRESS | Waves 1-4 planned above |
| 2 | PENDING | Round-2 architect review (architects use recall-first prompts) |
| 3 | PENDING | |
| 4 | PENDING | |
| 5 | PENDING | Final — ship after this regardless |
