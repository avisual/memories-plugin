# Memory Consolidation

Consolidation is periodic maintenance that keeps the memory graph healthy — like sleep for the brain.

## What consolidation does

Run via the `reflect` MCP tool or automatically during migration/reatomise:

### 1. Merge near-duplicates

Atoms with cosine similarity ≥ `MEMORIES_DEDUP_THRESHOLD` (default 0.92) are merged. The higher-confidence atom survives; the other is soft-deleted. Synapses from both are merged onto the survivor.

This handles the case where similar knowledge was stored twice, slightly differently worded.

### 2. Decay weak synapses

Synapses that haven't been activated recently have their `strength` reduced. The decay rate is controlled by `MEMORIES_CONSOLIDATION__DECAY_RATE`. Synapses that decay below `MEMORIES_CONSOLIDATION__PRUNE_THRESHOLD` (default 0.05) are pruned entirely.

### 3. Prune orphan connections

Synapses pointing to deleted atoms are removed.

### 4. Promote frequently accessed atoms

Atoms with high `access_count` and `activated_count` have their `confidence` score boosted, making them more prominent in future recalls.

## When to run consolidation

| Situation | Action |
|-----------|--------|
| After a major project completes | `reflect()` in Claude Code |
| After `backfill` or `migrate` | Runs automatically |
| After `reatomise` | Runs automatically |
| Weekly maintenance | `reflect()` |
| Graph feels noisy/irrelevant | `reflect(dry_run=True)` first to preview |

## Using the reflect tool

```python
# Full consolidation
reflect()

# Preview without changes
reflect(dry_run=True)

# Limit to one project
reflect(scope="project:myapp")
```

## CLI maintenance commands

```bash
# Merge near-duplicate atoms + split blobs + relink
uv run python -m memories reatomise --verbose

# Re-run auto_link for all atoms (fill missing synapses)
uv run python -m memories relink --verbose

# Rename fragmented regions to canonical names
uv run python -m memories normalise --verbose
```

## What consolidation does NOT do

- It does not permanently delete atoms with `hard=True` — use `forget(atom_id=X, hard=True)` for that
- It does not modify atom content — use `amend` for that
- It does not change importance scores — those are set manually via `remember` or `amend`

## Relationship to Hebbian learning

Consolidation and Hebbian learning are complementary:

- **Hebbian learning** (runs at session end): *strengthens* connections between co-activated atoms
- **Consolidation** (runs periodically): *removes* connections that were never activated and *merges* duplicates

Together they maintain a graph that grows richer with use and stays clean over time.
