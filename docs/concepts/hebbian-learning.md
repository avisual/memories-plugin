# Hebbian Learning

> "Neurons that fire together wire together." — Donald Hebb, 1949

Hebbian learning is how memories automatically strengthens connections between atoms that appear together in the same session.

## The core idea

When two memories are both recalled in the same Claude Code session, that co-activation is a signal that they're related in practice — even if the automated auto-link step didn't connect them strongly at creation time. The more often two atoms are recalled together, the stronger their connection becomes.

## How it works in memories

### During a session

Every hook (prompt-submit, post-tool, pre-tool) records the IDs of atoms it surfaces into the `hook_session_atoms` table, keyed by the Claude Code `session_id`.

### At session end (`Stop` hook)

The stop hook reads all atom IDs accumulated during the session and calls `session_end_learning`:

1. For every pair of atoms that co-occurred, the synapse between them (if it exists) has its `strength` increased by `MEMORIES_LEARNING__HEBBIAN_INCREMENT` (default 0.05), capped at 1.0.

2. If no synapse exists between two frequently co-occurring atoms, a new one is created with the `related-to` relationship.

3. The `activated_count` counter on each synapse increments, tracking lifetime co-activations.

### Mid-session checkpoint (`PreCompact` hook)

If context compaction fires before the session ends, Hebbian learning runs as a checkpoint. The session atoms accumulated so far are processed without deleting the table — the Stop hook will still run its own pass at the end.

## Auto-linking (creation-time)

When a new atom is created via `remember`, `auto_link` immediately searches for existing atoms with cosine similarity ≥ `MEMORIES_LEARNING__AUTO_LINK_THRESHOLD` (default 0.6) and creates `related-to` synapses to the top matches.

This seeds the graph at creation time. Hebbian learning then refines and strengthens connections based on actual usage patterns over time.

## Effect over time

After many sessions working on the same project:

- Atoms that are always recalled together (e.g., "Redis expiry" and "cache warming") build strong bidirectional synapses
- Atoms that co-occur infrequently maintain weaker connections
- Connections that were never activated decay during consolidation (see [Memory Consolidation](consolidation.md))

The result is a graph that reflects *your actual working patterns*, not just semantic similarity at write time.

## Sub-agent learning

When Claude Code spawns a sub-agent via the `Task` tool, the sub-agent accumulates its own set of session atoms. At sub-agent stop, those atoms are:

1. Processed with Hebbian learning for the sub-agent session
2. Propagated to the parent session's `hook_session_atoms` table

The parent's stop hook then runs a unified Hebbian pass over both sets together, linking atoms from the main session and sub-agent sessions into a single coherent co-activation graph.

## Tuning

```bash
# Stronger Hebbian increment per co-activation
export MEMORIES_LEARNING__HEBBIAN_INCREMENT=0.1

# Lower auto-link threshold (more connections at creation time)
export MEMORIES_LEARNING__AUTO_LINK_THRESHOLD=0.5
```
