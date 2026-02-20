# MCP Tools Reference

avisual memories exposes 13 MCP tools that AI agents call directly via the MCP protocol.

> **Tool names have no `memory_` prefix.** Call them as `remember`, `recall`, `connect`, etc.

---

## remember

Store a new memory atom. Auto-creates synaptic connections to related existing memories.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `content` | string | Yes | The knowledge to store. Keep atomic: one concept, ~50–200 tokens. |
| `type` | string | Yes | Atom type — see types below |
| `region` | string | No | Logical grouping e.g. `project:myapp`, `technical`. Auto-inferred if empty. |
| `tags` | list[string] | No | Keyword tags for filtering |
| `severity` | string | No | Antipatterns only: `low`, `medium`, `high`, `critical` |
| `instead` | string | No | Antipatterns only: the recommended alternative |
| `source_project` | string | No | Project this memory comes from |
| `source_file` | string | No | File this memory relates to |
| `importance` | float | No | Priority 0–1 (default 0.5). Use 0.7–1.0 for critical knowledge. |

**Atom types:** `fact`, `experience`, `skill`, `preference`, `insight`, `antipattern`

**Returns:**
```json
{
  "atom_id": 42,
  "atom": { "id": 42, "content": "...", "type": "preference", ... },
  "synapses_created": 3,
  "related_atoms": [...]
}
```

**Examples:**
```python
# Store a preference
remember(
    content="Use pytest over unittest for Python testing",
    type="preference"
)

# Store an antipattern with alternatives
remember(
    content="Using rm for file deletion is unrecoverable",
    type="antipattern",
    severity="high",
    instead="Use trash() to move files to .Trash/ so they are recoverable",
    importance=0.8
)
```

---

## recall

Search memories using semantic similarity with spreading activation.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | Yes | Natural language search query |
| `budget_tokens` | int | No | Max tokens to return (default 2000) |
| `depth` | int | No | Spreading activation hops from seed atoms (default 2) |
| `region` | string | No | Restrict to a specific region. Empty = all regions. |
| `types` | list[string] | No | Restrict to specific atom types. Empty = all types. |
| `include_antipatterns` | bool | No | Surface antipattern warnings (default true) |

**Returns:**
```json
{
  "atoms": [
    {
      "id": 42,
      "content": "Use pytest over unittest",
      "type": "preference",
      "score": 0.91,
      "score_breakdown": {
        "vector_similarity": 0.87,
        "spread_activation": 0.72,
        "recency": 0.10,
        "bm25": 0.05
      }
    }
  ],
  "antipatterns": [...],
  "budget_used": 840,
  "budget_remaining": 1160,
  "seed_count": 5,
  "total_activated": 23,
  "compression_level": 0
}
```

**Example:**
```python
recall(query="Python testing frameworks", budget_tokens=3000)
```

---

## connect

Create or strengthen a synaptic connection between two atoms.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `source_id` | int | Yes | Source atom ID |
| `target_id` | int | Yes | Target atom ID |
| `relationship` | string | Yes | Relationship type — see below |
| `strength` | float | No | Connection weight 0–1 (default 0.5) |

**Relationship types:** `related-to`, `caused-by`, `part-of`, `contradicts`, `supersedes`, `elaborates`, `warns-against`

**Returns:**
```json
{
  "synapse_id": 1234,
  "synapse": { "source_id": 42, "target_id": 88, "relationship": "elaborates", "strength": 0.7 },
  "source_summary": "Use pytest...",
  "target_summary": "Python testing best practices..."
}
```

**Example:**
```python
connect(source_id=42, target_id=88, relationship="elaborates", strength=0.7)
```

---

## forget

Soft-delete or hard-delete a memory atom.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `atom_id` | int | Yes | The atom to delete |
| `hard` | bool | No | If false (default): soft-delete (recoverable). If true: permanent. |

**Returns:**
```json
{
  "status": "deleted",
  "atom_id": 42,
  "hard": false,
  "synapses_affected": 5
}
```

**Example:**
```python
forget(atom_id=42)          # soft-delete (recoverable via DB)
forget(atom_id=42, hard=True)  # permanent
```

---

## amend

Update an existing atom's content, type, tags, or confidence. Re-embeds and re-links if content changes.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `atom_id` | int | Yes | Atom to update |
| `content` | string | No | New content. Empty = keep current. |
| `type` | string | No | New type. Empty = keep current. |
| `tags` | list[string] | No | New tags. Null = keep current. Empty list = clear all. |
| `confidence` | float | No | New confidence 0–1. Pass -1 (default) to keep current. |

**Returns:**
```json
{
  "atom": { "id": 42, "content": "...", ... },
  "new_synapses": 2,
  "removed_synapses": 1
}
```

**Example:**
```python
amend(atom_id=42, content="Use pytest AND coverage.py for testing", confidence=0.9)
```

---

## reflect

Trigger memory consolidation — like sleep for the brain.

Decays unused connections, prunes weak synapses, merges near-duplicate atoms, and promotes frequently accessed memories.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `scope` | string | No | `"all"` (default) or a region name to limit consolidation |
| `dry_run` | bool | No | If true, preview without applying changes |

**Returns:**
```json
{
  "merged": 3,
  "decayed": 12,
  "pruned": 45,
  "promoted": 8,
  "compressed": 2,
  "dry_run": false,
  "details": [...]
}
```

**Example:**
```python
reflect()                        # full consolidation
reflect(dry_run=True)           # preview only
reflect(scope="project:myapp")  # limit to one project
```

---

## status

Get memory system health and statistics.

**Parameters:** None

**Returns:**
```json
{
  "total_atoms": 5760,
  "total_synapses": 84566,
  "regions": [{"name": "project:myapp", "count": 142}, ...],
  "avg_confidence": 0.78,
  "stale_atoms": 23,
  "orphan_atoms": 4,
  "db_size_mb": 69.89,
  "embedding_model": "nomic-embed-text",
  "current_session_id": "...",
  "ollama_healthy": true
}
```

**Example:**
```python
status()
```

---

## pathway

Visualize the connection graph radiating from a specific atom.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `atom_id` | int | Yes | Starting atom to explore from |
| `depth` | int | No | Max hops from the starting atom (default 2) |
| `min_strength` | float | No | Minimum synapse strength to follow (default 0.1) |

**Returns:**
```json
{
  "nodes": [{ "id": 42, "content": "...", "type": "fact", "region": "technical" }],
  "edges": [{ "source": 42, "target": 88, "relationship": "elaborates", "strength": 0.7 }],
  "clusters": { "technical": [42, 88], "project:myapp": [103] }
}
```

**Example:**
```python
pathway(atom_id=42, depth=3)
```

---

## stats

Get hook invocation statistics and memory effectiveness metrics.

**Parameters:** None

**Returns:** Hook counts by period, avg atoms returned, relevance scores, budget utilisation, top atoms, novelty filter pass rates, latency by hook type.

**Example:**
```python
stats()
```

---

## create_task

Create a task atom with lifecycle tracking (pending → active → done → archived).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `content` | string | Yes | Task description |
| `region` | string | No | Logical grouping (defaults to `tasks`) |
| `tags` | list[string] | No | Tags for categorization |
| `importance` | float | No | Priority 0–1 (default 0.5) |
| `status` | string | No | Initial status: `pending` (default), `active`, `done`, `archived` |

**Returns:** Same as `remember` — atom_id, atom, synapses_created, related_atoms.

---

## update_task

Update task status and optionally flag linked memories as stale.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task_id` | int | Yes | Task atom ID |
| `status` | string | Yes | New status: `pending`, `active`, `done`, `archived` |
| `flag_linked_memories` | bool | No | When true (default) and status is `done`/`archived`, reduces confidence of linked memories by 0.1 |

**Returns:**
```json
{
  "task": { ... },
  "linked_memories_flagged": 5
}
```

---

## list_tasks

List task atoms with optional filters.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `status` | string | No | Filter by status. Empty = all. |
| `region` | string | No | Filter by region. Empty = all. |

**Returns:**
```json
{
  "tasks": [...],
  "count": 12
}
```

---

## stale_memories

Find memories linked to completed tasks that may be outdated.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `min_completed_tasks` | int | No | Minimum number of completed tasks linked for a memory to be considered stale (default 1) |

**Returns:** List of atoms with reduced confidence due to linked completed tasks.

---

## Error format

All tools return errors as:
```json
{
  "error": "ValueError",
  "detail": "Atom 999 not found",
  "traceback": "ValueError: Atom 999 not found"
}
```
