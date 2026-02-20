# Spreading Activation

Spreading activation is how memories retrieves related context beyond simple vector search.

## The problem with pure vector search

Vector similarity finds atoms whose content *literally resembles* your query. But the most useful context is often related *indirectly* — knowledge that connects to what you're asking through a chain of concepts, even if individual atoms don't score highly in isolation.

## How spreading activation works

### Step 1: Seed selection

The query is embedded via Ollama (`nomic-embed-text`). The top-K most similar atoms are found via ANN search in the `atoms_vec` vector table. These become **seeds** — the starting points for activation.

BM25 keyword search (FTS5) runs in parallel and adds any high-scoring atoms not already in the seed set.

### Step 2: Activation propagation

Each seed atom has an activation value proportional to its similarity score. Activation spreads outward through the synapse graph:

```
seed atom (activation=0.9)
  → related atom via "elaborates" synapse (strength=0.7)
      → activation = 0.9 × 0.7 × decay_factor = 0.63
      → related atom via "caused-by" synapse (strength=0.5)
          → activation = 0.63 × 0.5 × decay_factor = 0.315
```

Activation decays with each hop (controlled by `MEMORIES_RETRIEVAL__SPREAD_DEPTH`, default 2). Stronger synapses carry more activation. The same atom can receive activation from multiple paths — values are summed.

### Step 3: Multi-signal ranking

Every activated atom gets a composite score:

| Signal | Weight | Description |
|--------|--------|-------------|
| Vector similarity | 40% | How closely the content matches the query embedding |
| Spreading activation | 25% | How much activation reached this atom via the graph |
| Recency | varies | Boost for recently accessed atoms |
| Confidence | varies | Atom's confidence score |
| Importance | varies | Manually set importance weight |
| BM25 | varies | Keyword match score from FTS5 |
| Newness | varies | Slight boost for recently created atoms |

### Step 4: Budget fitting

Results are sorted by composite score. Atoms are packed into the token budget (`budget_tokens`) from highest score down. When the budget is nearly full, atoms are progressively compressed (content truncated) rather than dropped.

A region diversity cap (`MEMORIES_REGION_DIVERSITY_CAP`, default 5) prevents any single project from dominating the results.

## Example

Query: `"Redis cache expiry"`

**Seeds found by vector search:**
- "Redis expiry is lazy + periodic active sweep" (similarity 0.88)
- "Use SCAN not KEYS for Redis iteration" (similarity 0.72)

**Activation spreads to connected atoms:**
- "Redis cluster hash slots" (connected via `related-to` to the SCAN atom)
- "Cache warming strategy for high-traffic endpoints" (connected via `caused-by` to the expiry atom)

**Result:** All four atoms surface, even though the latter two don't directly match "cache expiry" — they're contextually adjacent via the neural graph.

## Why this matters

In practice, spreading activation surfaces:
- **Antipatterns** connected to the relevant domain via `warns-against` synapses
- **Architectural decisions** linked to the technology you're asking about
- **Past mistakes** that share context with the current task

This is the difference between "find what matches" and "find what's relevant."

## Tuning

```bash
# Deeper activation (more context, slower)
export MEMORIES_RETRIEVAL__SPREAD_DEPTH=3

# Restrict to direct matches only
recall(query="...", depth=1)

# More atoms per project
export MEMORIES_REGION_DIVERSITY_CAP=10
```
