# Best Practices

## Storing memories

### Keep atoms atomic
One concept per atom, ~50–200 tokens. The neural graph forms richer connections between small atoms than between large blobs.

```python
# Bad — too many concepts in one atom
remember(
    content="Redis uses SCAN instead of KEYS for iteration. KEYS blocks. Also Redis cluster uses hash slots. Expiry is lazy + active.",
    type="fact"
)

# Good — one concept each
remember(content="Use SCAN for Redis key iteration — KEYS blocks the server", type="antipattern", instead="SCAN", severity="high")
remember(content="Redis cluster distributes keys across 16384 hash slots", type="fact")
remember(content="Redis expiry is lazy (on access) plus periodic active sweep", type="fact")
```

### Use antipatterns liberally

Antipatterns are the highest-value memory type — they prevent repeated mistakes. When something goes wrong, store it immediately:

```python
remember(
    content="FastAPI background tasks don't work with SQLite connections — must use connection pool",
    type="antipattern",
    severity="high",
    instead="Use a connection pool (e.g. aiosqlite with pool_size=5)",
    importance=0.8
)
```

### Set importance correctly

| Importance | Use for |
|-----------|---------|
| 0.8–1.0 | Antipatterns, security issues, data loss risks, critical decisions |
| 0.6–0.7 | Key architectural decisions, frequently used skills |
| 0.4–0.5 | General facts, observations (default) |
| 0.2–0.3 | Low-signal notes, temporary context |

### Tag for retrieval

Tags enable filtering and improve BM25 keyword search:

```python
remember(
    content="Redshift SORTKEY must match the most common filter column",
    type="fact",
    tags=["redshift", "performance", "sql"],
    source_project="data-warehouse"
)
```

## Recall

### Be specific
```python
# Weak — too broad
recall("Redis")

# Strong — specific context
recall("Redis key expiry and eviction policies for cache warming")
```

### Use region filtering for project work
```python
# Only surface memories from this project
recall("authentication flow", region="project:myapp")
```

### Increase depth for exploratory queries
```python
# depth=3 follows 3 hops of spreading activation — surfaces related context
recall("database migrations", depth=3)
```

## Graph maintenance

### Run consolidation periodically

After large sessions or heavy development, clean up the graph:

```python
# In Claude Code — use the MCP tool
reflect()

# Check what it would do first
reflect(dry_run=True)
```

### Keep the graph connected

After importing atoms (e.g. `backfill` or `migrate`):

```bash
uv run python -m memories relink --verbose
```

### Fix region fragmentation

If atoms end up in generic regions (`general`, `project:git`), normalise them:

```bash
uv run python -m memories normalise --verbose
```

### Split blob atoms

If the backfill imported long multi-concept observations, split them:

```bash
uv run python -m memories reatomise --verbose
```

## Workflow patterns

### Project onboarding (new machine or new dev)

```bash
# 1. Install and set up
uv run python -m memories setup

# 2. Backfill from existing Claude transcripts
uv run python -m memories backfill

# 3. Migrate from claude-mem if applicable
uv run python -m memories migrate --source ~/.claude-mem/claude-mem.db

# 4. Relink the graph
uv run python -m memories relink

# 5. Verify
uv run python -m memories diagnose
uv run python -m memories stats
```

### End of a major feature

```bash
# 1. Let the Stop hook run Hebbian learning (happens automatically)
# 2. Run consolidation to merge any duplicates
uv run python -m memories eval "feature you just built" --verbose
# Review what's in memory, then:
reflect()  # via MCP tool in Claude Code
```

### Debugging a recurring problem

When you hit the same issue twice, it should be an antipattern:

```python
# In Claude Code:
remember(
    content="[Specific description of the problem and what triggers it]",
    type="antipattern",
    severity="high",
    instead="[What to do instead]",
    importance=0.85,
    source_project="myapp"
)
```

## What NOT to store

- Don't store ephemeral state ("current PR is #234") — it becomes stale immediately
- Don't store secrets, credentials, or PII
- Don't store things Claude can look up from code (file paths, function signatures) — store *patterns* and *lessons*, not reference material
- Don't store observations shorter than ~30 characters — too little signal for embeddings
