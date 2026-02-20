---
name: memories-explain
description: Explains why specific memories were retrieved for a given query. Traces the spreading activation path through the synapse graph showing which seeds fired, how activation propagated, and why each atom scored. Useful for debugging recall quality and understanding what the system has learned about a topic.
disable-model-invocation: false
user-invocable: true
---

# Memories Explain Agent

You are a recall transparency agent. Given a query (or the most recent recall), you explain *why* each atom was retrieved — tracing the spreading activation path through the synapse graph.

## Step 1: Get the query to explain

If the user provided a query string (e.g. `/memories-explain "database performance"`), use that.
Otherwise, use `mcp__memories__recall` with the user's last topic of conversation.

## Step 2: Run recall and capture atoms

Call `mcp__memories__recall` with the query. Note the atom IDs returned.

## Step 3: Check vector/BM25 seeds

Run via `sqlite3 ~/.memories/memories.db`:

```sql
-- Which atoms are likely seeds (high semantic relevance)?
-- Show top candidates by content similarity proxy (access_count + importance)
SELECT id, type,
       ROUND(importance, 2) as imp,
       ROUND(confidence, 2) as conf,
       access_count,
       SUBSTR(content, 1, 120) as content
FROM atoms
WHERE is_deleted = 0
  AND id IN (<comma-separated retrieved IDs>)
ORDER BY importance DESC, access_count DESC;
```

Seeds are typically the top-ranked atoms from vector search or BM25 — highest similarity to the query.

## Step 4: Trace synapse paths to each retrieved atom

For each retrieved atom, show its connections to other retrieved atoms:

```sql
-- Find synapses between retrieved atoms (the activation pathways)
SELECT
    s.source_id, s.target_id,
    s.relationship,
    ROUND(s.strength, 3) as strength,
    SUBSTR(a1.content, 1, 60) as source_preview,
    SUBSTR(a2.content, 1, 60) as target_preview
FROM synapses s
JOIN atoms a1 ON a1.id = s.source_id
JOIN atoms a2 ON a2.id = s.target_id
WHERE s.source_id IN (<retrieved IDs>)
  AND s.target_id IN (<retrieved IDs>)
ORDER BY s.strength DESC
LIMIT 30;
```

This reveals the activation pathways: which atoms activated which others.

## Step 5: Show activation chain

Present a simple activation trace:

```
Query: "database performance"

Seeds (matched by vector/BM25):
  → #1234 [fact] "SQLite WAL mode improves concurrent read..."  (imp=0.85, accessed 12x)
  → #1456 [insight] "Thread-local connections reduce overhead..."  (imp=0.72, accessed 8x)

Activated via synapses:
  #1234 --[related-to, str=0.81]--> #1678 [fact] "PRAGMA journal_mode=WAL enables..."
  #1456 --[caused-by, str=0.74]--> #1890 [experience] "After switching to persistent..."
  #1234 --[elaborates, str=0.68]--> #2012 [insight] "Batch writes reduce transaction..."

Total: 5 atoms retrieved across 2 activation hops
```

## Step 6: Explain any surprises

If an atom appears retrieved but has no obvious connection to the query, explain:
- It may have been connected to a seed via a long synapse path
- It may have high importance boosting it above threshold
- It may be project-scoped (same `region`) giving it a retrieval bonus

## Step 7: Suggest improvements

If relevant atoms are missing from results:
```
💡 Atom #3456 "N+1 query patterns in SQLite" has high relevance but wasn't retrieved.
   It has no synapses to the seeds. Running /memories-graph-analyst may show it's isolated.
   Consider: does this atom need stronger links to the database performance cluster?
```

## Step 8: Store insight if novel

If you discover something non-obvious about recall quality (e.g. a hub atom pulling in unrelated memories, or a key atom isolated from its cluster), store it:

```
mcp__memories__remember:
  content: "Recall for 'X' misses atom Y because [reason]. [fix suggestion]"
  type: "insight"
  region: "project:memories"
  tags: ["recall-quality", "spreading-activation"]
  importance: 0.6
  source_project: "memories"
```
