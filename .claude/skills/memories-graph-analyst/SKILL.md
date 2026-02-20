---
name: memories-graph-analyst
description: Graph topology analyst for the memories project. Queries the live SQLite database to measure synapse graph health — degree distributions, hub detection, strength histograms, cluster coefficients, and Hebbian learning effectiveness. Use after implementing Hebbian changes to verify graph is healthy, not degenerate.
disable-model-invocation: false
user-invocable: true
---

# Memories Graph Analyst

You are a graph topology specialist for the `memories` project in the local memories repository. You query the live SQLite database (`~/.memories/memories.db`) to measure the health of the synapse graph empirically — not by reading code, but by inspecting actual data.

## Step 1: Recall prior graph health snapshots

Call `mcp__memories__recall` with:
- "memories graph topology health"
- "memories synapse degree distribution"
- "memories hebbian graph analysis"

If prior snapshots exist, this run is a comparison.

## Step 2: Run graph health queries

Connect to the live database and run the following analyses. Use the Bash tool with `sqlite3 ~/.memories/memories.db`.

### 2a. Basic graph statistics
```sql
SELECT
    COUNT(*) as total_synapses,
    ROUND(AVG(strength), 4) as avg_strength,
    ROUND(MIN(strength), 4) as min_strength,
    ROUND(MAX(strength), 4) as max_strength,
    COUNT(DISTINCT relationship) as relationship_types
FROM synapses;

SELECT relationship, COUNT(*) as cnt, ROUND(AVG(strength),3) as avg_str
FROM synapses
GROUP BY relationship
ORDER BY cnt DESC;
```

### 2b. Strength distribution histogram (key Hebbian health indicator)
```sql
-- Should show a healthy bell curve, NOT a binary pile at 0.05 and 1.0
SELECT
    CASE
        WHEN strength < 0.1 THEN '0.0-0.1'
        WHEN strength < 0.2 THEN '0.1-0.2'
        WHEN strength < 0.3 THEN '0.2-0.3'
        WHEN strength < 0.4 THEN '0.3-0.4'
        WHEN strength < 0.5 THEN '0.4-0.5'
        WHEN strength < 0.6 THEN '0.5-0.6'
        WHEN strength < 0.7 THEN '0.6-0.7'
        WHEN strength < 0.8 THEN '0.7-0.8'
        WHEN strength < 0.9 THEN '0.8-0.9'
        ELSE '0.9-1.0'
    END as bucket,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM synapses), 1) as pct
FROM synapses
GROUP BY bucket
ORDER BY bucket;
```

### 2c. Hub detection — inbound degree (related-to only)
```sql
-- Inbound cap is 50. Healthy graph: few hubs near 50, most nodes <10
SELECT
    target_id,
    COUNT(*) as inbound_degree,
    (SELECT content FROM atoms WHERE id = target_id LIMIT 1) as atom_preview
FROM synapses
WHERE relationship = 'related-to'
GROUP BY target_id
HAVING inbound_degree > 10
ORDER BY inbound_degree DESC
LIMIT 20;

-- Distribution summary
SELECT
    CASE
        WHEN inbound_degree <= 5 THEN '1-5'
        WHEN inbound_degree <= 10 THEN '6-10'
        WHEN inbound_degree <= 20 THEN '11-20'
        WHEN inbound_degree <= 50 THEN '21-50'
        ELSE '>50 (BUG)'
    END as degree_bucket,
    COUNT(*) as node_count
FROM (
    SELECT target_id, COUNT(*) as inbound_degree
    FROM synapses WHERE relationship = 'related-to'
    GROUP BY target_id
) sub
GROUP BY degree_bucket
ORDER BY degree_bucket;
```

### 2d. Outbound degree distribution
```sql
SELECT
    CASE
        WHEN outbound <= 5 THEN '1-5'
        WHEN outbound <= 10 THEN '6-10'
        WHEN outbound <= 20 THEN '11-20'
        WHEN outbound <= 50 THEN '21-50'
        ELSE '>50'
    END as degree_bucket,
    COUNT(*) as node_count
FROM (
    SELECT source_id, COUNT(*) as outbound
    FROM synapses WHERE relationship = 'related-to'
    GROUP BY source_id
) sub
GROUP BY degree_bucket
ORDER BY degree_bucket;
```

### 2e. Inhibitory vs excitatory balance
```sql
-- Healthy: contradicts/supersedes/warns-against are rare but present
SELECT
    relationship,
    COUNT(*) as count,
    ROUND(AVG(strength), 3) as avg_strength,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM synapses), 2) as pct_of_total
FROM synapses
GROUP BY relationship
ORDER BY count DESC;
```

### 2f. Hebbian learning activity (recently strengthened)
```sql
-- How active is Hebbian reinforcement? Should see synapses activated recently
SELECT
    CASE
        WHEN last_activated_at IS NULL THEN 'never'
        WHEN last_activated_at > datetime('now', '-1 day') THEN 'today'
        WHEN last_activated_at > datetime('now', '-7 days') THEN 'this week'
        WHEN last_activated_at > datetime('now', '-30 days') THEN 'this month'
        ELSE 'older'
    END as activity,
    COUNT(*) as count,
    ROUND(AVG(strength), 3) as avg_strength
FROM synapses
GROUP BY activity
ORDER BY count DESC;
```

### 2g. Atom connectivity (isolated atoms are a problem)
```sql
-- Atoms with no synapses at all — orphans
SELECT COUNT(*) as orphan_atoms
FROM atoms a
WHERE is_deleted = 0
AND NOT EXISTS (
    SELECT 1 FROM synapses s
    WHERE s.source_id = a.id OR s.target_id = a.id
);

-- Connectivity distribution
SELECT
    CASE
        WHEN degree = 0 THEN 'isolated'
        WHEN degree <= 2 THEN '1-2 connections'
        WHEN degree <= 5 THEN '3-5 connections'
        WHEN degree <= 10 THEN '6-10 connections'
        ELSE '>10 connections'
    END as connectivity,
    COUNT(*) as atom_count
FROM (
    SELECT a.id, COUNT(s.id) as degree
    FROM atoms a
    LEFT JOIN synapses s ON s.source_id = a.id OR s.target_id = a.id
    WHERE a.is_deleted = 0
    GROUP BY a.id
) sub
GROUP BY connectivity;
```

## Step 3: Health assessment

Evaluate each metric against healthy targets:

| Metric | Healthy | Warning | Problem |
|--------|---------|---------|---------|
| Strength distribution | Bell curve centred 0.3-0.6 | Heavy tails | >30% at extremes (0.0-0.1 or 0.9-1.0) |
| Max inbound degree | <40 | 40-50 | Any >50 (cap broken) |
| Inhibitory ratio | 2-8% of edges | <1% or >15% | >20% |
| Orphan atoms | <5% | 5-15% | >15% |
| Hebbian activity (this week) | >10% of synapses | <5% | <1% |
| related-to % of all synapses | 60-80% | >90% | >95% (no semantic diversity) |

## Step 4: Store snapshot in memory

```
mcp__memories__remember:
  content: "Graph health YYYY-MM-DD: N synapses, avg_strength=X, strength histogram=[brief], max_inbound=Y, orphan_pct=Z%, hebbian_active_week=K%. [healthy/warning/problem] — [key finding]"
  type: "fact"
  region: "project:memories"
  tags: ["graph-health", "topology", "benchmark"]
  importance: 0.75
  source_project: "memories"
```

## Step 5: Flag degenerate states

Immediately flag if:
- Any inbound degree >50 (cap broken)
- Strength distribution is bimodal (>30% below 0.1, >30% above 0.9) — BCM not working
- >15% orphan atoms — Hebbian or auto_link broken
- Inhibitory synapses >20% — contradiction detection misfiring
- Zero synapses with `last_activated_at` in past week — Hebbian learning dead
