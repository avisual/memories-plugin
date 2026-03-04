---
name: memories-health
description: Health check for the memories system. Verifies hooks are configured, counts memories by type/region, shows recent activity, and diagnoses common problems (empty DB, hooks not firing, consolidation stale). Run this if recall feels broken or stale.
disable-model-invocation: false
user-invocable: true
---

# Memories Health Check

You are a diagnostic agent for the memories system. Your job is to quickly assess whether the system is working correctly and surface any problems to the user.

## Step 1: Check database accessibility

```bash
sqlite3 ~/.memories/memories.db "SELECT COUNT(*) FROM atoms WHERE is_deleted = 0;"
```

If this fails or returns 0, the database is missing or empty — report immediately.

## Step 2: Memory inventory

```sql
-- Count by type
SELECT type, COUNT(*) as count
FROM atoms WHERE is_deleted = 0
GROUP BY type ORDER BY count DESC;

-- Count by region (project awareness)
SELECT
    COALESCE(region, 'unset') as region,
    COUNT(*) as count
FROM atoms WHERE is_deleted = 0
GROUP BY region ORDER BY count DESC LIMIT 10;

-- Recent activity
SELECT
    DATE(created_at) as day,
    COUNT(*) as new_memories
FROM atoms WHERE is_deleted = 0
  AND created_at > datetime('now', '-7 days')
GROUP BY day ORDER BY day DESC;
```

Run via: `sqlite3 ~/.memories/memories.db "<sql>"`

## Step 3: Hook configuration check

```bash
# Check hooks are wired up in Claude Code settings
cat ~/.claude/settings.json 2>/dev/null | python3 -c "
import json, sys
s = json.load(sys.stdin)
hooks = s.get('hooks', {})
for event, cmds in hooks.items():
    print(f'{event}: {len(cmds)} hook(s)')
" 2>/dev/null || echo "No settings.json found"
```

Healthy output should show at least: `UserPromptSubmit`, `Stop` (or `PostToolUse`) hooks present.

## Step 4: Recent consolidation check

```sql
-- When did consolidation last run?
SELECT
    MAX(updated_at) as last_consolidation,
    COUNT(*) as atoms_consolidated
FROM atoms
WHERE is_deleted = 0
  AND updated_at > created_at;

-- Feedback backlog (high = consolidation not processing)
SELECT COUNT(*) as unprocessed_feedback
FROM atom_feedback
WHERE processed_at IS NULL;
```

Healthy: `last_consolidation` within past 24h, `unprocessed_feedback` < 100.

## Step 5: Synapse graph quick check

```sql
SELECT
    COUNT(*) as total_synapses,
    ROUND(AVG(strength), 3) as avg_strength,
    COUNT(DISTINCT relationship) as edge_types
FROM synapses;
```

Healthy: synapses > 0, avg_strength between 0.2–0.8, edge_types >= 3.

## Step 6: MCP server reachability

Call `mcp__memories__recall` with query "test health check".

- If it returns atoms → MCP server is live and connected
- If it errors → MCP server is not running (run `uv run python -m memories`)
- If it returns 0 atoms and DB has memories → recall pipeline issue

## Step 7: Report

Summarise findings as:

```
## Memory System Health Report

DB: X atoms (Y types, Z regions)
Hooks: [configured / missing — which ones]
Consolidation: last ran N hours ago, M unprocessed items
Synapse graph: P synapses, avg strength Q
MCP: [live / unreachable]

Status: [HEALTHY / WARNING: <reason> / PROBLEM: <reason>]
```

Flag any of these immediately:
- 0 atoms in a non-fresh install
- `UserPromptSubmit` hook missing (memories won't be retrieved)
- `Stop` hook missing (memories won't be stored after sessions)
- Consolidation last ran > 48h ago
- Unprocessed feedback > 500 (consolidation stuck)
- MCP unreachable
