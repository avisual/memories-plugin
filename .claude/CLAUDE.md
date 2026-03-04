# Memories Project

Brain-like memory system for Claude Code using neural graph architecture with spreading activation.

## Development

- Run tests: `uv run pytest`
- Run MCP server: `uv run python -m memories`
- Health check: `uv run python -m memories health`
- All async code uses `asyncio` — tests use `pytest-asyncio` with `asyncio_mode = "auto"`

## Skills

Seven Claude Code skills are included in `.claude/skills/`. Invoke with `/skill-name`.

### User-facing (for anyone using memories)

| Skill | When to use |
|-------|-------------|
| `/memories-health` | Recall feels broken or stale — diagnoses hooks, DB, consolidation, MCP |
| `/memories-prune` | Monthly cleanup — surfaces low-quality atoms for review and soft-delete |
| `/memories-explain` | Understand why specific memories were retrieved for a query |

### Developer (for contributing to the memories project)

| Skill | When to use |
|-------|-------------|
| `/memories-perf-architect` | Before/after performance changes — audits N+1, SQL safety, complexity |
| `/memories-hebbian-architect` | Before/after learning changes — audits LTP/LTD, spreading activation, graph health |
| `/memories-benchmark` | Measures actual recall/remember latency and SQL query counts vs baseline |
| `/memories-graph-analyst` | Inspects live synapse graph — strength histogram, hub detection, orphan atoms |

All architect skills recall prior findings from the memory system before auditing and store new findings back — so each run builds on the last.

## Sub-Agent Memory Pattern

All architect skills are memory-aware by design. For other Task-based agents (research, implementation), include the memory pattern explicitly in the Task prompt:

**At the start of the agent prompt:**
> Before starting: call `mcp__memories__recall` with 1-2 specific queries to retrieve prior
> findings. Use `region="project:memories"` to scope results to this project.
> Example: `mcp__memories__recall(query="Hebbian synapse performance", region="project:memories")`

**At the end of the agent prompt:**
> After completing: store key findings using `mcp__memories__remember`. Use:
> - `type="insight"` for architectural conclusions or design decisions
> - `type="antipattern"` for mistakes to avoid (include `severity` and `instead` fields)
> - `type="experience"` for what was done and what happened
> - `type="skill"` for reusable techniques
> - `region="project:memories"` to keep atoms project-scoped
> Example: `mcp__memories__remember(content="N+1 queries in _score_atoms...", type="antipattern",
>   region="project:memories", severity="medium", instead="Pass atom_map from _spread_activation")`

This ensures agent work accumulates in the graph rather than being lost after each session.
