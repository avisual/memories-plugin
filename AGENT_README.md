# memories — Agent README

**Brain-like persistent memory for Claude Code.** Neural graph with spreading activation, auto-linking, Hebbian learning, and hook-based injection into every session.

If you are a Claude Code agent working in this repository, this document tells you everything you need to know.

---

## What this project is

`memories` is a Python package (`src/memories/`) that gives Claude Code a persistent memory across sessions. It:

1. **Stores knowledge as atoms** — short facts, experiences, insights, skills, antipatterns, preferences, tasks — in a SQLite database with vector embeddings.
2. **Links atoms into a neural graph** — synapses with Hebbian strengthening connect related atoms so spreading activation finds context beyond pure keyword/vector match.
3. **Injects relevant atoms into every Claude prompt** — via Claude Code hooks (UserPromptSubmit, SessionStart, Stop, etc.) that run as shell commands.
4. **Exposes 13 MCP tools** (`remember`, `recall`, `connect`, `forget`, `amend`, `reflect`, `status`, `pathway`, `stats`, `create_task`, `update_task`, `list_tasks`, `stale_memories`) for use during interactive sessions.

---

## Repository layout

```
src/memories/
├── brain.py          Central orchestrator — the public API all other code calls
├── atoms.py          CRUD for memory atoms (nodes)
├── synapses.py       CRUD for synapse connections (edges)
├── retrieval.py      Spreading activation recall with multi-signal ranking
├── learning.py       Auto-linking, Hebbian strengthening, novelty assessment
├── context.py        Token budget manager with progressive compression
├── storage.py        SQLite + sqlite-vec + FTS5 backend; schema + migrations
├── embeddings.py     Ollama embedding client with caching and batching
├── consolidation.py  Memory decay, pruning, merging (like sleep)
├── config.py         Environment-variable-driven configuration
├── server.py         MCP server exposing the 13 tools
├── cli.py            Hook handlers + CLI commands (eval, stats, feedback, …)
├── __main__.py       Entry point — dispatches CLI or runs MCP server
├── migrate.py        One-time migration from claude-mem
└── setup.py          Install/uninstall hooks and MCP config
```

---

## Core data model

### Atom
```
id, content, type, region, confidence, importance, access_count,
last_accessed_at, created_at, source_project, source_session,
source_file, tags, severity, instead, is_deleted, task_status
```
**Types:** `fact`, `experience`, `skill`, `preference`, `insight`, `antipattern`, `task`

### Synapse
```
id, source_id, target_id, relationship, strength, bidirectional,
activated_count, last_activated_at, created_at
```
**Relationships:** `related-to`, `caused-by`, `part-of`, `contradicts`, `supersedes`, `elaborates`, `warns-against`

### Key tables
| Table | Purpose |
|-------|---------|
| `atoms` | All memory nodes |
| `atoms_fts` | FTS5 virtual table for BM25 keyword search |
| `atoms_vec` | sqlite-vec virtual table for ANN vector search |
| `synapses` | Graph edges |
| `hook_stats` | Hook invocation telemetry (latency, scores, atom counts) |
| `atom_feedback` | User good/bad signals that nudge atom confidence |
| `hook_session_atoms` | Accumulates atom IDs per session for Hebbian learning |
| `active_sessions` | Running sessions for sub-agent lineage detection |
| `session_lineage` | Parent→child session relationships |
| `embedding_cache` | Content-hash → embedding cache |

---

## Recall algorithm (4 steps)

1. **Vector search** — embed the query via Ollama (`nomic-embed-text`), find top-K seed atoms via ANN in `atoms_vec`.
2. **BM25 search** — FTS5 keyword search, merge seeds missed by vector search.
3. **Spreading activation** — propagate activation energy through synapse graph (configurable depth), decaying with distance and synapse strength.
4. **Multi-signal ranking** — composite score from: `vector_similarity × 0.4 + spread_activation × 0.25 + recency + confidence + frequency + importance + newness + bm25`. Apply region diversity cap. Fit to token budget with progressive compression.

Result structure:
```python
{
  "atoms": [{"id", "content", "type", "score", "score_breakdown", ...}],
  "antipatterns": [...],   # always surfaced, up to 3
  "budget_used": int,
  "budget_remaining": int,
  "seed_count": int,
  "total_activated": int,
  "compression_level": int,
}
```

---

## CLI commands

```bash
uv run python -m memories health        # system status
uv run python -m memories stats         # hook telemetry, relevance trend, feedback ratio
uv run python -m memories eval "<prompt>" [--project <name>] [--verbose]
                                        # show what Claude sees for any prompt
uv run python -m memories feedback <atom_id> good|bad
                                        # mark a recalled atom; nudges confidence ±0.05/0.10
uv run python -m memories diagnose      # full system diagnostics
uv run python -m memories backfill      # scan all Claude transcripts, store novel insights
uv run python -m memories relink        # re-run auto_link for all atoms
uv run python -m memories normalise     # rename fragmented region names to canonical
uv run python -m memories reatomise     # split large blob atoms into discrete facts via LLM
uv run python -m memories hook session-start   # hooks (read JSON from stdin)
uv run python -m memories hook prompt-submit
uv run python -m memories hook pre-tool
uv run python -m memories hook post-tool
uv run python -m memories hook stop
```

---

## MCP tools (interactive sessions only)

| Tool | Description |
|------|-------------|
| `remember` | Store a new atom |
| `recall` | Semantic search with spreading activation |
| `connect` | Create a synapse between two atoms |
| `forget` | Soft-delete an atom |
| `amend` | Update atom content/confidence |
| `reflect` | Consolidation pass (decay, prune, merge) |
| `status` | System health summary |
| `pathway` | Show synapse paths between atoms |
| `stats` | Hook invocation stats and memory effectiveness metrics |
| `create_task` | Create a task atom with lifecycle tracking |
| `update_task` | Update task status and optionally flag linked memories as stale |
| `list_tasks` | List task atoms with optional status/region filters |
| `stale_memories` | Find memories linked to completed tasks that may be outdated |

---

## Hook lifecycle

| Hook | When | What it does |
|------|------|-------------|
| `SessionStart` | Session opens | Recalls project-specific context, registers session |
| `UserPromptSubmit` | Each prompt | Recalls relevant atoms for the prompt, injects as context |
| `PreToolUse` | Before Task/Bash | Captures Claude's intent as insight/experience atom |
| `PostToolUse` | After each tool | Captures novel outputs (errors, file changes) |
| `PostResponse` | After each response | Extracts facts/skills/antipatterns from response |
| `Stop` | Session ends | Reads transcript, stores insights, runs Hebbian learning |
| `SubagentStop` | Sub-agent ends | Same as stop, propagates atoms to parent session |
| `PreCompact` | Context compaction | Checkpoints Hebbian learning mid-session |

---

## Development

```bash
uv run pytest           # 935 tests
uv run pytest -q        # quiet mode
uv run python -m memories health    # sanity check
```

### Config (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORIES_DB_PATH` | `~/.memories/memories.db` | SQLite path |
| `MEMORIES_OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `MEMORIES_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `MEMORIES_HOOK_BUDGET_PCT` | `0.05` | Fraction of context window for hook injection |
| `MEMORIES_DISTILL_THINKING` | `false` | Extract atomic facts from thinking blocks |
| `MEMORIES_DISTILL_MODEL` | `llama3.2:3b` | Model for atomic fact extraction |

---

## Multi-window safety

Each Claude Code window runs its own server process; all share `~/.memories/memories.db` via SQLite WAL mode.
