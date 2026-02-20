# avisual memories

[![Tests](https://github.com/avisual/memories-plugin/actions/workflows/tests.yml/badge.svg)](https://github.com/avisual/memories-plugin/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/badge/PyPI-coming%20soon-lightgrey)](https://github.com/avisual/memories-plugin)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-935%20passing-success)](https://github.com/avisual/memories)

> **The only AI memory system that prevents mistakes _before_ they happen.**

A brain-like memory system for AI agents that gives them persistent, searchable long-term memory with neural-inspired retrieval — plus **antipatterns** that proactively warn agents when they're about to repeat past mistakes.

Instead of starting every conversation from scratch, memories recalls relevant context from past sessions — what you worked on, what went wrong, what patterns you use — and injects it as context before each prompt.

## Why avisual memories?

Unlike basic RAG systems or commercial memory APIs, avisual memories is built like an actual brain:

| Feature | avisual memories | Mem0 ($249/mo) | LangChain Memory | ChromaDB |
|---------|------------------|----------------|------------------|----------|
| **Antipatterns** (proactive warnings) | ✅ | ❌ | ❌ | ❌ |
| Spreading activation (neural retrieval) | ✅ | ❌ | ❌ | ❌ |
| Hebbian learning (auto-connection) | ✅ | ❌ | ❌ | ❌ |
| Memory consolidation (decay/merge) | ✅ | ❌ | ❌ | ❌ |
| Local-first (no API costs) | ✅ | ❌ | ✅ | ✅ |
| Open source (MIT) | ✅ | ❌ | ✅ | ✅ |
| Multi-agent memory sharing | ✅ | ✅ | ❌ | ❌ |
| Works with any AI agent | ✅ | ✅ | ✅ | ✅ |

**The antipatterns feature alone saves hours of debugging.** When your agent tries to repeat a mistake (like using `rm` instead of `trash`, or browser automation that got blocked), memories surfaces a warning _before_ the command runs.

## How it works

Memories stores knowledge as **atoms** (discrete facts, experiences, skills, antipatterns) connected by **synapses** (weighted relationships). Retrieval uses **spreading activation** — when you recall one memory, activation flows through connected memories like neural pathways, surfacing related knowledge you didn't explicitly search for.

The system integrates with Claude Code through two channels:
- **MCP server** — 13 tools Claude can call directly (remember, recall, connect, forget, amend, reflect, status, pathway, stats, create_task, update_task, list_tasks, stale_memories)
- **Hooks** — automatic context injection on every prompt, plus learning from tool outputs

### Architecture

```
┌─────────────────────────────────────────────────┐
│                   Claude Code                    │
│                                                  │
│  UserPromptSubmit ──→ Hook ──→ Recall memories   │
│  PostToolUse ──────→ Hook ──→ Learn from errors  │
│  Stop ─────────────→ Hook ──→ Hebbian learning   │
│                                                  │
│  MCP Tools: remember, recall, connect, forget,   │
│             amend, reflect, status, pathway       │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│                 memories server                   │
│                                                  │
│  Brain ──→ Retrieval (spreading activation)      │
│        ──→ Learning (Hebbian, auto-linking)       │
│        ──→ Consolidation (decay, merge, prune)   │
│        ──→ Context (budget compression)          │
│                                                  │
│  Storage: SQLite + sqlite-vec + FTS5             │
│  Embeddings: Ollama (nomic-embed-text, 768-dim)  │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Install from Source

```bash
# 1. Clone and install
git clone https://github.com/avisual/memories.git
cd memories
uv sync

# 2. Run interactive setup
uv run python -m memories setup --interactive
```

The setup wizard will:
- ✓ Check if Ollama is installed (and tell you how to install it)
- ✓ Start Ollama daemon if needed (with your permission)
- ✓ Download the embedding model (with your permission)
- ✓ Create the database directory
- ✓ Register the MCP server in Claude Code (with your permission)
- ✓ Run a health check

Start a new Claude Code session and memories will be active!

## Tell Claude to Install This

**Copy and paste this prompt to Claude Code:**

```
Install the avisual-memories MCP server for me. Run these commands:

1. Clone: git clone https://github.com/avisual/memories.git && cd memories && uv sync
2. Install Ollama if needed: brew install ollama && ollama serve &
3. Pull embedding model: ollama pull nomic-embed-text
4. Run setup: uv run python -m memories setup --non-interactive
5. Verify: uv run python -m memories diagnose

Then restart Claude Code and I'll have persistent memory!
```

This gives Claude explicit commands to run - no ambiguity, no recursion.

## Installation

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **[Ollama](https://ollama.ai/)** — local embedding model server
- **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** — Anthropic's CLI

### 1. Clone and install

```bash
git clone https://github.com/avisual/memories.git
cd memories
uv sync
```

### 2. Run setup

```bash
# Interactive setup with prompts
uv run python -m memories setup

# Non-interactive setup (automated)
uv run python -m memories setup --non-interactive
```

### 3. Verify installation

```bash
# Run diagnostics
uv run python -m memories diagnose

# Or run health check
uv run python -m memories health
```

You should see all components marked as `[ok]`.

### Manual Setup (Alternative)

If you prefer manual configuration, follow these steps:

#### Install and start Ollama

```bash
# macOS
brew install ollama
ollama serve &

# Pull the embedding model
ollama pull nomic-embed-text
```

#### Verify the installation

```bash
uv run python -m memories health
```

You should see:

```
memories health check:
  atoms: 0
  synapses: 0
  regions: 0
  db_size: 0.05 MB
  ollama: healthy
  model: nomic-embed-text
```

#### Register the MCP server

Add to your `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "memories": {
      "type": "stdio",
      "command": "/path/to/memories/.venv/bin/python",
      "args": ["-m", "memories"]
    }
  }
}
```

Replace `/path/to/memories` with your actual clone path.

#### Configure hooks

Add to your `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/memories/.venv/bin/python -m memories hook prompt-submit",
            "timeout": 15
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/memories/.venv/bin/python -m memories hook post-tool",
            "timeout": 10
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/memories/.venv/bin/python -m memories hook stop",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

Replace `/path/to/memories` with your actual clone path.

#### Start a new Claude Code session

The memory system is now active. On each prompt, relevant memories are automatically recalled and injected as context.

## MCP Tools

Once registered, Claude can use these tools directly:

| Tool | Description |
|------|-------------|
| `remember` | Store a new memory atom. Auto-creates synaptic connections to related memories. |
| `recall` | Search memories using semantic similarity with spreading activation. |
| `connect` | Create or strengthen a connection between two memories. |
| `forget` | Soft-delete (recoverable) or hard-delete a memory. |
| `amend` | Update an existing memory. Re-embeds and re-links if content changes. |
| `reflect` | Run memory consolidation — decay, prune, merge, promote. Like sleep. |
| `status` | Get system health: atom/synapse counts, regions, DB size, Ollama status. |
| `pathway` | Visualize the connection graph radiating from a specific memory. |

## Hooks

Hooks run automatically during Claude Code sessions:

| Hook | Event | What it does |
|------|-------|-------------|
| `prompt-submit` | UserPromptSubmit | Recalls relevant memories and injects them as context before every prompt |
| `post-tool` | PostToolUse | Learns from Bash errors, file edits, and tool outputs (novelty-gated before storing) |
| `pre-tool` | PreToolUse | Reads thinking blocks before tool execution for richer context capture |
| `stop` | Stop | Reads session transcript, applies Hebbian learning, propagates sub-agent atoms to parent session |
| `subagent-stop` | SubagentStop | Same as stop — runs in sub-agent (Task) sessions and merges atoms into the parent's learning graph |
| `pre-compact` | PreCompact | Checkpoints Hebbian learning mid-session so atoms aren't lost if context compacts before stop fires |

### Sub-agent learning

When Claude Code spawns sub-agents (via the Task tool), each sub-agent runs its own `stop` hook. Memories detects the parent session automatically (by project + recency) and merges the sub-agent's atoms into the parent's co-activation graph. The parent's final `stop` then runs one consolidated Hebbian pass linking everything together.

## Configuration

All settings use environment variables with the `MEMORIES_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORIES_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `MEMORIES_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `MEMORIES_EMBEDDING_DIMS` | `768` | Embedding dimensions |
| `MEMORIES_DB_PATH` | `~/.memories/memories.db` | Database file path |
| `MEMORIES_CONTEXT_WINDOW_TOKENS` | `200000` | Model context window size |
| `MEMORIES_HOOK_BUDGET_PCT` | `0.02` | Hook injection budget (% of context window) |
| `MEMORIES_DEDUP_THRESHOLD` | `0.92` | Cosine similarity above which a new atom is skipped as a near-duplicate |
| `MEMORIES_REGION_DIVERSITY_CAP` | `5` | Maximum atoms per project returned in a single retrieval pass |
| `MEMORIES_DISTILL_THINKING` | `false` | Use a local LLM to extract atomic facts from Claude thinking blocks |
| `MEMORIES_DISTILL_MODEL` | `llama3.2:3b` | Ollama model used for fact extraction (any generative model works) |

Nested config uses double underscores:

```bash
# Change spreading activation depth
export MEMORIES_RETRIEVAL__SPREAD_DEPTH=3

# Change Hebbian learning increment
export MEMORIES_LEARNING__HEBBIAN_INCREMENT=0.1
```

### Atomic fact extraction (optional)

When `MEMORIES_DISTILL_THINKING=true`, the stop hook uses a local Ollama generative model to extract 2–5 discrete facts from each Claude thinking block, storing each as a separate atom. This produces a denser, more precisely-linked memory graph at the cost of additional Ollama inference time.

```bash
# Enable with default model (llama3.2:3b)
ollama pull llama3.2:3b
export MEMORIES_DISTILL_THINKING=true

# Or use a different model
export MEMORIES_DISTILL_MODEL=mistral:7b
```

## CLI Commands

Run `python -m memories <command>` (or `uv run python -m memories <command>` from source):

| Command | Description |
|---------|-------------|
| `setup` | Interactive or non-interactive setup wizard |
| `health` | Quick health check (DB, Ollama, model) |
| `diagnose` | Full diagnostics across all components |
| `stats` | Session stats, hook performance, top atoms, latency |
| `backfill` | Scan all `~/.claude/projects/` transcripts and store novel insights as atoms. Auto-relinks the graph when done. Safe to run repeatedly. |
| `relink` | Re-run `auto_link` for every atom to fill any missing synapses. Idempotent — existing synapses are strengthened, not duplicated. |
| `normalise` | Rename fragmented region aliases to canonical names (e.g. merges `general`, `project:git` → `project:utils`). |
| `reatomise` | Split large blob atoms into 2–5 atomic facts using a local LLM, soft-delete the originals, then auto-relink the graph. Requires Ollama. |
| `migrate` | Import atoms from a legacy claude-mem database |

### Verifying injection

To see exactly what memories are injected for a given prompt:

```bash
echo '{"session_id":"x","prompt":"YOUR QUESTION","cwd":"'$(pwd)'"}' | \
  python -m memories hook prompt-submit
```

### Backfilling historical transcripts

```bash
# Basic backfill (novelty-gated, safe to re-run)
python -m memories backfill

# With verbose output and LLM fact extraction
MEMORIES_DISTILL_THINKING=true python -m memories backfill --verbose
```

### Maintaining graph quality

```bash
# After a backfill, fix any region fragmentation:
python -m memories normalise --verbose

# Split large blob atoms into atomic facts (requires Ollama):
python -m memories reatomise --verbose

# Re-wire the whole graph (runs auto_link for every atom):
python -m memories relink --verbose
```

## Memory Types

| Type | Description |
|------|-------------|
| `fact` | A verified piece of knowledge |
| `experience` | Something learned from practice |
| `skill` | A how-to or technique |
| `preference` | A personal or project preference |
| `insight` | A derived observation or conclusion |
| `antipattern` | A known mistake to avoid (surfaced as warnings during recall) |
| `task` | A tracked task with lifecycle (pending → in_progress → completed) |

## Data Storage

All data is stored locally in `~/.memories/`:

- `memories.db` — SQLite database with sqlite-vec for vector search and FTS5 for keyword search
- `backups/` — Automatic backups (configurable count, default 5)

No data leaves your machine. Embeddings are generated locally via Ollama.

## Migration from claude-mem

If you have existing observations from the [claude-mem](https://github.com/thedotmack/claude-mem) plugin:

```bash
uv run python -m memories migrate --source ~/.claude-mem/claude-mem.db
```

Use `--dry-run` to preview without making changes.

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_retrieval.py

# Health check
uv run python -m memories health
```

## Troubleshooting

### Multi-window support

Multiple Claude Code windows can share memories simultaneously. Each window spawns its own server process; they all access the same `~/.memories/memories.db` safely via SQLite WAL mode.

To see how many servers are running:
```bash
uv run python -m memories diagnose
```

### "Ollama server unreachable"

Make sure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama manually
ollama serve

# Or as a service (macOS)
brew services start ollama
```

### "Model not found: nomic-embed-text"

Pull the embedding model:
```bash
ollama pull nomic-embed-text
```

### "Health check failed"

Run full diagnostics:
```bash
uv run python -m memories diagnose
```

This will check:
- MCP server status
- Ollama installation and daemon
- Required model availability
- Database health
- Configuration files

### Database issues

If you encounter database corruption or lock issues:

```bash
# Check database status
uv run python -m memories health

# Backup and reset (WARNING: deletes all memories)
mv ~/.memories/memories.db ~/.memories/memories.db.backup
uv run python -m memories health  # Creates fresh DB
```

### Uninstall

To remove memories from Claude Code configuration:
```bash
uv run python -m memories setup --uninstall
```

This removes MCP server registration and hooks but keeps your `~/.memories/` data directory. To delete all data:
```bash
rm -rf ~/.memories/
```

### Getting Help

- Check diagnostics: `uv run python -m memories diagnose`
- View stats: `uv run python -m memories stats`
- Check health: `uv run python -m memories health`
- Open an issue: https://github.com/avisual/memories/issues

## Documentation

- **[Getting Started Guide](https://avisual.github.io/memories/getting-started/)** - Installation and setup
- **[Antipatterns Deep Dive](https://avisual.github.io/memories/concepts/antipatterns/)** - How mistake prevention works
- **[Spreading Activation](https://avisual.github.io/memories/concepts/spreading-activation/)** - Neural-inspired retrieval explained
- **[API Reference](https://avisual.github.io/memories/api/)** - MCP tools documentation
- **[Best Practices](https://avisual.github.io/memories/guides/best-practices/)** - Tips for effective memory management

## Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Add tests for new features
4. Ensure all 935 tests pass: `uv run pytest`
5. Submit a PR

## License

MIT - see [LICENSE](LICENSE) file for details.

## Links

- **PyPI**: Coming soon
- **Documentation**: https://avisual.github.io/memories/
- **GitHub**: https://github.com/avisual/memories
- **Issues**: https://github.com/avisual/memories/issues
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
