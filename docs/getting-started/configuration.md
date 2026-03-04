# Configuration

All settings are controlled via environment variables with the `MEMORIES_` prefix. No config file needed.

## Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORIES_DB_PATH` | `~/.memories/memories.db` | SQLite database file path |
| `MEMORIES_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `MEMORIES_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `MEMORIES_EMBEDDING_DIMS` | `768` | Embedding dimensions (must match model) |

## Hook Injection

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORIES_CONTEXT_WINDOW_TOKENS` | `200000` | Model context window size |
| `MEMORIES_HOOK_BUDGET_PCT` | `0.02` | Default fraction of context window for hook injection. Differentiated per hook: session-start=3%, prompt-submit=2%, pre-tool=0.5% |

## Deduplication

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORIES_DEDUP_THRESHOLD` | `0.92` | Cosine similarity above which a new atom is skipped as a near-duplicate of an existing one |
| `MEMORIES_REGION_DIVERSITY_CAP` | `2` | Max atoms per region returned in a single recall |

## Thinking Block Distillation (Optional)

When enabled, the stop hook uses a local Ollama generative model to extract 2–5 atomic facts from Claude's thinking blocks, storing each as a separate atom.

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORIES_DISTILL_THINKING` | `false` | Enable atomic fact extraction from thinking blocks |
| `MEMORIES_DISTILL_MODEL` | `llama3.2:3b` | Ollama model for fact extraction (any generative model works) |

```bash
# Enable with default model
ollama pull llama3.2:3b
export MEMORIES_DISTILL_THINKING=true

# Or use a different model
export MEMORIES_DISTILL_MODEL=mistral:7b
```

## Nested Config (Retrieval & Learning)

Nested config uses double underscores:

```bash
# Spreading activation depth (default 2)
export MEMORIES_RETRIEVAL__SPREAD_DEPTH=3

# Hebbian learning increment per co-activation
export MEMORIES_LEARNING__HEBBIAN_INCREMENT=0.1

# Minimum synapse strength to auto-create
export MEMORIES_LEARNING__AUTO_LINK_THRESHOLD=0.6
```

## Setting Variables

### Per-session (shell export)
```bash
export MEMORIES_HOOK_BUDGET_PCT=0.03
claude  # new session picks up the env var
```

### Persistent (shell profile)
Add to `~/.zshrc` or `~/.bashrc`:
```bash
export MEMORIES_DISTILL_THINKING=true
export MEMORIES_DISTILL_MODEL=llama3.2:3b
```

### Per-project (Claude Code hooks)
You can prefix hook commands in `~/.claude/settings.json` to pass variables only for specific projects.

## Verifying Config

```bash
uv run python -m memories health
```

Shows the active embedding model and database path. Full config introspection:

```bash
python -c "from memories.config import get_config; import json; c = get_config(); print(c)"
```
