# Troubleshooting

## Quick diagnostics

Always start here:

```bash
uv run python -m memories diagnose
```

This checks: running server processes, Ollama installation, Ollama daemon, embedding model, database health, and Claude Code config files.

---

## Ollama issues

### "Ollama server unreachable" / embedding failures

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Or as a macOS service
brew services start ollama
```

### "Model not found: nomic-embed-text"

```bash
ollama pull nomic-embed-text
```

---

## MCP server issues

### memories tools not available in Claude Code

1. Check the server is registered globally:
   ```bash
   grep -A5 '"memories"' ~/.claude.json
   ```
   Should show a `stdio` entry pointing to the `.venv/bin/python` binary.

2. If missing, re-run setup:
   ```bash
   uv run python -m memories setup
   ```

3. Restart Claude Code after any config change.

### Multiple windows — do I need to do anything?

No. Each Claude Code window spawns its own `memories` server process. All processes share `~/.memories/memories.db` safely via SQLite WAL mode. Check how many are running:

```bash
uv run python -m memories diagnose | head -3
```

---

## Hook issues

### Memories not appearing before prompts

1. Check hooks are configured:
   ```bash
   grep -A5 "UserPromptSubmit" ~/.claude/settings.json
   ```

2. Test the hook manually:
   ```bash
   echo '{"session_id":"test","prompt":"how does Redis work","cwd":"'$(pwd)'"}' | \
     uv run python -m memories hook prompt-submit
   ```

3. Check hook latency — hooks that exceed their timeout are silently skipped. Increase the timeout in `settings.json` if Ollama is slow.

### Hook fires but returns nothing relevant

The system only returns atoms scoring above the `min_score` floor (0.25 by default). If you have few atoms, run a backfill:

```bash
uv run python -m memories backfill
```

---

## Database issues

### Health check shows 0 atoms

If you've just installed, the database starts empty. Memories accumulate automatically as you use Claude Code. To pre-populate from past transcripts:

```bash
uv run python -m memories backfill
```

### Migrating from claude-mem

```bash
uv run python -m memories migrate --source ~/.claude-mem/claude-mem.db --dry-run
uv run python -m memories migrate --source ~/.claude-mem/claude-mem.db
```

### Database corruption / reset

```bash
# Back up first
cp ~/.memories/memories.db ~/.memories/memories.db.backup

# Reset (all memories lost)
rm ~/.memories/memories.db
uv run python -m memories health  # recreates fresh DB
```

---

## Performance issues

### Slow hook response / timeouts

- Ollama on first call downloads nothing — it may just be cold. Run `uv run python -m memories health` to warm it up.
- If consistently slow, check Ollama resource usage: `ollama ps`
- Reduce hook budget: `export MEMORIES_HOOK_BUDGET_PCT=0.01`

### DB growing too large

Run consolidation to prune weak/duplicate atoms:

```bash
uv run python -m memories diagnose   # check current size
# In Claude Code: use the `reflect` MCP tool
# Or via CLI for full graph maintenance:
uv run python -m memories reatomise
uv run python -m memories normalise
```

---

## Uninstall

Remove memories from Claude Code config while keeping data:
```bash
uv run python -m memories setup --uninstall
```

Delete all data:
```bash
rm -rf ~/.memories/
```
