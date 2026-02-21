# Claude Code Integration

This guide covers the full integration between memories and Claude Code: MCP server registration, hook configuration, and what happens during a session.

## How it works

memories integrates with Claude Code through two independent channels:

1. **MCP server** — Claude can call the 13 memory tools (`remember`, `recall`, etc.) directly during a session
2. **Hooks** — shell commands that fire automatically at lifecycle events (session start, each prompt, tool use, session end)

Both channels access the same `~/.memories/memories.db` database.

## Setup

### Automated (recommended)

```bash
uv run python -m memories setup
```

This registers the MCP server globally and configures hooks interactively.

### Manual MCP registration

Add to `~/.claude.json` under `mcpServers`:

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

Replace `/path/to/memories` with your actual clone path (e.g. `/Users/you/git/memories`).

### Manual hook configuration

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [{
          "type": "command",
          "command": "/path/to/memories/.venv/bin/python -m memories hook session-start",
          "timeout": 15
        }]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [{
          "type": "command",
          "command": "/path/to/memories/.venv/bin/python -m memories hook prompt-submit",
          "timeout": 15
        }]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Task|Bash",
        "hooks": [{
          "type": "command",
          "command": "/path/to/memories/.venv/bin/python -m memories hook pre-tool",
          "timeout": 10
        }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit",
        "hooks": [{
          "type": "command",
          "command": "/path/to/memories/.venv/bin/python -m memories hook post-tool",
          "timeout": 10
        }]
      }
    ],
    "Stop": [
      {
        "hooks": [{
          "type": "command",
          "command": "/path/to/memories/.venv/bin/python -m memories hook stop",
          "timeout": 30
        }]
      }
    ],
    "SubagentStop": [
      {
        "hooks": [{
          "type": "command",
          "command": "/path/to/memories/.venv/bin/python -m memories hook subagent-stop",
          "timeout": 30
        }]
      }
    ],
    "PreCompact": [
      {
        "hooks": [{
          "type": "command",
          "command": "/path/to/memories/.venv/bin/python -m memories hook pre-compact",
          "timeout": 15
        }]
      }
    ]
  }
}
```

## What happens during a session

### Session start (`SessionStart` hook)
- Registers the session and detects parent session for sub-agents
- Bridges the Claude Code session ID to the Brain for contextual encoding and session priming
- Recalls up to 3% of context window tokens of project-specific memories
- Injects them as a `system-reminder` block before Claude's first response

### Each prompt (`UserPromptSubmit` hook)
- Uses the user's prompt as a recall query
- Runs two parallel recalls: project-scoped (2% budget) + cross-project global (half budget), then merges and deduplicates results
- Output includes confidence scores, atom IDs, and synapse pathways so Claude can assess reliability
- Returns relevant atoms + antipattern warnings as a `system-reminder`
- Claude sees this context before generating a response

### Before tools (`PreToolUse` hook — Task and Bash only)
- Recalls antipattern warnings relevant to the command about to run (0.5% budget, antipatterns only)
- Captures Claude's intent (sub-agent prompts, Bash descriptions) as `insight`/`experience` atoms
- Novelty-gated: only stores if the content is sufficiently different from existing atoms

### After tools (`PostToolUse` hook — Bash/Write/Edit/MultiEdit/NotebookEdit)
- Captures errors from Bash commands as `antipattern` atoms
- Captures file edits as `experience` atoms
- Novelty-gated before storing

### Session end (`Stop` / `SubagentStop` hooks)
- Reads the JSONL transcript and extracts Claude's thinking blocks and final responses
- Stores novel insights as atoms
- Runs Hebbian co-activation learning: atoms that appeared together in this session have their shared synapses strengthened
- Sub-agent atoms propagate to the parent session's learning graph

### Context compaction (`PreCompact` hook)
- Checkpoints Hebbian learning mid-session so progress isn't lost if the session compacts before `Stop` fires

## Multi-window support

Multiple Claude Code windows can run simultaneously. Each window spawns its own `memories` server process; all share `~/.memories/memories.db` via SQLite WAL mode. No configuration needed — it just works.

## Verifying injection

To see exactly what memories are injected for a given prompt:

```bash
echo '{"session_id":"x","prompt":"YOUR QUESTION","cwd":"'$(pwd)'"}' | \
  uv run python -m memories hook prompt-submit
```

Or use the `eval` CLI command:

```bash
uv run python -m memories eval "your question here" --verbose
```

## Sub-agent memory sharing

When Claude Code spawns sub-agents via the `Task` tool, each sub-agent runs its own `subagent-stop` hook. memories automatically detects the parent session (by project + recency) and merges the sub-agent's atoms into the parent's co-activation graph. The parent's final `stop` then runs one consolidated Hebbian pass.

## Checking hook performance

```bash
uv run python -m memories stats
```

Shows per-hook invocation counts, average atoms returned, relevance scores, latency, and novelty pass rates.
