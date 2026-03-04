# Installation

## Prerequisites

- **Python 3.13+** required
- **Ollama** for local embeddings (free, no API key)
- **AI agent** that supports MCP (Claude Code, Aider, etc.)

## Install from Source

```bash
git clone https://github.com/avisual/memories-plugin.git
cd memories-plugin
uv sync
```

## Install Ollama

### macOS

```bash
brew install ollama
brew services start ollama
```

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows

Download from [ollama.ai](https://ollama.ai/)

## Pull Embedding Model

```bash
ollama pull nomic-embed-text
```

This downloads a 274MB model for local embeddings (no API costs!).

## Run Setup

### Interactive Setup (Recommended)

```bash
uv run python -m memories setup --interactive
```

The wizard will:
- ✓ Check Ollama installation
- ✓ Start Ollama if needed
- ✓ Download embedding model
- ✓ Create database directory
- ✓ Register MCP server with your AI agent
- ✓ Run health check

### Non-Interactive Setup

```bash
uv run python -m memories setup --non-interactive
```

For automation/scripts (assumes Ollama already running).

## Verify Installation

```bash
uv run python -m memories diagnose
```

Should show:
```
memories diagnostics
========================================

  [ok] 1 memories server(s) running (PIDs: 12345)

  [ok] Ollama installed
  [ok] Ollama daemon running
  [ok] Model available: nomic-embed-text

Database:
  Total atoms: 0
  Total synapses: 0
  ...
All systems operational!
```

## Configuration

Default locations:
- **Data**: `~/.memories/`
- **Database**: `~/.memories/memories.db`

All settings are environment variables — no config file needed. See [Configuration](configuration.md).

## Next Steps

- [Quick Start](quick-start.md) - Try it out!
- [Configuration](configuration.md) - Customize settings
- [Troubleshooting](../guides/troubleshooting.md) - Common issues
