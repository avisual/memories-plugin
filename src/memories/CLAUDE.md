# Source: memories

14 modules implementing a brain-like memory system:
- brain.py: Central orchestrator and MCP tool implementations
- atoms.py / synapses.py: Core data layer (CRUD for nodes and edges)
- retrieval.py: Spreading activation recall with multi-signal ranking
- learning.py: Auto-linking, Hebbian strengthening, novelty assessment
- context.py: Token budget manager with progressive compression
- storage.py: SQLite + sqlite-vec + FTS5 backend
- embeddings.py: Ollama embedding client with caching and batching
- consolidation.py: Memory decay, pruning, merging (like sleep)
- config.py: Environment-variable-driven configuration
- server.py: MCP server exposing 8 tools
- cli.py: Hook handlers for Claude Code lifecycle events
- migrate.py: One-time migration from claude-mem
- credentials.py: Secret retrieval via pass/env vars
