# Source: memories

16 modules implementing a brain-like memory system:
- brain.py: Central orchestrator — the public API all other code calls
- atoms.py / synapses.py: Core data layer (CRUD for nodes and edges)
- retrieval.py: Spreading activation recall with multi-signal ranking
- learning.py: Auto-linking, Hebbian strengthening, novelty assessment
- context.py: Token budget manager with progressive compression
- storage.py: SQLite + sqlite-vec + FTS5 backend; schema + migrations
- embeddings.py: Ollama embedding client with caching and batching
- consolidation.py: Memory decay, pruning, merging (like sleep)
- config.py: Environment-variable-driven configuration
- server.py: MCP server exposing 14 tools
- cli.py: Hook handlers + CLI commands (eval, stats, feedback, …)
- __main__.py: Entry point — dispatches CLI or runs MCP server
- setup.py: Install/uninstall hooks and MCP config
- migrate.py: One-time migration from claude-mem
- credentials.py: Secret retrieval via pass/env vars
