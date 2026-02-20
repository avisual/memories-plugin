# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (Wave 11 — 2026-02-20)
- Bash error tightening: 4 specific error signatures now correctly classified
- `_SKIP_TOOLS` set consolidated into a single canonical definition
- Dead branch removal for cleaner hook dispatch

### Fixed (Wave 11 — 2026-02-20)
- Bash hook no longer misclassifies benign output as errors for the 4 hardened signatures

### Added (Wave 10 — 2026-02-20)
- Auto-learning loop: project-scoped recall now primes Hebbian co-activation at session start
- Sub-agent documentation added to AGENT_README covering spawning patterns and atom propagation
- `region_diversity_cap` reduced from 5 to 2 to reduce retrieval noise and surface cross-project patterns

### Changed (Wave 10 — 2026-02-20)
- `MEMORIES_REGION_DIVERSITY_CAP` default lowered from 5 to 2

### Fixed (Wave 9 — 2026-02-20)
- Outbound hub cap now enforced inside `hebbian_update` to prevent runaway hub growth
- Minor cleanup of unused imports and dead code paths

### Fixed (Wave 8 — 2026-02-20)
- Removed redundant `recall` call that fired twice per prompt, halving unnecessary DB load
- Antipattern classification fixed: atoms with `severity`/`instead` fields now correctly typed as `antipattern` rather than `insight`
- Decay coefficient corrected so daily decay rate matches documented target

### Fixed (Wave 7 — 2026-02-20)
- TOCTOU race in hub-cap enforcement: strength check and cap write now performed atomically
- Double-decay bug eliminated: consolidation no longer applies decay twice when both pruning and decay passes run in the same cycle
- Consolidation batch size capped to prevent memory spikes on large graphs

### Added (Wave 6 — 2026-02-20)
- Graph topology analysis: hub detection and orphan-atom reporting in `status`
- Dormant pruning: synapses inactive for >90 days are soft-pruned during consolidation
- BFS batch loading for spreading activation: neighbour atoms fetched in a single query per hop
- Type-differentiated decay rates: `antipattern` atoms decay slower than `experience` atoms

### Changed (Wave 6 — 2026-02-20)
- Spreading activation BFS now batches SQL reads per depth level instead of one query per atom

### Fixed (Wave 5 — 2026-02-20)
- Synapse correctness: duplicate synapses no longer created when `connect` is called for an existing pair
- CLI safety: hooks exit with code 0 on non-fatal errors to avoid blocking Claude Code
- Consolidation robustness: exceptions during decay/prune are caught and logged rather than crashing
- Brain cleanup: stale session records removed on startup to prevent lineage drift
- Lazy imports used for heavy dependencies so CLI startup time is not penalised

### Added (Wave 4 — 2026-02-20)
- Brain singleton for CLI hooks: all three hooks (prompt-submit, post-tool, stop) now share a single `Brain` instance per process, eliminating redundant initialisation

### Fixed (Wave 4 — 2026-02-20)
- Double Hebbian firing eliminated: `stop` hook no longer fires a second learning pass when sub-agent atoms are merged

### Added (Wave 3 — 2026-02-20)
- Batch learning reads: `get_session_atoms` fetches all session atom IDs in one query
- `RETURNING` clauses on `INSERT`/`UPDATE` statements replace follow-up `SELECT` round-trips
- Concurrent distillation: thinking-block fact extraction runs in a thread pool
- SQL aggregation pushed to the database for region diversity cap enforcement

### Fixed (Wave 2 — 2026-02-20)
- Batch neighbour fetches: spreading activation fetches all neighbours per node in a single `IN` query
- Proportional LTD: long-term depression now scales with synapse strength rather than applying a flat decrement
- Refractory spreading activation: nodes already visited in BFS are skipped to prevent re-activation cycles

### Fixed (Wave 1 — 2026-02-20)
- Hebbian correctness: co-activation updates now use BCM-style multiplicative increment proportional to current strength
- All-synapse strengthening: `hebbian_update` strengthens every co-activated synapse pair, not just the first
- Inhibitory exclusion: antipattern atoms are excluded from Hebbian potentiation to prevent mistake-loops
- SQL hardening: `hebbian_update` uses parameterised queries throughout

### Changed
- **Multi-window support**: removed singleton enforcement from `__main__.py`. Each Claude Code window now spawns its own MCP server process; all instances share `~/.memories/memories.db` safely via SQLite WAL mode. `diagnose` now counts all running server processes via `pgrep` instead of reading a lock file.

### Added
- **Task atom type** with lifecycle management (pending → active → done → archived)
- Tasks auto-link to related memories on creation
- Completing tasks flags linked memories as stale for review/cleanup
- 4 new MCP tools: `create_task`, `update_task`, `list_tasks`, `stale_memories`
- `get_stale_memories()` query to find memories linked to completed tasks
- Schema migration auto-adds `task_status` column to existing databases
- 17 new tests for task functionality

### Fixed
- Empty/whitespace queries now return empty result instead of crashing

### Improved
- Public API exports (Brain, Atom, Synapse, type constants)
- PEP 561 compliance with py.typed marker
- Documentation accuracy (tool counts, memory types)

## [0.1.0] - 2026-02-11

### Added
- Initial public release
- Core memory system with spreading activation retrieval
- Antipattern detection and proactive warnings
- Hebbian learning for connection strengthening
- Memory consolidation (decay, pruning, merging)
- Local embeddings via Ollama (nomic-embed-text)
- MCP server integration for AI agents
- SQLite + sqlite-vec + FTS5 storage
- Comprehensive test suite (502 tests)
- 1000-atom scalability tests
- Batch atom fetching optimization (67x speedup)

### Features
- `memory_recall(query)` - Search with spreading activation
- `memory_remember(content, type)` - Store new memories
- `memory_connect(source, target, type)` - Create explicit connections
- `memory_forget(id)` - GDPR-compliant deletion
- `memory_amend(id, content)` - Update existing memories
- `memory_reflect()` - Trigger consolidation
- `memory_status()` - System statistics
- `memory_pathway(source, target)` - Find conceptual paths

### Performance
- 602 atoms/sec creation rate
- 1128 synapses/sec creation rate
- 0.7ms average FTS query time
- 1545 recalls/sec

[0.1.0]: https://github.com/avisual/memories-plugin/releases/tag/v0.1.0
