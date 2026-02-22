# Test Coverage Analysis & Hebbian Architecture Review

**Date:** 2026-02-22
**Overall Coverage:** 69% → **75%** after new tests (5,075 statements, 1,256 missed)
**Tests:** 1,044 → **1,127** passing across 48 test files

---

## Part 1: Coverage by Module

| Module | Stmts | Miss | Cover | Priority |
|--------|-------|------|-------|----------|
| `config.py` | 161 | 2 | 99% | - |
| `atoms.py` | 265 | 11 | 96% | Low |
| `context.py` | 107 | 4 | 96% | Low |
| `learning.py` | 311 | 16 | 95% | Low |
| `synapses.py` | 298 | 16 | 95% | Low |
| `consolidation.py` | 739 | 54 | 93% | Low |
| `retrieval.py` | 306 | 23 | 92% | Medium |
| `storage.py` | 332 | 41 | 88% | Medium |
| `process_manager.py` | 64 | 11 | 83% | Low |
| `brain.py` | 408 | 90 | 78% | **High** |
| `embeddings.py` | 179 | 40 | 78% | **High** |
| `server.py` | 130 | 30 | 77% | **High** |
| `cli.py` | 1,221 | 716 | 41% | **Critical** |
| `ollama_manager.py` | 71 | 55 | 23% | Medium |
| `setup.py` | 275 | 250 | 9% | Medium |
| `__main__.py` | 12 | 12 | 0% | Low |
| `credentials.py` | 33 | 33 | 0% | Low |
| `migrate.py` | 157 | 157 | 0% | Low |

---

## Part 2: High-Priority Coverage Gaps

### 1. `cli.py` — 41% coverage (716 missed statements) — CRITICAL

This is the largest module (1,221 statements) and the primary integration layer between Claude Code hooks and the Brain. Nearly 60% of it is untested.

**Untested areas:**

- **`_hook_session_start`** (lines 345-424): The full session-start flow including sub-agent lineage detection, project-scoped recall, and hook stat recording. This is the entry point for every Claude Code session.

- **`_hook_prompt_submit`** (lines 445-523): Dual-recall (project + global), antipattern deduplication, and atom formatting. The main recall pathway for user prompts.

- **`_hook_pre_tool`** (lines 1646-1718): Pre-tool antipattern recall for Bash/Task, novelty assessment, and experience storage. Critical safety feature.

- **`_hook_pre_compact`** (lines 1733-1774): Mid-session Hebbian checkpoint. Important for learning continuity.

- **`_hook_post_tool_failure`** (lines 1782+): Error capture and storage.

- **`_hook_session_end` / `_run_session_stop_logic`** (lines 1879-1991): Session-end Hebbian learning, atom formatting, final cleanup. The sole trigger for `session_end_learning`.

- **`_hook_context_expand`** (lines 2008-2075): Context expansion recall.

- **`backfill_transcripts`** (lines 1075-1151): Transcript backfill with novelty gating and distillation.

- **`dispatch()`** (lines 2172-2253): The main CLI routing function.

- **Various formatting/utility functions**: `_format_atom_line`, `_format_pathways`, `_project_name`, `_hook_budget`, `_extract_pre_tool_content`.

**Recommended tests:**

```
test_cli_session_start_with_project_recall
test_cli_session_start_subagent_lineage_detection
test_cli_prompt_submit_dual_recall_merging
test_cli_prompt_submit_antipattern_deduplication
test_cli_pre_tool_antipattern_warning
test_cli_pre_tool_novelty_gating
test_cli_pre_compact_hebbian_checkpoint
test_cli_post_tool_failure_capture
test_cli_session_end_hebbian_learning
test_cli_context_expand_recall
test_cli_dispatch_routing
test_cli_format_atom_line_types
test_cli_format_pathways
test_cli_project_name_extraction
test_cli_hook_budget_calculation
```

### 2. `brain.py` — 78% coverage (90 missed statements) — HIGH

**Untested areas:**

- **`status()`** (lines 715-760): Health stats gathering — orphan atoms, DB size, Ollama health check, region listing. Used by `/memories-health`.

- **`update_task()`** (lines 1247-1292): Task status updates with linked memory flagging. Core task management.

- **`get_stale_memories()`** (lines 1347-1414): Finding memories linked to completed tasks. Used by the `stale_memories` MCP tool.

- **`initialize()` edge cases** (lines 206-217): Error handling during initialization.

- **`remember()` edge cases** (lines 282-295): Deduplication and interference handling branches.

**Recommended tests:**

```
test_brain_status_returns_all_fields
test_brain_status_orphan_atom_counting
test_brain_update_task_flags_linked_memories
test_brain_update_task_noop_when_not_done
test_brain_get_stale_memories
test_brain_get_stale_memories_min_completed_filter
test_brain_remember_deduplication
test_brain_remember_interference_detection
```

### 3. `server.py` — 77% coverage (30 missed statements) — HIGH

**Untested areas:**

- **`stats()`** (lines 473-478): Hook stats MCP tool.
- **`create_task()`** (lines 520-531): Task creation MCP tool.
- **`update_task()`** (lines 559-568): Task update MCP tool.
- **`list_tasks()`** (lines 590-598): Task listing MCP tool.
- **`stale_memories()`** (lines 626-633): Stale memory detection MCP tool.

All of these are simple pass-through wrappers around Brain methods, but they normalize parameters (empty string to None) and handle exceptions, which should be verified.

**Recommended tests:**

```
test_server_stats_returns_dict
test_server_create_task
test_server_update_task
test_server_list_tasks
test_server_stale_memories
test_server_tools_handle_exceptions
```

### 4. `embeddings.py` — 78% coverage (40 missed statements) — HIGH

**Untested areas:**

- **`_ensure_ollama_ready()`** (lines 153-167): Ollama readiness check and error reporting.
- **`embed_text()` error paths** (lines 240-246): ConnectionError and ResponseError handling.
- **`_embed_batch()` error paths** (lines 514-522): Batch embed error handling.
- **`search_similar()` vec unavailable** (lines 340-342): Vector search when sqlite-vec not loaded.

**Recommended tests:**

```
test_embed_text_connection_error
test_embed_text_response_error
test_embed_batch_connection_error
test_embed_batch_response_error
test_embed_batch_length_mismatch
test_search_similar_no_vec
test_ensure_ollama_ready_failure
```

### 5. `storage.py` — 88% coverage (41 missed statements) — MEDIUM

**Untested areas:**

- **Thread-local connection management** (lines 433-461): Connection pooling across threads.
- **`execute_batch_write()`** (lines 494-498): Batch write operations.
- **`execute_transaction()`** (lines 524-552): Transaction wrapping with rollback.
- **`get_db_size_mb()`** (lines 632-666): Database size calculation.

### 6. `ollama_manager.py` — 23% coverage — MEDIUM

Most of the Ollama lifecycle management (auto-start, model pull, health checks) is untested. This is understandable since it requires a running Ollama server, but mock-based tests would catch regressions.

### 7. `setup.py` — 9% coverage — MEDIUM

The installation flow creates directories, pulls models, registers MCP servers, and configures hooks. Important for first-run experience but difficult to test without filesystem mocking.

---

## Part 3: Structural Test Gaps (not captured by line coverage)

### A. No integration test for the full hook lifecycle

The system has 43 test files, but none test the complete hook lifecycle: `session-start` -> `prompt-submit` (N times) -> `pre-tool` -> `post-tool` -> `session-end`. This is the actual usage pattern and would catch interaction bugs between hooks (e.g., atoms accumulated in `hook_session_atoms` during prompt-submit being correctly consumed by session-end Hebbian learning).

### B. No test for concurrent hook execution

In practice, multiple hooks can fire in rapid succession (e.g., `prompt-submit` immediately followed by `pre-tool`). The Brain singleton and SQLite WAL mode should handle this, but there are no concurrency tests for the hook path.

### C. No test for context budget integration with hooks

The `_hook_budget()` function calculates a token budget (2% of 200K = 4,000 tokens), but no test verifies that hook output actually fits within this budget. If the context budget compression fails or produces oversized output, it could blow up the Claude Code context window.

### D. No test for region diversity cap in recall

The `region_diversity_cap = 2` setting in config limits atoms per region in recall results. This is tested in `test_retrieval.py` but not in the integration path through `brain.recall()` -> hook output.

### E. No negative test for synapse type weight coherence

The `SynapseTypeWeights` dataclass has 8 fields, and the `RetrievalEngine.__init__` builds a lookup dict by replacing `-` with `_` in relationship names. If a new synapse type is added to `RELATIONSHIP_TYPES` without a corresponding weight, it silently falls back to 0.5. A test should verify all types have explicit weights.

---

## Part 4: Hebbian Architecture & Recall Quality Review

### Assessment: The Learning System Is Well-Designed

After reviewing `retrieval.py`, `learning.py`, `synapses.py`, `consolidation.py`, and `config.py`, the Hebbian architecture is sound and implements several biologically-grounded mechanisms correctly. All items from the improvement plan v2 (Waves 1-4) are confirmed implemented.

### What the system does well:

1. **BCM multiplicative increment** (`synapses.py:816`): `strength + increment * (1 - strength)` correctly implements diminishing returns at high strength. This prevents synapse saturation.

2. **Proportional LTD** with floor: `strength * (1 - ltd_fraction)` with `MIN(ltd_min_floor, ...)` prevents immortal weak synapses. Well-calibrated at 15% per cycle.

3. **Refractory gating** (`retrieval.py:743`): Visited set gates frontier expansion only, preserving superposition (convergent path additivity). This is the correct biological interpretation.

4. **Fanout normalization** (`retrieval.py:712`): `1/sqrt(fanout)` using pre-filter degree penalizes hubs correctly.

5. **Multi-signal scoring** (`retrieval.py:872-881`): 8 signals with configurable weights (vector=0.30, spread=0.25, BM25=0.12, confidence=0.12, importance=0.11, recency=0.08, frequency=0.02, newness=0.15 bonus). Well-balanced.

6. **Inhibitory exclusion** (`synapses.py:707-723`): `contradicts`, `supersedes`, `warns-against`, and `encoded-with` are excluded from Hebbian strengthening. Inhibitory-only pairs don't trigger new `related-to` creation.

7. **Cue overload protection** (`synapses.py:792-807`): `max_new_pairs_per_session=50` prevents O(n^2) hub formation in large sessions.

8. **Temporal weighting** (`synapses.py:775-784`): Pairs within 300s window get full increment; distant pairs get 0.5x. Good implementation of temporal contiguity.

9. **Synaptic Tagging and Capture** (`config.py:109-118`): STC with `stc_tagged_strength=0.25` and `stc_capture_window_days=14` implements Frey & Morris (1997) correctly.

10. **Hybrid decay** (`config.py:201-212`): Exponential-to-power-law transition at 90 days implements Wixted & Ebbesen (1991) for heavy-tail preservation.

### Areas for improvement in recall quality:

#### Medium Priority

**M1 — Session priming boost is very small (0.05)**

`_SESSION_PRIME_BOOST = 0.05` (`retrieval.py:82`) means atoms seen earlier in the session barely register as seeds. Since these atoms already proved relevant, a value of 0.10-0.15 would better implement working-memory priming without overwhelming genuine vector matches. The current value is below `min_activation = 0.1`, meaning primed atoms that don't match the vector query won't even propagate through spreading activation.

**M2 — No reconsolidation on re-access**

When an atom is recalled and then the user provides feedback or modifies related knowledge, the system doesn't update the atom's embedding or re-evaluate its synapses. Biological memory undergoes reconsolidation when re-activated — the act of retrieval makes a memory temporarily labile and subject to updating. Adding a lightweight reconsolidation check (re-run auto_link on high-confidence recalls) would improve graph quality over time.

**M3 — ~~Encoded-with synapses are not created~~ CORRECTED: They ARE created**

`brain.py:264-296` creates `encoded-with` synapses during `remember()` when `_current_session_id` is set. The CLI bridges this at `cli.py:351`. This is working correctly — session atoms get bidirectional `encoded-with` links at strength 0.15, capped to 10 per atom. No fix needed.

#### Low Priority

**L1 — No depth-aware decay weighting**

`retrieval.py:726`: `self._cfg.decay_factor` is applied as a constant 0.85 per hop. Since `current_activation` at depth 2 already incorporates the depth-1 decay, this compounds correctly (0.85^2 = 0.7225 total at depth 2). However, the system could benefit from a depth-aware decay schedule (e.g., steeper decay at greater depths) to more aggressively penalize distant associations and keep the result set tightly focused around the query.

**L2 — No primacy/recency effect in session Hebbian learning**

All atoms in a session are treated equally for Hebbian learning. Biological memory shows primacy (first items) and recency (last items) effects. Weighting the first and last few atoms in a session more heavily would improve learning signal quality.

**L3 — Contradiction detection is heuristic-only**

`learning.py:996-1099`: The contradiction detector uses keyword matching (negation words, antonym pairs, value assertions). This catches obvious contradictions but misses subtle ones. Since the system already has an embedding engine, comparing the semantic divergence of assertion-bearing sub-clauses would improve precision.

---

## Part 5: Prioritized Recommendations

### Tier 1 — High Impact, Achievable

1. **Test the CLI hook lifecycle end-to-end** — One integration test covering session-start through session-end with Hebbian learning verification. This tests the most critical untested path (716 lines) and validates learning actually happens through the hook system.

2. **Test `brain.status()` and `brain.get_stale_memories()`** — These power the health check and task management features. 8-10 focused unit tests.

3. **Test MCP server task management tools** — 5 tests for `create_task`, `update_task`, `list_tasks`, `stale_memories`, `stats`. Simple pass-through verification.

4. **Fix session prime boost** — Raise `_SESSION_PRIME_BOOST` from 0.05 to 0.12 so primed atoms exceed `min_activation=0.1` and actually propagate.

### Tier 2 — Good ROI

5. **Test embedding error paths** — 7 mock-based tests for ConnectionError, ResponseError, batch mismatches. Guards against silent failures.

6. **Test pre-tool antipattern warning path** — The safety-critical path that warns before Bash execution is completely untested.

7. **Test pre-compact Hebbian checkpoint** — Validates that mid-session learning persists.

8. **Verify decay_factor is applied correctly per depth** — Add a test that confirms depth-2 activation is lower than depth-1 for the same synapse strength.

### Tier 3 — Completeness

9. **Add synapse type weight coherence test** — Verify all `RELATIONSHIP_TYPES` have explicit `SynapseTypeWeights` entries.

10. **Test `ollama_manager.py` with mocks** — Basic health check, model pull, auto-start tests.

11. **Test `backfill_transcripts()`** — Important for onboarding, currently 0% covered.

12. ~~**Create `encoded-with` synapses during session**~~ — Already implemented in `brain.py:264-296`.
