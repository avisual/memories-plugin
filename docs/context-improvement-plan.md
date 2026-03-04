# Context Injection Improvement Plan

## Architecture Overview

The memories system injects context into Claude Code via two paths:

1. **CLI Hooks** (13 registered, 2 inject context): `session-start` and `prompt-submit` hooks recall memories and print formatted text to stdout, which Claude Code inserts into the system prompt.
2. **MCP Server** (`recall` tool): Claude explicitly calls this during conversations when it needs to look something up.

### Current Flow
```
Hook fires → cli.py reads stdin JSON → brain.recall(query, budget, region)
  → vector search + BM25 seeds → spreading activation → multi-factor scoring
  → budget compression → formatted text → stdout → Claude sees it
```

### Key Files
- `src/memories/cli.py` — hook handlers, query construction, output formatting
- `src/memories/retrieval.py` — scoring formula, spreading activation, seed search
- `src/memories/context.py` — budget calculation, compression levels, atom formatters
- `src/memories/brain.py` — recall orchestration, session tracking
- `src/memories/server.py` — MCP tool definitions
- `hooks/hooks.json` — hook registration config

---

## Findings by Section

### A. Hook Injection Quality

| ID | Severity | Finding | Impact |
|----|----------|---------|--------|
| A1 | **High** | `session-start` query is `"project context for {project}"` — a meta-description that has low semantic similarity to actual stored content | Session starts miss most relevant memories |
| A2 | **High** | Both hooks lock recall to `region=project:{project}` — cross-project knowledge is invisible | General antipatterns, skills, and facts from other projects never surface |
| A3 | **Medium** | `pre-tool` hook does not recall — missed opportunity to surface warnings before Bash/Task execution | Antipatterns about tool usage are only shown reactively |
| A4 | **Medium** | Antipatterns appear in both `atoms` and `antipatterns` result lists — duplicated in output | Wastes budget, confusing output |
| A5 | **Low** | No framing preamble tells Claude these are recalled memories vs. fresh instructions | LLM may over-weight or under-weight recalled content |

### B. Retrieval & Scoring

| ID | Severity | Finding | Impact |
|----|----------|---------|--------|
| B1 | **High** | BM25 weight at 0.05 is too low — exact keyword matches contribute almost nothing to score | Misses on precise technical queries (function names, error codes) |
| B2 | **Medium** | FTS query sanitizer strips tokens < 3 chars — drops "Go", "CI", "DB", "UI" | Short technical abbreviations are invisible to BM25 |
| B3 | **Medium** | Confidence weight (0.07) is undervalued — difference between confidence 1.0 and 0.3 is only 0.049 points | Low-quality memories compete equally with verified ones |
| B4 | **Low** | Newness bonus makes score range [0, 1.15] instead of [0, 1.0] — min_score threshold is effectively lower | Marginal impact on filtering |

### C. Context Formatting

| ID | Severity | Finding | Impact |
|----|----------|---------|--------|
| C1 | **High** | No confidence/score signal in hook output — Claude can't self-weight memory reliability | All memories look equally authoritative |
| C2 | **High** | Hook formatting ignores pathways — Claude sees isolated facts, not how they connect | Loses the graph structure that makes the system valuable |
| C3 | **Medium** | Antipattern `severity` and `instead` fields are buried in content string, not structured | Claude can't prioritize warnings by severity |
| C4 | **Medium** | No region/source metadata in output — Claude can't see where a memory came from | Can't assess staleness or project-relevance |
| C5 | **Low** | No atom IDs in hook output — Claude can't reference specific memories for feedback | Prevents memory correction loop |

### D. Budget & Efficiency

| ID | Severity | Finding | Impact |
|----|----------|---------|--------|
| D1 | **Medium** | `hook_budget_pct=0.02` (2%) with `context_window_tokens=200k` gives only 4000 tokens | Very tight budget limits how many memories can be surfaced |
| D2 | **Medium** | `session-start` and `prompt-submit` use the same flat budget — no priority differentiation | Session start could use more budget (one-time cost) vs. per-prompt |
| D3 | **Low** | Compression level 0 includes synapse descriptions but they're rarely populated in the atom dict | Wasted formatting effort |

---

## Implementation Plan: 7 Iteration Waves

### Iteration 1: Hook Query Quality + Cross-Region Recall (High Impact)

**Focus**: Fix the two biggest issues — bad session-start query and region-locked recall.

**Changes**:

1. **A1 — Better session-start query**: Replace `"project context for {project}"` with a multi-faceted query that recalls by region alone (no semantic query needed for broad project recall), OR use recent session topics as the query seed.

2. **A2 — Cross-region recall on prompt-submit**: Do two recalls — one scoped to the project region, one global (unscoped) — then merge results. This surfaces cross-project knowledge without drowning in irrelevant content.

3. **A4 — Deduplicate antipatterns from atoms**: Before formatting, remove atoms from the antipatterns list that already appear in the regular atoms list.

**Files**: `src/memories/cli.py`

**Architect review**: Performance + Context after implementation.

---

### Iteration 2: Scoring Formula Rebalancing (Medium-High Impact)

**Focus**: Fix BM25, confidence, and FTS token filtering to improve retrieval precision.

**Changes**:

1. **B1 — Raise BM25 weight**: Increase from 0.05 to 0.12 so exact keyword matches meaningfully contribute.

2. **B2 — Lower FTS minimum token length**: Change `_sanitize_fts_query` minimum from 3 to 2 characters so "Go", "CI", "DB", "UI" are searchable.

3. **B3 — Raise confidence weight**: Increase from 0.07 to 0.12, reduce frequency from 0.07 to 0.05 (verified quality > mere access count). Rebalance weights to sum to 1.0.

**Files**: `src/memories/retrieval.py`, `src/memories/config.py`

**Architect review**: Hebbian + Performance after implementation.

---

### Iteration 3: Context Formatting Upgrade (High Impact)

**Focus**: Make the injected context more useful for the LLM.

**Changes**:

1. **C1 — Add confidence signal to output**: Show confidence in formatted output so Claude can weight reliability.

2. **C2 — Include pathways in hook output**: When atoms have graph connections to each other, show a brief pathway section so Claude understands the relationship structure.

3. **C3 — Structure antipattern metadata**: Format severity and "instead" fields as distinct lines in antipattern output.

4. **A5 — Add framing preamble**: Wrap hook output with a brief preamble: "Previously recalled memories (verify before acting on stale information):"

5. **C5 — Include atom IDs**: Add atom ID to each line so Claude can reference specific memories when giving feedback.

**Files**: `src/memories/cli.py`, `src/memories/context.py`

**Architect review**: Context + Performance after implementation.

---

### Iteration 4: Pre-Tool Recall + Session Priming (Medium Impact)

**Focus**: Surface warnings before tool execution and improve session continuity.

**Changes**:

1. **A3 — Pre-tool recall for warnings**: When `pre-tool` fires for Bash/Task, use the tool input as a recall query with `types=["antipattern"]` and a small budget. Surface relevant warnings before execution.

2. **Session-aware priming**: Pass `session_atom_ids` from the hook session table to recall, so atoms accessed earlier in the session get priming boost (this feature exists in retrieval.py but is not wired up in hooks).

**Files**: `src/memories/cli.py`, `hooks/hooks.json` (may need matcher update)

**Architect review**: Performance (ensure pre-tool recall is fast enough for the 10s timeout).

---

### Iteration 5: Budget Intelligence (Medium Impact)

**Focus**: Smarter budget allocation and differentiated budgets per hook type.

**Changes**:

1. **D1/D2 — Differentiated budgets**: `session-start` gets 3% budget (one-time, can be larger), `prompt-submit` keeps 2%, `pre-tool` gets 1% (must be fast).

2. **D3 — Skip empty synapse formatting**: Don't include synapse section in level 0 when atom has no populated synapses.

3. **Priority-aware compression**: Session-start uses "background" priority (more atoms, less detail), prompt-submit uses "critical" priority (fewer atoms, more detail) since it's query-specific.

**Files**: `src/memories/cli.py`, `src/memories/context.py`, `src/memories/config.py`

**Architect review**: Context review to validate budget allocation.

---

### Iteration 6: MCP Server Enrichment (Lower Impact)

**Focus**: Improve the MCP `recall` tool output to match hook improvements.

**Changes**:

1. Include pathways in the MCP recall response in a human-readable format.
2. Add a `explain` parameter to recall that returns score breakdown per atom.
3. Ensure antipattern deduplication applies to MCP path too.

**Files**: `src/memories/server.py`, `src/memories/brain.py`

**Architect review**: Full architect review (Hebbian + Performance + Context).

---

### Iteration 7: Final Polish + Validation (Cleanup)

**Focus**: Address any remaining findings from architect reviews, run full test suite, final cleanup.

**Changes**:
- Fix any remaining architect findings from iterations 1-6
- Ensure all tests pass
- Validate end-to-end with realistic queries
- Document the changes in improvement-plan-v2.md

**Architect review**: Final all-architect review to confirm PASS.

---

## Success Criteria

After all iterations:
1. `session-start` surfaces top-5 most relevant memories for the project (not generic padding) — **DONE**
2. `prompt-submit` surfaces cross-project knowledge when relevant — **DONE**
3. Claude sees confidence, pathways, and structured antipatterns in context — **DONE**
4. Pre-tool warnings fire before dangerous Bash/Task calls — **DONE**
5. All 1000+ tests pass — **DONE** (1037 passing, 5 Ollama-dependent skipped)
6. Performance architect PASS (no new N+1 queries, all hooks under timeout) — **PASS**
7. Hebbian architect PASS (no new learning rule regressions) — **PASS**

## Status: COMPLETE (2026-02-21)

All 7 iterations implemented, both architect reviews passed.
