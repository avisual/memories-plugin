# Changelog

All notable changes to LATTICE will live here. Versions follow
semantic versioning.

## [0.1.1] — Composition that works

The release where the parts actually started adding up. Live web
research, the two-LLM split, and code search all landed and were
proven on real multi-file, multi-step tasks driven by a 1.5B local
model on a 4-CPU box.

### Added

- **`Research` verb** + curl-cffi browser-impersonating web fetcher:
  the LLM can fetch documentation URLs at run time; the fetched
  content becomes a `fact` atom in the brain; next-cycle recall
  surfaces it. SSRF-guarded (localhost / private RFC1918 / file://
  / javascript: all blocked).

- **`TwoStageProposer`** (Planner picks a verb, Executor fills slots
  for that verb only). Lifted directly from the small-model failure
  mode observed live: one open-ended JSON call was overloaded; two
  focused calls are reliable. CLI: `--two-stage` on `lattice agent`
  and `lattice do`. `--executor-model` lets you mix sizes
  (e.g. 0.5B Planner + 1.5B Executor).

- **Semble code search** (Model2Vec + BM25, CPU, sub-second) for the
  SENSE organ. Per-cycle task query surfaces top-K relevant chunks
  in the observation instead of dumping the full symbol list.
  Optional [`search`] extra. CLI: `--semble`.

- **`--decompose`** on `lattice do` (previously `agent`-only):
  conjunction-split a single task string into multiple subtasks
  sharing one overlay.

### Fixed

- Observation builder: atom recall now runs every cycle (not just
  the first). Mid-loop atoms written by `Research` were previously
  invisible to the very next prompt.
- Observation builder: `research` step payload is now rendered in
  history hints so the model sees what it fetched.
- Stuck detector now also catches the case where the same verb is
  emitted N+ times consecutively regardless of kind (e.g. the
  `Research`-spam pattern observed live with 1.5B).
- Reverted an over-eager "LLM-first when task contains 'with/using/?'"
  heuristic that misfired on tasks like "with default 5.0" and
  bypassed a matching pattern.
- Verify gate: mypy subprocess now uses `sys.executable` instead of
  bare `python` so the venv's mypy is found.
- `_Verb.confidence` now defaults to 0.5 — small models routinely
  forget the field; defaulting unblocks correct-shape actions.

### Live evidence (run on this machine)

**Composition with web research + LLM + atom feedback**
(Qwen-1.5B, two-stage, no patterns matched):
```
lattice do "Add Flask-CORS support to src/app.py. ... emit Research
  with url=https://flask-cors.readthedocs.io/en/latest/. After
  researching, emit AddImport for the flask_cors module." \
  --model Qwen/Qwen2.5-1.5B-Instruct --two-stage --write
# cycle 1: Planner→Research, Executor→Research(flask-cors-docs)
# cycle 2: Planner→AddImport, Executor→AddImport(flask_cors, [CORS])
+from flask_cors import CORS
```

**Two-file multi-subtask** (Qwen-1.5B, two-stage, decompose, semble):
```
lattice do "Two subtasks. First: add an import of CORS from
  flask_cors to src/app.py — if you don't know flask_cors, emit
  Research(...) first. Second: add a bool field named cors_enabled
  default True to class Config in src/config.py." \
  --two-stage --semble --decompose
# src/app.py: +from flask_cors import CORS
# src/config.py: +    cors_enabled: bool = True
```

### Tests

- 222 unit tests passing (+13 vs 0.1.0).
- 4 live tests gated on `LATTICE_LIVE_WEB=1` or `LATTICE_LLM_SMOKE=1`
  (real curl-cffi fetch, real MiniLM/Model2Vec embedding, real
  transformers model generation).

## [0.1.0] — Alpha

First releaseable cut. The system runs end-to-end on a 4-CPU box
with no GPU and produces verified diffs against real Python code.

### Architecture

- **Typed action DSL** (Organ 3): 11 verbs, all Pydantic-validated
  with discriminated-union dispatch. Invalid actions cannot reach
  the compiler.
- **Action compiler** (libcst, Python): all 5 mutating verbs
  (AddImport, AddField, AddParameter, WrapInTry, AddTest) plus
  RenameSymbol with cross-file reference rewriting.
- **Atom store** (Organ 1, v0): SQLite + sentence-transformers
  MiniLM embeddings. Semantic recall, no external service.
- **Symbol graph** (Organ 2): libcst walk yielding typed `Symbol`
  records (file, dotted name, kind, line).
- **Pattern proposer**: deterministic regex routing for common
  natural-language phrasings. No LLM call needed for: imports
  (plain/from/alias), rename, add-parameter, add-field, wrap-in-
  try, add-test.
- **Local LLM proposer**: Qwen2.5-0.5B / 1.5B / arbitrary HF model
  with JSON-schema-validated output, loose coercion, retry-with-
  error correction, multi-candidate sampling for pre-flight.
- **Hosted LLM proposer**: Anthropic API with auto-generated tool
  specs from the action union's Pydantic schemas. Requires
  `ANTHROPIC_API_KEY` and the `[hosted]` extra.
- **Agent loop**: stack-machine model — one action per cycle,
  observable state controls the program counter, no multi-step
  planning required of the LLM. Stuck-detection breaks loops.
  Pre-flight simulation rejects no-op candidates.
- **Multi-subtask runner**: planner-executor split with shared
  overlay across subtasks. Advances on either MarkDone or stuck.
- **Brain feedback**: experience atoms written after every
  successful agent run; brain accumulates lived knowledge.
- **Seed pack**: 96 curated atoms covering Python conventions,
  antipatterns, common libraries.
- **Verify**: ast.parse always on; optional mypy type-check via
  `--types` + the `[typecheck]` extra.
- **Apply-to-disk**: atomic write-then-rename, path-safety guards
  matching FilesystemWorkspace.

### CLI

- `lattice do "task"` — one-shot: init brain if needed, run agent,
  write file.
- `lattice init` — initialise a brain in a project.
- `lattice agent` — full-control multi-step agent.
- `lattice intent` — deterministic Intent expansion across files.
- `lattice apply` — single-action JSON application.
- `lattice brain inspect/dump/import/export` — curate the brain.
- `lattice atom add/recall/seed` — lower-level brain ops.

### Tests

- 205 unit tests covering every layer.
- 2 live tests gated on `LATTICE_LLM_SMOKE=1` (model download).
- GitHub Actions workflow for CI.

### Examples

- 5 runnable shell scripts (`examples/01_hello.sh` ... `05_brain_grow.sh`)
  demonstrating each major capability.

### Verified live

- Multi-file Intent expansion: 4 typed actions across 3 files.
- Pattern-driven tasks: `import request from flask` and friends,
  zero LLM calls.
- LLM-driven slot-filling: AddParameter on a class method (Qwen-1.5B).
- Cross-file rename: real lattice production code refactored in a
  tempdir, both the definition and references updated.

### Known limitations

- Multi-step LLM tasks at small scale (0.5–1.5B) remain brittle;
  mitigated by patterns + the planner-executor split.
- Aliased imports (`from X import Y as Z`) not yet resolved by
  RenameSymbol.
- No test-run verify gate yet (only parse + optional type-check).
- Python only; cross-language is not built.
- Activation steering, apprentice (Organ 8), macro evolution
  (Organ 9) — design exists, not built.
