# Changelog

All notable changes to LATTICE will live here. Versions follow
semantic versioning.

## [0.2.0] — Every organ has a v0; refactoring tool is real

The release where every organ in DESIGN.md has working code with
tests, the verb set covers what real refactoring needs structurally,
and the IMPROVES dial actually saves LLM calls via learned templates.

### Added

- **AddDecorator verb** — apply a decorator to a function or class.
  Compiler handles outermost/innermost positioning, idempotency.
  Pattern: 'Add @cached decorator to function f in src/x.py' /
  'Decorate function f in src/x.py with @app.route("/health")'.

- **DeleteSymbol verb** — remove a function/class/method by dotted
  name. Top-level removal AND method-of-class removal. require_present
  flag for soft 'ensure absent' semantics.

- **AddStatement function-scoped positions** — `start_of_function` and
  `end_of_function` (plus the existing module-level `end` and
  `top_after_imports`). With a `target` SymbolRef, inserts inside a
  specific function body. Closes the inside-the-body gap.

- **Multi-action template inference (Organ 9 phase 3)** — apprentice
  now learns CHAINS, not just single actions. Multi-step traces of
  the same shape produce a chain of substituted action templates;
  ApprenticeProposer emits each step as a candidate; the agent
  loop's pre-flight picks whichever step isn't yet a no-op. Real
  IMPROVES across multi-action workflows.

- **Apprentice (Organ 8 v0)** — template-based proposer that fires
  for novel tasks whose shape matches learned traces. Three training
  examples of 'Add an import of X to Y' → infer the template →
  arbitrary new (X, Y) get a typed Action via pure substitution. No
  LLM call.

- **STEER (Organ 4 v0)** — importance-weighted recall + priming
  block at the prompt head. EVOLVE-boosted atoms sit in the LLM's
  most-attended prompt position.

- **POPULATION (Organ 6 v0)** — composite-scored candidates
  (`conf + lines_added + (1.0 if not no-op)`). Replaces the prior
  'first non-no-op wins' with proper beam scoring.

- **Benchmark expansion** — pattern tier grew from 8 to 15 tasks
  covering: imports (plain/from/alias), rename cross-file,
  add-parameter, add-field, add-decorator (3 variants), multi-step
  decompose, wrap-in-try, insert-at-start/end-of-function,
  delete function / method-of-class.

- **In-process benchmark mode + `--repeat N`** — shares the LLM
  model across tasks (~10× faster than subprocess) and produces real
  pass-rate measurements per task.

### Fixed

- Confidence-field default (0.5) — small models routinely forget the
  field; missing-required-field validations were a real failure mode.
- Quoted-default capture in the AddField / AddParameter patterns —
  `default "1.0"` now preserves the string literal instead of
  truncating to `1.0` (the number).

### Scoreboard (last live run on a 4-CPU box, no GPU)

  pattern: 15/15
  llm:      2/2 single, 6/6 ×3 (Qwen-0.5B & 1.5B two-stage)
  research: 1/1 (Qwen-1.5B + curl-cffi web fetch)
  total measured: 18/18 single, 30/30 across repeats

### Architecture parity against DESIGN.md (v0 implementations)

  Organ 1  (lattice store)   — partial (atom store + EVOLVE traces)
  Organ 2  (SENSE)            — done (libcst + Semble)
  Organ 3  (ACT)              — 10 mutating verbs compile to diffs
  Organ 4  (STEER)            — v0 lightweight (importance priming)
  Organ 5  (world model)      — partial (preflight scoring)
  Organ 6  (POPULATION)       — v0 (composite scoring beam)
  Organ 7  (VERIFY)            — parse + optional mypy
  Organ 8  (DISTILL)           — v0 (template-substitution apprentice)
  Organ 9  (EVOLVE)            — phases 1-3 (discover, boost, infer)
  Organ 10 (HUMAN INTERFACE)  — CLI + brain inspect/dump/import

Heavyweight versions of Organs 4 and 8 (real activation patching and
trained policy nets) still TODO; their interfaces and v0 paths are
in place so the heavyweight implementations plug in cleanly.

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
