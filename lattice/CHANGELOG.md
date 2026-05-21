# Changelog

All notable changes to LATTICE will live here. Versions follow
semantic versioning.

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
