# LATTICE

**A coding harness that makes small LLMs reliable.**

LATTICE is built on a single observation: most "code changes a coding
agent needs to make" are structural mappings — adding an import,
renaming a symbol, adding a parameter. A regex can do those correctly;
a 500M-parameter model gets them wrong. Reserve the LLM for the
genuinely open-ended cases. Give the LLM a typed action vocabulary so
it can't emit syntactically invalid code. Give it a brain so it knows
your codebase's conventions. Verify every output before it touches a
file.

Built on the [`memories-plugin`](../) Hebbian graph as inspiration;
ships its own atom store so no external service is required.

## What it actually does (live, today, on a 4-CPU box, no GPU)

```bash
$ lattice do "Add an import of request from flask to src/myapp/handlers.py"
+from flask import request

$ lattice do "Add a keyword-only parameter timeout of type float to function get of class Client in src/api.py"
-    def get(self, url: str) -> str:
+    def get(self, url: str, *, timeout: float = None) -> str:

$ lattice do "Rename charge to take_payment in src/billing.py"
# Renames the definition in billing.py AND every reference across
# every .py file in the project (3 files touched, 2 changed).
```

Each command:
1. Auto-initialises a brain (96 seed atoms) on first run.
2. Tries pattern-matching first — these tasks need zero LLM calls.
3. Falls back to a small local LLM (Qwen2.5-0.5B-Instruct, ~500MB)
   for tasks the patterns don't cover.
4. Compiles the action to a real file diff via libcst.
5. Verifies the result parses (and optionally type-checks with mypy).
6. Writes the file. Records what it did as an experience atom.

## Install + try it

```bash
cd lattice
uv venv --python 3.13
uv pip install -e '.[llm]'        # ~1GB: torch + transformers + MiniLM
source .venv/bin/activate
lattice init                       # in your project root: seeds .lattice/brain.db
lattice do "your task here"
```

Optional extras:
- `'.[hosted]'` for the Anthropic API path (requires `ANTHROPIC_API_KEY`).
- `'.[typecheck]'` to add mypy verify (`--types` flag).

## CLI

| Command | What it does |
|---|---|
| `lattice do "task"` | One-shot: init if needed, run agent, write file. |
| `lattice init [dir]` | Initialise a brain with 96 seed atoms. |
| `lattice agent` | Multi-step agent with full control over model / brain / steps. |
| `lattice intent` | Expand a high-level Intent into many typed actions (no LLM). |
| `lattice apply` | Apply a single hand-written Action JSON. |
| `lattice brain inspect/dump/import/export` | Curate the atom store. |
| `lattice atom add/recall/seed` | Lower-level brain operations. |

## What's reliable, what's not (honest)

| | What works | What doesn't yet |
|---|---|---|
| **Verbs** | All 5 mutating verbs compile (AddImport, AddField, AddParameter, WrapInTry, AddTest) plus single-file & cross-file RenameSymbol | Aliased imports (`from x import y as z`) not yet resolved during rename |
| **Patterns** | Imports (plain/from/alias), rename, add parameter, add field — deterministic, no LLM | WrapInTry, AddTest patterns not yet added |
| **Small LLM (0.5–1.5B)** | Tasks within the pattern vocabulary; single-slot fills the LLM handles well (e.g. AddParameter on a known method) | Multi-step planning — model hallucinates after first action. Mitigated by stuck-detection and the planner-executor split. |
| **Hosted LLM (Anthropic)** | Architecturally complete: tool-spec generation, retry, validation. Untested without an API key. | Live API call not verified in this environment. |
| **Verify** | `ast.parse` always on; `mypy` opt-in via `--types` | Test-run gate (pytest on affected tests) — not built yet |
| **Cross-language** | Python only | TS / JS / Rust / Go — not built |
| **Brain** | 96 seed atoms, semantic recall via MiniLM, Hebbian feedback writes experience atoms on success, curatable via JSON | Activation steering, apprentice (Organ 8), macro evolution — design exists, not built |

## The design (one paragraph)

The harness is the fetch-decode-execute cycle; the LLM is the
instruction emitter. The LLM never plans — it answers "given this
observable state, what's the next single instruction?" each cycle.
The harness manages the program counter, advances when the next
instruction would be a no-op (the goal is observably met), and
detects when the LLM is stuck repeating itself. A brain of atoms,
recalled by semantic similarity, becomes hints in the observation
so the LLM benefits from project knowledge without needing to
remember it. A pattern proposer handles common natural-language
phrasings deterministically before the LLM ever gets called.
**This is what makes a small model usable for real work.**

The full architecture (10 organs, the compositional spine, the
build sketch) is in [DESIGN.md](DESIGN.md).

## Tests

```bash
uv run pytest tests/ -q -k "not local_llm and not minilm"
# 200+ passing tests in ~7 seconds
```

Two tests are gated on real model downloads, enabled by setting
`LATTICE_LLM_SMOKE=1`.

## Layout

```
lattice/
├── DESIGN.md                          # full architecture & rationale
├── examples/                          # runnable shell scripts
├── pyproject.toml
├── src/lattice/
│   ├── actions/                       # ACT: typed action DSL  (Organ 3)
│   ├── atoms/                         # the brain                (Organ 1)
│   ├── compiler/                      # action → diff
│   ├── orchestrator/                  # agent loop + planner    (Organ 6)
│   ├── propose/                       # pattern + LLM + hosted   (Organ 4)
│   ├── sense/                         # symbol graph            (Organ 2)
│   ├── verify/                        # syntactic + type-check  (Organ 7)
│   └── apply/                         # write-to-disk
└── tests/
```
