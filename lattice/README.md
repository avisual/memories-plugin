# LATTICE

A compositional, self-improving reasoning substrate for small coding LLMs.

> An organism, not a prompt. Senses, muscles, memory, dreams — and small
> typed primitives that grow into bigger forms with use.

LATTICE is a coding harness for non-frontier LLMs. It replaces blob-text
context and free-form code generation with:

- **Typed observations** of a code subgraph instead of file dumps.
- **A typed action DSL** as the LLM's only code-modifying output —
  invalid syntax, undefined symbols, and broken imports are deleted by
  construction.
- **Activation steering** from a Hebbian memory graph — recalled atoms
  bias the model's hidden state instead of consuming context tokens.
- **A world model** that simulates the consequence of an action before
  it ever touches a file.
- **A distilled apprentice** trained continuously from successful
  traces — over time, most decisions stop needing the LLM at all.
- **A DSL that grows itself**: macros → recipes → idioms emerge from
  recurring action patterns by the same Hebbian rule that links memory
  atoms.

Built on the [`memories-plugin`](../) Hebbian graph as substrate.
The full design is in [DESIGN.md](DESIGN.md).

## Status

v0 working end-to-end through the composition spine: one intent
expands into many typed actions, every action compiles to a verified
diff. What runs today:

- **Action DSL** (Organ 3): 10 typed verbs, Pydantic-validated,
  discriminated-union parser, JSON-schema export for constrained
  decoding.
- **Action compiler** (Python, libcst): `AddImport`, `AddField`,
  `AddParameter` produce real file diffs. Idempotent on already-
  present state. `WrapInTry`, `AddTest`, `RenameSymbol` raise
  `UnsupportedAction` until they land.
- **Symbol-graph extractor** (Organ 2, Python): walks files via
  libcst, yields typed `Symbol{file, name, kind, line}` records.
- **Orchestrator**: takes a high-level `Intent`, expands it into many
  typed `Action`s, runs each through compile + verify, returns an
  `ExecutionReport` bundling diffs and outcomes.
- **Syntactic verify** (Organ 7, partial): every compiled diff is
  parsed with `ast.parse`.
- **CLI**: `python -m lattice apply` (single action) and
  `python -m lattice intent` (multi-action expansion).

Not yet: world model, steering, lattice store, apprentice, evolution,
interface surfaces beyond the CLI.

## Try it

Install:

```bash
cd lattice
uv pip install -e ".[dev]"
```

### Single action

```bash
mkdir -p /tmp/demo
cat > /tmp/demo/charge.py <<'PY'
"""Charge a customer's card."""
import os

class ChargeProcessor:
    def charge(self, amount: int) -> None:
        pass
PY

echo '{"verb":"AddImport","file":{"path":"charge.py"},"module":"stripe","confidence":0.9}' \
  | uv run python -m lattice apply /tmp/demo --action -
```

```diff
--- a/charge.py
+++ b/charge.py
@@ -1,5 +1,6 @@
 """Charge a customer's card."""
 import os
+import stripe
```

### Multi-file intent (composition)

One Intent expands into N typed Actions across the whole workspace:

```bash
echo '{
  "kind": "AddParameterToAllMatching",
  "function_name": "charge",
  "parameter_name": "dry_run",
  "parameter_type": "bool",
  "parameter_default": "False",
  "keyword_only": true
}' | uv run python -m lattice intent /tmp/demo --intent -
```

The orchestrator walks the workspace, finds every `charge()` (function
or method), emits one `AddParameter` action per match, compiles each
through libcst, and verifies the output parses. Output is a sequence
of unified diffs across all touched files. Nothing is written to disk
— the diff is the deliverable.

## Layout

```
lattice/
├── DESIGN.md                 # the full architecture & rationale
├── pyproject.toml
├── src/lattice/
│   ├── actions/              # ACT: typed action DSL  (Organ 3)
│   ├── store/                # Lattice store          (Organ 1)
│   ├── sense/                # Typed observations     (Organ 2)
│   ├── steer/                # Activation steering    (Organ 4)
│   ├── world_model/          # Imagination            (Organ 5)
│   ├── population/           # Evolutionary search    (Organ 6)
│   ├── verify/               # Silent ground truth    (Organ 7)
│   ├── learn/                # Hebbian updates        (LEARN)
│   ├── distill/              # The apprentice         (Organ 8)
│   ├── evolve/               # DSL growth             (Organ 9)
│   └── interface/            # Human surfaces         (Organ 10)
└── tests/
```
