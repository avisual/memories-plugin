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

v0 working end-to-end on the action → diff path (Months 1–2 of the
build sketch). What runs today:

- **Action DSL** (Organ 3): 10 typed verbs, Pydantic-validated,
  discriminated-union parser, JSON-schema export for constrained
  decoding.
- **Action compiler** (Python, libcst): `AddImport`, `AddField`,
  `AddParameter` produce real file diffs. Idempotent (no-op on
  already-present state). `WrapInTry`, `AddTest`, `RenameSymbol`
  raise `UnsupportedAction` until they land.
- **Syntactic verify** (Organ 7, partial): every compiled diff is
  parsed with `ast.parse` before being printed.
- **CLI**: `python -m lattice apply <workspace> --action <json>` takes
  a typed action as JSON and emits a unified diff on stdout.

Not yet: world model, steering, lattice store, apprentice, evolution,
interface surfaces beyond the CLI.

## Try it

```bash
cd lattice
uv pip install -e ".[dev]"

mkdir -p /tmp/latticedemo
cat > /tmp/latticedemo/charge.py <<'PY'
"""Charge a customer's card."""
import os

class ChargeProcessor:
    api_key: str = ""

    def charge(self, amount: int) -> None:
        pass
PY

echo '{"verb":"AddImport","file":{"path":"charge.py"},"module":"stripe","confidence":0.9}' \
  | uv run python -m lattice apply /tmp/latticedemo --action -
```

Output:

```diff
--- a/charge.py
+++ b/charge.py
@@ -1,5 +1,6 @@
 """Charge a customer's card."""
 import os
+import stripe

 class ChargeProcessor:
```

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
