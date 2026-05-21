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

v0 scaffolding — Months 1–2 of the build sketch in DESIGN.md.
Currently in this package:

- The typed action DSL (10 starter verbs, Pydantic-validated)
- Module skeletons for the remaining nine organs

Nothing wires to an LLM yet. That comes next.

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
