"""ATOMS — lattice's own minimal memory store.

A small SQLite-backed atom store with embedding-based recall. Built so
LATTICE has a learning surface without depending on external services
(no Ollama, no embedding API). Same conceptual model as
memories-plugin — atoms, types, importance, regions — but tailored
for the lattice loop and runnable on the CPU you already have.

The store is intentionally small (~300 lines). Once memories-plugin
runs in this environment we add a `MemoriesAtomStore` adapter behind
the same Protocol; nothing in the orchestrator changes.
"""

from lattice.atoms.atom import Atom, AtomType
from lattice.atoms.embedder import Embedder, MiniLMEmbedder
from lattice.atoms.feedback import (
    record_antipattern,
    record_experience,
    summarize_trace_for_experience,
)
from lattice.atoms.io import export_atoms, import_atoms, stats
from lattice.atoms.seed import SEED_ATOMS, SeedAtom, seed_store
from lattice.atoms.store import AtomStore, RecallResult, SQLiteAtomStore

__all__ = [
    "Atom",
    "AtomStore",
    "AtomType",
    "Embedder",
    "MiniLMEmbedder",
    "RecallResult",
    "SEED_ATOMS",
    "SQLiteAtomStore",
    "SeedAtom",
    "export_atoms",
    "import_atoms",
    "record_antipattern",
    "record_experience",
    "seed_store",
    "stats",
    "summarize_trace_for_experience",
]
