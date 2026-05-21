"""SENSE — Organ 2.

Typed observations of the codebase: symbols, edges, salience. Replaces
blob-text context with a few hundred structured tokens.

- symbol_graph (libcst, Python-only for v0): the workspace's Symbols
  (functions, classes, methods).
- semble_search (optional [search] extra): fast Model2Vec+BM25 code
  retrieval, ranked chunks per task — focused observation for big
  codebases.
"""

from lattice.sense.semble_search import CodeChunk, SembleCodeSearch, maybe_code_search
from lattice.sense.symbol_graph import (
    Symbol,
    SymbolKind,
    extract_symbols,
    find_symbols,
    walk_workspace,
)

__all__ = [
    "CodeChunk",
    "SembleCodeSearch",
    "Symbol",
    "SymbolKind",
    "extract_symbols",
    "find_symbols",
    "maybe_code_search",
    "walk_workspace",
]
