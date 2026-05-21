"""SENSE — Organ 2.

Typed observations of the codebase: symbols, edges, salience. Replaces
blob-text context with a few hundred structured tokens. Tree-sitter
(here: libcst, Python-only for v0) does the AST extraction.
"""

from lattice.sense.symbol_graph import (
    Symbol,
    SymbolKind,
    extract_symbols,
    find_symbols,
    walk_workspace,
)

__all__ = [
    "Symbol",
    "SymbolKind",
    "extract_symbols",
    "find_symbols",
    "walk_workspace",
]
