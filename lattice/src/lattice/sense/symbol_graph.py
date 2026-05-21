"""Symbol-graph extraction — walks Python source into typed Symbols.

A Symbol is the unit the lattice store will eventually hold as a node;
for now we just extract them so the orchestrator can find call sites,
methods, and classes by name and predicate without re-parsing.

This is deliberately tiny — libcst already does the heavy lifting.
A future iteration adds typed edges (calls, mutates, instantiates)
and stores both nodes and edges in the lattice store (Organ 1).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable

import libcst as cst
from pydantic import BaseModel, ConfigDict, Field

from lattice.compiler.workspace import Workspace


class SymbolKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"


class Symbol(BaseModel):
    model_config = ConfigDict(frozen=True)
    file: str
    name: str = Field(description="Dotted name within the file, e.g. 'Outer.Inner.method'.")
    kind: SymbolKind
    line: int = Field(ge=1, description="1-indexed line of the def/class statement.")


def extract_symbols(file: str, source: str) -> list[Symbol]:
    """Return a flat list of Symbols defined in *source*.

    Methods inside classes carry the dotted path 'ClassName.method'.
    Nested classes get dotted paths too: 'Outer.Inner'. Module-level
    constants are out of scope for v0 (kept simple; add later).
    """
    module = cst.parse_module(source)
    wrapper = cst.MetadataWrapper(module)
    positions = wrapper.resolve(cst.metadata.PositionProvider)
    symbols: list[Symbol] = []
    _walk(wrapper.module.body, prefix=(), file=file, positions=positions, out=symbols)
    return symbols


def _walk(
    statements: Iterable[cst.BaseStatement],
    *,
    prefix: tuple[str, ...],
    file: str,
    positions: dict,
    out: list[Symbol],
) -> None:
    for stmt in statements:
        if isinstance(stmt, cst.ClassDef):
            name = stmt.name.value
            dotted = ".".join((*prefix, name))
            out.append(
                Symbol(
                    file=file,
                    name=dotted,
                    kind=SymbolKind.CLASS,
                    line=positions[stmt].start.line,
                )
            )
            if isinstance(stmt.body, cst.IndentedBlock):
                _walk(
                    stmt.body.body,
                    prefix=(*prefix, name),
                    file=file,
                    positions=positions,
                    out=out,
                )
        elif isinstance(stmt, cst.FunctionDef):
            name = stmt.name.value
            dotted = ".".join((*prefix, name))
            kind = SymbolKind.METHOD if prefix else SymbolKind.FUNCTION
            out.append(
                Symbol(
                    file=file,
                    name=dotted,
                    kind=kind,
                    line=positions[stmt].start.line,
                )
            )


def walk_workspace(workspace: Workspace, files: Iterable[str]) -> list[Symbol]:
    """Extract symbols from every file in *files* via *workspace*.

    Files that can't be parsed are skipped (callers can introspect by
    pre-checking with workspace.exists / their own parse).
    """
    out: list[Symbol] = []
    for path in files:
        if not workspace.exists(path):
            continue
        try:
            src = workspace.read(path)
            out.extend(extract_symbols(path, src))
        except cst.ParserSyntaxError:
            continue
    return out


def find_symbols(
    symbols: Iterable[Symbol],
    *,
    leaf_name: str | None = None,
    kinds: tuple[SymbolKind, ...] | None = None,
) -> list[Symbol]:
    """Filter *symbols* by leaf-name (last dotted component) and kind."""
    out: list[Symbol] = []
    for sym in symbols:
        leaf = sym.name.rsplit(".", 1)[-1]
        if leaf_name is not None and leaf != leaf_name:
            continue
        if kinds is not None and sym.kind not in kinds:
            continue
        out.append(sym)
    return out
