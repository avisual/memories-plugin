"""Tests for the symbol-graph extractor."""

from __future__ import annotations

import textwrap

from lattice.compiler.workspace import DictWorkspace
from lattice.sense import (
    Symbol,
    SymbolKind,
    extract_symbols,
    find_symbols,
    walk_workspace,
)


def test_extracts_module_function():
    src = "def f():\n    pass\n"
    syms = extract_symbols("a.py", src)
    assert syms == [Symbol(file="a.py", name="f", kind=SymbolKind.FUNCTION, line=1)]


def test_extracts_class_and_method():
    src = textwrap.dedent("""\
        class C:
            def m(self):
                pass
    """)
    syms = extract_symbols("a.py", src)
    names = {s.name: s.kind for s in syms}
    assert names == {"C": SymbolKind.CLASS, "C.m": SymbolKind.METHOD}


def test_extracts_nested_class():
    src = textwrap.dedent("""\
        class Outer:
            class Inner:
                def m(self):
                    pass
    """)
    syms = extract_symbols("a.py", src)
    names = {s.name for s in syms}
    assert names == {"Outer", "Outer.Inner", "Outer.Inner.m"}


def test_lines_are_1_indexed():
    src = "\n\ndef f():\n    pass\n"
    syms = extract_symbols("a.py", src)
    assert syms[0].line == 3


def test_find_symbols_by_leaf_name():
    src = textwrap.dedent("""\
        def charge():
            pass

        class A:
            def charge(self):
                pass

        class B:
            def refund(self):
                pass
    """)
    syms = extract_symbols("a.py", src)
    matches = find_symbols(syms, leaf_name="charge")
    assert {s.name for s in matches} == {"charge", "A.charge"}


def test_find_symbols_by_kind():
    src = textwrap.dedent("""\
        def f():
            pass

        class C:
            def m(self):
                pass
    """)
    syms = extract_symbols("a.py", src)
    only_methods = find_symbols(syms, kinds=(SymbolKind.METHOD,))
    assert [s.name for s in only_methods] == ["C.m"]


def test_walk_workspace_across_files():
    ws = DictWorkspace(
        {
            "a.py": "def charge():\n    pass\n",
            "b.py": "class B:\n    def charge(self):\n        pass\n",
        }
    )
    syms = walk_workspace(ws, ["a.py", "b.py"])
    files_for_charge = {s.file for s in find_symbols(syms, leaf_name="charge")}
    assert files_for_charge == {"a.py", "b.py"}


def test_walk_workspace_skips_unparseable():
    ws = DictWorkspace(
        {
            "ok.py": "def f():\n    pass\n",
            "bad.py": "def f(:\n",  # invalid syntax
        }
    )
    syms = walk_workspace(ws, ["ok.py", "bad.py"])
    # Should still get f from ok.py and not crash.
    assert any(s.name == "f" and s.file == "ok.py" for s in syms)
