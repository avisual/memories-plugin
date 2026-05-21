"""Tests for ChangeReturnType, ModifyDocstring, and MoveSymbol."""

from __future__ import annotations

import textwrap

import pytest

from lattice.actions import (
    ChangeReturnType,
    FileRef,
    ModifyDocstring,
    MoveSymbol,
    SymbolRef,
    TypeExpr,
)
from lattice.compiler import (
    CompileError,
    DictWorkspace,
    SymbolNotFound,
    UnsupportedAction,
    compile_action,
)


def _parses(src: str) -> bool:
    import ast
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


class TestChangeReturnType:
    def test_adds_annotation_when_missing(self):
        src = textwrap.dedent("""\
            def f(x):
                return x
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ChangeReturnType(
                symbol=SymbolRef(file="a.py", name="f"),
                return_type=TypeExpr(expr="int"),
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "def f(x) -> int:" in out

    def test_replaces_existing_annotation(self):
        src = textwrap.dedent("""\
            def f(x) -> str:
                return str(x)
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ChangeReturnType(
                symbol=SymbolRef(file="a.py", name="f"),
                return_type=TypeExpr(expr="str | None"),
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "def f(x) -> str | None:" in out
        assert "def f(x) -> str:" not in out

    def test_idempotent(self):
        src = "def f(x) -> int:\n    return 1\n"
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ChangeReturnType(
                symbol=SymbolRef(file="a.py", name="f"),
                return_type=TypeExpr(expr="int"),
                confidence=0.9,
            ),
            ws,
        )
        assert result.is_noop

    def test_method_via_dotted_name(self):
        src = textwrap.dedent("""\
            class API:
                def get(self, path):
                    return ""
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ChangeReturnType(
                symbol=SymbolRef(file="a.py", name="API.get"),
                return_type=TypeExpr(expr="dict[str, int]"),
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "def get(self, path) -> dict[str, int]:" in out

    def test_invalid_type_expr_rejected(self):
        ws = DictWorkspace({"a.py": "def f(): pass\n"})
        with pytest.raises(Exception):
            compile_action(
                ChangeReturnType(
                    symbol=SymbolRef(file="a.py", name="f"),
                    return_type=TypeExpr(expr="@@@"),
                    confidence=0.9,
                ),
                ws,
            )


class TestModifyDocstring:
    def test_adds_docstring_to_function(self):
        src = "def f():\n    return 1\n"
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ModifyDocstring(
                file=FileRef(path="a.py"),
                symbol=SymbolRef(file="a.py", name="f"),
                docstring="Compute the answer.",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert '"""Compute the answer."""' in out
        # Docstring sits as the FIRST statement in the function body.
        i_doc = out.index('"""Compute the answer."""')
        i_ret = out.index("return 1")
        assert i_doc < i_ret

    def test_replaces_existing_docstring(self):
        src = textwrap.dedent('''\
            def f():
                """Old."""
                return 1
        ''')
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ModifyDocstring(
                file=FileRef(path="a.py"),
                symbol=SymbolRef(file="a.py", name="f"),
                docstring="New.",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert '"""New."""' in out
        assert '"""Old."""' not in out

    def test_module_docstring_when_symbol_is_none(self):
        src = "import os\n\n\ndef f():\n    return 1\n"
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ModifyDocstring(
                file=FileRef(path="a.py"),
                docstring="Module-level doc.",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert out.lstrip().startswith('"""Module-level doc."""')

    def test_module_docstring_replaces_existing(self):
        src = textwrap.dedent('''\
            """Old module doc."""

            import os
        ''')
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ModifyDocstring(
                file=FileRef(path="a.py"),
                docstring="Fresh module doc.",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert '"""Fresh module doc."""' in out
        assert '"""Old module doc."""' not in out

    def test_idempotent(self):
        src = textwrap.dedent('''\
            def f():
                """Hello."""
                return 1
        ''')
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ModifyDocstring(
                file=FileRef(path="a.py"),
                symbol=SymbolRef(file="a.py", name="f"),
                docstring="Hello.",
                confidence=0.9,
            ),
            ws,
        )
        assert result.is_noop

    def test_adds_to_class(self):
        src = "class C:\n    pass\n"
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ModifyDocstring(
                file=FileRef(path="a.py"),
                symbol=SymbolRef(file="a.py", name="C"),
                docstring="A class.",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert '"""A class."""' in out


class TestMoveSymbol:
    def test_moves_top_level_function(self):
        ws = DictWorkspace(
            {
                "src/old.py": "def keep():\n    pass\n\n\ndef movee():\n    return 42\n",
                "src/new.py": "",
            }
        )
        result = compile_action(
            MoveSymbol(
                symbol=SymbolRef(file="src/old.py", name="movee"),
                target_file=FileRef(path="src/new.py"),
                position="end",
                confidence=0.9,
            ),
            ws,
        )
        old_after = next(c.after for c in result.file_changes if c.path == "src/old.py")
        new_after = next(c.after for c in result.file_changes if c.path == "src/new.py")
        assert _parses(old_after)
        assert _parses(new_after)
        assert "def keep" in old_after
        assert "def movee" not in old_after
        assert "def movee" in new_after
        assert "return 42" in new_after

    def test_moves_class_with_decorators(self):
        ws = DictWorkspace(
            {
                "src/old.py": textwrap.dedent("""\
                    from dataclasses import dataclass


                    @dataclass
                    class Movee:
                        x: int = 0
                """),
                "src/new.py": "",
            }
        )
        result = compile_action(
            MoveSymbol(
                symbol=SymbolRef(file="src/old.py", name="Movee"),
                target_file=FileRef(path="src/new.py"),
                confidence=0.9,
            ),
            ws,
        )
        new_after = next(c.after for c in result.file_changes if c.path == "src/new.py")
        assert "@dataclass" in new_after
        assert "class Movee" in new_after

    def test_creates_target_file_if_missing(self):
        ws = DictWorkspace({"src/old.py": "def movee():\n    pass\n"})
        result = compile_action(
            MoveSymbol(
                symbol=SymbolRef(file="src/old.py", name="movee"),
                target_file=FileRef(path="src/brand_new.py"),
                confidence=0.9,
            ),
            ws,
        )
        new_after = next(c.after for c in result.file_changes if c.path == "src/brand_new.py")
        assert "def movee" in new_after

    def test_missing_symbol_raises(self):
        ws = DictWorkspace({"src/old.py": "def keep(): pass\n", "src/new.py": ""})
        with pytest.raises(SymbolNotFound):
            compile_action(
                MoveSymbol(
                    symbol=SymbolRef(file="src/old.py", name="ghost"),
                    target_file=FileRef(path="src/new.py"),
                    confidence=0.9,
                ),
                ws,
            )

    def test_method_dotted_name_rejected(self):
        ws = DictWorkspace(
            {"src/old.py": "class C:\n    def m(self): pass\n", "src/new.py": ""}
        )
        with pytest.raises(UnsupportedAction):
            compile_action(
                MoveSymbol(
                    symbol=SymbolRef(file="src/old.py", name="C.m"),
                    target_file=FileRef(path="src/new.py"),
                    confidence=0.9,
                ),
                ws,
            )

    def test_same_file_rejected(self):
        ws = DictWorkspace({"src/a.py": "def f(): pass\n"})
        with pytest.raises(CompileError):
            compile_action(
                MoveSymbol(
                    symbol=SymbolRef(file="src/a.py", name="f"),
                    target_file=FileRef(path="src/a.py"),
                    confidence=0.9,
                ),
                ws,
            )
