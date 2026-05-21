"""Tests for the DeleteSymbol verb (compiler + pattern)."""

from __future__ import annotations

import textwrap

import pytest

from lattice.actions import DeleteSymbol, SymbolRef
from lattice.compiler import DictWorkspace, SymbolNotFound, compile_action
from lattice.propose import task_to_action


def _parses(src: str) -> bool:
    import ast
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


class TestDeleteSymbolCompiler:
    def test_delete_top_level_function(self):
        src = textwrap.dedent("""\
            def keep_me() -> None:
                pass


            def delete_me() -> None:
                pass
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            DeleteSymbol(
                symbol=SymbolRef(file="a.py", name="delete_me"),
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "def keep_me" in out
        assert "def delete_me" not in out

    def test_delete_class(self):
        src = textwrap.dedent("""\
            class KeepMe:
                pass


            class DeleteMe:
                pass
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            DeleteSymbol(
                symbol=SymbolRef(file="a.py", name="DeleteMe"),
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "class KeepMe" in out
        assert "class DeleteMe" not in out

    def test_delete_method_via_dotted_name(self):
        src = textwrap.dedent("""\
            class API:
                def keep(self) -> None:
                    pass

                def delete_me(self) -> None:
                    pass
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            DeleteSymbol(
                symbol=SymbolRef(file="a.py", name="API.delete_me"),
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "def keep" in out
        assert "def delete_me" not in out
        assert "class API:" in out

    def test_delete_only_method_leaves_pass(self):
        """Removing a class's only method must keep the class parseable."""
        src = textwrap.dedent("""\
            class C:
                def only_method(self) -> None:
                    pass
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            DeleteSymbol(
                symbol=SymbolRef(file="a.py", name="C.only_method"),
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "def only_method" not in out
        # Class body collapsed to `pass`.
        assert "class C:" in out
        assert "pass" in out

    def test_missing_symbol_raises_by_default(self):
        ws = DictWorkspace({"a.py": "def f(): pass\n"})
        with pytest.raises(SymbolNotFound):
            compile_action(
                DeleteSymbol(
                    symbol=SymbolRef(file="a.py", name="ghost"),
                    confidence=0.9,
                ),
                ws,
            )

    def test_missing_symbol_no_op_when_require_present_false(self):
        ws = DictWorkspace({"a.py": "def f(): pass\n"})
        result = compile_action(
            DeleteSymbol(
                symbol=SymbolRef(file="a.py", name="ghost"),
                require_present=False,
                confidence=0.9,
            ),
            ws,
        )
        assert result.is_noop

    def test_missing_method_class_present_no_op_when_soft(self):
        ws = DictWorkspace({"a.py": "class C:\n    pass\n"})
        result = compile_action(
            DeleteSymbol(
                symbol=SymbolRef(file="a.py", name="C.ghost"),
                require_present=False,
                confidence=0.9,
            ),
            ws,
        )
        assert result.is_noop

    def test_missing_class_for_method_raises_by_default(self):
        ws = DictWorkspace({"a.py": "def f(): pass\n"})
        with pytest.raises(SymbolNotFound):
            compile_action(
                DeleteSymbol(
                    symbol=SymbolRef(file="a.py", name="GhostClass.method"),
                    confidence=0.9,
                ),
                ws,
            )


class TestDeleteSymbolPattern:
    def test_delete_function(self):
        a = task_to_action("Delete function dead_code in src/util.py")
        assert isinstance(a, DeleteSymbol)
        assert a.symbol.name == "dead_code"
        assert a.symbol.file == "src/util.py"

    def test_remove_class(self):
        a = task_to_action("Remove class LegacyConfig in src/cfg.py")
        assert isinstance(a, DeleteSymbol)
        assert a.symbol.name == "LegacyConfig"

    def test_delete_method_of_class(self):
        a = task_to_action(
            "Delete method old_charge of class Billing from src/billing.py"
        )
        assert isinstance(a, DeleteSymbol)
        assert a.symbol.name == "Billing.old_charge"
        assert a.symbol.file == "src/billing.py"

    def test_drop_function(self):
        a = task_to_action("Drop function unused in src/x.py")
        assert isinstance(a, DeleteSymbol)
        assert a.symbol.name == "unused"

    def test_unrelated_returns_none(self):
        assert task_to_action("Something else entirely") is None
