"""Tests for the AddDecorator verb (compiler + pattern)."""

from __future__ import annotations

import textwrap

import pytest

from lattice.actions import AddDecorator, SymbolRef
from lattice.compiler import DictWorkspace, SymbolNotFound, compile_action
from lattice.propose import task_to_action


def _parses(src: str) -> bool:
    import ast
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


class TestAddDecoratorCompiler:
    def test_adds_simple_decorator(self):
        src = textwrap.dedent("""\
            def health() -> dict:
                return {"ok": True}
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            AddDecorator(
                symbol=SymbolRef(file="a.py", name="health"),
                decorator='app.route("/health")',
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert '@app.route("/health")' in out
        assert "def health() -> dict:" in out

    def test_adds_to_class(self):
        src = textwrap.dedent("""\
            class Config:
                debug: bool = False
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            AddDecorator(
                symbol=SymbolRef(file="a.py", name="Config"),
                decorator="dataclass(frozen=True)",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "@dataclass(frozen=True)" in out
        assert "class Config:" in out

    def test_outermost_position_stacks_above_existing(self):
        src = textwrap.dedent("""\
            @existing
            def f():
                pass
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            AddDecorator(
                symbol=SymbolRef(file="a.py", name="f"),
                decorator="newer",
                position="outermost",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        # @newer must appear ABOVE @existing.
        i_newer = lines.index("@newer")
        i_existing = lines.index("@existing")
        assert i_newer < i_existing

    def test_innermost_position_sits_below_existing(self):
        src = textwrap.dedent("""\
            @existing
            def f():
                pass
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            AddDecorator(
                symbol=SymbolRef(file="a.py", name="f"),
                decorator="newer",
                position="innermost",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        i_newer = lines.index("@newer")
        i_existing = lines.index("@existing")
        assert i_existing < i_newer

    def test_idempotent_when_decorator_already_present(self):
        src = textwrap.dedent("""\
            @cached
            def f():
                pass
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            AddDecorator(
                symbol=SymbolRef(file="a.py", name="f"),
                decorator="cached",
                confidence=0.9,
            ),
            ws,
        )
        assert result.is_noop

    def test_method_via_dotted_name(self):
        src = textwrap.dedent("""\
            class API:
                def get(self) -> str:
                    return ""
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            AddDecorator(
                symbol=SymbolRef(file="a.py", name="API.get"),
                decorator="staticmethod",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "@staticmethod" in out

    def test_missing_symbol_raises(self):
        ws = DictWorkspace({"a.py": "x = 1\n"})
        with pytest.raises(SymbolNotFound):
            compile_action(
                AddDecorator(
                    symbol=SymbolRef(file="a.py", name="does_not_exist"),
                    decorator="cached",
                    confidence=0.9,
                ),
                ws,
            )

    def test_invalid_decorator_expression_raises(self):
        ws = DictWorkspace(
            {"a.py": "def f():\n    pass\n"}
        )
        with pytest.raises(Exception):
            compile_action(
                AddDecorator(
                    symbol=SymbolRef(file="a.py", name="f"),
                    decorator="@@bogus",  # not a valid expression
                    confidence=0.9,
                ),
                ws,
            )


class TestAddDecoratorPattern:
    def test_add_X_to_function_Y_in_FILE(self):
        a = task_to_action("Add @cached decorator to function f in src/app.py")
        assert isinstance(a, AddDecorator)
        assert a.symbol.name == "f"
        assert a.symbol.file == "src/app.py"
        assert a.decorator == "cached"

    def test_add_routed_decorator(self):
        a = task_to_action('Add app.route("/health") decorator to function health in src/app.py')
        assert isinstance(a, AddDecorator)
        assert a.decorator.startswith("app.route")
        assert "/health" in a.decorator

    def test_method_of_class(self):
        a = task_to_action(
            "Add @classmethod to function from_dict of class Config in src/c.py"
        )
        assert isinstance(a, AddDecorator)
        assert a.symbol.name == "Config.from_dict"
        assert a.decorator == "classmethod"

    def test_decorate_phrasing(self):
        a = task_to_action(
            "Decorate function f in src/app.py with @cached"
        )
        assert isinstance(a, AddDecorator)
        assert a.decorator == "cached"
        assert a.symbol.name == "f"

    def test_unrelated_task_returns_none(self):
        assert task_to_action("Some other unrelated task") is None
