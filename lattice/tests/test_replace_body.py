"""Tests for the ReplaceBody verb (compiler + pattern)."""

from __future__ import annotations

import textwrap

import pytest

from lattice.actions import ReplaceBody, SymbolRef
from lattice.compiler import CompileError, DictWorkspace, SymbolNotFound, compile_action
from lattice.propose import task_to_action


def _parses(src: str) -> bool:
    import ast
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


class TestReplaceBodyCompiler:
    def test_replaces_simple_body(self):
        src = textwrap.dedent("""\
            def charge(amount: int) -> int:
                return amount
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ReplaceBody(
                symbol=SymbolRef(file="a.py", name="charge"),
                body="amount = max(0, amount)\nreturn amount * 100",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        # New body is in.
        assert "amount = max(0, amount)" in out
        assert "return amount * 100" in out
        # Signature preserved.
        assert "def charge(amount: int) -> int:" in out
        # Old single-statement body gone.
        assert "    return amount\n" not in out

    def test_preserves_decorators(self):
        src = textwrap.dedent("""\
            @cached
            @app.route("/health")
            def health() -> dict:
                return {"ok": False}
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ReplaceBody(
                symbol=SymbolRef(file="a.py", name="health"),
                body='return {"ok": True}',
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "@cached" in out
        assert '@app.route("/health")' in out
        assert 'return {"ok": True}' in out

    def test_method_via_dotted(self):
        src = textwrap.dedent("""\
            class C:
                def go(self, x: int) -> int:
                    return -1
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ReplaceBody(
                symbol=SymbolRef(file="a.py", name="C.go"),
                body="if x < 0:\n    return 0\nreturn x",
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "if x < 0:" in out
        assert "return x\n" in out

    def test_idempotent_on_same_body(self):
        src = textwrap.dedent("""\
            def f():
                return 1
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            ReplaceBody(
                symbol=SymbolRef(file="a.py", name="f"),
                body="return 1",
                confidence=0.9,
            ),
            ws,
        )
        assert result.is_noop

    def test_invalid_body_rejected(self):
        ws = DictWorkspace({"a.py": "def f():\n    return 1\n"})
        with pytest.raises(CompileError):
            compile_action(
                ReplaceBody(
                    symbol=SymbolRef(file="a.py", name="f"),
                    body="@@ not python @@",
                    confidence=0.9,
                ),
                ws,
            )

    def test_missing_symbol_raises(self):
        ws = DictWorkspace({"a.py": "def f():\n    return 1\n"})
        with pytest.raises(SymbolNotFound):
            compile_action(
                ReplaceBody(
                    symbol=SymbolRef(file="a.py", name="ghost"),
                    body="return 0",
                    confidence=0.9,
                ),
                ws,
            )


class TestReplaceBodyPattern:
    def test_replace_body_phrasing(self):
        a = task_to_action(
            "Replace the body of function charge in src/billing.py with `return 0`"
        )
        assert isinstance(a, ReplaceBody)
        assert a.symbol.name == "charge"
        assert a.symbol.file == "src/billing.py"
        assert a.body == "return 0"

    def test_rewrite_phrasing(self):
        a = task_to_action(
            "Rewrite function go of class Worker in src/w.py to `return None`"
        )
        assert isinstance(a, ReplaceBody)
        assert a.symbol.name == "Worker.go"

    def test_unrelated_returns_none(self):
        assert task_to_action("Something else entirely") is None
