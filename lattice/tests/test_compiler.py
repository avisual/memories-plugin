"""Tests for the Python action compiler.

The compiler's contract:
- Every CompiledAction.after parses as valid Python.
- The before/after is byte-equal when the action is a no-op.
- SymbolRefs that don't resolve raise SymbolNotFound.
- Non-mutating verbs raise NonMutatingAction (handled by orchestrator).
- Unsupported verbs raise UnsupportedAction (forward-compat fence).
"""

from __future__ import annotations

import ast
import textwrap

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lattice.actions import (
    AddField,
    AddImport,
    AddParameter,
    Branch,
    Expr,
    FileRef,
    MarkBlocked,
    RecallMore,
    RenameSymbol,
    RevealBody,
    SpanRef,
    SymbolRef,
    TypeExpr,
    WrapInTry,
)
from lattice.compiler import (
    DictWorkspace,
    NonMutatingAction,
    SymbolNotFound,
    UnsupportedAction,
    WorkspaceError,
    compile_action,
)


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# AddImport
# ---------------------------------------------------------------------------


class TestAddImport:
    def test_into_empty_file(self):
        ws = DictWorkspace({"a.py": ""})
        result = compile_action(
            AddImport(file=FileRef(path="a.py"), module="json", confidence=0.9), ws
        )
        assert _parses(result.file_changes[0].after)
        assert "import json" in result.file_changes[0].after

    def test_first_import_lands_after_docstring(self):
        src = textwrap.dedent('''\
            """Module docstring."""


            CONST = 1
        ''')
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            AddImport(file=FileRef(path="a.py"), module="json", confidence=0.9), ws
        )
        out = result.file_changes[0].after
        assert _parses(out)
        lines = out.splitlines()
        assert lines[0].strip().startswith('"""')
        json_idx = next(i for i, line in enumerate(lines) if line.strip() == "import json")
        const_idx = next(i for i, line in enumerate(lines) if line.startswith("CONST"))
        assert json_idx > 0
        assert json_idx < const_idx

    def test_after_existing_imports(self):
        src = textwrap.dedent("""\
            import os
            import sys

            CONST = 1
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            AddImport(file=FileRef(path="a.py"), module="json", confidence=0.9), ws
        )
        out = result.file_changes[0].after
        assert _parses(out)
        lines = out.splitlines()
        json_idx = next(i for i, line in enumerate(lines) if line.strip() == "import json")
        const_idx = next(i for i, line in enumerate(lines) if line.startswith("CONST"))
        assert json_idx < const_idx
        # Inserted after existing import block (after sys, before blank+CONST).
        assert lines[json_idx - 1].strip() == "import sys"

    def test_idempotent_exact_match(self):
        src = "import json\n"
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            AddImport(file=FileRef(path="a.py"), module="json", confidence=0.9), ws
        )
        assert result.is_noop
        assert result.file_changes[0].diff == ""

    def test_from_form(self):
        ws = DictWorkspace({"a.py": ""})
        result = compile_action(
            AddImport(
                file=FileRef(path="a.py"),
                module="typing",
                names=["Optional", "Sequence"],
                confidence=0.9,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "from typing import Optional, Sequence" in out

    def test_with_alias(self):
        ws = DictWorkspace({"a.py": ""})
        result = compile_action(
            AddImport(
                file=FileRef(path="a.py"), module="numpy", alias="np", confidence=0.9
            ),
            ws,
        )
        assert "import numpy as np" in result.file_changes[0].after

    def test_dotted_module(self):
        ws = DictWorkspace({"a.py": ""})
        result = compile_action(
            AddImport(file=FileRef(path="a.py"), module="os.path", confidence=0.9), ws
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "import os.path" in out

    def test_missing_file_raises(self):
        ws = DictWorkspace({})
        with pytest.raises(WorkspaceError):
            compile_action(
                AddImport(file=FileRef(path="missing.py"), module="json", confidence=0.5),
                ws,
            )


# ---------------------------------------------------------------------------
# AddField
# ---------------------------------------------------------------------------


class TestAddField:
    def _ws(self, src: str = "") -> DictWorkspace:
        default_src = textwrap.dedent("""\
            class Point:
                x: int = 0

                def shift(self, dx: int) -> None:
                    self.x += dx
        """)
        return DictWorkspace({"a.py": src or default_src})

    def test_into_simple_class(self):
        ws = self._ws()
        result = compile_action(
            AddField(
                cls=SymbolRef(file="a.py", name="Point"),
                name="y",
                type=TypeExpr(expr="int"),
                default=Expr(code="0"),
                confidence=0.8,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "y: int = 0" in out
        # Field added in field-block, before the method.
        idx_y = out.index("y: int = 0")
        idx_method = out.index("def shift")
        assert idx_y < idx_method

    def test_idempotent(self):
        ws = self._ws()
        result = compile_action(
            AddField(
                cls=SymbolRef(file="a.py", name="Point"),
                name="x",
                type=TypeExpr(expr="int"),
                default=Expr(code="0"),
                confidence=0.8,
            ),
            ws,
        )
        assert result.is_noop

    def test_after_docstring(self):
        src = textwrap.dedent('''\
            class Point:
                """A 2D point."""

                def shift(self, dx: int) -> None:
                    self.x += dx
        ''')
        ws = self._ws(src=src)
        result = compile_action(
            AddField(
                cls=SymbolRef(file="a.py", name="Point"),
                name="x",
                type=TypeExpr(expr="int"),
                default=Expr(code="0"),
                confidence=0.8,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "x: int = 0" in out
        # Docstring still first.
        first_real_line = next(
            line for line in out.splitlines() if line.strip() and not line.startswith("class")
        )
        assert first_real_line.strip().startswith('"""')

    def test_missing_class_raises(self):
        ws = self._ws()
        with pytest.raises(SymbolNotFound):
            compile_action(
                AddField(
                    cls=SymbolRef(file="a.py", name="DoesNotExist"),
                    name="x",
                    type=TypeExpr(expr="int"),
                    confidence=0.8,
                ),
                ws,
            )

    def test_nested_class_dotted_lookup(self):
        src = textwrap.dedent("""\
            class Outer:
                class Inner:
                    a: int = 0
        """)
        ws = DictWorkspace({"a.py": src})
        result = compile_action(
            AddField(
                cls=SymbolRef(file="a.py", name="Outer.Inner"),
                name="b",
                type=TypeExpr(expr="str"),
                default=Expr(code='"x"'),
                confidence=0.8,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert 'b: str = "x"' in out


# ---------------------------------------------------------------------------
# AddParameter
# ---------------------------------------------------------------------------


class TestAddParameter:
    def _ws(self) -> DictWorkspace:
        src = textwrap.dedent("""\
            def charge(amount: int) -> None:
                pass


            class Processor:
                def refund(self, charge_id: str) -> None:
                    pass
        """)
        return DictWorkspace({"a.py": src})

    def test_appends_to_function(self):
        ws = self._ws()
        result = compile_action(
            AddParameter(
                function=SymbolRef(file="a.py", name="charge"),
                name="customer_id",
                type=TypeExpr(expr="str"),
                confidence=0.7,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "def charge(amount: int, customer_id: str)" in out

    def test_method_via_dotted_name(self):
        ws = self._ws()
        result = compile_action(
            AddParameter(
                function=SymbolRef(file="a.py", name="Processor.refund"),
                name="partial",
                type=TypeExpr(expr="bool"),
                default=Expr(code="False"),
                confidence=0.7,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "def refund(self, charge_id: str, partial: bool = False)" in out

    def test_keyword_only(self):
        ws = self._ws()
        result = compile_action(
            AddParameter(
                function=SymbolRef(file="a.py", name="charge"),
                name="dry_run",
                type=TypeExpr(expr="bool"),
                default=Expr(code="False"),
                keyword_only=True,
                confidence=0.7,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "def charge(amount: int, *, dry_run: bool = False)" in out

    def test_idempotent(self):
        ws = self._ws()
        result = compile_action(
            AddParameter(
                function=SymbolRef(file="a.py", name="charge"),
                name="amount",
                type=TypeExpr(expr="int"),
                confidence=0.7,
            ),
            ws,
        )
        assert result.is_noop

    def test_missing_function_raises(self):
        ws = self._ws()
        with pytest.raises(SymbolNotFound):
            compile_action(
                AddParameter(
                    function=SymbolRef(file="a.py", name="does_not_exist"),
                    name="x",
                    type=TypeExpr(expr="int"),
                    confidence=0.7,
                ),
                ws,
            )

    def test_position_zero_prepends(self):
        ws = self._ws()
        result = compile_action(
            AddParameter(
                function=SymbolRef(file="a.py", name="charge"),
                name="ctx",
                type=TypeExpr(expr="Context"),
                position=0,
                confidence=0.7,
            ),
            ws,
        )
        out = result.file_changes[0].after
        assert _parses(out)
        assert "def charge(ctx: Context, amount: int)" in out


# ---------------------------------------------------------------------------
# Dispatch / error routing
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_recall_more_is_non_mutating(self):
        with pytest.raises(NonMutatingAction):
            compile_action(RecallMore(query="x", confidence=0.5), DictWorkspace())

    def test_reveal_body_is_non_mutating(self):
        with pytest.raises(NonMutatingAction):
            compile_action(
                RevealBody(symbol=SymbolRef(file="a.py", name="f"), confidence=0.5),
                DictWorkspace({"a.py": ""}),
            )

    def test_branch_is_non_mutating(self):
        with pytest.raises(NonMutatingAction):
            compile_action(Branch(rationale="r", confidence=0.5), DictWorkspace())

    def test_mark_blocked_is_non_mutating(self):
        with pytest.raises(NonMutatingAction):
            compile_action(
                MarkBlocked(reason_code="needs_human", detail="x", confidence=0.5),
                DictWorkspace(),
            )

    def test_wrap_in_try_unsupported(self):
        with pytest.raises(UnsupportedAction):
            compile_action(
                WrapInTry(
                    span=SpanRef(file="a.py", start_line=1, end_line=2),
                    exception_type=TypeExpr(expr="Exception"),
                    confidence=0.5,
                ),
                DictWorkspace({"a.py": "pass\npass\n"}),
            )

    def test_rename_symbol_unsupported(self):
        with pytest.raises(UnsupportedAction):
            compile_action(
                RenameSymbol(
                    symbol=SymbolRef(file="a.py", name="x"),
                    new_name="y",
                    confidence=0.5,
                ),
                DictWorkspace({"a.py": ""}),
            )


# ---------------------------------------------------------------------------
# Property-based: anything we compile is valid Python.
# ---------------------------------------------------------------------------


_MODULE_NAMES = st.from_regex(r"\A[a-z_][a-z0-9_]{0,12}(\.[a-z_][a-z0-9_]{0,12}){0,2}\Z", fullmatch=True)
_IDENTIFIERS = st.from_regex(r"\A[a-z_][a-z0-9_]{0,12}\Z", fullmatch=True).filter(
    lambda s: s
    not in {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
        "match",
        "case",
    }
)


@given(module=_MODULE_NAMES, alias=st.one_of(st.none(), _IDENTIFIERS))
@settings(max_examples=50, deadline=None)
def test_property_add_import_always_parses(module: str, alias: str | None) -> None:
    ws = DictWorkspace({"a.py": "x = 1\n"})
    action = AddImport(
        file=FileRef(path="a.py"), module=module, alias=alias, confidence=0.5
    )
    result = compile_action(action, ws)
    assert _parses(result.file_changes[0].after)


@given(
    field_name=_IDENTIFIERS,
    type_expr=st.sampled_from(["int", "str", "bool", "list[int]", "dict[str, int]"]),
)
@settings(max_examples=50, deadline=None)
def test_property_add_field_always_parses(field_name: str, type_expr: str) -> None:
    src = textwrap.dedent("""\
        class C:
            existing: int = 0
    """)
    ws = DictWorkspace({"a.py": src})
    action = AddField(
        cls=SymbolRef(file="a.py", name="C"),
        name=field_name,
        type=TypeExpr(expr=type_expr),
        confidence=0.5,
    )
    result = compile_action(action, ws)
    assert _parses(result.file_changes[0].after)


@given(
    param_name=_IDENTIFIERS,
    type_expr=st.sampled_from(["int", "str", "bool", "list[int]"]),
    keyword_only=st.booleans(),
)
@settings(max_examples=50, deadline=None)
def test_property_add_parameter_always_parses(
    param_name: str, type_expr: str, keyword_only: bool
) -> None:
    src = textwrap.dedent("""\
        def f(a: int) -> None:
            pass
    """)
    ws = DictWorkspace({"a.py": src})
    action = AddParameter(
        function=SymbolRef(file="a.py", name="f"),
        name=param_name,
        type=TypeExpr(expr=type_expr),
        keyword_only=keyword_only,
        confidence=0.5,
    )
    result = compile_action(action, ws)
    assert _parses(result.file_changes[0].after)
