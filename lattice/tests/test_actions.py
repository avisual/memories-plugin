"""Property and unit tests for the typed action DSL.

The DSL's contract:
- Every action that passes Pydantic validation has a well-formed shape.
- An action dict either parses into exactly one verb, or raises.
- Constructed actions round-trip through JSON without information loss.
- Boundary constraints (confidence ∈ [0,1], paths not absolute, etc.)
  are enforced at construction time, never silently coerced.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from lattice.actions import (
    Action,
    AddField,
    AddImport,
    AddParameter,
    AddTest,
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
    parse_action,
)
from lattice.actions.action import action_json_schema

# ---------- helpers ----------

VALID_PATHS = st.from_regex(r"\A[a-z][a-z0-9_/]*\.py\Z", fullmatch=True)
VALID_NAMES = st.from_regex(r"\A[a-z_][a-z0-9_]*\Z", fullmatch=True)
CONFIDENCE = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def _symbol(path: str = "src/payments/charge.py", name: str = "charge") -> SymbolRef:
    return SymbolRef(file=path, name=name)


def _file(path: str = "src/payments/charge.py") -> FileRef:
    return FileRef(path=path)


# ---------- shape & validation ----------


class TestRefs:
    def test_file_rejects_absolute_path(self):
        with pytest.raises(ValidationError):
            FileRef(path="/etc/passwd")

    def test_file_rejects_traversal(self):
        with pytest.raises(ValidationError):
            FileRef(path="src/../secrets.py")

    def test_symbol_rejects_absolute_file(self):
        with pytest.raises(ValidationError):
            SymbolRef(file="/abs/path.py", name="x")

    def test_span_rejects_inverted_range(self):
        with pytest.raises(ValidationError):
            SpanRef(file="a.py", start_line=10, end_line=5)

    def test_span_accepts_single_line(self):
        SpanRef(file="a.py", start_line=5, end_line=5)

    def test_type_expr_rejects_empty(self):
        with pytest.raises(ValidationError):
            TypeExpr(expr="")

    def test_refs_are_frozen(self):
        f = _file()
        with pytest.raises(ValidationError):
            f.path = "other.py"


class TestAddImport:
    def test_bare_import(self):
        a = AddImport(file=_file(), module="json", confidence=0.9)
        assert a.names is None and a.alias is None

    def test_from_import(self):
        AddImport(file=_file(), module="json", names=["loads", "dumps"], confidence=0.5)

    def test_alias_with_multiple_names_rejected(self):
        with pytest.raises(ValidationError):
            AddImport(
                file=_file(),
                module="json",
                names=["loads", "dumps"],
                alias="j",
                confidence=0.5,
            )

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValidationError):
            AddImport(file=_file(), module="json", names=["loads", "loads"], confidence=0.5)

    def test_empty_names_rejected(self):
        with pytest.raises(ValidationError):
            AddImport(file=_file(), module="json", names=[], confidence=0.5)


class TestRenameSymbol:
    def test_basic(self):
        RenameSymbol(symbol=_symbol(), new_name="charge_customer", confidence=0.7)

    def test_identity_rename_rejected(self):
        with pytest.raises(ValidationError):
            RenameSymbol(symbol=_symbol(name="charge"), new_name="charge", confidence=0.7)

    def test_dotted_name_compared_by_leaf(self):
        with pytest.raises(ValidationError):
            RenameSymbol(
                symbol=_symbol(name="ChargeProcessor.refund"),
                new_name="refund",
                confidence=0.7,
            )


class TestAddParameter:
    def test_basic(self):
        AddParameter(
            function=_symbol(),
            name="customer_id",
            type=TypeExpr(expr="str"),
            confidence=0.6,
        )

    def test_keyword_only_with_position_rejected(self):
        with pytest.raises(ValidationError):
            AddParameter(
                function=_symbol(),
                name="x",
                type=TypeExpr(expr="int"),
                position=0,
                keyword_only=True,
                confidence=0.6,
            )


class TestWrapInTry:
    def test_composes_actions_in_handler(self):
        inner = MarkBlocked(reason_code="external_dependency", detail="db down", confidence=0.9)
        WrapInTry(
            span=SpanRef(file="a.py", start_line=10, end_line=20),
            exception_type=TypeExpr(expr="ConnectionError"),
            handler_body=[inner],
            confidence=0.8,
        )

    def test_empty_handler_ok(self):
        WrapInTry(
            span=SpanRef(file="a.py", start_line=10, end_line=20),
            exception_type=TypeExpr(expr="Exception"),
            confidence=0.8,
        )


class TestAddTest:
    def test_test_name_prefix_required(self):
        with pytest.raises(ValidationError):
            AddTest(
                target=_symbol(),
                test_name="charges_a_customer",
                given=Expr(code="c = Charge(1)"),
                when=Expr(code="c.run()"),
                then=Expr(code="assert c.done"),
                confidence=0.7,
            )

    def test_valid(self):
        AddTest(
            target=_symbol(),
            test_name="test_charges_a_customer",
            given=Expr(code="c = Charge(1)"),
            when=Expr(code="c.run()"),
            then=Expr(code="assert c.done"),
            confidence=0.7,
        )


# ---------- discriminated union ----------


class TestParseAction:
    def test_parses_each_verb(self):
        examples = [
            {"verb": "AddImport", "file": {"path": "a.py"}, "module": "json", "confidence": 0.5},
            {
                "verb": "RenameSymbol",
                "symbol": {"file": "a.py", "name": "x"},
                "new_name": "y",
                "confidence": 0.5,
            },
            {
                "verb": "AddField",
                "cls": {"file": "a.py", "name": "C"},
                "name": "f",
                "type": {"expr": "int"},
                "confidence": 0.5,
            },
            {
                "verb": "AddParameter",
                "function": {"file": "a.py", "name": "f"},
                "name": "x",
                "type": {"expr": "int"},
                "confidence": 0.5,
            },
            {
                "verb": "WrapInTry",
                "span": {"file": "a.py", "start_line": 1, "end_line": 2},
                "exception_type": {"expr": "Exception"},
                "confidence": 0.5,
            },
            {
                "verb": "AddTest",
                "target": {"file": "a.py", "name": "f"},
                "test_name": "test_f",
                "given": {"code": "1"},
                "when": {"code": "1"},
                "then": {"code": "assert 1"},
                "confidence": 0.5,
            },
            {"verb": "RecallMore", "query": "rate limits", "confidence": 0.5},
            {"verb": "RevealBody", "symbol": {"file": "a.py", "name": "f"}, "confidence": 0.5},
            {
                "verb": "MarkBlocked",
                "reason_code": "needs_human",
                "detail": "ambiguous",
                "confidence": 0.5,
            },
            {"verb": "Branch", "rationale": "try alternative", "confidence": 0.5},
        ]
        for data in examples:
            action = parse_action(data)
            assert action.verb == data["verb"]

    def test_unknown_verb_rejected(self):
        with pytest.raises(ValidationError):
            parse_action({"verb": "Nuke", "target": "everything", "confidence": 1.0})

    def test_missing_verb_rejected(self):
        with pytest.raises(ValidationError):
            parse_action({"module": "json", "file": {"path": "a.py"}, "confidence": 0.5})

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            parse_action(
                {
                    "verb": "RecallMore",
                    "query": "x",
                    "confidence": 0.5,
                    "rogue_field": True,
                }
            )

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            parse_action({"verb": "Branch", "rationale": "x", "confidence": 1.5})
        with pytest.raises(ValidationError):
            parse_action({"verb": "Branch", "rationale": "x", "confidence": -0.1})


# ---------- round-trip ----------


@given(
    module=st.from_regex(r"\A[a-z][a-z0-9_.]{0,30}\Z", fullmatch=True),
    confidence=CONFIDENCE,
)
def test_add_import_round_trip(module: str, confidence: float):
    a = AddImport(file=_file(), module=module, confidence=confidence)
    blob = a.model_dump_json()
    restored = parse_action(json.loads(blob))
    assert restored == a


@given(
    new_name=VALID_NAMES,
    confidence=CONFIDENCE,
)
def test_rename_round_trip(new_name: str, confidence: float):
    # Avoid the identity-rename validator by using a fixed distinct source name.
    if new_name == "charge":
        return
    a = RenameSymbol(symbol=_symbol(name="charge"), new_name=new_name, confidence=confidence)
    restored = parse_action(json.loads(a.model_dump_json()))
    assert restored == a


@given(
    rationale=st.text(min_size=1, max_size=280).filter(lambda s: s.strip()),
    confidence=CONFIDENCE,
)
def test_branch_round_trip(rationale: str, confidence: float):
    a = Branch(rationale=rationale, confidence=confidence)
    restored = parse_action(json.loads(a.model_dump_json()))
    assert restored == a


# ---------- schema export ----------


def test_json_schema_lists_all_verbs():
    schema = action_json_schema()
    blob = json.dumps(schema)
    for verb in [
        "AddImport",
        "RenameSymbol",
        "AddField",
        "AddParameter",
        "WrapInTry",
        "AddTest",
        "RecallMore",
        "RevealBody",
        "MarkBlocked",
        "Branch",
    ]:
        assert verb in blob, f"verb {verb!r} missing from JSON schema"


def test_json_schema_is_serializable():
    json.dumps(action_json_schema())  # must not raise
