"""Tests for the deterministic pattern proposer.

Covers the natural-language patterns that map to typed Actions
without invoking the LLM. Each pattern should match its target
phrasings and ONLY its target phrasings (no false positives on
similar but distinct tasks).
"""

from __future__ import annotations

import pytest

from lattice.actions import AddImport, RenameSymbol
from lattice.propose import ObservationContext, PatternProposer, task_to_action


class TestPlainImport:
    def test_basic(self):
        a = task_to_action("Add an import of os to src/main.py")
        assert isinstance(a, AddImport)
        assert a.module == "os"
        assert a.names is None
        assert a.file.path == "src/main.py"

    def test_dotted_module(self):
        a = task_to_action("Add an import of os.path to src/a.py")
        assert isinstance(a, AddImport)
        assert a.module == "os.path"

    def test_with_the_module_phrase(self):
        a = task_to_action("Add an import of the json module to src/b.py")
        assert isinstance(a, AddImport)
        assert a.module == "json"

    def test_quoted_module(self):
        a = task_to_action("Add an import of 'stripe' to src/payments/charge.py")
        assert isinstance(a, AddImport)
        assert a.module == "stripe"

    def test_into_synonym(self):
        a = task_to_action("Add an import of json into src/c.py")
        assert isinstance(a, AddImport)
        assert a.module == "json"


class TestFromImport:
    def test_basic(self):
        a = task_to_action("Add an import of request from flask to src/app.py")
        assert isinstance(a, AddImport)
        assert a.module == "flask"
        assert a.names == ["request"]

    def test_alt_phrasing(self):
        a = task_to_action("import Optional from typing in src/utils.py")
        assert isinstance(a, AddImport)
        assert a.module == "typing"
        assert a.names == ["Optional"]


class TestAliasImport:
    def test_basic(self):
        a = task_to_action("Add an import of numpy as np to src/calc.py")
        assert isinstance(a, AddImport)
        assert a.module == "numpy"
        assert a.alias == "np"


class TestRename:
    def test_basic(self):
        a = task_to_action("Rename charge to take_payment in src/billing.py")
        assert isinstance(a, RenameSymbol)
        assert a.symbol.name == "charge"
        assert a.new_name == "take_payment"
        assert a.symbol.file == "src/billing.py"


class TestAddParameter:
    def test_function_with_class(self):
        from lattice.actions import AddParameter

        a = task_to_action(
            "Add a keyword-only parameter named timeout of type float to function get of class Client in src/api.py"
        )
        assert isinstance(a, AddParameter)
        assert a.function.name == "Client.get"
        assert a.name == "timeout"
        assert a.type.expr == "float"
        assert a.keyword_only is True

    def test_function_module_level(self):
        from lattice.actions import AddParameter

        a = task_to_action("Add a parameter user_id of type str to function login in src/auth.py")
        assert isinstance(a, AddParameter)
        assert a.function.name == "login"
        assert a.name == "user_id"
        assert a.keyword_only is False

    def test_with_default(self):
        from lattice.actions import AddParameter

        a = task_to_action(
            "Add a parameter retries of type int with default 3 to function call_api in src/x.py"
        )
        assert isinstance(a, AddParameter)
        assert a.default.code == "3"
        assert a.type.expr == "int"


class TestAddField:
    def test_basic(self):
        from lattice.actions import AddField

        a = task_to_action(
            "Add a field version of type str with default 1.0 to class Config in src/config.py"
        )
        assert isinstance(a, AddField)
        assert a.cls.name == "Config"
        assert a.name == "version"
        assert a.type.expr == "str"

    def test_attribute_synonym(self):
        from lattice.actions import AddField

        a = task_to_action(
            "Add an attribute api_key of type str to class Client in src/api.py"
        )
        assert isinstance(a, AddField)
        assert a.name == "api_key"


class TestWrapInTry:
    def test_basic(self):
        from lattice.actions import WrapInTry

        a = task_to_action("Wrap lines 5-8 of src/io.py in a try/except for IOError")
        assert isinstance(a, WrapInTry)
        assert a.span.file == "src/io.py"
        assert a.span.start_line == 5
        assert a.span.end_line == 8
        assert a.exception_type.expr == "IOError"

    def test_through_keyword(self):
        from lattice.actions import WrapInTry

        a = task_to_action("Wrap lines 1 through 10 of src/a.py in try/except for ValueError")
        assert isinstance(a, WrapInTry)
        assert (a.span.start_line, a.span.end_line) == (1, 10)

    def test_no_exception_defaults_to_exception(self):
        from lattice.actions import WrapInTry

        a = task_to_action("Wrap lines 5-8 of src/a.py in a try/except")
        assert isinstance(a, WrapInTry)
        assert a.exception_type.expr == "Exception"


class TestAddTestPattern:
    def test_basic(self):
        from lattice.actions import AddTest

        a = task_to_action("Add a smoke test for charge in src/billing.py")
        assert isinstance(a, AddTest)
        assert a.target.name == "charge"
        assert a.target.file == "src/billing.py"
        assert a.test_name == "test_charge"

    def test_with_explicit_name(self):
        from lattice.actions import AddTest

        a = task_to_action(
            "Add a test called test_charge_zero for charge in src/billing.py"
        )
        assert isinstance(a, AddTest)
        assert a.test_name == "test_charge_zero"


class TestNonMatches:
    def test_unrelated_task(self):
        assert task_to_action("Refactor the whole codebase") is None

    def test_ambiguous(self):
        assert task_to_action("Please help me") is None

    def test_empty(self):
        assert task_to_action("") is None


class TestProposerProtocol:
    def test_returns_action_when_matches(self):
        p = PatternProposer()
        out = p.propose(ObservationContext(task="Add an import of os to src/x.py"))
        assert len(out) == 1
        assert out[0].verb == "AddImport"

    def test_returns_empty_when_no_match(self):
        p = PatternProposer()
        out = p.propose(ObservationContext(task="Refactor everything"))
        assert out == []


class TestComposite:
    def test_first_non_empty_wins(self):
        from lattice.propose.mock import MockProposer
        from lattice.propose import CompositeProposer

        # Pattern matches => MockProposer never gets called.
        p = CompositeProposer(
            [
                PatternProposer(),
                MockProposer(batches=[[]]),  # would return []
            ]
        )
        out = p.propose(ObservationContext(task="Add an import of os to src/x.py"))
        assert len(out) == 1
        assert out[0].verb == "AddImport"

    def test_falls_through_when_first_empty(self):
        from lattice.actions import Branch
        from lattice.propose import CompositeProposer
        from lattice.propose.mock import MockProposer

        p = CompositeProposer(
            [
                PatternProposer(),  # nothing matches
                MockProposer(batches=[[Branch(rationale="x", confidence=0.5)]]),
            ]
        )
        out = p.propose(ObservationContext(task="something the pattern doesn't know"))
        assert len(out) == 1
        assert out[0].verb == "Branch"
