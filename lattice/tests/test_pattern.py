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
