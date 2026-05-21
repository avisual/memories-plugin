"""VERIFY — Organ 7.

Silent ground truth: parse, type-check, run the affected tests in a
sandbox. Failures don't reach the user without the agent loop getting
a chance to react.

What's implemented:
- syntactic gate (ast.parse) — always on, fast.
- type-check gate (mypy in a tempdir) — opt-in via [typecheck] extras.
- test-run gate (pytest in a sandbox copy) — opt-in via the agent's
  --tests flag. Auto-discovers tests by mirror (src/foo.py ->
  tests/test_foo.py) and by content reference (test files that
  import the touched module).

Failure modes the test gate catches that ast.parse and mypy miss:
- ImportError at runtime due to broken refactor.
- Assertion failures when behavior changed.
- AttributeError when a method's signature shifted unexpectedly.
"""

from lattice.verify.syntactic import SyntacticOutcome, verify_syntactic
from lattice.verify.tests import TestVerifyOutcome, verify_tests
from lattice.verify.types import TypeCheckOutcome, verify_types

__all__ = [
    "SyntacticOutcome",
    "TestVerifyOutcome",
    "TypeCheckOutcome",
    "verify_syntactic",
    "verify_tests",
    "verify_types",
]
