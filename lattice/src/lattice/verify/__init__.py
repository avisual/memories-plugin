"""VERIFY — Organ 7.

Silent ground truth: type checker, lint, narrow test subset, in a
sandbox. Failures don't reach the user — they become atoms (and
counterfactuals).

v0 implements:
- syntactic gate (ast.parse) — always on, fast.
- type-check gate (mypy in a tempdir) — optional via [typecheck] extras.

Test-run gate (pytest on affected tests) lands with the coverage map
in the world model.
"""

from lattice.verify.syntactic import SyntacticOutcome, verify_syntactic
from lattice.verify.types import TypeCheckOutcome, verify_types

__all__ = [
    "SyntacticOutcome",
    "TypeCheckOutcome",
    "verify_syntactic",
    "verify_types",
]
