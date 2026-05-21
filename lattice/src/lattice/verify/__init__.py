"""VERIFY — Organ 7.

Silent ground truth: type checker, lint, narrow test subset, in a
sandbox, ≤2s budget. Failures don't reach the user — they become atoms
(and counterfactuals).

v0 implements only the syntactic gate: the compiler's output must parse
as valid Python. Type-check and test-run gates land when the world
model is in place.
"""

from lattice.verify.syntactic import SyntacticOutcome, verify_syntactic

__all__ = ["SyntacticOutcome", "verify_syntactic"]
