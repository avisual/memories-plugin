"""ORCHESTRATOR — the v0 reasoning loop driver.

Takes a high-level Intent, expands it into a sequence of typed Actions
(the composition spine: intent → primitives), compiles each, verifies
each, and returns an ExecutionReport bundling all diffs and outcomes.

This is the harness's central loop in its simplest form. Future
iterations replace the rule-based `expand_intent` with an LLM that
emits Actions via constrained decoding, add the world model in front
of compile, the population search around it, and the apprentice taking
over the easy cases.
"""

from lattice.orchestrator.intent import (
    AddParameterToAllMatching,
    Intent,
    expand_intent,
)
from lattice.orchestrator.run import ExecutionReport, execute_plan

__all__ = [
    "AddParameterToAllMatching",
    "ExecutionReport",
    "Intent",
    "execute_plan",
    "expand_intent",
]
