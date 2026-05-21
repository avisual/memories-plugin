"""PROPOSE — sources of typed Actions.

Every Proposer turns a typed Observation into one or more typed
Actions. The boundary is intentional: the orchestrator's reasoning
loop never knows whether the actions came from a rule, a small LLM,
a hosted LLM, the apprentice (Organ 8), or a human.

Implementations:
- MockProposer: deterministic, for tests and the v0 demo.
- LocalLLMProposer: a small instruct model via transformers, with
  Pydantic-validated JSON output + retry-on-parse-failure.
- HostedLLMProposer: Anthropic API with tool-use; constrained
  decoding via tool schema = the action verb set. Quality jump
  for users with ANTHROPIC_API_KEY.
- (planned) ApprenticeProposer: the distilled policy net (Organ 8).
"""

from lattice.propose.base import (
    ObservationContext,
    Proposer,
    ProposerError,
    proposer_for_intent,
)
from lattice.propose.mock import MockProposer
from lattice.propose.pattern import (
    CompositeProposer,
    PatternProposer,
    task_to_action,
)

__all__ = [
    "CompositeProposer",
    "MockProposer",
    "ObservationContext",
    "PatternProposer",
    "Proposer",
    "ProposerError",
    "proposer_for_intent",
    "task_to_action",
]


def _make_two_stage_proposer():  # pragma: no cover (import deferred)
    from lattice.propose.two_stage import TwoStageProposer

    return TwoStageProposer


def _make_apprentice_proposer():  # pragma: no cover (import deferred)
    from lattice.distill.apprentice import ApprenticeProposer

    return ApprenticeProposer


def _make_hosted_proposer_class():  # pragma: no cover (import deferred)
    from lattice.propose.hosted import HostedLLMProposer

    return HostedLLMProposer
