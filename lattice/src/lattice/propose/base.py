"""Proposer interface — the boundary between the harness and the LLM.

An ObservationContext is the structured payload a Proposer sees: the
task, a code subgraph (Symbol list for v0), and any free-form hints.
The Proposer returns up to N typed Actions; the orchestrator decides
which to keep.

This is deliberately small. Once the world model lands the Observation
gains more fields (free-energy signal, recalled atoms, etc.) but the
Protocol stays the same — additions are non-breaking.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from lattice.actions import Action
from lattice.sense import Symbol


class ObservationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: str = Field(min_length=1, description="One-line description of the goal.")
    symbols: tuple[Symbol, ...] = Field(
        default=(),
        description="Symbols the proposer may operate on (typically top-K relevant).",
    )
    hints: tuple[str, ...] = Field(
        default=(),
        description="Optional free-form hints surfaced from memory / recent failures.",
    )


class ProposerError(Exception):
    """A proposer failed to produce a usable Action."""


@runtime_checkable
class Proposer(Protocol):
    def propose(self, obs: ObservationContext, n: int = 1) -> list[Action]:
        """Return up to *n* candidate Actions for the observation.

        May return fewer than *n* candidates (or zero) when the proposer
        cannot generate confident output. Raises ProposerError when the
        upstream model produced nothing parseable after retries.
        """
        ...


def proposer_for_intent(intent: str) -> Proposer:
    """Convenience: pick a default proposer based on environment.

    Today returns a MockProposer; once the LocalLLMProposer is on the
    install path with a model cached, the selection here will become
    `LocalLLMProposer` by default.
    """
    from lattice.propose.mock import MockProposer

    return MockProposer.empty()
