"""Deterministic Proposer — used by tests and the v0 smoke loop.

Holds a pre-defined sequence of action lists; `propose` returns the
next entry. When exhausted, returns an empty list.
"""

from __future__ import annotations

from lattice.actions import Action
from lattice.propose.base import ObservationContext, Proposer


class MockProposer:
    """Returns pre-canned action batches in order."""

    def __init__(self, batches: list[list[Action]]) -> None:
        self._batches = list(batches)
        self._calls = 0

    @classmethod
    def empty(cls) -> "MockProposer":
        return cls(batches=[])

    @property
    def calls(self) -> int:
        return self._calls

    def propose(self, obs: ObservationContext, n: int = 1) -> list[Action]:
        self._calls += 1
        if not self._batches:
            return []
        batch = self._batches.pop(0)
        return batch[:n] if n > 0 else batch


# Static check that MockProposer satisfies the Proposer protocol.
_: Proposer = MockProposer.empty()
