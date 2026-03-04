"""memories -- brain-like memory system for Claude Code.

Quick start::

    from memories import Brain

    async def main():
        brain = Brain()
        await brain.initialize()

        result = await brain.remember("Redis SCAN is O(N)", type="fact")
        memories = await brain.recall("How does Redis SCAN work?")

        await brain.shutdown()

For lower-level access, import from submodules::

    from memories.atoms import Atom, AtomManager, ATOM_TYPES
    from memories.synapses import Synapse, SynapseManager, RELATIONSHIP_TYPES
    from memories.retrieval import RetrievalEngine, RecallResult
    from memories.consolidation import ConsolidationEngine, ConsolidationResult
"""

from __future__ import annotations

__version__ = "0.1.0"

# Public API exports
from memories.brain import Brain
from memories.atoms import Atom, ATOM_TYPES, SEVERITY_LEVELS, TASK_STATUSES
from memories.synapses import Synapse, RELATIONSHIP_TYPES

__all__ = [
    "__version__",
    "Brain",
    "Atom",
    "ATOM_TYPES",
    "SEVERITY_LEVELS",
    "TASK_STATUSES",
    "Synapse",
    "RELATIONSHIP_TYPES",
]
