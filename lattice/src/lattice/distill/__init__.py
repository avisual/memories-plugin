"""DISTILL — Organ 8.

The IMPROVES dial: over time, more decisions get made WITHOUT calling
the LLM. DESIGN.md describes a trained policy net; this v0 takes the
lightweight path explicitly named in the same design doc — distil
recurring patterns into a runtime proposer that fires when confident.

What's actually implemented (composes from prior organs):

- PatternProposer (`propose/pattern.py`) is the *hand-curated*
  distilled rules. It runs first in the proposer chain and handles
  ~50% of common task phrasings without invoking the LLM.

- EVOLVE phase 2 (`atoms/evolve.py:boost_recurrent_traces`) is the
  *learned* distillation: when an action sequence recurs ≥ N times,
  its trace atoms get an importance boost, recall surfaces them
  higher, and the LLM next time sees a strong nudge toward that
  pattern. That's distillation via memory, not via fresh weights.

- ApprenticeProposer (this module) closes the loop further: at
  startup it scans the atom store for high-importance trace atoms
  and exposes them as a runtime proposer. When a new task's
  embedding closely matches a learned trace, the apprentice emits
  the trace's first action AS A CANDIDATE — the population scorer
  in Organ 6 then decides whether to use it.

  Net effect: tasks the system has done before get a relevant,
  pre-vetted candidate without an LLM call. The LLM is reserved
  for genuinely novel work. The composite proposer chain becomes:

    [PatternProposer, ApprenticeProposer, LLMProposer]

The heavyweight Organ 8 (trained policy net) builds on this:
the trace store BECOMES the training data; a tiny encoder learns
(observation → action) mappings; the apprentice becomes a neural
head. Out of scope here.
"""

from lattice.distill.apprentice import ApprenticeProposer

__all__ = ["ApprenticeProposer"]
