"""STEER — Organ 4.

Two implementation paths are described in DESIGN.md:

- **Heavyweight (the headline)**: activation patching. The top-K
  recalled atoms are summed (weighted by activation) into a steering
  vector applied to the LLM's residual stream at selected layers.
  Requires open-weight model access + transformer_lens / nnsight +
  per-architecture layer probes. Not built in this session.

- **Lightweight (the fallback DESIGN.md acknowledges)**: render the
  highest-importance atoms in a strong position in the prompt — at
  the top of the user message, marked PRIMING. Influence-by-attention-
  position rather than influence-by-residual-stream-injection.

This module implements the lightweight path. It's a clean function
the agent loop calls to get the PRIMING block; the agent's
`_build_observation` then includes it as the lead hint. EVOLVE's
boost_recurrent_traces (Organ 9 phase 2) is what makes the priming
ALSO adapt to lived experience — high-importance trace atoms surface
above generic seed atoms because of `score = cosine + 0.10 * importance`
in the recall scorer.

Net effect: atoms the system has learned matter from doing the task
N times end up in the LLM's most-attended-to prompt position. That's
the v0 STEER.

When the heavyweight path lands, this module gains an
`apply_steering_vector(model, vector)` entry point and the
LocalLLMProposer routes there for open-weight models.
"""

from __future__ import annotations

from typing import Any


def top_priming_atoms(
    atom_store: Any,
    task: str,
    *,
    k: int = 2,
    min_importance: float = 0.6,
) -> list:
    """Return up to K high-importance atoms relevant to the task.

    Used by the agent loop to construct a PRIMING block at the head
    of the LLM's user message. Atoms must clear `min_importance` so
    only patterns the system has lived through (EVOLVE-boosted) make
    it in — generic seeds at importance 0.55 stay in the regular
    hint pool.

    Returns an empty list when no store is available or no atom clears
    the bar; the agent loop then skips the PRIMING block entirely.
    """
    if atom_store is None or k <= 0 or not task.strip():
        return []
    try:
        results = atom_store.recall(task, k=max(k * 2, 4))
    except Exception:  # noqa: BLE001
        return []
    primed = [r for r in results if r.atom.importance >= min_importance]
    return primed[:k]


def render_priming_block(atoms: list) -> str:
    """Render priming atoms as a single string fit for the prompt head.

    Empty input -> empty string (caller skips the block entirely).
    """
    if not atoms:
        return ""
    lines = ["PRIMING — patterns this system has lived through:"]
    for r in atoms:
        # Strip the JSON trailer that trace atoms carry; humans (and the
        # model) read better without it.
        content = r.atom.content
        marker = "\nJSON: "
        idx = content.rfind(marker)
        if idx >= 0:
            content = content[:idx]
        lines.append(f"  • [importance {r.atom.importance:.2f}] {content[:300]}")
    return "\n".join(lines)
