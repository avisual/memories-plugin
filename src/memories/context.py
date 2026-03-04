"""Context window budget manager for the memories system.

Controls how many tokens of memory to inject into the current context
window, applying progressive compression when the budget is tight.

The manager works with four compression levels:

- **Level 0** (full): Complete atom content with synapse info, tags, and
  all metadata.
- **Level 1** (content-only): Atom type, content, and confidence score.
- **Level 2** (summaries): One-line summary per atom (first sentence or
  first 100 characters).
- **Level 3** (bullets): Minimal bullet points with type and the first
  50 characters of content.

Usage::

    from memories.context import ContextBudget

    budget = ContextBudget()
    token_limit = budget.budget_for_recall(estimated_used=80_000, priority="critical")
    result = budget.compress_to_budget(atoms, token_limit)
    print(result.formatted_text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from memories.config import get_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4
"""Rough approximation: ~4 characters per token for English text."""

_PRIORITY_PARAMS: dict[str, tuple[float, int]] = {
    "critical": (0.15, 8_000),
    "background": (0.05, 2_000),
    "minimal": (0.02, 500),
}
"""Mapping of priority level to (fraction of remaining, hard cap)."""

_MAX_COMPRESSION_LEVEL = 3
"""Highest compression level supported."""

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_AtomFormatter = Callable[[dict], str]
"""Signature for a function that renders a single atom dict to a string."""


# ---------------------------------------------------------------------------
# Budget result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BudgetResult:
    """Outcome of compressing a list of atoms to fit a token budget.

    Attributes
    ----------
    atoms:
        The atoms that fit within the allocated budget, ordered by
        descending relevance score.
    budget_used:
        Number of tokens consumed by the formatted text.
    budget_total:
        Total token budget that was available.
    compression_level:
        The compression level that was applied (0 = full detail,
        3 = minimal bullets).
    formatted_text:
        The human-readable string ready for injection into the context
        window.
    """

    atoms: list[dict] = field(default_factory=list)
    budget_used: int = 0
    budget_total: int = 0
    compression_level: int = 0
    formatted_text: str = ""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _first_sentence(text: str, *, max_chars: int = 100) -> str:
    """Extract the first sentence from *text*, capped at *max_chars*.

    A sentence boundary is detected by a period, exclamation mark, or
    question mark followed by a space.  If no boundary is found within
    *max_chars*, the text is simply truncated at the character limit.
    """
    normalised = text.replace("\n", " ").strip()

    for i, ch in enumerate(normalised[:max_chars]):
        if ch in ".!?" and i + 1 < len(normalised) and normalised[i + 1] == " ":
            return normalised[: i + 1]

    if len(normalised) <= max_chars:
        return normalised

    return normalised[:max_chars].rstrip()


# ---------------------------------------------------------------------------
# Context budget manager
# ---------------------------------------------------------------------------


class ContextBudget:
    """Calculate and enforce token budgets for memory recall injection.

    Parameters
    ----------
    model_context_window:
        Override the model context window size in tokens.  When *None*,
        the value is read from :func:`memories.config.get_config`.
    """

    def __init__(self, model_context_window: int | None = None) -> None:
        cfg = get_config()
        self.model_context_window: int = (
            model_context_window or cfg.context_window_tokens
        )

    # ------------------------------------------------------------------
    # Budget calculation
    # ------------------------------------------------------------------

    def budget_for_recall(
        self,
        estimated_used: int = 0,
        priority: str = "background",
    ) -> int:
        """Calculate how many tokens we can spend on memory recall.

        Parameters
        ----------
        estimated_used:
            Number of tokens already consumed in the current context
            window (system prompt, conversation history, tool results,
            etc.).
        priority:
            Recall urgency level:

            - ``"critical"`` -- direct user question about past work.
              Up to 15 % of remaining window, hard cap 8 000 tokens.
            - ``"background"`` -- automatic recall for context.
              Up to 5 % of remaining window, hard cap 2 000 tokens.
            - ``"minimal"`` -- context is tight, inject very little.
              Up to 2 % of remaining window, hard cap 500 tokens.

        Returns
        -------
        int
            Maximum number of tokens that may be used for recall.
        """
        remaining = max(0, self.model_context_window - estimated_used)

        fraction, cap = _PRIORITY_PARAMS.get(
            priority,
            _PRIORITY_PARAMS["minimal"],
        )
        return min(int(remaining * fraction), cap)

    # ------------------------------------------------------------------
    # Compression pipeline
    # ------------------------------------------------------------------

    def compress_to_budget(
        self,
        atoms: list[dict],
        budget_tokens: int,
    ) -> BudgetResult:
        """Progressively compress *atoms* to fit within *budget_tokens*.

        Atoms are first sorted by ``score`` descending (highest-value
        memories first).  The method then tries each compression level
        in order, greedily accumulating atoms until the budget would be
        exceeded.  If a higher-detail level does not fit even one atom,
        the next compression level is attempted.

        Parameters
        ----------
        atoms:
            A list of dicts with keys: ``id``, ``content``, ``type``,
            ``confidence``, ``score``, ``tags``, and optionally
            ``synapses`` (a list of synapse dicts with ``relationship``,
            ``content``, and ``strength``).
        budget_tokens:
            Maximum number of tokens to consume.

        Returns
        -------
        BudgetResult
            The atoms that fit, the tokens used, and the formatted text.
        """
        if not atoms or budget_tokens <= 0:
            return BudgetResult(
                atoms=[],
                budget_used=0,
                budget_total=max(budget_tokens, 0),
                compression_level=0,
                formatted_text="",
            )

        # Sort by score descending so the most relevant atoms come first.
        sorted_atoms = sorted(
            atoms,
            key=lambda a: a.get("score", 0.0),
            reverse=True,
        )

        # Try each compression level until we fit at least one atom.
        formatters: list[_AtomFormatter] = [
            self._format_atom_level0,
            self._format_atom_level1,
            self._format_atom_level2,
            self._format_atom_level3,
        ]

        for level, formatter in enumerate(formatters):
            selected, text, tokens_used = self._accumulate(
                sorted_atoms,
                formatter,
                budget_tokens,
            )
            if selected:
                return BudgetResult(
                    atoms=selected,
                    budget_used=tokens_used,
                    budget_total=budget_tokens,
                    compression_level=level,
                    formatted_text=text,
                )

        # Even the most aggressive compression could not fit a single
        # atom -- return an empty result.
        return BudgetResult(
            atoms=[],
            budget_used=0,
            budget_total=budget_tokens,
            compression_level=_MAX_COMPRESSION_LEVEL,
            formatted_text="",
        )

    # ------------------------------------------------------------------
    # Greedy accumulator
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate(
        atoms: list[dict],
        formatter: _AtomFormatter,
        budget_tokens: int,
    ) -> tuple[list[dict], str, int]:
        """Greedily add atoms using *formatter* until budget is exhausted.

        Returns
        -------
        tuple
            ``(selected_atoms, formatted_text, tokens_used)``
        """
        selected: list[dict] = []
        parts: list[str] = []
        tokens_used = 0

        for atom in atoms:
            rendered = formatter(atom)
            atom_tokens = ContextBudget.estimate_tokens(rendered)

            if tokens_used + atom_tokens > budget_tokens:
                break

            selected.append(atom)
            parts.append(rendered)
            tokens_used += atom_tokens

        return selected, "\n".join(parts), tokens_used

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count based on character length.

        Uses the common heuristic of ~4 characters per token for
        English text.  Always returns at least 1.

        Parameters
        ----------
        text:
            The text to estimate.

        Returns
        -------
        int
            Estimated token count (>= 1).
        """
        return max(1, len(text) // _CHARS_PER_TOKEN)

    # ------------------------------------------------------------------
    # Per-atom formatters
    # ------------------------------------------------------------------

    @staticmethod
    def _format_atom_level0(atom: dict) -> str:
        """Full detail: content, type, confidence, tags, synapse descriptions.

        Example output::

            [fact] (confidence: 0.95, tags: redis, performance)
            Redis SCAN is O(N) over full keyspace, not per-call.
              -> related-to: "Redis cluster sharding" (strength: 0.8)
              -> warns-against: "Never use KEYS in production" (strength: 0.9)
        """
        atom_type = atom.get("type", "unknown")
        confidence = atom.get("confidence", 1.0)
        tags = atom.get("tags", "")
        content = atom.get("content", "")
        synapses: list[dict] = atom.get("synapses") or []

        header_parts = [f"[{atom_type}] (confidence: {confidence}"]
        if tags:
            header_parts.append(f", tags: {tags}")
        header = "".join(header_parts) + ")"

        lines = [header, content]

        for syn in synapses:
            relationship = syn.get("relationship", "related-to")
            syn_content = syn.get("content", "")
            strength = syn.get("strength", 0.5)
            lines.append(
                f'  -> {relationship}: "{syn_content}" (strength: {strength})'
            )

        return "\n".join(lines)

    @staticmethod
    def _format_atom_level1(atom: dict) -> str:
        """Content only: type prefix, content, and confidence.

        Example output::

            [fact|0.95] Redis SCAN is O(N) over full keyspace, not per-call.
        """
        atom_type = atom.get("type", "unknown")
        confidence = atom.get("confidence", 1.0)
        content = atom.get("content", "")
        return f"[{atom_type}|{confidence}] {content}"

    @staticmethod
    def _format_atom_level2(atom: dict) -> str:
        """Summary: first sentence or first 100 characters of content.

        Example output::

            - Redis SCAN is O(N) over full keyspace
        """
        content = atom.get("content", "")
        summary = _first_sentence(content, max_chars=100)
        return f"- {summary}"

    @staticmethod
    def _format_atom_level3(atom: dict) -> str:
        """Bullets: [type] first 50 characters with ellipsis.

        Example output::

            * [fact] Redis SCAN is O(N) over full key...
        """
        atom_type = atom.get("type", "unknown")
        content = atom.get("content", "")
        truncated = content[:50].rstrip()
        if len(content) > 50:
            truncated += "..."
        return f"* [{atom_type}] {truncated}"

    # ------------------------------------------------------------------
    # Batch formatters (convenience wrappers)
    # ------------------------------------------------------------------

    def format_atoms_level0(self, atoms: list[dict]) -> str:
        """Format all *atoms* at compression level 0 (full detail)."""
        return "\n".join(self._format_atom_level0(a) for a in atoms)

    def format_atoms_level1(self, atoms: list[dict]) -> str:
        """Format all *atoms* at compression level 1 (content only)."""
        return "\n".join(self._format_atom_level1(a) for a in atoms)

    def format_atoms_level2(self, atoms: list[dict]) -> str:
        """Format all *atoms* at compression level 2 (summaries)."""
        return "\n".join(self._format_atom_level2(a) for a in atoms)

    def format_atoms_level3(self, atoms: list[dict]) -> str:
        """Format all *atoms* at compression level 3 (bullets)."""
        return "\n".join(self._format_atom_level3(a) for a in atoms)
