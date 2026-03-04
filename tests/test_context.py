"""Comprehensive tests for the context.py module.

Covers ContextBudget.estimate_tokens, budget_for_recall, compress_to_budget,
the _first_sentence helper, and all four formatter levels.
"""

from __future__ import annotations

import pytest

from memories.context import BudgetResult, ContextBudget, _first_sentence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atom(
    *,
    id: int = 1,
    content: str = "Some memory content.",
    type: str = "fact",
    confidence: float = 0.9,
    score: float = 0.5,
    tags: str = "",
    synapses: list[dict] | None = None,
) -> dict:
    atom: dict = {
        "id": id,
        "content": content,
        "type": type,
        "confidence": confidence,
        "score": score,
        "tags": tags,
    }
    if synapses is not None:
        atom["synapses"] = synapses
    return atom


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    """Tests for ContextBudget.estimate_tokens."""

    def test_empty_string_returns_one(self):
        assert ContextBudget.estimate_tokens("") == 1

    def test_short_text_returns_one(self):
        # 3 chars -> 3 // 4 = 0 -> max(1, 0) = 1
        assert ContextBudget.estimate_tokens("abc") == 1

    def test_exactly_four_chars(self):
        assert ContextBudget.estimate_tokens("abcd") == 1

    def test_proportional_scaling(self):
        text = "a" * 400
        assert ContextBudget.estimate_tokens(text) == 100

    def test_large_text(self):
        text = "x" * 4000
        assert ContextBudget.estimate_tokens(text) == 1000


# ---------------------------------------------------------------------------
# budget_for_recall
# ---------------------------------------------------------------------------

class TestBudgetForRecall:
    """Tests for ContextBudget.budget_for_recall."""

    def setup_method(self):
        self.budget = ContextBudget(model_context_window=200_000)

    def test_critical_priority(self):
        # remaining = 200_000 - 100_000 = 100_000
        # 100_000 * 0.15 = 15_000 -> capped at 8_000
        result = self.budget.budget_for_recall(estimated_used=100_000, priority="critical")
        assert result == 8_000

    def test_critical_below_cap(self):
        # remaining = 200_000 - 190_000 = 10_000
        # 10_000 * 0.15 = 1_500 (below cap of 8_000)
        result = self.budget.budget_for_recall(estimated_used=190_000, priority="critical")
        assert result == 1_500

    def test_background_priority(self):
        # remaining = 200_000 - 100_000 = 100_000
        # 100_000 * 0.05 = 5_000 -> capped at 2_000
        result = self.budget.budget_for_recall(estimated_used=100_000, priority="background")
        assert result == 2_000

    def test_background_below_cap(self):
        # remaining = 200_000 - 195_000 = 5_000
        # 5_000 * 0.05 = 250 (below cap of 2_000)
        result = self.budget.budget_for_recall(estimated_used=195_000, priority="background")
        assert result == 250

    def test_minimal_priority(self):
        # remaining = 200_000 - 100_000 = 100_000
        # 100_000 * 0.02 = 2_000 -> capped at 500
        result = self.budget.budget_for_recall(estimated_used=100_000, priority="minimal")
        assert result == 500

    def test_minimal_below_cap(self):
        # remaining = 200_000 - 198_000 = 2_000
        # 2_000 * 0.02 = 40 (below cap of 500)
        result = self.budget.budget_for_recall(estimated_used=198_000, priority="minimal")
        assert result == 40

    def test_unknown_priority_falls_back_to_minimal(self):
        result_unknown = self.budget.budget_for_recall(estimated_used=100_000, priority="unknown_thing")
        result_minimal = self.budget.budget_for_recall(estimated_used=100_000, priority="minimal")
        assert result_unknown == result_minimal

    def test_zero_remaining_context(self):
        result = self.budget.budget_for_recall(estimated_used=200_000, priority="critical")
        assert result == 0

    def test_over_window_clamps_to_zero(self):
        # estimated_used exceeds the window -> remaining = max(0, ...) = 0
        result = self.budget.budget_for_recall(estimated_used=300_000, priority="critical")
        assert result == 0


# ---------------------------------------------------------------------------
# compress_to_budget
# ---------------------------------------------------------------------------

class TestCompressToBudget:
    """Tests for ContextBudget.compress_to_budget."""

    def setup_method(self):
        self.budget = ContextBudget(model_context_window=200_000)

    def test_empty_atoms_returns_empty_result(self):
        result = self.budget.compress_to_budget([], budget_tokens=1000)
        assert result.atoms == []
        assert result.budget_used == 0
        assert result.budget_total == 1000
        assert result.formatted_text == ""

    def test_zero_budget_returns_empty_result(self):
        atoms = [_make_atom()]
        result = self.budget.compress_to_budget(atoms, budget_tokens=0)
        assert result.atoms == []
        assert result.budget_used == 0
        assert result.budget_total == 0
        assert result.formatted_text == ""

    def test_negative_budget_returns_empty_result(self):
        atoms = [_make_atom()]
        result = self.budget.compress_to_budget(atoms, budget_tokens=-10)
        assert result.atoms == []
        assert result.budget_total == 0

    def test_sorts_by_score_descending(self):
        atoms = [
            _make_atom(id=1, score=0.3, content="low"),
            _make_atom(id=2, score=0.9, content="high"),
            _make_atom(id=3, score=0.6, content="mid"),
        ]
        result = self.budget.compress_to_budget(atoms, budget_tokens=5000)
        returned_ids = [a["id"] for a in result.atoms]
        assert returned_ids == [2, 3, 1]

    def test_greedy_accumulation_stops_at_budget(self):
        # Each level-0 atom has a header + content. Create atoms that individually
        # fit but collectively exceed a tight budget.
        small_atom = _make_atom(content="Short.")  # small footprint
        big_atom = _make_atom(content="x" * 400, score=0.1)  # ~100 tokens at level 0

        # Give budget that fits small but not both at level 0
        small_rendered = ContextBudget._format_atom_level0(small_atom)
        small_tokens = ContextBudget.estimate_tokens(small_rendered)

        # Budget fits exactly the small atom
        result = self.budget.compress_to_budget(
            [small_atom, big_atom],
            budget_tokens=small_tokens,
        )
        assert len(result.atoms) == 1

    def test_level0_chosen_when_budget_is_large(self):
        atoms = [_make_atom(content="Hello world.")]
        result = self.budget.compress_to_budget(atoms, budget_tokens=5000)
        assert result.compression_level == 0
        assert "[fact]" in result.formatted_text
        assert "confidence:" in result.formatted_text

    def test_falls_to_higher_compression_when_level0_too_large(self):
        # Create an atom whose level-0 rendering exceeds a tight budget,
        # but whose level-1 rendering fits.
        atom = _make_atom(
            content="A short fact.",
            tags="redis, performance, caching",
            synapses=[
                {"relationship": "related-to", "content": "Some synapse info", "strength": 0.8},
                {"relationship": "warns-against", "content": "Another synapse", "strength": 0.9},
            ],
        )
        level0 = ContextBudget._format_atom_level0(atom)
        level1 = ContextBudget._format_atom_level1(atom)
        level0_tokens = ContextBudget.estimate_tokens(level0)
        level1_tokens = ContextBudget.estimate_tokens(level1)

        # Budget between level1 and level0
        budget = level0_tokens - 1
        assert budget >= level1_tokens, "Test setup: level1 should fit"

        result = self.budget.compress_to_budget([atom], budget_tokens=budget)
        assert result.compression_level >= 1
        assert len(result.atoms) == 1

    def test_falls_to_level3_bullets(self):
        # Budget so tight only level 3 can fit
        atom = _make_atom(content="Tiny.")
        level3 = ContextBudget._format_atom_level3(atom)
        level3_tokens = ContextBudget.estimate_tokens(level3)
        level2 = ContextBudget._format_atom_level2(atom)
        level2_tokens = ContextBudget.estimate_tokens(level2)

        # If level 2 and level 3 have same size for tiny content, use an atom
        # with longer content to create a gap.
        atom = _make_atom(content="A moderately lengthy content that will be summarized differently at each level.")
        level2 = ContextBudget._format_atom_level2(atom)
        level3 = ContextBudget._format_atom_level3(atom)
        level2_tokens = ContextBudget.estimate_tokens(level2)
        level3_tokens = ContextBudget.estimate_tokens(level3)

        if level3_tokens < level2_tokens:
            result = self.budget.compress_to_budget([atom], budget_tokens=level2_tokens - 1)
            assert result.compression_level == 3
        else:
            # Level 3 and 2 are same size; just verify level 3 works at all
            result = self.budget.compress_to_budget([atom], budget_tokens=level3_tokens)
            assert result.compression_level <= 3
            assert len(result.atoms) == 1

    def test_nothing_fits_returns_empty_with_max_compression(self):
        # Budget of 1 token -- even level 3 won't fit
        atom = _make_atom(content="Something that takes more than 4 chars.")
        result = self.budget.compress_to_budget([atom], budget_tokens=1)
        assert result.atoms == []
        assert result.compression_level == 3
        assert result.formatted_text == ""

    def test_budget_result_fields(self):
        atoms = [_make_atom(content="Test content.")]
        result = self.budget.compress_to_budget(atoms, budget_tokens=5000)
        assert isinstance(result, BudgetResult)
        assert result.budget_total == 5000
        assert result.budget_used > 0
        assert result.budget_used <= result.budget_total
        assert len(result.formatted_text) > 0


# ---------------------------------------------------------------------------
# _first_sentence
# ---------------------------------------------------------------------------

class TestFirstSentence:
    """Tests for the _first_sentence helper."""

    def test_period_followed_by_space(self):
        assert _first_sentence("Hello world. More text here.") == "Hello world."

    def test_exclamation_boundary(self):
        assert _first_sentence("Watch out! Danger ahead.") == "Watch out!"

    def test_question_boundary(self):
        assert _first_sentence("Is this working? Yes it is.") == "Is this working?"

    def test_no_boundary_short_text(self):
        assert _first_sentence("No sentence boundary here") == "No sentence boundary here"

    def test_no_boundary_truncated_at_max_chars(self):
        long_text = "a" * 200
        result = _first_sentence(long_text, max_chars=100)
        assert len(result) == 100

    def test_period_at_end_without_trailing_space(self):
        # Period at end of text with no following space: no boundary detected
        assert _first_sentence("Just one sentence.") == "Just one sentence."

    def test_newlines_normalized_to_spaces(self):
        assert _first_sentence("Line one.\n More text.") == "Line one."

    def test_empty_string(self):
        assert _first_sentence("") == ""


# ---------------------------------------------------------------------------
# Formatter output verification
# ---------------------------------------------------------------------------

class TestFormatters:
    """Tests for the four compression-level formatters."""

    def test_level0_full_detail(self):
        atom = _make_atom(
            type="fact",
            confidence=0.95,
            tags="redis, caching",
            content="Redis SCAN is O(N).",
            synapses=[
                {"relationship": "related-to", "content": "Cluster sharding", "strength": 0.8},
            ],
        )
        text = ContextBudget._format_atom_level0(atom)
        assert text.startswith("[fact] (confidence: 0.95, tags: redis, caching)")
        assert "Redis SCAN is O(N)." in text
        assert '-> related-to: "Cluster sharding" (strength: 0.8)' in text

    def test_level0_no_tags_no_synapses(self):
        atom = _make_atom(type="insight", confidence=0.7, tags="", content="Something.")
        text = ContextBudget._format_atom_level0(atom)
        assert text.startswith("[insight] (confidence: 0.7)")
        assert "tags:" not in text
        assert "->" not in text

    def test_level1_content_only(self):
        atom = _make_atom(type="skill", confidence=0.85, content="Use pytest fixtures.")
        text = ContextBudget._format_atom_level1(atom)
        assert text == "[skill|0.85] Use pytest fixtures."

    def test_level2_summary(self):
        atom = _make_atom(content="First sentence here. Second sentence follows.")
        text = ContextBudget._format_atom_level2(atom)
        assert text == "- First sentence here."

    def test_level2_no_boundary(self):
        atom = _make_atom(content="No sentence boundary at all")
        text = ContextBudget._format_atom_level2(atom)
        assert text == "- No sentence boundary at all"

    def test_level3_short_content(self):
        atom = _make_atom(type="preference", content="Short.")
        text = ContextBudget._format_atom_level3(atom)
        assert text == "* [preference] Short."
        assert "..." not in text

    def test_level3_long_content_truncated(self):
        atom = _make_atom(type="fact", content="A" * 80)
        text = ContextBudget._format_atom_level3(atom)
        assert text.startswith("* [fact] ")
        assert text.endswith("...")
        # 9 chars for "* [fact] ", then 50 chars, then "..."
        body = text[len("* [fact] "):]
        assert body == "A" * 50 + "..."
