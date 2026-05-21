"""Tests for the observation-hint context budget (Phase 3).

Hints accumulate per cycle (priming + atom recall + history +
workspace files + step trail). On a long multi-step run they can
exceed the small model's context. `_apply_context_budget` prunes
to a char budget while preserving priority — FIX REQUIRED + PLAN
must survive even when the budget is tight.
"""

from __future__ import annotations

from lattice.orchestrator.agent import _apply_context_budget, _hint_priority


def test_hint_priority_orders_classes_correctly():
    """The priority function returns numerically lower for more-important hints."""
    assert _hint_priority("FIX REQUIRED — broke parse") < _hint_priority("PLAN:")
    assert _hint_priority("PLAN:\n  [>] 1. step") < _hint_priority(
        "WORKSPACE FILES (use these EXACT paths"
    )
    assert _hint_priority("WORKSPACE FILES x") < _hint_priority(
        "skill (score 0.42): something"
    )
    assert _hint_priority("step 1 [edit] AddImport") < _hint_priority(
        "FILES ALREADY EDITED THIS TASK: src/x.py"
    )
    assert _hint_priority("FILES ALREADY EDITED") < _hint_priority(
        "Look at the history."
    )


def test_budget_no_op_when_within_limit():
    """A small list of hints stays untouched when total chars < budget."""
    hints = ["a", "b", "c"]
    assert _apply_context_budget(hints, max_chars=100) == ["a", "b", "c"]


def test_budget_drops_lowest_priority_first():
    """Out of budget, the 'Look at the history' coda drops before step trail."""
    hints = [
        "FIX REQUIRED — broken",  # priority 0
        "PLAN:\n  [>] 1. step",   # priority 1
        "step 1 [edit] AddImport",  # priority 6
        "Look at the history. If the task is complete, emit MarkDone.",  # priority 8
    ]
    # Budget tight enough to drop the lowest-priority hint but keep the rest.
    total = sum(len(h) for h in hints)
    pruned = _apply_context_budget(hints, max_chars=total - 20)
    # The 'Look at the history' line dropped first.
    assert not any("Look at the history" in h for h in pruned)
    # FIX REQUIRED, PLAN, step trail survived.
    assert any("FIX REQUIRED" in h for h in pruned)
    assert any("PLAN:" in h for h in pruned)
    assert any("step 1" in h for h in pruned)


def test_budget_preserves_original_order():
    """Pruned hints come back in their original order (priority is for
    *which* survive, not *how* they're rendered)."""
    hints = [
        "FIX REQUIRED — A",
        "low priority noise " * 20,
        "PLAN: B",
    ]
    pruned = _apply_context_budget(hints, max_chars=50)
    # FIX REQUIRED first (originally idx 0), PLAN second (originally idx 2).
    assert pruned[0].startswith("FIX REQUIRED")
    assert pruned[-1].startswith("PLAN:")


def test_budget_truncates_oversized_top_priority_hint():
    """If a FIX REQUIRED hint alone is bigger than the budget, truncate
    rather than drop — the model MUST see it to self-correct."""
    big = "FIX REQUIRED " + "X" * 5000
    hints = [big, "step 1 [edit] x", "Look at the history"]
    pruned = _apply_context_budget(hints, max_chars=500)
    assert pruned, "FIX REQUIRED should survive even when oversized"
    assert pruned[0].startswith("FIX REQUIRED")
    assert "[truncated]" in pruned[0]


def test_budget_skips_low_priority_when_high_priority_filled_budget():
    """If a high-priority hint consumes the budget, lower-priority hints drop."""
    hints = [
        "FIX REQUIRED — " + "X" * 200,
        "step 1 [edit] AddImport",
        "step 2 [edit] AddImport",
    ]
    pruned = _apply_context_budget(hints, max_chars=230)
    assert any(p.startswith("FIX REQUIRED") for p in pruned)
    # Steps should drop.
    assert not any(p.startswith("step ") for p in pruned)
