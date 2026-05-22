"""Unit tests for audit_brain verdict logic.

The verdict text is what users read after a 30-minute audit. Make sure
the branches behave correctly without spending 30 minutes on each.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from lattice.benchmarks.audit_brain import AuditRow, _print_audit


def _row(name, on_pass, off_pass, on_steps, off_steps, runs=2, tier="llm"):
    return AuditRow(
        task_name=name,
        tier=tier,
        runs=runs,
        brain_on_passes=on_pass,
        brain_off_passes=off_pass,
        brain_on_steps=on_steps,
        brain_off_steps=off_steps,
        brain_on_time_s=0.0,
        brain_off_time_s=0.0,
    )


def _verdict_of(rows):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_audit(rows)
    return buf.getvalue()


def test_verdict_helps_when_passes_increase():
    rows = [_row("t1", on_pass=2, off_pass=1, on_steps=4, off_steps=4)]
    out = _verdict_of(rows)
    assert "HELPS" in out
    assert "+1 passes" in out


def test_verdict_hurts_when_passes_decrease():
    rows = [_row("t1", on_pass=0, off_pass=2, on_steps=4, off_steps=4)]
    out = _verdict_of(rows)
    assert "HURTS" in out
    assert "2 fewer" in out


def test_verdict_helps_via_step_efficiency_when_passes_tied():
    """Pass rates equal but brain ON uses fewer total steps.

    This is the key new signal — binary pass/fail can't detect
    'brain converged in 2 cycles instead of 4'. Step delta does.
    """
    rows = [_row("t1", on_pass=2, off_pass=2, on_steps=4, off_steps=8)]
    out = _verdict_of(rows)
    assert "HELPS via step efficiency" in out
    assert "4 fewer total steps" in out


def test_verdict_hurts_via_step_inefficiency_when_passes_tied():
    rows = [_row("t1", on_pass=2, off_pass=2, on_steps=10, off_steps=4)]
    out = _verdict_of(rows)
    assert "HURTS via step inefficiency" in out
    assert "6 MORE total steps" in out


def test_verdict_decoration_when_both_tied():
    """Pass rates AND step counts identical — the strict decoration case."""
    rows = [_row("t1", on_pass=2, off_pass=2, on_steps=4, off_steps=4)]
    out = _verdict_of(rows)
    assert "DECORATION on this task set" in out
    # The diagnostic checklist points users at LATTICE_BRAIN_DEBUG.
    assert "LATTICE_BRAIN_DEBUG=1" in out


def test_verdict_decoration_pattern_only_is_expected():
    """Pattern-only audits are STRUCTURALLY tied — that's not a bug."""
    rows = [_row("p1", on_pass=2, off_pass=2, on_steps=4, off_steps=4, tier="pattern")]
    out = _verdict_of(rows)
    assert "EXPECTED" in out
    assert "Pattern proposer is deterministic" in out


def test_step_count_per_task_printed():
    """The per-row table now shows on-steps / off-steps averages."""
    rows = [_row("t1", on_pass=2, off_pass=2, on_steps=6, off_steps=10, runs=2)]
    out = _verdict_of(rows)
    # average on steps = 6/2 = 3.0, average off steps = 10/2 = 5.0
    assert "3.0" in out
    assert "5.0" in out
    # And the net step delta gets printed in the overall block.
    assert "NET step delta" in out
