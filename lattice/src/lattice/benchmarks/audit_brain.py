"""Brain-effect audit — does the substrate actually pay for itself?

Runs the benchmark twice in-process: once with the brain on (priming
hints, recall, apprentice templates all active), once with --no-brain
(everything brain-side off). Reports per-task pass-rate and timing
deltas so we can SEE whether atoms move outcomes.

Usage:

    python -m lattice.benchmarks.audit_brain [--repeat N] [--tier TIER]

The audit is meaningful primarily for the LLM tier — the pattern tier
runs the deterministic regex path regardless of brain state, so deltas
there are noise. Default: --tier llm --repeat 2.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from lattice.benchmarks.run import _run_one_inprocess, _setup_workspace
from lattice.benchmarks.tasks import TASKS, BenchTask


@dataclass
class AuditRow:
    task_name: str
    tier: str
    brain_on_passes: int = 0
    brain_off_passes: int = 0
    brain_on_time_s: float = 0.0
    brain_off_time_s: float = 0.0
    brain_on_failures: list[str] = field(default_factory=list)
    brain_off_failures: list[str] = field(default_factory=list)
    runs: int = 0


def _run_one_with_brain_flag(task: BenchTask, use_brain: bool, llm_cache: dict):
    """Wrap _run_one_inprocess so brain on/off is toggled per run."""
    # The simplest path: build the task's CLI flags with --no-brain
    # appended when use_brain is False. _run_one_inprocess reads
    # the flags through _parse_task_flags which already understands
    # the agent CLI flags.
    flags = task.flags
    if not use_brain and "--no-brain" not in flags:
        flags = (*flags, "--no-brain")
    shadowed = BenchTask(
        name=task.name,
        tier=task.tier,
        task=task.task,
        files=dict(task.files),
        expects=dict(task.expects),
        not_expects=dict(task.not_expects),
        flags=flags,
        timeout_s=task.timeout_s,
    )
    return _run_one_inprocess(shadowed, llm_cache)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lattice-bench-audit-brain")
    parser.add_argument(
        "--tier", default="llm",
        choices=("all", "pattern", "llm", "research"),
        help="Which tier to audit (default: llm — pattern is brain-independent).",
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--name", default="", help="Substring filter for task names.")
    args = parser.parse_args(argv)

    selected = TASKS
    if args.tier != "all":
        selected = tuple(t for t in selected if t.tier == args.tier)
    if args.name:
        selected = tuple(t for t in selected if args.name in t.name)
    if not selected:
        sys.stderr.write("(no tasks selected)\n")
        return 0

    sys.stderr.write(
        f"auditing brain effect on {len(selected)} task(s) "
        f"x {args.repeat} repeat(s) x 2 conditions = "
        f"{len(selected) * args.repeat * 2} runs\n"
    )

    # Separate caches so brain-on and brain-off don't share apprentice
    # state through the LLM proposer's cached weights (model weights
    # are safe to share; the apprentice store is per-run anyway).
    llm_cache_on: dict = {}
    llm_cache_off: dict = {}
    rows: list[AuditRow] = []

    for task in selected:
        row = AuditRow(task_name=task.name, tier=task.tier, runs=args.repeat)
        for i in range(args.repeat):
            sys.stderr.write(f"  -> {task.name} run {i + 1}/{args.repeat} ... ")
            sys.stderr.flush()

            t0 = time.time()
            res_on = _run_one_with_brain_flag(task, use_brain=True, llm_cache=llm_cache_on)
            row.brain_on_time_s += time.time() - t0
            if res_on.passed:
                row.brain_on_passes += 1
            else:
                row.brain_on_failures.append(res_on.detail[:200])

            t0 = time.time()
            res_off = _run_one_with_brain_flag(task, use_brain=False, llm_cache=llm_cache_off)
            row.brain_off_time_s += time.time() - t0
            if res_off.passed:
                row.brain_off_passes += 1
            else:
                row.brain_off_failures.append(res_off.detail[:200])

            sys.stderr.write(
                f"on={'P' if res_on.passed else 'F'} "
                f"off={'P' if res_off.passed else 'F'}\n"
            )
        rows.append(row)

    _print_audit(rows)
    return 0


def _print_audit(rows: list[AuditRow]) -> None:
    name_w = max((len(r.task_name) for r in rows), default=8)
    print(
        f"\n{'task':<{name_w}}  {'tier':<8}  "
        f"{'brain on':>10}  {'brain off':>10}  "
        f"{'on avg s':>9}  {'off avg s':>10}  delta"
    )
    print("-" * (name_w + 70))
    for r in rows:
        on_rate = f"{r.brain_on_passes}/{r.runs}"
        off_rate = f"{r.brain_off_passes}/{r.runs}"
        on_avg = r.brain_on_time_s / max(1, r.runs)
        off_avg = r.brain_off_time_s / max(1, r.runs)
        delta = r.brain_on_passes - r.brain_off_passes
        marker = ""
        if delta > 0:
            marker = "  ← brain helped"
        elif delta < 0:
            marker = "  ← brain hurt"
        print(
            f"{r.task_name:<{name_w}}  {r.tier:<8}  "
            f"{on_rate:>10}  {off_rate:>10}  "
            f"{on_avg:>8.1f}s  {off_avg:>9.1f}s  {delta:+d}{marker}"
        )

    total_on = sum(r.brain_on_passes for r in rows)
    total_off = sum(r.brain_off_passes for r in rows)
    total_runs = sum(r.runs for r in rows)
    print()
    print(f"OVERALL brain ON:  {total_on}/{total_runs}")
    print(f"OVERALL brain OFF: {total_off}/{total_runs}")
    print(f"NET DELTA:         {total_on - total_off:+d}")

    on_time = sum(r.brain_on_time_s for r in rows)
    off_time = sum(r.brain_off_time_s for r in rows)
    print(f"\ntotal on-time:  {on_time:.1f}s")
    print(f"total off-time: {off_time:.1f}s")
    print(f"time delta:     {on_time - off_time:+.1f}s")

    if total_on == total_off:
        print(
            "\nVerdict: brain is currently DECORATION on this task set. "
            "No outcome shift. Needs work before more verbs are added."
        )
    elif total_on > total_off:
        print(
            f"\nVerdict: brain HELPS — +{total_on - total_off} passes "
            f"vs brain-off baseline. Substrate pays for itself."
        )
    else:
        print(
            f"\nVerdict: brain HURTS — {total_off - total_on} fewer "
            "passes when on. The current brain wiring is net negative; "
            "fix before building further."
        )


if __name__ == "__main__":
    raise SystemExit(main())
