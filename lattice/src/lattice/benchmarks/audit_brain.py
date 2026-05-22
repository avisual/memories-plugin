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
import shutil
import sys
import tempfile
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
    # Cumulative step counts across all runs of this task. Step-delta
    # is the second-tier signal (after pass-rate): if brain ON converges
    # in fewer cycles even when both ON and OFF eventually pass, that's
    # measurable brain help that the binary pass/fail misses.
    brain_on_steps: int = 0
    brain_off_steps: int = 0


def _run_one_with_brain_flag(
    task: BenchTask,
    use_brain: bool,
    llm_cache: dict,
    brain_db_path: Path | None,
):
    """Wrap _run_one_inprocess so brain on/off is toggled per run.

    Passes `brain_db_path` straight through to the runner so the audit
    can persist atoms across runs in the same condition. Without that
    persistence, brain decision-weighting has no data to act on and
    any 'brain helps' verdict gets masked by the fresh-store baseline.
    """
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
    return _run_one_inprocess(shadowed, llm_cache, brain_db_path=brain_db_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lattice-bench-audit-brain")
    parser.add_argument(
        "--tier", default="llm",
        choices=("all", "pattern", "llm", "research"),
        help="Which tier to audit (default: llm — pattern is brain-independent).",
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--name", default="", help="Substring filter for task names.")
    parser.add_argument(
        "--fresh-brain",
        action="store_true",
        help=(
            "Use a fresh brain DB per task (default: persistent across "
            "tasks in the same condition). Persistent is the meaningful "
            "audit — atoms accumulated in earlier runs are what the "
            "decision-weighting actually scores future candidates against."
        ),
    )
    parser.add_argument(
        "--keep-brain-db",
        default="",
        help=(
            "Path to a directory where the brain-on.db and brain-off.db "
            "files should be COPIED at the end of the audit (before the "
            "tmp dir is removed). Without this, the audit DBs disappear "
            "as soon as the verdict prints — making post-hoc `brain "
            "audit-state` introspection impossible. Specify a directory; "
            "the audit creates it if missing."
        ),
    )
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

    # PERSISTENT BRAIN ACROSS RUNS (in the same condition).
    # Without this, every run starts with a fresh brain.db — the
    # decision-weighting has nothing to score against, the audit only
    # tests the (modest) effect of priming + recall hints on the LLM
    # prompt. With shared persistence, atoms written during run N are
    # visible when scoring candidates in run N+1, which is the actual
    # mechanism by which the brain pays for itself.
    audit_root: Path | None = None
    brain_on_path: Path | None = None
    brain_off_path: Path | None = None
    if not args.fresh_brain:
        audit_root = Path(tempfile.mkdtemp(prefix="lattice-audit-"))
        brain_on_path = audit_root / "brain-on.db"
        brain_off_path = audit_root / "brain-off.db"
        sys.stderr.write(
            f"persistent brain mode: on={brain_on_path} off={brain_off_path}\n"
        )

    try:
        for task in selected:
            row = AuditRow(task_name=task.name, tier=task.tier, runs=args.repeat)
            for i in range(args.repeat):
                sys.stderr.write(f"  -> {task.name} run {i + 1}/{args.repeat} ... ")
                sys.stderr.flush()

                t0 = time.time()
                res_on = _run_one_with_brain_flag(
                    task,
                    use_brain=True,
                    llm_cache=llm_cache_on,
                    brain_db_path=brain_on_path,
                )
                row.brain_on_time_s += time.time() - t0
                row.brain_on_steps += getattr(res_on, "step_count", 0)
                if res_on.passed:
                    row.brain_on_passes += 1
                else:
                    row.brain_on_failures.append(res_on.detail[:200])

                t0 = time.time()
                res_off = _run_one_with_brain_flag(
                    task,
                    use_brain=False,
                    llm_cache=llm_cache_off,
                    brain_db_path=brain_off_path,
                )
                row.brain_off_time_s += time.time() - t0
                row.brain_off_steps += getattr(res_off, "step_count", 0)
                if res_off.passed:
                    row.brain_off_passes += 1
                else:
                    row.brain_off_failures.append(res_off.detail[:200])

                sys.stderr.write(
                    f"on={'P' if res_on.passed else 'F'}({res_on.step_count}s) "
                    f"off={'P' if res_off.passed else 'F'}({res_off.step_count}s)\n"
                )
            rows.append(row)
    finally:
        if audit_root is not None and audit_root.exists():
            if args.keep_brain_db:
                keep_dir = Path(args.keep_brain_db)
                keep_dir.mkdir(parents=True, exist_ok=True)
                for db in (brain_on_path, brain_off_path):
                    if db is not None and db.exists():
                        shutil.copy2(db, keep_dir / db.name)
                sys.stderr.write(
                    f"copied brain DBs to {keep_dir.resolve()}\n"
                )
            shutil.rmtree(audit_root, ignore_errors=True)

    _print_audit(rows)
    return 0


def _print_audit(rows: list[AuditRow]) -> None:
    name_w = max((len(r.task_name) for r in rows), default=8)
    print(
        f"\n{'task':<{name_w}}  {'tier':<8}  "
        f"{'brain on':>10}  {'brain off':>10}  "
        f"{'on steps':>9}  {'off steps':>10}  delta"
    )
    print("-" * (name_w + 70))
    for r in rows:
        on_rate = f"{r.brain_on_passes}/{r.runs}"
        off_rate = f"{r.brain_off_passes}/{r.runs}"
        on_steps_avg = r.brain_on_steps / max(1, r.runs)
        off_steps_avg = r.brain_off_steps / max(1, r.runs)
        delta = r.brain_on_passes - r.brain_off_passes
        marker = ""
        if delta > 0:
            marker = "  ← brain helped"
        elif delta < 0:
            marker = "  ← brain hurt"
        print(
            f"{r.task_name:<{name_w}}  {r.tier:<8}  "
            f"{on_rate:>10}  {off_rate:>10}  "
            f"{on_steps_avg:>8.1f}   {off_steps_avg:>8.1f}   {delta:+d}{marker}"
        )

    total_on = sum(r.brain_on_passes for r in rows)
    total_off = sum(r.brain_off_passes for r in rows)
    total_runs = sum(r.runs for r in rows)
    total_on_steps = sum(r.brain_on_steps for r in rows)
    total_off_steps = sum(r.brain_off_steps for r in rows)
    print()
    print(f"OVERALL brain ON:  {total_on}/{total_runs}")
    print(f"OVERALL brain OFF: {total_off}/{total_runs}")
    print(f"NET pass delta:    {total_on - total_off:+d}")
    # Step delta — second-tier brain signal. Negative means brain ON
    # converged in fewer total steps across all runs (good — brain
    # cuts cycles even when both eventually pass).
    print(
        f"NET step delta:    {total_on_steps - total_off_steps:+d} "
        f"(ON={total_on_steps} OFF={total_off_steps})"
    )

    on_time = sum(r.brain_on_time_s for r in rows)
    off_time = sum(r.brain_off_time_s for r in rows)
    print(f"\ntotal on-time:  {on_time:.1f}s")
    print(f"total off-time: {off_time:.1f}s")
    print(f"time delta:     {on_time - off_time:+.1f}s")

    # Pattern-only audits are STRUCTURALLY expected to show no
    # outcome delta — the pattern proposer routes by deterministic
    # regex and the brain cannot move what it cannot influence.
    # An LLM-tier (or mixed) audit with no delta is the meaningful
    # signal that the brain wiring isn't paying for itself.
    tiers_present = {r.tier for r in rows}
    pattern_only = tiers_present == {"pattern"}

    if total_on == total_off:
        if pattern_only:
            print(
                "\nVerdict: brain DECORATION on pattern tier — EXPECTED. "
                "Pattern proposer is deterministic; brain cannot move "
                "outcomes here. Re-run with --tier llm for the meaningful "
                "signal."
            )
        elif total_on_steps < total_off_steps:
            # Pass-rate tied but brain cut cycles. Real signal — brain
            # CAN'T flip a passing task to a failing one or vice versa,
            # so the only place its effect shows up is convergence
            # speed. Negative step delta is the brain-helps proxy
            # when both conditions clear the binary bar.
            saved = total_off_steps - total_on_steps
            pct = 100.0 * saved / max(1, total_off_steps)
            print(
                f"\nVerdict: brain HELPS via step efficiency — "
                f"{saved} fewer total steps with brain ON ({pct:.0f}%). "
                "Pass rates tied (both clear the bar), but brain ON "
                "converges in fewer cycles. Real signal even though "
                "binary pass-rate didn't move."
            )
        elif total_on_steps > total_off_steps:
            extra = total_on_steps - total_off_steps
            pct = 100.0 * extra / max(1, total_off_steps)
            print(
                f"\nVerdict: brain HURTS via step inefficiency — "
                f"{extra} MORE total steps with brain ON ({pct:.0f}%). "
                "Pass rates tied. Brain ON is adding cycles without "
                "yielding pass-rate benefit — the recall+priming is "
                "currently distracting the loop rather than focusing it."
            )
        else:
            print(
                "\nVerdict: brain DECORATION on this task set. "
                "Pass rates AND step counts identical. Either the "
                "tasks are too easy for the LLM (brain irrelevant), "
                "the persistent-brain isn't accumulating useful atoms, "
                "or the decision-weighting isn't tipping the choices "
                "enough to flip results. Diagnostics:\n"
                "  1. Re-run with LATTICE_BRAIN_DEBUG=1 to see per-cycle "
                "brain_delta on each candidate and whether 'BRAIN FLIPPED' "
                "fired.\n"
                "  2. Inspect what got written with\n"
                "     lattice brain audit-state --db <keep-brain-db>/brain-on.db\n"
                "  3. Add tasks where the small model SOMETIMES fails — "
                "without failures there's no ANTIPATTERN atom for the "
                "brain to discourage wrong choices with."
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
