"""Benchmark runner — measures lattice on the task corpus.

Usage:

    python -m lattice.benchmarks.run            # all tasks, all tiers
    python -m lattice.benchmarks.run pattern    # only pattern-tier tasks (fast)
    python -m lattice.benchmarks.run llm        # only LLM-tier (slow)
    python -m lattice.benchmarks.run research   # only research-tier (slowest)
    python -m lattice.benchmarks.run --json     # JSON-lines output for CI

Each task runs in a fresh tempdir. Exit code = number of failed tasks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from lattice.benchmarks.tasks import TASKS, BenchTask


@dataclass
class TaskResult:
    task: BenchTask
    passed: bool
    elapsed_s: float
    detail: str = ""
    final_files: dict[str, str] = field(default_factory=dict)


def _setup_workspace(task: BenchTask, root: Path) -> None:
    for rel_path, content in task.files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _check_expectations(task: BenchTask, root: Path) -> tuple[bool, str, dict[str, str]]:
    finals: dict[str, str] = {}
    missing: list[str] = []
    forbidden: list[str] = []

    for rel_path, must_contain in task.expects.items():
        target = root / rel_path
        if not target.is_file():
            missing.append(f"file not produced: {rel_path}")
            continue
        text = target.read_text(encoding="utf-8")
        finals[rel_path] = text
        for needle in must_contain:
            if needle not in text:
                missing.append(f"{rel_path} missing substring: {needle!r}")

    for rel_path, must_not_contain in task.not_expects.items():
        target = root / rel_path
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8")
        finals.setdefault(rel_path, text)
        for needle in must_not_contain:
            if needle in text:
                forbidden.append(f"{rel_path} should not contain: {needle!r}")

    if missing or forbidden:
        return False, "; ".join(missing + forbidden), finals
    return True, "", finals


def _run_one(task: BenchTask) -> TaskResult:
    root = Path(tempfile.mkdtemp(prefix="lattice-bench-"))
    try:
        _setup_workspace(task, root)
        cmd = ["lattice", "do", task.task, "--write", "--max-steps", "4", *task.flags]
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=task.timeout_s,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return TaskResult(
                task=task,
                passed=False,
                elapsed_s=elapsed,
                detail=f"timeout after {task.timeout_s:.0f}s",
            )
        elapsed = time.time() - start

        ok, detail, finals = _check_expectations(task, root)
        if not ok and proc.returncode != 0:
            detail = f"{detail} | exit={proc.returncode} stderr_tail={proc.stderr[-200:]!r}"
        return TaskResult(
            task=task,
            passed=ok,
            elapsed_s=elapsed,
            detail=detail,
            final_files=finals,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _print_table(results: list[TaskResult]) -> None:
    name_w = max(len(r.task.name) for r in results)
    print(f"\n{'name':<{name_w}}  {'tier':<8}  {'pass':<5}  {'time':>7}  detail")
    print("-" * (name_w + 32 + 30))
    for r in results:
        symbol = "PASS" if r.passed else "FAIL"
        print(
            f"{r.task.name:<{name_w}}  {r.task.tier:<8}  {symbol:<5}  "
            f"{r.elapsed_s:>6.1f}s  {r.detail}"
        )

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    by_tier: dict[str, tuple[int, int]] = {}
    for r in results:
        ok, tot = by_tier.get(r.task.tier, (0, 0))
        by_tier[r.task.tier] = (ok + (1 if r.passed else 0), tot + 1)

    print()
    print(f"OVERALL: {passed}/{total}")
    for tier in ("pattern", "llm", "research"):
        if tier in by_tier:
            ok, tot = by_tier[tier]
            print(f"  {tier}: {ok}/{tot}")


def _print_json(results: list[TaskResult]) -> None:
    for r in results:
        sys.stdout.write(
            json.dumps(
                {
                    "name": r.task.name,
                    "tier": r.task.tier,
                    "passed": r.passed,
                    "elapsed_s": round(r.elapsed_s, 2),
                    "detail": r.detail,
                }
            )
            + "\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lattice-bench")
    parser.add_argument(
        "tier",
        nargs="?",
        default="all",
        choices=("all", "pattern", "llm", "research"),
        help="Which tier to run (default: all).",
    )
    parser.add_argument("--json", action="store_true", help="JSON-lines output.")
    parser.add_argument(
        "--name",
        default="",
        help="Run only the named task (substring match).",
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

    sys.stderr.write(f"running {len(selected)} task(s)...\n")
    results: list[TaskResult] = []
    for task in selected:
        sys.stderr.write(f"  -> {task.name} [{task.tier}] ... ")
        sys.stderr.flush()
        r = _run_one(task)
        results.append(r)
        sys.stderr.write(
            f"{'PASS' if r.passed else 'FAIL'} ({r.elapsed_s:.1f}s)\n"
        )

    if args.json:
        _print_json(results)
    else:
        _print_table(results)

    return sum(1 for r in results if not r.passed)


if __name__ == "__main__":
    raise SystemExit(main())
