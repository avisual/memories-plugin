"""Benchmark runner — measures lattice on the task corpus.

Usage:

    python -m lattice.benchmarks.run            # all tasks, subprocess mode
    python -m lattice.benchmarks.run pattern    # only pattern-tier (fast)
    python -m lattice.benchmarks.run llm        # only LLM-tier (slow)
    python -m lattice.benchmarks.run research   # only research-tier (slowest)
    python -m lattice.benchmarks.run --in-process  # share model + brain
                                                    # across tasks; ~30x faster
                                                    # on LLM/research tiers
    python -m lattice.benchmarks.run --repeat 5    # run each task 5 times,
                                                    # report pass-rate
    python -m lattice.benchmarks.run --json        # JSON-lines for CI

Each task runs in a fresh tempdir workspace. The atom store ships
seeded but otherwise fresh per task. In --in-process mode, the LLM
model is loaded ONCE and reused across all tasks (huge speedup);
each task still gets an isolated workspace + brain.

Exit code = number of failed task-runs (task * repeats).
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
from typing import Any

from lattice.benchmarks.tasks import TASKS, BenchTask


@dataclass
class TaskResult:
    task: BenchTask
    passed: bool
    elapsed_s: float
    detail: str = ""
    final_files: dict[str, str] = field(default_factory=dict)
    run_idx: int = 0  # 0-indexed when --repeat > 1
    # Steps the agent loop took before terminating. With brain ON,
    # a well-primed loop may converge in fewer cycles even when
    # both ON and OFF eventually pass — this captures that delta.
    # 0 when the run errored before producing a trace.
    step_count: int = 0


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


def _run_one_subprocess(task: BenchTask) -> TaskResult:
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


def _parse_task_flags(flags: tuple[str, ...]) -> dict[str, str | bool]:
    """Translate the subprocess-style flag list into an in-process config dict."""
    out: dict[str, str | bool] = {
        "two_stage": False,
        "decompose": False,
        "hosted": False,
        "model": "",
        "executor_model": "",
    }
    it = iter(flags)
    for tok in it:
        if tok == "--two-stage":
            out["two_stage"] = True
        elif tok == "--decompose":
            out["decompose"] = True
        elif tok == "--hosted":
            out["hosted"] = True
        elif tok == "--model":
            out["model"] = next(it)
        elif tok == "--executor-model":
            out["executor_model"] = next(it)
    return out


def _build_inproc_proposer(cfg: dict[str, str | bool], cache: dict) -> Any:
    """Build a Composite proposer, caching the LLM stage across calls.

    The Pattern proposer is cheap to recreate. The LocalLLM /
    TwoStage proposer loads transformers + a model — expensive — so
    we cache by (model, executor_model, two_stage) and reuse.
    """
    from lattice.propose import CompositeProposer, PatternProposer

    key = (cfg.get("model", ""), cfg.get("executor_model", ""), bool(cfg.get("two_stage")))
    if key not in cache:
        if cfg["two_stage"]:
            from lattice.propose.two_stage import TwoStageProposer

            cache[key] = TwoStageProposer(
                planner_model=cfg["model"] or None,
                executor_model=cfg["executor_model"] or None,
            )
        else:
            from lattice.propose.local import LocalLLMProposer

            cache[key] = LocalLLMProposer(model_name=cfg["model"] or None)
    return CompositeProposer([PatternProposer(), cache[key]])


def _run_one_inprocess(
    task: BenchTask,
    llm_cache: dict,
    *,
    brain_db_path: Path | None = None,
) -> TaskResult:
    """Run a single task in-process, reusing cached LLM weights across tasks.

    When `brain_db_path` is provided, it overrides the per-task tempdir
    brain location. The audit harness uses this to persist a single
    brain.db across multiple task runs, so atoms accumulated in run N
    are available when scoring candidates in run N+1. Without that
    override, brain decision-weighting has no data to act on and any
    'brain helps' verdict is masked by the fresh-store condition.
    """
    from lattice.apply import write_final
    from lattice.atoms import SQLiteAtomStore, seed_store
    from lattice.compiler import FilesystemWorkspace
    from lattice.orchestrator import (
        AgentLoop,
        decompose,
        run_subtasks,
    )

    root = Path(tempfile.mkdtemp(prefix="lattice-bench-"))
    start = time.time()
    try:
        _setup_workspace(task, root)
        if brain_db_path is not None:
            brain_path = brain_db_path
        else:
            brain_path = root / ".lattice" / "brain.db"
        brain_path.parent.mkdir(parents=True, exist_ok=True)
        store = SQLiteAtomStore(brain_path)
        try:
            if store.count() == 0:
                seed_store(store)
            cfg = _parse_task_flags(task.flags)
            proposer = _build_inproc_proposer(cfg, llm_cache)
            workspace = FilesystemWorkspace(root)

            step_count = 0
            trace_ok = False  # Will gate experience-atom recording.
            if cfg["decompose"]:
                subtasks = decompose(task.task)
                report = run_subtasks(
                    subtasks,
                    proposer=proposer,
                    workspace=workspace,
                    atom_store=store,
                    max_steps_per_subtask=4,
                )
                final_files = report.final_files
                # Sum step counts across subtask traces. report.traces
                # is the list of per-subtask AgentTrace objects.
                step_count = sum(
                    len(getattr(t, "steps", ())) for t in getattr(report, "traces", ())
                )
                trace_ok = bool(getattr(report, "ok", False))
                trace_for_experience = report
            else:
                loop = AgentLoop(
                    proposer=proposer,
                    workspace=workspace,
                    atom_store=store,
                    max_steps=4,
                )
                trace = loop.run(task.task)
                final_files = trace.final_files
                step_count = len(trace.steps)
                trace_ok = trace.ok
                trace_for_experience = trace

            # WRITE EXPERIENCE + TRACE ATOMS on full success. Without
            # this, the audit's in-process runs never populate
            # region='traces' — so the apprentice can't fire on
            # subsequent runs, blocking the brain's biggest payoff
            # path (LLM-skipping template reuse). The CLI does this
            # already in __main__.py; the bench runner missed it.
            #
            # Gating on trace_ok matches the partial-success guard
            # we added to __main__.py — only fully-completed traces
            # become experience/trace atoms.
            if trace_ok and final_files:
                try:
                    from lattice.atoms.feedback import (
                        record_experience,
                        summarize_trace_for_experience,
                    )

                    actions, files = summarize_trace_for_experience(
                        trace_for_experience
                    )
                    if actions:
                        record_experience(
                            store=store,
                            task=task.task,
                            actions_summary=actions,
                            files_touched=files,
                        )
                except Exception:  # noqa: BLE001
                    # Bench experience recording is best-effort; never
                    # let it sink an otherwise-successful run.
                    pass

            if final_files:
                class _Report:
                    pass

                rep = _Report()
                rep.final_files = final_files  # type: ignore[attr-defined]
                write_final(rep, root=root)
        finally:
            store.close()

        elapsed = time.time() - start
        ok, detail, finals = _check_expectations(task, root)
        return TaskResult(
            task=task,
            passed=ok,
            elapsed_s=elapsed,
            detail=detail,
            final_files=finals,
            step_count=step_count,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start
        return TaskResult(
            task=task,
            passed=False,
            elapsed_s=elapsed,
            detail=f"in-process exception: {type(exc).__name__}: {exc}",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _print_table(results: list[TaskResult], *, repeat: int) -> None:
    name_w = max(len(r.task.name) for r in results)

    # When repeating, summarise per task (pass-rate).
    if repeat > 1:
        by_task: dict[str, list[TaskResult]] = {}
        for r in results:
            by_task.setdefault(r.task.name, []).append(r)
        print(
            f"\n{'name':<{name_w}}  {'tier':<8}  {'rate':<6}  "
            f"{'avg time':>9}  details"
        )
        print("-" * (name_w + 40 + 30))
        for name, rs in by_task.items():
            ok = sum(1 for r in rs if r.passed)
            tot = len(rs)
            avg = sum(r.elapsed_s for r in rs) / tot
            tier = rs[0].task.tier
            fail_msgs = [r.detail for r in rs if not r.passed]
            details = (fail_msgs[0][:120] + " ...") if fail_msgs else ""
            print(
                f"{name:<{name_w}}  {tier:<8}  {ok}/{tot:<3}  "
                f"{avg:>8.1f}s  {details}"
            )
    else:
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
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run each task N times; report pass-rate.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help=(
            "Run tasks in-process, sharing the LLM model across tasks. "
            "30x+ faster on LLM/research tiers (one cold model load, "
            "many warm inference runs)."
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

    n_runs = len(selected) * max(1, args.repeat)
    sys.stderr.write(
        f"running {len(selected)} task(s) x {args.repeat} repeat(s) = "
        f"{n_runs} run(s)"
        + (" (in-process)\n" if args.in_process else "\n")
    )

    llm_cache: dict = {}
    results: list[TaskResult] = []
    for task in selected:
        for i in range(max(1, args.repeat)):
            tag = f" run {i + 1}/{args.repeat}" if args.repeat > 1 else ""
            sys.stderr.write(f"  -> {task.name} [{task.tier}]{tag} ... ")
            sys.stderr.flush()
            r = (
                _run_one_inprocess(task, llm_cache)
                if args.in_process
                else _run_one_subprocess(task)
            )
            r.run_idx = i
            results.append(r)
            sys.stderr.write(
                f"{'PASS' if r.passed else 'FAIL'} ({r.elapsed_s:.1f}s)\n"
            )

    if args.json:
        _print_json(results)
    else:
        _print_table(results, repeat=args.repeat)

    return sum(1 for r in results if not r.passed)


if __name__ == "__main__":
    raise SystemExit(main())
