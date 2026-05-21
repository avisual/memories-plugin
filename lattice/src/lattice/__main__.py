"""Command-line driver for LATTICE.

Subcommands:

    python -m lattice apply   <workspace> [--action JSON | -]
    python -m lattice intent  <workspace> [--intent JSON | -] [--files a.py,b.py]
    python -m lattice propose <workspace> --task "..." [--model NAME]

`apply` takes a single typed Action and prints its diff.
`intent` takes a high-level Intent, expands it into many typed Actions
(the composition spine: intent → primitives), compiles and verifies
each, and prints a multi-file unified diff.
`propose` asks a small local LLM (default Qwen2.5-0.5B) to emit an
Action for a task, then runs compile + verify + diff. The full
non-text reasoning loop driven by a real model.

No filesystem mutation — diffs are printed, not applied.

Exit: 0 on success, 1 on compile/verify failure, 2 on input errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO

from pydantic import TypeAdapter, ValidationError

from lattice.actions import parse_action
from lattice.compiler import (
    CompileError,
    FilesystemWorkspace,
    compile_action,
)
from lattice.orchestrator import (
    ExecutionReport,
    Intent,
    execute_plan,
    expand_intent,
)
from lattice.propose import Proposer
from lattice.verify import verify_syntactic


def _read_action_json(arg: str | None, stdin: IO[str]) -> dict:
    if arg is None or arg == "-":
        raw = stdin.read()
    elif arg.startswith("@"):
        with open(arg[1:], encoding="utf-8") as fh:
            raw = fh.read()
    else:
        raw = arg
    raw = raw.strip()
    if not raw:
        raise SystemExit("error: empty action JSON on stdin")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: action JSON is malformed: {exc}") from exc


def _apply(args: argparse.Namespace) -> int:
    workspace = FilesystemWorkspace(args.workspace)
    payload = _read_action_json(args.action, sys.stdin)
    try:
        action = parse_action(payload)
    except ValidationError as exc:
        sys.stderr.write(f"action did not validate:\n{exc}\n")
        return 2

    try:
        compiled = compile_action(action, workspace)
    except CompileError as exc:
        sys.stderr.write(f"compile error ({type(exc).__name__}): {exc}\n")
        return 1

    outcome = verify_syntactic(compiled)
    if not outcome.ok:
        for path, msg in outcome.errors:
            sys.stderr.write(f"verify failed in {path}: {msg}\n")
        return 1

    if compiled.is_noop:
        sys.stderr.write("(no-op: action did not change the workspace)\n")
        return 0

    for change in compiled.file_changes:
        if change.diff:
            sys.stdout.write(change.diff)
            if not change.diff.endswith("\n"):
                sys.stdout.write("\n")

    if args.write:
        from lattice.apply import write_compiled

        written = write_compiled(compiled, root=args.workspace)
        sys.stderr.write(f"wrote {len(written)} file(s)\n")
    return 0


_intent_adapter: TypeAdapter[Intent] = TypeAdapter(Intent)


def _agent(args: argparse.Namespace) -> int:
    from lattice.atoms import (
        SQLiteAtomStore,
        record_experience,
        summarize_trace_for_experience,
    )
    from lattice.orchestrator import AgentLoop, decompose, run_subtasks

    workspace = FilesystemWorkspace(args.workspace)
    atom_store = SQLiteAtomStore(args.atom_db) if args.atom_db else None
    try:
        from lattice.distill import ApprenticeProposer
        from lattice.propose import CompositeProposer, PatternProposer

        rule_based: Proposer = PatternProposer()  # type: ignore[assignment]
        # When --no-brain is set, the apprentice never sees the store
        # — every learned-template / boosted-trace shortcut is off.
        apprentice_store = None if getattr(args, "no_brain", False) else atom_store
        apprentice: Proposer = ApprenticeProposer(  # type: ignore[assignment]
            atom_store=apprentice_store
        )

        if args.hosted:
            from lattice.propose.hosted import HostedLLMProposer

            sys.stderr.write(f"using hosted proposer ({args.model or 'default Anthropic model'})...\n")
            llm_proposer: Proposer = HostedLLMProposer(model=args.model)  # type: ignore[assignment]
        elif args.two_stage:
            from lattice.propose.two_stage import TwoStageProposer

            sys.stderr.write(
                "loading two-stage local LLMs (planner + executor)...\n"
            )
            llm_proposer = TwoStageProposer(  # type: ignore[assignment]
                planner_model=args.model,
                executor_model=args.executor_model,
            )
        else:
            from lattice.propose.local import LocalLLMProposer

            sys.stderr.write(f"loading local model{(' ' + args.model) if args.model else ''}...\n")
            llm_proposer = (
                LocalLLMProposer(model_name=args.model)
                if args.model
                else LocalLLMProposer()
            )

        # Chain order: hand-written patterns first (fast, deterministic),
        # then apprentice (learned patterns from boosted traces), then
        # the LLM (novel / open-ended interpretation).
        proposer = CompositeProposer([rule_based, apprentice, llm_proposer])

        if args.subtasks:
            subtasks = [s.strip() for s in args.subtasks.split("|") if s.strip()]
        elif args.decompose:
            subtasks = decompose(args.task)
        else:
            subtasks = []

        if subtasks:
            sys.stderr.write(f"running {len(subtasks)} subtask(s):\n")
            for i, st in enumerate(subtasks, 1):
                sys.stderr.write(f"  [{i}] {st}\n")
            report = run_subtasks(
                subtasks,
                proposer=proposer,
                workspace=workspace,
                atom_store=atom_store,
                max_steps_per_subtask=args.max_steps,
            )
            sys.stderr.write(f"\nterminated_by={report.terminated_by}\n")
            for i, trace in enumerate(report.traces, 1):
                sys.stderr.write(f"  subtask {i}: {trace.terminated_by}\n")
                for step in trace.steps:
                    sys.stderr.write(
                        f"    step {step.step} [{step.kind}] {step.verb}\n"
                    )
            for diff in report.consolidated_diffs:
                sys.stdout.write(diff)
                if not diff.endswith("\n"):
                    sys.stdout.write("\n")
            if args.write and report.final_files:
                from lattice.apply import write_final

                written = write_final(report, root=args.workspace)
                sys.stderr.write(f"wrote {len(written)} file(s)\n")

            if atom_store is not None and report.final_files:
                actions, files = summarize_trace_for_experience(report)
                if actions:
                    record_experience(
                        store=atom_store,
                        task=args.task or "; ".join(report.subtasks),
                        actions_summary=actions,
                        files_touched=files,
                    )
                    sys.stderr.write("recorded experience atom\n")
            return 0 if report.ok else 1

        code_search = None
        if getattr(args, "semble", False):
            from lattice.sense.semble_search import maybe_code_search

            code_search = maybe_code_search(args.workspace)
            if code_search is None:
                sys.stderr.write(
                    "warning: --semble requested but semble not importable; continuing without it\n"
                )

        loop = AgentLoop(
            proposer=proposer,
            workspace=workspace,
            atom_store=atom_store,
            max_steps=args.max_steps,
            type_check=args.types,
            run_tests=getattr(args, "tests", False),
            workspace_root=args.workspace,
            code_search=code_search,
            use_brain=not getattr(args, "no_brain", False),
        )
        trace = loop.run(args.task)

        sys.stderr.write(f"\nterminated_by={trace.terminated_by}\n")
        for step in trace.steps:
            sys.stderr.write(
                f"  step {step.step} [{step.kind}] {step.verb}"
                + (f" :: {step.error[:120]}" if step.error else "")
                + (f" :: {step.payload[:120]}" if step.payload else "")
                + "\n"
            )

        for diff in trace.consolidated_diffs:
            sys.stdout.write(diff)
            if not diff.endswith("\n"):
                sys.stdout.write("\n")

        if args.write and trace.final_files:
            from lattice.apply import write_final

            written = write_final(trace, root=args.workspace)
            sys.stderr.write(f"wrote {len(written)} file(s)\n")

        if atom_store is not None and trace.final_files:
            actions, files = summarize_trace_for_experience(trace)
            if actions:
                record_experience(
                    store=atom_store,
                    task=args.task,
                    actions_summary=actions,
                    files_touched=files,
                )
                sys.stderr.write("recorded experience atom\n")

        return 0 if trace.ok else 1
    finally:
        if atom_store is not None:
            atom_store.close()


def _atom_add(args: argparse.Namespace) -> int:
    from lattice.atoms import AtomType, SQLiteAtomStore

    try:
        atom_type = AtomType(args.type)
    except ValueError:
        sys.stderr.write(
            f"unknown atom type {args.type!r}; valid: {', '.join(t.value for t in AtomType)}\n"
        )
        return 2

    tags = tuple(t.strip() for t in args.tags.split(",") if t.strip())
    store = SQLiteAtomStore(args.db)
    try:
        atom = store.add(
            args.content,
            type=atom_type,
            region=args.region,
            tags=tags,
            importance=args.importance,
        )
    finally:
        store.close()
    sys.stdout.write(f"added atom id={atom.id} type={atom.type.value}\n")
    return 0


def _do(args: argparse.Namespace) -> int:
    """One-shot convenience: init if needed, then run agent."""
    from lattice.atoms import SQLiteAtomStore, seed_store

    root = Path(args.workspace).resolve()
    if not root.is_dir():
        sys.stderr.write(f"workspace is not a directory: {root}\n")
        return 2

    brain_path = (root / args.brain).resolve()
    brain_path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteAtomStore(brain_path)
    try:
        if store.count() == 0:
            sys.stderr.write("first run — seeding brain with built-in atom pack...\n")
            seed_store(store)
    finally:
        store.close()

    task = " ".join(args.task)
    # Forward to _agent via a stand-in namespace.
    agent_ns = argparse.Namespace(
        workspace=str(root),
        task=task,
        atom_db=str(brain_path),
        model=args.model,
        max_steps=args.max_steps,
        subtasks=args.subtasks,
        decompose=args.decompose,
        hosted=args.hosted,
        two_stage=args.two_stage,
        executor_model=args.executor_model,
        types=args.types,
        tests=args.tests,
        semble=args.semble,
        no_brain=args.no_brain,
        write=not args.no_write,
    )
    return _agent(agent_ns)


def _init(args: argparse.Namespace) -> int:
    from lattice.atoms import SQLiteAtomStore, seed_store

    root = Path(args.workspace).resolve()
    if not root.exists():
        sys.stderr.write(f"workspace does not exist: {root}\n")
        return 2
    if not root.is_dir():
        sys.stderr.write(f"workspace is not a directory: {root}\n")
        return 2

    brain_path = (root / args.brain).resolve()
    try:
        brain_path.relative_to(root)
    except ValueError:
        sys.stderr.write(f"brain path must be inside the workspace: {brain_path}\n")
        return 2
    brain_path.parent.mkdir(parents=True, exist_ok=True)

    if brain_path.exists():
        sys.stderr.write(f"brain already exists: {brain_path}\n")
    else:
        sys.stderr.write(f"creating brain at {brain_path}\n")

    store = SQLiteAtomStore(brain_path)
    try:
        seeded = 0
        if not args.no_seed and store.count() == 0:
            sys.stderr.write("loading seed atom pack ...\n")
            seeded = seed_store(store)
        total = store.count()
    finally:
        store.close()

    sys.stdout.write(
        f"lattice initialised\n"
        f"  workspace: {root}\n"
        f"  brain:     {brain_path}\n"
        f"  atoms:     {total} (seeded {seeded} this run)\n"
        f"\n"
        f"Try:\n"
        f"  lattice agent {root} --atom-db {brain_path} --task '<your task>'\n"
        f"  lattice brain inspect --db {brain_path} --task '<a task>'\n"
        f"  lattice brain dump    --db {brain_path}\n"
    )
    return 0


def _brain_inspect(args: argparse.Namespace) -> int:
    from lattice.atoms import SQLiteAtomStore

    store = SQLiteAtomStore(args.db)
    try:
        results = store.recall(args.task, k=args.k)
    finally:
        store.close()

    if not results:
        sys.stderr.write("(no atoms recalled)\n")
        return 0
    sys.stdout.write(f"brain hints for task: {args.task!r}\n")
    for r in results:
        sys.stdout.write(
            f"  [{r.score:.3f}] {r.atom.type.value:<12} {r.atom.region:<18} {r.atom.content}\n"
        )
    return 0


def _brain_dump(args: argparse.Namespace) -> int:
    from lattice.atoms import SQLiteAtomStore, stats

    store = SQLiteAtomStore(args.db)
    try:
        layout = stats(store)
        total = store.count()
    finally:
        store.close()

    sys.stdout.write(f"brain contains {total} atom(s)\n")
    for region, counts in sorted(layout.items()):
        region_total = sum(counts.values())
        label = region or "(no region)"
        sys.stdout.write(f"  {label}: {region_total}\n")
        for atype, count in sorted(counts.items()):
            sys.stdout.write(f"    - {atype}: {count}\n")
    return 0


def _brain_import(args: argparse.Namespace) -> int:
    from lattice.atoms import SQLiteAtomStore, import_atoms

    store = SQLiteAtomStore(args.db)
    try:
        added = import_atoms(store, args.file)
    finally:
        store.close()
    sys.stdout.write(f"imported {added} atom(s) from {args.file}\n")
    return 0


def _brain_export(args: argparse.Namespace) -> int:
    from lattice.atoms import SQLiteAtomStore, export_atoms

    store = SQLiteAtomStore(args.db)
    try:
        count = export_atoms(store, args.file)
    finally:
        store.close()
    sys.stdout.write(f"exported {count} atom(s) to {args.file}\n")
    return 0


def _brain_audit_state(args: argparse.Namespace) -> int:
    """Snapshot the brain's actual state — the kind of view you want
    to read AFTER an audit to understand what the system learned.

    Reports:
      - total atoms, breakdown by region/type
      - importance histogram in 0.1 buckets (so you can see drift
        from the seed baseline)
      - top-N hottest atoms (highest importance) per region
      - count of step-outcome atoms (the ones the brain decision-
        weighting actually consults) by SKILL / ANTIPATTERN

    This is the introspection that lets you answer 'is the brain
    accumulating useful signal?' without staring at SQL.
    """
    import sqlite3

    db = args.db
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT id, content, type, region, importance, access_count, "
            "created_at FROM atoms"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        sys.stdout.write(f"(brain at {db} is empty)\n")
        return 0

    total = len(rows)
    by_type: dict[str, int] = {}
    by_region: dict[str, int] = {}
    importance_bins = [0] * 10  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
    for _id, _content, atype, region, importance, _ac, _ca in rows:
        by_type[atype] = by_type.get(atype, 0) + 1
        by_region[region or "(none)"] = by_region.get(region or "(none)", 0) + 1
        idx = min(9, max(0, int(importance * 10)))
        importance_bins[idx] += 1

    sys.stdout.write(f"brain at {db}\n")
    sys.stdout.write(f"  total atoms:    {total}\n\n")

    sys.stdout.write("by type:\n")
    for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]):
        sys.stdout.write(f"  {t:<14} {c}\n")

    sys.stdout.write("\nby region:\n")
    for r, c in sorted(by_region.items(), key=lambda kv: -kv[1]):
        sys.stdout.write(f"  {r:<22} {c}\n")

    sys.stdout.write("\nimportance histogram (bucket=0.1):\n")
    max_bin = max(importance_bins) or 1
    for i, count in enumerate(importance_bins):
        lo, hi = i / 10, (i + 1) / 10
        bar = "#" * int(40 * count / max_bin)
        sys.stdout.write(f"  [{lo:.1f}-{hi:.1f})  {count:>4}  {bar}\n")

    # Step-outcome breakdown — the atoms brain_score actually consults.
    step_atoms = [r for r in rows if r[3] == "steps"]
    if step_atoms:
        sys.stdout.write(
            f"\nstep-outcome atoms (region='steps'):  {len(step_atoms)}\n"
        )
        skills = sum(1 for r in step_atoms if r[2] == "skill")
        antis = sum(1 for r in step_atoms if r[2] == "antipattern")
        sys.stdout.write(f"  skill (successes):       {skills}\n")
        sys.stdout.write(f"  antipattern (failures):  {antis}\n")

    # Top-N hottest atoms (highest importance, after Hebbian updates).
    n = args.top
    sys.stdout.write(f"\ntop {n} hottest atoms (highest importance):\n")
    rows_sorted = sorted(rows, key=lambda r: -r[4])
    for _id, content, atype, region, importance, ac, _ca in rows_sorted[:n]:
        snippet = content[:80].replace("\n", " ")
        sys.stdout.write(
            f"  [imp={importance:.2f} ac={ac:>3} {atype:<11} {region or '-':<14}] {snippet}\n"
        )
    return 0


def _evolve(args: argparse.Namespace) -> int:
    from lattice.atoms import SQLiteAtomStore, boost_recurrent_traces, discover

    store = SQLiteAtomStore(args.db)
    try:
        report = discover(store, min_recurrence=args.min_recurrence)
        boosted = 0
        if args.apply:
            boosted = boost_recurrent_traces(
                store, min_recurrence=args.min_recurrence
            )
    finally:
        store.close()

    sys.stdout.write(
        f"scanned {report.total_traces} trace(s); "
        f"{len(report.candidates)} macro candidate(s) at "
        f"min_recurrence={args.min_recurrence}\n"
    )
    if not report.candidates:
        sys.stdout.write(
            "(no recurring patterns yet — run agent tasks then come back)\n"
        )
        return 0
    for i, cand in enumerate(report.candidates, 1):
        sys.stdout.write(
            f"\n  [{i}] x{cand.sample_count}  "
            f"sequence: {' → '.join(cand.action_sequence)}\n"
        )
        for st in cand.sample_tasks[:3]:
            sys.stdout.write(f"      sample task: {st[:160]}\n")
    if args.apply:
        sys.stdout.write(
            f"\napplied: boosted importance on {boosted} trace atom(s); "
            "future recall will surface them as stronger hints.\n"
        )
    return 0


def _atom_seed(args: argparse.Namespace) -> int:
    from lattice.atoms import SQLiteAtomStore, seed_store

    store = SQLiteAtomStore(args.db)
    try:
        added = seed_store(store)
    finally:
        store.close()
    sys.stdout.write(f"seeded {added} atom(s) into {args.db}\n")
    return 0


def _atom_recall(args: argparse.Namespace) -> int:
    from lattice.atoms import SQLiteAtomStore

    store = SQLiteAtomStore(args.db)
    try:
        results = store.recall(args.query, k=args.k)
    finally:
        store.close()
    if not results:
        sys.stderr.write("(no atoms recalled)\n")
        return 0
    for r in results:
        sys.stdout.write(
            f"[{r.score:.3f}] {r.atom.type.value:<12} {r.atom.content}\n"
        )
    return 0


def _propose(args: argparse.Namespace) -> int:
    from lattice.orchestrator import execute_plan
    from lattice.propose import ObservationContext
    from lattice.propose.local import LocalLLMProposer
    from lattice.sense import walk_workspace

    workspace = FilesystemWorkspace(args.workspace)
    files = (
        [p.strip() for p in args.files.split(",") if p.strip()]
        if args.files
        else workspace.iter_files(suffix=".py")
    )
    symbols = walk_workspace(workspace, files)

    hints: tuple[str, ...] = ()
    if args.atom_db:
        from lattice.atoms import AtomType, SQLiteAtomStore

        store = SQLiteAtomStore(args.atom_db)
        try:
            results = store.recall(args.task, k=args.hint_count)
            antipattern_results = store.recall(
                args.task, k=2, types=(AtomType.ANTIPATTERN,)
            )
        finally:
            store.close()

        hint_list: list[str] = []
        for res in antipattern_results:
            hint_list.append(f"antipattern (score {res.score:.2f}): {res.atom.content}")
        seen = {r.atom.id for r in antipattern_results}
        for res in results:
            if res.atom.id in seen:
                continue
            hint_list.append(
                f"{res.atom.type.value} (score {res.score:.2f}): {res.atom.content}"
            )
        hints = tuple(hint_list[: args.hint_count])
        if hints:
            sys.stderr.write(f"recalled {len(hints)} hint(s) from atom store\n")

    obs = ObservationContext(
        task=args.task, symbols=tuple(symbols[:30]), hints=hints
    )
    sys.stderr.write(f"loading model{(' ' + args.model) if args.model else ''}...\n")
    proposer = LocalLLMProposer(model_name=args.model) if args.model else LocalLLMProposer()

    try:
        actions = proposer.propose(obs)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"proposer failed: {type(exc).__name__}: {exc}\n")
        return 1
    if not actions:
        sys.stderr.write("(proposer returned no actions)\n")
        return 1

    sys.stderr.write(f"proposed action:\n  {actions[0].model_dump_json()}\n")

    report = execute_plan(actions, workspace)
    if not report.ok:
        for step in report.steps:
            if step.ok:
                continue
            if step.error:
                sys.stderr.write(f"  step failed: {step.error}\n")
            elif step.verify and step.verify.errors:
                for path, msg in step.verify.errors:
                    sys.stderr.write(f"  verify failed {path}: {msg}\n")
        return 1

    for diff in report.consolidated_diffs:
        sys.stdout.write(diff)
        if not diff.endswith("\n"):
            sys.stdout.write("\n")
    if not report.consolidated_diffs:
        sys.stderr.write("(action was a no-op against the current workspace)\n")

    if args.write and report.consolidated_diffs:
        from lattice.apply import write_final

        written = write_final(report, root=args.workspace)
        sys.stderr.write(f"wrote {len(written)} file(s)\n")
    return 0


def _intent(args: argparse.Namespace) -> int:
    workspace = FilesystemWorkspace(args.workspace)
    payload = _read_action_json(args.intent, sys.stdin)
    try:
        intent = _intent_adapter.validate_python(payload)
    except ValidationError as exc:
        sys.stderr.write(f"intent did not validate:\n{exc}\n")
        return 2

    if args.files:
        files = [p.strip() for p in args.files.split(",") if p.strip()]
    else:
        files = workspace.iter_files(suffix=".py")

    actions = expand_intent(intent, workspace, files)
    if not actions:
        sys.stderr.write("(intent expanded to zero actions — no matching symbols)\n")
        return 0

    sys.stderr.write(f"expanded to {len(actions)} action(s)\n")
    report: ExecutionReport = execute_plan(actions, workspace)

    if not report.ok:
        for step in report.steps:
            if step.ok:
                continue
            if step.error:
                sys.stderr.write(f"  step failed: {step.error}\n")
            elif step.verify and step.verify.errors:
                for path, msg in step.verify.errors:
                    sys.stderr.write(f"  verify failed {path}: {msg}\n")
        return 1

    # Consolidated diffs (one per touched file) are more readable than
    # per-step diffs when actions chain. Both are available on the report.
    for diff in report.consolidated_diffs:
        sys.stdout.write(diff)
        if not diff.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lattice")
    sub = parser.add_subparsers(dest="cmd", required=True)

    apply_p = sub.add_parser("apply", help="Compile and print the diff for a typed action.")
    apply_p.add_argument("workspace", help="Repository root the action operates against.")
    apply_p.add_argument(
        "--action",
        default="-",
        help="Action JSON: literal string, '-' for stdin, or '@FILE' to read from a file.",
    )
    apply_p.add_argument(
        "--write",
        action="store_true",
        help="Actually write the compiled changes to disk after verify passes.",
    )
    apply_p.set_defaults(func=_apply)

    intent_p = sub.add_parser(
        "intent",
        help="Expand a high-level intent into many typed actions and print the diffs.",
    )
    intent_p.add_argument("workspace", help="Repository root the intent operates against.")
    intent_p.add_argument(
        "--intent",
        default="-",
        help="Intent JSON: literal string, '-' for stdin, or '@FILE' to read from a file.",
    )
    intent_p.add_argument(
        "--files",
        default="",
        help="Comma-separated repository-relative .py paths. Default: walk workspace.",
    )
    intent_p.add_argument(
        "--write",
        action="store_true",
        help="Actually write the compiled changes to disk after verify passes.",
    )
    intent_p.set_defaults(func=_intent)

    propose_p = sub.add_parser(
        "propose",
        help="Ask a local LLM to emit an Action for a task; compile + verify + diff.",
    )
    propose_p.add_argument("workspace", help="Repository root the action operates against.")
    propose_p.add_argument("--task", required=True, help="One-line task description.")
    propose_p.add_argument(
        "--files",
        default="",
        help="Comma-separated repository-relative .py paths to surface as symbol context.",
    )
    propose_p.add_argument(
        "--model",
        default=None,
        help="HuggingFace model name. Default: Qwen/Qwen2.5-0.5B-Instruct (~500MB, CPU).",
    )
    propose_p.add_argument(
        "--atom-db",
        default=None,
        help="Path to a lattice atom store; relevant atoms become hints to the LLM.",
    )
    propose_p.add_argument(
        "--hint-count",
        type=int,
        default=4,
        help="Max hints to surface from the atom store (default 4).",
    )
    propose_p.add_argument(
        "--write",
        action="store_true",
        help="Actually write the compiled changes to disk after verify passes.",
    )
    propose_p.set_defaults(func=_propose)

    do_p = sub.add_parser(
        "do",
        help="One-shot: init if needed, then agent. Defaults the workspace to '.' and the brain to .lattice/brain.db.",
    )
    do_p.add_argument(
        "task", nargs="+", help='Natural-language task, e.g. lattice do "add stripe import to src/foo.py"'
    )
    do_p.add_argument("--workspace", default=".")
    do_p.add_argument("--brain", default=".lattice/brain.db")
    do_p.add_argument("--max-steps", type=int, default=4)
    do_p.add_argument("--model", default=None)
    do_p.add_argument("--hosted", action="store_true")
    do_p.add_argument("--two-stage", action="store_true",
                      help="Planner+Executor split (recommended for sub-1.5B models).")
    do_p.add_argument("--executor-model", default=None)
    do_p.add_argument("--types", action="store_true")
    do_p.add_argument("--tests", action="store_true",
                      help="Run pytest on the affected tests after each edit.")
    do_p.add_argument("--no-brain", action="store_true",
                      help="Disable brain influence (audit mode).")
    do_p.add_argument(
        "--decompose",
        action="store_true",
        help="Deterministically split the task on conjunctions; run one agent loop per subtask.",
    )
    do_p.add_argument(
        "--subtasks",
        default="",
        help="Pipe-separated subtasks; overrides --decompose.",
    )
    do_p.add_argument(
        "--semble",
        action="store_true",
        help="Use Semble for task-relevant code search.",
    )
    do_p.add_argument("--write", action="store_true", help="Default for `do`: ON. Use --no-write to dry-run.")
    do_p.add_argument("--no-write", action="store_true")
    do_p.set_defaults(func=_do)

    init_p = sub.add_parser(
        "init",
        help="Initialize a lattice brain in a new project (seeded + ready to run).",
    )
    init_p.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Project root; defaults to current directory.",
    )
    init_p.add_argument(
        "--brain",
        default=".lattice/brain.db",
        help="Path (relative to workspace) for the brain SQLite file.",
    )
    init_p.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip loading the built-in seed atom pack.",
    )
    init_p.set_defaults(func=_init)

    agent_p = sub.add_parser(
        "agent",
        help="Run a multi-step agent loop with a local LLM; one action per turn.",
    )
    agent_p.add_argument("workspace")
    agent_p.add_argument("--task", required=True)
    agent_p.add_argument(
        "--atom-db", default=None, help="Atom store DB; used for hints + RecallMore."
    )
    agent_p.add_argument(
        "--model",
        default=None,
        help="HuggingFace model name. Default: Qwen/Qwen2.5-0.5B-Instruct.",
    )
    agent_p.add_argument("--max-steps", type=int, default=6)
    agent_p.add_argument(
        "--hosted",
        action="store_true",
        help=(
            "Use the hosted Anthropic proposer (requires ANTHROPIC_API_KEY "
            "and the [hosted] extras). Much more reliable than the local "
            "0.5B for non-trivial tasks."
        ),
    )
    agent_p.add_argument(
        "--two-stage",
        action="store_true",
        help=(
            "Split the local LLM call into Planner (pick a verb) + "
            "Executor (fill slots). Both can be the same small model. "
            "Improves reliability with sub-1.5B models on multi-step tasks."
        ),
    )
    agent_p.add_argument(
        "--executor-model",
        default=None,
        help="Optional second model name for the Executor stage (otherwise the same as --model).",
    )
    agent_p.add_argument(
        "--semble",
        action="store_true",
        help=(
            "Use Semble (fast Model2Vec+BM25 code search) to surface "
            "task-relevant code chunks in the observation. Requires "
            "the [search] extras."
        ),
    )
    agent_p.add_argument(
        "--types",
        action="store_true",
        help=(
            "Run mypy against each candidate edit's after-content; reject "
            "edits that introduce type errors. Requires the [typecheck] "
            "extras."
        ),
    )
    agent_p.add_argument(
        "--tests",
        action="store_true",
        help=(
            "Run pytest on the affected tests in a sandbox after each "
            "candidate edit. Reject edits that break tests. Requires "
            "pytest to be installed (already a [dev] dep)."
        ),
    )
    agent_p.add_argument(
        "--no-brain",
        action="store_true",
        help=(
            "Disable brain influence on the loop (no priming block, no "
            "recall hints in the observation, no apprentice candidates). "
            "Used by the audit to measure 'does the substrate pay for itself'."
        ),
    )
    agent_p.add_argument(
        "--subtasks",
        default="",
        help=(
            "Pipe-separated list of subtasks. When set, the agent runs once "
            "per subtask with shared state. Overrides --decompose."
        ),
    )
    agent_p.add_argument(
        "--decompose",
        action="store_true",
        help=(
            "Deterministically split --task on conjunctions (then / ; / "
            "numbered lists) and run one agent loop per subtask."
        ),
    )
    agent_p.add_argument(
        "--write",
        action="store_true",
        help="Write the agent's final files to disk after the loop ends.",
    )
    agent_p.set_defaults(func=_agent)

    atom_p = sub.add_parser("atom", help="Manage the lattice atom store.")
    atom_sub = atom_p.add_subparsers(dest="atom_cmd", required=True)

    add_p = atom_sub.add_parser("add", help="Add an atom to the store.")
    add_p.add_argument("--db", required=True, help="Atom-store DB path.")
    add_p.add_argument("--content", required=True)
    add_p.add_argument(
        "--type", default="fact", help="Atom type (fact|experience|skill|antipattern|...)"
    )
    add_p.add_argument("--region", default="")
    add_p.add_argument("--tags", default="", help="Comma-separated tags.")
    add_p.add_argument("--importance", type=float, default=0.5)
    add_p.set_defaults(func=_atom_add)

    recall_p = atom_sub.add_parser("recall", help="Recall atoms matching a query.")
    recall_p.add_argument("--db", required=True, help="Atom-store DB path.")
    recall_p.add_argument("--query", required=True)
    recall_p.add_argument("--k", type=int, default=5)
    recall_p.set_defaults(func=_atom_recall)

    seed_p = atom_sub.add_parser(
        "seed",
        help="Load the built-in seed atoms (python conventions/antipatterns/skills) into a store.",
    )
    seed_p.add_argument("--db", required=True, help="Atom-store DB path.")
    seed_p.set_defaults(func=_atom_seed)

    brain_p = sub.add_parser("brain", help="Inspect and curate the lattice atom store.")
    brain_sub = brain_p.add_subparsers(dest="brain_cmd", required=True)

    inspect_p = brain_sub.add_parser(
        "inspect",
        help="Show what hints the brain would surface for a given task.",
    )
    inspect_p.add_argument("--db", required=True)
    inspect_p.add_argument("--task", required=True)
    inspect_p.add_argument("--k", type=int, default=6)
    inspect_p.set_defaults(func=_brain_inspect)

    dump_p = brain_sub.add_parser("dump", help="Group atoms by region and type.")
    dump_p.add_argument("--db", required=True)
    dump_p.set_defaults(func=_brain_dump)

    import_p = brain_sub.add_parser(
        "import",
        help="Load atoms from a JSON or YAML file into the store.",
    )
    import_p.add_argument("--db", required=True)
    import_p.add_argument("file", help="Path to a .json or .yaml/.yml file.")
    import_p.set_defaults(func=_brain_import)

    export_p = brain_sub.add_parser(
        "export", help="Dump every atom in the store to a JSON file."
    )
    export_p.add_argument("--db", required=True)
    export_p.add_argument("file", help="Output .json path.")
    export_p.set_defaults(func=_brain_export)

    state_p = brain_sub.add_parser(
        "audit-state",
        help=(
            "Snapshot the brain's state — type/region breakdown, "
            "importance histogram, and top hottest atoms. The view "
            "you want after running a benchmark or audit to see "
            "what the system actually learned."
        ),
    )
    state_p.add_argument("--db", required=True)
    state_p.add_argument(
        "--top", type=int, default=10,
        help="Number of hottest atoms to print (default 10).",
    )
    state_p.set_defaults(func=_brain_audit_state)

    evolve_p = sub.add_parser(
        "evolve",
        help=(
            "Mine successful traces for recurring task→action-sequence "
            "patterns; print macro candidates."
        ),
    )
    evolve_p.add_argument("--db", required=True, help="Atom-store DB path.")
    evolve_p.add_argument(
        "--min-recurrence",
        type=int,
        default=3,
        help="Minimum number of identical action-sequence runs to promote (default 3).",
    )
    evolve_p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Boost the importance of trace atoms whose action-sequence "
            "recurs at least --min-recurrence times. After this, atom "
            "recall surfaces those patterns as stronger hints to the LLM."
        ),
    )
    evolve_p.set_defaults(func=_evolve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
