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
    from lattice.propose.local import LocalLLMProposer

    workspace = FilesystemWorkspace(args.workspace)
    atom_store = SQLiteAtomStore(args.atom_db) if args.atom_db else None
    try:
        sys.stderr.write(f"loading model{(' ' + args.model) if args.model else ''}...\n")
        proposer = (
            LocalLLMProposer(model_name=args.model)
            if args.model
            else LocalLLMProposer()
        )

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

        loop = AgentLoop(
            proposer=proposer,
            workspace=workspace,
            atom_store=atom_store,
            max_steps=args.max_steps,
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
