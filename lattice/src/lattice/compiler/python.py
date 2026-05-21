"""Python action compiler — turns Actions into CompiledActions.

Uses libcst so the resulting source preserves formatting and comments.
All transforms are pure; the compiler never touches the filesystem.

Currently implements: AddImport, AddField, AddParameter.
Other mutating verbs raise UnsupportedAction until they land.
"""

from __future__ import annotations

import libcst as cst

from lattice.actions import (
    Action,
    AddDecorator,
    AddField,
    AddFunction,
    AddImport,
    AddParameter,
    AddStatement,
    AddTest,
    Branch,
    ChangeReturnType,
    DeleteSymbol,
    MarkBlocked,
    MarkDone,
    ModifyDocstring,
    MoveSymbol,
    RecallMore,
    RenameSymbol,
    ReplaceBody,
    Research,
    RevealBody,
    WrapInTry,
)
from lattice.compiler.diff import unified_diff
from lattice.compiler.errors import (
    CompileError,
    NonMutatingAction,
    SymbolNotFound,
    UnsupportedAction,
)
from lattice.compiler.types import CompiledAction, FileChange
from lattice.compiler.workspace import Workspace


def compile_action(action: Action, workspace: Workspace) -> CompiledAction:
    """Compile a typed action into file changes against *workspace*.

    Pure: the workspace is read-only here. The returned CompiledAction
    carries `before` / `after` content for each affected file plus a
    unified diff. Applying it to disk is verify's job.
    """
    try:
        match action:
            case AddImport():
                return _compile_add_import(action, workspace)
            case AddField():
                return _compile_add_field(action, workspace)
            case AddParameter():
                return _compile_add_parameter(action, workspace)
            case WrapInTry():
                return _compile_wrap_in_try(action, workspace)
            case AddTest():
                return _compile_add_test(action, workspace)
            case AddStatement():
                return _compile_add_statement(action, workspace)
            case AddFunction():
                return _compile_add_function(action, workspace)
            case AddDecorator():
                return _compile_add_decorator(action, workspace)
            case DeleteSymbol():
                return _compile_delete_symbol(action, workspace)
            case ChangeReturnType():
                return _compile_change_return_type(action, workspace)
            case ModifyDocstring():
                return _compile_modify_docstring(action, workspace)
            case MoveSymbol():
                return _compile_move_symbol(action, workspace)
            case ReplaceBody():
                return _compile_replace_body(action, workspace)
            case RenameSymbol():
                return _compile_rename_symbol(action, workspace)
            case RecallMore() | RevealBody() | MarkBlocked() | MarkDone() | Branch() | Research():
                raise NonMutatingAction(
                    f"{action.verb!r} does not produce file changes; "
                    "the orchestrator handles it directly"
                )
            case _:
                raise UnsupportedAction(f"unknown action: {type(action).__name__}")
    except (cst.CSTValidationError, cst.ParserSyntaxError) as exc:
        raise CompileError(f"libcst rejected {action.verb!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# AddImport
# ---------------------------------------------------------------------------


def _compile_add_import(action: AddImport, workspace: Workspace) -> CompiledAction:
    path = action.file.path
    before = workspace.read(path)
    new_stmt = _build_import_stmt(action)
    rendered_new_stmt = cst.Module(body=[new_stmt]).code.rstrip("\n")

    module = cst.parse_module(before)

    for stmt in module.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                if _imports_equivalent(small, new_stmt.body[0]):
                    return CompiledAction(
                        verb=action.verb,
                        file_changes=(
                            FileChange(path=path, before=before, after=before, diff=""),
                        ),
                    )

    insertion_index = _last_top_level_import_index(module) + 1
    new_body = (*module.body[:insertion_index], new_stmt, *module.body[insertion_index:])
    new_module = module.with_changes(body=new_body)
    after = new_module.code
    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


def _build_import_stmt(action: AddImport) -> cst.SimpleStatementLine:
    asname = cst.AsName(name=cst.Name(action.alias)) if action.alias else None
    if action.names is None:
        node: cst.BaseSmallStatement = cst.Import(
            names=[cst.ImportAlias(name=_dotted(action.module), asname=asname)]
        )
    else:
        names = [
            cst.ImportAlias(
                name=cst.Name(n),
                asname=(asname if len(action.names) == 1 else None),
            )
            for n in action.names
        ]
        node = cst.ImportFrom(module=_dotted(action.module), names=names)
    return cst.SimpleStatementLine(body=[node])


def _dotted(name: str) -> cst.BaseExpression:
    parts = name.split(".")
    expr: cst.BaseExpression = cst.Name(parts[0])
    for part in parts[1:]:
        expr = cst.Attribute(value=expr, attr=cst.Name(part))
    return expr


def _imports_equivalent(a: cst.BaseSmallStatement, b: cst.BaseSmallStatement) -> bool:
    """Compare two import statements by their canonical code form."""
    return _render(a) == _render(b)


def _render(node: cst.CSTNode) -> str:
    return cst.Module(body=[cst.SimpleStatementLine(body=[node])]).code.strip()  # type: ignore[list-item]


def _last_top_level_import_index(module: cst.Module) -> int:
    """Index of the last top-level import.

    If no imports exist, returns the index *just after* a leading
    docstring (if present), so PEP-8 placement is preserved when we add
    the very first import to a previously import-free file.
    """
    last = -1
    for i, stmt in enumerate(module.body):
        if isinstance(stmt, cst.SimpleStatementLine) and stmt.body and isinstance(
            stmt.body[0], (cst.Import, cst.ImportFrom)
        ):
            last = i
    if last >= 0:
        return last
    if module.body and _is_docstring(module.body[0]):
        return 0  # insertion happens at index 1, after the docstring.
    return -1  # insertion happens at index 0, at the top of the file.


# ---------------------------------------------------------------------------
# AddField  /  AddParameter — share symbol-resolution helpers
# ---------------------------------------------------------------------------


def _find_class(module: cst.Module, dotted_name: str) -> cst.ClassDef:
    node = _find_symbol(module, dotted_name)
    if not isinstance(node, cst.ClassDef):
        raise SymbolNotFound(
            f"expected class at {dotted_name!r}, "
            f"found {type(node).__name__ if node else 'nothing'}"
        )
    return node


def _find_function(module: cst.Module, dotted_name: str) -> cst.FunctionDef:
    node = _find_symbol(module, dotted_name)
    if not isinstance(node, cst.FunctionDef):
        raise SymbolNotFound(
            f"expected function at {dotted_name!r}, "
            f"found {type(node).__name__ if node else 'nothing'}"
        )
    return node


def _find_symbol(module: cst.Module, dotted_name: str) -> cst.CSTNode | None:
    parts = dotted_name.split(".")
    container: cst.Module | cst.ClassDef | cst.FunctionDef = module
    for i, part in enumerate(parts):
        found = _find_named_child(container, part)
        if found is None:
            return None
        if i == len(parts) - 1:
            return found
        if not isinstance(found, (cst.ClassDef, cst.FunctionDef)):
            return None
        container = found
    return None


def _find_named_child(
    container: cst.Module | cst.ClassDef | cst.FunctionDef, name: str
) -> cst.CSTNode | None:
    if isinstance(container, cst.Module):
        statements = container.body
    else:
        body = container.body
        if isinstance(body, cst.IndentedBlock):
            statements = body.body
        else:
            return None
    for stmt in statements:
        candidate = _name_of(stmt)
        if candidate == name:
            return stmt
    return None


def _name_of(node: cst.CSTNode) -> str | None:
    if isinstance(node, (cst.ClassDef, cst.FunctionDef)):
        return node.name.value
    return None


# ---------------------------------------------------------------------------
# AddField
# ---------------------------------------------------------------------------


def _compile_add_field(action: AddField, workspace: Workspace) -> CompiledAction:
    path = action.cls.file
    before = workspace.read(path)
    module = cst.parse_module(before)
    cls = _find_class(module, action.cls.name)

    if _class_has_field(cls, action.name):
        return CompiledAction(
            verb=action.verb,
            file_changes=(FileChange(path=path, before=before, after=before, diff=""),),
        )

    field_stmt = _build_field_stmt(action)
    new_cls = cls.with_changes(body=_insert_field(cls.body, field_stmt))
    new_module = module.deep_replace(cls, new_cls)
    after = new_module.code

    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


def _class_has_field(cls: cst.ClassDef, name: str) -> bool:
    body = cls.body
    if not isinstance(body, cst.IndentedBlock):
        return False
    for stmt in body.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                if isinstance(small, cst.AnnAssign) and isinstance(small.target, cst.Name):
                    if small.target.value == name:
                        return True
                if isinstance(small, cst.Assign):
                    for target in small.targets:
                        if isinstance(target.target, cst.Name) and target.target.value == name:
                            return True
    return False


def _build_field_stmt(action: AddField) -> cst.SimpleStatementLine:
    annotation = cst.Annotation(annotation=cst.parse_expression(action.type.expr))
    value = cst.parse_expression(action.default.code) if action.default else None
    ann_assign = cst.AnnAssign(
        target=cst.Name(action.name),
        annotation=annotation,
        value=value,
    )
    return cst.SimpleStatementLine(body=[ann_assign])


def _insert_field(
    body: cst.BaseSuite, field_stmt: cst.SimpleStatementLine
) -> cst.BaseSuite:
    if not isinstance(body, cst.IndentedBlock):
        return cst.IndentedBlock(body=[field_stmt])

    insertion_index = 0
    saw_docstring = False
    for i, stmt in enumerate(body.body):
        if i == 0 and _is_docstring(stmt):
            insertion_index = i + 1
            saw_docstring = True
            continue
        if isinstance(stmt, cst.SimpleStatementLine) and stmt.body and isinstance(
            stmt.body[0], (cst.AnnAssign, cst.Assign)
        ):
            insertion_index = i + 1
        else:
            if not saw_docstring:
                break
            saw_docstring = False
            if insertion_index == 0:
                insertion_index = i
            break

    new_body = (*body.body[:insertion_index], field_stmt, *body.body[insertion_index:])
    return body.with_changes(body=new_body)


def _is_docstring(stmt: cst.BaseStatement) -> bool:
    if not isinstance(stmt, cst.SimpleStatementLine):
        return False
    if not stmt.body:
        return False
    first = stmt.body[0]
    return (
        isinstance(first, cst.Expr)
        and isinstance(first.value, (cst.SimpleString, cst.ConcatenatedString))
    )


# ---------------------------------------------------------------------------
# AddParameter
# ---------------------------------------------------------------------------


def _compile_add_parameter(action: AddParameter, workspace: Workspace) -> CompiledAction:
    path = action.function.file
    before = workspace.read(path)
    module = cst.parse_module(before)
    fn = _find_function(module, action.function.name)

    if _function_has_parameter(fn, action.name):
        return CompiledAction(
            verb=action.verb,
            file_changes=(FileChange(path=path, before=before, after=before, diff=""),),
        )

    new_params = _insert_parameter(fn.params, action)
    new_fn = fn.with_changes(params=new_params)
    new_module = module.deep_replace(fn, new_fn)
    after = new_module.code

    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# RenameSymbol
# ---------------------------------------------------------------------------


def _compile_rename_symbol(
    action: RenameSymbol, workspace: Workspace
) -> CompiledAction:
    """Rename a symbol across the workspace.

    Walks every .py file in the workspace; in each:
    - In the file that DEFINES the symbol, rewrites the def/class
      name AND every bare-name reference.
    - In every OTHER file, rewrites bare-name references to the
      symbol's leaf name (heuristic: same leaf, same intent).

    Limitation: bare-name aliasing across import-as is not yet
    detected. e.g. `from old_module import old_name as ox; ox()` —
    we don't touch the `ox` reference. A future iteration uses the
    lattice symbol graph (Organ 1) to resolve aliases precisely.

    The dotted name in action.symbol selects the definition; only the
    leaf component is renamed (renaming 'Foo.bar' to 'baz' renames
    just the method, not the class).
    """
    def_path = action.symbol.file
    def_before = workspace.read(def_path)
    def_module = cst.parse_module(def_before)

    node = _find_symbol(def_module, action.symbol.name)
    if node is None:
        raise SymbolNotFound(
            f"could not find symbol {action.symbol.name!r} in {def_path}"
        )
    if not isinstance(node, (cst.FunctionDef, cst.ClassDef)):
        raise SymbolNotFound(
            f"symbol {action.symbol.name!r} is not a function or class; "
            f"got {type(node).__name__}"
        )

    old_leaf = node.name.value
    new_leaf = action.new_name

    if old_leaf == new_leaf:
        return CompiledAction(
            verb=action.verb,
            file_changes=(
                FileChange(path=def_path, before=def_before, after=def_before, diff=""),
            ),
        )

    transformer = _RenameTransformer(old_leaf=old_leaf, new_leaf=new_leaf)

    file_changes: list[FileChange] = []

    new_def_module = def_module.visit(transformer)
    def_after = new_def_module.code
    file_changes.append(
        FileChange(
            path=def_path,
            before=def_before,
            after=def_after,
            diff=unified_diff(path=def_path, before=def_before, after=def_after),
        )
    )

    other_files = [p for p in workspace.iter_files(suffix=".py") if p != def_path]
    for path in other_files:
        try:
            before = workspace.read(path)
        except Exception:  # noqa: BLE001
            continue
        if old_leaf not in before:
            continue
        try:
            module = cst.parse_module(before)
        except cst.ParserSyntaxError:
            continue
        new_module = module.visit(transformer)
        after = new_module.code
        if after == before:
            continue
        file_changes.append(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            )
        )

    return CompiledAction(verb=action.verb, file_changes=tuple(file_changes))


class _RenameTransformer(cst.CSTTransformer):
    """Rewrites Name and FunctionDef/ClassDef name tokens.

    Only rewrites the leaf identifier; doesn't try to rewrite imports
    that bind the symbol to a local alias (that's a follow-up requiring
    scope analysis).
    """

    def __init__(self, *, old_leaf: str, new_leaf: str) -> None:
        self._old = old_leaf
        self._new = new_leaf

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        if updated_node.name.value == self._old:
            return updated_node.with_changes(name=cst.Name(self._new))
        return updated_node

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        if updated_node.name.value == self._old:
            return updated_node.with_changes(name=cst.Name(self._new))
        return updated_node

    def leave_Name(
        self,
        original_node: cst.Name,
        updated_node: cst.Name,
    ) -> cst.Name:
        if updated_node.value == self._old:
            return cst.Name(self._new)
        return updated_node


# ---------------------------------------------------------------------------
# WrapInTry
# ---------------------------------------------------------------------------


def _compile_wrap_in_try(action: WrapInTry, workspace: Workspace) -> CompiledAction:
    if action.handler_body or action.finally_body:
        raise UnsupportedAction(
            "WrapInTry handler_body/finally_body composition not yet supported; "
            "emit an empty handler_body and compose follow-up actions instead"
        )

    path = action.span.file
    before = workspace.read(path)
    module = cst.parse_module(before)
    wrapper = cst.MetadataWrapper(module)
    positions = wrapper.resolve(cst.metadata.PositionProvider)

    start = action.span.start_line
    end = action.span.end_line

    enclosing, parent_body, indices = _find_span_in_module(
        wrapper.module, positions, start, end
    )
    if enclosing is None or not indices:
        raise CompileError(
            f"no top-level or block statements fall within lines {start}-{end} of {path}"
        )

    statements_to_wrap = [parent_body[i] for i in indices]
    try_stmt = _build_try_stmt(statements_to_wrap, action.exception_type.expr)

    new_body = list(parent_body)
    first = indices[0]
    last = indices[-1]
    new_body[first : last + 1] = [try_stmt]

    if enclosing is wrapper.module:
        new_module = wrapper.module.with_changes(body=tuple(new_body))
    else:
        block = enclosing.body  # type: ignore[union-attr]
        new_block = block.with_changes(body=tuple(new_body))
        new_module = wrapper.module.deep_replace(
            enclosing, enclosing.with_changes(body=new_block)
        )

    after = new_module.code
    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


def _find_span_in_module(
    module: cst.Module,
    positions: dict,
    start_line: int,
    end_line: int,
) -> tuple[cst.CSTNode | None, list[cst.BaseStatement], list[int]]:
    """Locate the statements that fall within [start_line, end_line].

    Returns (enclosing_node, body_list, indices). enclosing_node is the
    Module or the FunctionDef/ClassDef whose body holds the statements.
    """

    def scan(node: cst.CSTNode, body: list[cst.BaseStatement]) -> tuple[
        cst.CSTNode | None, list[cst.BaseStatement], list[int]
    ]:
        indices: list[int] = []
        for i, stmt in enumerate(body):
            pos = positions[stmt]
            if pos.end.line < start_line:
                continue
            if pos.start.line > end_line:
                break
            indices.append(i)
            if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)) and isinstance(
                stmt.body, cst.IndentedBlock
            ):
                inner_indices = [
                    j
                    for j, child in enumerate(stmt.body.body)
                    if positions[child].start.line >= start_line
                    and positions[child].end.line <= end_line
                ]
                if inner_indices:
                    return stmt, list(stmt.body.body), inner_indices
        return node, body, indices

    return scan(module, list(module.body))


def _build_try_stmt(
    statements: list[cst.BaseStatement], exception_expr: str
) -> cst.Try:
    """Build a try/except wrapping *statements* with `except <exc>: pass`."""
    try_body = cst.IndentedBlock(body=tuple(statements))
    handler = cst.ExceptHandler(
        type=cst.parse_expression(exception_expr),
        body=cst.IndentedBlock(body=(cst.SimpleStatementLine(body=[cst.Pass()]),)),
    )
    return cst.Try(body=try_body, handlers=[handler])


# ---------------------------------------------------------------------------
# AddTest
# ---------------------------------------------------------------------------


def _test_path_for(target_file: str) -> str:
    """Default test-file convention: tests/test_<basename>.py at repo root."""
    base = target_file.rsplit("/", 1)[-1]
    stem = base[:-3] if base.endswith(".py") else base
    return f"tests/test_{stem}.py"


def _compile_add_test(action: AddTest, workspace: Workspace) -> CompiledAction:
    test_path = _test_path_for(action.target.file)
    before = workspace.read(test_path) if workspace.exists(test_path) else ""

    if before:
        module = cst.parse_module(before)
        if _function_exists_in_module(module, action.test_name):
            return CompiledAction(
                verb=action.verb,
                file_changes=(
                    FileChange(path=test_path, before=before, after=before, diff=""),
                ),
            )

    test_fn = _build_test_function(action)
    if before:
        module = cst.parse_module(before)
        new_body = (*module.body, cst.EmptyLine(), test_fn)
        new_module = module.with_changes(body=new_body)
    else:
        new_module = cst.Module(body=(test_fn,))
    after = new_module.code

    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=test_path,
                before=before,
                after=after,
                diff=unified_diff(path=test_path, before=before, after=after),
            ),
        ),
    )


def _function_exists_in_module(module: cst.Module, name: str) -> bool:
    return any(
        isinstance(stmt, cst.FunctionDef) and stmt.name.value == name
        for stmt in module.body
    )


def _build_test_function(action: AddTest) -> cst.FunctionDef:
    body_lines = [
        cst.SimpleStatementLine(body=[_statement_from_code(action.given.code)]),
        cst.SimpleStatementLine(body=[_statement_from_code(action.when.code)]),
        cst.SimpleStatementLine(body=[_statement_from_code(action.then.code)]),
    ]
    return cst.FunctionDef(
        name=cst.Name(action.test_name),
        params=cst.Parameters(),
        body=cst.IndentedBlock(body=tuple(body_lines)),
        returns=cst.Annotation(annotation=cst.Name("None")),
        leading_lines=(cst.EmptyLine(), cst.EmptyLine()),
    )


def _statement_from_code(code: str) -> cst.BaseSmallStatement:
    """Parse *code* as a single small statement.

    Accepts assert/assign/expression statements. Raises CompileError if
    libcst can't parse the snippet as one statement.
    """
    try:
        parsed = cst.parse_statement(code)
    except cst.ParserSyntaxError as exc:
        raise CompileError(f"could not parse statement {code!r}: {exc}") from exc
    if isinstance(parsed, cst.SimpleStatementLine) and len(parsed.body) == 1:
        return parsed.body[0]
    raise CompileError(
        f"expected a single small statement, got {type(parsed).__name__} for {code!r}"
    )


# ---------------------------------------------------------------------------
# AddStatement
# ---------------------------------------------------------------------------


def _compile_add_statement(
    action: AddStatement, workspace: Workspace
) -> CompiledAction:
    """Insert a statement at the requested position.

    Module-level positions ('end', 'top_after_imports') insert into
    the module's top-level body. Function-scoped positions
    ('start_of_function', 'end_of_function') insert into a specific
    function/method's body — schema-required `target` carries the
    SymbolRef.

    `code` is parsed with libcst.parse_module — invalid Python is
    rejected here. Idempotent on exact-match.
    """
    path = action.file.path
    before = workspace.read(path)

    try:
        new_block = cst.parse_module(action.code)
    except cst.ParserSyntaxError as exc:
        raise CompileError(f"AddStatement.code is not valid Python: {exc}") from exc
    if not new_block.body:
        raise CompileError("AddStatement.code parsed to zero statements")

    module = cst.parse_module(before)

    if action.position in ("start_of_function", "end_of_function"):
        assert action.target is not None  # enforced by model_post_init
        fn = _find_function(module, action.target.name)

        body = fn.body
        if not isinstance(body, cst.IndentedBlock):
            raise CompileError(
                f"{action.target.name!r} has no IndentedBlock body; "
                "cannot insert statements"
            )

        # Idempotency: skip if every candidate already appears in the body.
        candidate_renders = [_render_statements([s]) for s in new_block.body]
        existing_renders = {_render_statements([s]) for s in body.body}
        if candidate_renders and all(r in existing_renders for r in candidate_renders):
            return CompiledAction(
                verb=action.verb,
                file_changes=(
                    FileChange(path=path, before=before, after=before, diff=""),
                ),
            )

        if action.position == "start_of_function":
            # After docstring (if any), otherwise at index 0.
            insert_at = 1 if body.body and _is_docstring(body.body[0]) else 0
        else:  # end_of_function
            insert_at = len(body.body)

        new_inner = (
            *body.body[:insert_at],
            *new_block.body,
            *body.body[insert_at:],
        )
        new_body = body.with_changes(body=new_inner)
        new_fn = fn.with_changes(body=new_body)
        new_module = module.deep_replace(fn, new_fn)
    else:
        insertion_index = _statement_insertion_index(module, action.position)

        candidate_renders = [_render_statements([s]) for s in new_block.body]
        existing_renders = {_render_statements([s]) for s in module.body}
        if candidate_renders and all(r in existing_renders for r in candidate_renders):
            return CompiledAction(
                verb=action.verb,
                file_changes=(
                    FileChange(path=path, before=before, after=before, diff=""),
                ),
            )

        new_body = (
            *module.body[:insertion_index],
            *new_block.body,
            *module.body[insertion_index:],
        )
        new_module = module.with_changes(body=new_body)

    after = new_module.code
    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


def _render_statements(statements: tuple | list) -> str:
    if not statements:
        return ""
    return cst.Module(body=tuple(statements)).code.strip()


def _statement_insertion_index(module: cst.Module, position: str) -> int:
    """Resolve a position keyword to a body index.

    'end' — after every existing top-level statement (length of body).
    'top_after_imports' — after the docstring (if any) and the import
        block (if any). If neither exists, returns 0 (very top of file).
    """
    if position == "end":
        return len(module.body)

    index = 0
    if module.body and _is_docstring(module.body[0]):
        index = 1
    while index < len(module.body):
        stmt = module.body[index]
        if isinstance(stmt, cst.SimpleStatementLine) and stmt.body and isinstance(
            stmt.body[0], (cst.Import, cst.ImportFrom)
        ):
            index += 1
            continue
        break
    return index


# ---------------------------------------------------------------------------
# ReplaceBody
# ---------------------------------------------------------------------------


def _compile_replace_body(
    action: ReplaceBody, workspace: Workspace
) -> CompiledAction:
    """Swap a function's body for new code; signature + decorators kept."""
    path = action.symbol.file
    before = workspace.read(path)
    module = cst.parse_module(before)
    fn = _find_function(module, action.symbol.name)

    # Parse the new body as a module, then re-wrap as the function's
    # IndentedBlock. Reject malformed input here.
    try:
        new_module_body = cst.parse_module(action.body)
    except cst.ParserSyntaxError as exc:
        raise CompileError(
            f"ReplaceBody.body is not valid Python: {exc}"
        ) from exc
    if not new_module_body.body:
        raise CompileError("ReplaceBody.body parsed to zero statements")

    new_indented = cst.IndentedBlock(body=tuple(new_module_body.body))

    # Idempotency: compare rendered body source.
    if isinstance(fn.body, cst.IndentedBlock):
        old_src = cst.Module(body=fn.body.body).code.strip()
        new_src = cst.Module(body=new_module_body.body).code.strip()
        if old_src == new_src:
            return CompiledAction(
                verb=action.verb,
                file_changes=(
                    FileChange(path=path, before=before, after=before, diff=""),
                ),
            )

    new_fn = fn.with_changes(body=new_indented)
    new_module = module.deep_replace(fn, new_fn)
    after = new_module.code

    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# ChangeReturnType
# ---------------------------------------------------------------------------


def _compile_change_return_type(
    action: ChangeReturnType, workspace: Workspace
) -> CompiledAction:
    """Replace a function's return annotation."""
    path = action.symbol.file
    before = workspace.read(path)
    module = cst.parse_module(before)
    fn = _find_function(module, action.symbol.name)

    try:
        new_annotation_expr = cst.parse_expression(action.return_type.expr)
    except cst.ParserSyntaxError as exc:
        raise CompileError(
            f"ChangeReturnType.return_type is not a valid expression: {exc}"
        ) from exc

    # Idempotency: compare the rendered source of the existing return
    # annotation (if any) to the requested one.
    requested_src = cst.Module(body=[]).code_for_node(new_annotation_expr).strip()
    if fn.returns is not None:
        current_src = cst.Module(body=[]).code_for_node(fn.returns.annotation).strip()
        if current_src == requested_src:
            return CompiledAction(
                verb=action.verb,
                file_changes=(FileChange(path=path, before=before, after=before, diff=""),),
            )

    new_fn = fn.with_changes(returns=cst.Annotation(annotation=new_annotation_expr))
    new_module = module.deep_replace(fn, new_fn)
    after = new_module.code

    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# ModifyDocstring
# ---------------------------------------------------------------------------


def _compile_modify_docstring(
    action: ModifyDocstring, workspace: Workspace
) -> CompiledAction:
    """Set the docstring on a function, class, or module.

    For functions/classes (action.symbol set), inserts the docstring as
    the first statement of the body (or replaces an existing first
    docstring). For modules (symbol=None), inserts at module top (or
    replaces an existing module docstring).
    """
    path = action.file.path
    before = workspace.read(path)
    module = cst.parse_module(before)

    new_docstring_literal = cst.SimpleString(_quote_docstring(action.docstring))
    new_doc_stmt = cst.SimpleStatementLine(body=[cst.Expr(value=new_docstring_literal)])

    if action.symbol is None:
        # Module docstring.
        body = list(module.body)
        if body and _is_docstring(body[0]):
            existing = body[0]
            if isinstance(existing, cst.SimpleStatementLine) and isinstance(
                existing.body[0], cst.Expr
            ):
                existing_value = existing.body[0].value
                if (
                    isinstance(existing_value, cst.SimpleString)
                    and existing_value.value == new_docstring_literal.value
                ):
                    return CompiledAction(
                        verb=action.verb,
                        file_changes=(
                            FileChange(path=path, before=before, after=before, diff=""),
                        ),
                    )
            body[0] = new_doc_stmt
        else:
            body.insert(0, new_doc_stmt)
        new_module = module.with_changes(body=tuple(body))
    else:
        target = _find_symbol(module, action.symbol.name)
        if target is None or not isinstance(target, (cst.FunctionDef, cst.ClassDef)):
            raise SymbolNotFound(
                f"ModifyDocstring target must be function or class; got "
                f"{type(target).__name__ if target else 'nothing'}"
            )
        body = target.body
        if not isinstance(body, cst.IndentedBlock):
            raise CompileError(f"{action.symbol.name!r} has no IndentedBlock body")
        body_stmts = list(body.body)
        if body_stmts and _is_docstring(body_stmts[0]):
            existing = body_stmts[0]
            if isinstance(existing, cst.SimpleStatementLine) and isinstance(
                existing.body[0], cst.Expr
            ):
                existing_value = existing.body[0].value
                if (
                    isinstance(existing_value, cst.SimpleString)
                    and existing_value.value == new_docstring_literal.value
                ):
                    return CompiledAction(
                        verb=action.verb,
                        file_changes=(
                            FileChange(path=path, before=before, after=before, diff=""),
                        ),
                    )
            body_stmts[0] = new_doc_stmt
        else:
            body_stmts.insert(0, new_doc_stmt)
        new_body = body.with_changes(body=tuple(body_stmts))
        new_target = target.with_changes(body=new_body)
        new_module = module.deep_replace(target, new_target)

    after = new_module.code
    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


def _quote_docstring(text: str) -> str:
    """Wrap *text* in triple-double-quotes, escaping inner triple-quotes."""
    safe = text.replace('"""', '\\"\\"\\"')
    return f'"""{safe}"""'


# ---------------------------------------------------------------------------
# MoveSymbol
# ---------------------------------------------------------------------------


def _compile_move_symbol(action: MoveSymbol, workspace: Workspace) -> CompiledAction:
    """Move a top-level function or class from source to target file.

    v0 scope: top-level def/class only. Methods (dotted names) are not
    yet supported. Imports referencing the symbol are NOT updated (a
    follow-up RenameSymbol can redirect them).
    """
    if "." in action.symbol.name:
        raise UnsupportedAction(
            "MoveSymbol v0 supports top-level def/class only; "
            "method moves are a follow-up."
        )

    source_path = action.symbol.file
    target_path = action.target_file.path

    if source_path == target_path:
        raise CompileError(
            "MoveSymbol: source and target files must differ"
        )

    source_before = workspace.read(source_path)
    source_module = cst.parse_module(source_before)

    # Find the target node in the source module.
    target_node = None
    for stmt in source_module.body:
        if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)) and stmt.name.value == action.symbol.name:
            target_node = stmt
            break
    if target_node is None:
        raise SymbolNotFound(
            f"MoveSymbol: could not find top-level {action.symbol.name!r} "
            f"in {source_path}"
        )

    # Remove from source.
    new_source_body = tuple(
        s for s in source_module.body
        if not (isinstance(s, (cst.FunctionDef, cst.ClassDef)) and s.name.value == action.symbol.name)
    )
    new_source_module = source_module.with_changes(body=new_source_body)
    source_after = new_source_module.code

    # Add to target.
    if workspace.exists(target_path):
        target_before = workspace.read(target_path)
    else:
        target_before = ""
    target_module = cst.parse_module(target_before)

    if _function_exists_in_module(target_module, action.symbol.name) or _class_exists_in_module(
        target_module, action.symbol.name
    ):
        # Already there. If we also removed it from source, that's the
        # whole move — still a real diff. Otherwise no-op.
        if source_before == source_after:
            return CompiledAction(
                verb=action.verb,
                file_changes=(
                    FileChange(path=source_path, before=source_before, after=source_before, diff=""),
                    FileChange(path=target_path, before=target_before, after=target_before, diff=""),
                ),
            )
        new_target_module = target_module
    else:
        insertion_index = _statement_insertion_index(target_module, action.position)
        leading_lines = (cst.EmptyLine(), cst.EmptyLine()) if insertion_index > 0 else ()
        moved_node = target_node.with_changes(leading_lines=leading_lines)
        new_target_body = (
            *target_module.body[:insertion_index],
            moved_node,
            *target_module.body[insertion_index:],
        )
        new_target_module = target_module.with_changes(body=new_target_body)

    target_after = new_target_module.code

    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=source_path,
                before=source_before,
                after=source_after,
                diff=unified_diff(path=source_path, before=source_before, after=source_after),
            ),
            FileChange(
                path=target_path,
                before=target_before,
                after=target_after,
                diff=unified_diff(path=target_path, before=target_before, after=target_after),
            ),
        ),
    )


def _class_exists_in_module(module: cst.Module, name: str) -> bool:
    return any(
        isinstance(stmt, cst.ClassDef) and stmt.name.value == name
        for stmt in module.body
    )


# ---------------------------------------------------------------------------
# DeleteSymbol
# ---------------------------------------------------------------------------


def _compile_delete_symbol(
    action: DeleteSymbol, workspace: Workspace
) -> CompiledAction:
    """Remove a function/class/method definition by dotted name."""
    path = action.symbol.file
    before = workspace.read(path)
    module = cst.parse_module(before)

    dotted = action.symbol.name

    if "." in dotted:
        cls_name, _, leaf = dotted.rpartition(".")
        try:
            cls = _find_class(module, cls_name)
        except SymbolNotFound:
            if action.require_present:
                raise
            return CompiledAction(
                verb=action.verb,
                file_changes=(FileChange(path=path, before=before, after=before, diff=""),),
            )
        body = cls.body
        if not isinstance(body, cst.IndentedBlock):
            raise CompileError(f"{cls_name!r} body is not an IndentedBlock")
        kept = []
        removed = False
        for stmt in body.body:
            if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)) and stmt.name.value == leaf:
                removed = True
                continue
            kept.append(stmt)
        if not removed:
            if action.require_present:
                raise SymbolNotFound(
                    f"DeleteSymbol could not find {dotted!r} in {path}"
                )
            return CompiledAction(
                verb=action.verb,
                file_changes=(FileChange(path=path, before=before, after=before, diff=""),),
            )
        if not kept:
            # Class would become empty — leave a `pass` placeholder so it parses.
            kept = [cst.SimpleStatementLine(body=[cst.Pass()])]
        new_cls = cls.with_changes(body=body.with_changes(body=tuple(kept)))
        new_module = module.deep_replace(cls, new_cls)
    else:
        new_body = []
        removed = False
        for stmt in module.body:
            if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)) and stmt.name.value == dotted:
                removed = True
                continue
            new_body.append(stmt)
        if not removed:
            if action.require_present:
                raise SymbolNotFound(
                    f"DeleteSymbol could not find {dotted!r} in {path}"
                )
            return CompiledAction(
                verb=action.verb,
                file_changes=(FileChange(path=path, before=before, after=before, diff=""),),
            )
        new_module = module.with_changes(body=tuple(new_body))

    after = new_module.code
    if after == before:
        return CompiledAction(
            verb=action.verb,
            file_changes=(FileChange(path=path, before=before, after=before, diff=""),),
        )
    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# AddDecorator
# ---------------------------------------------------------------------------


def _compile_add_decorator(
    action: AddDecorator, workspace: Workspace
) -> CompiledAction:
    """Apply a decorator to an existing function or class.

    `decorator` is parsed with libcst.parse_expression — anything that
    isn't a valid Python expression is rejected at compile time.
    Idempotent: returns a no-op when an exactly-equal decorator is
    already present on the target.

    position='outermost' (default) places the new decorator ABOVE any
    existing ones (the typical Pythonic shape). 'innermost' places it
    directly above the def/class.
    """
    path = action.symbol.file
    before = workspace.read(path)
    module = cst.parse_module(before)

    target = _find_symbol(module, action.symbol.name)
    if target is None or not isinstance(target, (cst.FunctionDef, cst.ClassDef)):
        raise SymbolNotFound(
            f"AddDecorator target must be a function or class; "
            f"got {type(target).__name__ if target else 'nothing'} for {action.symbol.name!r}"
        )

    try:
        decorator_expr = cst.parse_expression(action.decorator)
    except cst.ParserSyntaxError as exc:
        raise CompileError(
            f"AddDecorator.decorator is not a valid expression: {exc}"
        ) from exc

    new_decorator = cst.Decorator(decorator=decorator_expr)
    new_decorator_src = _decorator_source(new_decorator)

    existing = list(target.decorators)
    for d in existing:
        if _decorator_source(d) == new_decorator_src:
            return CompiledAction(
                verb=action.verb,
                file_changes=(FileChange(path=path, before=before, after=before, diff=""),),
            )

    if action.position == "innermost":
        new_decorators = (*existing, new_decorator)
    else:
        new_decorators = (new_decorator, *existing)

    new_target = target.with_changes(decorators=new_decorators)
    new_module = module.deep_replace(target, new_target)
    after = new_module.code

    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


def _decorator_source(d: cst.Decorator) -> str:
    """Canonical-source string for a decorator (for idempotency check)."""
    # Wrap in a dummy class body so libcst will render the decorator
    # in context; strip the wrapper from the resulting text.
    rendered = cst.Module(
        body=[
            cst.ClassDef(
                name=cst.Name("_X"),
                body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
                decorators=[d],
            )
        ]
    ).code
    line = rendered.splitlines()[0].strip()
    return line


# ---------------------------------------------------------------------------
# AddFunction
# ---------------------------------------------------------------------------


def _compile_add_function(
    action: AddFunction, workspace: Workspace
) -> CompiledAction:
    """Insert a complete function (with decorators) at module level."""
    path = action.file.path
    before = workspace.read(path)

    try:
        parsed = cst.parse_statement(action.source)
    except cst.ParserSyntaxError as exc:
        raise CompileError(f"AddFunction.source is not a valid statement: {exc}") from exc

    if not isinstance(parsed, cst.FunctionDef):
        raise CompileError(
            f"AddFunction.source must parse to a FunctionDef, got {type(parsed).__name__}"
        )

    module = cst.parse_module(before)
    if _function_exists_in_module(module, parsed.name.value):
        return CompiledAction(
            verb=action.verb,
            file_changes=(FileChange(path=path, before=before, after=before, diff=""),),
        )

    insertion_index = _statement_insertion_index(module, action.position)

    # Insert with a blank-line gap before the function if it's not at top.
    leading_lines = (cst.EmptyLine(), cst.EmptyLine()) if insertion_index > 0 else ()
    new_fn = parsed.with_changes(leading_lines=leading_lines)

    new_body = (
        *module.body[:insertion_index],
        new_fn,
        *module.body[insertion_index:],
    )
    new_module = module.with_changes(body=new_body)
    after = new_module.code

    return CompiledAction(
        verb=action.verb,
        file_changes=(
            FileChange(
                path=path,
                before=before,
                after=after,
                diff=unified_diff(path=path, before=before, after=after),
            ),
        ),
    )


def _function_has_parameter(fn: cst.FunctionDef, name: str) -> bool:
    p = fn.params
    for group in (p.params, p.kwonly_params, p.posonly_params):
        for param in group:
            if param.name.value == name:
                return True
    if p.star_arg and isinstance(p.star_arg, cst.Param) and p.star_arg.name.value == name:
        return True
    if p.star_kwarg and p.star_kwarg.name.value == name:
        return True
    return False


def _insert_parameter(params: cst.Parameters, action: AddParameter) -> cst.Parameters:
    annotation = cst.Annotation(annotation=cst.parse_expression(action.type.expr))
    default = cst.parse_expression(action.default.code) if action.default else None
    new_param = cst.Param(
        name=cst.Name(action.name),
        annotation=annotation,
        default=default,
    )

    if action.keyword_only:
        existing = list(params.kwonly_params)
        existing.append(new_param)
        star_arg: cst.Param | cst.ParamStar | cst.MaybeSentinel = params.star_arg
        if isinstance(star_arg, cst.MaybeSentinel):
            star_arg = cst.ParamStar()
        return params.with_changes(kwonly_params=tuple(existing), star_arg=star_arg)

    existing = list(params.params)
    if action.position is not None and 0 <= action.position <= len(existing):
        existing.insert(action.position, new_param)
    else:
        existing.append(new_param)
    return params.with_changes(params=tuple(existing))
