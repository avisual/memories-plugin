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
    AddField,
    AddImport,
    AddParameter,
    AddTest,
    Branch,
    MarkBlocked,
    RecallMore,
    RenameSymbol,
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
            case WrapInTry() | AddTest() | RenameSymbol():
                raise UnsupportedAction(
                    f"compiler does not yet implement {action.verb!r}"
                )
            case RecallMore() | RevealBody() | MarkBlocked() | Branch():
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
