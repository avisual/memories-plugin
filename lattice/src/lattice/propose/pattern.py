"""PatternProposer — deterministic shortcuts for common task phrasings.

Recognizes a small set of natural-language patterns and emits the exact
typed Action directly. No LLM needed for these — they're structural
mappings from English to the DSL. The harness should not punt to a
500M-parameter model for things a regex can solve correctly.

The orchestrator tries this proposer first; if nothing matches, it
falls through to whatever LLM proposer is configured.

Patterns covered (case-insensitive, file paths quoted or unquoted):

- 'add an import of <X> to <FILE>'              → AddImport(plain)
- 'add an import of <X> from <Y> to <FILE>'     → AddImport(from-form)
- 'import <X> from <Y> in <FILE>'               → AddImport(from-form)
- 'import <X> as <ALIAS> in/to <FILE>'          → AddImport(alias)
- 'add a <type> parameter <NAME> to <FUNC>'     → AddParameter (heuristic)
- 'add a <type> field <NAME> to <CLASS>'        → AddField (heuristic)
- 'rename <OLD> to <NEW> in <FILE>'             → RenameSymbol

Patterns are intentionally narrow — false positives are worse than
falling through to the LLM. When the regex matches but a slot is
empty, no action is emitted and the LLM gets the task.
"""

from __future__ import annotations

import re

from lattice.actions import (
    Action,
    AddDecorator,
    AddField,
    AddImport,
    AddParameter,
    AddStatement,
    AddTest,
    Expr,
    FileRef,
    RenameSymbol,
    SpanRef,
    SymbolRef,
    TypeExpr,
    WrapInTry,
)
from lattice.propose.base import ObservationContext, Proposer


_PATH = r"['\"`]?(?P<file>[\w/][\w/.-]+\.py)['\"`]?"
_ID = r"['\"`]?(?P<id>[A-Za-z_][\w.]*)['\"`]?"
_LEAF = r"['\"`]?[A-Za-z_]\w*['\"`]?"


_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 'add a keyword-only parameter <NAME> of type <T> [with default <D>] to function <DOTTED> in <FILE>'
    (
        re.compile(
            r"""
            add\s+(?:an?\s+)?(?P<kw>keyword[\s-]only\s+)?
            parameter\s+(?:named\s+|called\s+)?
            ['"`]?(?P<name>[A-Za-z_]\w*)['"`]?
            \s+of\s+type\s+
            ['"`]?(?P<typ>[A-Za-z_][\w\[\].,\s|]*?)['"`]?
            (?:\s+(?:with\s+)?defaul[tT]\s+
                (?P<default>"[^"]*"|'[^']*'|`[^`]*`|[^\s,;]+)
            )?
            \s+to\s+(?:function|method)\s+
            ['"`]?(?P<func>[A-Za-z_][\w.]*)['"`]?
            (?:\s+(?:of\s+class\s+|in\s+class\s+)
                ['"`]?(?P<cls>[A-Za-z_]\w*)['"`]?
            )?
            \s+(?:in|inside)\s+
            """ + _PATH,
            re.IGNORECASE | re.VERBOSE,
        ),
        "add_parameter",
    ),
    # 'add a field <NAME> of type <T> [with default <D>] to class <X> in <FILE>'
    (
        re.compile(
            r"""
            add\s+(?:an?\s+)?
            (?:(?P<typ_pre>[A-Za-z_][\w\[\].,\s|]*?)\s+)?
            (?:field|attribute)\s+(?:named\s+|called\s+)?
            ['"`]?(?P<name>[A-Za-z_]\w*)['"`]?
            (?:\s+of\s+type\s+['"`]?(?P<typ>[A-Za-z_][\w\[\].,\s|]*?)['"`]?)?
            (?:\s+(?:with\s+)?defaul[tT]\s+
                (?P<default>"[^"]*"|'[^']*'|`[^`]*`|[^\s,;]+)
            )?
            \s+to\s+(?:class\s+)?
            ['"`]?(?P<cls>[A-Za-z_]\w*)['"`]?
            \s+(?:in|inside)\s+
            """ + _PATH,
            re.IGNORECASE | re.VERBOSE,
        ),
        "add_field",
    ),
    # 'add an import of <X> from <Y> to <FILE>' / 'import X from Y in FILE'
    (
        re.compile(
            r"""
            (?:add\s+(?:an?\s+)?import\s+of\s+|import\s+)
            ['"`]?(?P<name>[A-Za-z_]\w*)['"`]?
            \s+from\s+
            ['"`]?(?P<module>[A-Za-z_][\w.]*)['"`]?
            \s+(?:in|to|into)\s+
            """ + _PATH,
            re.IGNORECASE | re.VERBOSE,
        ),
        "from_import",
    ),
    # 'add an import of X as ALIAS to FILE'
    (
        re.compile(
            r"""
            (?:add\s+(?:an?\s+)?import\s+of\s+|import\s+)
            ['"`]?(?P<module>[A-Za-z_][\w.]*)['"`]?
            \s+as\s+
            ['"`]?(?P<alias>[A-Za-z_]\w*)['"`]?
            \s+(?:in|to|into)\s+
            """ + _PATH,
            re.IGNORECASE | re.VERBOSE,
        ),
        "alias_import",
    ),
    # 'add an import of X to FILE' / 'add an import of the X module to FILE'
    (
        re.compile(
            r"""
            (?:add\s+(?:an?\s+)?import\s+of\s+(?:the\s+)?|import\s+(?:the\s+)?)
            ['"`]?(?P<module>[A-Za-z_][\w.]*)['"`]?
            (?:\s+module)?
            \s+(?:in|to|into)\s+
            """ + _PATH,
            re.IGNORECASE | re.VERBOSE,
        ),
        "plain_import",
    ),
    # 'rename OLD to NEW in FILE'
    (
        re.compile(
            r"""
            rename\s+
            ['"`]?(?P<old>[A-Za-z_]\w*)['"`]?
            \s+to\s+
            ['"`]?(?P<new>[A-Za-z_]\w*)['"`]?
            \s+(?:in|inside)\s+
            """ + _PATH,
            re.IGNORECASE | re.VERBOSE,
        ),
        "rename",
    ),
    # 'add a @decorator to function FUNC [of class C] in FILE'
    # 'apply @decorator to function FUNC in FILE'
    # 'decorate function FUNC in FILE with @decorator'
    (
        re.compile(
            r"""
            (?:
                (?:add|apply)\s+(?:an?\s+|the\s+)?(?:@\s*)?
                ['"`]?(?P<decorator_a>[A-Za-z_][\w.()\[\]'"=,\s/_-]*?)['"`]?
                \s+(?:decorator\s+)?to\s+(?:function|method|class)\s+
                ['"`]?(?P<func_a>[A-Za-z_][\w.]*)['"`]?
                (?:\s+(?:of\s+class\s+|in\s+class\s+)
                    ['"`]?(?P<cls_a>[A-Za-z_]\w*)['"`]?
                )?
                \s+(?:in|inside)\s+
                (?P<file_a>['"`]?[\w/][\w/.-]+\.py['"`]?)
            )
            |
            (?:
                decorate\s+(?:function|method|class)\s+
                ['"`]?(?P<func_b>[A-Za-z_][\w.]*)['"`]?
                (?:\s+(?:of\s+class\s+|in\s+class\s+)
                    ['"`]?(?P<cls_b>[A-Za-z_]\w*)['"`]?
                )?
                \s+(?:in|inside)\s+
                (?P<file_b>['"`]?[\w/][\w/.-]+\.py['"`]?)
                \s+with\s+(?:an?\s+|the\s+)?(?:@\s*)?
                ['"`]?(?P<decorator_b>[A-Za-z_][\w.()\[\]'"=,\s/_-]*?)['"`]?
            )
            $
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
        "add_decorator",
    ),
    # 'wrap lines N-M of FILE in try/except for EXC' / 'wrap lines N to M in FILE with a try/except for EXC'
    (
        re.compile(
            r"""
            wrap\s+lines?\s+
            (?P<start>\d+)
            \s*(?:-|to|through)\s*
            (?P<end>\d+)
            \s+(?:of|in|inside)\s+
            """ + _PATH + r"""
            \s+(?:in|with|using)\s+(?:an?\s+)?try(?:\s*/\s*except|\s+except)
            (?:\s+for\s+
                ['"`]?(?P<exc>[A-Za-z_][\w.]*)['"`]?
            )?
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
        "wrap_in_try",
    ),
    # 'add a test NAME for FUNC in FILE' / 'add a smoke test for FUNC in FILE'
    (
        re.compile(
            r"""
            add\s+(?:an?\s+)?(?P<kind>smoke\s+|unit\s+|integration\s+)?test\s+
            (?:(?:named\s+|called\s+)
                ['"`]?(?P<tname>test_[A-Za-z_]\w*)['"`]?
                \s+
            )?
            for\s+(?:function\s+|method\s+)?
            ['"`]?(?P<func>[A-Za-z_][\w.]*)['"`]?
            \s+(?:in|inside)\s+
            """ + _PATH,
            re.IGNORECASE | re.VERBOSE,
        ),
        "add_test",
    ),
    # 'Insert <code> at the start/end of function FUNC in FILE'
    # 'Add <code> at the top/bottom of function FUNC in FILE'
    (
        re.compile(
            r"""
            (?:insert|add)\s+
            (?P<stmt>['"`].+?['"`]|`.+?`)
            \s+at\s+(?:the\s+)?(?P<where>start|top|beginning|end|bottom)
            \s+of\s+(?:function|method)\s+
            ['"`]?(?P<func>[A-Za-z_][\w.]*)['"`]?
            (?:\s+(?:of\s+class\s+|in\s+class\s+)
                ['"`]?(?P<cls>[A-Za-z_]\w*)['"`]?
            )?
            \s+(?:in|inside)\s+
            """ + _PATH,
            re.IGNORECASE | re.VERBOSE,
        ),
        "add_statement_in_function",
    ),
]


def _match(task: str) -> tuple[str, dict[str, str]] | None:
    for pattern, kind in _PATTERNS:
        m = pattern.search(task)
        if m:
            return kind, m.groupdict()
    return None


def task_to_action(task: str) -> Action | None:
    """Map *task* to a typed Action via the pattern table; None if no match."""
    matched = _match(task)
    if matched is None:
        return None
    kind, groups = matched
    # The decorator patterns use file_a/file_b instead of file (two
    # alternation branches in one regex). For every other kind, 'file'
    # is the standard group name.
    file_path = groups.get("file") or ""
    if file_path:
        file_ref = FileRef(path=file_path)
    else:
        file_ref = None  # type: ignore[assignment]

    if kind == "plain_import":
        return AddImport(file=file_ref, module=groups["module"], confidence=0.95)
    if kind == "from_import":
        return AddImport(
            file=file_ref,
            module=groups["module"],
            names=[groups["name"]],
            confidence=0.95,
        )
    if kind == "alias_import":
        return AddImport(
            file=file_ref,
            module=groups["module"],
            alias=groups["alias"],
            confidence=0.95,
        )
    if kind == "rename":
        return RenameSymbol(
            symbol=SymbolRef(file=file_path, name=groups["old"]),
            new_name=groups["new"],
            confidence=0.9,
        )
    if kind == "add_parameter":
        func = groups["func"]
        if groups.get("cls"):
            func = f"{groups['cls']}.{func}"
        default = groups.get("default")
        return AddParameter(
            function=SymbolRef(file=file_path, name=func),
            name=groups["name"],
            type=TypeExpr(expr=groups["typ"].strip()),
            default=Expr(code=default) if default else None,
            keyword_only=bool(groups.get("kw")),
            confidence=0.9,
        )
    if kind == "add_field":
        typ_str = (groups.get("typ") or groups.get("typ_pre") or "").strip()
        if not typ_str:
            return None
        default = groups.get("default")
        return AddField(
            cls=SymbolRef(file=file_path, name=groups["cls"]),
            name=groups["name"],
            type=TypeExpr(expr=typ_str),
            default=Expr(code=default) if default else None,
            confidence=0.9,
        )
    if kind == "wrap_in_try":
        try:
            start = int(groups["start"])
            end = int(groups["end"])
        except (KeyError, TypeError, ValueError):
            return None
        if end < start:
            return None
        exc_expr = groups.get("exc") or "Exception"
        return WrapInTry(
            span=SpanRef(file=file_path, start_line=start, end_line=end),
            exception_type=TypeExpr(expr=exc_expr),
            confidence=0.85,
        )
    if kind == "add_test":
        func = groups["func"]
        leaf = func.rsplit(".", 1)[-1]
        test_name = groups.get("tname") or f"test_{leaf}"
        return AddTest(
            target=SymbolRef(file=file_path, name=func),
            test_name=test_name,
            given=Expr(code="..."),
            when=Expr(code=f"result = {leaf}()"),
            then=Expr(code="assert result is not None"),
            confidence=0.7,
        )
    if kind == "add_statement_in_function":
        # Strip the matched quote-bracket pair.
        raw = groups["stmt"].strip()
        if raw and raw[0] in ('"', "'", "`"):
            raw = raw[1:-1]
        if not raw.strip():
            return None
        where = (groups.get("where") or "").lower()
        if where in ("start", "top", "beginning"):
            position = "start_of_function"
        else:
            position = "end_of_function"
        func = groups["func"]
        if groups.get("cls"):
            func = f"{groups['cls']}.{func}"
        return AddStatement(
            file=FileRef(path=file_path),
            code=raw,
            position=position,  # type: ignore[arg-type]
            target=SymbolRef(file=file_path, name=func),
            confidence=0.9,
        )
    if kind == "add_decorator":
        # Two branches collapsed; pull whichever group fired.
        func = groups.get("func_a") or groups.get("func_b")
        cls = groups.get("cls_a") or groups.get("cls_b")
        decorator = (groups.get("decorator_a") or groups.get("decorator_b") or "").strip()
        file_match = groups.get("file_a") or groups.get("file_b") or ""
        # Strip optional surrounding quotes the regex tolerated.
        file_clean = file_match.strip("'\"`")
        if not (func and decorator and file_clean):
            return None
        if cls:
            func = f"{cls}.{func}"
        # Strip a stray leading '@' if the user wrote "add @cached to ..."
        decorator = decorator.lstrip("@").strip()
        return AddDecorator(
            symbol=SymbolRef(file=file_clean, name=func),
            decorator=decorator,
            confidence=0.9,
        )
    return None


class PatternProposer:
    """Deterministic rule-based proposer. Returns [] when no pattern matches."""

    def propose(self, obs: ObservationContext, n: int = 1) -> list[Action]:
        action = task_to_action(obs.task)
        return [action] if action is not None else []


# Static check that PatternProposer satisfies the Proposer Protocol.
_: Proposer = PatternProposer()


class CompositeProposer:
    """Try proposers in order; return the first non-empty result.

    The agent CLI builds this as [PatternProposer, LLMProposer]. When
    the pattern matches deterministically, we use it — that's a
    feature, not a bug. The LLM gets called for everything the patterns
    don't cover (novel tasks, tasks needing Research, multi-slot
    interpretive work). A prior 'LLM-first when task has these words'
    heuristic was removed after it misfired on 'with default 5.0' —
    treating 'with' as a complexity flag and bypassing a matching
    pattern.
    """

    def __init__(self, proposers: list[Proposer]) -> None:
        self._proposers = proposers

    def propose(self, obs: ObservationContext, n: int = 1) -> list[Action]:
        for p in self._proposers:
            out = p.propose(obs, n=n)
            if out:
                return out
        return []
