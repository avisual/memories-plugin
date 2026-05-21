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
    AddField,
    AddImport,
    AddParameter,
    Expr,
    FileRef,
    RenameSymbol,
    SymbolRef,
    TypeExpr,
)
from lattice.propose.base import ObservationContext, Proposer


_PATH = r"['\"`]?(?P<file>[\w/][\w/.-]+\.py)['\"`]?"
_ID = r"['\"`]?(?P<id>[A-Za-z_][\w.]*)['\"`]?"
_LEAF = r"['\"`]?[A-Za-z_]\w*['\"`]?"


_PATTERNS: list[tuple[re.Pattern[str], str]] = [
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
    file_path = groups["file"]
    file_ref = FileRef(path=file_path)

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

    Used by the agent CLI to try fast deterministic rules first, then
    fall through to the LLM for anything the rules don't cover.
    """

    def __init__(self, proposers: list[Proposer]) -> None:
        self._proposers = proposers

    def propose(self, obs: ObservationContext, n: int = 1) -> list[Action]:
        for p in self._proposers:
            out = p.propose(obs, n=n)
            if out:
                return out
        return []
