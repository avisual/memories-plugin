"""Code search via Semble — fast, focused code retrieval for SENSE.

Semble (https://minish.ai/packages/semble/) is a CPU-only code search
library built for agents. It produces ranked code chunks for a natural-
language or code query in milliseconds — much more focused than
'dump every symbol in the workspace' as the LLM's observation.

The lattice integration is intentionally thin: index the workspace
once per AgentLoop, query per cycle with the (possibly evolving) task
description, surface the top-K chunks as hints in the next prompt.

Lattice still works without semble installed; this is an opt-in
upgrade gated on the `[search]` extras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CodeChunk:
    file: str
    start_line: int
    end_line: int
    content: str
    score: float = 0.0


class SembleCodeSearch:
    """Wrap a SembleIndex with lazy load + per-instance caching."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace_root must be an existing directory: {self.root}")
        self._index: Any = None

    def _load(self) -> None:
        if self._index is not None:
            return
        try:
            from semble import SembleIndex
        except ImportError as exc:
            raise RuntimeError(
                "semble not installed. Run: uv pip install -e '.[search]'"
            ) from exc
        self._index = SembleIndex.from_path(str(self.root))

    def search(self, query: str, *, top_k: int = 5) -> list[CodeChunk]:
        if not query.strip():
            return []
        self._load()
        results = self._index.search(query, top_k=top_k)
        out: list[CodeChunk] = []
        for r in results:
            chunk = r.chunk
            out.append(
                CodeChunk(
                    file=str(chunk.file_path),
                    start_line=int(chunk.start_line),
                    end_line=int(chunk.end_line),
                    content=str(chunk.content),
                    score=float(getattr(r, "score", 0.0)),
                )
            )
        return out


def maybe_code_search(workspace_root: str | Path) -> SembleCodeSearch | None:
    """Construct a SembleCodeSearch if semble is installed, else None.

    Convenience for the agent loop, which should degrade silently when
    the user hasn't installed the [search] extra.
    """
    try:
        import semble  # noqa: F401
    except ImportError:
        return None
    try:
        return SembleCodeSearch(workspace_root)
    except Exception:
        return None
