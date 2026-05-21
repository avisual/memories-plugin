"""Atom — the unit of stored experience.

Same conceptual taxonomy as memories-plugin (fact, experience, skill,
preference, insight, antipattern, task), with extras for the verb-
level composites (macro, recipe, idiom) defined in DESIGN.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AtomType(StrEnum):
    FACT = "fact"
    EXPERIENCE = "experience"
    SKILL = "skill"
    PREFERENCE = "preference"
    INSIGHT = "insight"
    ANTIPATTERN = "antipattern"
    TASK = "task"
    MACRO = "macro"
    RECIPE = "recipe"
    IDIOM = "idiom"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Atom(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int | None = None
    content: str = Field(min_length=1)
    type: AtomType = AtomType.FACT
    region: str = ""
    tags: tuple[str, ...] = ()
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    access_count: int = 0
    created_at: str = Field(default_factory=_now)
    last_accessed_at: str | None = None
