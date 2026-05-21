"""Shared types for the compiler output."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FileChange(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str
    before: str
    after: str
    diff: str

    @property
    def is_noop(self) -> bool:
        return self.before == self.after


class CompiledAction(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    verb: str = Field(description="The verb name of the compiled action.")
    file_changes: tuple[FileChange, ...]

    @property
    def is_noop(self) -> bool:
        return all(fc.is_noop for fc in self.file_changes)
