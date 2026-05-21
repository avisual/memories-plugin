"""Embedder — turns text into dense vectors.

Default backend: a small sentence-transformers model (MiniLM-L6, 90MB,
384-dim, CPU-friendly). Lazy-loaded so the module is importable without
the [llm] extras installed.

Custom embedders (your own model, an API client, a deterministic hash
for tests) satisfy the same Embedder Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of *texts*. Returns shape (n, dim), normalized L2."""
        ...

    @property
    def dim(self) -> int: ...


class MiniLMEmbedder:
    """sentence-transformers/all-MiniLM-L6-v2 — small, CPU-fine, 384-dim."""

    _DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    _DIM = 384

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self._DEFAULT_MODEL
        self._model = None  # type: ignore[assignment]

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. Run: uv pip install -e '.[llm]'"
            ) from exc
        self._model = SentenceTransformer(self.model_name, device="cpu")

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        assert self._model is not None
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32, copy=False)

    @property
    def dim(self) -> int:
        return self._DIM
