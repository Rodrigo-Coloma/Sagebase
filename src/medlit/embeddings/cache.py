"""On-disk embedding cache. Keyed by sha1(model_name + text)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from medlit.logging import logger


class EmbeddingCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, text: str) -> str:
        h = hashlib.sha1()
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        # Shard by first 2 chars to avoid huge dirs.
        return self.cache_dir / key[:2] / f"{key}.npy"

    def get_many(self, model: str, texts: Sequence[str]) -> tuple[list[int], list[np.ndarray]]:
        """Return (hit_indices, hit_vectors)."""
        hit_idx: list[int] = []
        hit_vec: list[np.ndarray] = []
        for i, t in enumerate(texts):
            p = self._path(self._key(model, t))
            if p.exists():
                try:
                    hit_vec.append(np.load(p))
                    hit_idx.append(i)
                except (OSError, ValueError) as e:  # corrupted file
                    logger.warning(f"corrupted cache entry {p}: {e}")
                    p.unlink(missing_ok=True)
        return hit_idx, hit_vec

    def put_many(self, model: str, texts: Sequence[str], vectors: np.ndarray) -> None:
        for t, v in zip(texts, vectors, strict=True):
            p = self._path(self._key(model, t))
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, v.astype(np.float32, copy=False))
