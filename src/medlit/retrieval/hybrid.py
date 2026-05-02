"""Reciprocal Rank Fusion: merges dense + sparse rankings without
needing per-system score calibration."""

from __future__ import annotations

from collections.abc import Sequence

from medlit.models import Chunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[Chunk, float]]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[Chunk, float]]:
    """Fuse multiple rankings via RRF: score = sum_i 1 / (k + rank_i).

    `rankings` is a list of result lists (each: list of (chunk, score)).
    The original scores are ignored — only ranks are used.
    """
    fused: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            key = chunk.id or f"{chunk.paper_id}:{chunk.chunk_index}"
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            chunk_by_id.setdefault(key, chunk)
    out = sorted(
        ((chunk_by_id[cid], score) for cid, score in fused.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return out[:top_n] if top_n else out
