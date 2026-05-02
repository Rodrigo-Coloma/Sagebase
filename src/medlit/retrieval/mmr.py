"""Maximal Marginal Relevance for diversity in retrieved chunks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from medlit.models import Chunk


def maximal_marginal_relevance(
    query_vec: np.ndarray,
    candidates: Sequence[tuple[Chunk, np.ndarray]],
    *,
    k: int,
    lambda_mult: float = 0.5,
) -> list[Chunk]:
    """Greedy MMR. `candidates`: list of (chunk, embedding) — embeddings must be unit-normalized."""
    if not candidates:
        return []
    chunks, embs = zip(*candidates, strict=True)
    embs_arr = np.asarray(list(embs))
    sim_to_query = embs_arr @ query_vec
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        best_idx = -1
        best_score = -np.inf
        for i in remaining:
            if not selected:
                score = sim_to_query[i]
            else:
                sim_to_selected = float(np.max(embs_arr[selected] @ embs_arr[i]))
                score = lambda_mult * sim_to_query[i] - (1 - lambda_mult) * sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [chunks[i] for i in selected]
