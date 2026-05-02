"""Paper-level deduplication helpers."""

from __future__ import annotations

from collections.abc import Iterable

from medlit.models import Paper


def dedup_papers(papers: Iterable[Paper]) -> list[Paper]:
    """De-duplicate by DOI > PMID > arXiv id > title-hash. Stable order kept."""
    seen: set[str] = set()
    out: list[Paper] = []
    for p in papers:
        keys: list[str] = []
        if p.doi:
            keys.append(f"doi:{p.doi.lower()}")
        if p.pmid:
            keys.append(f"pmid:{p.pmid}")
        if p.arxiv_id:
            keys.append(f"arxiv:{p.arxiv_id}")
        if not keys and p.title:
            keys.append(f"title:{p.title.lower().strip()}")
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        out.append(p)
    return out
