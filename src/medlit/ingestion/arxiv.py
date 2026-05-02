"""arXiv ingestion via the `arxiv` library (wraps the export.arxiv.org API)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from medlit.ingestion.base import BaseIngester, IngestionQuery
from medlit.logging import logger
from medlit.models import AccessStatus, Author, Paper, Source

if TYPE_CHECKING:
    import arxiv as _arxiv  # type: ignore[import-not-found]


class ArxivIngester(BaseIngester):
    source = Source.ARXIV

    def __init__(self, *, page_size: int = 100, delay_seconds: float = 3.0) -> None:
        import arxiv  # lazy: optional in some envs

        self._arxiv = arxiv
        self._client = arxiv.Client(page_size=page_size, delay_seconds=delay_seconds)

    async def search(self, query: IngestionQuery) -> AsyncIterator[Paper]:
        if not query.query:
            return
        arxiv = self._arxiv
        search = arxiv.Search(
            query=query.query,
            max_results=query.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        logger.info(f"arXiv search: {query.query!r} (max={query.max_results})")
        results = await asyncio.to_thread(lambda: list(self._client.results(search)))
        for r in results:
            paper = self._to_paper(r)
            if (query.date_from and paper.publication_date and paper.publication_date < query.date_from):
                continue
            if (query.date_to and paper.publication_date and paper.publication_date > query.date_to):
                continue
            yield paper

    async def fetch_one(self, identifier: str) -> Paper | None:
        arxiv = self._arxiv
        search = arxiv.Search(id_list=[identifier])
        results = await asyncio.to_thread(lambda: list(self._client.results(search)))
        return self._to_paper(results[0]) if results else None

    @staticmethod
    def _to_paper(r: Any) -> Paper:
        arxiv_id = r.entry_id.rsplit("/", 1)[-1]
        # strip version suffix v1, v2, ...
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.split("v")[0]
        authors = [Author(name=a.name) for a in r.authors]
        return Paper(
            id=Paper.make_id(arxiv_id=arxiv_id, title=r.title),
            title=r.title.strip(),
            authors=authors,
            abstract=r.summary.strip() if r.summary else None,
            journal=r.journal_ref,
            publication_date=r.published.date() if r.published else None,
            doi=r.doi,
            arxiv_id=arxiv_id,
            keywords=list(r.categories or []),
            source=Source.ARXIV,
            url=r.entry_id,  # type: ignore[arg-type]
            access_status=AccessStatus.OPEN,
        )
