"""Crossref DOI lookup. Returns metadata; full-text retrieval delegated to
Unpaywall (open-access only, per project copyright policy)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from typing import Any

from medlit.ingestion.base import BaseIngester, IngestionQuery
from medlit.logging import logger
from medlit.models import AccessStatus, Author, Paper, Source
from medlit.utils.http import RateLimiter, get_with_retry, http_client

CROSSREF_API = "https://api.crossref.org/works"


class CrossrefIngester(BaseIngester):
    source = Source.CROSSREF

    def __init__(self, *, mailto: str | None = None, rate_limit_rps: float = 5.0) -> None:
        self.mailto = mailto
        self._limiter = RateLimiter(rps=rate_limit_rps)

    async def search(self, query: IngestionQuery) -> AsyncIterator[Paper]:
        if not query.query:
            return
        params: dict[str, str] = {"query": query.query, "rows": str(query.max_results)}
        if self.mailto:
            params["mailto"] = self.mailto
        if query.date_from:
            params["filter"] = f"from-pub-date:{query.date_from.isoformat()}"
        async with http_client() as client:
            resp = await get_with_retry(client, CROSSREF_API, limiter=self._limiter, params=params)
            data: dict[str, Any] = resp.json()
        items = data.get("message", {}).get("items", [])
        logger.info(f"crossref returned {len(items)} items for {query.query!r}")
        for item in items:
            yield self._to_paper(item)

    async def fetch_one(self, identifier: str) -> Paper | None:
        url = f"{CROSSREF_API}/{identifier}"
        params: dict[str, str] = {"mailto": self.mailto} if self.mailto else {}
        async with http_client() as client:
            try:
                resp = await get_with_retry(client, url, limiter=self._limiter, params=params)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"crossref fetch failed for {identifier}: {e}")
                return None
            data = resp.json()
        return self._to_paper(data["message"])

    def _to_paper(self, item: dict[str, Any]) -> Paper:
        doi = item.get("DOI")
        title_list = item.get("title") or []
        title = title_list[0] if title_list else "(untitled)"
        abstract = item.get("abstract")
        authors = [
            Author(
                name=" ".join(filter(None, [a.get("given"), a.get("family")])).strip()
                or a.get("name", "Unknown"),
                affiliation=(a.get("affiliation") or [{}])[0].get("name") if a.get("affiliation") else None,
                orcid=a.get("ORCID"),
            )
            for a in item.get("author", [])
        ]
        # date-parts: [[YYYY, MM, DD]]
        date_parts = (
            item.get("issued", {}).get("date-parts")
            or item.get("published-print", {}).get("date-parts")
            or item.get("published-online", {}).get("date-parts")
            or [[]]
        )[0]
        pub_date: date | None = None
        if date_parts:
            try:
                y = int(date_parts[0])
                m = int(date_parts[1]) if len(date_parts) > 1 else 1
                d = int(date_parts[2]) if len(date_parts) > 2 else 1
                pub_date = date(y, m, d)
            except (ValueError, TypeError):
                pub_date = None
        journal = (item.get("container-title") or [None])[0]
        pub_types = [item["type"]] if item.get("type") else []
        return Paper(
            id=Paper.make_id(doi=doi, title=title),
            title=title,
            authors=authors,
            abstract=abstract,
            journal=journal,
            publication_date=pub_date,
            doi=doi,
            publication_types=pub_types,
            source=Source.CROSSREF,
            url=item.get("URL"),  # type: ignore[arg-type]
            access_status=AccessStatus.UNKNOWN,
        )
