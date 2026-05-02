"""bioRxiv / medRxiv ingestion via their public JSON API.

API docs: https://api.biorxiv.org/

Note: bioRxiv's `details` endpoint serves all preprints in a date window — we
retrieve and filter client-side by the query string. For broad keyword search
this is not as efficient as PubMed, but is the only sanctioned route.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from datetime import date, timedelta
from typing import Any

from medlit.ingestion.base import BaseIngester, IngestionQuery
from medlit.logging import logger
from medlit.models import AccessStatus, Author, Paper, Source
from medlit.utils.http import RateLimiter, get_with_retry, http_client

API_BASE = "https://api.biorxiv.org/details"


class BiorxivIngester(BaseIngester):
    source = Source.BIORXIV

    def __init__(self, *, server: str = "biorxiv", rate_limit_rps: float = 2.0) -> None:
        if server not in {"biorxiv", "medrxiv"}:
            raise ValueError("server must be 'biorxiv' or 'medrxiv'")
        self.server = server
        self._limiter = RateLimiter(rps=rate_limit_rps)
        self.source = Source.BIORXIV if server == "biorxiv" else Source.MEDRXIV

    async def search(self, query: IngestionQuery) -> AsyncIterator[Paper]:
        date_to = query.date_to or date.today()
        date_from = query.date_from or (date_to - timedelta(days=30))
        pattern = re.compile(query.query, re.IGNORECASE) if query.query else None
        logger.info(
            f"{self.server} window {date_from}..{date_to} query={query.query!r}"
        )
        cursor = 0
        emitted = 0
        async with http_client(timeout=30.0) as client:
            while emitted < query.max_results:
                url = f"{API_BASE}/{self.server}/{date_from}/{date_to}/{cursor}"
                resp = await get_with_retry(client, url, limiter=self._limiter)
                payload: dict[str, Any] = resp.json()
                msg = payload.get("messages", [{}])[0]
                items = payload.get("collection", [])
                if not items:
                    break
                for item in items:
                    if pattern:
                        haystack = " ".join(
                            str(item.get(k, ""))
                            for k in ("title", "abstract", "authors", "category")
                        )
                        if not pattern.search(haystack):
                            continue
                    paper = self._to_paper(item)
                    yield paper
                    emitted += 1
                    if emitted >= query.max_results:
                        return
                count = int(msg.get("count", 0))
                cursor += count
                if count == 0:
                    break

    async def fetch_one(self, identifier: str) -> Paper | None:
        # identifier is a DOI like "10.1101/2024.01.01.000001"
        url = f"{API_BASE}/{self.server}/{identifier}"
        async with http_client() as client:
            resp = await get_with_retry(client, url, limiter=self._limiter)
            data: dict[str, Any] = resp.json()
        items = data.get("collection") or []
        return self._to_paper(items[0]) if items else None

    def _to_paper(self, item: dict[str, Any]) -> Paper:
        doi = item.get("doi")
        title = item.get("title") or "(untitled)"
        abstract = item.get("abstract")
        authors_raw = item.get("authors") or ""
        authors = [
            Author(name=a.strip())
            for a in re.split(r";|,", str(authors_raw))
            if a.strip()
        ]
        date_str = item.get("date")
        try:
            pub_date = date.fromisoformat(date_str) if date_str else None
        except ValueError:
            pub_date = None
        url = f"https://doi.org/{doi}" if doi else None
        return Paper(
            id=Paper.make_id(doi=doi, title=title),
            title=title,
            authors=authors,
            abstract=abstract,
            journal=self.server,
            publication_date=pub_date,
            doi=doi,
            keywords=[item.get("category")] if item.get("category") else [],
            source=self.source,
            url=url,  # type: ignore[arg-type]
            access_status=AccessStatus.OPEN,
        )
