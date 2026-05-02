"""PubMed / PMC ingestion via NCBI E-utilities.

We deliberately avoid Biopython's blocking Bio.Entrez here in favor of
async httpx so we play well with FastAPI. The same endpoints are used.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from datetime import date, datetime
from typing import Any

from lxml import etree

from medlit.config import get_settings
from medlit.ingestion.base import BaseIngester, IngestionQuery
from medlit.logging import logger
from medlit.models import AccessStatus, Author, Paper, Source
from medlit.utils.http import RateLimiter, get_with_retry, http_client

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH = f"{EUTILS}/esearch.fcgi"
EFETCH = f"{EUTILS}/efetch.fcgi"
ESUMMARY = f"{EUTILS}/esummary.fcgi"
PMC_OA = f"{EUTILS}/efetch.fcgi"   # db=pmc returns NXML for OA papers


class PubMedIngester(BaseIngester):
    source = Source.PUBMED

    def __init__(
        self,
        *,
        api_key: str | None = None,
        tool: str | None = None,
        email: str | None = None,
        rate_limit_rps: float | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.ncbi_api_key
        self.tool = tool or settings.ncbi_tool
        self.email = email or settings.ncbi_email
        rps = rate_limit_rps or (
            settings.ingestion.pubmed.rate_limit_rps if self.api_key else 3
        )
        self._limiter = RateLimiter(rps=rps)

    # ------------------------------------------------------------------ public
    async def search(self, query: IngestionQuery) -> AsyncIterator[Paper]:
        term = self._build_term(query)
        logger.info(f"PubMed search: {term!r} (max={query.max_results})")
        pmids = await self._esearch(term, retmax=query.max_results)
        if not pmids:
            return
        # Stream in batches of 100 to respect URL length & memory.
        batch = 100
        for i in range(0, len(pmids), batch):
            chunk_ids = pmids[i : i + batch]
            for paper in await self._efetch_pubmed(chunk_ids):
                yield paper

    async def fetch_one(self, identifier: str) -> Paper | None:
        papers = await self._efetch_pubmed([identifier])
        return papers[0] if papers else None

    async def fetch_pmc_fulltext(self, pmcid: str) -> str | None:
        """Return raw NXML for an open-access PMC article, or None."""
        pmcid_norm = pmcid.removeprefix("PMC")
        params = self._common_params() | {"db": "pmc", "id": pmcid_norm, "rettype": "xml"}
        async with http_client() as client:
            resp = await get_with_retry(client, PMC_OA, limiter=self._limiter, params=params)
            text = resp.text
        return text or None

    # ---------------------------------------------------------------- internals
    def _common_params(self) -> dict[str, str]:
        params: dict[str, str] = {"tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _build_term(self, q: IngestionQuery) -> str:
        parts: list[str] = []
        if q.query:
            parts.append(q.query)
        if q.authors:
            parts.extend(f"{a}[Author]" for a in q.authors)
        if q.mesh_terms:
            parts.extend(f"{m}[MeSH Terms]" for m in q.mesh_terms)
        if q.journal:
            parts.append(f"{q.journal}[Journal]")
        if q.date_from or q.date_to:
            df = q.date_from.strftime("%Y/%m/%d") if q.date_from else "1900/01/01"
            dt = q.date_to.strftime("%Y/%m/%d") if q.date_to else date.today().strftime(
                "%Y/%m/%d"
            )
            parts.append(f"({df}:{dt}[Date - Publication])")
        return " AND ".join(f"({p})" for p in parts) if parts else ""

    async def _esearch(self, term: str, *, retmax: int) -> list[str]:
        params = self._common_params() | {
            "db": "pubmed",
            "term": term,
            "retmax": str(retmax),
            "retmode": "json",
            "sort": "pub_date",
        }
        async with http_client() as client:
            resp = await get_with_retry(client, ESEARCH, limiter=self._limiter, params=params)
            data: dict[str, Any] = resp.json()
        return list(data.get("esearchresult", {}).get("idlist", []))

    async def _efetch_pubmed(self, pmids: list[str]) -> list[Paper]:
        if not pmids:
            return []
        params = self._common_params() | {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        async with http_client() as client:
            resp = await get_with_retry(client, EFETCH, limiter=self._limiter, params=params)
            xml = resp.content
        return parse_pubmed_xml(xml)


# --------------------------------------------------------------------- parsing
def parse_pubmed_xml(xml: bytes) -> list[Paper]:
    """Parse PubMed efetch XML into a list of Paper objects."""
    root = etree.fromstring(xml)  # noqa: S320 - trusted source
    out: list[Paper] = []
    for art in root.findall(".//PubmedArticle"):
        out.append(_parse_article(art))
    return out


def _parse_article(art: etree._Element) -> Paper:
    pmid = _text(art.find(".//PMID"))
    title = _text(art.find(".//ArticleTitle")) or "(untitled)"
    abstract = _join_abstract(art.findall(".//Abstract/AbstractText"))
    journal = _text(art.find(".//Journal/Title"))
    pub_date = _parse_pub_date(art.find(".//Article/Journal/JournalIssue/PubDate"))
    doi = _find_article_id(art, "doi")
    pmcid = _find_article_id(art, "pmc")
    authors = _parse_authors(art.findall(".//AuthorList/Author"))
    mesh = [
        _text(d) or ""
        for d in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
        if _text(d)
    ]
    pub_types = [
        _text(p) or ""
        for p in art.findall(".//PublicationTypeList/PublicationType")
        if _text(p)
    ]
    keywords = [_text(k) or "" for k in art.findall(".//KeywordList/Keyword") if _text(k)]
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
    paper_id = Paper.make_id(doi=doi, pmid=pmid, title=title)
    access = AccessStatus.OPEN if pmcid else AccessStatus.UNKNOWN
    return Paper(
        id=paper_id,
        title=title,
        authors=authors,
        abstract=abstract,
        journal=journal,
        publication_date=pub_date,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        mesh_terms=mesh,
        keywords=keywords,
        publication_types=pub_types,
        source=Source.PUBMED,
        url=url,  # type: ignore[arg-type]
        access_status=access,
    )


def _text(el: etree._Element | None) -> str | None:
    if el is None:
        return None
    s = "".join(el.itertext()).strip()
    return s or None


def _join_abstract(nodes: Iterable[etree._Element]) -> str | None:
    parts: list[str] = []
    for n in nodes:
        label = n.get("Label")
        text = "".join(n.itertext()).strip()
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts) if parts else None


def _parse_pub_date(el: etree._Element | None) -> date | None:
    if el is None:
        return None
    medline = _text(el.find("MedlineDate"))
    if medline:
        for fmt in ("%Y %b %d", "%Y %b", "%Y"):
            try:
                return datetime.strptime(medline.split("-")[0].strip(), fmt).date()
            except ValueError:
                continue
        # Last resort: pull a 4-digit year out (handles "2024 Spring", "Winter 2024", etc.)
        import re
        m = re.search(r"(\d{4})", medline)
        if m:
            try:
                return date(int(m.group(1)), 1, 1)
            except ValueError:
                pass
    year = _text(el.find("Year"))
    month = _text(el.find("Month")) or "Jan"
    day = _text(el.find("Day")) or "1"
    if not year:
        return None
    for fmt in ("%Y %b %d", "%Y %m %d"):
        try:
            return datetime.strptime(f"{year} {month} {day}", fmt).date()
        except ValueError:
            continue
    try:
        return date(int(year), 1, 1)
    except ValueError:
        return None


def _find_article_id(art: etree._Element, id_type: str) -> str | None:
    for aid in art.findall(".//ArticleIdList/ArticleId"):
        if aid.get("IdType") == id_type:
            return _text(aid)
    return None


def _parse_authors(nodes: Iterable[etree._Element]) -> list[Author]:
    authors: list[Author] = []
    for n in nodes:
        last = _text(n.find("LastName")) or ""
        fore = _text(n.find("ForeName")) or _text(n.find("Initials")) or ""
        collective = _text(n.find("CollectiveName"))
        affiliation = _text(n.find(".//Affiliation"))
        name = collective or f"{fore} {last}".strip()
        if not name:
            continue
        authors.append(Author(name=name, affiliation=affiliation))
    return authors
