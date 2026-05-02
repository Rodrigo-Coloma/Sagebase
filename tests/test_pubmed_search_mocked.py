from __future__ import annotations

from pathlib import Path

import pytest
import respx
from httpx import Response

from medlit.ingestion.base import IngestionQuery
from medlit.ingestion.pubmed import PubMedIngester

FIXTURES = Path(__file__).parent / "fixtures"

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@pytest.mark.asyncio
async def test_pubmed_search_returns_papers():
    fetch_xml = (FIXTURES / "pubmed_two.xml").read_bytes()
    with respx.mock(assert_all_called=False) as m:
        m.get(ESEARCH).mock(
            return_value=Response(
                200,
                json={"esearchresult": {"idlist": ["11111111", "22222222"]}},
            )
        )
        m.get(EFETCH).mock(return_value=Response(200, content=fetch_xml))

        ingester = PubMedIngester(api_key=None, email="t@t", tool="test", rate_limit_rps=100)
        q = IngestionQuery(query="base editing", max_results=10)
        out = []
        async for p in ingester.search(q):
            out.append(p)
        assert len(out) == 2
        pmids = {p.pmid for p in out}
        assert pmids == {"11111111", "22222222"}


@pytest.mark.asyncio
async def test_pubmed_search_term_includes_filters():
    captured: dict[str, str] = {}

    def _capture(request):
        captured.update(dict(request.url.params))
        return Response(200, json={"esearchresult": {"idlist": []}})

    with respx.mock(assert_all_called=False) as m:
        m.get(ESEARCH).mock(side_effect=_capture)

        ingester = PubMedIngester(api_key="KEY", email="t@t", tool="test", rate_limit_rps=100)
        q = IngestionQuery(
            query="CRISPR",
            max_results=5,
            authors=["Liu D"],
            mesh_terms=["Gene Editing"],
            journal="Nature",
        )
        async for _ in ingester.search(q):
            pass
        term = captured.get("term", "")
        assert "CRISPR" in term
        assert "Liu D[Author]" in term
        assert "Gene Editing[MeSH Terms]" in term
        assert "Nature[Journal]" in term
        assert captured.get("api_key") == "KEY"
