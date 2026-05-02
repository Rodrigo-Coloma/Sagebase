from __future__ import annotations

from medlit.models import Chunk, Paper, Source


def test_paper_make_id_prefers_doi():
    a = Paper.make_id(doi="10.1/abc", pmid="123", title="x")
    b = Paper.make_id(doi="10.1/abc", pmid="999", title="y")
    assert a == b


def test_paper_make_id_normalizes_doi_case():
    a = Paper.make_id(doi="10.1/ABC")
    b = Paper.make_id(doi="10.1/abc")
    assert a == b


def test_paper_doi_normalization():
    p = Paper(id="x", title="t", source=Source.CROSSREF, doi="https://doi.org/10.1/ABC")
    assert p.doi == "10.1/abc"


def test_citation_token_priority():
    p = Paper(id="x", title="t", source=Source.PUBMED, pmid="42", doi="10.1/x")
    assert p.citation_token == "PMID:42"
    p2 = Paper(id="x", title="t", source=Source.CROSSREF, doi="10.1/x")
    assert p2.citation_token == "DOI:10.1/x"


def test_chunk_id():
    assert Chunk.make_id("p1", 3) == "p1:3"
