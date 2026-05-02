from __future__ import annotations

from medlit.models import Paper, Source
from medlit.utils.dedup import dedup_papers


def test_dedup_by_doi():
    a = Paper(id="1", title="A", source=Source.PUBMED, doi="10.1/x")
    b = Paper(id="2", title="B", source=Source.CROSSREF, doi="10.1/x")
    out = dedup_papers([a, b])
    assert len(out) == 1
    assert out[0] is a  # first occurrence wins


def test_dedup_by_pmid_when_no_doi():
    a = Paper(id="1", title="A", source=Source.PUBMED, pmid="42")
    b = Paper(id="2", title="A different", source=Source.PUBMED, pmid="42")
    out = dedup_papers([a, b])
    assert len(out) == 1


def test_dedup_keeps_unique():
    a = Paper(id="1", title="A", source=Source.PUBMED, pmid="1")
    b = Paper(id="2", title="B", source=Source.PUBMED, pmid="2")
    assert len(dedup_papers([a, b])) == 2
