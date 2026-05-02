from __future__ import annotations

from datetime import date
from pathlib import Path

from medlit.ingestion.pubmed import parse_pubmed_xml

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_pubmed_two_articles():
    xml = (FIXTURES / "pubmed_two.xml").read_bytes()
    papers = parse_pubmed_xml(xml)
    assert len(papers) == 2

    p1 = papers[0]
    assert p1.pmid == "11111111"
    assert p1.doi == "10.1038/s41586-023-99999"
    assert p1.pmcid == "PMC9999999"
    assert p1.journal == "Nature"
    assert p1.publication_date == date(2023, 8, 15)
    assert "BACKGROUND" in (p1.abstract or "")
    assert any(a.name == "David Liu" for a in p1.authors)
    assert "CRISPR-Cas Systems" in p1.mesh_terms
    assert "Randomized Controlled Trial" in p1.publication_types
    assert p1.citation_token == "PMID:11111111"

    p2 = papers[1]
    assert p2.publication_date == date(2024, 1, 1)  # MedlineDate "2024 Spring" -> year fallback
    assert p2.authors[0].name == "Editas Consortium"


def test_pubmed_id_stable_across_normalizations():
    xml = (FIXTURES / "pubmed_two.xml").read_bytes()
    papers = parse_pubmed_xml(xml)
    p1 = papers[0]
    # Building an id from the same DOI should be stable.
    from medlit.models import Paper as PaperModel
    assert p1.id == PaperModel.make_id(doi=p1.doi, pmid=p1.pmid, title=p1.title)
