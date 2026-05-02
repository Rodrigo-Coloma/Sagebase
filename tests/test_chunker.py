from __future__ import annotations

from pathlib import Path

from medlit.config import ChunkingConfig
from medlit.models import Paper, Source
from medlit.processing.chunker import SectionAwareChunker, count_tokens
from medlit.processing.pmc_xml import parse_pmc_xml

FIXTURES = Path(__file__).parent / "fixtures"


def _paper() -> Paper:
    return Paper(id="p1", title="t", source=Source.PUBMED, pmid="1", doi="10.1/x")


def test_chunk_paper_uses_abstract_when_no_full_text():
    paper = _paper()
    paper = paper.model_copy(update={"abstract": "Hello world. " * 20})
    chunker = SectionAwareChunker()
    chunks = chunker.chunk_paper(paper)
    assert chunks
    assert all(c.section == "Abstract" for c in chunks)
    assert all(c.title == paper.title for c in chunks)
    assert all(c.citation_token == "PMID:1" for c in chunks)


def test_chunk_pmc_keeps_section_titles():
    parsed = parse_pmc_xml((FIXTURES / "pmc_sample.xml").read_bytes())
    paper = _paper()
    chunks = SectionAwareChunker(ChunkingConfig(chunk_size=64)).chunk_pmc(paper, parsed)
    sections = {c.section for c in chunks}
    assert "Methods" in sections
    assert any("Methods > Cell culture" in s for s in sections if s)
    assert "Results" in sections


def test_chunk_pmc_table_is_atomic():
    parsed = parse_pmc_xml((FIXTURES / "pmc_sample.xml").read_bytes())
    paper = _paper()
    chunks = SectionAwareChunker(ChunkingConfig(chunk_size=10)).chunk_pmc(paper, parsed)
    table_chunks = [c for c in chunks if c.text.startswith("[Table 1]")]
    assert table_chunks, "Table placeholder should appear as its own chunk"


def test_count_tokens_monotonic():
    assert count_tokens("hello") < count_tokens("hello world goodbye")
