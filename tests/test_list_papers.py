from __future__ import annotations

from datetime import date

import numpy as np

from medlit.models import Chunk, Source


def test_list_papers_aggregates_chunks(fake_embedder, in_memory_store):  # type: ignore[no-untyped-def]
    chunks = [
        Chunk(
            id=f"p1:{i}",
            paper_id="p1",
            text=f"chunk {i} of paper one",
            chunk_index=i,
            section="Body",
            title="Paper One",
            authors=["Liu D", "Smith J"],
            journal="Nature",
            publication_date=date(2024, 5, 1),
            source=Source.PUBMED,
            pmid="111",
            citation_token="PMID:111",
            url="https://pubmed.ncbi.nlm.nih.gov/111/",
        )
        for i in range(3)
    ]
    chunks.append(
        Chunk(
            id="p2:0",
            paper_id="p2",
            text="single chunk paper",
            chunk_index=0,
            title="Paper Two",
            authors=["Doe A"],
            source=Source.ARXIV,
            arxiv_id="2401.00001",
            citation_token="arXiv:2401.00001",
            url="https://arxiv.org/abs/2401.00001",
        )
    )
    vecs = fake_embedder.embed_documents([c.text for c in chunks])
    in_memory_store.upsert(chunks, vecs)

    papers = in_memory_store.list_papers()
    assert len(papers) == 2
    by_id = {p.paper_id: p for p in papers}
    assert by_id["p1"].chunk_count == 3
    assert by_id["p1"].title == "Paper One"
    assert by_id["p1"].authors == ["Liu D", "Smith J"]
    assert by_id["p1"].url == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert by_id["p2"].chunk_count == 1
    assert by_id["p2"].source == Source.ARXIV
