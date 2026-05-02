"""High-level orchestration: ingest -> parse -> chunk -> embed -> upsert.

This is the primary entry point used by both the CLI and the API.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from medlit.config import get_settings
from medlit.embeddings.factory import CachedEmbedder, build_embedder
from medlit.ingestion.base import BaseIngester, IngestionQuery
from medlit.ingestion.manual import ManualIngester
from medlit.ingestion.pubmed import PubMedIngester
from medlit.ingestion.unpaywall import UnpaywallClient
from medlit.logging import logger
from medlit.models import Chunk, Paper, Source
from medlit.processing.chunker import SectionAwareChunker
from medlit.processing.pdf import parse_pdf
from medlit.processing.pmc_xml import parse_pmc_xml
from medlit.storage.base import VectorStore
from medlit.storage.factory import build_vector_store
from medlit.utils.dedup import dedup_papers


class IngestionPipeline:
    def __init__(
        self,
        *,
        embedder: CachedEmbedder | None = None,
        store: VectorStore | None = None,
        chunker: SectionAwareChunker | None = None,
        unpaywall: UnpaywallClient | None = None,
    ) -> None:
        self.embedder = embedder or build_embedder()
        self.store = store or build_vector_store()
        self.store.ensure_collection(dimension=self.embedder.dimension)
        self.chunker = chunker or SectionAwareChunker(get_settings().chunking)
        self.unpaywall = unpaywall or UnpaywallClient()
        self._seen_paper_ids: set[str] = set()

    # -------------------------------------------------------------- main API
    async def ingest_from(
        self,
        ingester: BaseIngester,
        query: IngestionQuery,
        *,
        with_full_text: bool = True,
    ) -> AsyncIterator[Paper]:
        """Search via ingester, chunk, embed, and upsert. Yields papers
        as they're indexed (so callers can stream progress)."""
        async for paper in ingester.search(query):
            try:
                indexed = await self._process_paper(paper, ingester=ingester, with_full_text=with_full_text)
                if indexed:
                    yield paper
            except Exception as e:  # noqa: BLE001
                logger.exception(f"failed to ingest {paper.id}: {e}")

    async def ingest_files(self, paths: list[Path]) -> AsyncIterator[Paper]:
        ingester = ManualIngester(paths)
        async for paper in self.ingest_from(ingester, IngestionQuery(), with_full_text=True):
            yield paper

    # ----------------------------------------------------------- per-paper
    async def _process_paper(
        self,
        paper: Paper,
        *,
        ingester: BaseIngester,
        with_full_text: bool,
    ) -> bool:
        if paper.id in self._seen_paper_ids:
            logger.debug(f"skip already-seen {paper.id}")
            return False
        self._seen_paper_ids.add(paper.id)

        chunks = await self._make_chunks(paper, ingester=ingester, with_full_text=with_full_text)
        if not chunks:
            logger.warning(f"no chunks produced for {paper.title!r}")
            return False
        texts = [c.text for c in chunks]
        vectors = await asyncio.to_thread(self.embedder.embed_documents, texts)
        await asyncio.to_thread(self.store.upsert, chunks, vectors)
        logger.info(f"indexed {len(chunks)} chunks for {paper.title[:80]!r}")
        return True

    async def _make_chunks(
        self,
        paper: Paper,
        *,
        ingester: BaseIngester,
        with_full_text: bool,
    ) -> list[Chunk]:
        # 1) Manual file path -> parse PDF/XML/text
        if paper.source == Source.MANUAL:
            path_str = paper.raw.get("path") if paper.raw else None
            if path_str:
                return self._chunk_local_file(paper, Path(path_str))

        # 2) PubMed paper with PMCID -> fetch PMC XML if open access
        if with_full_text and isinstance(ingester, PubMedIngester) and paper.pmcid:
            try:
                xml = await ingester.fetch_pmc_fulltext(paper.pmcid)
                if xml:
                    parsed = await asyncio.to_thread(parse_pmc_xml, xml)
                    chunks = self.chunker.chunk_pmc(paper, parsed)
                    if chunks:
                        return chunks
            except Exception as e:  # noqa: BLE001
                logger.warning(f"PMC fulltext fetch failed for {paper.pmcid}: {e}")

        # 3) Try Unpaywall PDF (open-access only)
        if with_full_text and paper.doi and get_settings().ingestion.unpaywall_enabled:
            pdf_url = await self.unpaywall.best_oa_pdf_url(paper.doi)
            if pdf_url:
                pdf_path = await self._download(pdf_url, key=paper.id)
                if pdf_path:
                    return self._chunk_local_file(paper, pdf_path)

        # 4) Fallback: just chunk the abstract.
        return self.chunker.chunk_paper(paper)

    def _chunk_local_file(self, paper: Paper, path: Path) -> list[Chunk]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = parse_pdf(path)
            return self.chunker.chunk_pages(paper, [(p.page_number, p.text) for p in pages])
        if suffix == ".xml":
            parsed = parse_pmc_xml(path.read_bytes())
            return self.chunker.chunk_pmc(paper, parsed)
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            paper = paper.model_copy(update={"full_text": text})
            return self.chunker.chunk_paper(paper)
        logger.warning(f"unsupported file type: {path}")
        return []

    async def _download(self, url: str, *, key: str) -> Path | None:
        cache_dir = Path(get_settings().medlit_cache_dir) / "pdfs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{key}.pdf"
        if path.exists():
            return path
        from medlit.utils.http import http_client

        try:
            async with http_client(timeout=60.0, follow_redirects=True) as client:  # type: ignore[arg-type]
                resp = await client.get(url)
                resp.raise_for_status()
                if "pdf" not in resp.headers.get("content-type", "").lower():
                    return None
                path.write_bytes(resp.content)
                return path
        except Exception as e:  # noqa: BLE001
            logger.warning(f"download failed {url}: {e}")
            return None


# ------------------------------------------------------------------ convenience
async def ingest_from_papers(papers: list[Paper], pipeline: IngestionPipeline) -> int:
    """Ingest a pre-fetched list of papers. Returns count indexed."""
    deduped = dedup_papers(papers)
    indexed = 0
    for p in deduped:
        try:
            ok = await pipeline._process_paper(  # noqa: SLF001
                p, ingester=ManualIngester([]), with_full_text=False
            )
            if ok:
                indexed += 1
        except Exception as e:  # noqa: BLE001
            logger.exception(f"ingest_from_papers failed for {p.id}: {e}")
    return indexed
