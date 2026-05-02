"""Section-aware chunker.

Strategy:
1. If we have IMRaD sections (from PMC XML or PDF heading detection), chunk
   *within* each section so a chunk never crosses Methods <-> Results.
2. Within a section, use RecursiveCharacterTextSplitter targeted at a token
   budget (default 512) with overlap (default 50). Tokens counted with
   tiktoken (cl100k_base) — close enough for both ST and OpenAI models.
3. Tables/figures (already rendered as `[Table 1] ...` placeholders by the
   PMC parser) are kept as their own atomic chunk so we never split them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from medlit.config import ChunkingConfig
from medlit.models import Chunk, Paper
from medlit.processing.pmc_xml import ParsedPMC


@lru_cache(maxsize=1)
def _encoder():  # type: ignore[no-untyped-def]
    """Lazily load tiktoken's cl100k_base; fall back to char/4 estimate
    if the BPE blob can't be downloaded."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001
        return None


_TABLE_OR_FIG = re.compile(r"^\[(Table|Figure|Fig\.?)[^\]]*\]", re.IGNORECASE)
_HEADING_RE = re.compile(
    r"^(abstract|introduction|background|methods?|materials and methods|"
    r"results?|discussion|conclusions?|references?)\s*$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class _Block:
    section: str
    text: str
    page_number: int | None = None
    atomic: bool = False  # do not further split (e.g. a table)


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is None:
        # Fallback: ~4 characters per token. Good enough for chunk sizing.
        return max(1, len(text) // 4)
    return len(enc.encode(text, disallowed_special=()))


class SectionAwareChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        # Clamp overlap so it stays strictly smaller than chunk size even when
        # callers pass an unusually small chunk_size.
        chunk_chars = max(self.config.chunk_size * 4, 16)
        overlap_chars = min(self.config.chunk_overlap * 4, chunk_chars // 2)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=count_tokens,
            is_separator_regex=False,
        )

    # ---------------------------------------------------------------- inputs
    def chunk_paper(self, paper: Paper, *, full_text_blocks: list[_Block] | None = None) -> list[Chunk]:
        """Chunk a Paper. If `full_text_blocks` provided, use them; otherwise
        fall back to the abstract."""
        blocks = full_text_blocks
        if not blocks:
            if paper.abstract:
                blocks = [_Block(section="Abstract", text=paper.abstract)]
            elif paper.full_text:
                blocks = [_Block(section="Body", text=paper.full_text)]
            else:
                return []
        return self._chunks_from_blocks(paper, blocks)

    def chunk_pmc(self, paper: Paper, parsed: ParsedPMC) -> list[Chunk]:
        blocks: list[_Block] = []
        for section, text in parsed.iter_flat():
            for piece in _split_table_blocks(section, text):
                blocks.append(piece)
        return self._chunks_from_blocks(paper, blocks)

    def chunk_pages(self, paper: Paper, pages: list[tuple[int, str]]) -> list[Chunk]:
        """Chunk page-text from a PDF. Heading detection is heuristic."""
        blocks: list[_Block] = []
        current_section = "Body"
        for page_no, text in pages:
            for line_block in _split_by_headings(text):
                section, body = line_block
                if section:
                    current_section = section
                if not body.strip():
                    continue
                for piece in _split_table_blocks(current_section, body, page=page_no):
                    blocks.append(piece)
        return self._chunks_from_blocks(paper, blocks)

    # ------------------------------------------------------------- internals
    def _chunks_from_blocks(self, paper: Paper, blocks: list[_Block]) -> list[Chunk]:
        chunks: list[Chunk] = []
        idx = 0
        for block in blocks:
            pieces: list[str]
            if block.atomic or count_tokens(block.text) <= self.config.chunk_size:
                pieces = [block.text]
            else:
                pieces = self._splitter.split_text(block.text)
            for piece in pieces:
                if not piece.strip():
                    continue
                chunk_id = Chunk.make_id(paper.id, idx)
                chunks.append(self._build_chunk(paper, chunk_id, idx, block, piece))
                idx += 1
        return chunks

    @staticmethod
    def _build_chunk(
        paper: Paper, chunk_id: str, index: int, block: _Block, text: str
    ) -> Chunk:
        return Chunk(
            id=chunk_id,
            paper_id=paper.id,
            text=text,
            chunk_index=index,
            section=block.section,
            page_number=block.page_number,
            token_count=count_tokens(text),
            title=paper.title,
            authors=[a.name for a in paper.authors],
            journal=paper.journal,
            publication_date=paper.publication_date,
            doi=paper.doi,
            pmid=paper.pmid,
            arxiv_id=paper.arxiv_id,
            mesh_terms=paper.mesh_terms,
            publication_types=paper.publication_types,
            source=paper.source,
            url=str(paper.url) if paper.url else None,
            citation_token=paper.citation_token,
        )


def _split_by_headings(text: str) -> list[tuple[str | None, str]]:
    """Heuristic IMRaD heading split for PDF text."""
    lines = text.split("\n")
    out: list[tuple[str | None, str]] = []
    buf: list[str] = []
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if _HEADING_RE.match(stripped):
            if buf:
                out.append((current, "\n".join(buf).strip()))
                buf = []
            current = stripped.title()
        else:
            buf.append(line)
    if buf:
        out.append((current, "\n".join(buf).strip()))
    return out or [(None, text)]


def _split_table_blocks(section: str, text: str, *, page: int | None = None) -> list[_Block]:
    """Pull figure/table placeholder paragraphs out as atomic blocks."""
    paragraphs = re.split(r"\n{2,}", text)
    out: list[_Block] = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        atomic = bool(_TABLE_OR_FIG.match(p))
        out.append(_Block(section=section, text=p, page_number=page, atomic=atomic))
    return out


__all__ = ["SectionAwareChunker", "count_tokens"]
