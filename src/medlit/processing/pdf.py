"""PDF parsing with PyMuPDF (fitz). Yields per-page text with page numbers.

For complex multi-column or scanned PDFs, install the `unstructured` extra
and call `parse_pdf_unstructured` as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medlit.logging import logger
from medlit.processing.normalizer import normalize_text


@dataclass(slots=True)
class PageText:
    page_number: int  # 1-indexed
    text: str


def parse_pdf(path: str | Path) -> list[PageText]:
    """Extract text page-by-page. Empty pages are skipped."""
    import fitz  # PyMuPDF; imported lazily so tests w/o it can still load module

    pages: list[PageText] = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            raw = page.get_text("text") or ""
            cleaned = normalize_text(raw)
            if cleaned:
                pages.append(PageText(page_number=i, text=cleaned))
    logger.debug(f"parsed {len(pages)} non-empty pages from {path}")
    return pages


def parse_pdf_unstructured(path: str | Path) -> list[PageText]:
    """Optional fallback using `unstructured` for complex layouts."""
    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "Install medlit[unstructured] to use the unstructured fallback."
        ) from e

    elements = partition_pdf(filename=str(path), strategy="hi_res")
    by_page: dict[int, list[str]] = {}
    for el in elements:
        pn = getattr(el.metadata, "page_number", None) or 1
        by_page.setdefault(pn, []).append(str(el))
    return [
        PageText(page_number=pn, text=normalize_text("\n".join(parts)))
        for pn, parts in sorted(by_page.items())
        if parts
    ]
