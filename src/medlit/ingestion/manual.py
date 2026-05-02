"""Manual ingestion: PDF, XML, or plain text files supplied by the user."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from medlit.ingestion.base import BaseIngester, IngestionQuery
from medlit.logging import logger
from medlit.models import AccessStatus, Paper, Source


class ManualIngester(BaseIngester):
    """Wrap a single uploaded file (or directory of files) as Paper objects.

    The actual *parsing* of file contents happens later in the processing
    layer; this ingester just produces a Paper shell with metadata so the
    pipeline downstream can hand the file path to the right parser.
    """

    source = Source.MANUAL

    def __init__(self, paths: list[Path]) -> None:
        self.paths = [Path(p) for p in paths]

    async def search(self, query: IngestionQuery) -> AsyncIterator[Paper]:
        for p in self.paths:
            if not p.exists():
                logger.warning(f"manual ingest: path missing: {p}")
                continue
            if p.is_dir():
                for child in sorted(p.rglob("*")):
                    if child.is_file() and child.suffix.lower() in {".pdf", ".xml", ".txt", ".md"}:
                        yield self._shell_paper(child)
            else:
                yield self._shell_paper(p)

    async def fetch_one(self, identifier: str) -> Paper | None:
        path = Path(identifier)
        if not path.exists() or not path.is_file():
            return None
        return self._shell_paper(path)

    @staticmethod
    def _shell_paper(path: Path) -> Paper:
        title = path.stem.replace("_", " ").strip() or path.name
        return Paper(
            id=Paper.make_id(title=str(path.resolve())),
            title=title,
            source=Source.MANUAL,
            access_status=AccessStatus.OPEN,
            url=None,
            raw={"path": str(path.resolve()), "suffix": path.suffix.lower()},
        )
