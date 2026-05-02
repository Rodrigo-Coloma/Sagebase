"""Common ingestion interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date

from medlit.models import Paper, Source


@dataclass(slots=True)
class IngestionQuery:
    """Generic query parameters; individual ingesters consume what they need."""

    query: str | None = None
    max_results: int = 50
    date_from: date | None = None
    date_to: date | None = None
    authors: list[str] | None = None
    mesh_terms: list[str] | None = None
    journal: str | None = None
    extra: dict[str, str] | None = None


class BaseIngester(ABC):
    """Abstract ingester. All sources yield normalized `Paper` objects."""

    source: Source

    @abstractmethod
    async def search(self, query: IngestionQuery) -> AsyncIterator[Paper]:
        """Yield papers matching the query. Streams to allow large result sets."""
        ...

    @abstractmethod
    async def fetch_one(self, identifier: str) -> Paper | None:
        """Fetch a single paper by source-native identifier."""
        ...

    async def close(self) -> None:
        """Override if the ingester holds open resources."""
        return None
