"""Pluggable ingestion sources. All implement BaseIngester -> Paper."""

from medlit.ingestion.base import BaseIngester, IngestionQuery
from medlit.ingestion.arxiv import ArxivIngester
from medlit.ingestion.biorxiv import BiorxivIngester
from medlit.ingestion.crossref import CrossrefIngester
from medlit.ingestion.manual import ManualIngester
from medlit.ingestion.pubmed import PubMedIngester
from medlit.ingestion.unpaywall import UnpaywallClient

__all__ = [
    "ArxivIngester",
    "BaseIngester",
    "BiorxivIngester",
    "CrossrefIngester",
    "IngestionQuery",
    "ManualIngester",
    "PubMedIngester",
    "UnpaywallClient",
]
