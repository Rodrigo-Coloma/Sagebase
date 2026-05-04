"""Qdrant-backed VectorStore. Dense via HNSW, sparse via local BM25.

Qdrant supports server-side sparse vectors but configuring them requires
specific tokenizer setup; for portability we keep BM25 in-process via
rank_bm25. The retrieval layer fuses dense + sparse with RRF.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import date

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from rank_bm25 import BM25Okapi

from medlit.logging import logger
from medlit.models import Chunk, Source
from medlit.storage.base import PaperSummary, SearchFilter, VectorStore


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Qdrant points need int or UUID ids; deterministic UUID5 from our string."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"medlit:{chunk_id}"))


class QdrantStore(VectorStore):
    def __init__(
        self,
        *,
        url: str,
        collection: str,
        api_key: str | None = None,
    ) -> None:
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []

    # -------------------------------------------------------------- collection
    def ensure_collection(self, *, dimension: int) -> None:
        """Create the collection if missing, or validate its dimension.

        Catches the common foot-gun where the embedder configuration changes
        (e.g. switching from BGE 1024 -> OpenAI 3072) but the existing Qdrant
        collection was created at the old dim. Without this check Qdrant
        silently rejects every upsert, the per-paper try/except in the
        ingestion pipeline swallows it, and ingestion 'completes' with
        processed=0.
        """
        try:
            info = self.client.get_collection(self.collection)
        except Exception:
            info = None  # collection does not exist

        if info is not None:
            existing_dim = self._existing_dim(info)
            if existing_dim == dimension:
                return
            count = int(info.points_count or 0)
            if count == 0:
                logger.warning(
                    f"qdrant collection {self.collection!r} dim mismatch "
                    f"(existing {existing_dim} != embedder {dimension}); "
                    f"empty collection — recreating"
                )
                self.client.delete_collection(self.collection)
            else:
                raise RuntimeError(
                    f"Qdrant collection {self.collection!r} has dimension "
                    f"{existing_dim} but the embedder produces {dimension}-dim "
                    f"vectors. {count} existing vectors would be invalidated. "
                    f"To proceed, wipe the collection manually:\n"
                    f"  curl -X DELETE {self.client._client.host}:"
                    f"{self.client._client.port}/collections/{self.collection}"
                )

        logger.info(f"creating qdrant collection {self.collection!r} dim={dimension}")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
        )
        # Index payload fields used in filters.
        for field, schema in (
            ("publication_date", qm.PayloadSchemaType.DATETIME),
            ("journal", qm.PayloadSchemaType.KEYWORD),
            ("authors", qm.PayloadSchemaType.KEYWORD),
            ("mesh_terms", qm.PayloadSchemaType.KEYWORD),
            ("publication_types", qm.PayloadSchemaType.KEYWORD),
            ("source", qm.PayloadSchemaType.KEYWORD),
            ("paper_id", qm.PayloadSchemaType.KEYWORD),
            ("doi", qm.PayloadSchemaType.KEYWORD),
            ("pmid", qm.PayloadSchemaType.KEYWORD),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"payload index {field}: {e}")

    @staticmethod
    def _existing_dim(info: object) -> int | None:
        """Pull the vector size out of qdrant's CollectionInfo, tolerating
        both single-vector and named-vector configs."""
        try:
            vectors = info.config.params.vectors  # type: ignore[attr-defined]
            if hasattr(vectors, "size"):
                return int(vectors.size)
            if isinstance(vectors, dict):
                # Named vectors: take the first one's size for our single-vector use.
                first = next(iter(vectors.values()))
                return int(first.size)
        except Exception:  # noqa: BLE001
            return None
        return None

    # ---------------------------------------------------------------- upsert
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors / chunks length mismatch")
        payloads = self._chunks_payload(chunks)
        # Stash the original chunk id in the payload so we can recover it.
        for c, p in zip(chunks, payloads, strict=True):
            p["chunk_id"] = c.id
        points = [
            qm.PointStruct(
                id=_chunk_id_to_uuid(c.id),
                vector=v.tolist(),
                payload=p,
            )
            for c, v, p in zip(chunks, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        self._bm25 = None  # invalidate sparse index

    # ---------------------------------------------------------------- searches
    def dense_search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        filter_: SearchFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        qfilter = self._build_filter(filter_)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [(_payload_to_chunk(h.payload), float(h.score)) for h in hits]

    def sparse_search(
        self,
        query_text: str,
        *,
        top_k: int,
        filter_: SearchFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        # Build / refresh local BM25 index over (optionally pre-filtered) chunks.
        chunks = self._all_chunks(filter_=filter_)
        if not chunks:
            return []
        if self._bm25 is None or self._bm25_chunks is not chunks:
            tokenized = [_tokenize(c.text) for c in chunks]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_chunks = chunks
        scores = self._bm25.get_scores(_tokenize(query_text))
        order = np.argsort(scores)[::-1][:top_k]
        return [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    # -------------------------------------------------------------- management
    def delete_paper(self, paper_id: str) -> int:
        sel = qm.FilterSelector(
            filter=qm.Filter(must=[qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id))])
        )
        before = self._count()
        self.client.delete(collection_name=self.collection, points_selector=sel, wait=True)
        after = self._count()
        self._bm25 = None
        return max(0, before - after)

    def get_paper_chunks(self, paper_id: str) -> list[Chunk]:
        flt = qm.Filter(
            must=[qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id))]
        )
        out: list[Chunk] = []
        offset: int | str | None = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=256,
                with_payload=True,
                offset=offset,
            )
            out.extend(_payload_to_chunk(p.payload) for p in points)
            if offset is None:
                break
        out.sort(key=lambda c: c.chunk_index)
        return out

    def list_papers(self) -> list[PaperSummary]:
        by_paper: dict[str, dict] = {}
        counts: dict[str, int] = {}
        offset: int | str | None = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=512,
                with_payload=True,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                pid = payload.get("paper_id")
                if not pid:
                    continue
                counts[pid] = counts.get(pid, 0) + 1
                by_paper.setdefault(pid, payload)
            if offset is None:
                break
        out: list[PaperSummary] = []
        for pid, payload in by_paper.items():
            pub_date_str = payload.get("publication_date")
            try:
                pub_date = date.fromisoformat(pub_date_str) if pub_date_str else None
            except (TypeError, ValueError):
                pub_date = None
            out.append(
                PaperSummary(
                    paper_id=pid,
                    title=payload.get("title"),
                    authors=payload.get("authors") or [],
                    journal=payload.get("journal"),
                    publication_date=pub_date,
                    source=Source(payload.get("source") or Source.MANUAL.value),
                    doi=payload.get("doi"),
                    pmid=payload.get("pmid"),
                    arxiv_id=payload.get("arxiv_id"),
                    url=payload.get("url"),
                    citation_token=payload.get("citation_token"),
                    chunk_count=counts[pid],
                )
            )
        out.sort(key=lambda p: (p.publication_date is None, p.publication_date), reverse=True)
        return out

    def stats(self) -> dict[str, object]:
        info = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "vectors_count": info.points_count,
            "status": info.status.value if info.status else "unknown",
        }

    # ---------------------------------------------------------------- helpers
    def _count(self) -> int:
        return int(self.client.get_collection(self.collection).points_count or 0)

    def _all_chunks(self, *, filter_: SearchFilter | None) -> list[Chunk]:
        flt = self._build_filter(filter_)
        out: list[Chunk] = []
        offset: int | str | None = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=512,
                with_payload=True,
                offset=offset,
            )
            out.extend(_payload_to_chunk(p.payload) for p in points)
            if offset is None or len(out) > 50_000:
                # Cap to keep BM25 memory reasonable; users with large corpora
                # should use Qdrant's native sparse vectors instead.
                break
        return out

    def _build_filter(self, filter_: SearchFilter | None) -> qm.Filter | None:
        if filter_ is None:
            return None
        must: list[qm.FieldCondition | qm.Filter] = []
        if filter_.date_from or filter_.date_to:
            range_cond = qm.DatetimeRange(
                gte=_to_dt(filter_.date_from) if filter_.date_from else None,
                lte=_to_dt(filter_.date_to) if filter_.date_to else None,
            )
            must.append(qm.FieldCondition(key="publication_date", range=range_cond))
        if filter_.journals:
            must.append(qm.FieldCondition(key="journal", match=qm.MatchAny(any=filter_.journals)))
        if filter_.authors:
            must.append(qm.FieldCondition(key="authors", match=qm.MatchAny(any=filter_.authors)))
        if filter_.mesh_terms:
            must.append(qm.FieldCondition(key="mesh_terms", match=qm.MatchAny(any=filter_.mesh_terms)))
        if filter_.publication_types:
            must.append(
                qm.FieldCondition(
                    key="publication_types", match=qm.MatchAny(any=filter_.publication_types)
                )
            )
        if filter_.sources:
            must.append(qm.FieldCondition(key="source", match=qm.MatchAny(any=filter_.sources)))
        if filter_.paper_ids:
            must.append(qm.FieldCondition(key="paper_id", match=qm.MatchAny(any=filter_.paper_ids)))
        return qm.Filter(must=must) if must else None


def _to_dt(d: date) -> str:
    return f"{d.isoformat()}T00:00:00Z"


def _tokenize(text: str) -> list[str]:
    import re
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]


def _payload_to_chunk(payload: dict | None) -> Chunk:
    payload = payload or {}
    pub_date_str = payload.get("publication_date")
    pub_date = date.fromisoformat(pub_date_str) if pub_date_str else None
    return Chunk(
        id=payload.get("chunk_id") or "",
        paper_id=payload.get("paper_id", ""),
        text=payload.get("text", ""),
        chunk_index=payload.get("chunk_index", 0),
        section=payload.get("section"),
        page_number=payload.get("page_number"),
        token_count=payload.get("token_count"),
        title=payload.get("title"),
        authors=payload.get("authors") or [],
        journal=payload.get("journal"),
        publication_date=pub_date,
        doi=payload.get("doi"),
        pmid=payload.get("pmid"),
        arxiv_id=payload.get("arxiv_id"),
        mesh_terms=payload.get("mesh_terms") or [],
        publication_types=payload.get("publication_types") or [],
        source=Source(payload.get("source") or Source.MANUAL.value),
        url=payload.get("url"),
        citation_token=payload.get("citation_token"),
    )
