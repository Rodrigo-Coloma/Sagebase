"""ChromaDB-backed VectorStore (lightweight local alternative)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
from rank_bm25 import BM25Okapi

from medlit.logging import logger
from medlit.models import Chunk, Source
from medlit.storage.base import SearchFilter, VectorStore


class ChromaStore(VectorStore):
    def __init__(self, *, persist_dir: str, collection: str) -> None:
        try:
            import chromadb
        except ImportError as e:  # pragma: no cover - optional
            raise RuntimeError("Install medlit[chroma] to use ChromaStore.") from e
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection
        self._collection = None
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []

    def ensure_collection(self, *, dimension: int) -> None:
        # Chroma manages dimension automatically once the first vector is added.
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"chroma collection {self.collection_name!r} ready (dim={dimension})")

    @property
    def col(self):  # type: ignore[no-untyped-def]
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = self._chunks_payload(chunks)
        self.col.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=documents,
            metadatas=metadatas,
        )
        self._bm25 = None

    def dense_search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        filter_: SearchFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        where = self._build_where(filter_)
        res = self.col.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where or None,
        )
        out: list[tuple[Chunk, float]] = []
        for i, _id in enumerate(res["ids"][0]):
            md = res["metadatas"][0][i]
            doc = res["documents"][0][i]
            distance = res["distances"][0][i]
            out.append((_md_to_chunk(_id, doc, md), float(1 - distance)))
        return out

    def sparse_search(
        self,
        query_text: str,
        *,
        top_k: int,
        filter_: SearchFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        chunks = self._all_chunks(filter_=filter_)
        if not chunks:
            return []
        if self._bm25 is None or self._bm25_chunks is not chunks:
            self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
            self._bm25_chunks = chunks
        scores = self._bm25.get_scores(_tokenize(query_text))
        order = np.argsort(scores)[::-1][:top_k]
        return [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def delete_paper(self, paper_id: str) -> int:
        before = self.col.count()
        self.col.delete(where={"paper_id": paper_id})
        after = self.col.count()
        self._bm25 = None
        return max(0, before - after)

    def get_paper_chunks(self, paper_id: str) -> list[Chunk]:
        res = self.col.get(where={"paper_id": paper_id})
        out = [_md_to_chunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)]
        out.sort(key=lambda c: c.chunk_index)
        return out

    def stats(self) -> dict[str, object]:
        return {"collection": self.collection_name, "vectors_count": self.col.count()}

    # ---------------------------------------------------------------- helpers
    def _all_chunks(self, *, filter_: SearchFilter | None) -> list[Chunk]:
        where = self._build_where(filter_)
        res = self.col.get(where=where or None)
        return [
            _md_to_chunk(i, d, m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

    @staticmethod
    def _build_where(filter_: SearchFilter | None) -> dict:
        if filter_ is None:
            return {}
        clauses: list[dict] = []
        if filter_.journals:
            clauses.append({"journal": {"$in": filter_.journals}})
        if filter_.sources:
            clauses.append({"source": {"$in": filter_.sources}})
        if filter_.paper_ids:
            clauses.append({"paper_id": {"$in": filter_.paper_ids}})
        if filter_.date_from:
            clauses.append({"publication_date": {"$gte": filter_.date_from.isoformat()}})
        if filter_.date_to:
            clauses.append({"publication_date": {"$lte": filter_.date_to.isoformat()}})
        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}


def _tokenize(text: str) -> list[str]:
    import re
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]


def _md_to_chunk(_id: str, text: str, md: dict) -> Chunk:
    pub_date_str = md.get("publication_date")
    pub_date = date.fromisoformat(pub_date_str) if pub_date_str else None
    return Chunk(
        id=_id,
        paper_id=md.get("paper_id", ""),
        text=text,
        chunk_index=md.get("chunk_index", 0),
        section=md.get("section"),
        page_number=md.get("page_number"),
        token_count=md.get("token_count"),
        title=md.get("title"),
        authors=md.get("authors") or [],
        journal=md.get("journal"),
        publication_date=pub_date,
        doi=md.get("doi"),
        pmid=md.get("pmid"),
        arxiv_id=md.get("arxiv_id"),
        mesh_terms=md.get("mesh_terms") or [],
        publication_types=md.get("publication_types") or [],
        source=Source(md.get("source") or Source.MANUAL.value),
        url=md.get("url"),
        citation_token=md.get("citation_token"),
    )
