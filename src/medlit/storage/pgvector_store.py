"""pgvector-backed VectorStore (optional).

Schema is created on `ensure_collection`. Sparse search uses Postgres
`websearch_to_tsquery` over a generated `tsvector` column for portability —
not as good as BM25 at top-k retrieval, but adequate as an alternative.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np

from medlit.logging import logger
from medlit.models import Chunk, Source
from medlit.storage.base import PaperSummary, SearchFilter, VectorStore


class PgVectorStore(VectorStore):
    def __init__(self, *, dsn: str, collection: str) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError("Install medlit[pgvector] to use PgVectorStore.") from e
        self._psycopg = psycopg
        self._register_vector = register_vector
        self.dsn = dsn
        self.table = collection.replace("-", "_")
        self.conn = psycopg.connect(dsn, autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def ensure_collection(self, *, dimension: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({dimension}),
                    metadata JSONB NOT NULL,
                    publication_date DATE,
                    journal TEXT,
                    source TEXT,
                    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_emb_ix "
                f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_tsv_ix "
                f"ON {self.table} USING GIN (tsv)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_paper_ix "
                f"ON {self.table} (paper_id)"
            )
        logger.info(f"pgvector table {self.table!r} ready (dim={dimension})")

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        payloads = self._chunks_payload(chunks)
        rows = []
        for c, v, p in zip(chunks, vectors, payloads, strict=True):
            rows.append(
                (
                    c.id,
                    c.paper_id,
                    c.text,
                    v.tolist(),
                    self._psycopg.types.json.Jsonb(p),
                    c.publication_date,
                    c.journal,
                    c.source.value,
                )
            )
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table}
                  (id, paper_id, text, embedding, metadata, publication_date, journal, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                  text = EXCLUDED.text,
                  embedding = EXCLUDED.embedding,
                  metadata = EXCLUDED.metadata,
                  publication_date = EXCLUDED.publication_date,
                  journal = EXCLUDED.journal,
                  source = EXCLUDED.source
                """,
                rows,
            )

    def dense_search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        filter_: SearchFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        where, params = self._build_where(filter_)
        sql = (
            f"SELECT id, text, metadata, 1 - (embedding <=> %s) AS score "
            f"FROM {self.table} {where} ORDER BY embedding <=> %s LIMIT %s"
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, (query_vector.tolist(), *params, query_vector.tolist(), top_k))
            rows = cur.fetchall()
        return [(_row_to_chunk(r[0], r[1], r[2]), float(r[3])) for r in rows]

    def sparse_search(
        self,
        query_text: str,
        *,
        top_k: int,
        filter_: SearchFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        where, params = self._build_where(filter_)
        sql = (
            f"SELECT id, text, metadata, "
            f"ts_rank(tsv, websearch_to_tsquery('english', %s)) AS score "
            f"FROM {self.table} "
            f"{where} {'AND' if where else 'WHERE'} "
            f"tsv @@ websearch_to_tsquery('english', %s) "
            f"ORDER BY score DESC LIMIT %s"
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, (query_text, *params, query_text, top_k))
            rows = cur.fetchall()
        return [(_row_to_chunk(r[0], r[1], r[2]), float(r[3])) for r in rows]

    def delete_paper(self, paper_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE paper_id = %s", (paper_id,))
            return cur.rowcount

    def get_paper_chunks(self, paper_id: str) -> list[Chunk]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, metadata FROM {self.table} WHERE paper_id = %s "
                f"ORDER BY (metadata->>'chunk_index')::int",
                (paper_id,),
            )
            rows = cur.fetchall()
        return [_row_to_chunk(r[0], r[1], r[2]) for r in rows]

    def list_papers(self) -> list[PaperSummary]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT paper_id, COUNT(*) AS n,
                       (array_agg(metadata ORDER BY (metadata->>'chunk_index')::int))[1] AS md
                FROM {self.table}
                GROUP BY paper_id
                """
            )
            rows = cur.fetchall()
        out: list[PaperSummary] = []
        for paper_id, n, md in rows:
            md = md or {}
            pub_date_str = md.get("publication_date")
            try:
                pub_date = date.fromisoformat(pub_date_str) if pub_date_str else None
            except (TypeError, ValueError):
                pub_date = None
            out.append(
                PaperSummary(
                    paper_id=paper_id,
                    title=md.get("title"),
                    authors=md.get("authors") or [],
                    journal=md.get("journal"),
                    publication_date=pub_date,
                    source=Source(md.get("source") or Source.MANUAL.value),
                    doi=md.get("doi"),
                    pmid=md.get("pmid"),
                    arxiv_id=md.get("arxiv_id"),
                    url=md.get("url"),
                    citation_token=md.get("citation_token"),
                    chunk_count=int(n),
                )
            )
        out.sort(key=lambda p: (p.publication_date is None, p.publication_date), reverse=True)
        return out

    def stats(self) -> dict[str, object]:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            n = cur.fetchone()[0]
        return {"collection": self.table, "vectors_count": n}

    def close(self) -> None:
        self.conn.close()

    def _build_where(self, filter_: SearchFilter | None) -> tuple[str, list[object]]:
        if filter_ is None:
            return "", []
        parts: list[str] = []
        params: list[object] = []
        if filter_.date_from:
            parts.append("publication_date >= %s")
            params.append(filter_.date_from)
        if filter_.date_to:
            parts.append("publication_date <= %s")
            params.append(filter_.date_to)
        if filter_.journals:
            parts.append("journal = ANY(%s)")
            params.append(filter_.journals)
        if filter_.sources:
            parts.append("source = ANY(%s)")
            params.append(filter_.sources)
        if filter_.paper_ids:
            parts.append("paper_id = ANY(%s)")
            params.append(filter_.paper_ids)
        return ("WHERE " + " AND ".join(parts), params) if parts else ("", [])


def _row_to_chunk(_id: str, text: str, md: dict) -> Chunk:
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
