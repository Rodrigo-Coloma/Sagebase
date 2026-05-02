"""medlit CLI."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from medlit.config import get_settings
from medlit.embeddings.factory import build_embedder
from medlit.generation.base import GenerationRequest
from medlit.generation.factory import build_generator
from medlit.ingestion.arxiv import ArxivIngester
from medlit.ingestion.base import IngestionQuery
from medlit.ingestion.biorxiv import BiorxivIngester
from medlit.ingestion.crossref import CrossrefIngester
from medlit.ingestion.pubmed import PubMedIngester
from medlit.logging import logger, setup_logging
from medlit.pipeline import IngestionPipeline
from medlit.retrieval.pipeline import RetrievalPipeline
from medlit.retrieval.reranker import CrossEncoderReranker
from medlit.storage.base import SearchFilter
from medlit.storage.factory import build_vector_store

app = typer.Typer(
    name="medlit",
    help="Vectorized medical literature RAG.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
ingest_app = typer.Typer(no_args_is_help=True, help="Ingest papers from various sources.")
app.add_typer(ingest_app, name="ingest")
console = Console()


def _build_pipeline() -> IngestionPipeline:
    setup_logging()
    return IngestionPipeline()


def _parse_date(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


# ---------------------------------------------------------------------- ingest
@ingest_app.command("pubmed")
def ingest_pubmed(
    query: Annotated[str, typer.Option("--query", "-q", help="Free-text or boolean PubMed query.")],
    max_results: Annotated[int, typer.Option("--max-results", help="Cap.")] = 100,
    date_from: Annotated[str | None, typer.Option("--date-from", help="YYYY-MM-DD.")] = None,
    date_to: Annotated[str | None, typer.Option("--date-to", help="YYYY-MM-DD.")] = None,
    no_full_text: Annotated[bool, typer.Option("--no-full-text", help="Abstract-only.")] = False,
) -> None:
    pipeline = _build_pipeline()
    ingester = PubMedIngester()
    iq = IngestionQuery(
        query=query,
        max_results=max_results,
        date_from=_parse_date(date_from),
        date_to=_parse_date(date_to),
    )

    async def _run() -> None:
        count = 0
        async for _ in pipeline.ingest_from(ingester, iq, with_full_text=not no_full_text):
            count += 1
            console.print(f"[green]✓[/] indexed {count}/{max_results}")
        console.print(f"[bold green]done.[/] {count} papers indexed.")

    asyncio.run(_run())


@ingest_app.command("arxiv")
def ingest_arxiv(
    query: Annotated[str, typer.Option("--query", "-q")],
    max_results: Annotated[int, typer.Option("--max-results")] = 50,
) -> None:
    pipeline = _build_pipeline()
    ingester = ArxivIngester()
    iq = IngestionQuery(query=query, max_results=max_results)

    async def _run() -> None:
        async for paper in pipeline.ingest_from(ingester, iq):
            console.print(f"[green]✓[/] {paper.title[:100]}")

    asyncio.run(_run())


@ingest_app.command("biorxiv")
def ingest_biorxiv(
    query: Annotated[str, typer.Option("--query", "-q")] = "",
    max_results: Annotated[int, typer.Option("--max-results")] = 50,
    server: Annotated[str, typer.Option("--server", help="biorxiv|medrxiv")] = "biorxiv",
    date_from: Annotated[str | None, typer.Option("--date-from")] = None,
    date_to: Annotated[str | None, typer.Option("--date-to")] = None,
) -> None:
    pipeline = _build_pipeline()
    ingester = BiorxivIngester(server=server)
    iq = IngestionQuery(
        query=query or None,
        max_results=max_results,
        date_from=_parse_date(date_from),
        date_to=_parse_date(date_to),
    )

    async def _run() -> None:
        async for paper in pipeline.ingest_from(ingester, iq):
            console.print(f"[green]✓[/] {paper.title[:100]}")

    asyncio.run(_run())


@ingest_app.command("file")
def ingest_file(
    paths: Annotated[list[Path], typer.Argument(help="Files or directories.")],
) -> None:
    pipeline = _build_pipeline()

    async def _run() -> None:
        async for paper in pipeline.ingest_files(paths):
            console.print(f"[green]✓[/] {paper.title[:100]}")

    asyncio.run(_run())


@ingest_app.command("doi")
def ingest_doi(doi: Annotated[str, typer.Argument(help="DOI like 10.1038/s41586-023-xxxx")]) -> None:
    pipeline = _build_pipeline()

    async def _run() -> None:
        cr = CrossrefIngester(mailto=get_settings().unpaywall_email)
        paper = await cr.fetch_one(doi)
        if paper is None:
            console.print(f"[red]not found:[/] {doi}")
            raise typer.Exit(1)
        # Try to enrich with PubMed if we can find a PMID via NCBI's idconv.
        # Skipped for simplicity; pipeline will Unpaywall the DOI for full text.
        ok = await pipeline._process_paper(  # noqa: SLF001
            paper, ingester=cr, with_full_text=True
        )
        if ok:
            console.print(f"[green]✓[/] indexed {paper.title[:100]}")
        else:
            console.print(f"[yellow]skipped[/] {paper.title[:100]}")

    asyncio.run(_run())


# ---------------------------------------------------------------------- search
@app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Natural language query.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 10,
    date_from: Annotated[str | None, typer.Option("--date-from")] = None,
    date_to: Annotated[str | None, typer.Option("--date-to")] = None,
    journal: Annotated[list[str] | None, typer.Option("--journal")] = None,
    mesh: Annotated[list[str] | None, typer.Option("--mesh")] = None,
    no_rerank: Annotated[bool, typer.Option("--no-rerank")] = False,
) -> None:
    setup_logging()
    settings = get_settings()
    embedder = build_embedder()
    store = build_vector_store()
    store.ensure_collection(dimension=embedder.dimension)
    reranker = None if no_rerank or not settings.reranker.enabled else CrossEncoderReranker(settings.reranker.model)
    pipe = RetrievalPipeline(embedder=embedder, store=store, reranker=reranker)
    flt = SearchFilter(
        date_from=_parse_date(date_from),
        date_to=_parse_date(date_to),
        journals=journal,
        mesh_terms=mesh,
    )
    results = pipe.search(query, filter_=flt, top_k=top_k)

    table = Table(title=f"Top {len(results)} for: {query}")
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Cite")
    table.add_column("Section")
    table.add_column("Title")
    for i, r in enumerate(results, start=1):
        table.add_row(
            str(i),
            f"{r.score:.3f}",
            r.chunk.citation_token or "",
            (r.chunk.section or "")[:30],
            (r.chunk.title or "")[:80],
        )
    console.print(table)
    for i, r in enumerate(results, start=1):
        console.print(f"\n[bold]#{i}[/] [{r.chunk.citation_token}] {r.chunk.title}")
        console.print(r.chunk.text[:600] + ("..." if len(r.chunk.text) > 600 else ""))


# ---------------------------------------------------------------------- ask
@app.command("ask")
def ask(
    question: Annotated[str, typer.Argument()],
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 8,
    date_from: Annotated[str | None, typer.Option("--date-from")] = None,
    date_to: Annotated[str | None, typer.Option("--date-to")] = None,
    no_rerank: Annotated[bool, typer.Option("--no-rerank")] = False,
) -> None:
    setup_logging()
    settings = get_settings()
    embedder = build_embedder()
    store = build_vector_store()
    store.ensure_collection(dimension=embedder.dimension)
    reranker = None if no_rerank or not settings.reranker.enabled else CrossEncoderReranker(settings.reranker.model)
    retr = RetrievalPipeline(embedder=embedder, store=store, reranker=reranker)
    flt = SearchFilter(date_from=_parse_date(date_from), date_to=_parse_date(date_to))
    results = retr.search(question, filter_=flt, top_k=top_k)
    if not results:
        console.print("[yellow]No relevant context found.[/]")
        raise typer.Exit(1)

    gen = build_generator()
    req = GenerationRequest(
        question=question,
        context=results,
        max_tokens=settings.generation.max_tokens,
        temperature=settings.generation.temperature,
    )

    async def _run() -> None:
        console.print(f"[bold]Question:[/] {question}\n")
        console.print("[bold]Sources:[/]")
        for i, r in enumerate(results, start=1):
            console.print(f"  {i}. [{r.chunk.citation_token}] {r.chunk.title}")
        console.print("\n[bold]Answer:[/]\n")
        async for piece in gen.stream(req):
            console.print(piece, end="", soft_wrap=True)
        console.print("")

    asyncio.run(_run())


# ---------------------------------------------------------------------- stats
@app.command("stats")
def stats() -> None:
    setup_logging()
    store = build_vector_store()
    embedder = build_embedder()
    store.ensure_collection(dimension=embedder.dimension)
    s = store.stats()
    table = Table(title="medlit stats")
    table.add_column("Key")
    table.add_column("Value")
    for k, v in s.items():
        table.add_row(str(k), str(v))
    console.print(table)


# ---------------------------------------------------------------------- serve
@app.command("serve")
def serve(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Start the FastAPI server."""
    setup_logging()
    import uvicorn

    uvicorn.run("medlit.api.app:app", host=host, port=port, reload=reload)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
