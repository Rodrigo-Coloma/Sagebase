"""Ingestion endpoints (async — return a job id, poll status separately).

All background runners are wrapped with `_safe_job` so that any exception —
including ones thrown before the job's status is updated to "running" — gets
written to the job record's `error` field and the status flips to "failed".
Without this wrapper, FastAPI's BackgroundTasks swallow exceptions silently
and the UI sees a job stuck in "pending" forever.
"""

from __future__ import annotations

import shutil
import traceback
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from medlit.api.deps import get_ingestion
from medlit.api.jobs import registry
from medlit.api.schemas import (
    ArxivIngestRequest,
    BiorxivIngestRequest,
    DoiIngestRequest,
    IngestJobResponse,
    PubMedIngestRequest,
)
from medlit.config import get_settings
from medlit.ingestion.arxiv import ArxivIngester
from medlit.ingestion.base import IngestionQuery
from medlit.ingestion.biorxiv import BiorxivIngester
from medlit.ingestion.crossref import CrossrefIngester
from medlit.ingestion.pubmed import PubMedIngester
from medlit.logging import logger
from medlit.models import IngestionJob, Source

router = APIRouter(prefix="/ingest", tags=["ingest"])


# ----------------------------------------------------------------- error capture
async def _safe_job(
    job_id: str,
    runner: Callable[..., Awaitable[None]],
    *args: object,
) -> None:
    """Run a job runner and capture any exception into the job record."""
    try:
        await runner(*args)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception(f"job {job_id} failed")
        try:
            await registry.update(job_id, status="failed", error=tb[-2000:])
        except Exception:  # noqa: BLE001
            logger.exception(f"could not record failure for job {job_id}")


# ------------------------------------------------------------------ runners
async def _run_pubmed_job(job_id: str, req: PubMedIngestRequest) -> None:
    pipeline = get_ingestion()
    ingester = PubMedIngester()
    iq = IngestionQuery(
        query=req.query,
        max_results=req.max_results,
        date_from=req.date_from,
        date_to=req.date_to,
    )
    await registry.update(job_id, status="running")
    processed = 0
    async for _ in pipeline.ingest_from(ingester, iq, with_full_text=req.with_full_text):
        processed += 1
        await registry.update(job_id, processed=processed)
    await registry.update(job_id, status="completed", processed=processed, total=processed)


async def _run_arxiv_job(job_id: str, req: ArxivIngestRequest) -> None:
    pipeline = get_ingestion()
    ingester = ArxivIngester()
    iq = IngestionQuery(query=req.query, max_results=req.max_results)
    await registry.update(job_id, status="running")
    processed = 0
    async for _ in pipeline.ingest_from(ingester, iq):
        processed += 1
        await registry.update(job_id, processed=processed)
    await registry.update(job_id, status="completed", processed=processed, total=processed)


async def _run_biorxiv_job(job_id: str, req: BiorxivIngestRequest) -> None:
    pipeline = get_ingestion()
    ingester = BiorxivIngester(server=req.server)
    iq = IngestionQuery(
        query=req.query,
        max_results=req.max_results,
        date_from=req.date_from,
        date_to=req.date_to,
    )
    await registry.update(job_id, status="running")
    processed = 0
    async for _ in pipeline.ingest_from(ingester, iq):
        processed += 1
        await registry.update(job_id, processed=processed)
    await registry.update(job_id, status="completed", processed=processed, total=processed)


async def _run_doi_job(job_id: str, req: DoiIngestRequest) -> None:
    pipeline = get_ingestion()
    cr = CrossrefIngester(mailto=get_settings().unpaywall_email)
    await registry.update(job_id, status="running")
    paper = await cr.fetch_one(req.doi)
    if paper is None:
        await registry.update(job_id, status="failed", error="DOI not found in Crossref")
        return
    ok = await pipeline._process_paper(paper, ingester=cr, with_full_text=True)  # noqa: SLF001
    await registry.update(
        job_id,
        status="completed",
        processed=1 if ok else 0,
        total=1,
        failed=0 if ok else 1,
    )


async def _run_upload_job(job_id: str, paths: list[Path]) -> None:
    pipeline = get_ingestion()
    await registry.update(job_id, status="running")
    processed = 0
    async for _ in pipeline.ingest_files(paths):
        processed += 1
        await registry.update(job_id, processed=processed)
    await registry.update(job_id, status="completed", processed=processed, total=processed)


# ------------------------------------------------------------------ routes
@router.post("/pubmed", response_model=IngestJobResponse)
async def ingest_pubmed(req: PubMedIngestRequest, bg: BackgroundTasks) -> IngestJobResponse:
    job = await registry.create(source=Source.PUBMED, query=req.query)
    bg.add_task(_safe_job, job.id, _run_pubmed_job, job.id, req)
    return IngestJobResponse(job=job)


@router.post("/arxiv", response_model=IngestJobResponse)
async def ingest_arxiv(req: ArxivIngestRequest, bg: BackgroundTasks) -> IngestJobResponse:
    job = await registry.create(source=Source.ARXIV, query=req.query)
    bg.add_task(_safe_job, job.id, _run_arxiv_job, job.id, req)
    return IngestJobResponse(job=job)


@router.post("/biorxiv", response_model=IngestJobResponse)
async def ingest_biorxiv(req: BiorxivIngestRequest, bg: BackgroundTasks) -> IngestJobResponse:
    job = await registry.create(
        source=Source.BIORXIV if req.server == "biorxiv" else Source.MEDRXIV,
        query=req.query,
    )
    bg.add_task(_safe_job, job.id, _run_biorxiv_job, job.id, req)
    return IngestJobResponse(job=job)


@router.post("/doi", response_model=IngestJobResponse)
async def ingest_doi(req: DoiIngestRequest, bg: BackgroundTasks) -> IngestJobResponse:
    job = await registry.create(source=Source.CROSSREF, query=req.doi)
    bg.add_task(_safe_job, job.id, _run_doi_job, job.id, req)
    return IngestJobResponse(job=job)


@router.post("/upload", response_model=IngestJobResponse)
async def ingest_upload(
    bg: BackgroundTasks,
    files: list[UploadFile] = File(...),  # noqa: B008
) -> IngestJobResponse:
    settings = get_settings()
    upload_dir = Path(settings.medlit_data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for f in files:
        if not f.filename:
            continue
        dest = upload_dir / f.filename
        with dest.open("wb") as fh:
            shutil.copyfileobj(f.file, fh)
        saved_paths.append(dest)
    if not saved_paths:
        raise HTTPException(status_code=400, detail="No files received.")
    job = await registry.create(source=Source.MANUAL, query=",".join(p.name for p in saved_paths))
    bg.add_task(_safe_job, job.id, _run_upload_job, job.id, saved_paths)
    return IngestJobResponse(job=job)


@router.get("/jobs/{job_id}", response_model=IngestionJob)
async def get_job(job_id: str) -> IngestionJob:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.get("/jobs", response_model=list[IngestionJob])
async def list_jobs() -> list[IngestionJob]:
    return registry.list()
