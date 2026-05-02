"""In-memory job tracker for async ingestion endpoints.

For production, swap this for Postgres or Redis (the docker-compose file ships
with a Postgres profile for that purpose).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from medlit.models import IngestionJob, Source


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}
        self._lock = asyncio.Lock()

    async def create(self, *, source: Source, query: str | None) -> IngestionJob:
        async with self._lock:
            job = IngestionJob(id=str(uuid.uuid4()), source=source, query=query)
            self._jobs[job.id] = job
            return job

    async def update(self, job_id: str, **fields: object) -> IngestionJob:
        async with self._lock:
            job = self._jobs[job_id]
            for k, v in fields.items():
                setattr(job, k, v)
            job.updated_at = datetime.utcnow()
            return job

    def get(self, job_id: str) -> IngestionJob | None:
        return self._jobs.get(job_id)

    def list(self) -> list[IngestionJob]:
        return list(self._jobs.values())


registry = JobRegistry()
