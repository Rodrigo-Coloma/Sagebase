"""APScheduler-based watch runner.

Each watch is a saved query (PubMed/arXiv/bioRxiv) that re-ingests on a
schedule. Deduplication happens automatically because the upsert is keyed
by chunk id (paper_id:chunk_index)."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from medlit.ingestion.arxiv import ArxivIngester
from medlit.ingestion.base import IngestionQuery
from medlit.ingestion.biorxiv import BiorxivIngester
from medlit.ingestion.pubmed import PubMedIngester
from medlit.logging import logger
from medlit.pipeline import IngestionPipeline

_INTERVAL_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$")


@dataclass(slots=True)
class Watch:
    name: str
    source: str
    query: str
    max_results: int
    schedule: dict[str, Any]
    filters: dict[str, Any]


def load_watches(path: str | Path) -> list[Watch]:
    p = Path(path)
    if not p.exists():
        logger.warning(f"watches file not found: {p}")
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [
        Watch(
            name=w["name"],
            source=w["source"],
            query=w.get("query", ""),
            max_results=int(w.get("max_results", 100)),
            schedule=dict(w.get("schedule") or {}),
            filters=dict(w.get("filters") or {}),
        )
        for w in (data.get("watches") or [])
    ]


def _trigger_from(schedule: dict[str, Any]) -> CronTrigger | IntervalTrigger:
    if "cron" in schedule:
        return CronTrigger.from_crontab(str(schedule["cron"]))
    if "every" in schedule:
        m = _INTERVAL_RE.match(str(schedule["every"]))
        if not m:
            raise ValueError(f"invalid 'every' value: {schedule['every']!r}")
        n, unit = int(m.group(1)), m.group(2)
        kwargs = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]
        return IntervalTrigger(**{kwargs: n})
    raise ValueError("schedule must contain 'cron' or 'every'")


class WatchScheduler:
    def __init__(self, *, pipeline: IngestionPipeline | None = None) -> None:
        self.pipeline = pipeline or IngestionPipeline()
        self.scheduler = AsyncIOScheduler()

    def add(self, watch: Watch) -> None:
        trigger = _trigger_from(watch.schedule)
        self.scheduler.add_job(
            self._run_watch,
            trigger=trigger,
            args=[watch],
            id=watch.name,
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info(f"scheduled watch {watch.name!r} ({watch.source})")

    def start(self) -> None:
        self.scheduler.start()

    def stop(self) -> None:
        self.scheduler.shutdown(wait=False)

    async def _run_watch(self, watch: Watch) -> None:
        logger.info(f"running watch: {watch.name}")
        date_from: date | None = None
        if "date_window_days" in watch.filters:
            date_from = date.today() - timedelta(days=int(watch.filters["date_window_days"]))
        iq = IngestionQuery(
            query=watch.query, max_results=watch.max_results, date_from=date_from
        )
        try:
            if watch.source == "pubmed":
                ingester = PubMedIngester()
            elif watch.source == "arxiv":
                ingester = ArxivIngester()  # type: ignore[assignment]
            elif watch.source in {"biorxiv", "medrxiv"}:
                ingester = BiorxivIngester(server=watch.source)  # type: ignore[assignment]
            else:
                logger.warning(f"unsupported watch source: {watch.source}")
                return
            count = 0
            async for _ in self.pipeline.ingest_from(ingester, iq):
                count += 1
            logger.info(f"watch {watch.name!r} indexed {count} new papers")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"watch {watch.name} failed: {e}")


def main(config_path: str = "config/watches.yaml") -> None:  # pragma: no cover
    """Run all configured watches forever."""
    from medlit.logging import setup_logging

    setup_logging()
    sched = WatchScheduler()
    for w in load_watches(config_path):
        sched.add(w)
    sched.start()
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        sched.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
