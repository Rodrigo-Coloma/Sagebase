"""Rate-limited async HTTP client primitives shared across ingesters."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class RateLimiter:
    """Token-bucket-ish limiter: at most `rps` requests per second."""

    def __init__(self, rps: float) -> None:
        if rps <= 0:
            raise ValueError("rps must be positive")
        self._min_interval = 1.0 / rps
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


@asynccontextmanager
async def http_client(timeout: float = 30.0, **kwargs: object) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=timeout, **kwargs) as client:  # type: ignore[arg-type]
        yield client


def retry_policy(attempts: int = 4) -> AsyncRetrying:
    """Standard retry: exponential backoff on transient HTTP errors."""
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(
            (httpx.HTTPError, httpx.TimeoutException, httpx.RemoteProtocolError)
        ),
        reraise=True,
    )


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    limiter: RateLimiter | None = None,
    **kwargs: object,
) -> httpx.Response:
    """GET with rate limiting + retry. Raises for status."""
    async for attempt in retry_policy():
        with attempt:
            if limiter is not None:
                await limiter.acquire()
            resp = await client.get(url, **kwargs)  # type: ignore[arg-type]
            resp.raise_for_status()
            return resp
    raise RuntimeError("unreachable")  # for mypy
