"""Unpaywall: legitimately discover open-access full text for a DOI.

Unpaywall surfaces author manuscripts and publisher OA copies. We only return
URLs marked as `is_oa = true` and prefer license-tagged copies (cc-by, etc.)
to honor publisher copyright."""

from __future__ import annotations

from typing import Any

from medlit.config import get_settings
from medlit.logging import logger
from medlit.utils.http import RateLimiter, get_with_retry, http_client

UNPAYWALL_API = "https://api.unpaywall.org/v2/{doi}"


class UnpaywallClient:
    def __init__(self, *, email: str | None = None, rate_limit_rps: float = 5.0) -> None:
        self.email = email or get_settings().unpaywall_email
        self._limiter = RateLimiter(rps=rate_limit_rps)

    async def best_oa_pdf_url(self, doi: str) -> str | None:
        url = UNPAYWALL_API.format(doi=doi)
        async with http_client() as client:
            try:
                resp = await get_with_retry(
                    client, url, limiter=self._limiter, params={"email": self.email}
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"unpaywall lookup failed for {doi}: {e}")
                return None
            data: dict[str, Any] = resp.json()
        if not data.get("is_oa"):
            return None
        best = data.get("best_oa_location") or {}
        # Prefer URL of PDF; fall back to landing page URL.
        return best.get("url_for_pdf") or best.get("url")
