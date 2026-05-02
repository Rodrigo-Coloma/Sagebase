"""Loguru setup. Idempotent: safe to call multiple times."""

from __future__ import annotations

import sys

from loguru import logger

from medlit.config import get_settings

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.logging.level,
        format=settings.logging.format,
        backtrace=False,
        diagnose=False,
    )
    _CONFIGURED = True


__all__ = ["logger", "setup_logging"]
