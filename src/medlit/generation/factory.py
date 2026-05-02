"""Generator factory."""

from __future__ import annotations

from medlit.config import GenerationConfig, get_settings
from medlit.generation.base import BaseGenerator


def build_generator(config: GenerationConfig | None = None) -> BaseGenerator:
    settings = get_settings()
    cfg = config or settings.generation
    if cfg.backend == "anthropic":
        from medlit.generation.anthropic import AnthropicGenerator

        return AnthropicGenerator(model=cfg.model, api_key=settings.anthropic_api_key)
    if cfg.backend == "openai":
        from medlit.generation.openai import OpenAIGenerator

        return OpenAIGenerator(model=cfg.model, api_key=settings.openai_api_key)
    if cfg.backend == "ollama":
        from medlit.generation.ollama import OllamaGenerator

        return OllamaGenerator(model=cfg.model)
    raise ValueError(f"unknown generation backend: {cfg.backend}")
