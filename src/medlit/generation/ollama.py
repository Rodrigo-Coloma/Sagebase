"""Ollama (local) generator (optional)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from medlit.generation.base import BaseGenerator, GenerationRequest
from medlit.generation.prompts import build_messages


class OllamaGenerator(BaseGenerator):
    def __init__(self, *, model: str = "llama3.1", host: str = "http://localhost:11434") -> None:
        try:
            from ollama import AsyncClient
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError("Install medlit[ollama] to use OllamaGenerator.") from e
        self.model = model
        self._client = AsyncClient(host=host)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        messages = build_messages(request.question, request.context)
        stream = await self._client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={"temperature": request.temperature, "num_predict": request.max_tokens},
        )
        async for chunk in stream:
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
