"""OpenAI chat completions generator (optional)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from medlit.generation.base import BaseGenerator, GenerationRequest
from medlit.generation.prompts import build_messages


class OpenAIGenerator(BaseGenerator):
    def __init__(self, *, model: str = "gpt-4o", api_key: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError("Install medlit[openai] to use OpenAIGenerator.") from e
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        messages = build_messages(request.question, request.context)
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
        )
        async for event in stream:
            choice = event.choices[0] if event.choices else None
            if not choice:
                continue
            delta = choice.delta.content or ""
            if delta:
                yield delta
