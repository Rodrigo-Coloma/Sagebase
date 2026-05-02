"""Anthropic Claude generator. Default backend."""

from __future__ import annotations

from collections.abc import AsyncIterator

from medlit.generation.base import BaseGenerator, GenerationRequest
from medlit.generation.prompts import SYSTEM_PROMPT, build_messages


class AnthropicGenerator(BaseGenerator):
    def __init__(self, *, model: str = "claude-sonnet-4-5", api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self._client = AsyncAnthropic(api_key=api_key)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        msgs = build_messages(request.question, request.context)
        # Anthropic takes `system` as a separate argument; build_messages emits a
        # system role we strip here.
        user_msgs = [m for m in msgs if m["role"] != "system"]
        async with self._client.messages.stream(
            model=self.model,
            system=SYSTEM_PROMPT,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=[{"role": m["role"], "content": m["content"]} for m in user_msgs],
        ) as stream:
            async for text in stream.text_stream:
                yield text
