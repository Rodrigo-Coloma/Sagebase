"""Generator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from medlit.models import RetrievalResult


@dataclass(slots=True)
class GenerationRequest:
    question: str
    context: Sequence[RetrievalResult]
    max_tokens: int = 1024
    temperature: float = 0.2


class BaseGenerator(ABC):
    model: str

    @abstractmethod
    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Yield answer tokens as they arrive."""
        ...

    async def generate(self, request: GenerationRequest) -> str:
        chunks: list[str] = []
        async for piece in self.stream(request):
            chunks.append(piece)
        return "".join(chunks)
