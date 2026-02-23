"""OpenAI-compatible chat provider for DeepSeek, GPT, and other vendors.

Uses the openai SDK which is compatible with DeepSeek's API.
"""

import time
from dataclasses import dataclass
from typing import Optional

import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger()

# Approximate pricing per 1M tokens (input/output)
MODEL_PRICING = {
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost based on known pricing."""
    pricing = MODEL_PRICING.get(model, (1.0, 3.0))
    return (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000


@dataclass
class ChatResponse:
    """Response from a chat completion."""

    content: str
    model: str
    cost: float
    input_tokens: int
    output_tokens: int
    duration_ms: int


class ChatProvider:
    """OpenAI-compatible chat provider supporting DeepSeek, GPT, etc."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """Send chat completion request."""
        used_model = model or self.model
        start = time.monotonic()

        response = await self.client.chat.completions.create(
            model=used_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        duration_ms = int((time.monotonic() - start) * 1000)
        choice = response.choices[0]
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return ChatResponse(
            content=choice.message.content or "",
            model=response.model or used_model,
            cost=_estimate_cost(used_model, input_tokens, output_tokens),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )

    async def classify(
        self,
        prompt: str,
        system: str,
        model: Optional[str] = None,
    ) -> str:
        """Quick classification call with low max_tokens."""
        response = await self.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=100,
            temperature=0.0,
        )
        return response.content
