"""LLM-based fact extraction from conversations."""

import json
from typing import Any, Optional

import structlog

from .models import UserFact

logger = structlog.get_logger()

EXTRACT_FACTS_SYSTEM = """\
Extract any new facts about the user from this exchange.
Return a JSON array: [{"category": "preference|personal|work|location|contact|technical", "fact": "..."}]
If no new facts, return [].

Categories:
- preference: likes/dislikes, style preferences
- personal: name, age, family, hobbies
- work: company, role, projects
- location: city, timezone, country
- contact: email, phone (only if explicitly shared)
- technical: programming languages, tools, frameworks they use"""

SUMMARIZE_SYSTEM = """\
Summarize this conversation session in 2-3 sentences.
Focus on: decisions made, tasks completed, key topics discussed.
Start with the date if available."""


class FactExtractor:
    """Extract user facts and generate summaries using a cheap LLM."""

    def __init__(self, chat_provider: Any = None) -> None:
        self._provider = chat_provider

    async def extract(
        self,
        user_message: str,
        bot_response: str,
        user_id: int = 0,
    ) -> list[UserFact]:
        """Extract new facts from a message exchange."""
        if not self._provider:
            return []

        if len(user_message) < 10 and len(bot_response) < 10:
            return []

        exchange = f"User: {user_message[:500]}\nAssistant: {bot_response[:500]}"

        try:
            raw = await self._provider.classify(
                prompt=exchange,
                system=EXTRACT_FACTS_SYSTEM,
            )
            data = json.loads(raw.strip())

            if not isinstance(data, list):
                return []

            facts = []
            for item in data:
                if isinstance(item, dict) and "category" in item and "fact" in item:
                    facts.append(
                        UserFact(
                            user_id=user_id,
                            category=item["category"],
                            fact=item["fact"],
                            source="auto_extract",
                        )
                    )
            return facts

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.debug("Fact extraction parse error", error=str(exc))
            return []
        except Exception as exc:
            logger.warning("Fact extraction failed", error=str(exc))
            return []

    async def summarize(self, messages: list[dict[str, str]]) -> Optional[str]:
        """Generate a conversation summary."""
        if not self._provider or len(messages) < 2:
            return None

        # Build a condensed transcript
        lines = []
        for msg in messages[-20:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]
            lines.append(f"{role}: {content}")

        transcript = "\n".join(lines)

        try:
            response = await self._provider.chat(
                messages=[
                    {"role": "system", "content": SUMMARIZE_SYSTEM},
                    {"role": "user", "content": transcript},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return response.content.strip()
        except Exception as exc:
            logger.warning("Conversation summarization failed", error=str(exc))
            return None
