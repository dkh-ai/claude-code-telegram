"""Reminder plugin — set and manage reminders."""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from ..base import PluginResponse

logger = structlog.get_logger()

EXTRACT_SYSTEM_PROMPT = """\
Extract reminder details from the user message.
Return JSON: {"text": "what to remind", "delay_minutes": N, "recurring": null or "daily"|"weekly"}
If you can't parse the reminder, return {"error": "reason"}.
Examples:
- "remind me in 2 hours to call mom" → {"text": "call mom", "delay_minutes": 120, "recurring": null}
- "напомни через 30 минут проверить тесты" → {"text": "проверить тесты", "delay_minutes": 30, "recurring": null}
"""


@dataclass
class Reminder:
    """A reminder record."""

    user_id: int
    text: str
    remind_at: datetime
    recurring: Optional[str] = None


class ReminderPlugin:
    """Set and manage reminders via LLM extraction."""

    name: str = "reminder"
    description: str = "Set and manage reminders"
    patterns: list[re.Pattern[str]] = [
        re.compile(
            r"(remind|напомни|напоминани|через \d+\s*(мин|час|сек)|timer|будильник"
            r"|alarm|поставь\s+таймер|запланируй)",
            re.I,
        ),
    ]
    model: str = "gpt-4o-mini"

    def __init__(self, storage: Any = None, scheduler: Any = None) -> None:
        self._storage = storage
        self._scheduler = scheduler
        self._pending: list[Reminder] = []

    async def can_handle(self, message: str, context: dict[str, Any]) -> float:
        """Check if this looks like a reminder request."""
        for pattern in self.patterns:
            if pattern.search(message):
                return 0.9
        return 0.0

    async def handle(
        self,
        message: str,
        context: dict[str, Any],
        chat_provider: Any,
    ) -> PluginResponse:
        """Extract reminder details and store."""
        try:
            raw = await chat_provider.classify(
                prompt=message,
                system=EXTRACT_SYSTEM_PROMPT,
            )
            data = json.loads(raw.strip())

            if "error" in data:
                return PluginResponse(
                    content=f"Не удалось распознать напоминание: {data['error']}",
                    model=self.model,
                )

            text = data.get("text", message)
            delay = int(data.get("delay_minutes", 30))
            recurring = data.get("recurring")

            remind_at = datetime.now(timezone.utc)
            # Calculate remind_at from delay
            from datetime import timedelta

            remind_at = remind_at + timedelta(minutes=delay)

            user_id = context.get("user_id", 0)
            reminder = Reminder(
                user_id=user_id,
                text=text,
                remind_at=remind_at,
                recurring=recurring,
            )

            # Store in memory (DB storage wired later)
            self._pending.append(reminder)

            # Store in DB if storage available
            if self._storage:
                try:
                    db = self._storage.db_manager
                    async with db.get_connection() as conn:
                        await conn.execute(
                            """INSERT INTO reminders
                            (user_id, reminder_text, remind_at, recurring)
                            VALUES (?, ?, ?, ?)""",
                            (user_id, text, remind_at.isoformat(), recurring),
                        )
                        await conn.commit()
                except Exception as e:
                    logger.warning("Failed to store reminder in DB", error=str(e))

            # Format confirmation
            if delay < 60:
                time_str = f"{delay} мин"
            else:
                hours = delay // 60
                mins = delay % 60
                time_str = f"{hours}ч" + (f" {mins}мин" if mins else "")

            recurring_str = f" (повтор: {recurring})" if recurring else ""

            return PluginResponse(
                content=f"⏰ Напоминание установлено!\n\n"
                f"📝 {text}\n"
                f"⏱ Через {time_str}{recurring_str}",
                model=self.model,
                metadata={"reminder_at": remind_at.isoformat()},
            )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Reminder extraction failed", error=str(exc))
            return PluginResponse(
                content="Не удалось распознать время напоминания. "
                "Попробуй: 'напомни через 30 минут позвонить маме'",
                model=self.model,
            )
