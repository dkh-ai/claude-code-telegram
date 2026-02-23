"""Memory manager — recall and store user facts and summaries."""

from typing import Any, Optional

import structlog

from .extractor import FactExtractor
from .models import MemoryContext, UserFact

logger = structlog.get_logger()


class MemoryManager:
    """Manages persistent user memory: facts, summaries, working memory."""

    def __init__(
        self,
        db_manager: Any = None,
        extractor: Optional[FactExtractor] = None,
    ) -> None:
        self._db = db_manager
        self._extractor = extractor or FactExtractor()

    async def recall(self, user_id: int, message: str = "") -> MemoryContext:
        """Retrieve relevant memories for a user."""
        facts = await self._get_user_facts(user_id)
        summaries = await self._search_summaries(user_id, message, limit=3)

        return MemoryContext(
            facts=facts,
            summaries=summaries,
        )

    async def extract_and_store(
        self,
        user_id: int,
        message: str,
        response: str,
    ) -> None:
        """Extract new facts from an exchange and store them."""
        if not self._extractor:
            return

        new_facts = await self._extractor.extract(message, response, user_id)
        for fact in new_facts:
            await self._upsert_fact(user_id, fact)

    async def summarize_session(
        self,
        user_id: int,
        session_messages: list[dict[str, str]],
        session_id: Optional[str] = None,
    ) -> None:
        """Generate and store a conversation summary."""
        if not self._extractor:
            return

        summary = await self._extractor.summarize(session_messages)
        if summary:
            await self._store_summary(user_id, summary, session_id)

    def format_for_prompt(self, memory: MemoryContext) -> str:
        """Format memory context as text for system prompt injection."""
        parts: list[str] = []

        if memory.facts:
            facts_text = "\n".join(
                f"- [{f.category}] {f.fact}" for f in memory.facts
            )
            parts.append(f"Known facts about the user:\n{facts_text}")

        if memory.summaries:
            summaries_text = "\n".join(f"- {s}" for s in memory.summaries)
            parts.append(f"Recent conversation context:\n{summaries_text}")

        return "\n\n".join(parts)

    # --- DB operations ---

    async def _get_user_facts(self, user_id: int) -> list[UserFact]:
        """Fetch all facts for a user."""
        if not self._db:
            return []

        try:
            async with self._db.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM user_memory WHERE user_id = ? ORDER BY updated_at DESC",
                    (user_id,),
                )
                rows = await cursor.fetchall()
                return [UserFact.from_row(dict(row)) for row in rows]
        except Exception as exc:
            logger.warning("Failed to fetch user facts", error=str(exc))
            return []

    async def _upsert_fact(self, user_id: int, fact: UserFact) -> None:
        """Insert or update a fact."""
        if not self._db:
            return

        try:
            async with self._db.get_connection() as conn:
                await conn.execute(
                    """INSERT INTO user_memory (user_id, category, fact, source, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, category, fact) DO UPDATE SET
                        confidence = excluded.confidence,
                        updated_at = CURRENT_TIMESTAMP""",
                    (user_id, fact.category, fact.fact, fact.source, fact.confidence),
                )
                await conn.commit()
        except Exception as exc:
            logger.warning("Failed to upsert fact", error=str(exc))

    async def _search_summaries(
        self, user_id: int, query: str, limit: int = 3
    ) -> list[str]:
        """Search conversation summaries by keyword."""
        if not self._db:
            return []

        try:
            async with self._db.get_connection() as conn:
                # Simple keyword search; LIKE is good enough for now
                if query and len(query) > 5:
                    # Extract first significant word
                    words = [w for w in query.split() if len(w) > 3]
                    if words:
                        keyword = words[0]
                        cursor = await conn.execute(
                            """SELECT summary FROM conversation_summaries
                            WHERE user_id = ? AND (summary LIKE ? OR key_topics LIKE ?)
                            ORDER BY created_at DESC LIMIT ?""",
                            (user_id, f"%{keyword}%", f"%{keyword}%", limit),
                        )
                        rows = await cursor.fetchall()
                        if rows:
                            return [row[0] for row in rows]

                # Fallback: most recent summaries
                cursor = await conn.execute(
                    """SELECT summary FROM conversation_summaries
                    WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
                    (user_id, limit),
                )
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
        except Exception as exc:
            logger.warning("Failed to search summaries", error=str(exc))
            return []

    async def _store_summary(
        self, user_id: int, summary: str, session_id: Optional[str] = None
    ) -> None:
        """Store a conversation summary."""
        if not self._db:
            return

        try:
            async with self._db.get_connection() as conn:
                await conn.execute(
                    """INSERT INTO conversation_summaries
                    (user_id, session_id, summary)
                    VALUES (?, ?, ?)""",
                    (user_id, session_id, summary),
                )
                await conn.commit()
        except Exception as exc:
            logger.warning("Failed to store summary", error=str(exc))
