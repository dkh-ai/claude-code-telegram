"""Test MemoryManager — recall and store user facts and summaries."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.extractor import FactExtractor
from src.memory.manager import MemoryManager
from src.memory.models import MemoryContext, UserFact


class TestMemoryManager:
    """Test MemoryManager for memory recall, storage, and formatting."""

    def test_construction_without_dependencies(self) -> None:
        """Test MemoryManager can be created without db_manager or extractor."""
        manager = MemoryManager()
        assert manager._db is None
        assert manager._extractor is not None  # Default FactExtractor created

    def test_construction_with_db_manager(self) -> None:
        """Test MemoryManager can be created with db_manager."""
        db_manager = MagicMock()
        manager = MemoryManager(db_manager=db_manager)
        assert manager._db is db_manager

    def test_construction_with_custom_extractor(self) -> None:
        """Test MemoryManager can be created with custom extractor."""
        extractor = FactExtractor()
        manager = MemoryManager(extractor=extractor)
        assert manager._extractor is extractor

    def test_construction_creates_default_extractor_when_none_provided(self) -> None:
        """Test MemoryManager creates default FactExtractor when extractor is None."""
        manager = MemoryManager(extractor=None)
        assert manager._extractor is not None
        assert isinstance(manager._extractor, FactExtractor)

    async def test_recall_returns_empty_context_without_db(self) -> None:
        """Test recall() returns empty MemoryContext when no db_manager."""
        manager = MemoryManager()
        context = await manager.recall(user_id=123, message="test")

        assert isinstance(context, MemoryContext)
        assert context.facts == []
        assert context.summaries == []

    async def test_recall_fetches_facts_and_summaries(self) -> None:
        """Test recall() fetches facts and summaries from database."""
        db_manager = MagicMock()
        manager = MemoryManager(db_manager=db_manager)

        # Mock database operations
        fact1 = UserFact(user_id=123, category="preference", fact="likes Python")
        fact2 = UserFact(user_id=123, category="work", fact="software engineer")

        manager._get_user_facts = AsyncMock(return_value=[fact1, fact2])
        manager._search_summaries = AsyncMock(
            return_value=["Summary 1", "Summary 2", "Summary 3"]
        )

        context = await manager.recall(user_id=123, message="tell me about Python")

        assert len(context.facts) == 2
        assert context.facts[0].fact == "likes Python"
        assert context.facts[1].fact == "software engineer"
        assert len(context.summaries) == 3
        assert context.summaries[0] == "Summary 1"

        # Verify internal methods were called correctly
        manager._get_user_facts.assert_called_once_with(123)
        manager._search_summaries.assert_called_once_with(123, "tell me about Python", limit=3)

    async def test_extract_and_store_does_nothing_without_extractor(self) -> None:
        """Test extract_and_store() does nothing when no extractor."""
        manager = MemoryManager(extractor=None)
        manager._extractor = None  # Force None

        # Should not raise exception
        await manager.extract_and_store(
            user_id=123,
            message="I love Python",
            response="That's great!",
        )

    async def test_extract_and_store_calls_extractor_and_upserts_facts(self) -> None:
        """Test extract_and_store() calls extractor.extract and upserts facts."""
        extractor = MagicMock()
        fact1 = UserFact(user_id=456, category="preference", fact="likes Python")
        fact2 = UserFact(user_id=456, category="work", fact="data scientist")
        extractor.extract = AsyncMock(return_value=[fact1, fact2])

        manager = MemoryManager(extractor=extractor)
        manager._upsert_fact = AsyncMock()

        await manager.extract_and_store(
            user_id=456,
            message="I love Python and work as a data scientist",
            response="That's wonderful!",
        )

        # Verify extractor was called
        extractor.extract.assert_called_once_with(
            "I love Python and work as a data scientist",
            "That's wonderful!",
            456,
        )

        # Verify _upsert_fact was called for each fact
        assert manager._upsert_fact.call_count == 2
        manager._upsert_fact.assert_any_call(456, fact1)
        manager._upsert_fact.assert_any_call(456, fact2)

    async def test_extract_and_store_handles_empty_facts_list(self) -> None:
        """Test extract_and_store() handles empty facts list from extractor."""
        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value=[])

        manager = MemoryManager(extractor=extractor)
        manager._upsert_fact = AsyncMock()

        await manager.extract_and_store(
            user_id=789,
            message="Short",
            response="Ok",
        )

        # Verify extractor was called but no upserts happened
        extractor.extract.assert_called_once()
        manager._upsert_fact.assert_not_called()

    async def test_summarize_session_does_nothing_without_extractor(self) -> None:
        """Test summarize_session() does nothing when no extractor."""
        manager = MemoryManager(extractor=None)
        manager._extractor = None  # Force None

        # Should not raise exception
        await manager.summarize_session(
            user_id=123,
            session_messages=[{"role": "user", "content": "Hello"}],
            session_id="session-1",
        )

    async def test_summarize_session_calls_extractor_and_stores_summary(self) -> None:
        """Test summarize_session() calls extractor.summarize and stores result."""
        extractor = MagicMock()
        extractor.summarize = AsyncMock(return_value="User discussed Python programming.")

        manager = MemoryManager(extractor=extractor)
        manager._store_summary = AsyncMock()

        messages = [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language"},
        ]

        await manager.summarize_session(
            user_id=100,
            session_messages=messages,
            session_id="session-abc",
        )

        # Verify extractor was called
        extractor.summarize.assert_called_once_with(messages)

        # Verify summary was stored
        manager._store_summary.assert_called_once_with(
            100, "User discussed Python programming.", "session-abc"
        )

    async def test_summarize_session_does_not_store_none_summary(self) -> None:
        """Test summarize_session() does not store when extractor returns None."""
        extractor = MagicMock()
        extractor.summarize = AsyncMock(return_value=None)

        manager = MemoryManager(extractor=extractor)
        manager._store_summary = AsyncMock()

        messages = [{"role": "user", "content": "Hi"}]

        await manager.summarize_session(
            user_id=200,
            session_messages=messages,
        )

        # Verify extractor was called
        extractor.summarize.assert_called_once()

        # Verify summary was NOT stored
        manager._store_summary.assert_not_called()

    def test_format_for_prompt_with_empty_memory(self) -> None:
        """Test format_for_prompt() returns empty string for empty memory."""
        manager = MemoryManager()
        memory = MemoryContext()

        result = manager.format_for_prompt(memory)

        assert result == ""

    def test_format_for_prompt_with_facts_only(self) -> None:
        """Test format_for_prompt() formats facts correctly."""
        manager = MemoryManager()
        fact1 = UserFact(user_id=1, category="preference", fact="likes Python")
        fact2 = UserFact(user_id=1, category="work", fact="software engineer")
        fact3 = UserFact(user_id=1, category="location", fact="lives in SF")

        memory = MemoryContext(facts=[fact1, fact2, fact3])

        result = manager.format_for_prompt(memory)

        assert "Known facts about the user:" in result
        assert "- [preference] likes Python" in result
        assert "- [work] software engineer" in result
        assert "- [location] lives in SF" in result
        assert "Recent conversation context:" not in result

    def test_format_for_prompt_with_summaries_only(self) -> None:
        """Test format_for_prompt() formats summaries correctly."""
        manager = MemoryManager()
        summaries = [
            "Discussed Python async programming",
            "Asked about database optimization",
            "Talked about deployment strategies",
        ]

        memory = MemoryContext(summaries=summaries)

        result = manager.format_for_prompt(memory)

        assert "Recent conversation context:" in result
        assert "- Discussed Python async programming" in result
        assert "- Asked about database optimization" in result
        assert "- Talked about deployment strategies" in result
        assert "Known facts about the user:" not in result

    def test_format_for_prompt_with_both_facts_and_summaries(self) -> None:
        """Test format_for_prompt() formats both facts and summaries."""
        manager = MemoryManager()
        fact1 = UserFact(user_id=1, category="preference", fact="likes Python")
        fact2 = UserFact(user_id=1, category="work", fact="data engineer")

        summaries = ["Previously discussed databases", "Worked on API design"]

        memory = MemoryContext(facts=[fact1, fact2], summaries=summaries)

        result = manager.format_for_prompt(memory)

        # Should contain both sections separated by double newline
        assert "Known facts about the user:" in result
        assert "- [preference] likes Python" in result
        assert "- [work] data engineer" in result
        assert "Recent conversation context:" in result
        assert "- Previously discussed databases" in result
        assert "- Worked on API design" in result

        # Verify sections are separated by double newline
        assert "\n\n" in result

    def test_format_for_prompt_preserves_order(self) -> None:
        """Test format_for_prompt() preserves order of facts and summaries."""
        manager = MemoryManager()
        fact1 = UserFact(user_id=1, category="a", fact="first fact")
        fact2 = UserFact(user_id=1, category="b", fact="second fact")
        fact3 = UserFact(user_id=1, category="c", fact="third fact")

        memory = MemoryContext(facts=[fact1, fact2, fact3])

        result = manager.format_for_prompt(memory)

        # Find positions of each fact in the output
        pos1 = result.find("first fact")
        pos2 = result.find("second fact")
        pos3 = result.find("third fact")

        # Verify order is preserved
        assert pos1 < pos2 < pos3

    async def test_get_user_facts_returns_empty_without_db(self) -> None:
        """Test _get_user_facts() returns empty list without db_manager."""
        manager = MemoryManager()
        facts = await manager._get_user_facts(user_id=123)
        assert facts == []

    async def test_get_user_facts_fetches_from_database(self) -> None:
        """Test _get_user_facts() fetches facts from database."""
        db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Setup mock database response
        mock_row1 = {
            "user_id": 123,
            "category": "preference",
            "fact": "likes Python",
            "source": "auto_extract",
            "confidence": 1.0,
            "created_at": None,
            "updated_at": None,
        }
        mock_row2 = {
            "user_id": 123,
            "category": "work",
            "fact": "software engineer",
            "source": "manual",
            "confidence": 0.9,
            "created_at": None,
            "updated_at": None,
        }

        mock_cursor.fetchall = AsyncMock(return_value=[mock_row1, mock_row2])
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)
        facts = await manager._get_user_facts(user_id=123)

        assert len(facts) == 2
        assert facts[0].fact == "likes Python"
        assert facts[0].source == "auto_extract"
        assert facts[1].fact == "software engineer"
        assert facts[1].source == "manual"

    async def test_get_user_facts_handles_database_exception(self) -> None:
        """Test _get_user_facts() handles database exceptions gracefully."""
        db_manager = MagicMock()
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(
            side_effect=Exception("Database error")
        )
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)
        facts = await manager._get_user_facts(user_id=456)

        # Should return empty list on error
        assert facts == []

    async def test_search_summaries_returns_empty_without_db(self) -> None:
        """Test _search_summaries() returns empty list without db_manager."""
        manager = MemoryManager()
        summaries = await manager._search_summaries(user_id=123, query="test", limit=3)
        assert summaries == []

    async def test_search_summaries_with_keyword_search(self) -> None:
        """Test _search_summaries() performs keyword search when query is long enough."""
        db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Setup mock database response
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("Discussed Python async programming",),
                ("Talked about Python decorators",),
            ]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)
        summaries = await manager._search_summaries(
            user_id=789, query="tell me about Python programming", limit=5
        )

        assert len(summaries) == 2
        assert summaries[0] == "Discussed Python async programming"
        assert summaries[1] == "Talked about Python decorators"

    async def test_search_summaries_falls_back_to_recent_when_no_keyword_matches(self) -> None:
        """Test _search_summaries() falls back to most recent when no keyword matches."""
        db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # First execute returns empty (no keyword matches)
        # Second execute returns recent summaries
        mock_cursor.fetchall = AsyncMock(
            side_effect=[
                [],  # No keyword matches
                [("Recent summary 1",), ("Recent summary 2",)],  # Recent summaries
            ]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)
        summaries = await manager._search_summaries(
            user_id=100, query="tell me about databases", limit=2
        )

        assert len(summaries) == 2
        assert summaries[0] == "Recent summary 1"
        assert summaries[1] == "Recent summary 2"

    async def test_search_summaries_uses_recent_for_short_query(self) -> None:
        """Test _search_summaries() uses recent summaries when query is too short."""
        db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.fetchall = AsyncMock(
            return_value=[("Recent summary",)]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)
        summaries = await manager._search_summaries(user_id=200, query="test", limit=1)

        # Should go straight to recent summaries (query too short)
        assert len(summaries) == 1
        assert summaries[0] == "Recent summary"

    async def test_search_summaries_handles_database_exception(self) -> None:
        """Test _search_summaries() handles database exceptions gracefully."""
        db_manager = MagicMock()
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(
            side_effect=Exception("Database error")
        )
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)
        summaries = await manager._search_summaries(user_id=300, query="test query", limit=3)

        # Should return empty list on error
        assert summaries == []

    async def test_upsert_fact_does_nothing_without_db(self) -> None:
        """Test _upsert_fact() does nothing without db_manager."""
        manager = MemoryManager()
        fact = UserFact(user_id=123, category="test", fact="test fact")

        # Should not raise exception
        await manager._upsert_fact(user_id=123, fact=fact)

    async def test_upsert_fact_inserts_into_database(self) -> None:
        """Test _upsert_fact() inserts fact into database."""
        db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)
        fact = UserFact(
            user_id=456,
            category="preference",
            fact="likes Python",
            source="auto_extract",
            confidence=0.95,
        )

        await manager._upsert_fact(user_id=456, fact=fact)

        # Verify database execute was called
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]

        assert "INSERT INTO user_memory" in sql
        assert "ON CONFLICT" in sql
        assert params == (456, "preference", "likes Python", "auto_extract", 0.95)

        # Verify commit was called
        mock_conn.commit.assert_called_once()

    async def test_upsert_fact_handles_database_exception(self) -> None:
        """Test _upsert_fact() handles database exceptions gracefully."""
        db_manager = MagicMock()
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(
            side_effect=Exception("Database error")
        )
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)
        fact = UserFact(user_id=789, category="test", fact="test fact")

        # Should not raise exception
        await manager._upsert_fact(user_id=789, fact=fact)

    async def test_store_summary_does_nothing_without_db(self) -> None:
        """Test _store_summary() does nothing without db_manager."""
        manager = MemoryManager()

        # Should not raise exception
        await manager._store_summary(
            user_id=123, summary="Test summary", session_id="session-1"
        )

    async def test_store_summary_inserts_into_database(self) -> None:
        """Test _store_summary() inserts summary into database."""
        db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)

        await manager._store_summary(
            user_id=100,
            summary="User discussed Python programming",
            session_id="session-abc-123",
        )

        # Verify database execute was called
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]

        assert "INSERT INTO conversation_summaries" in sql
        assert params == (100, "session-abc-123", "User discussed Python programming")

        # Verify commit was called
        mock_conn.commit.assert_called_once()

    async def test_store_summary_handles_none_session_id(self) -> None:
        """Test _store_summary() handles None session_id."""
        db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)

        await manager._store_summary(user_id=200, summary="Summary without session", session_id=None)

        # Verify database execute was called with None session_id
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params == (200, None, "Summary without session")

    async def test_store_summary_handles_database_exception(self) -> None:
        """Test _store_summary() handles database exceptions gracefully."""
        db_manager = MagicMock()
        db_manager.get_connection = MagicMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(
            side_effect=Exception("Database error")
        )
        db_manager.get_connection.return_value.__aexit__ = AsyncMock()

        manager = MemoryManager(db_manager=db_manager)

        # Should not raise exception
        await manager._store_summary(user_id=300, summary="Test summary")
