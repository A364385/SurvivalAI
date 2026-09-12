import unittest
from datetime import datetime, timedelta

from app.core.generation.manager import GenerationManager
from app.core.memory.store import InMemoryStore
from app.core.models.generation import (
    GenerationLifecycleState,
    DeathTrigger,
    OperatingCost,
    GenerationState,
)
from app.core.models.memory import MemoryType
from app.core.event_bus.event_bus import EventBus
from app.agents.registry import AgentRegistry
from app.services.market_data.mock_provider import MockMarketDataProvider
from app.services.news.mock_provider import MockNewsProvider
from app.services.llm.mock_provider import MockLLMProvider
from app.utils.time import now_utc


class TestGenerationManager(unittest.TestCase):
    def setUp(self):
        self.now = now_utc()
        self.memory = InMemoryStore()
        self.event_bus = EventBus()
        self.registry = AgentRegistry()
        self.market_data = MockMarketDataProvider()
        self.news = MockNewsProvider()
        self.llm = MockLLMProvider({"response": "test"})

        self.manager = GenerationManager(
            memory_store=self.memory,
            event_bus=self.event_bus,
            agent_registry=self.registry,
            market_data_provider=self.market_data,
            news_provider=self.news,
            llm_provider=self.llm,
            initial_capital=100000.0,
            minimum_survival_threshold=10000.0,
            maximum_drawdown_threshold=0.5,
        )

    def test_create_generation(self):
        """Test creating a new generation."""
        state = self.manager.create_generation()

        self.assertIsNotNone(state)
        self.assertEqual(state.generation_number, 1)
        self.assertEqual(state.lifecycle_state, GenerationLifecycleState.CREATED)
        self.assertEqual(state.starting_capital, 100000.0)
        self.assertEqual(state.current_capital, 100000.0)
        self.assertIsNone(state.parent_generation_id)

    def test_get_generation(self):
        """Test retrieving a generation."""
        state = self.manager.create_generation()
        retrieved = self.manager.get_generation(state.generation_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.generation_id, state.generation_id)

    def test_cannot_execute_trade(self):
        """Test that GenerationManager cannot execute trades."""
        with self.assertRaises(PermissionError):
            self.manager.execute_trade()

    def test_cannot_activate_strategy(self):
        """Test that GenerationManager cannot directly activate strategies."""
        with self.assertRaises(PermissionError):
            self.manager.activate_strategy()


if __name__ == "__main__":
    unittest.main()
