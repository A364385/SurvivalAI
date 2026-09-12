import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from app.core.runtime.survival_runtime import SurvivalRuntime
from app.core.runtime.models import RuntimeState, RuntimeConfig, CyclePhase, RuntimeStateSnapshot, CycleRecord, FailureType
from app.core.memory.store import InMemoryStore
from app.core.models.memory import MemoryType
from app.core.event_bus.event_bus import EventBus
from app.core.generation.manager import GenerationManager
from app.agents.registry import AgentRegistry
from app.services.market_data.mock_provider import MockMarketDataProvider
from app.services.news.mock_provider import MockNewsProvider
from app.services.execution.mock_provider import MockExecutionProvider
from app.core.portfolio.synchronizer import PortfolioSynchronizer
from app.services.health.provider_health import ProviderHealthChecker
from app.utils.time import now_utc


class TestSurvivalRuntime(unittest.TestCase):
    def setUp(self):
        self.memory = InMemoryStore()
        self.event_bus = EventBus()
        self.agent_registry = AgentRegistry()
        self.market_data = MockMarketDataProvider()
        self.news = MockNewsProvider()
        self.execution = MockExecutionProvider()
        self.portfolio_sync = PortfolioSynchronizer(self.execution)
        self.health_checker = ProviderHealthChecker()

        self.generation_manager = GenerationManager(
            memory_store=self.memory,
            event_bus=self.event_bus,
            agent_registry=self.agent_registry,
            market_data_provider=self.market_data,
            news_provider=self.news,
            llm_provider=None,
            initial_capital=100000.0,
        )

        self.config = RuntimeConfig(
            cycle_interval_seconds=1,
            test_mode=True,
            paper_mode_required=True,
        )

        self.runtime = SurvivalRuntime(
            memory_store=self.memory,
            event_bus=self.event_bus,
            generation_manager=self.generation_manager,
            agent_registry=self.agent_registry,
            market_data_provider=self.market_data,
            news_provider=self.news,
            execution_provider=self.execution,
            portfolio_synchronizer=self.portfolio_sync,
            provider_health_checker=self.health_checker,
            config=self.config,
        )

    def test_runtime_initialization(self):
        """Test runtime initialization."""
        self.assertIsNotNone(self.runtime)
        self.assertEqual(self.runtime.current_state, RuntimeState.STARTING)
        self.assertIsNotNone(self.runtime.runtime_id)

    def test_state_transitions(self):
        """Test valid state transitions."""
        # STARTING -> HEALTH_CHECK
        self.assertTrue(self.runtime._transition_to(RuntimeState.HEALTH_CHECK))
        self.assertEqual(self.runtime.current_state, RuntimeState.HEALTH_CHECK)

        # HEALTH_CHECK -> INITIALIZING_GENERATION
        self.assertTrue(self.runtime._transition_to(RuntimeState.INITIALIZING_GENERATION))
        self.assertEqual(self.runtime.current_state, RuntimeState.INITIALIZING_GENERATION)

        # Invalid transition
        self.assertFalse(self.runtime._transition_to(RuntimeState.STARTING))

    def test_invalid_state_transition(self):
        """Test that invalid state transitions are rejected."""
        self.runtime.current_state = RuntimeState.STARTING
        # Cannot transition from STARTING to OBSERVING directly
        self.assertFalse(self.runtime._transition_to(RuntimeState.OBSERVING))

    def test_health_check(self):
        """Test health check functionality."""
        self.runtime._transition_to(RuntimeState.HEALTH_CHECK)
        result = self.runtime._perform_health_check()
        self.assertTrue(result)

    def test_generation_initialization(self):
        """Test generation initialization."""
        self.runtime._transition_to(RuntimeState.INITIALIZING_GENERATION)
        result = self.runtime._initialize_generation()
        self.assertTrue(result)
        self.assertIsNotNone(self.runtime.active_generation_id)

    def test_market_data_collection(self):
        """Test market data collection."""
        data = self.runtime._collect_market_data()
        self.assertIsNotNone(data)
        self.assertIn("clock", data)

    def test_news_collection(self):
        """Test news collection."""
        data = self.runtime._collect_news()
        self.assertIsNotNone(data)
        self.assertIn("news", data)

    def test_crisis_data_collection(self):
        """Test crisis data collection."""
        data = self.runtime._collect_crisis_data()
        self.assertIsNotNone(data)

    def test_survival_check(self):
        """Test survival check."""
        self.runtime.active_generation_id = self.generation_manager.create_generation().generation_id
        self.generation_manager.start_generation(self.runtime.active_generation_id)

        result = self.runtime._check_survival()
        self.assertIsNotNone(result)
        self.assertIn("death_triggered", result)
        self.assertFalse(result["death_triggered"])

    def test_survival_check_with_death(self):
        """Test survival check with death condition."""
        gen = self.generation_manager.create_generation()
        self.generation_manager.start_generation(gen.generation_id)
        self.runtime.active_generation_id = gen.generation_id

        # Verify the generation is alive
        result_before = self.runtime._check_survival()
        self.assertFalse(result_before["death_triggered"])

        # Trigger death condition - this will automatically kill the generation
        self.generation_manager.update_capital(gen.generation_id, 5000.0)

        # Verify the generation is now dead
        gen_after = self.generation_manager.get_generation(gen.generation_id)
        self.assertEqual(gen_after.lifecycle_state.value, "DEAD")

    def test_cannot_execute_live_trade(self):
        """Test that runtime cannot execute live trades."""
        with self.assertRaises(PermissionError):
            self.runtime.execute_live_trade()

    def test_cannot_activate_live_trading(self):
        """Test that runtime cannot activate live trading."""
        with self.assertRaises(PermissionError):
            self.runtime.activate_live_trading()

    def test_failure_recording(self):
        """Test failure recording."""
        self.runtime._record_failure(FailureType.TRANSIENT, "test_component", "test error")

        # Verify it was stored (MemoryType is an enum, not a string)
        records = self.memory.query(memory_type=MemoryType.ERROR)
        self.assertTrue(len(records) > 0)


if __name__ == "__main__":
    unittest.main()
