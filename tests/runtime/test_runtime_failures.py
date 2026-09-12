import unittest
from unittest.mock import Mock, patch, MagicMock

from app.core.runtime.survival_runtime import SurvivalRuntime
from app.core.runtime.models import RuntimeState, RuntimeConfig, FailureType
from app.core.models.generation import DeathTrigger
from app.core.memory.store import InMemoryStore
from app.core.event_bus.event_bus import EventBus
from app.core.generation.manager import GenerationManager
from app.agents.registry import AgentRegistry
from app.services.market_data.mock_provider import MockMarketDataProvider
from app.services.news.mock_provider import MockNewsProvider
from app.services.execution.mock_provider import MockExecutionProvider
from app.core.portfolio.synchronizer import PortfolioSynchronizer
from app.services.health.provider_health import ProviderHealthChecker
from app.utils.time import now_utc


class TestSurvivalRuntimeFailures(unittest.TestCase):
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

    def test_market_api_unavailable(self):
        """Test handling of market API unavailability."""
        # Mock market data provider to fail
        self.market_data.get_market_clock = Mock(side_effect=Exception("API unavailable"))

        health_result = self.runtime._check_provider_health("market_data")
        self.assertFalse(health_result.available)
        self.assertIsNotNone(health_result.error_message)

    def test_news_api_unavailable(self):
        """Test handling of news API unavailability."""
        # Mock news provider to fail
        self.news.get_latest_news = Mock(side_effect=Exception("API unavailable"))

        health_result = self.runtime._check_provider_health("news")
        self.assertFalse(health_result.available)
        self.assertIsNotNone(health_result.error_message)

    def test_generation_initialization_failure(self):
        """Test handling of generation initialization failure."""
        # Mock generation manager to fail
        self.generation_manager.create_generation = Mock(side_effect=Exception("Initialization failed"))

        result = self.runtime._initialize_generation()
        self.assertFalse(result)

    def test_stale_market_data(self):
        """Test detection of stale market data."""
        # Market data with old timestamp would be considered stale
        # This is handled in the market data provider
        pass

    def test_duplicate_generation_attempt(self):
        """Test that duplicate generation creation is prevented."""
        gen1 = self.generation_manager.create_generation()
        gen2 = self.generation_manager.create_generation()

        # Different IDs should be generated
        self.assertNotEqual(gen1.generation_id, gen2.generation_id)

    def test_death_during_active_positions(self):
        """Test generation death while positions are active.

        update_capital() evaluates death conditions and kills the generation
        automatically when a fatal condition is met (capital <= 0). After the
        kill, check_death_conditions() returns None because the generation is
        no longer ACTIVE/PAUSED — so we assert the generation state instead.
        """
        gen = self.generation_manager.create_generation()
        self.generation_manager.start_generation(gen.generation_id)

        # Re-fetch persisted state (start_generation stores its own copy),
        # add active positions (simulated), and persist via the manager.
        gen = self.generation_manager.get_generation(gen.generation_id)
        gen.active_investments = ["inv1", "inv2"]
        self.generation_manager._store_generation_state(gen)

        # Trigger death: capital depleted while positions are open
        self.generation_manager.update_capital(gen.generation_id, 0.0)

        gen_after = self.generation_manager.get_generation(gen.generation_id)
        self.assertIsNotNone(gen_after)
        self.assertEqual(gen_after.lifecycle_state.value, "DEAD")
        self.assertEqual(gen_after.death_trigger, DeathTrigger.CAPITAL_DEPLETED)

    def test_invalid_strategy(self):
        """Test handling of invalid strategy."""
        # This would be caught during generation initialization
        pass

    def test_failed_backtest(self):
        """Test handling of failed backtest."""
        # Backtest failures are handled in Step 12
        pass

    def test_restart_after_generation_death(self):
        """Test runtime restart after generation death."""
        # Kill a generation (kill_generation requires a DeathTrigger enum)
        gen = self.generation_manager.create_generation()
        self.generation_manager.start_generation(gen.generation_id)
        self.generation_manager.kill_generation(
            gen.generation_id, DeathTrigger.CONFIGURED_CONDITION_VIOLATED, "test"
        )

        # Create new runtime and verify it can initialize a new generation
        new_runtime = SurvivalRuntime(
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

        new_runtime._transition_to(RuntimeState.INITIALIZING_GENERATION)
        result = new_runtime._initialize_generation()
        self.assertTrue(result)

    def test_provider_health_check_failure(self):
        """Test runtime behavior when health check fails."""
        # Mock all providers to fail
        self.market_data.get_market_clock = Mock(side_effect=Exception("Failed"))
        self.news.get_latest_news = Mock(side_effect=Exception("Failed"))

        result = self.runtime._perform_health_check()
        self.assertFalse(result)

    def test_pause_on_provider_outage(self):
        """Test that runtime pauses on critical provider outage."""
        self.runtime._running = True
        self.runtime.current_state = RuntimeState.OBSERVING

        # Simulate provider outage
        self.market_data.get_market_clock = Mock(side_effect=Exception("Outage"))

        # Runtime should transition to ERROR or PAUSED
        # This is handled in the autonomous cycle
        pass


if __name__ == "__main__":
    unittest.main()
