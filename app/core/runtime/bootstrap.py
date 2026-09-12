"""Runtime bootstrap factory.

Wires the complete SurvivalAI system from configuration:

- providers (mock for test mode, Alpaca paper for production),
- agents (research, risk, safety, CEO, strategy),
- generation manager, memory, event bus,
- the SurvivalRuntime with all Step 15 services.

Production mode is internet-connected and strictly paper-trading-only.
Test mode uses deterministic mocks and never touches the network.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.agents.registry import AgentRegistry
from app.agents.ceo.agent import CEOAgent
from app.agents.crisis_risk.agent import CrisisRiskAgent
from app.agents.deep_looker.agent import DeepLookerAgent
from app.agents.investment_safety.agent import InvestmentSafetyManagerAgent
from app.agents.market_research.agent import MarketResearchAgent
from app.agents.news_research.agent import NewsResearchAgent
from app.agents.risk_manager.agent import RiskManagerAgent
from app.agents.strategy_updater.agent import StrategyUpdaterAgent
from app.core.event_bus.event_bus import EventBus
from app.core.generation.manager import GenerationManager
from app.core.memory.store import InMemoryStore, MemoryStore
from app.core.runtime.models import RuntimeConfig
from app.core.runtime.survival_runtime import SurvivalRuntime
from app.services.execution.mock_provider import MockExecutionProvider
from app.services.llm.mock_provider import MockLLMProvider
from app.services.market_data.mock_provider import MockMarketDataProvider
from app.services.news.mock_provider import MockNewsProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)


class BootstrapResult:
    """Container for the wired system."""

    def __init__(
        self,
        runtime: SurvivalRuntime,
        agent_registry: AgentRegistry,
        config: RuntimeConfig,
        components: Dict[str, Any],
    ):
        self.runtime = runtime
        self.agent_registry = agent_registry
        self.config = config
        self.components = components


def build_test_system(
    config: Optional[RuntimeConfig] = None,
    initial_capital: float = 100000.0,
) -> BootstrapResult:
    """Build a fully-wired deterministic test system (no network access).

    Uses mock providers for market data, news, LLM, and execution so the
    entire autonomous loop can run offline and deterministically.
    """
    config = config or RuntimeConfig(test_mode=True)

    memory_store = InMemoryStore()
    event_bus = EventBus()
    agent_registry = AgentRegistry()

    market_data = MockMarketDataProvider()
    news = MockNewsProvider()
    llm = MockLLMProvider()
    from app.services.execution.mock_provider import MockExecutionProvider
    execution = MockExecutionProvider(initial_cash=initial_capital)

    from app.core.portfolio.synchronizer import PortfolioSynchronizer
    portfolio_sync = PortfolioSynchronizer(execution)

    # --- Agents -----------------------------------------------------------
    from app.agents.market_research.agent import MarketResearchAgent
    from app.agents.news_research.agent import NewsResearchAgent
    from app.agents.crisis_risk.agent import CrisisRiskAgent
    from app.agents.deep_looker.agent import DeepLookerAgent
    from app.agents.risk_manager.agent import RiskManagerAgent
    from app.agents.investment_safety.agent import InvestmentSafetyManagerAgent
    from app.agents.ceo.agent import CEOAgent
    from app.agents.strategy_updater.agent import StrategyUpdaterAgent

    market_agent = MarketResearchAgent(
        agent_id="market_research",
        market_data_provider=market_data,
        llm_provider=llm,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    news_agent = NewsResearchAgent(
        agent_id="news_research",
        news_provider=news,
        llm_provider=llm,
        memory_store=memory_store,
    )
    crisis_agent = CrisisRiskAgent(
        agent_id="crisis_risk",
        llm_provider=llm,
        news_provider=news,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    deep_looker = DeepLookerAgent(
        agent_id="deep_looker",
        llm_provider=llm,
        market_data_provider=market_data,
        news_provider=news,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    risk_manager = RiskManagerAgent(
        agent_id="risk_manager",
        llm_provider=llm,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    safety_manager = InvestmentSafetyManagerAgent(
        agent_id="investment_safety",
        llm_provider=llm,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    strategy_updater = StrategyUpdaterAgent(
        agent_id="strategy_updater",
        llm_provider=llm,
        memory_store=memory_store,
    )

    agent_registry.register(market_agent)
    agent_registry.register(news_agent)
    agent_registry.register(crisis_agent)
    agent_registry.register(deep_looker)
    agent_registry.register(risk_manager)
    agent_registry.register(safety_manager)
    agent_registry.register(strategy_updater)

    ceo = CEOAgent(
        agent_id="ceo",
        agent_registry=agent_registry,
        llm_provider=llm,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    agent_registry.register(ceo)

    generation_manager = GenerationManager(
        memory_store=memory_store,
        event_bus=event_bus,
        agent_registry=agent_registry,
        market_data_provider=market_data,
        news_provider=news,
        llm_provider=llm,
        initial_capital=initial_capital,
    )

    runtime = SurvivalRuntime(
        memory_store=memory_store,
        event_bus=event_bus,
        generation_manager=generation_manager,
        agent_registry=agent_registry,
        market_data_provider=market_data,
        news_provider=news,
        execution_provider=execution,
        portfolio_synchronizer=portfolio_sync,
        config=config,
    )

    components = {
        "memory_store": memory_store,
        "event_bus": event_bus,
        "market_data_provider": market_data,
        "news_provider": news,
        "llm_provider": llm,
        "execution_provider": execution,
        "generation_manager": generation_manager,
        "ceo_agent": ceo,
        "risk_manager_agent": risk_manager,
        "safety_manager_agent": safety_manager,
        "strategy_updater_agent": strategy_updater,
    }
    return BootstrapResult(runtime, agent_registry, config, components)


def build_production_system(
    config: Optional[RuntimeConfig] = None,
    initial_capital: float = 100000.0,
) -> BootstrapResult:
    """Build the internet-connected, paper-trading-only production system.

    Credentials come exclusively from environment variables (never hardcoded,
    never logged). If paper-trading configuration is missing the build FAILS
    SAFE: no runtime is returned and nothing executes.
    """
    config = config or RuntimeConfig(test_mode=False)

    # Fail-safe: refuse to build without paper credentials.
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Production bootstrap requires ALPACA_API_KEY and ALPACA_API_SECRET "
            "environment variables for PAPER trading. Refusing to start without "
            "paper-trading configuration (fail-safe: no execution)."
        )

    memory_store = InMemoryStore()
    event_bus = EventBus()
    agent_registry = AgentRegistry()

    from app.services.market_data.alpaca_provider import AlpacaMarketDataProvider
    from app.services.news.alpaca_provider import AlpacaNewsProvider
    from app.services.execution.alpaca_provider import AlpacaPaperExecutionProvider

    market_data = AlpacaMarketDataProvider()
    news = AlpacaNewsProvider()
    execution = AlpacaPaperExecutionProvider(
        api_key=api_key,
        api_secret=api_secret,
    )

    # LLM: optional; falls back to deterministic mock if not configured.
    llm = _build_llm_provider()

    from app.agents.market_research.agent import MarketResearchAgent
    from app.agents.news_research.agent import NewsResearchAgent
    from app.agents.crisis_risk.agent import CrisisRiskAgent
    from app.agents.deep_looker.agent import DeepLookerAgent
    from app.agents.risk_manager.agent import RiskManagerAgent
    from app.agents.investment_safety.agent import InvestmentSafetyManagerAgent
    from app.agents.ceo.agent import CEOAgent
    from app.agents.strategy_updater.agent import StrategyUpdaterAgent

    market_agent = MarketResearchAgent(
        agent_id="market_research",
        market_data_provider=market_data,
        llm_provider=llm,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    news_agent = NewsResearchAgent(
        agent_id="news_research",
        news_provider=news,
        llm_provider=llm,
        memory_store=memory_store,
    )
    crisis_agent = CrisisRiskAgent(
        agent_id="crisis_risk",
        llm_provider=llm,
        news_provider=news,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    deep_looker = DeepLookerAgent(
        agent_id="deep_looker",
        llm_provider=llm,
        market_data_provider=market_data,
        news_provider=news,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    risk_manager = RiskManagerAgent(
        agent_id="risk_manager",
        llm_provider=llm,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    safety_manager = InvestmentSafetyManagerAgent(
        agent_id="investment_safety",
        llm_provider=llm,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    strategy_updater = StrategyUpdaterAgent(
        agent_id="strategy_updater",
        llm_provider=llm,
        memory_store=memory_store,
    )

    agent_registry.register(market_agent)
    agent_registry.register(news_agent)
    agent_registry.register(crisis_agent)
    agent_registry.register(deep_looker)
    agent_registry.register(risk_manager)
    agent_registry.register(safety_manager)
    agent_registry.register(strategy_updater)

    ceo = CEOAgent(
        agent_id="ceo",
        agent_registry=agent_registry,
        llm_provider=llm,
        memory_store=memory_store,
        event_publisher=event_bus.publish,
    )
    agent_registry.register(ceo)

    generation_manager = GenerationManager(
        memory_store=memory_store,
        event_bus=event_bus,
        agent_registry=agent_registry,
        market_data_provider=market_data,
        news_provider=news,
        llm_provider=llm,
        initial_capital=initial_capital,
    )

    runtime = SurvivalRuntime(
        memory_store=memory_store,
        event_bus=event_bus,
        generation_manager=generation_manager,
        agent_registry=agent_registry,
        market_data_provider=market_data,
        news_provider=news,
        execution_provider=AlpacaPaperExecutionProvider(
            api_key=api_key, api_secret=api_secret
        ),
        config=config,
    )

    components = {
        "market_data_provider": market_data,
        "news_provider": news,
        "llm_provider": llm,
        "execution_provider": "AlpacaPaperExecutionProvider (PAPER ONLY)",
        "generation_manager": generation_manager,
    }
    return BootstrapResult(runtime, agent_registry, config, components)


@dataclass
class LLMStack:
    """The assembled LLM stack: one router, one usage tracker."""

    router: Any
    usage_tracker: Any
    provider_type: str
    # "fallback" | "connected" | "configured" | "unavailable" | "failed"
    provider_state: str
    provider_detail: str = ""
    rebound_agents: int = 0

    def describe(self) -> str:
        """One human-readable line for the launcher."""
        return f"[llm] {self.provider_detail} (agents bound: {self.rebound_agents})"


def build_llm_stack(runtime: Any, memory_store: MemoryStore,
                    generation_manager: Any,
                    components: Optional[Dict[str, Any]] = None) -> LLMStack:
    """Build the LLM stack and make the router the provider agents call.

    This is the single owner of the real configuration path. The launcher
    (`scripts/run_local.py`) and the integration tests both call it, so the
    shipping configuration is exercised by tests instead of living inline in
    `main()`.

    Provider selection is environment-driven:
    `SURVIVALAI_LLM_PROVIDER` = lm_studio | ollama | gemini | claude (empty
    means the deterministic mock). Remote providers additionally need
    GEMINI_API_KEY / ANTHROPIC_API_KEY; when anything is missing or unreachable
    the stack degrades to the mock instead of failing to boot.

    Precondition: `generation_manager.memory_store` must be the same
    `memory_store` passed here (as `scripts/run_local.py` wires it), otherwise
    cost attribution would be written to a different store than the one the
    dashboard and generation reads use.
    """
    from app.services.llm.model_router import RouterLLMProvider
    from app.services.llm.usage_tracker import LLMUsageTracker

    # Estimated LLM costs are attributed to the active generation's operating
    # costs, so API spend shows up in the survival/death math.
    usage_tracker = LLMUsageTracker(
        memory_store=memory_store,
        generation_manager=generation_manager,
        generation_id_resolver=lambda: runtime.active_generation_id,
    )
    router = RouterLLMProvider(
        default_provider=(components or {}).get("llm_provider"),
        usage_tracker=usage_tracker,
        generation_id_resolver=lambda: runtime.active_generation_id,
    )

    provider_type = os.getenv("SURVIVALAI_LLM_PROVIDER", "").lower()
    state, detail = "fallback", "deterministic mock (no provider configured)"
    if provider_type in ("lm_studio", "ollama"):
        from app.services.llm.local_providers import build_local_provider
        try:
            local_provider = build_local_provider(
                provider_type,
                endpoint=os.getenv("SURVIVALAI_LLM_ENDPOINT"),
                model=os.getenv("SURVIVALAI_LLM_MODEL"),
            )
            if local_provider.health_check().get("available"):
                router.register_role_provider("general", local_provider)
                state, detail = "connected", f"local provider '{provider_type}' connected"
            else:
                state = "unavailable"
                detail = f"provider '{provider_type}' not reachable — using fallback"
        except Exception as e:
            state, detail = "failed", f"provider setup failed ({e}) — using fallback"
    elif provider_type in ("gemini", "claude"):
        result = router.configure_remote_provider(
            provider_type, model=os.getenv("SURVIVALAI_LLM_MODEL"),
        )
        if result.get("ok"):
            state = "configured"
            detail = f"remote provider '{provider_type}' configured (cost tracking on)"
        else:
            state = "failed"
            detail = f"remote provider setup failed: {result.get('error')} — using fallback"

    # Make the router the provider every agent actually calls, so routing,
    # remote providers and usage/cost tracking apply to real agent traffic.
    rebound = router.bind_to_agents(getattr(runtime, "agent_registry", None))
    if components is not None:
        components["llm_provider"] = router
    runtime.llm_provider = router

    return LLMStack(router=router, usage_tracker=usage_tracker,
                    provider_type=provider_type, provider_state=state,
                    provider_detail=detail, rebound_agents=rebound)


def _build_llm_provider():
    """Build the LLM provider from environment configuration.

    Falls back to the deterministic mock when no LLM credentials are set —
    agents then operate with deterministic analysis only (no fabrication).
    """
    provider_name = os.getenv("LLM_PROVIDER", "").lower()
    if provider_name in ("", "none", "mock"):
        return MockLLMProvider()
    # Real LLM adapters are a future step; until configured, stay deterministic.
    logger.warning(
        "LLM_PROVIDER '%s' has no adapter yet; using deterministic mock provider.",
        provider_name,
    )
    return MockLLMProvider()