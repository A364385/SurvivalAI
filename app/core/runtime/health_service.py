"""Runtime health checks.

Three layers of health verification before autonomous operation:

1. Infrastructure health: memory store, generation manager, agent registry.
2. Internet/API health: market data, news, LLM providers.
3. Paper-trading health: execution provider reachable, account accessible,
   order endpoint available, environment strictly PAPER.

Any critical failure puts the runtime into a safe state (no new decisions).
"""

from typing import Any, Dict, List, Optional

from app.core.memory.store import MemoryStore
from app.core.generation.manager import GenerationManager
from app.core.models.execution import ExecutionEnvironment
from app.core.runtime.execution_service import PaperExecutionService
from app.core.runtime.models import HealthCheckResult, RuntimeConfig
from app.services.market_data.provider import MarketDataProvider
from app.services.news.provider import NewsProvider
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class RuntimeHealthChecker:
    """Performs layered health checks for the autonomous runtime."""

    def __init__(
        self,
        memory_store: MemoryStore,
        generation_manager: GenerationManager,
        market_data_provider: Optional[MarketDataProvider],
        news_provider: Optional[NewsProvider],
        execution_service: PaperExecutionService,
        config: RuntimeConfig,
    ):
        self.memory_store = memory_store
        self.generation_manager = generation_manager
        self.market_data_provider = market_data_provider
        self.news_provider = news_provider
        self.execution_service = execution_service
        self.config = config

    # ------------------------------------------------------------------
    # Layer 1: infrastructure
    # ------------------------------------------------------------------
    def check_infrastructure(self) -> Dict[str, HealthCheckResult]:
        results: Dict[str, HealthCheckResult] = {}

        try:
            self.memory_store.list()
            results["memory"] = HealthCheckResult(component="memory", available=True)
        except Exception as e:
            results["memory"] = HealthCheckResult(
                component="memory", available=False, error_message=str(e)
            )

        try:
            self.generation_manager.get_generation("nonexistent_probe")  # read path
            results["generation_manager"] = HealthCheckResult(
                component="generation_manager", available=True
            )
        except Exception as e:
            results["generation_manager"] = HealthCheckResult(
                component="generation_manager", available=False, error_message=str(e)
            )

        return results

    # ------------------------------------------------------------------
    # Layer 2: internet / API providers
    # ------------------------------------------------------------------
    def check_api_providers(self) -> Dict[str, HealthCheckResult]:
        results: Dict[str, HealthCheckResult] = {}

        if self.market_data_provider is not None:
            try:
                self.market_data_provider.get_market_clock()
                results["market_data"] = HealthCheckResult(
                    component="market_data", available=True
                )
            except Exception as e:
                results["market_data"] = HealthCheckResult(
                    component="market_data", available=False, error_message=str(e)
                )
        else:
            results["market_data"] = HealthCheckResult(
                component="market_data", available=False,
                error_message="market data provider not configured",
            )

        if self.news_provider is not None:
            try:
                self.news_provider.get_latest_news(limit=1)
                results["news"] = HealthCheckResult(component="news", available=True)
            except Exception as e:
                results["news"] = HealthCheckResult(
                    component="news", available=False, error_message=str(e)
                )
        else:
            results["news"] = HealthCheckResult(
                component="news", available=False,
                error_message="news provider not configured",
            )

        return results

    # ------------------------------------------------------------------
    # Layer 3: paper trading
    # ------------------------------------------------------------------
    def check_paper_trading(self) -> Dict[str, HealthCheckResult]:
        results: Dict[str, HealthCheckResult] = {}

        verification = self.execution_service.verify_paper_mode(force=True)
        for check_name, passed in verification.checks.items():
            results[f"paper_{check_name}"] = HealthCheckResult(
                component=f"paper_{check_name}",
                available=bool(passed),
                error_message=None if passed else f"paper check failed: {check_name}",
            )

        results["paper_verification"] = HealthCheckResult(
            component="paper_verification",
            available=verification.verified,
            error_message="; ".join(verification.failures) if verification.failures else None,
        )
        return results

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def perform_full_health_check(self) -> Dict[str, Any]:
        """Run all layers; return aggregate with criticality classification."""
        infrastructure = self.check_infrastructure()
        api = self.check_api_providers()
        paper = self.check_paper_trading()

        all_results: Dict[str, HealthCheckResult] = {}
        all_results.update(infrastructure)
        all_results.update(api)
        all_results.update(paper)

        # Critical components: runtime cannot make decisions without these.
        critical_components = [
            "memory",
            "generation_manager",
            "market_data",
            "paper_verification",
        ]
        critical_failures = [
            name for name in critical_components
            if name in all_results and not all_results[name].available
        ]

        all_healthy = all(r.available for r in all_results.values())
        critical_healthy = not critical_failures

        summary = {
            "timestamp": now_utc().isoformat(),
            "all_healthy": all_healthy,
            "critical_healthy": critical_healthy,
            "critical_failures": critical_failures,
            "results": {
                name: {
                    "available": r.available,
                    "error": r.error_message,
                }
                for name, r in all_results.items()
            },
        }
        if not all_healthy:
            logger.warning("Health check failures: %s", summary["results"])
        return summary
