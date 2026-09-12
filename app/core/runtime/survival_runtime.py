"""SurvivalRuntime — central autonomous runtime service (Step 15).

Coordinates the full end-to-end paper-trading loop:

START -> HEALTH CHECK -> GENERATION START -> OBSERVE (market/news/crisis)
-> RESEARCH (market/news/crisis in parallel via CEO) -> DEEP ANALYSIS
-> RISK CHECK -> DECIDE -> PAPER EXECUTE (if approved) -> PORTFOLIO SYNC
-> INVESTMENT MONITORING -> LEARN -> SURVIVAL CHECK -> REPEAT
-> GENERATION DEATH -> TRANSITION -> SUCCESSOR -> GENERATION 2 START

Safety architecture:
- Paper trading only: every order passes PaperOnlyExecutionProvider and
  paper-mode verification; live trading is architecturally impossible.
- The CEO/Orchestrator runs the decision pipeline; the Risk Manager gate
  cannot be bypassed (BLOCKED never executes).
- Decision gating prevents blind trading every cycle.
- Idempotency for orders, generation creation, learning, costs, transitions.
- Failures are classified and recorded; transient outages pause new
  decisions instead of crashing the loop.
"""

import threading
import time
from typing import Any, Dict, List, Optional

from app.agents.registry import AgentRegistry
from app.core.event_bus.event_bus import EventBus
from app.core.generation.manager import GenerationManager
from app.core.memory.store import MemoryStore
from app.core.models.generation import DeathTrigger
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.models.risk import PortfolioRiskState, PositionExposure
from app.core.models.task import Task
from app.core.portfolio.synchronizer import PortfolioSynchronizer
from app.core.runtime.audit import AuditTrail
from app.core.runtime.cost_service import CostAccountingService
from app.core.runtime.decision_gate import DecisionGate
from app.core.safety.capital_protection import CapitalProtectionLayer
from app.core.runtime.watchdog import Watchdog
from app.core.runtime.execution_service import PaperExecutionService
from app.core.runtime.health_service import RuntimeHealthChecker
from app.core.runtime.idempotency import IdempotencyManager
from app.core.runtime.learning_service import ExperienceCollector, LearningCycle
from app.core.runtime.models import (
    CyclePhase,
    CycleRecord,
    FailureRecord,
    FailureType,
    HealthCheckResult,
    RuntimeConfig,
    RuntimeState,
    RuntimeStateSnapshot,
)
from app.core.runtime.monitoring_service import InvestmentMonitor, PortfolioMonitor
from app.core.runtime.recovery import RuntimeRecovery
from app.core.runtime.scheduler import RuntimeScheduler
from app.core.runtime.transition_service import GenerationTransitionService
from app.services.execution.provider import ExecutionProvider
from app.services.market_data.provider import MarketDataProvider
from app.services.news.provider import NewsProvider
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class SurvivalRuntime:
    """Central runtime service for autonomous SurvivalAI operation.

    Coordinates health checks, generation lifecycle, the autonomous cycle
    (observe -> research -> analyze -> risk -> decide -> execute -> monitor
    -> learn -> survival check), paper execution, monitoring, learning,
    cost accounting, and generation transitions.

    The runtime never executes live trades and never bypasses the CEO /
    Risk Manager pipeline.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        event_bus: EventBus,
        generation_manager: GenerationManager,
        agent_registry: AgentRegistry,
        market_data_provider: MarketDataProvider,
        news_provider: NewsProvider,
        execution_provider: ExecutionProvider,
        portfolio_synchronizer: Optional[PortfolioSynchronizer] = None,
        provider_health_checker=None,
        config: Optional[RuntimeConfig] = None,
    ):
        self.memory_store = memory_store
        self.event_bus = event_bus
        self.generation_manager = generation_manager
        self.agent_registry = agent_registry
        self.market_data_provider = market_data_provider
        self.news_provider = news_provider
        self.execution_provider = execution_provider
        self.portfolio_synchronizer = portfolio_synchronizer
        self.provider_health_checker = provider_health_checker
        self.config = config or RuntimeConfig()

        self.runtime_id = generate_id("runtime")
        self.current_state = RuntimeState.STARTING
        self.previous_state: Optional[RuntimeState] = None
        self.active_generation_id: Optional[str] = None
        self.current_cycle_id: Optional[str] = None
        self.cycle_phase: Optional[CyclePhase] = None
        self.completed_cycles = 0

        self._running = False
        self._lock = threading.RLock()
        self._consecutive_critical_failures = 0
        self._last_market_data: Dict[str, Any] = {}
        self._last_news: Dict[str, Any] = {}
        self._last_crisis: Dict[str, Any] = {}
        self._last_health: Dict[str, Any] = {}

        # --- composed services -------------------------------------------------
        self.capital_protection = CapitalProtectionLayer()
        self.audit = AuditTrail(self.memory_store)
        self.watchdog = Watchdog()
        self.watchdog.on_unsafe = self._on_watchdog_unsafe
        self._last_heartbeat: Optional[float] = None
        self.idempotency = IdempotencyManager(self.memory_store)
        self.recovery = RuntimeRecovery(self.memory_store, generation_manager)
        self.execution_service = PaperExecutionService(
            execution_provider=execution_provider,
            market_data_provider=market_data_provider,
            memory_store=self.memory_store,
            idempotency_manager=self.idempotency,
            config=self.config,
        )
        self.health_checker = RuntimeHealthChecker(
            memory_store=self.memory_store,
            generation_manager=generation_manager,
            market_data_provider=market_data_provider,
            news_provider=news_provider,
            execution_service=self.execution_service,
            config=self.config,
        )
        self.decision_gate = DecisionGate(self.memory_store, self.config)
        self.portfolio_monitor = PortfolioMonitor(
            execution_provider, self.memory_store, self.config
        )
        self.investment_monitor = InvestmentMonitor(
            memory_store=self.memory_store,
            market_data_provider=market_data_provider,
            safety_agent=self.agent("investment_safety"),
            config=self.config,
        )
        self.experience_collector = ExperienceCollector(
            memory_store=self.memory_store,
            market_data_provider=market_data_provider,
            config=self.config,
        )
        self.learning_cycle = LearningCycle(
            memory_store=self.memory_store,
            strategy_agent=self.agent("strategy_updater"),
            idempotency_manager=self.idempotency,
            config=self.config,
        )
        self.cost_service = CostAccountingService(
            memory_store=self.memory_store,
            idempotency_manager=self.idempotency,
            config=self.config,
        )
        self.transition_service = GenerationTransitionService(
            memory_store=self.memory_store,
            generation_manager=generation_manager,
            execution_service=self.execution_service,
            experience_collector=self.experience_collector,
            learning_cycle=self.learning_cycle,
            idempotency_manager=self.idempotency,
        )
        self.scheduler = RuntimeScheduler()
        self._register_scheduled_tasks()

        # State transition validation
        self._valid_transitions = {
            RuntimeState.STARTING: [RuntimeState.HEALTH_CHECK, RuntimeState.ERROR, RuntimeState.STOPPED],
            RuntimeState.HEALTH_CHECK: [RuntimeState.INITIALIZING_GENERATION, RuntimeState.ERROR, RuntimeState.STOPPED],
            RuntimeState.INITIALIZING_GENERATION: [RuntimeState.OBSERVING, RuntimeState.ERROR, RuntimeState.STOPPED],
            RuntimeState.OBSERVING: [RuntimeState.RESEARCHING, RuntimeState.MONITORING, RuntimeState.PAUSED, RuntimeState.ERROR],
            RuntimeState.RESEARCHING: [RuntimeState.ANALYZING, RuntimeState.MONITORING, RuntimeState.PAUSED, RuntimeState.ERROR],
            RuntimeState.ANALYZING: [RuntimeState.RISK_CHECK, RuntimeState.MONITORING, RuntimeState.PAUSED, RuntimeState.ERROR],
            RuntimeState.RISK_CHECK: [RuntimeState.DECIDING, RuntimeState.MONITORING, RuntimeState.PAUSED, RuntimeState.ERROR],
            RuntimeState.DECIDING: [RuntimeState.EXECUTING, RuntimeState.MONITORING, RuntimeState.PAUSED, RuntimeState.ERROR],
            RuntimeState.EXECUTING: [RuntimeState.MONITORING, RuntimeState.LEARNING, RuntimeState.PAUSED, RuntimeState.ERROR],
            RuntimeState.MONITORING: [RuntimeState.LEARNING, RuntimeState.SURVIVAL_CHECK, RuntimeState.PAUSED, RuntimeState.ERROR],
            RuntimeState.LEARNING: [RuntimeState.SURVIVAL_CHECK, RuntimeState.PAUSED, RuntimeState.ERROR],
            RuntimeState.SURVIVAL_CHECK: [RuntimeState.GENERATION_TRANSITION, RuntimeState.OBSERVING, RuntimeState.PAUSED, RuntimeState.ERROR],
            RuntimeState.GENERATION_TRANSITION: [RuntimeState.INITIALIZING_GENERATION, RuntimeState.STOPPED, RuntimeState.ERROR],
            RuntimeState.PAUSED: [RuntimeState.OBSERVING, RuntimeState.STOPPING, RuntimeState.ERROR],
            RuntimeState.STOPPING: [RuntimeState.STOPPED, RuntimeState.ERROR],
            RuntimeState.STOPPED: [],
            RuntimeState.ERROR: [RuntimeState.HEALTH_CHECK, RuntimeState.STOPPING, RuntimeState.STOPPED],
        }

    def _on_watchdog_unsafe(self, failed_checks) -> None:
        """Watchdog safety valve: pause instead of continuing blindly."""
        names = ", ".join(c.name for c in failed_checks)
        logger.warning("Watchdog triggered safe pause: %s", names)
        self.audit.record_event(
            self.active_generation_id or "unknown",
            "watchdog", "safe_pause", "PAUSED",
            {"failed_checks": names},
        )
        try:
            self.pause()
        except Exception as e:
            logger.error("Watchdog pause failed: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def agent(self, agent_id: str):
        """Retrieve an agent from the registry (None if absent/disabled)."""
        retrieved = self.agent_registry.retrieve(agent_id)
        if retrieved is None or not self.agent_registry.check_enabled(agent_id):
            return None
        return retrieved

    def _register_scheduled_tasks(self) -> None:
        cfg = self.config
        self.scheduler.register(
            "portfolio_sync", cfg.portfolio_sync_interval_seconds, self._scheduled_portfolio_sync
        )
        self.scheduler.register(
            "investment_monitoring", cfg.investment_monitor_interval_seconds, self._scheduled_investment_monitoring
        )
        self.scheduler.register(
            "learning", cfg.learning_interval_seconds, self._scheduled_learning
        )
        self.scheduler.register(
            "health_check", cfg.health_check_interval_seconds, self._scheduled_health_check
        )
        self.scheduler.register(
            "cost_accounting", cfg.learning_interval_seconds, self._scheduled_cost_accounting
        )

    # ------------------------------------------------------------------
    # Lifecycle: start / stop / pause / resume
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Start the autonomous runtime (health checks -> generation -> cycle)."""
        with self._lock:
            if self._running:
                logger.warning("Runtime is already running")
                return False
            self._running = True

        try:
            # Attempt recovery from a previous process (restart/recovery).
            self._recover_from_restart()

            if not self._perform_health_check():
                self._transition_to(RuntimeState.ERROR)
                self._running = False
                return False

            if not self._initialize_generation():
                self._transition_to(RuntimeState.ERROR)
                self._running = False
                return False

            self._transition_to(RuntimeState.OBSERVING)
            self._run_autonomous_cycle()
            return True

        except Exception as e:
            logger.error(f"Failed to start runtime: {e}")
            self._transition_to(RuntimeState.ERROR)
            self._running = False
            return False

    def stop(self) -> bool:
        """Stop the autonomous runtime gracefully."""
        with self._lock:
            if not self._running:
                logger.warning("Runtime is not running")
                return False
            self._running = False

        try:
            self._transition_to(RuntimeState.STOPPING)
            self._transition_to(RuntimeState.STOPPED)
            logger.info("Runtime stopped successfully")
            return True
        except Exception as e:
            logger.error(f"Error stopping runtime: {e}")
            self._transition_to(RuntimeState.ERROR)
            return False

    def pause(self) -> bool:
        """Pause the runtime (safe pause: no new decisions)."""
        with self._lock:
            if not self._running:
                return False
            return self._transition_to(RuntimeState.PAUSED)

    def resume(self) -> bool:
        """Resume the runtime from PAUSED."""
        with self._lock:
            if not self._running:
                return False
            if self.current_state != RuntimeState.PAUSED:
                return False
            return self._transition_to(RuntimeState.OBSERVING)

    # ------------------------------------------------------------------
    # Restart / recovery
    # ------------------------------------------------------------------
    def _recover_from_restart(self) -> None:
        """Recover state after a process restart.

        Restores the active generation, reconciles portfolio and orders, and
        identifies unfinished operations. Never assumes the previous process
        completed anything.
        """
        try:
            active_gen_id = self.recovery.recover_active_generation()
            if active_gen_id:
                gen = self.generation_manager.get_generation(active_gen_id)
                if gen and gen.lifecycle_state.value in ("ACTIVE", "PAUSED"):
                    self.active_generation_id = active_gen_id
                    logger.info(
                        "Recovered active generation %s from previous runtime",
                        active_gen_id,
                    )

            unfinished = self.recovery.identify_unfinished_operations()
            if unfinished:
                logger.warning(
                    "Found %d unfinished operation(s) from previous runtime; "
                    "they will not be blindly resumed.",
                    len(unfinished),
                )

            # Reconcile orders tracked at the provider (adopts orphaned orders).
            self.execution_service.reconcile_open_orders()
        except Exception as e:
            logger.error("Restart recovery failed (continuing safely): %s", e)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def _transition_to(self, new_state: RuntimeState) -> bool:
        """Transition to a new state with validation."""
        with self._lock:
            valid_transitions = self._valid_transitions.get(self.current_state, [])
            if new_state not in valid_transitions:
                logger.error(f"Invalid state transition: {self.current_state} -> {new_state}")
                return False

            self.previous_state = self.current_state
            self.current_state = new_state

            snapshot = RuntimeStateSnapshot(
                runtime_id=self.runtime_id,
                generation_id=self.active_generation_id,
                current_state=self.current_state,
                previous_state=self.previous_state,
                timestamp=now_utc(),
                cycle_id=self.current_cycle_id,
                cycle_phase=self.cycle_phase,
                health_status={
                    name: bool(info.get("available"))
                    for name, info in (self._last_health.get("results") or {}).items()
                } if self._last_health else {},
            )
            self._store_state_snapshot(snapshot)

            logger.info(f"Runtime state transition: {self.previous_state} -> {self.current_state}")
            return True

    # ------------------------------------------------------------------
    # Autonomous cycle
    # ------------------------------------------------------------------
    def _run_autonomous_cycle(self):
        """Run the main autonomous loop with configurable interval."""
        while self._running:
            try:
                if self.config.max_cycles and self.completed_cycles >= self.config.max_cycles:
                    logger.info("Reached max_cycles (%d); stopping.", self.config.max_cycles)
                    self.stop()
                    break

                if self.current_state == RuntimeState.PAUSED:
                    time.sleep(min(self.config.cycle_interval_seconds, 1.0))
                    continue

                if self.current_state == RuntimeState.ERROR:
                    logger.error("Runtime in ERROR state, stopping")
                    self.stop()
                    break

                self.current_cycle_id = generate_id("cycle")
                cycle_record = CycleRecord(
                    cycle_id=self.current_cycle_id,
                    generation_id=self.active_generation_id or "unknown",
                    start_timestamp=now_utc(),
                    end_timestamp=None,
                    phase=CyclePhase.OBSERVE,
                )

                self._execute_cycle(cycle_record)
                self.completed_cycles += 1

                if not self._running:
                    break

                time.sleep(self.config.cycle_interval_seconds)

            except Exception as e:
                logger.error(f"Error in autonomous cycle: {e}")
                self._record_failure(FailureType.CRITICAL, "autonomous_cycle", str(e))
                self._consecutive_failure_bump()

    def _consecutive_failure_bump(self) -> None:
        self._consecutive_critical_failures += 1
        if (
            self._consecutive_critical_failures
            >= self.config.max_consecutive_critical_failures
        ):
            logger.error(
                "Too many consecutive critical failures (%d); stopping runtime.",
                self._consecutive_critical_failures,
            )
            self.stop()

    def _execute_cycle(self, cycle_record: CycleRecord):
        """Execute a single autonomous cycle through all phases."""
        try:
            # Watchdog: pause instead of continuing blindly when unsafe.
            self._last_heartbeat = time.time()
            watchdog_result = self.watchdog.run_checks(extra={
                "last_heartbeat": self._last_heartbeat,
                "provider_health": self._last_health or None,
            })
            if watchdog_result.get("should_pause"):
                cycle_record.warnings.append("watchdog_safe_pause")
                self._safe_pause(cycle_record)
                return

            # ---------------- OBSERVE ----------------
            self.cycle_phase = CyclePhase.OBSERVE
            self._transition_to(RuntimeState.OBSERVING)
            market_data = self._collect_market_data()
            self._collect_news()
            self._collect_crisis_data()
            cycle_record.triggered_actions.append("observe")

            # Scheduled background tasks (portfolio sync, health, etc.)
            scheduled = self.scheduler.run_due_tasks()
            for result in scheduled:
                if result.get("status") == "FAILED":
                    cycle_record.warnings.append(
                        f"scheduled task failed: {result.get('task')}"
                    )

            # Refresh health snapshot for gating decisions.
            self._last_health = self.health_checker.perform_full_health_check()
            providers_healthy = bool(self._last_health.get("critical_healthy"))

            # Safe pause: critical infrastructure unavailable -> no new decisions.
            if not providers_healthy:
                cycle_record.warnings.append("safe_pause: critical health failures")
                self._safe_pause(cycle_record)
                return

            # ---------------- RESEARCH / ANALYZE / RISK / DECIDE ----------------
            # The CEO orchestrates research (parallel), deep analysis, and the
            # Risk Manager gate internally; the runtime gates and dispatches.
            gate_result = self._evaluate_decision_gate(market_data, providers_healthy)
            cycle_record.agent_results["decision_gate"] = {
                "allowed": gate_result["allowed"],
                "reason": gate_result["reason"],
            }

            if gate_result["allowed"]:
                self.cycle_phase = CyclePhase.RESEARCH
                self._transition_to(RuntimeState.RESEARCHING)
                cycle_record.triggered_actions.append("research_dispatch")

                self.cycle_phase = CyclePhase.ANALYZE
                self._transition_to(RuntimeState.ANALYZING)

                self.cycle_phase = CyclePhase.RISK_CHECK
                self._transition_to(RuntimeState.RISK_CHECK)

                self.cycle_phase = CyclePhase.DECIDE
                self._transition_to(RuntimeState.DECIDING)
                decision = self._execute_decision_pipeline(gate_result)
                cycle_record.decision_result = decision
                cycle_record.triggered_actions.append("decision")

                # ---------------- EXECUTE (paper only, if approved) ----------------
                if decision.get("decision") == "INVEST":
                    self.cycle_phase = CyclePhase.EXECUTE
                    self._transition_to(RuntimeState.EXECUTING)
                    execution_result = self._execute_paper_execution(decision)
                    cycle_record.execution_result = execution_result
                    cycle_record.triggered_actions.append("paper_execution")
            else:
                logger.info("Decision gate blocked new decision: %s", gate_result["reason"])

            # ---------------- MONITOR ----------------
            self.cycle_phase = CyclePhase.MONITOR
            self._transition_to(RuntimeState.MONITORING)
            portfolio_state = self._monitor_portfolio()
            cycle_record.portfolio_state = portfolio_state
            investment_results = self._monitor_existing_investments()
            cycle_record.agent_results["investment_monitoring"] = investment_results

            # ---------------- LEARN ----------------
            self.cycle_phase = CyclePhase.LEARN
            self._transition_to(RuntimeState.LEARNING)
            experiences = self._collect_experiences()
            cycle_record.agent_results["experiences"] = experiences

            # ---------------- SURVIVAL CHECK ----------------
            self.cycle_phase = CyclePhase.SURVIVAL_CHECK
            self._transition_to(RuntimeState.SURVIVAL_CHECK)
            survival_result = self._check_survival()
            cycle_record.survival_state = survival_result

            cycle_record.end_timestamp = now_utc()
            cycle_record.status = "COMPLETED"
            self._store_cycle_record(cycle_record)

            # Handle generation death (transition pipeline).
            if survival_result.get("death_triggered"):
                self._handle_generation_death(survival_result)

            self._consecutive_critical_failures = 0

        except Exception as e:
            logger.error(f"Error executing cycle: {e}")
            cycle_record.status = "FAILED"
            cycle_record.errors.append(str(e))
            self._store_cycle_record(cycle_record)
            self._record_failure(FailureType.CRITICAL, "cycle_execution", str(e))
            self._consecutive_failure_bump()

    def _safe_pause(self, cycle_record: CycleRecord) -> None:
        """Enter safe pause: monitoring continues, new decisions stop."""
        cycle_record.warnings.append("entering_safe_pause")
        self._record_failure(
            FailureType.RECOVERABLE, "safe_pause", "critical health check failed"
        )
        # Monitoring of existing positions continues where data is available.
        portfolio_state = self._monitor_portfolio()
        cycle_record.portfolio_state = portfolio_state
        cycle_record.end_timestamp = now_utc()
        cycle_record.status = "SAFE_PAUSED"
        self._store_cycle_record(cycle_record)
        self._transition_to(RuntimeState.PAUSED)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------
    def _perform_health_check(self) -> bool:
        """Perform comprehensive layered health check. Returns bool for compat."""
        self._transition_to(RuntimeState.HEALTH_CHECK)
        self._last_health = self.health_checker.perform_full_health_check()
        if not self._last_health.get("critical_healthy"):
            logger.error(f"Health check failed: {self._last_health}")
            return False
        logger.info("Health check passed")
        return True

    def _check_provider_health(self, provider_type: str) -> HealthCheckResult:
        """Check health of a specific provider (compatibility API)."""
        try:
            if provider_type == "market_data":
                self.market_data_provider.get_market_clock()
            elif provider_type == "news":
                self.news_provider.get_latest_news(limit=1)
            elif provider_type == "execution":
                verification = self.execution_service.verify_paper_mode(force=True)
                return HealthCheckResult(
                    component=provider_type,
                    available=verification.verified,
                    error_message="; ".join(verification.failures) or None,
                )

            return HealthCheckResult(component=provider_type, available=True)

        except Exception as e:
            return HealthCheckResult(
                component=provider_type,
                available=False,
                error_message=str(e),
            )

    def _check_memory_health(self) -> HealthCheckResult:
        try:
            self.memory_store.list()
            return HealthCheckResult(component="memory", available=True)
        except Exception as e:
            return HealthCheckResult(
                component="memory", available=False, error_message=str(e)
            )

    def _check_generation_manager_health(self) -> HealthCheckResult:
        try:
            self.generation_manager.get_generation("nonexistent_probe")
            return HealthCheckResult(component="generation_manager", available=True)
        except Exception as e:
            return HealthCheckResult(
                component="generation_manager", available=False, error_message=str(e)
            )

    # ------------------------------------------------------------------
    # Generation initialization
    # ------------------------------------------------------------------
    def _initialize_generation(self) -> bool:
        """Initialize or load active generation (idempotent)."""
        self._transition_to(RuntimeState.INITIALIZING_GENERATION)

        try:
            # Try to load existing active generation (recovery path).
            if self.active_generation_id:
                gen = self.generation_manager.get_generation(self.active_generation_id)
                if gen and gen.lifecycle_state.value in ["ACTIVE", "PAUSED"]:
                    logger.info(
                        "Reusing active generation %s", self.active_generation_id
                    )
                    return True

            # Idempotent generation creation for this runtime boot.
            operation_id = f"boot_{self.runtime_id}"
            if not self.idempotency.begin(
                "generation_creation",
                operation_id,
                generation_id=self.active_generation_id or "new",
            ):
                logger.warning("Duplicate generation creation blocked for boot")
                return self.active_generation_id is not None

            gen = self.generation_manager.create_generation()
            success = self.generation_manager.start_generation(gen.generation_id)

            if success:
                self.active_generation_id = gen.generation_id
                self.idempotency.complete(
                    "generation_creation",
                    operation_id,
                    {"generation_id": gen.generation_id},
                )
                logger.info(f"Initialized generation: {self.active_generation_id}")
                return True

            self.idempotency.fail(
                "generation_creation", operation_id, "start_generation returned False"
            )
            return False

        except Exception as e:
            logger.error(f"Failed to initialize generation: {e}")
            return False

    # ------------------------------------------------------------------
    # Observation (market data / news / crisis)
    # ------------------------------------------------------------------
    def _collect_market_data(self) -> Dict[str, Any]:
        """Collect current market data with staleness detection."""
        try:
            clock = self.market_data_provider.get_market_clock()
            data = {
                "clock": clock,
                "timestamp": now_utc(),
                "fresh": True,
            }
            # Staleness detection: provider timestamp age vs threshold.
            clock_ts = getattr(clock, "timestamp", None)
            if clock_ts is not None:
                age = (now_utc() - clock_ts).total_seconds()
                data["data_age_seconds"] = age
                data["fresh"] = age <= self.config.stale_data_threshold_seconds
            self._last_market_data = data
            return data
        except Exception as e:
            logger.error(f"Failed to collect market data: {e}")
            self._record_failure(FailureType.TRANSIENT, "market_data", str(e))
            self._last_market_data = {"timestamp": now_utc(), "fresh": False, "error": str(e)}
            return self._last_market_data

    def _collect_news(self) -> Dict[str, Any]:
        """Collect recent news through the news provider."""
        try:
            news = self.news_provider.get_latest_news(limit=10)
            data = {"news": news, "timestamp": now_utc(), "count": len(news)}
            self._last_news = data
            return data
        except Exception as e:
            logger.error(f"Failed to collect news: {e}")
            self._record_failure(FailureType.TRANSIENT, "news", str(e))
            self._last_news = {"news": [], "timestamp": now_utc(), "error": str(e)}
            return self._last_news

    def _collect_crisis_data(self) -> Dict[str, Any]:
        """Crisis data is collected through the Crisis Risk Agent pipeline."""
        return {"timestamp": now_utc()}

    # ------------------------------------------------------------------
    # Decision gating + CEO pipeline
    # ------------------------------------------------------------------
    def _evaluate_decision_gate(
        self, market_data: Dict[str, Any], providers_healthy: bool
    ) -> Dict[str, Any]:
        """Evaluate the decision gate for the primary watched symbol."""
        symbol = self.config.watched_symbols[0] if self.config.watched_symbols else "AAPL"

        portfolio = self._portfolio_risk_state()
        open_orders = []
        try:
            open_orders = self.execution_provider.get_open_orders()
        except Exception:
            pass

        market_fresh = bool(market_data.get("fresh", False))
        gate = self.decision_gate.evaluate(
            asset=symbol,
            generation_id=self.active_generation_id or "unknown",
            market_data_fresh=market_fresh,
            providers_healthy=providers_healthy,
            crisis_severity=self._last_crisis_severity(),
            portfolio_value=portfolio.portfolio_value if portfolio else None,
            available_cash=portfolio.available_cash if portfolio else None,
            open_position_count=len(portfolio.positions) if portfolio else None,
            open_order_count=len(open_orders),
        )
        return {"allowed": gate.allowed, "reason": gate.reason, "asset": symbol}

    def _portfolio_risk_state(self) -> Optional[PortfolioRiskState]:
        """Build a PortfolioRiskState from the live paper account."""
        try:
            account = self.execution_provider.get_account()
            positions = self.execution_provider.get_positions()
        except Exception as e:
            logger.error("Failed to fetch portfolio state: %s", e)
            return None

        exposures = [
            PositionExposure(
                symbol=p.symbol,
                market_value=p.market_value,
            )
            for p in positions
        ]
        return PortfolioRiskState(
            portfolio_value=account.equity,
            available_cash=account.cash,
            positions=exposures,
            historical_peak_value=account.equity,
            timestamp=now_utc(),
        )

    def _execute_decision_pipeline(self, gate_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run the CEO orchestration pipeline for a gated candidate."""
        symbol = gate_result.get("asset", "AAPL")
        generation_id = self.active_generation_id or "unknown"

        portfolio = self._portfolio_risk_state()
        if portfolio is None:
            return {
                "decision": "INSUFFICIENT_DATA",
                "reason": "portfolio state unavailable",
            }

        position_value = portfolio.portfolio_value * self.config.proposed_position_pct

        ceo_task = Task(
            task_id=generate_id("task"),
            requesting_agent="SurvivalRuntime",
            target_agent="ceo",
            task_type="orchestration",
            priority=1,
            input_data={
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": symbol,
                "objective": f"Evaluate paper investment candidate {symbol} for survival-optimized portfolio",
                "generation_id": generation_id,
                "requested_position_size": position_value,
                "context": {
                    "portfolio_state": portfolio,
                },
            },
            created_at=now_utc(),
        )

        try:
            result = self._run_agent("ceo", ceo_task)
        except Exception as e:
            logger.error(f"Decision pipeline failed: {e}")
            self._record_failure(FailureType.CRITICAL, "ceo", str(e))
            return {"decision": "FAILED", "reason": str(e)}

        analysis = result.analysis or {}
        proposal = analysis.get("decision_proposal") or {}
        decision_type = proposal.get("decision_type") or analysis.get("decision") or "INSUFFICIENT_DATA"

        # Record the decision for cooldowns and the audit trail.
        decision_id = proposal.get("decision_id") or generate_id("decision")
        self.decision_gate.record_decision(symbol, str(decision_type), str(decision_id))

        # --- CAPITAL PROTECTION LAYER (deterministic, above Risk Manager) ---
        # Runs after the CEO/Risk Manager pipeline and before execution. AI
        # output cannot pass it; only broker/account numbers decide.
        proposed_value = proposal.get("position_size")
        if not isinstance(proposed_value, (int, float)) or proposed_value <= 0:
            proposed_value = portfolio.portfolio_value * self.config.proposed_position_pct
        protection = self.capital_protection.evaluate(
            symbol=symbol,
            proposed_position_value=float(proposed_value),
            portfolio_value=portfolio.portfolio_value,
            available_cash=portfolio.available_cash,
            current_positions=[
                {"symbol": p.symbol, "market_value": p.market_value}
                for p in portfolio.positions
            ],
            current_drawdown=0.0,  # drawdown tracked via survival system below
        )

        self._store_decision_record(
            decision_id=str(decision_id),
            generation_id=generation_id,
            asset=symbol,
            decision_type=str(decision_type),
            proposal=proposal,
            run_id=analysis.get("run_id"),
        )

        decision_payload = {
            "decision": str(decision_type),
            "decision_id": str(decision_id),
            "run_id": analysis.get("run_id"),
            "reason": proposal.get("decision_reason", ""),
            "confidence": proposal.get("confidence", 0.0),
            "position_size": proposal.get("position_size"),
            "capital_protection": protection.to_dict(),
        }

        # Hard veto: protection failure downgrades any INVEST decision.
        if str(decision_type) == "INVEST" and not protection.passed:
            logger.warning(
                "Capital protection BLOCKED invest decision %s: %s",
                decision_id, "; ".join(protection.reasons),
            )
            decision_payload["decision"] = "DO_NOT_INVEST"
            decision_payload["blocked_by"] = "capital_protection"
            decision_payload["original_decision"] = str(decision_type)

        self.audit.record_decision(
            generation_id=generation_id,
            decision_id=str(decision_id),
            decision_type=str(decision_payload["decision"]),
            confidence=float(decision_payload.get("confidence", 0.0) or 0.0),
            risk_result=str(proposal.get("risk_assessment", "UNKNOWN")),
            capital_protection_result=("PASS" if protection.passed else "BLOCKED"),
            execution_result=None,
            sources=[str(r) for r in (analysis.get("agent_results") or {}).keys()] if isinstance(analysis.get("agent_results"), dict) else [],
            output=proposal,
        )

        return decision_payload

    def _store_decision_record(
        self,
        decision_id: str,
        generation_id: str,
        asset: str,
        decision_type: str,
        proposal: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> None:
        """Persist a DecisionRecord for the audit trail."""
        record = MemoryRecord(
            memory_id=f"decision_{decision_id}",
            memory_type=MemoryType.DECISION,
            generation_id=generation_id,
            timestamp=now_utc(),
            source_agent="SurvivalRuntime",
            importance=10,
            content={
                "decision_id": decision_id,
                "run_id": run_id,
                "decision_type": decision_type,
                "asset": asset,
                "proposal": proposal,
                "timestamp": now_utc().isoformat(),
            },
            metadata={},
        )
        self.memory_store.save(record)

    def _run_agent(self, agent_name: str, task: Task):
        """Run an agent by registry id using the standard Task contract."""
        agent = self.agent_registry.retrieve(agent_name)
        if agent is None:
            raise ValueError(f"Agent not found: {agent_name}")
        if not self.agent_registry.check_enabled(agent_name):
            raise ValueError(f"Agent is disabled: {agent_name}")
        return agent.process_task(task)

    # ------------------------------------------------------------------
    # Paper execution
    # ------------------------------------------------------------------
    def _execute_paper_execution(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an approved decision through the paper execution service.

        Final execution gate: Capital Protection is re-checked at execution
        time with live account numbers — AI cannot slide an order past it.
        """
        if str(decision.get("decision")) != "INVEST":
            return {"status": "SKIPPED", "reason": f"decision={decision.get('decision')}"}

        symbol = self.config.watched_symbols[0] if self.config.watched_symbols else "AAPL"
        position_value = decision.get("position_size")
        if not position_value or position_value <= 0:
            portfolio = self._portfolio_risk_state()
            position_value = (
                portfolio.portfolio_value * self.config.proposed_position_pct
                if portfolio else 0.0
            )
        if position_value <= 0:
            return {"status": "SKIPPED", "reason": "no position value"}

        # Final deterministic gate (defense in depth): re-evaluate with the
        # live account state right before submission.
        portfolio = self._portfolio_risk_state()
        if portfolio is not None:
            protection = self.capital_protection.evaluate(
                symbol=symbol,
                proposed_position_value=float(position_value),
                portfolio_value=portfolio.portfolio_value,
                available_cash=portfolio.available_cash,
                current_positions=[
                    {"symbol": p.symbol, "market_value": p.market_value}
                    for p in portfolio.positions
                ],
            )
            if not protection.passed:
                logger.warning(
                    "Execution-time capital protection blocked order: %s",
                    "; ".join(protection.reasons),
                )
                self.audit.record_event(
                    self.active_generation_id or "unknown",
                    "execution_gate", f"order:{symbol}", "BLOCKED",
                    {"reasons": protection.reasons},
                )
                return {
                    "status": "BLOCKED",
                    "reason": "capital_protection",
                    "details": protection.reasons,
                }

        result = self.execution_service.execute_decision(
            decision_id=str(decision.get("decision_id", generate_id("decision"))),
            symbol=symbol,
            position_value=float(position_value),
            generation_id=self.active_generation_id or "unknown",
            thesis=str(decision.get("reason", "")),
        )
        self.audit.record_event(
            self.active_generation_id or "unknown",
            "execution", f"order:{symbol}", str(result.get("status", "UNKNOWN")),
            {"decision_id": decision.get("decision_id")},
        )
        return result

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    def _monitor_portfolio(self) -> Dict[str, Any]:
        """Synchronize portfolio state from the paper provider."""
        result = self.portfolio_monitor.synchronize()
        if result.get("status") == "SYNCED":
            # Keep generation capital aligned with account equity.
            if self.active_generation_id:
                try:
                    self.generation_manager.update_capital(
                        self.active_generation_id, float(result.get("equity", 0.0))
                    )
                except Exception as e:
                    logger.warning("Capital sync failed: %s", e)
        # Feed the dashboard live charts (equity / P/L / drawdown / cash).
        self._publish_dashboard_point(result)
        return result

    def _publish_dashboard_point(self, sync_result: Dict[str, Any]) -> None:
        """Push a portfolio point to the dashboard state hub if attached."""
        dashboard_state = getattr(self, "dashboard_state", None)
        if dashboard_state is None:
            return
        try:
            equity = float(sync_result.get("equity", 0.0) or 0.0)
            cash = float(sync_result.get("cash", 0.0) or 0.0)
            realized = float(sync_result.get("realized_pl", 0.0) or 0.0)
            unrealized = float(sync_result.get("unrealized_pl", 0.0) or 0.0)
            if equity > 0:
                dashboard_state.record_portfolio_point(
                    equity=equity, cash=cash,
                    realized_pl=realized, unrealized_pl=unrealized,
                )
                dashboard_state.broadcast("portfolio", {
                    "equity": equity, "cash": cash,
                    "pl": round(realized + unrealized, 2),
                    "generation_id": self.active_generation_id,
                })
        except Exception as e:
            logger.debug("Dashboard publish skipped: %s", e)

    def _monitor_existing_investments(self) -> List[Dict[str, Any]]:
        """Run Investment Safety Manager for open investments."""
        if not self.active_generation_id:
            return []
        return self.investment_monitor.monitor_open_investments(
            self.active_generation_id
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _collect_experiences(self) -> List[Dict[str, Any]]:
        """Collect experiences from executed decisions."""
        if not self.active_generation_id:
            return []
        return self.experience_collector.collect_experiences(
            self.active_generation_id
        )

    def _scheduled_learning(self) -> Dict[str, Any]:
        if not self.active_generation_id:
            return {"status": "SKIPPED", "reason": "no active generation"}
        return self.learning_cycle.run_learning_cycle(self.active_generation_id)

    def _scheduled_portfolio_sync(self) -> Dict[str, Any]:
        return self.portfolio_monitor.synchronize()

    def _scheduled_health_check(self) -> Dict[str, Any]:
        self._last_health = self.health_checker.perform_full_health_check()
        return {
            "status": "COMPLETED",
            "critical_healthy": self._last_health.get("critical_healthy"),
        }

    def _scheduled_investment_monitoring(self) -> Dict[str, Any]:
        if not self.active_generation_id:
            return {"status": "SKIPPED", "reason": "no active generation"}
        results = self.investment_monitor.monitor_open_investments(
            self.active_generation_id
        )
        return {"status": "COMPLETED", "monitored": len(results)}

    def _scheduled_cost_accounting(self) -> Dict[str, Any]:
        if not self.active_generation_id:
            return {"status": "SKIPPED", "reason": "no active generation"}
        return self.cost_service.apply_configured_costs(self.active_generation_id)

    # ------------------------------------------------------------------
    # Survival
    # ------------------------------------------------------------------
    def _check_survival(self) -> Dict[str, Any]:
        """Check survival conditions for the active generation."""
        try:
            if not self.active_generation_id:
                return {"death_triggered": False}

            gen = self.generation_manager.get_generation(self.active_generation_id)
            if gen is None or gen.lifecycle_state.value in ("DEAD", "DYING"):
                return {
                    "death_triggered": gen is not None,
                    "already_dead": True,
                    "trigger": gen.death_trigger.value if gen and gen.death_trigger else None,
                }

            trigger = self.generation_manager.check_death_conditions(self.active_generation_id)
            death_triggered = trigger is not None

            return {
                "death_triggered": death_triggered,
                "trigger": trigger.value if trigger else None,
                "timestamp": now_utc().isoformat(),
            }
        except Exception as e:
            logger.error(f"Survival check failed: {e}")
            self._record_failure(FailureType.CRITICAL, "survival_check", str(e))
            return {"death_triggered": False}

    def _handle_generation_death(self, survival_result: Dict[str, Any]):
        """Handle generation death through the transition service."""
        try:
            if survival_result.get("already_dead"):
                # Death was already processed (e.g. by update_capital); run
                # the transition pipeline if not yet done.
                raw_trigger = survival_result.get("trigger") or "CAPITAL_DEPLETED"
                trigger = DeathTrigger(normalize_trigger(raw_trigger))
            else:
                trigger = DeathTrigger(survival_result["trigger"])

            self._transition_to(RuntimeState.GENERATION_TRANSITION)

            pipeline = self.transition_service.handle_generation_death(
                generation_id=self.active_generation_id,
                trigger=trigger,
                reason=survival_result.get("reason")
                or f"Survival condition triggered: {trigger.value}",
            )
            logger.info(
                "Generation transition pipeline status: %s", pipeline.get("status")
            )

            # If a validated successor exists, start it; otherwise stay safe.
            successor = self.transition_service.create_successor_generation(
                self.active_generation_id
            )
            if successor and successor.get("started"):
                self.active_generation_id = successor["successor_generation_id"]
                self._transition_to(RuntimeState.INITIALIZING_GENERATION)
                self._transition_to(RuntimeState.OBSERVING)
            else:
                logger.warning(
                    "No successor started; generation remains SUCCESSOR_PENDING."
                )
                self._transition_to(RuntimeState.STOPPING)
                self._running = False

        except Exception as e:
            logger.error(f"Generation death handling failed: {e}")
            self._record_failure(FailureType.CRITICAL, "generation_death", str(e))

    def _last_crisis_severity(self) -> Optional[str]:
        """Extract the highest crisis severity from the last crisis research."""
        crisis_result = self._last_crisis.get("result")
        if crisis_result is None:
            return None
        try:
            events = (crisis_result.analysis or {}).get("events", []) or []
            severities = [
                str(e.get("severity", "")).upper()
                for e in events
                if isinstance(e, dict)
            ]
            for level in ("CRITICAL", "SEVERE", "HIGH"):
                if level in severities:
                    return level
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _store_state_snapshot(self, snapshot: RuntimeStateSnapshot):
        try:
            record = MemoryRecord(
                memory_id=f"runtime_state_{self.runtime_id}_{snapshot.timestamp.isoformat()}",
                memory_type=MemoryType.FACT,
                generation_id=self.active_generation_id or "unknown",
                timestamp=now_utc(),
                source_agent="SurvivalRuntime",
                importance=6,
                content={
                    "runtime_id": snapshot.runtime_id,
                    "generation_id": snapshot.generation_id,
                    "current_state": snapshot.current_state.value,
                    "previous_state": snapshot.previous_state.value if snapshot.previous_state else None,
                    "timestamp": snapshot.timestamp.isoformat(),
                    "cycle_id": snapshot.cycle_id,
                    "cycle_phase": snapshot.cycle_phase.value if snapshot.cycle_phase else None,
                    "health_status": snapshot.health_status,
                    "error_count": snapshot.error_count,
                    "last_error": snapshot.last_error,
                    "last_error_timestamp": snapshot.last_error_timestamp,
                },
                metadata={},
            )
            self.memory_store.save(record)
        except Exception as e:
            logger.error(f"Failed to store state snapshot: {e}")

    def _store_cycle_record(self, cycle_record: CycleRecord):
        try:
            record = MemoryRecord(
                memory_id=f"cycle_{cycle_record.cycle_id}",
                memory_type=MemoryType.FACT,
                generation_id=cycle_record.generation_id,
                timestamp=now_utc(),
                source_agent="SurvivalRuntime",
                importance=8,
                content={
                    "cycle_id": cycle_record.cycle_id,
                    "generation_id": cycle_record.generation_id,
                    "start_timestamp": cycle_record.start_timestamp.isoformat(),
                    "end_timestamp": cycle_record.end_timestamp.isoformat() if cycle_record.end_timestamp else None,
                    "phase": cycle_record.phase.value,
                    "triggered_actions": cycle_record.triggered_actions,
                    "agent_results": _safe_serialize(cycle_record.agent_results),
                    "decision_result": cycle_record.decision_result,
                    "execution_result": cycle_record.execution_result,
                    "portfolio_state": cycle_record.portfolio_state,
                    "survival_state": cycle_record.survival_state,
                    "errors": cycle_record.errors,
                    "warnings": cycle_record.warnings,
                    "status": cycle_record.status,
                },
                metadata={},
            )
            self.memory_store.save(record)
        except Exception as e:
            logger.error(f"Failed to store cycle record: {e}")

    def _record_failure(
        self,
        failure_type: FailureType,
        component: str,
        error_message: str,
    ):
        """Record a classified failure."""
        try:
            failure = FailureRecord(
                failure_id=generate_id("failure"),
                failure_type=failure_type,
                component=component,
                error_message=error_message,
                timestamp=now_utc(),
                cycle_id=self.current_cycle_id,
            )

            record = MemoryRecord(
                memory_id=f"failure_{failure.failure_id}",
                memory_type=MemoryType.ERROR,
                generation_id=self.active_generation_id or "unknown",
                timestamp=now_utc(),
                source_agent="SurvivalRuntime",
                importance=8,
                content={
                    "failure_id": failure.failure_id,
                    "failure_type": failure.failure_type.value,
                    "component": failure.component,
                    "error_message": failure.error_message,
                    "timestamp": failure.timestamp.isoformat(),
                    "cycle_id": failure.cycle_id,
                    "retry_count": failure.retry_count,
                    "resolved": failure.resolved,
                    "resolution_timestamp": failure.resolution_timestamp,
                },
                metadata={},
            )
            self.memory_store.save(record)
        except Exception as e:
            logger.error(f"Failed to record failure: {e}")

    # Permission boundaries
    def execute_live_trade(self, *args, **kwargs):
        raise PermissionError("SurvivalRuntime cannot execute live trades")

    def activate_live_trading(self, *args, **kwargs):
        raise PermissionError("SurvivalRuntime cannot activate live trading")


def normalize_trigger(value: str) -> str:
    """Normalize a trigger string to a DeathTrigger value."""
    normalized = str(value).upper()
    for trigger in DeathTrigger:
        if trigger.value == normalized:
            return normalized
    return "CAPITAL_DEPLETED"


def _safe_serialize(value: Any) -> Any:
    """Best-effort serialization of cycle agent results for memory storage."""
    if isinstance(value, dict):
        return {k: _safe_serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)