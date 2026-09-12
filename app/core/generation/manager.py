from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from app.core.memory.store import MemoryStore
from app.core.models.generation import (
    GenerationLifecycleState,
    DeathTrigger,
    OperatingCost,
    GenerationState,
    InheritedKnowledge,
    GenerationComparisonResult,
    DeathCondition,
    ExperienceSelectionCriteria,
)
from app.core.models.memory import (
    MemoryType,
    GenerationRecord,
    Experience,
    StrategyVersion,
    StrategyStatus,
    DeathReport,
)
from app.core.models.strategy import StrategyChangeProposal, ProposalStatus
from app.core.models.events import (
    GenerationCreated,
    GenerationInitializing,
    GenerationStarted,
    GenerationPaused,
    GenerationDying,
    GenerationDied,
    SuccessorGenerationRequested,
    SuccessorGenerationCreated,
    StrategyCandidateCreated,
    StrategyCandidateApproved,
    StrategyCandidateRejected,
)
from app.core.event_bus.event_bus import EventBus
from app.services.market_data.provider import MarketDataProvider
from app.services.news.provider import NewsProvider
from app.services.llm.provider import LLMProvider
from app.agents.registry import AgentRegistry
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class GenerationManager:
    """Manages generation lifecycle, inheritance, and evolution."""

    def __init__(
        self,
        memory_store: MemoryStore,
        event_bus: EventBus,
        agent_registry: AgentRegistry,
        market_data_provider: Optional[MarketDataProvider] = None,
        news_provider: Optional[NewsProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        initial_capital: float = 100000.0,
        minimum_survival_threshold: float = 10000.0,
        maximum_drawdown_threshold: float = 0.5,
    ):
        self.memory_store = memory_store
        self.event_bus = event_bus
        self.agent_registry = agent_registry
        self.market_data_provider = market_data_provider
        self.news_provider = news_provider
        self.llm_provider = llm_provider
        self.initial_capital = initial_capital
        self.minimum_survival_threshold = minimum_survival_threshold
        self.maximum_drawdown_threshold = maximum_drawdown_threshold

        self.active_generation: Optional[GenerationState] = None
        self.death_conditions: List[DeathCondition] = self._initialize_death_conditions()

    def _initialize_death_conditions(self) -> List[DeathCondition]:
        """Initialize default death conditions."""
        return [
            DeathCondition(
                condition_id="death_capital_depleted",
                trigger=DeathTrigger.CAPITAL_DEPLETED,
                threshold=0.0,
                is_fatal=True,
                description="Capital completely depleted",
            ),
            DeathCondition(
                condition_id="death_min_survival",
                trigger=DeathTrigger.MINIMUM_SURVIVAL_THRESHOLD,
                threshold=self.minimum_survival_threshold,
                is_fatal=True,
                description="Capital below minimum survival threshold",
            ),
            DeathCondition(
                condition_id="death_max_drawdown",
                trigger=DeathTrigger.MAXIMUM_DRAWDOWN_EXCEEDED,
                threshold=self.maximum_drawdown_threshold,
                is_fatal=True,
                description="Maximum drawdown exceeded",
            ),
        ]

    def create_generation(
        self,
        parent_generation_id: Optional[str] = None,
        strategy_version: Optional[str] = None,
        inherited_knowledge: Optional[List[InheritedKnowledge]] = None,
        inherited_strategies: Optional[List[str]] = None,
    ) -> GenerationState:
        """Create a new generation."""
        # Determine generation number
        generation_number = 1
        if parent_generation_id:
            parent = self.get_generation(parent_generation_id)
            if parent:
                generation_number = parent.generation_number + 1

        # Get or create strategy version
        if not strategy_version:
            strategy_version = self._get_or_create_default_strategy(generation_number)

        generation_id = generate_id("gen")

        state = GenerationState(
            generation_id=generation_id,
            parent_generation_id=parent_generation_id,
            generation_number=generation_number,
            strategy_version=strategy_version,
            lifecycle_state=GenerationLifecycleState.CREATED,
            creation_timestamp=now_utc(),
            start_timestamp=None,
            end_timestamp=None,
            starting_capital=self.initial_capital,
            current_capital=self.initial_capital,
            ending_capital=None,
            return_percentage=None,
            maximum_drawdown=0.0,
            lifespan_days=None,
            cause_of_death=None,
            death_trigger=None,
            death_timestamp=None,
            inherited_experience_refs=[k.knowledge_id for k in inherited_knowledge] if inherited_knowledge else [],
            inherited_strategy_refs=inherited_strategies or [],
        )

        # Store in memory
        self._store_generation_state(state)

        # Emit event
        self.event_bus.publish(
            GenerationCreated(
                event_id=generate_id("evt"),
                timestamp=now_utc(),
                event_type="GenerationCreated",
                generation_id=generation_id,
                parent_generation_id=parent_generation_id,
                generation_number=generation_number,
                strategy_version=strategy_version,
                starting_capital=self.initial_capital,
            )
        )

        return state

    def start_generation(self, generation_id: str) -> bool:
        """Start a generation after infrastructure checks."""
        state = self.get_generation(generation_id)
        if not state:
            logger.error(f"Generation {generation_id} not found")
            return False

        if state.lifecycle_state != GenerationLifecycleState.CREATED:
            logger.error(f"Generation {generation_id} is not in CREATED state")
            return False

        # Infrastructure health check
        infrastructure_status = self._check_infrastructure()
        if not infrastructure_status["all_available"]:
            logger.error(f"Infrastructure check failed: {infrastructure_status}")
            self.event_bus.publish(
                GenerationInitializing(
                    event_id=generate_id("evt"),
                    timestamp=now_utc(),
                    event_type="GenerationInitializing",
                    generation_id=generation_id,
                    strategy_version=state.strategy_version,
                    infrastructure_check_status="FAILED",
                )
            )
            return False

        # Update state
        state.lifecycle_state = GenerationLifecycleState.INITIALIZING
        state.start_timestamp = now_utc()
        self._store_generation_state(state)

        # Load inherited knowledge
        if state.inherited_experience_refs:
            self._load_inherited_knowledge(state)

        # Transition to ACTIVE
        state.lifecycle_state = GenerationLifecycleState.ACTIVE
        self._store_generation_state(state)

        # Set as active generation
        self.active_generation = state

        # Emit events
        self.event_bus.publish(
            GenerationInitializing(
                event_id=generate_id("evt"),
                timestamp=now_utc(),
                event_type="GenerationInitializing",
                generation_id=generation_id,
                strategy_version=state.strategy_version,
                infrastructure_check_status="SUCCESS",
            )
        )

        self.event_bus.publish(
            GenerationStarted(
                event_id=generate_id("evt"),
                timestamp=now_utc(),
                event_type="GenerationStarted",
                generation_id=generation_id,
                starting_capital=state.starting_capital,
            )
        )

        logger.info(f"Generation {generation_id} started successfully")
        return True

    def pause_generation(self, generation_id: str, reason: str) -> bool:
        """Pause a generation."""
        state = self.get_generation(generation_id)
        if not state:
            return False

        if state.lifecycle_state != GenerationLifecycleState.ACTIVE:
            return False

        state.lifecycle_state = GenerationLifecycleState.PAUSED
        self._store_generation_state(state)

        self.event_bus.publish(
            GenerationPaused(
                event_id=generate_id("evt"),
                timestamp=now_utc(),
                event_type="GenerationPaused",
                generation_id=generation_id,
                reason=reason,
                current_capital=state.current_capital,
            )
        )

        return True

    def check_death_conditions(self, generation_id: str) -> Optional[DeathTrigger]:
        """Check if any death conditions are triggered."""
        state = self.get_generation(generation_id)
        if not state:
            return None

        if state.lifecycle_state not in [GenerationLifecycleState.ACTIVE, GenerationLifecycleState.PAUSED]:
            return None

        for condition in self.death_conditions:
            if not condition.enabled:
                continue

            triggered = False

            if condition.trigger == DeathTrigger.CAPITAL_DEPLETED:
                triggered = state.current_capital <= condition.threshold
            elif condition.trigger == DeathTrigger.MINIMUM_SURVIVAL_THRESHOLD:
                triggered = state.current_capital < condition.threshold
            elif condition.trigger == DeathTrigger.MAXIMUM_DRAWDOWN_EXCEEDED:
                triggered = state.maximum_drawdown > condition.threshold

            if triggered:
                logger.warning(f"Death condition triggered: {condition.description}")
                return condition.trigger

        return None

    def kill_generation(self, generation_id: str, trigger: DeathTrigger, reason: str) -> bool:
        """Kill a generation and create death report."""
        state = self.get_generation(generation_id)
        if not state:
            return False

        # Update state
        state.lifecycle_state = GenerationLifecycleState.DYING
        state.death_trigger = trigger
        state.death_timestamp = now_utc()
        self._store_generation_state(state)

        # Emit dying event
        self.event_bus.publish(
            GenerationDying(
                event_id=generate_id("evt"),
                timestamp=now_utc(),
                event_type="GenerationDying",
                generation_id=generation_id,
                trigger=trigger.value,
                reason=reason,
                current_capital=state.current_capital,
                maximum_drawdown=state.maximum_drawdown,
            )
        )

        # Create death report
        death_report = self._create_death_report(state, reason)

        # Finalize state
        state.lifecycle_state = GenerationLifecycleState.DEAD
        state.end_timestamp = now_utc()
        state.ending_capital = state.current_capital
        state.lifespan_days = (state.end_timestamp - state.start_timestamp).days if state.start_timestamp else 0
        state.return_percentage = (
            (state.ending_capital - state.starting_capital) / state.starting_capital
            if state.starting_capital > 0
            else 0.0
        )
        state.cause_of_death = reason
        self._store_generation_state(state)

        # Store death report
        self._store_death_report(death_report)

        # Emit died event
        self.event_bus.publish(
            GenerationDied(
                event_id=generate_id("evt"),
                timestamp=now_utc(),
                event_type="GenerationDied",
                generation_id=generation_id,
                reason=reason,
                survival_time_days=state.lifespan_days,
            )
        )

        # Clear active generation if this was the active one
        if self.active_generation and self.active_generation.generation_id == generation_id:
            self.active_generation = None

        logger.info(f"Generation {generation_id} died: {reason}")
        return True

    def request_successor(self, generation_id: str, strategy_candidate_id: Optional[str] = None) -> bool:
        """Request creation of a successor generation."""
        state = self.get_generation(generation_id)
        if not state:
            return False

        if state.lifecycle_state != GenerationLifecycleState.DEAD:
            return False

        state.lifecycle_state = GenerationLifecycleState.SUCCESSOR_PENDING
        self._store_generation_state(state)

        self.event_bus.publish(
            SuccessorGenerationRequested(
                event_id=generate_id("evt"),
                timestamp=now_utc(),
                event_type="SuccessorGenerationRequested",
                parent_generation_id=generation_id,
                reason="Generation died, requesting successor",
                strategy_candidate_id=strategy_candidate_id,
            )
        )

        return True

    def update_capital(self, generation_id: str, new_capital: float) -> bool:
        """Update generation capital and check death conditions."""
        state = self.get_generation(generation_id)
        if not state:
            return False

        state.current_capital = new_capital
        self._apply_capital_change(state)
        return True

    def _apply_capital_change(self, state: GenerationState) -> None:
        """Persist a mutated state after a capital change and check death.

        Callers MUST pass the same `state` instance they mutated: re-reading
        the generation here would silently discard concurrent field changes
        (this previously lost every operating cost that was recorded).
        """
        peak = max(state.starting_capital, state.current_capital)
        drawdown = (peak - state.current_capital) / peak if peak > 0 else 0.0
        state.maximum_drawdown = max(state.maximum_drawdown, drawdown)

        self._store_generation_state(state)

        trigger = self.check_death_conditions(state.generation_id)
        if trigger:
            self.kill_generation(
                state.generation_id,
                trigger,
                f"Death condition triggered: {trigger.value} (capital: {state.current_capital:.2f}, drawdown: {state.maximum_drawdown:.2%})",
            )

    def add_operating_cost(self, generation_id: str, cost: OperatingCost) -> bool:
        """Record an operating cost and reduce capital by its amount."""
        state = self.get_generation(generation_id)
        if not state:
            return False

        state.operating_costs.append(cost)
        state.total_operating_costs += cost.amount
        state.current_capital -= cost.amount

        # Persist this same state so the cost and the capital change land
        # together, then run the shared death-condition check.
        self._apply_capital_change(state)
        return True

    def get_generation(self, generation_id: str) -> Optional[GenerationState]:
        """Retrieve generation state."""
        records = self.memory_store.query(
            memory_type=MemoryType.GENERATION,
            generation_id=generation_id,
        )

        if not records:
            return None

        record = records[0]
        try:
            content = record.content.copy()
            # Convert string enums back to enum values
            if isinstance(content.get("lifecycle_state"), str):
                content["lifecycle_state"] = GenerationLifecycleState(content["lifecycle_state"])
            if isinstance(content.get("death_trigger"), str):
                content["death_trigger"] = DeathTrigger(content["death_trigger"])
            if content.get("death_timestamp"):
                content["death_timestamp"] = datetime.fromisoformat(content["death_timestamp"]) if isinstance(content["death_timestamp"], str) else content["death_timestamp"]
            if content.get("creation_timestamp"):
                content["creation_timestamp"] = datetime.fromisoformat(content["creation_timestamp"]) if isinstance(content["creation_timestamp"], str) else content["creation_timestamp"]
            if content.get("start_timestamp"):
                content["start_timestamp"] = datetime.fromisoformat(content["start_timestamp"]) if isinstance(content["start_timestamp"], str) else content["start_timestamp"]
            if content.get("end_timestamp"):
                content["end_timestamp"] = datetime.fromisoformat(content["end_timestamp"]) if isinstance(content["end_timestamp"], str) else content["end_timestamp"]

            # Reconstruct operating costs. Stores may hand back already-built
            # OperatingCost objects (in-memory, or revived from SQLite), so
            # only plain mappings need constructing.
            if content.get("operating_costs"):
                content["operating_costs"] = [
                    c if isinstance(c, OperatingCost) else OperatingCost(**c)
                    for c in content["operating_costs"]
                ]

            return GenerationState(**content)
        except Exception as e:
            logger.error(f"Failed to parse generation state: {e}")
            return None

    def get_parent_generation(self, generation_id: str) -> Optional[GenerationState]:
        """Get parent generation."""
        state = self.get_generation(generation_id)
        if not state or not state.parent_generation_id:
            return None

        return self.get_generation(state.parent_generation_id)

    def get_child_generations(self, generation_id: str) -> List[GenerationState]:
        """Get child generations."""
        all_generations = self.memory_store.query(memory_type=MemoryType.GENERATION)
        children = []

        for record in all_generations:
            try:
                state = GenerationState(**record.content)
                if state.parent_generation_id == generation_id:
                    children.append(state)
            except Exception:
                pass

        return children

    def get_generation_lineage(self, generation_id: str) -> List[GenerationState]:
        """Get full lineage from root to current generation."""
        lineage = []
        current = self.get_generation(generation_id)

        while current:
            lineage.insert(0, current)
            current = self.get_parent_generation(current.generation_id)

        return lineage

    def compare_generations(self, generation_a_id: str, generation_b_id: str) -> Optional[GenerationComparisonResult]:
        """Compare two generations."""
        state_a = self.get_generation(generation_a_id)
        state_b = self.get_generation(generation_b_id)

        if not state_a or not state_b:
            return None

        return_generation = None
        if state_b.return_percentage and state_a.return_percentage:
            if state_b.return_percentage > state_a.return_percentage:
                return_generation = generation_b_id
            elif state_a.return_percentage > state_b.return_percentage:
                return_generation = generation_a_id

        comparison = GenerationComparisonResult(
            generation_a_id=generation_a_id,
            generation_b_id=generation_b_id,
            comparison_timestamp=now_utc(),
            total_return_difference=(state_b.return_percentage or 0) - (state_a.return_percentage or 0),
            maximum_drawdown_difference=state_b.maximum_drawdown - state_a.maximum_drawdown,
            survival_duration_difference=(state_b.lifespan_days or 0) - (state_a.lifespan_days or 0),
            ending_capital_difference=state_b.ending_capital - state_a.ending_capital if state_b.ending_capital and state_a.ending_capital else 0.0,  # Fixed: was state.a
            number_of_investments_difference=0,  # Would need investment tracking
            win_rate_difference=None,
            transaction_cost_difference=0.0,
            operating_cost_difference=state_b.total_operating_costs - state_a.total_operating_costs,
            crisis_performance_difference={},
            market_regime_performance_difference={},
            benchmark_performance_difference={},
            strategy_version_difference=f"{state_a.strategy_version} -> {state_b.strategy_version}",
            risk_violation_count_difference=0,
            blocked_investment_count_difference=0,
            major_failures=[],
            fitness_scores={
                f"{generation_a_id}": self._calculate_fitness(state_a),
                f"{generation_b_id}": self._calculate_fitness(state_b),
            },
            recommended_generation=return_generation,
        )

        return comparison

    def _check_infrastructure(self) -> Dict[str, Any]:
        """Check infrastructure availability."""
        status = {
            "all_available": True,
            "market_data": False,
            "news": False,
            "llm": False,
            "memory": False,
            "registry": False,
        }

        try:
            if self.market_data_provider:
                self.market_data_provider.get_market_clock()
                status["market_data"] = True
        except Exception:
            status["market_data"] = False
            status["all_available"] = False

        try:
            if self.news_provider:
                self.news_provider.get_latest_news(limit=1)
                status["news"] = True
        except Exception:
            status["news"] = False
            status["all_available"] = False

        try:
            if self.llm_provider:
                status["llm"] = True
        except Exception:
            status["llm"] = False
            status["all_available"] = False

        try:
            self.memory_store.list()
            status["memory"] = True
        except Exception:
            status["memory"] = False
            status["all_available"] = False

        try:
            self.agent_registry.list_agents()
            status["registry"] = True
        except Exception:
            status["registry"] = False
            status["all_available"] = False

        return status

    def _get_or_create_default_strategy(self, generation_number: int) -> str:
        """Get or create default strategy for a generation."""
        from app.core.models.memory import MemoryRecord

        strategy_id = f"strategy_v{generation_number}"

        existing = self.memory_store.query(
            memory_type=MemoryType.STRATEGY,
            strategy_id=strategy_id,
        )

        if existing:
            return strategy_id

        # Create default strategy
        strategy = StrategyVersion(
            strategy_id=strategy_id,
            version=f"{generation_number}.0",
            parent_strategy_id=None,
            created_at=now_utc(),
            generation_id=f"gen_{generation_number}",
            parameters={
                "max_position_size": 0.1,
                "cash_reserve": 0.2,
                "max_drawdown": 0.3,
            },
            rules={
                "diversification": True,
                "risk_limit": 0.15,
            },
            description=f"Default strategy for generation {generation_number}",
            status=StrategyStatus.ACTIVE,
        )

        record = MemoryRecord(
            memory_id=generate_id("mem_strategy"),
            memory_type=MemoryType.STRATEGY,
            generation_id=f"gen_{generation_number}",
            timestamp=now_utc(),
            source_agent="GenerationManager",
            importance=10,
            content={
                "strategy_id": strategy.strategy_id,
                "version": strategy.version,
                "parent_strategy_id": strategy.parent_strategy_id,
                "created_at": strategy.created_at,
                "generation_id": strategy.generation_id,
                "parameters": strategy.parameters,
                "rules": strategy.rules,
                "description": strategy.description,
                "status": strategy.status,
            },
            metadata={},
        )
        self.memory_store.save(record)

        return strategy_id

    def _load_inherited_knowledge(self, state: GenerationState) -> None:
        """Load inherited knowledge from parent generation."""
        if not state.inherited_experience_refs or not state.inherited_strategy_refs:
            return

        # Load Experiences
        experiences = []
        for ref_id in state.inherited_experience_refs:
            records = self.memory_store.query(
                memory_type=MemoryType.EXPERIENCE,
                memory_id=ref_id,
            )
            if records:
                # The ref_id is the memory_id
                experiences.append(records[0].content)

        # Load Strategies
        strategies = []
        for ref_id in state.inherited_strategy_refs:
            records = self.memory_store.query(
                memory_type=MemoryType.STRATEGY,
                strategy_id=ref_id,
            )
            if records:
                strategies.append(records[0].content)

        logger.info(
            f"Loaded {len(experiences)} experiences and {len(strategies)} "
            "strategies from parent generation for {state.generation_id}."
        )

    def _create_death_report(self, state: GenerationState, reason: str) -> DeathReport:
        """Create death report for a generation."""
        return DeathReport(
            generation_id=state.generation_id,
            starting_capital=state.starting_capital,
            final_capital=state.current_capital,
            lifespan=state.lifespan_days or 0,
            total_return=state.return_percentage or 0.0,
            maximum_drawdown=state.maximum_drawdown,
            largest_losses=[],
            major_decisions=[],
            failed_decisions=[],
            successful_decisions=[],
            agent_performance=[],
            market_conditions={},
            crisis_conditions={},
            active_strategy=state.strategy_version,
            suspected_causes=[reason],
            lessons_for_future_analysis=[],
        )

    def _store_generation_state(self, state: GenerationState) -> None:
        """Store generation state in memory."""
        from app.core.models.memory import MemoryRecord

        record = MemoryRecord(
            memory_id=f"gen_state_{state.generation_id}",
            memory_type=MemoryType.GENERATION,
            generation_id=state.generation_id,
            timestamp=now_utc(),
            source_agent="GenerationManager",
            importance=10,
            content={
                "generation_id": state.generation_id,
                "parent_generation_id": state.parent_generation_id,
                "generation_number": state.generation_number,
                "strategy_version": state.strategy_version,
                "lifecycle_state": state.lifecycle_state,
                "creation_timestamp": state.creation_timestamp,
                "start_timestamp": state.start_timestamp,
                "end_timestamp": state.end_timestamp,
                "starting_capital": state.starting_capital,
                "current_capital": state.current_capital,
                "ending_capital": state.ending_capital,
                "return_percentage": state.return_percentage,
                "maximum_drawdown": state.maximum_drawdown,
                "lifespan_days": state.lifespan_days,
                "cause_of_death": state.cause_of_death,
                "death_trigger": state.death_trigger,
                "death_timestamp": state.death_timestamp,
                "death_details": state.death_details,
                "inherited_experience_refs": state.inherited_experience_refs,
                "inherited_strategy_refs": state.inherited_strategy_refs,
                "successor_generation_id": state.successor_generation_id,
                "operating_costs": state.operating_costs,
                "total_operating_costs": state.total_operating_costs,
                "active_investments": state.active_investments,
                "metrics": state.metrics,
            },
            metadata={},
        )
        self.memory_store.save(record)

    def _store_death_report(self, report: DeathReport) -> None:
        """Store death report in memory."""
        from app.core.models.memory import MemoryRecord

        record = MemoryRecord(
            memory_id=f"death_report_{report.generation_id}",
            memory_type=MemoryType.DEATH,
            generation_id=report.generation_id,
            timestamp=now_utc(),
            source_agent="GenerationManager",
            importance=10,
            content={
                "generation_id": report.generation_id,
                "starting_capital": report.starting_capital,
                "final_capital": report.final_capital,
                "lifespan": report.lifespan,
                "total_return": report.total_return,
                "maximum_drawdown": report.maximum_drawdown,
                "largest_losses": report.largest_losses,
                "major_decisions": report.major_decisions,
                "failed_decisions": report.failed_decisions,
                "successful_decisions": report.successful_decisions,
                "agent_performance": [
                    {
                        "agent_id": ap.agent_id,
                        "generation_id": ap.generation_id,
                        "tasks_completed": ap.tasks_completed,
                        "tasks_failed": ap.tasks_failed,
                        "useful_predictions": ap.useful_predictions,
                        "incorrect_predictions": ap.incorrect_predictions,
                        "contribution_score": ap.contribution_score,
                        "cost": ap.cost,
                        "performance_notes": ap.performance_notes,
                    }
                    for ap in report.agent_performance
                ],
                "market_conditions": report.market_conditions,
                "crisis_conditions": report.crisis_conditions,
                "active_strategy": report.active_strategy,
                "suspected_causes": report.suspected_causes,
                "lessons_for_future_analysis": report.lessons_for_future_analysis,
            },
            metadata={},
        )
        self.memory_store.save(record)

    def _calculate_fitness(self, state: GenerationState) -> float:
        """Calculate simple fitness score for a generation."""
        if not state.return_percentage:
            return 0.0

        # Simple fitness: penalize drawdown heavily, reward return
        drawdown_penalty = state.maximum_drawdown * 2.0
        return_score = state.return_percentage if state.return_percentage > 0 else state.return_percentage * 0.5

        fitness = return_score - drawdown_penalty
        return max(0.0, fitness)

    # Permission boundaries
    def execute_trade(self, *args, **kwargs):
        raise PermissionError("GenerationManager cannot execute trades")

    def activate_strategy(self, *args, **kwargs):
        raise PermissionError("GenerationManager cannot directly activate strategies")
