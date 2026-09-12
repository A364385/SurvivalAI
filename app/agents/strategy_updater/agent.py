from dataclasses import asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.agents.base import BaseAgent
from app.core.memory.store import MemoryStore
from app.core.models.agent import AgentConfig, AgentResult, AgentStatus
from app.core.models.memory import (
    MemoryType,
    Experience,
    DecisionRecord,
    InvestmentRecord,
    AgentPerformanceRecord,
    DeathReport,
    StrategyVersion,
    StrategyStatus,
)
from app.core.models.strategy import (
    ProposalStatus,
    StrategyChangeProposal,
)
from app.core.models.task import Task
from app.services.llm.provider import LLMProvider
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class StrategyUpdaterAgent(BaseAgent):
    """Analyzes historical performance and proposes strategy improvements.

    The Strategy Updater learns from past decisions, outcomes, and generation
    deaths to identify weaknesses and propose candidate strategy changes. It
    NEVER directly modifies the active strategy.
    """

    SYSTEM_PROMPT = """You are the SurvivalAI Strategy Updater.
    Your role is to analyze historical performance and propose strategy improvements.
    Never directly modify the active strategy. Always create proposals with evidence.
    """

    def __init__(
        self,
        agent_id: str,
        llm_provider: Optional[LLMProvider] = None,
        memory_store: Optional[MemoryStore] = None,
        configuration: Optional[AgentConfig] = None,
        minimum_observations: int = 10,
        minimum_confidence: float = 0.6,
    ):
        config = configuration or AgentConfig(
            version="1.0.0", model=None, max_tokens=2000
        )
        super().__init__(
            agent_id=agent_id,
            agent_name="StrategyUpdaterAgent",
            role="Strategy Learning and Proposal",
            description="Analyzes historical performance and proposes strategy improvements based on evidence.",
            version="1.0.0",
            configuration=config,
            capabilities=[
                "historical_analysis",
                "pattern_detection",
                "strategy_proposal",
                "backtest_triggering",
                "strategy_comparison",
            ],
        )
        self.llm_provider = llm_provider
        self.memory_store = memory_store
        self.minimum_observations = minimum_observations
        self.minimum_confidence = minimum_confidence

    def _get_active_strategy(self) -> Optional[StrategyVersion]:
        """Retrieve the currently active strategy."""
        if not self.memory_store:
            return None

        strategies = self.memory_store.query(
            memory_type=MemoryType.STRATEGY,
            status=StrategyStatus.ACTIVE,
        )

        if not strategies:
            return None

        # Return the most recent active strategy
        latest = max(strategies, key=lambda s: s.timestamp)
        return StrategyVersion(**latest.content)

    def _analyze_historical_experiences(
        self, generation_id: Optional[str] = None
    ) -> List[Experience]:
        """Analyze historical experiences to identify patterns."""
        if not self.memory_store:
            return []

        query_params = {"memory_type": MemoryType.EXPERIENCE}
        if generation_id:
            query_params["generation_id"] = generation_id

        records = self.memory_store.query(**query_params)
        experiences = []

        for record in records:
            try:
                exp = Experience(**record.content)
                experiences.append(exp)
            except Exception as e:
                logger.warning(f"Failed to parse experience: {e}")

        return experiences

    def _analyze_historical_decisions(
        self, generation_id: Optional[str] = None
    ) -> List[DecisionRecord]:
        """Analyze historical decisions to identify patterns."""
        if not self.memory_store:
            return []

        query_params = {"memory_type": MemoryType.DECISION}
        if generation_id:
            query_params["generation_id"] = generation_id

        records = self.memory_store.query(**query_params)
        decisions = []

        for record in records:
            try:
                decision = DecisionRecord(**record.content)
                decisions.append(decision)
            except Exception as e:
                logger.warning(f"Failed to parse decision: {e}")

        return decisions

    def _analyze_historical_investments(
        self, generation_id: Optional[str] = None
    ) -> List[InvestmentRecord]:
        """Analyze historical investments to identify patterns."""
        if not self.memory_store:
            return []

        query_params = {"memory_type": MemoryType.INVESTMENT}
        if generation_id:
            query_params["generation_id"] = generation_id

        records = self.memory_store.query(**query_params)
        investments = []

        for record in records:
            try:
                investment = InvestmentRecord(**record.content)
                investments.append(investment)
            except Exception as e:
                logger.warning(f"Failed to parse investment: {e}")

        return investments

    def _analyze_death_reports(self) -> List[DeathReport]:
        """Analyze death reports to identify fatal weaknesses."""
        if not self.memory_store:
            return []

        records = self.memory_store.query(memory_type=MemoryType.DEATH)
        reports = []

        for record in records:
            try:
                report = DeathReport(**record.content)
                reports.append(report)
            except Exception as e:
                logger.warning(f"Failed to parse death report: {e}")

        return reports

    def _identify_repeated_losses(
        self, investments: List[InvestmentRecord]
    ) -> List[Dict[str, Any]]:
        """Identify patterns in repeated losses."""
        losses = []

        for inv in investments:
            if (
                inv.status.value == "CLOSED"
                and inv.realized_profit_loss is not None
                and inv.realized_profit_loss < 0
            ):
                losses.append(
                    {
                        "asset": inv.asset,
                        "loss": inv.realized_profit_loss,
                        "entry_timestamp": inv.entry_timestamp,
                        "exit_timestamp": inv.exit_timestamp,
                        "thesis": inv.investment_thesis,
                        "time_horizon": inv.time_horizon,
                        "risk_level": inv.risk_level,
                    }
                )

        return losses

    def _identify_successful_patterns(
        self, investments: List[InvestmentRecord]
    ) -> List[Dict[str, Any]]:
        """Identify patterns in successful investments."""
        successes = []

        for inv in investments:
            if (
                inv.status.value == "CLOSED"
                and inv.realized_profit_loss is not None
                and inv.realized_profit_loss > 0
            ):
                successes.append(
                    {
                        "asset": inv.asset,
                        "gain": inv.realized_profit_loss,
                        "entry_timestamp": inv.entry_timestamp,
                        "exit_timestamp": inv.exit_timestamp,
                        "thesis": inv.investment_thesis,
                        "time_horizon": inv.time_horizon,
                        "risk_level": inv.risk_level,
                    }
                )

        return successes

    def _identify_weaknesses(
        self,
        experiences: List[Experience],
        decisions: List[DecisionRecord],
        investments: List[InvestmentRecord],
        death_reports: List[DeathReport],
    ) -> List[str]:
        """Identify strategy weaknesses from historical data."""
        weaknesses = []

        # Check for repeated losses
        losses = self._identify_repeated_losses(investments)
        if len(losses) >= self.minimum_observations:
            weaknesses.append("Repeated investment losses detected")

        # Check for excessive drawdowns in death reports
        for report in death_reports:
            if report.maximum_drawdown > 0.5:  # 50% drawdown
                weaknesses.append(
                    f"Severe drawdown in generation {report.generation_id}: {report.maximum_drawdown:.2%}"
                )

        # Check for agent failures
        agent_performance = self.memory_store.query(
            memory_type=MemoryType.AGENT_PERFORMANCE
        ) if self.memory_store else []

        for record in agent_performance:
            try:
                perf = AgentPerformanceRecord(**record.content)
                if perf.tasks_failed > perf.tasks_completed * 0.3:
                    weaknesses.append(
                        f"High failure rate for agent {perf.agent_id}: {perf.tasks_failed}/{perf.tasks_completed}"
                    )
            except Exception:
                pass

        return weaknesses

    def _create_strategy_proposal(
        self,
        active_strategy: StrategyVersion,
        weaknesses: List[str],
        supporting_experiences: List[str],
        generation_id: str,
    ) -> StrategyChangeProposal:
        """Create a strategy change proposal."""
        # Simple deterministic proposal logic
        # In production, this would use LLM for more sophisticated analysis

        proposed_parameters = active_strategy.parameters.copy()
        changed_rules = {}
        unchanged_rules = {}

        # Example: if excessive losses detected, reduce position size
        if "Repeated investment losses" in " ".join(weaknesses):
            if "max_position_size" in proposed_parameters:
                changed_rules["max_position_size"] = {
                    "old": proposed_parameters["max_position_size"],
                    "new": proposed_parameters["max_position_size"] * 0.8,
                    "reason": "Reduce position size to limit loss exposure",
                }
                proposed_parameters["max_position_size"] *= 0.8

        # Preserve unchanged rules
        for rule_key, rule_value in active_strategy.rules.items():
            if rule_key not in changed_rules:
                unchanged_rules[rule_key] = rule_value

        motivation = "; ".join(weaknesses) if weaknesses else "Regular strategy review"

        proposal = StrategyChangeProposal(
            proposal_id=generate_id("proposal"),
            parent_strategy_id=active_strategy.strategy_id,
            proposed_parameters=proposed_parameters,
            changed_rules=changed_rules,
            unchanged_rules=unchanged_rules,
            motivation=motivation,
            supporting_experiences=supporting_experiences,
            supporting_decisions=[],
            supporting_outcomes=[],
            expected_effect="Reduced risk exposure while maintaining growth potential",
            risks=["May reduce growth during favorable conditions"],
            assumptions=["Historical patterns will continue to be relevant"],
            confidence=0.7,
            backtest_required=True,
            status=ProposalStatus.PROPOSED,
            created_at=now_utc(),
            generation_id=generation_id,
        )

        return proposal

    def process_task(self, task: Task) -> AgentResult:
        """Process a strategy update task."""
        try:
            generation_id = task.input_data.get("generation_id")
            force_proposal = task.input_data.get("force_proposal", False)

            # Get active strategy
            active_strategy = self._get_active_strategy()
            if not active_strategy:
                return AgentResult(
                    agent_id=self.agent_id,
                    task_id=task.task_id,
                    timestamp=now_utc(),
                    status=AgentStatus.FAILED,
                    summary="No active strategy found",
                    confidence=0.0,
                    analysis={"error": "No active strategy to update"},
                    errors=["No active strategy found"],
                )

            # Analyze historical data
            experiences = self._analyze_historical_experiences(generation_id)
            decisions = self._analyze_historical_decisions(generation_id)
            investments = self._analyze_historical_investments(generation_id)
            death_reports = self._analyze_death_reports()

            # Identify weaknesses
            weaknesses = self._identify_weaknesses(
                experiences, decisions, investments, death_reports
            )

            # Only create proposal if weaknesses found or forced
            if not weaknesses and not force_proposal:
                return AgentResult(
                    agent_id=self.agent_id,
                    task_id=task.task_id,
                    timestamp=now_utc(),
                    status=AgentStatus.SUCCESS,
                    summary="No significant weaknesses detected",
                    confidence=0.9,
                    analysis={
                        "weaknesses": [],
                        "recommendation": "Current strategy appears sound",
                    },
                )

            # Create proposal
            supporting_experiences = [exp.experience_id for exp in experiences]
            proposal = self._create_strategy_proposal(
                active_strategy, weaknesses, supporting_experiences, generation_id or "unknown"
            )

            # Store proposal in memory
            if self.memory_store:
                from app.core.models.memory import MemoryRecord

                memory_record = MemoryRecord(
                    memory_id=generate_id("mem_proposal"),
                    memory_type=MemoryType.STRATEGY,
                    generation_id=generation_id or "unknown",
                    timestamp=now_utc(),
                    source_agent=self.agent_id,
                    importance=9,
                    content=asdict(proposal),
                    metadata={"proposal_id": proposal.proposal_id},
                )
                self.memory_store.save(memory_record)

            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now_utc(),
                status=AgentStatus.SUCCESS,
                summary=f"Strategy change proposal created: {proposal.proposal_id}",
                confidence=proposal.confidence,
                analysis={
                    "proposal": asdict(proposal),
                    "weaknesses": weaknesses,
                    "supporting_evidence_count": len(supporting_experiences),
                },
            )

        except Exception as e:
            logger.error(f"Strategy update failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now_utc(),
                status=AgentStatus.FAILED,
                summary="Strategy update failed",
                confidence=0.0,
                analysis={"error": str(e)},
                errors=[str(e)],
            )

    # Permission boundaries
    def activate_strategy(self, *args, **kwargs):
        raise PermissionError("Strategy Updater cannot directly activate strategies")

    def modify_active_strategy(self, *args, **kwargs):
        raise PermissionError("Strategy Updater cannot directly modify active strategy")
