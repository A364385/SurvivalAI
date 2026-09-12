"""Learning services for the autonomous runtime.

- ExperienceCollector: evaluates outcomes of executed decisions, closes
  investments, and writes Experience records for the learning loop.
- LearningCycle: runs the StrategyUpdaterAgent periodically and records
  strategy proposals (never activates strategies directly).
"""

from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.core.memory.store import MemoryStore
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.models.task import Task
from app.core.runtime.idempotency import IdempotencyManager
from app.core.runtime.models import RuntimeConfig
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class ExperienceCollector:
    """Collects experiences from executed decisions and their outcomes.

    Outcome evaluation is deterministic: current position value vs entry
    cost, using provider-confirmed data only.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        market_data_provider,
        config: RuntimeConfig,
    ):
        self.memory_store = memory_store
        self.market_data_provider = market_data_provider
        self.config = config

    def _current_price(self, symbol: str) -> Optional[float]:
        try:
            trade = self.market_data_provider.get_latest_trade(symbol)
            price = getattr(trade, "price", None)
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        try:
            quote = self.market_data_provider.get_quote(symbol)
            bid = getattr(quote, "bid_price", None)
            ask = getattr(quote, "ask_price", None)
            if bid and ask and bid > 0 and ask > 0:
                return (bid + ask) / 2.0
        except Exception:
            pass
        return None

    def collect_experiences(self, generation_id: str) -> List[Dict[str, Any]]:
        """Evaluate open investments and record Experience records.

        Experiences are only recorded once per investment (idempotent).
        """
        collected: List[Dict[str, Any]] = []

        investment_records = self.memory_store.query(memory_type=MemoryType.INVESTMENT)
        open_investments = [
            r for r in investment_records
            if r.content.get("status") == "OPEN"
            and r.content.get("originating_generation") == generation_id
        ]

        # Idempotency: skip investments that already have an experience.
        existing_experience_records = self.memory_store.query(
            memory_type=MemoryType.EXPERIENCE
        )
        already_experienced = {
            r.content.get("investment_id") for r in existing_experience_records
            if r.content.get("investment_id")
        }

        for record in open_investments:
            content = record.content
            investment_id = content.get("investment_id")
            if investment_id in already_experienced:
                continue

            symbol = content.get("asset", "")
            quantity = float(content.get("quantity", 0.0))
            entry_price = float(content.get("entry_price", 0.0))
            entry_cost = quantity * entry_price

            price = self._current_price(symbol)
            if price is None:
                collected.append(
                    {
                        "investment_id": investment_id,
                        "status": "INSUFFICIENT_DATA",
                        "reason": "price unavailable; outcome not evaluated",
                    }
                )
                continue

            current_value = quantity * price
            unrealized_pl = current_value - entry_cost
            outcome = "PROFIT" if unrealized_pl > 0 else ("LOSS" if unrealized_pl < 0 else "NEUTRAL")
            reward = (unrealized_pl / entry_cost) if entry_cost > 0 else 0.0

            experience_id = generate_id("exp")
            experience = MemoryRecord(
                memory_id=f"experience_{experience_id}",
                memory_type=MemoryType.EXPERIENCE,
                generation_id=generation_id,
                timestamp=now_utc(),
                source_agent="ExperienceCollector",
                importance=9,
                content={
                    "experience_id": experience_id,
                    "investment_id": investment_id,
                    "decision_id": content.get("decision_id"),
                    "asset": symbol,
                    "event_type": f"INVESTMENT_{outcome}",
                    "context": {
                        "entry_price": entry_price,
                        "quantity": quantity,
                        "entry_cost": entry_cost,
                        "thesis": content.get("investment_thesis", ""),
                    },
                    "action": {
                        "type": "PAPER_BUY",
                        "order_id": content.get("order_id"),
                    },
                    "outcome": {
                        "current_price": price,
                        "current_value": current_value,
                        "unrealized_pl": unrealized_pl,
                        "return_pct": reward,
                    },
                    "result": outcome,
                    "reward": reward,
                    "loss": max(0.0, -unrealized_pl),
                    "lessons": self._lessons(outcome, reward),
                    "contributing_agents": [
                        "market_research", "news_research", "crisis_risk",
                        "deep_looker", "risk_manager", "ceo",
                    ],
                },
                metadata={},
            )
            self.memory_store.save(experience)
            collected.append(
                {
                    "investment_id": investment_id,
                    "experience_id": experience_id,
                    "status": "COLLECTED",
                    "outcome": outcome,
                    "return_pct": reward,
                }
            )
        return collected

    def _lessons(self, outcome: str, reward: float) -> List[str]:
        lessons: List[str] = []
        if outcome == "LOSS":
            if reward < -0.10:
                lessons.append("Loss exceeded 10%: entry thesis or risk sizing should be re-examined.")
            else:
                lessons.append("Minor loss within expected volatility range.")
        elif outcome == "PROFIT":
            lessons.append("Thesis produced gains; consider whether sizing can scale within risk limits.")
        else:
            lessons.append("Flat outcome; thesis evidence was not decisive.")
        return lessons


class LearningCycle:
    """Runs the StrategyUpdaterAgent on a bounded interval.

    Proposals are recorded with status PROPOSED. The LearningCycle never
    activates strategies — activation requires the validation pipeline
    (backtest, walk-forward, risk evaluation) from Step 12/13.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        strategy_agent: Optional[BaseAgent],
        idempotency_manager: IdempotencyManager,
        config: RuntimeConfig,
    ):
        self.memory_store = memory_store
        self.strategy_agent = strategy_agent
        self.idempotency = idempotency_manager
        self.config = config

    def run_learning_cycle(self, generation_id: str) -> Dict[str, Any]:
        """Run one learning cycle (idempotent per generation + timestamp hour)."""
        window = now_utc().strftime("%Y%m%d%H")
        operation_id = f"learning_{generation_id}_{window}"

        if not self.idempotency.begin(
            IdempotencyManager.OPERATION_LEARNING_CYCLE,
            operation_id,
            generation_id=generation_id,
        ):
            return {
                "status": "DUPLICATE_SKIPPED",
                "operation_id": operation_id,
                "previous": self.idempotency.get_operation(
                    IdempotencyManager.OPERATION_LEARNING_CYCLE, operation_id
                ),
            }

        try:
            if self.strategy_agent is None:
                self.idempotency.fail(
                    IdempotencyManager.OPERATION_LEARNING_CYCLE,
                    operation_id,
                    "strategy agent unavailable",
                )
                return {"status": "SKIPPED", "reason": "strategy agent unavailable"}

            task = Task(
                task_id=generate_id("task"),
                requesting_agent="SurvivalRuntime",
                target_agent="strategy_updater",
                task_type="strategy_update",
                priority=2,
                input_data={
                    "generation_id": generation_id,
                    "force_proposal": False,
                },
                created_at=now_utc(),
            )
            result = self.strategy_agent.process_task(task)

            outcome = {
                "status": "COMPLETED" if result.status.value == "SUCCESS" else "AGENT_FAILED",
                "summary": result.summary,
                "proposal_id": (result.analysis or {}).get("proposal", {}).get("proposal_id")
                if isinstance(result.analysis, dict) else None,
                "weaknesses": (result.analysis or {}).get("weaknesses", [])
                if isinstance(result.analysis, dict) else [],
            }
            self.idempotency.complete(
                IdempotencyManager.OPERATION_LEARNING_CYCLE,
                operation_id,
                outcome,
            )
            return outcome

        except Exception as e:
            self.idempotency.fail(
                IdempotencyManager.OPERATION_LEARNING_CYCLE,
                operation_id,
                str(e),
            )
            logger.error("Learning cycle failed: %s", e)
            return {"status": "FAILED", "error": str(e)}

    def record_strategy_proposal_outcome(
        self, proposal_id: str, backtest_passed: bool, reason: str = ""
    ) -> None:
        """Record the outcome of a proposal's validation pipeline.

        This does NOT activate anything. It only records PASSED/FAILED so the
        evolution pipeline can decide on successor strategy approval.
        """
        record = MemoryRecord(
            memory_id=f"proposal_outcome_{proposal_id}",
            memory_type=MemoryType.STRATEGY,
            generation_id="evolution",
            timestamp=now_utc(),
            source_agent="LearningCycle",
            importance=9,
            content={
                "proposal_id": proposal_id,
                "validation_outcome": "PASSED" if backtest_passed else "FAILED",
                "reason": reason,
                "status": "RECORDED",
            },
            metadata={},
        )
        self.memory_store.save(record)