"""Monitoring services for the autonomous runtime.

- PortfolioMonitor: synchronizes local portfolio state against the paper
  provider (source of truth) and reports account/position snapshots.
- InvestmentMonitor: runs the Investment Safety Manager for every open
  investment, records assessments, and flags EXIT_CANDIDATE investments for
  higher-level review. It never executes sells automatically.
"""

from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.core.memory.store import MemoryStore
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.models.task import Task
from app.core.runtime.models import RuntimeConfig
from app.services.execution.provider import ExecutionProvider
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class PortfolioMonitor:
    """Synchronizes portfolio state from the paper provider."""

    def __init__(
        self,
        execution_provider: ExecutionProvider,
        memory_store: MemoryStore,
        config: RuntimeConfig,
    ):
        self.execution_provider = execution_provider
        self.memory_store = memory_store
        self.config = config

    def synchronize(self) -> Dict[str, Any]:
        """Fetch account + positions from the provider (source of truth)."""
        try:
            account = self.execution_provider.get_account()
            positions = self.execution_provider.get_positions()
            open_orders = self.execution_provider.get_open_orders()
        except Exception as e:
            logger.error("Portfolio synchronization failed: %s", e)
            return {"status": "FAILED", "error": str(e)}

        snapshot = {
            "status": "SYNCED",
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "currency": account.currency,
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "average_entry_price": p.average_entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_profit_loss": p.unrealized_profit_loss,
                }
                for p in positions
            ],
            "open_orders": [
                {
                    "order_id": o.order_id,
                    "client_order_id": o.client_order_id,
                    "symbol": o.symbol,
                    "status": o.status.value,
                }
                for o in open_orders
            ],
            "timestamp": now_utc().isoformat(),
        }

        record = MemoryRecord(
            memory_id=f"portfolio_snapshot_{generate_id('snap')}",
            memory_type=MemoryType.FACT,
            generation_id="portfolio",
            timestamp=now_utc(),
            source_agent="PortfolioMonitor",
            importance=7,
            content=snapshot,
            metadata={},
        )
        self.memory_store.save(record)
        return snapshot


class InvestmentMonitor:
    """Runs Investment Safety Manager assessments for open investments.

    EXIT_CANDIDATE recommendations are recorded and flagged for higher-level
    review — the monitor never executes sells automatically.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        market_data_provider,
        safety_agent: Optional[BaseAgent],
        config: RuntimeConfig,
    ):
        self.memory_store = memory_store
        self.market_data_provider = market_data_provider
        self.safety_agent = safety_agent
        self.config = config

    def _current_price(self, symbol: str) -> Optional[float]:
        try:
            quote = self.market_data_provider.get_quote(symbol)
            bid = getattr(quote, "bid_price", None)
            ask = getattr(quote, "ask_price", None)
            if bid and ask and bid > 0 and ask > 0:
                return (bid + ask) / 2.0
        except Exception:
            pass
        try:
            trade = self.market_data_provider.get_latest_trade(symbol)
            price = getattr(trade, "price", None)
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        return None

    def monitor_open_investments(self, generation_id: str) -> List[Dict[str, Any]]:
        """Assess every open investment for the active generation."""
        if self.safety_agent is None:
            logger.warning("Investment safety agent unavailable; skipping monitoring")
            return [{"status": "SKIPPED", "reason": "safety agent unavailable"}]

        records = self.memory_store.query(memory_type=MemoryType.INVESTMENT)
        open_investments = [
            r for r in records
            if r.content.get("status") == "OPEN"
            and r.content.get("originating_generation") == generation_id
        ]

        results: List[Dict[str, Any]] = []
        for record in open_investments:
            content = record.content
            asset = content.get("asset", "")
            quantity = float(content.get("quantity", 0.0))
            entry_price = float(content.get("entry_price", 0.0))

            price = self._current_price(asset)
            if price is None or price <= 0:
                results.append(
                    {
                        "investment_id": content.get("investment_id"),
                        "asset": asset,
                        "status": "INSUFFICIENT_DATA",
                        "reason": "current price unavailable",
                    }
                )
                continue

            current_value = quantity * price
            task = Task(
                task_id=generate_id("task"),
                requesting_agent="SurvivalRuntime",
                target_agent="investment_safety",
                task_type="safety_assessment",
                priority=1,
                input_data={
                    "investment_record": dict(content),
                    "current_price": price,
                    "current_position_value": current_value,
                    "generation_id": generation_id,
                },
                created_at=now_utc(),
            )

            try:
                agent_result = self.safety_agent.process_task(task)
            except Exception as e:
                logger.error("Safety agent failed for %s: %s", asset, e)
                results.append(
                    {
                        "investment_id": content.get("investment_id"),
                        "asset": asset,
                        "status": "FAILED",
                        "error": str(e),
                    }
                )
                continue

            recommendation = (agent_result.analysis or {}).get("recommendation")
            assessment = (agent_result.analysis or {}).get("safety_assessment")

            summary = {
                "investment_id": content.get("investment_id"),
                "asset": asset,
                "status": "MONITORED",
                "recommendation": recommendation_value(assessment, agent_result),
                "current_price": price,
                "unrealized_pl": current_value - (quantity * entry_price),
            }

            # Record the assessment for the audit trail.
            self.memory_store.save(MemoryRecord(
                memory_id=f"safety_monitor_{generate_id('mon')}",
                memory_type=MemoryType.ANALYSIS,
                generation_id=generation_id,
                timestamp=now_utc(),
                source_agent="InvestmentMonitor",
                importance=8,
                content={
                    "investment_id": content.get("investment_id"),
                    "asset": asset,
                    "recommendation": summary["recommendation"],
                    "current_price": price,
                    "assessment_summary": summary,
                },
                metadata={},
            ))

            # EXIT_CANDIDATE is a recommendation only — never an automatic sell.
            if summary["recommendation"] == "EXIT_CANDIDATE":
                self.memory_store.save(MemoryRecord(
                    memory_id=f"exit_review_{generate_id('exit')}",
                    memory_type=MemoryType.DECISION,
                    generation_id=generation_id,
                    timestamp=now_utc(),
                    source_agent="InvestmentMonitor",
                    importance=10,
                    content={
                        "decision_type": "EXIT_CANDIDATE_REVIEW",
                        "investment_id": content.get("investment_id"),
                        "asset": asset,
                        "status": "PENDING_REVIEW",
                        "note": "EXIT_CANDIDATE requires higher-level review; no automatic sell executed.",
                    },
                    metadata={},
                ))
                summary["requires_review"] = True

            results.append(summary)
        return results


def recommendation_value(assessment, agent_result) -> str:
    """Extract the recommendation string from an assessment or result."""
    if assessment is not None:
        rec = getattr(assessment, "recommendation", None)
        if rec is not None:
            return getattr(rec, "value", str(rec))
    return str((agent_result.analysis or {}).get("recommendation", "UNKNOWN"))