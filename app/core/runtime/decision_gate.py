"""Decision gating for the autonomous cycle.

The runtime must not blindly trade every cycle. The DecisionGate evaluates
configurable reasons to skip a new investment decision: cooldowns, duplicate
decisions, insufficient data, exposure limits, crisis conditions, provider
outages, and stale market data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.core.memory.store import MemoryStore
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.runtime.models import RuntimeConfig
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


@dataclass
class GateDecision:
    """Outcome of a decision-gate evaluation."""
    allowed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class DecisionGate:
    """Deterministic gate that decides whether a new investment decision
    should be attempted in the current cycle.

    The gate is deliberately conservative: when in doubt, it blocks new
    decisions while allowing monitoring of existing positions to continue.
    """

    def __init__(self, memory_store: MemoryStore, config: RuntimeConfig):
        self.memory_store = memory_store
        self.config = config

    # ------------------------------------------------------------------
    # Cooldown tracking
    # ------------------------------------------------------------------
    def record_decision(self, asset: str, decision_type: str, decision_id: str) -> None:
        """Record that a decision was made for an asset (for cooldowns)."""
        record = MemoryRecord(
            memory_id=f"gate_decision_{decision_id}",
            memory_type=MemoryType.DECISION,
            generation_id="gate",
            timestamp=now_utc(),
            source_agent="DecisionGate",
            importance=5,
            content={
                "asset": asset,
                "decision_type": decision_type,
                "decision_id": decision_id,
                "timestamp": now_utc().isoformat(),
            },
            metadata={},
        )
        self.memory_store.save(record)

    def _last_decision_for_asset(self, asset: str) -> Optional[datetime]:
        records = self.memory_store.query(memory_type=MemoryType.DECISION)
        latest = None
        for record in records:
            if getattr(record, "source_agent", "") != "DecisionGate":
                continue
            content = record.content
            if content.get("asset") != asset:
                continue
            try:
                ts = datetime.fromisoformat(content["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if latest is None or ts > latest:
                latest = ts
        return latest

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        asset: str,
        generation_id: str,
        market_data_fresh: bool,
        providers_healthy: bool,
        crisis_severity: Optional[str] = None,
        portfolio_value: Optional[float] = None,
        available_cash: Optional[float] = None,
        open_position_count: Optional[int] = None,
        open_order_count: Optional[int] = None,
        research_confidence: Optional[float] = None,
    ) -> GateDecision:
        """Evaluate whether a new investment decision may proceed."""
        now = now_utc()
        blockers: List[str] = []

        # 1. Provider health
        if not providers_healthy:
            blockers.append("provider_outage")

        # 2. Market data freshness
        if not market_data_fresh:
            blockers.append("stale_or_missing_market_data")

        # 3. Cooldown since last decision on this asset
        last_decision = self._last_decision_for_asset(asset)
        if last_decision is not None:
            elapsed = (now - last_decision).total_seconds()
            if elapsed < self.config.decision_cooldown_seconds:
                blockers.append(
                    f"cooldown_active({elapsed:.0f}s < {self.config.decision_cooldown_seconds}s)"
                )

        # 4. Crisis conditions
        if crisis_severity and crisis_severity.upper() in ("HIGH", "CRITICAL", "SEVERE"):
            blockers.append(f"crisis_severity_{crisis_severity.upper()}")

        # 5. Exposure / position limits
        if portfolio_value is not None and portfolio_value <= 0:
            blockers.append("no_portfolio_value")
        if (
            available_cash is not None
            and portfolio_value is not None
            and portfolio_value > 0
            and available_cash / portfolio_value < 0.05
        ):
            blockers.append("insufficient_available_cash")
        if open_position_count is not None and open_position_count >= self.config.max_open_positions:
            blockers.append(f"max_open_positions({open_position_count})")
        if open_order_count is not None and open_order_count >= self.config.max_open_orders:
            blockers.append(f"max_open_orders({open_order_count})")

        # 6. Research confidence
        if research_confidence is not None and research_confidence < self.config.min_research_confidence:
            blockers.append(
                f"low_research_confidence({research_confidence:.2f} < {self.config.min_research_confidence:.2f})"
            )

        if blockers:
            return GateDecision(
                allowed=False,
                reason="; ".join(blockers),
                details={"asset": asset, "generation_id": generation_id},
            )

        return GateDecision(
            allowed=True,
            reason="all_gate_conditions_met",
            details={"asset": asset, "generation_id": generation_id},
        )