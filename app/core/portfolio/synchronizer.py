from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from app.core.models.execution import AccountState, PositionState, OrderResponse
from app.core.models.events import AccountUpdated, PositionUpdated
from app.services.execution.provider import ExecutionProvider
from app.utils.ids import generate_id
from app.utils.time import now_utc
from app.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class LocalPortfolio:
    """Local representation of portfolio synchronized from external paper source of truth."""
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    positions: Dict[str, PositionState] = field(default_factory=dict)
    open_orders: List[OrderResponse] = field(default_factory=list)
    last_synced_at: Optional[datetime] = None


class PortfolioSynchronizer:
    """Synchronizes local portfolio state against paper execution provider (source of truth)."""

    def __init__(self, execution_provider: ExecutionProvider):
        self._provider = execution_provider
        self._portfolio = LocalPortfolio()

    @property
    def portfolio(self) -> LocalPortfolio:
        return self._portfolio

    def synchronize(self) -> List[AccountUpdated | PositionUpdated]:
        """Fetches live paper state and reconciles local portfolio.
        Returns generated domain events reflecting changes.
        """
        events = []
        now = now_utc()

        # 1. Sync Account
        account: AccountState = self._provider.get_account()
        self._portfolio.equity = account.equity
        self._portfolio.cash = account.cash
        self._portfolio.buying_power = account.buying_power

        events.append(AccountUpdated(
            event_id=generate_id("evt_acc"),
            timestamp=now,
            event_type="AccountUpdated",
            equity=account.equity,
            cash=account.cash,
            buying_power=account.buying_power
        ))

        # 2. Sync Positions
        positions: List[PositionState] = self._provider.get_positions()
        self._portfolio.positions = {p.symbol: p for p in positions}

        for p in positions:
            events.append(PositionUpdated(
                event_id=generate_id("evt_pos"),
                timestamp=now,
                event_type="PositionUpdated",
                symbol=p.symbol,
                quantity=p.quantity,
                market_value=p.market_value,
                unrealized_pl=p.unrealized_profit_loss
            ))

        # 3. Sync Open Orders
        open_orders: List[OrderResponse] = self._provider.get_open_orders()
        self._portfolio.open_orders = open_orders

        self._portfolio.last_synced_at = now
        logger.info(f"Synchronized local portfolio: Equity=${account.equity:.2f}, Positions={len(positions)}, OpenOrders={len(open_orders)}")
        return events
