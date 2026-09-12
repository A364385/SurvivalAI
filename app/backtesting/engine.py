from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from app.core.models.market import Bar
from app.core.models.execution import OrderSide, OrderType
from app.core.models.risk import InvestmentProposal, PortfolioRiskState, RiskPolicy
from app.services.market_data.provider import MarketDataProvider
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class BacktestPhase(Enum):
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"


@dataclass
class TransactionCostConfig:
    """Configuration for transaction costs and slippage."""
    commission_per_trade: float = 1.0
    commission_per_share: float = 0.01
    spread_percentage: float = 0.001  # 0.1%
    slippage_percentage: float = 0.0005  # 0.05%
    market_impact_factor: float = 0.0001


@dataclass
class SimulatedPosition:
    """A position in the backtest portfolio."""
    symbol: str
    quantity: float
    average_entry_price: float
    entry_timestamp: datetime
    current_price: float
    market_value: float
    unrealized_pnl: float


@dataclass
class SimulatedTrade:
    """A trade executed in the backtest."""
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: float
    execution_price: float
    timestamp: datetime
    commission: float
    slippage: float
    total_cost: float


@dataclass
class SimulatedPortfolio:
    """The backtest portfolio state."""
    cash: float
    positions: Dict[str, SimulatedPosition] = field(default_factory=dict)
    initial_capital: float = 0.0
    current_value: float = 0.0
    total_return: float = 0.0
    peak_value: float = 0.0
    maximum_drawdown: float = 0.0
    trades: List[SimulatedTrade] = field(default_factory=list)


class DataLeakageProtection:
    """Protects against look-ahead bias and data leakage."""

    def __init__(self, current_timestamp: datetime):
        self.current_timestamp = current_timestamp
        self.lookahead_window_seconds = 60  # 1 minute buffer

    def is_data_available(self, data_timestamp: datetime) -> bool:
        """Check if data would have been available at current timestamp."""
        if data_timestamp is None:
            return False
        return data_timestamp <= self.current_timestamp

    def filter_future_data(self, data: List[Any], timestamp_attr: str) -> List[Any]:
        """Filter out data that would be in the future."""
        available = []
        for item in data:
            if hasattr(item, timestamp_attr):
                data_ts = getattr(item, timestamp_attr)
                if self.is_data_available(data_ts):
                    available.append(item)
        return available


class BacktestEngine:
    """Backtesting engine for strategy evaluation."""

    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        transaction_config: Optional[TransactionCostConfig] = None,
        risk_policy: Optional[RiskPolicy] = None,
    ):
        self.market_data_provider = market_data_provider
        self.transaction_config = transaction_config or TransactionCostConfig()
        self.risk_policy = risk_policy or RiskPolicy()
        self.portfolio: Optional[SimulatedPortfolio] = None
        self.current_timestamp: Optional[datetime] = None

    def initialize_portfolio(self, initial_capital: float) -> SimulatedPortfolio:
        """Initialize the backtest portfolio."""
        self.portfolio = SimulatedPortfolio(
            cash=initial_capital,
            initial_capital=initial_capital,
            current_value=initial_capital,
            peak_value=initial_capital,
        )
        return self.portfolio

    def update_portfolio_value(self, current_prices: Dict[str, float]) -> None:
        """Update portfolio value based on current prices."""
        if not self.portfolio:
            return

        total_value = self.portfolio.cash

        for symbol, position in self.portfolio.positions.items():
            current_price = current_prices.get(symbol, position.current_price)
            position.current_price = current_price
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.average_entry_price) * position.quantity
            total_value += position.market_value

        self.portfolio.current_value = total_value
        self.portfolio.total_return = (
            (total_value - self.portfolio.initial_capital)
            / self.portfolio.initial_capital
        )

        # Update peak and drawdown
        if total_value > self.portfolio.peak_value:
            self.portfolio.peak_value = total_value

        drawdown = (self.portfolio.peak_value - total_value) / self.portfolio.peak_value
        self.portfolio.maximum_drawdown = max(self.portfolio.maximum_drawdown, drawdown)

    def calculate_transaction_cost(
        self, symbol: str, side: OrderSide, quantity: float, price: float
    ) -> float:
        """Calculate total transaction cost including commission, spread, and slippage."""
        # Commission
        commission = self.transaction_config.commission_per_trade
        commission += self.transaction_config.commission_per_share * abs(quantity)

        # Spread cost (approximate)
        spread_cost = price * self.transaction_config.spread_percentage * abs(quantity)

        # Slippage cost
        slippage_cost = price * self.transaction_config.slippage_percentage * abs(quantity)

        # Market impact (for larger orders)
        market_impact = (
            price
            * self.transaction_config.market_impact_factor
            * abs(quantity)
            * (quantity / 1000)  # Simple scaling
        )

        total_cost = commission + spread_cost + slippage_cost + market_impact
        return total_cost

    def execute_trade(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> SimulatedTrade:
        """Execute a trade in the backtest portfolio."""
        if not self.portfolio:
            raise ValueError("Portfolio not initialized")

        # Apply slippage to execution price
        slippage_direction = 1 if side == OrderSide.BUY else -1
        slippage_amount = price * self.transaction_config.slippage_percentage
        execution_price = price + (slippage_direction * slippage_amount)

        # Calculate costs
        commission = self.transaction_config.commission_per_trade
        commission += self.transaction_config.commission_per_share * abs(quantity)
        slippage_cost = abs(price - execution_price) * abs(quantity)
        total_cost = commission + slippage_cost

        # Update portfolio
        if side == OrderSide.BUY:
            required = (execution_price * quantity) + total_cost
            if required > self.portfolio.cash:
                raise ValueError(f"Insufficient cash: need {required}, have {self.portfolio.cash}")

            self.portfolio.cash -= required

            if symbol in self.portfolio.positions:
                # Average into existing position
                existing = self.portfolio.positions[symbol]
                total_quantity = existing.quantity + quantity
                avg_price = (
                    (existing.average_entry_price * existing.quantity)
                    + (execution_price * quantity)
                ) / total_quantity
                existing.quantity = total_quantity
                existing.average_entry_price = avg_price
            else:
                self.portfolio.positions[symbol] = SimulatedPosition(
                    symbol=symbol,
                    quantity=quantity,
                    average_entry_price=execution_price,
                    entry_timestamp=timestamp,
                    current_price=execution_price,
                    market_value=execution_price * quantity,
                    unrealized_pnl=0.0,
                )

        else:  # SELL
            if symbol not in self.portfolio.positions:
                raise ValueError(f"No position in {symbol}")

            position = self.portfolio.positions[symbol]
            if quantity > position.quantity:
                raise ValueError(f"Cannot sell more than held: {quantity} > {position.quantity}")

            proceeds = (execution_price * quantity) - total_cost
            self.portfolio.cash += proceeds

            if quantity == position.quantity:
                del self.portfolio.positions[symbol]
            else:
                position.quantity -= quantity
                position.market_value = position.quantity * execution_price
                position.unrealized_pnl = (
                    (execution_price - position.average_entry_price) * position.quantity
                )

        # Record trade
        trade = SimulatedTrade(
            trade_id=generate_id("trade"),
            symbol=symbol,
            side=side,
            quantity=quantity,
            execution_price=execution_price,
            timestamp=timestamp,
            commission=commission,
            slippage=slippage_cost,
            total_cost=total_cost,
        )

        self.portfolio.trades.append(trade)
        return trade

    def check_risk_rules(
        self, proposal: InvestmentProposal, portfolio_state: PortfolioRiskState
    ) -> bool:
        """Check if an investment proposal violates risk rules."""
        # Simple check using risk policy
        position_pct = (
            proposal.proposed_position_value / portfolio_state.portfolio_value
        )

        if position_pct > self.risk_policy.max_single_position_pct:
            return False

        if portfolio_state.available_cash < (
            portfolio_state.portfolio_value * self.risk_policy.min_cash_reserve_pct
        ):
            return False

        return True

    def run_backtest(
        self,
        strategy: Dict[str, Any],
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
    ) -> Dict[str, Any]:
        """Run a backtest for a strategy on a symbol."""
        # Initialize portfolio
        self.initialize_portfolio(initial_capital)

        # Get historical bars
        bars = self.market_data_provider.get_historical_bars(
            symbol, "1D", start_date, end_date, limit=1000
        )

        if not bars:
            return {"error": "No historical data available"}

        results = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "trades": [],
            "portfolio_values": [],
        }

        for bar in bars:
            self.current_timestamp = bar.timestamp

            # Update portfolio value
            self.update_portfolio_value({symbol: bar.close})

            results["portfolio_values"].append(
                {
                    "timestamp": bar.timestamp,
                    "value": self.portfolio.current_value,
                    "cash": self.portfolio.cash,
                }
            )

            # Simple strategy: buy on green bar, sell on red bar
            # (This is a placeholder - real strategies would be more sophisticated)
            if bar.close > bar.open:  # Green bar
                try:
                    # Try to buy
                    quantity = int(
                        (self.portfolio.cash * 0.1) / bar.close
                    )  # Use 10% of cash
                    if quantity > 0:
                        trade = self.execute_trade(
                            symbol, OrderSide.BUY, quantity, bar.close, bar.timestamp
                        )
                        results["trades"].append(
                            {
                                "side": "BUY",
                                "quantity": quantity,
                                "price": bar.close,
                                "timestamp": bar.timestamp,
                            }
                        )
                except ValueError:
                    pass  # Insufficient cash
            elif bar.close < bar.open:  # Red bar
                try:
                    # Try to sell
                    if symbol in self.portfolio.positions:
                        position = self.portfolio.positions[symbol]
                        quantity = int(position.quantity * 0.5)  # Sell half
                        if quantity > 0:
                            trade = self.execute_trade(
                                symbol, OrderSide.SELL, quantity, bar.close, bar.timestamp
                            )
                            results["trades"].append(
                                {
                                    "side": "SELL",
                                    "quantity": quantity,
                                    "price": bar.close,
                                    "timestamp": bar.timestamp,
                                }
                            )
                except ValueError:
                    pass  # No position

        # Final update
        self.update_portfolio_value({symbol: bars[-1].close if bars else 0})

        results["final_capital"] = self.portfolio.current_value
        results["total_return"] = self.portfolio.total_return
        results["maximum_drawdown"] = self.portfolio.maximum_drawdown
        results["number_of_trades"] = len(self.portfolio.trades)

        return results
