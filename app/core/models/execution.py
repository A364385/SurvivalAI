from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class ExecutionEnvironment(str, Enum):
    """Execution environment. Only PAPER is allowed by design."""
    PAPER = "PAPER"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class OrderStatus(str, Enum):
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


@dataclass
class OrderRequest:
    """Client request to place a paper order."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    time_in_force: TimeInForce
    client_order_id: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    environment: ExecutionEnvironment = ExecutionEnvironment.PAPER

    def __post_init__(self):
        if self.environment != ExecutionEnvironment.PAPER:
            raise ValueError(f"Execution environment must be PAPER, got {self.environment}")


@dataclass
class OrderResponse:
    """Provider response acknowledging or detailing an order."""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    filled_quantity: float
    order_type: OrderType
    status: OrderStatus
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    average_fill_price: Optional[float] = None


@dataclass
class AccountState:
    """Snapshot of account cash, equity, and buying power."""
    equity: float
    cash: float
    buying_power: float
    currency: str
    status: str
    timestamp: datetime


@dataclass
class PositionState:
    """Current holding in an asset."""
    symbol: str
    quantity: float
    average_entry_price: float
    current_price: float
    market_value: float
    unrealized_profit_loss: float
    unrealized_profit_loss_percentage: float
