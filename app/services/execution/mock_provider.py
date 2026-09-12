from datetime import datetime
from typing import Dict, List, Optional
from app.core.models.execution import (
    ExecutionEnvironment, OrderSide, OrderType, TimeInForce,
    OrderStatus, OrderRequest, OrderResponse, AccountState, PositionState
)
from app.core.models.provider_errors import OrderNotFoundError, OrderRejectedError
from app.services.execution.provider import ExecutionProvider
from app.utils.ids import generate_id
from app.utils.time import now_utc

class MockExecutionProvider(ExecutionProvider):
    """In-memory mock execution provider for fast, reliable unit testing."""

    def __init__(self, initial_cash: float = 100000.0):
        self._orders: Dict[str, OrderResponse] = {}
        self._positions: Dict[str, PositionState] = {}
        self._cash: float = initial_cash
        self._equity: float = initial_cash
        self.auto_fill = True

    def submit_order(self, request: OrderRequest) -> OrderResponse:
        if request.environment != ExecutionEnvironment.PAPER:
            raise OrderRejectedError("Only PAPER execution is permitted.", provider_name="MockExecution")

        order_id = generate_id("mock_order")
        now = now_utc()
        
        status = OrderStatus.SUBMITTED
        filled_qty = 0.0
        avg_price = None
        filled_at = None

        if self.auto_fill:
            status = OrderStatus.FILLED
            filled_qty = request.quantity
            avg_price = request.limit_price or 100.0
            filled_at = now
            cost = filled_qty * avg_price

            if request.side == OrderSide.BUY:
                self._cash -= cost
                curr_pos = self._positions.get(request.symbol)
                if curr_pos:
                    new_qty = curr_pos.quantity + filled_qty
                    new_avg = ((curr_pos.average_entry_price * curr_pos.quantity) + cost) / new_qty
                    self._positions[request.symbol] = PositionState(
                        symbol=request.symbol,
                        quantity=new_qty,
                        average_entry_price=new_avg,
                        current_price=avg_price,
                        market_value=new_qty * avg_price,
                        unrealized_profit_loss=0.0,
                        unrealized_profit_loss_percentage=0.0
                    )
                else:
                    self._positions[request.symbol] = PositionState(
                        symbol=request.symbol,
                        quantity=filled_qty,
                        average_entry_price=avg_price,
                        current_price=avg_price,
                        market_value=cost,
                        unrealized_profit_loss=0.0,
                        unrealized_profit_loss_percentage=0.0
                    )
            elif request.side == OrderSide.SELL:
                self._cash += cost
                curr_pos = self._positions.get(request.symbol)
                if curr_pos:
                    remaining_qty = curr_pos.quantity - filled_qty
                    if remaining_qty <= 0:
                        del self._positions[request.symbol]
                    else:
                        self._positions[request.symbol] = PositionState(
                            symbol=request.symbol,
                            quantity=remaining_qty,
                            average_entry_price=curr_pos.average_entry_price,
                            current_price=avg_price,
                            market_value=remaining_qty * avg_price,
                            unrealized_profit_loss=0.0,
                            unrealized_profit_loss_percentage=0.0
                        )

        resp = OrderResponse(
            order_id=order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            filled_quantity=filled_qty,
            order_type=request.order_type,
            status=status,
            submitted_at=now,
            filled_at=filled_at,
            average_fill_price=avg_price
        )
        self._orders[order_id] = resp
        return resp

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self._orders:
            raise OrderNotFoundError(f"Order {order_id} not found.", provider_name="MockExecution")
        order = self._orders[order_id]
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False
        order.status = OrderStatus.CANCELLED
        return True

    def get_order(self, order_id: str) -> OrderResponse:
        if order_id not in self._orders:
            raise OrderNotFoundError(f"Order {order_id} not found.", provider_name="MockExecution")
        return self._orders[order_id]

    def list_orders(self, status: Optional[str] = None, limit: int = 50) -> List[OrderResponse]:
        results = list(self._orders.values())
        if status:
            results = [o for o in results if o.status == status or o.status.value == status]
        return results[:limit]

    def get_open_orders(self) -> List[OrderResponse]:
        return [o for o in self._orders.values() if o.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED)]

    def get_positions(self) -> List[PositionState]:
        return list(self._positions.values())

    def get_account(self) -> AccountState:
        pos_val = sum(p.market_value for p in self._positions.values())
        total_equity = self._cash + pos_val
        return AccountState(
            equity=total_equity,
            cash=self._cash,
            buying_power=self._cash * 2.0,
            currency="USD",
            status="ACTIVE",
            timestamp=now_utc()
        )
