import re
from typing import Optional
from app.core.models.execution import (
    ExecutionEnvironment, OrderRequest, OrderSide, OrderType, TimeInForce, AccountState, PositionState
)
from app.core.models.provider_errors import OrderRejectedError
from app.services.execution.provider import ExecutionProvider

SYMBOL_REGEX = re.compile(r'^[A-Z]{1,6}$')

class OrderValidator:
    """Pre-submission validation layer for paper orders.
    Validates structural correctness, account buying power, and position limits.
    """

    def __init__(self, execution_provider: ExecutionProvider):
        self._provider = execution_provider

    def validate(self, request: OrderRequest) -> None:
        """Validates an order request before submission.
        Raises OrderRejectedError if validation fails.
        """
        # 1. Environment validation
        if request.environment != ExecutionEnvironment.PAPER:
            raise OrderRejectedError("Environment must be PAPER.", provider_name="OrderValidator")

        # 2. Symbol validation
        if not request.symbol or not SYMBOL_REGEX.match(request.symbol):
            raise OrderRejectedError(f"Invalid symbol format: '{request.symbol}'. Must be 1-6 uppercase letters.", provider_name="OrderValidator")

        # 3. Quantity validation
        if request.quantity <= 0:
            raise OrderRejectedError(f"Order quantity must be positive. Received: {request.quantity}", provider_name="OrderValidator")

        # 4. Side validation
        if request.side not in (OrderSide.BUY, OrderSide.SELL):
            raise OrderRejectedError(f"Unsupported order side: {request.side}", provider_name="OrderValidator")

        # 5. Order Type validation
        if request.order_type not in (OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT):
            raise OrderRejectedError(f"Unsupported order type: {request.order_type}", provider_name="OrderValidator")

        if request.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and (request.limit_price is None or request.limit_price <= 0):
            raise OrderRejectedError("Limit price required and must be positive for LIMIT orders.", provider_name="OrderValidator")

        # 6. Time in force validation
        if request.time_in_force not in (TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK):
            raise OrderRejectedError(f"Unsupported time in force: {request.time_in_force}", provider_name="OrderValidator")

        # 7. Balance / Position validation
        if request.side == OrderSide.BUY:
            account: AccountState = self._provider.get_account()
            est_cost = request.quantity * (request.limit_price or 1.0) # conservative or estimated
            if request.limit_price and est_cost > account.buying_power:
                raise OrderRejectedError(
                    f"Insufficient buying power: required estimated ${est_cost:.2f}, available ${account.buying_power:.2f}",
                    provider_name="OrderValidator"
                )
        elif request.side == OrderSide.SELL:
            positions = self._provider.get_positions()
            holding = next((p for p in positions if p.symbol == request.symbol), None)
            if not holding or holding.quantity < request.quantity:
                current_qty = holding.quantity if holding else 0.0
                raise OrderRejectedError(
                    f"Insufficient position for SELL of {request.symbol}: held {current_qty}, requested {request.quantity}",
                    provider_name="OrderValidator"
                )
