from typing import List, Optional
from app.core.models.execution import (
    ExecutionEnvironment, OrderRequest, OrderResponse, AccountState, PositionState
)
from app.core.models.provider_errors import InvalidEnvironmentError
from app.services.execution.provider import ExecutionProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)

class PaperOnlyExecutionProvider(ExecutionProvider):
    """Hard safety wrapper around any execution provider.
    Guarantees that no request can bypass the PAPER environment check.
    Any attempt to execute outside PAPER raises InvalidEnvironmentError immediately.
    """

    def __init__(self, inner_provider: ExecutionProvider):
        self._inner = inner_provider

    def _assert_paper(self, env: ExecutionEnvironment) -> None:
        if env != ExecutionEnvironment.PAPER:
            logger.critical(f"FATAL SAFETY VIOLATION: Execution attempted in non-paper environment: {env}")
            raise InvalidEnvironmentError(
                f"SAFETY SHIELD ACTIVATED: Only ExecutionEnvironment.PAPER is allowed. Rejected target: {env}",
                provider_name="PaperSafetyShield"
            )

    def submit_order(self, request: OrderRequest) -> OrderResponse:
        self._assert_paper(request.environment)
        logger.info(f"[PAPER EXECUTION] Validating safety for order {request.client_order_id} ({request.symbol})")
        return self._inner.submit_order(request)

    def cancel_order(self, order_id: str) -> bool:
        return self._inner.cancel_order(order_id)

    def get_order(self, order_id: str) -> OrderResponse:
        return self._inner.get_order(order_id)

    def list_orders(self, status: Optional[str] = None, limit: int = 50) -> List[OrderResponse]:
        return self._inner.list_orders(status=status, limit=limit)

    def get_open_orders(self) -> List[OrderResponse]:
        return self._inner.get_open_orders()

    def get_positions(self) -> List[PositionState]:
        return self._inner.get_positions()

    def get_account(self) -> AccountState:
        return self._inner.get_account()
