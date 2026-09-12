from abc import ABC, abstractmethod
from typing import List, Optional
from app.core.models.execution import (
    OrderRequest, OrderResponse, AccountState, PositionState
)

class ExecutionProvider(ABC):
    """Abstract interface for paper execution and portfolio retrieval."""

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> OrderResponse:
        """Submit a new paper order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing open order."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> OrderResponse:
        """Fetch current status of an order."""
        pass

    @abstractmethod
    def list_orders(self, status: Optional[str] = None, limit: int = 50) -> List[OrderResponse]:
        """List orders matching status."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[OrderResponse]:
        """Fetch all currently open/unfilled orders."""
        pass

    @abstractmethod
    def get_positions(self) -> List[PositionState]:
        """Fetch all active positions held in paper account."""
        pass

    @abstractmethod
    def get_account(self) -> AccountState:
        """Fetch current account balances and buying power."""
        pass
