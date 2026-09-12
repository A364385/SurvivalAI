import unittest
from datetime import datetime
from app.core.models.execution import (
    ExecutionEnvironment, OrderSide, OrderType, TimeInForce,
    OrderStatus, OrderRequest, OrderResponse, AccountState, PositionState
)
from app.utils.time import now_utc

class TestExecutionModels(unittest.TestCase):
    def test_order_request_valid(self):
        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            client_order_id="req_1",
            limit_price=150.0,
            environment=ExecutionEnvironment.PAPER
        )
        self.assertEqual(req.environment, ExecutionEnvironment.PAPER)

    def test_order_request_non_paper_fails(self):
        with self.assertRaises(ValueError):
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id="req_2",
                environment="LIVE" # type: ignore
            )

    def test_order_response(self):
        resp = OrderResponse(
            order_id="ord_1",
            client_order_id="req_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            filled_quantity=10,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            submitted_at=now_utc(),
            average_fill_price=150.0
        )
        self.assertEqual(resp.status, OrderStatus.FILLED)

if __name__ == "__main__":
    unittest.main()
