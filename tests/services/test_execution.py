import unittest
from app.core.models.execution import (
    ExecutionEnvironment, OrderSide, OrderType, TimeInForce,
    OrderStatus, OrderRequest
)
from app.core.models.provider_errors import OrderNotFoundError
from app.services.execution.mock_provider import MockExecutionProvider

class TestExecutionService(unittest.TestCase):
    def setUp(self):
        self.provider = MockExecutionProvider(initial_cash=100000.0)

    def test_submit_buy_order(self):
        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            client_order_id="cli_1",
            limit_price=150.0,
            environment=ExecutionEnvironment.PAPER
        )
        resp = self.provider.submit_order(req)
        self.assertEqual(resp.status, OrderStatus.FILLED)
        self.assertEqual(resp.filled_quantity, 10)
        self.assertEqual(resp.symbol, "AAPL")

        # Verify account cash deducted
        acc = self.provider.get_account()
        self.assertEqual(acc.cash, 100000.0 - (10 * 150.0))

        # Verify position created
        positions = self.provider.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].symbol, "AAPL")
        self.assertEqual(positions[0].quantity, 10)

    def test_submit_sell_order(self):
        # First buy
        buy_req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            client_order_id="buy_1",
            limit_price=150.0
        )
        self.provider.submit_order(buy_req)

        # Then sell half
        sell_req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            client_order_id="sell_1",
            limit_price=160.0
        )
        resp = self.provider.submit_order(sell_req)
        self.assertEqual(resp.status, OrderStatus.FILLED)

        positions = self.provider.get_positions()
        self.assertEqual(positions[0].quantity, 5)

    def test_order_not_found(self):
        with self.assertRaises(OrderNotFoundError):
            self.provider.get_order("non_existent")

if __name__ == "__main__":
    unittest.main()
