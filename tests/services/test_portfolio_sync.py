import unittest
from app.core.models.execution import (
    ExecutionEnvironment, OrderSide, OrderType, TimeInForce, OrderRequest
)
from app.services.execution.mock_provider import MockExecutionProvider
from app.core.portfolio.synchronizer import PortfolioSynchronizer

class TestPortfolioSynchronizer(unittest.TestCase):
    def setUp(self):
        self.exec_provider = MockExecutionProvider(initial_cash=50000.0)
        self.sync = PortfolioSynchronizer(self.exec_provider)

    def test_initial_sync(self):
        events = self.sync.synchronize()
        self.assertEqual(self.sync.portfolio.equity, 50000.0)
        self.assertEqual(self.sync.portfolio.cash, 50000.0)
        self.assertEqual(len(self.sync.portfolio.positions), 0)
        self.assertTrue(any(e.event_type == "AccountUpdated" for e in events))

    def test_sync_after_fill(self):
        req = OrderRequest(
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            client_order_id="p_sync_1",
            limit_price=300.0
        )
        self.exec_provider.submit_order(req)
        events = self.sync.synchronize()

        self.assertEqual(self.sync.portfolio.cash, 50000.0 - 3000.0)
        self.assertIn("MSFT", self.sync.portfolio.positions)
        self.assertEqual(self.sync.portfolio.positions["MSFT"].quantity, 10)
        self.assertTrue(any(e.event_type == "PositionUpdated" for e in events))

if __name__ == "__main__":
    unittest.main()
