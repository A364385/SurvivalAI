import unittest
from app.core.models.execution import (
    ExecutionEnvironment, OrderSide, OrderType, TimeInForce, OrderRequest
)
from app.core.models.provider_errors import OrderRejectedError
from app.services.execution.mock_provider import MockExecutionProvider
from app.services.execution.validator import OrderValidator

class TestOrderValidator(unittest.TestCase):
    def setUp(self):
        self.exec_provider = MockExecutionProvider(initial_cash=500.0)
        self.validator = OrderValidator(self.exec_provider)

    def test_valid_order_passes(self):
        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=2,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            client_order_id="val_1",
            limit_price=100.0
        )
        # Should not raise
        self.validator.validate(req)

    def test_invalid_symbol_rejected(self):
        req = OrderRequest(
            symbol="bad_symbol_123",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            client_order_id="val_2",
            limit_price=100.0
        )
        with self.assertRaises(OrderRejectedError):
            self.validator.validate(req)

    def test_negative_quantity_rejected(self):
        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=-5,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            client_order_id="val_3",
            limit_price=100.0
        )
        with self.assertRaises(OrderRejectedError):
            self.validator.validate(req)

    def test_insufficient_buying_power_rejected(self):
        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            client_order_id="val_4",
            limit_price=200.0 # Requires $20,000, account has $1,000 buying power
        )
        with self.assertRaises(OrderRejectedError):
            self.validator.validate(req)

    def test_insufficient_position_for_sell_rejected(self):
        req = OrderRequest(
            symbol="NVDA",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            client_order_id="val_5",
            limit_price=120.0
        )
        with self.assertRaises(OrderRejectedError):
            self.validator.validate(req)

if __name__ == "__main__":
    unittest.main()
