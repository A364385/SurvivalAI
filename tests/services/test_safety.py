import unittest
from app.core.models.execution import (
    ExecutionEnvironment, OrderSide, OrderType, TimeInForce, OrderRequest
)
from app.core.models.provider_errors import InvalidEnvironmentError
from app.services.execution.mock_provider import MockExecutionProvider
from app.services.execution.safety import PaperOnlyExecutionProvider
from app.services.execution.alpaca_provider import AlpacaPaperExecutionProvider

class TestHardSafetyBoundary(unittest.TestCase):
    def setUp(self):
        self.mock_exec = MockExecutionProvider()
        self.safety_shield = PaperOnlyExecutionProvider(self.mock_exec)

    def test_paper_execution_allowed(self):
        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id="safe_1",
            environment=ExecutionEnvironment.PAPER
        )
        resp = self.safety_shield.submit_order(req)
        self.assertIsNotNone(resp.order_id)

    def test_live_execution_is_rejected(self):
        # Attempt to pass an invalid environment
        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id="unsafe_1",
            environment=ExecutionEnvironment.PAPER
        )
        # Force modify environment attribute to simulate malicious or accidental live bypass
        object.__setattr__(req, "environment", "LIVE")
        
        with self.assertRaises(InvalidEnvironmentError):
            self.safety_shield.submit_order(req)

    def test_alpaca_rejects_live_url(self):
        with self.assertRaises(InvalidEnvironmentError):
            AlpacaPaperExecutionProvider(
                api_key="mock_key",
                api_secret="mock_secret",
                base_url="https://api.alpaca.markets/v2" # Live URL rejected!
            )

if __name__ == "__main__":
    unittest.main()
