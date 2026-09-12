import unittest
from datetime import datetime, timedelta

from app.backtesting.engine import (
    BacktestEngine,
    TransactionCostConfig,
    SimulatedPortfolio,
    DataLeakageProtection,
)
from app.core.models.execution import OrderSide
from app.core.models.market import Bar
from app.services.market_data.mock_provider import MockMarketDataProvider
from app.utils.time import now_utc


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        self.now = now_utc()
        self.market_data = MockMarketDataProvider()
        self.engine = BacktestEngine(
            market_data_provider=self.market_data,
            transaction_config=TransactionCostConfig(),
        )

    def test_initialize_portfolio(self):
        """Test portfolio initialization."""
        portfolio = self.engine.initialize_portfolio(10000.0)

        self.assertEqual(portfolio.cash, 10000.0)
        self.assertEqual(portfolio.initial_capital, 10000.0)
        self.assertEqual(portfolio.current_value, 10000.0)
        self.assertEqual(portfolio.total_return, 0.0)
        self.assertEqual(len(portfolio.positions), 0)

    def test_calculate_transaction_cost(self):
        """Test transaction cost calculation."""
        cost = self.engine.calculate_transaction_cost(
            "AAPL", OrderSide.BUY, 100.0, 150.0
        )

        # Commission: 1.0 + 100 * 0.01 = 2.0
        # Spread: 150.0 * 0.001 * 100 = 15.0
        # Slippage: 150.0 * 0.0005 * 100 = 7.5
        # Market impact: 150.0 * 0.0001 * 100 * 0.1 = 0.15
        expected = 2.0 + 15.0 + 7.5 + 0.15
        self.assertAlmostEqual(cost, expected, places=2)

    def test_execute_buy_trade(self):
        """Test executing a buy trade."""
        self.engine.initialize_portfolio(10000.0)

        trade = self.engine.execute_trade(
            "AAPL", OrderSide.BUY, 10.0, 150.0, self.now
        )

        self.assertEqual(trade.symbol, "AAPL")
        self.assertEqual(trade.side, OrderSide.BUY)
        self.assertEqual(trade.quantity, 10.0)
        self.assertIn("AAPL", self.engine.portfolio.positions)
        self.assertLess(self.engine.portfolio.cash, 10000.0)

    def test_execute_sell_trade(self):
        """Test executing a sell trade."""
        self.engine.initialize_portfolio(10000.0)

        # First buy
        self.engine.execute_trade("AAPL", OrderSide.BUY, 10.0, 150.0, self.now)

        # Then sell
        trade = self.engine.execute_trade("AAPL", OrderSide.SELL, 5.0, 155.0, self.now)

        self.assertEqual(trade.symbol, "AAPL")
        self.assertEqual(trade.side, OrderSide.SELL)
        self.assertEqual(trade.quantity, 5.0)
        self.assertEqual(self.engine.portfolio.positions["AAPL"].quantity, 5.0)

    def test_insufficient_cash_for_buy(self):
        """Test that buy fails with insufficient cash."""
        self.engine.initialize_portfolio(100.0)

        with self.assertRaises(ValueError):
            self.engine.execute_trade("AAPL", OrderSide.BUY, 1000.0, 150.0, self.now)

    def test_sell_without_position(self):
        """Test that sell fails without position."""
        self.engine.initialize_portfolio(10000.0)

        with self.assertRaises(ValueError):
            self.engine.execute_trade("AAPL", OrderSide.SELL, 10.0, 150.0, self.now)

    def test_update_portfolio_value(self):
        """Test updating portfolio value."""
        self.engine.initialize_portfolio(10000.0)
        self.engine.execute_trade("AAPL", OrderSide.BUY, 10.0, 150.0, self.now)

        self.engine.update_portfolio_value({"AAPL": 155.0})

        self.assertGreater(self.engine.portfolio.current_value, 10000.0)
        self.assertEqual(self.engine.portfolio.positions["AAPL"].current_price, 155.0)

    def test_maximum_drawdown_tracking(self):
        """Test maximum drawdown tracking."""
        self.engine.initialize_portfolio(10000.0)

        # Simulate value drop
        self.engine.update_portfolio_value({"AAPL": 100.0})
        self.engine.portfolio.current_value = 8000.0
        self.engine.portfolio.peak_value = 10000.0

        drawdown = (10000.0 - 8000.0) / 10000.0
        self.engine.portfolio.maximum_drawdown = drawdown

        self.assertEqual(self.engine.portfolio.maximum_drawdown, 0.2)

    def test_data_leakage_protection(self):
        """Test data leakage protection."""
        current_time = self.now
        protection = DataLeakageProtection(current_time)

        # Past data should be available
        past_time = current_time - timedelta(hours=1)
        self.assertTrue(protection.is_data_available(past_time))

        # Future data should not be available
        future_time = current_time + timedelta(hours=1)
        self.assertFalse(protection.is_data_available(future_time))

    def test_filter_future_data(self):
        """Test filtering future data."""
        current_time = self.now
        protection = DataLeakageProtection(current_time)

        from dataclasses import dataclass

        @dataclass
        class TestData:
            timestamp: datetime
            value: float

        data = [
            TestData(current_time - timedelta(hours=1), 100.0),
            TestData(current_time + timedelta(hours=1), 200.0),
            TestData(current_time - timedelta(hours=2), 90.0),
        ]

        available = protection.filter_future_data(data, "timestamp")

        self.assertEqual(len(available), 2)  # Only past data
        self.assertEqual(available[0].value, 100.0)
        self.assertEqual(available[1].value, 90.0)

    def test_check_risk_rules_position_size(self):
        """Test risk rule check for position size."""
        from app.core.models.risk import InvestmentProposal, PortfolioRiskState

        proposal = InvestmentProposal(
            proposal_id="prop_1",
            asset="AAPL",
            proposed_position_value=20000.0,  # 20% of 100k
            asset_class="EQUITY",
        )

        portfolio_state = PortfolioRiskState(
            portfolio_value=100000.0,
            available_cash=50000.0,
            positions=[],
        )

        # Should pass with default policy (10% max)
        result = self.engine.check_risk_rules(proposal, portfolio_state)
        self.assertFalse(result)  # 20% > 10% max

    def test_check_risk_rules_cash_reserve(self):
        """Test risk rule check for cash reserve."""
        from app.core.models.risk import InvestmentProposal, PortfolioRiskState

        proposal = InvestmentProposal(
            proposal_id="prop_1",
            asset="AAPL",
            proposed_position_value=80000.0,
            asset_class="EQUITY",
        )

        portfolio_state = PortfolioRiskState(
            portfolio_value=100000.0,
            available_cash=90000.0,
            positions=[],
        )

        # Should fail - not enough cash reserve after investment
        result = self.engine.check_risk_rules(proposal, portfolio_state)
        self.assertFalse(result)

    def test_transaction_cost_config_defaults(self):
        """Test default transaction cost configuration."""
        config = TransactionCostConfig()

        self.assertEqual(config.commission_per_trade, 1.0)
        self.assertEqual(config.commission_per_share, 0.01)
        self.assertEqual(config.spread_percentage, 0.001)
        self.assertEqual(config.slippage_percentage, 0.0005)


if __name__ == "__main__":
    unittest.main()
