import unittest
from datetime import datetime, timezone
from app.core.models.market import Quote, Trade, Bar, MarketClock
from app.utils.time import now_utc

class TestMarketModels(unittest.TestCase):
    def test_quote_creation(self):
        now = now_utc()
        q = Quote(symbol="AAPL", bid_price=150.0, ask_price=150.10, bid_size=100, ask_size=200, timestamp=now)
        self.assertEqual(q.symbol, "AAPL")
        self.assertEqual(q.bid_price, 150.0)
        self.assertEqual(q.ask_price, 150.10)

    def test_trade_creation(self):
        now = now_utc()
        t = Trade(symbol="NVDA", price=120.5, size=50, timestamp=now)
        self.assertEqual(t.symbol, "NVDA")
        self.assertEqual(t.price, 120.5)

    def test_bar_creation(self):
        now = now_utc()
        b = Bar(symbol="SPY", timestamp=now, open=500.0, high=505.0, low=498.0, close=502.0, volume=100000)
        self.assertEqual(b.close, 502.0)

    def test_market_clock(self):
        now = now_utc()
        c = MarketClock(timestamp=now, is_open=True, next_open=now, next_close=now)
        self.assertTrue(c.is_open)

if __name__ == "__main__":
    unittest.main()
