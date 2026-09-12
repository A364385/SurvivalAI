import unittest
from datetime import datetime
from app.core.models.market import Quote, Trade, Bar
from app.core.models.provider_errors import ConfigurationError
from app.services.market_data.mock_provider import MockMarketDataProvider
from app.services.market_data.alpaca_provider import AlpacaMarketDataProvider
from app.utils.time import now_utc

class TestMarketDataService(unittest.TestCase):
    def test_mock_provider_quote(self):
        provider = MockMarketDataProvider()
        quote = provider.get_quote("AAPL")
        self.assertEqual(quote.symbol, "AAPL")
        self.assertGreater(quote.ask_price, quote.bid_price)

    def test_mock_provider_custom_quote(self):
        provider = MockMarketDataProvider()
        custom = Quote(symbol="TSLA", bid_price=200.0, ask_price=200.5, bid_size=5, ask_size=5, timestamp=now_utc())
        provider.set_quote(custom)
        fetched = provider.get_quote("TSLA")
        self.assertEqual(fetched.bid_price, 200.0)

    def test_mock_provider_bars(self):
        provider = MockMarketDataProvider()
        bars = provider.get_historical_bars("MSFT", "1Min", now_utc(), limit=5)
        self.assertEqual(len(bars), 5)
        self.assertEqual(bars[0].symbol, "MSFT")

    def test_mock_provider_clock(self):
        provider = MockMarketDataProvider(is_open=False)
        clock = provider.get_market_clock()
        self.assertFalse(clock.is_open)
        self.assertFalse(provider.get_market_status())

    def test_alpaca_missing_credentials(self):
        # Should raise ConfigurationError when no keys provided in env or init
        with self.assertRaises(ConfigurationError):
            AlpacaMarketDataProvider(api_key="", api_secret="")

if __name__ == "__main__":
    unittest.main()
