import unittest
from app.core.models.provider_errors import ConfigurationError
from app.services.news.alpaca_provider import AlpacaNewsProvider

class TestAlpacaNewsProvider(unittest.TestCase):
    def test_missing_credentials_raises(self):
        with self.assertRaises(ConfigurationError):
            AlpacaNewsProvider(api_key="", api_secret="")

    def test_initialization_with_credentials(self):
        provider = AlpacaNewsProvider(api_key="mock_key", api_secret="mock_secret")
        self.assertIsNotNone(provider)
        headers = provider._get_headers()
        self.assertEqual(headers["APCA-API-KEY-ID"], "mock_key")
        self.assertEqual(headers["APCA-API-SECRET-KEY"], "mock_secret")

if __name__ == "__main__":
    unittest.main()
