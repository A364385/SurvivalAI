import unittest
from app.core.models.news import NewsItem
from app.services.news.mock_provider import MockNewsProvider
from app.utils.time import now_utc

class TestNewsService(unittest.TestCase):
    def test_mock_news_defaults(self):
        provider = MockNewsProvider()
        news = provider.get_latest_news()
        self.assertTrue(len(news) > 0)
        self.assertIn("NVDA", news[0].related_symbols)

    def test_mock_news_filter_by_symbol(self):
        provider = MockNewsProvider()
        item = NewsItem(
            news_id="n1",
            timestamp=now_utc(),
            headline="Tesla launches robotaxi prototype",
            summary="Autonomous vehicle testing begins in selected markets.",
            source="Mock Wire",
            url="https://example.com/tsla",
            related_symbols=["TSLA"]
        )
        provider.add_news_item(item)
        results = provider.get_news_for_symbol("TSLA")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].headline, "Tesla launches robotaxi prototype")

if __name__ == "__main__":
    unittest.main()
