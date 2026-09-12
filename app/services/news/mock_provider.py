from typing import List
from app.core.models.news import NewsItem
from app.services.news.provider import NewsProvider
from app.utils.ids import generate_id
from app.utils.time import now_utc

class MockNewsProvider(NewsProvider):
    """In-memory mock news provider for testing and agent feeds."""

    def __init__(self, predefined_items: List[NewsItem] | None = None):
        self._news: List[NewsItem] = predefined_items or []

    def add_news_item(self, item: NewsItem) -> None:
        self._news.append(item)

    def get_latest_news(self, limit: int = 20) -> List[NewsItem]:
        if not self._news:
            return [
                NewsItem(
                    news_id=generate_id("news"),
                    timestamp=now_utc(),
                    headline="Tech stocks surge amid new chip advancements",
                    summary="Major semiconductor players reported record quarterly demand.",
                    source="Financial Times Mock",
                    url="https://example.com/tech-surge",
                    related_symbols=["NVDA", "AAPL"]
                )
            ]
        return self._news[:limit]

    def get_news_for_symbol(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        matches = [n for n in self._news if symbol in n.related_symbols]
        if not matches:
            return [
                NewsItem(
                    news_id=generate_id("news"),
                    timestamp=now_utc(),
                    headline=f"Quarterly earnings update for {symbol}",
                    summary=f"Analyst consensus remains cautiously optimistic on {symbol}.",
                    source="Mock Wire",
                    url=f"https://example.com/earnings/{symbol}",
                    related_symbols=[symbol]
                )
            ]
        return matches[:limit]
