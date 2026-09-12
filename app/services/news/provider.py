from abc import ABC, abstractmethod
from typing import List, Optional
from app.core.models.news import NewsItem

class NewsProvider(ABC):
    """Abstract interface for normalized news retrieval."""

    @abstractmethod
    def get_latest_news(self, limit: int = 20) -> List[NewsItem]:
        """Fetch latest market and general news items."""
        pass

    @abstractmethod
    def get_news_for_symbol(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        """Fetch news specifically mentioning an asset symbol."""
        pass
