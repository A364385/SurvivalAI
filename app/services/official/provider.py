from abc import ABC, abstractmethod
from typing import List
from app.core.models.news import NewsItem


class OfficialDataProvider(ABC):
    """Future adapter for government, central-bank, and IGO statements.

    Isolated behind an interface. Implementations must not scrape arbitrary websites
    inside agents; they normalize into NewsItem records.
    """

    @abstractmethod
    def get_official_statements(self, query: str, limit: int = 10) -> List[NewsItem]:
        pass
