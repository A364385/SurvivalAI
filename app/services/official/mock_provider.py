from typing import List
from app.core.models.news import NewsItem
from app.services.official.provider import OfficialDataProvider


class MockOfficialDataProvider(OfficialDataProvider):
    """Returns only statements explicitly configured. Never fabricates official data."""

    def __init__(self):
        self._items: List[NewsItem] = []

    def add_statement(self, item: NewsItem) -> None:
        self._items.append(item)

    def get_official_statements(self, query: str, limit: int = 10) -> List[NewsItem]:
        q = (query or "").lower()
        if not q:
            return self._items[:limit]
        matched = [
            i for i in self._items
            if q in i.headline.lower() or q in i.summary.lower()
            or any(q in c.lower() for c in i.related_countries)
        ]
        return matched[:limit]
