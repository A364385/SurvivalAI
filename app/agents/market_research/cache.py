from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from app.core.models.market import Bar
from app.utils.time import now_utc


@dataclass(frozen=True)
class CachedBarSeries:
    """Cached historical bars with retrieval time distinct from market timestamps."""
    bars: List[Bar]
    retrieved_at: datetime
    data_timestamp: Optional[datetime]


class MarketDataCache:
    """In-process TTL cache. Expired entries are never returned as current data."""

    def __init__(self, ttl_seconds: int = 60):
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, CachedBarSeries] = {}

    def _key(self, symbol: str, timeframe: str, limit: int) -> str:
        return f"{symbol}:{timeframe}:{limit}"

    def get(self, symbol: str, timeframe: str, limit: int = 0) -> Optional[CachedBarSeries]:
        key = self._key(symbol, timeframe, limit)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if now_utc() - entry.retrieved_at >= self.ttl:
            del self._cache[key]
            return None
        return entry

    def set(self, symbol: str, timeframe: str, bars: List[Bar], limit: int = 0) -> CachedBarSeries:
        key = self._key(symbol, timeframe, limit)
        data_timestamp = bars[-1].timestamp if bars else None
        entry = CachedBarSeries(
            bars=bars,
            retrieved_at=now_utc(),
            data_timestamp=data_timestamp,
        )
        self._cache[key] = entry
        return entry

    def clear(self) -> None:
        self._cache.clear()
