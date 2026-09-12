from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
from app.core.models.market import Quote, Trade, Bar, MarketClock

class MarketDataProvider(ABC):
    """Abstract interface for fetching market data. Isolates external providers."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Fetch current best bid and ask quote for a symbol."""
        pass

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Fetch quotes for multiple symbols."""
        pass

    @abstractmethod
    def get_latest_trade(self, symbol: str) -> Trade:
        """Fetch latest executed trade for a symbol."""
        pass

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Bar]:
        """Fetch historical bars (OHLCV) for a symbol."""
        pass

    @abstractmethod
    def get_market_clock(self) -> MarketClock:
        """Fetch current market open status and transition times."""
        pass

    @abstractmethod
    def get_market_status(self) -> bool:
        """Quick check if market is currently open."""
        pass
