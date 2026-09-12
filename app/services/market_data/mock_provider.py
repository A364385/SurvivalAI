from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from app.core.models.market import Quote, Trade, Bar, MarketClock
from app.core.models.provider_errors import (
    AuthenticationError, APIUnavailableError, RateLimitError,
)
from app.services.market_data.provider import MarketDataProvider
from app.utils.time import now_utc

class MockMarketDataProvider(MarketDataProvider):
    """In-memory mock provider for testing and offline development."""

    def __init__(self, is_open: bool = True):
        self._quotes: Dict[str, Quote] = {}
        self._trades: Dict[str, Trade] = {}
        self._bars: Dict[str, List[Bar]] = {}
        self._is_open = is_open
        self.should_fail_rate_limit = False
        self.should_fail_unavailable = False
        self.should_fail_auth = False
        self.historical_call_count = 0

    def set_quote(self, quote: Quote) -> None:
        self._quotes[quote.symbol] = quote

    def set_trade(self, trade: Trade) -> None:
        self._trades[trade.symbol] = trade

    def set_bars(self, symbol: str, bars: List[Bar]) -> None:
        self._bars[symbol] = bars

    def set_market_open(self, is_open: bool) -> None:
        self._is_open = is_open

    def get_quote(self, symbol: str) -> Quote:
        self._raise_injected_failure()
        if symbol in self._quotes:
            return self._quotes[symbol]
        # Return sensible mock default
        return Quote(
            symbol=symbol,
            bid_price=100.0,
            ask_price=100.05,
            bid_size=10,
            ask_size=10,
            timestamp=now_utc()
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        return {s: self.get_quote(s) for s in symbols}

    def get_latest_trade(self, symbol: str) -> Trade:
        if symbol in self._trades:
            return self._trades[symbol]
        return Trade(
            symbol=symbol,
            price=100.0,
            size=5,
            timestamp=now_utc()
        )

    def _raise_injected_failure(self) -> None:
        if self.should_fail_rate_limit:
            raise RateLimitError("Simulated market-data rate limit.", provider_name="MockMarketData")
        if self.should_fail_unavailable:
            raise APIUnavailableError("Simulated market-data outage.", provider_name="MockMarketData")
        if self.should_fail_auth:
            raise AuthenticationError("Simulated market-data auth failure.", provider_name="MockMarketData")

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Bar]:
        self.historical_call_count += 1
        self._raise_injected_failure()
        if symbol in self._bars:
            return self._bars[symbol][:limit]
        return [
            Bar(
                symbol=symbol,
                timestamp=now_utc() - timedelta(minutes=i),
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000
            ) for i in range(min(5, limit))
        ]

    def get_market_clock(self) -> MarketClock:
        now = now_utc()
        return MarketClock(
            timestamp=now,
            is_open=self._is_open,
            next_open=now + timedelta(hours=8),
            next_close=now + timedelta(hours=6)
        )

    def get_market_status(self) -> bool:
        return self._is_open
