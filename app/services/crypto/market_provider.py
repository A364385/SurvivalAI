"""Crypto market data provider abstraction.

Extends the general MarketDataProvider with crypto-specific capabilities
like 24/7 trading, BTC dominance, and crypto market-wide metrics.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from app.core.models.market import Quote, Trade, Bar, MarketClock
from app.core.models.crypto import (
    CryptoMarketWideConditions,
    CryptoAssetCorrelation,
)


class CryptoMarketDataProvider(ABC):
    """Abstract interface for fetching crypto market data.

    Crypto markets operate 24/7, so some assumptions from traditional
    market data providers (like market hours) need adjustment.
    """

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Fetch current best bid and ask quote for a crypto symbol."""
        pass

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Fetch quotes for multiple crypto symbols."""
        pass

    @abstractmethod
    def get_latest_trade(self, symbol: str) -> Trade:
        """Fetch latest executed trade for a crypto symbol."""
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
        """Fetch historical bars (OHLCV) for a crypto symbol."""
        pass

    @abstractmethod
    def get_market_clock(self) -> MarketClock:
        """Fetch current market clock (crypto markets are 24/7)."""
        pass

    @abstractmethod
    def get_market_status(self) -> bool:
        """Quick check if market is currently open (always True for crypto)."""
        pass

    @abstractmethod
    def get_market_wide_conditions(self) -> CryptoMarketWideConditions:
        """Fetch market-wide crypto conditions (total market cap, BTC dominance, etc.)."""
        pass

    @abstractmethod
    def get_correlation(self, asset_pair: str) -> Optional[CryptoAssetCorrelation]:
        """Fetch correlation between two crypto assets (e.g., BTC-ETH)."""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Return list of supported crypto symbols."""
        pass
