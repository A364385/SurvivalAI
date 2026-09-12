"""Crypto tokenomics data provider abstraction.

Provides token economic metrics like supply, inflation, market cap,
and unlock schedules.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from app.core.models.crypto import (
    CryptoTokenomics,
    TokenUnlock,
)


class CryptoFundamentalDataProvider(ABC):
    """Abstract interface for fetching crypto tokenomics data."""

    @abstractmethod
    def get_tokenomics(self, symbol: str) -> Optional[CryptoTokenomics]:
        """Fetch token economic metrics for a crypto symbol."""
        pass

    @abstractmethod
    def get_token_unlocks(self, symbol: str) -> List[TokenUnlock]:
        """Fetch scheduled token unlock events for a crypto symbol."""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Return list of supported crypto symbols for tokenomics data."""
        pass
