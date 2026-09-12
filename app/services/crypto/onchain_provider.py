"""On-chain data provider abstraction.

Provides blockchain metrics like active addresses, transaction volume,
exchange flows, and network activity.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from app.core.models.crypto import OnChainMetrics


class OnChainDataProvider(ABC):
    """Abstract interface for fetching on-chain blockchain data.

    Not all crypto assets support on-chain metrics (e.g., centralized
    stablecoins may not have transparent on-chain data). Providers
    should return None or INSUFFICIENT_DATA when unavailable.
    """

    @abstractmethod
    def get_on_chain_metrics(self, symbol: str) -> Optional[OnChainMetrics]:
        """Fetch on-chain metrics for a crypto symbol."""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Return list of supported crypto symbols for on-chain data."""
        pass
