from abc import ABC, abstractmethod
from typing import Dict, Optional


class MacroDataProvider(ABC):
    """Future adapter for inflation, rates, and other macro series. No fabricated values."""

    @abstractmethod
    def get_indicator(self, name: str, region: str = "GLOBAL") -> Optional[Dict]:
        """Return a normalized indicator payload or None if unavailable."""
        pass
