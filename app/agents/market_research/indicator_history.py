"""Indicator history for the fast loop and dashboard charts.

Computes deterministic indicator snapshots (via technical_indicators) over
recent bars and keeps a bounded in-memory history per symbol. The dashboard
reads this for indicator charts; nothing here touches an LLM.
"""

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from app.agents.market_research.technical_indicators import (
    compute_indicator_snapshot,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class IndicatorHistory:
    """Bounded per-symbol indicator snapshot history (thread-safe)."""

    def __init__(self, max_points: int = 600):
        self.max_points = max_points
        self._lock = threading.Lock()
        self._history: Dict[str, Deque[Dict[str, Any]]] = {}
        self._latest_bars: Dict[str, List[Any]] = {}

    def update_symbol(
        self,
        symbol: str,
        closes: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        snapshot = compute_indicator_snapshot(closes, highs, lows, volumes)
        point = {
            "ts": time.time(),
            "price": snapshot.get("price"),
            "rsi_14": snapshot.get("rsi_14"),
            "sma_20": snapshot.get("sma_20"),
            "sma_50": snapshot.get("sma_50"),
            "macd_histogram": snapshot.get("macd_histogram"),
            "volatility_20d": snapshot.get("volatility_20d"),
            "volume_ratio": snapshot.get("volume_ratio"),
        }
        with self._lock:
            history = self._history.setdefault(symbol, deque(maxlen=self.max_points))
            history.append(point)
            self._latest_bars[symbol] = list(closes or [])
        return snapshot

    def latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            history = self._history.get(symbol)
        return dict(history[-1]) if history else None

    def series(self, symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            history = self._history.get(symbol)
        if not history:
            return []
        return list(history)[-limit:]

    def symbols(self) -> List[str]:
        with self._lock:
            return list(self._history.keys())
