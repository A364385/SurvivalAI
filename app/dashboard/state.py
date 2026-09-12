"""Central dashboard state hub.

Holds the live state the dashboard renders, in-process (single local user):

- equity / P/L / drawdown / cash history for charts
- survival state, generation info
- control flags: fast loop, research loop, paper trading, learning,
  generation evolution, per-agent on/off
- provider settings (secrets NEVER returned unmasked)
- SSE subscriber registry for efficient live updates (no full-page reloads)
"""

import json
import queue
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set

from app.utils.time import now_utc

# Agents that may be toggled (safety-critical ones are NOT in this list).
TOGGLEABLE_AGENTS = [
    "news_research", "market_research", "deep_looker", "crisis_risk",
    "crypto_research", "strategy_updater",
]
# Safety-critical components that must never be disableable.
PROTECTED_COMPONENTS = [
    "risk_manager", "investment_safety", "capital_protection",
    "paper_trading_safety", "generation_system",
]


class DashboardState:
    """Thread-safe in-process state store + SSE broadcast hub."""

    def __init__(self, max_history: int = 1440):
        self._lock = threading.RLock()
        self.max_history = max_history
        # Time-series (bounded ring buffers)
        self.equity_history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.daily_pl_history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.drawdown_history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.cash_history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._peak_equity: float = 0.0
        # Flags
        self.fast_loop_enabled: bool = True
        self.research_loop_enabled: bool = True
        self.paper_trading_enabled: bool = True
        self.learning_enabled: bool = True
        self.generation_evolution_enabled: bool = True
        self.agent_toggles: Dict[str, bool] = {a: True for a in TOGGLEABLE_AGENTS}
        # Provider settings — secrets stored masked; raw kept only in-process
        # for connection tests, never returned by the API, never logged.
        self.provider_settings: Dict[str, Dict[str, Any]] = {}
        self._sse_queues: Set["queue.Queue"] = set()

    # ------------------------------------------------------------------
    # Time-series
    # ------------------------------------------------------------------
    def record_portfolio_point(self, equity: float, cash: float,
                               realized_pl: float = 0.0,
                               unrealized_pl: float = 0.0) -> None:
        ts = now_utc().isoformat()
        with self._lock:
            self._peak_equity = max(self._peak_equity, equity)
            drawdown = (
                (self._peak_equity - equity) / self._peak_equity
                if self._peak_equity > 0 else 0.0
            )
            self.equity_history.append({"ts": ts, "value": round(equity, 2)})
            self.cash_history.append({"ts": ts, "value": round(cash, 2)})
            self.daily_pl_history.append({
                "ts": ts, "value": round(realized_pl + unrealized_pl, 2),
            })
            self.drawdown_history.append({"ts": ts, "value": round(drawdown, 4)})

    def portfolio_snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {
                "equity": list(self.equity_history),
                "daily_pl": list(self.daily_pl_history),
                "drawdown": list(self.drawdown_history),
                "cash": list(self.cash_history),
            }

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def set_flag(self, name: str, value: bool) -> bool:
        """Set a control flag; refuses protected components. Returns success."""
        allowed_flags = {
            "fast_loop", "research_loop", "paper_trading", "learning",
            "generation_evolution",
        }
        protected = {"paper_trading_safety", "risk_manager", "investment_safety",
                     "capital_protection"}
        if name in protected:
            return False
        with self._lock:
            if name == "fast_loop":
                self.fast_loop_enabled = bool(value)
            elif name == "research_loop":
                self.research_loop_enabled = bool(value)
            elif name == "paper_trading":
                self.paper_trading_enabled = bool(value)
            elif name == "learning":
                self.learning_enabled = bool(value)
            elif name == "generation_evolution":
                self.generation_evolution_enabled = bool(value)
            else:
                return False
        return True

    def set_agent_enabled(self, agent_id: str, enabled: bool) -> bool:
        if agent_id not in TOGGLEABLE_AGENTS:
            return False  # safety-critical agents cannot be toggled off
        with self._lock:
            self.agent_toggles[agent_id] = bool(enabled)
        return True

    def controls_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "fast_loop": self.fast_loop_enabled,
                "research_loop": self.research_loop_enabled,
                "paper_trading": self.paper_trading_enabled,
                "learning": self.learning_enabled,
                "generation_evolution": self.generation_evolution_enabled,
                "agents": dict(self.agent_toggles),
                "protected": list(PROTECTED_COMPONENTS),
            }

    # ------------------------------------------------------------------
    # Provider settings (masked)
    # ------------------------------------------------------------------
    def set_provider_setting(self, provider: str, key: str, value: Any,
                             secret: bool = False) -> None:
        with self._lock:
            entry = self.provider_settings.setdefault(provider, {})
            if secret:
                entry[key] = _mask(str(value))
                entry[f"{key}__set"] = True
            else:
                entry[key] = value

    def provider_settings_masked(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = json.loads(json.dumps(self.provider_settings))  # deep copy
        return snapshot

    # ------------------------------------------------------------------
    # SSE broadcasting
    # ------------------------------------------------------------------
    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(maxsize=100)
        with self._lock:
            self._sse_queues.add(q)
        return q

    def unsubscribe(self, q: "queue.Queue") -> None:
        with self._lock:
            self._sse_queues.discard(q)

    def broadcast(self, event: str, data: Dict[str, Any]) -> None:
        payload = {"event": event, "data": data, "ts": now_utc().isoformat()}
        with self._lock:
            targets = list(self._sse_queues)
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass  # slow client: drop rather than block the runtime


def _mask(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 4:
        return "****"
    return secret[:2] + "*" * (len(secret) - 4) + secret[-2:]
