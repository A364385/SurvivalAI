"""Local watchdog.

Continuously verifies that the system is in a SAFE state to continue:
- runtime responsiveness (cycle heartbeat age)
- provider health (market/news/execution)
- database integrity (SQLite)
- model/LLM provider availability
- process memory usage (never blindly consume all RAM)
- stuck agents (long-running tasks)
- stale market data

When unsafe, the watchdog instructs the runtime to PAUSE (safe state:
monitoring continues, new decisions stop) rather than continuing blindly.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


@dataclass
class WatchdogConfig:
    max_heartbeat_age_seconds: float = 300.0
    max_memory_percent: float = 90.0
    max_stale_data_seconds: float = 900.0
    check_interval_seconds: float = 30.0
    consecutive_failures_before_pause: int = 2


@dataclass
class WatchdogCheck:
    name: str
    ok: bool
    detail: str = ""


class Watchdog:
    """Runs periodic safety checks; exposes `is_safe()` and a callback hook."""

    def __init__(
        self,
        config: Optional[WatchdogConfig] = None,
        checkers: Optional[List[Callable[[], WatchdogCheck]]] = None,
    ):
        self.config = config or WatchdogConfig()
        self._checkers: List[Callable[[], WatchdogCheck]] = list(checkers or [])
        self._failures = 0
        self._last_result: Dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.on_unsafe: Optional[Callable[[List[WatchdogCheck]], None]] = None
        self._paused_by_watchdog = False

    # ------------------------------------------------------------------
    def register_checker(self, checker: Callable[[], WatchdogCheck]) -> None:
        self._checkers.append(checker)

    def heartbeat_check(self, last_heartbeat_ts: Optional[float]) -> WatchdogCheck:
        if last_heartbeat_ts is None:
            return WatchdogCheck("heartbeat", True, "no heartbeat yet (startup)")
        age = time.time() - last_heartbeat_ts
        ok = age <= self.config.max_heartbeat_age_seconds
        return WatchdogCheck("heartbeat", ok, f"age={age:.0f}s")

    def memory_check(self) -> WatchdogCheck:
        try:
            import psutil
            proc = psutil.Process()
            percent = proc.memory_percent()
            ok = percent < self.config.max_memory_percent
            return WatchdogCheck("memory", ok, f"process uses {percent:.1f}% of RAM")
        except Exception:
            return WatchdogCheck("memory", True, "psutil unavailable — skipped")

    def stale_data_check(self, market_data_ts: Optional[float]) -> WatchdogCheck:
        if market_data_ts is None:
            return WatchdogCheck("stale_data", True, "no market data yet")
        age = time.time() - market_data_ts
        ok = age <= self.config.max_stale_data_seconds
        return WatchdogCheck("stale_data", ok, f"data age={age:.0f}s")

    def database_check(self, integrity_fn: Optional[Callable[[], bool]] = None) -> WatchdogCheck:
        if integrity_fn is None:
            return WatchdogCheck("database", True, "no integrity check configured")
        try:
            ok = bool(integrity_fn())
            return WatchdogCheck("database", ok, "integrity_check")
        except Exception as e:
            return WatchdogCheck("database", False, str(e))

    def provider_check(self, provider_health: Optional[Dict[str, Any]]) -> WatchdogCheck:
        if not provider_health:
            return WatchdogCheck("providers", True, "no health data yet")
        critical = provider_health.get("critical_healthy", True)
        return WatchdogCheck("providers", bool(critical),
                             f"critical_healthy={critical}")

    # ------------------------------------------------------------------
    def run_checks(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        checks: List[WatchdogCheck] = []
        extra = extra or {}
        try:
            checks.append(self.memory_check())
            if "last_heartbeat" in extra:
                checks.append(self.heartbeat_check(extra["last_heartbeat"]))
            if "market_data_ts" in extra:
                checks.append(self.stale_data_check(extra["market_data_ts"]))
            if "integrity_fn" in extra:
                checks.append(self.database_check(extra["integrity_fn"]))
            if "provider_health" in extra:
                checks.append(self.provider_check(extra["provider_health"]))
            for checker in self._checkers:
                try:
                    checks.append(checker())
                except Exception as e:
                    checks.append(WatchdogCheck(f"custom:{getattr(checker, '__name__', '?')}", False, str(e)))
        except Exception as e:
            logger.exception("Watchdog check execution failed")
            checks.append(WatchdogCheck("watchdog_itself", False, str(e)))

        failed = [c for c in checks if not c.ok]
        is_safe = len(failed) == 0
        with self._lock:
            self._failures = 0 if is_safe else self._failures + 1
            threshold = self.config.consecutive_failures_before_pause
            should_pause = self._failures >= threshold
            self._last_result = {
                "safe": is_safe,
                "should_pause": should_pause,
                "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
                "consecutive_failures": self._failures,
                "timestamp": now_utc().isoformat(),
            }
        if not is_safe:
            failed_names = ", ".join(c.name for c in failed)
            logger.warning("Watchdog unsafe: %s", failed_names)
            if should_pause and self.on_unsafe:
                try:
                    self.on_unsafe(failed)
                    self._paused_by_watchdog = True
                except Exception:
                    logger.exception("on_unsafe callback failed")
        else:
            self._paused_by_watchdog = False
        return self._last_result

    @property
    def last_result(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._last_result)

    @property
    def paused_by_watchdog(self) -> bool:
        return self._paused_by_watchdog

    # ------------------------------------------------------------------
    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        def loop():
            while not self._stop.is_set():
                try:
                    self.run_checks()
                except Exception:
                    logger.exception("Watchdog loop error")
                self._stop.wait(self.config.check_interval_seconds)

        self._thread = threading.Thread(target=loop, daemon=True, name="survivalai-watchdog")
        self._thread.start()

    def stop_background(self) -> None:
        self._stop.set()
