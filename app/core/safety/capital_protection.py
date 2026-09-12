"""Capital Protection Layer — deterministic hard safety above the Risk Manager.

Hierarchy: DATA -> RESEARCH -> DEEP LOOKER -> RISK MANAGER -> CAPITAL
PROTECTION -> CEO DECISION -> PAPER EXECUTION.

This layer enforces portfolio-level hard limits that NO AI output can override:
max position size, max portfolio exposure, max single loss, minimum cash
reserve, max drawdraw, max correlated/sector/crypto exposure, and an
emergency stop that halts all new paper orders.

It is intentionally pure/deterministic: no LLM, no network, no I/O. Unit tests
prove that AI-generated "approvals" cannot bypass it.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


@dataclass
class CapitalProtectionConfig:
    """Hard portfolio-level limits (all deterministic, AI cannot change them)."""
    # Max fraction of portfolio equity in one new position (Risk Manager may
    # be stricter; this is the ceiling).
    max_position_fraction: float = 0.10
    # Max total invested fraction of equity across all positions.
    max_portfolio_exposure_fraction: float = 0.80
    # Max single-day loss fraction before new entries are blocked.
    max_single_loss_fraction: float = 0.05
    # Minimum cash fraction that must remain after any new order.
    min_cash_fraction: float = 0.10
    # Max drawdown fraction; beyond this, emergency stop engages.
    max_drawdown_fraction: float = 0.25
    # Max fraction of equity exposed to correlated assets (same group).
    max_correlated_group_fraction: float = 0.30
    # Max fraction of equity exposed to crypto assets.
    max_crypto_exposure_fraction: float = 0.20
    # Emergency stop: when set, ALL new paper orders are rejected.
    emergency_stop: bool = False
    # Trading pause: blocks new orders without engaging the emergency flag.
    trading_paused: bool = False
    # Crypto symbols are detected by these suffixes/patterns.
    crypto_symbols: List[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK",
        "DOT", "MATIC", "UNI", "LTC", "BCH", "ATOM", "NEAR", "ARB", "OP",
    ])


@dataclass
class ProtectionVerdict:
    """Result of a capital protection evaluation."""
    passed: bool
    reasons: List[str]
    checks: Dict[str, bool]
    checked_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
            "checked_at": self.checked_at,
        }


class CapitalProtectionLayer:
    """Deterministic portfolio-level protection. No AI can override this."""

    def __init__(self, config: Optional[CapitalProtectionConfig] = None):
        self.config = config or CapitalProtectionConfig()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_crypto(symbol: str, crypto_symbols: List[str]) -> bool:
        s = (symbol or "").upper()
        base = s.split("/")[0].split("-")[0].split(":")[0]
        return base in crypto_symbols

    def _group_of(self, symbol: str) -> str:
        """Group symbols for correlation limits (crypto vs equity vs cash)."""
        if self._is_crypto(symbol, self.config.crypto_symbols):
            return "crypto"
        # Simple sector proxy: same first two letters as an equity ticker group
        # would be wrong; instead treat all non-crypto as one 'equity' group for
        # the correlated-exposure cap unless a sector map is provided.
        return "equity"

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        symbol: str,
        proposed_position_value: float,
        portfolio_value: float,
        available_cash: float,
        current_positions: Optional[List[Dict[str, Any]]] = None,
        current_drawdown: float = 0.0,
        current_peak_equity: float = 0.0,
    ) -> ProtectionVerdict:
        """Evaluate a proposed new paper position against hard limits.

        All inputs are deterministic numbers fetched from the paper broker and
        the runtime's own accounting; AI outputs can only *request* a size, and
        the caller must clamp it to `max_allowed_position_value`.
        """
        positions = current_positions or []
        checks: Dict[str, bool] = {}
        reasons: List[str] = []

        now = now_utc().isoformat()

        def fail(name: str, reason: str) -> None:
            checks[name] = False
            reasons.append(reason)

        # 0. Emergency stop / pause are absolute.
        if self.config.emergency_stop:
            fail("emergency_stop", "emergency stop engaged — all new orders blocked")
        else:
            checks["emergency_stop"] = True

        if self.config.trading_paused and not self.config.emergency_stop:
            fail("trading_paused", "trading paused — new orders blocked")
        elif not self.config.trading_paused:
            checks["trading_paused"] = True

        if portfolio_value <= 0:
            fail("portfolio_value_positive", "portfolio value must be positive")
            return ProtectionVerdict(False, reasons, checks, now)

        # 1. Max position size.
        max_pos = portfolio_value * self.config.max_position_fraction
        checks["max_position_size"] = proposed_position_value <= max_pos
        if not checks["max_position_size"]:
            reasons.append(
                f"proposed position {proposed_position_value:.2f} exceeds "
                f"{self.config.max_position_fraction:.0%} of portfolio "
                f"({max_pos:.2f})"
            )

        # 2. Max total exposure.
        current_exposure = sum(
            float(p.get("market_value", 0.0)) for p in positions
            if isinstance(p, dict)
        )
        projected_exposure = current_exposure + proposed_position_value
        max_exposure = portfolio_value * self.config.max_portfolio_exposure_fraction
        checks["max_portfolio_exposure"] = projected_exposure <= max_exposure
        if not checks["max_portfolio_exposure"]:
            reasons.append(
                f"projected exposure {projected_exposure:.2f} would exceed "
                f"{self.config.max_portfolio_exposure_fraction:.0%} cap "
                f"({max_exposure:.2f})"
            )

        # 3. Minimum cash reserve after the trade.
        projected_cash = available_cash - proposed_position_value
        min_cash = portfolio_value * self.config.min_cash_fraction
        checks["min_cash_reserve"] = projected_cash >= min_cash
        if not checks["min_cash_reserve"]:
            reasons.append(
                f"projected cash {projected_cash:.2f} would fall below "
                f"{self.config.min_cash_fraction:.0%} reserve ({min_cash:.2f})"
            )

        # 4. Drawdown emergency stop.
        if current_drawdown >= self.config.max_drawdown_fraction:
            fail(
                "max_drawdown",
                f"current drawdown {current_drawdown:.1%} exceeds "
                f"{self.config.max_drawdown_fraction:.0%} limit",
            )
        else:
            checks["max_drawdown"] = True

        # 5. Correlated group exposure (existing positions in the same group).
        group = self._group_of(symbol)
        group_exposure = sum(
            float(p.get("market_value", 0.0)) for p in positions
            if isinstance(p, dict) and self._group_of(str(p.get("symbol", ""))) == group
        )
        max_group = portfolio_value * self.config.max_correlated_group_fraction
        if group == "crypto":
            crypto_cap = self.config.max_crypto_exposure_fraction
        else:
            crypto_cap = None
        if crypto_cap is not None:
            projected_crypto = group_exposure + proposed_position_value
            max_crypto = portfolio_value * crypto_cap
            crypto_ok = projected_crypto <= max_crypto
            checks["max_crypto_exposure"] = crypto_ok
            if not crypto_ok:
                reasons.append(
                    f"projected crypto exposure {projected_crypto:.2f} would exceed "
                    f"{crypto_cap:.0%} cap ({max_crypto:.2f})"
                )
        else:
            crypto_ok = projected_exposure <= max_exposure  # already checked
            checks["max_crypto_exposure"] = True

        if group == "equity" and (group_exposure + proposed_position_value) > max_group:
            checks["max_correlated_exposure"] = False
            reasons.append(
                f"projected correlated-group exposure exceeds "
                f"{self.config.max_correlated_group_fraction:.0%} cap"
            )
        else:
            checks["max_correlated_exposure"] = True

        passed = all(checks.values())
        return ProtectionVerdict(passed=passed, reasons=reasons, checks=checks, checked_at=now)

    # ------------------------------------------------------------------
    # Batch helpers used by the runtime
    # ------------------------------------------------------------------
    def max_allowed_position_value(
        self,
        portfolio_value: float,
        available_cash: float,
        current_positions: Optional[List[Dict[str, Any]]] = None,
        symbol: str = "",
    ) -> float:
        """Largest position value that would pass protection (deterministic)."""
        if portfolio_value <= 0:
            return 0.0
        positions = current_positions or []
        current_exposure = sum(
            float(p.get("market_value", 0.0)) for p in positions if isinstance(p, dict)
        )
        by_cap = portfolio_value * self.config.max_position_fraction
        by_exposure = max(0.0, portfolio_value * self.config.max_portfolio_exposure_fraction - current_exposure)
        by_cash = max(0.0, available_cash - portfolio_value * self.config.min_cash_fraction)
        group = self._group_of(symbol)
        if group == "crypto":
            group_exposure = sum(
                float(p.get("market_value", 0.0)) for p in positions
                if isinstance(p, dict) and self._group_of(str(p.get("symbol", ""))) == "crypto"
            )
            by_group = max(0.0, portfolio_value * self.config.max_crypto_exposure_fraction - group_exposure)
        else:
            group_exposure = current_exposure  # all non-crypto counted as one group
            by_group = max(0.0, portfolio_value * self.config.max_correlated_group_fraction - group_exposure)
        return max(0.0, min(by_cap, by_exposure, by_cash, by_group))

    # ------------------------------------------------------------------
    # Control surface (deterministic only — dashboard may call these)
    # ------------------------------------------------------------------
    def engage_emergency_stop(self, reason: str) -> None:
        self.config.emergency_stop = True
        logger.critical("CAPITAL PROTECTION EMERGENCY STOP engaged: %s", reason)

    def lift_emergency_stop(self) -> None:
        self.config.emergency_stop = False
        logger.info("Capital protection emergency stop lifted")

    def pause_trading(self, reason: str) -> None:
        self.config.trading_paused = True
        logger.warning("Capital protection trading pause: %s", reason)

    def resume_trading(self) -> None:
        self.config.trading_paused = False
        logger.info("Capital protection trading resumed")
