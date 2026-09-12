from dataclasses import dataclass


HORIZON_BAR_LIMITS = {
    "short": 30,
    "medium": 60,
    "long": 220,
}


@dataclass(frozen=True)
class MarketAnalysisConfig:
    """Configurable analysis windows and detection thresholds.

    Historical volatility is the sample standard deviation of log returns,
    annualized with sqrt(252). This is NOT implied volatility.
    """

    default_horizon: str = "medium"
    default_timeframe: str = "1D"
    volatility_short_window: int = 5
    volatility_medium_window: int = 20
    rsi_period: int = 14
    roc_period: int = 14
    volume_average_window: int = 20
    unusual_volume_ratio: float = 2.0
    anomaly_volume_ratio: float = 2.5
    price_anomaly_threshold: float = 0.05
    gap_threshold: float = 0.025
    spread_anomaly_pct: float = 0.015
    high_volatility_threshold: float = 0.40
    low_volatility_threshold: float = 0.12
    sideways_sma_band: float = 0.015
    sideways_return_band: float = 0.02
    stale_quote_seconds: int = 900
    cache_ttl_seconds: int = 60
    trading_days_per_year: int = 252

    def bar_limit_for_horizon(self, horizon: str) -> int:
        return HORIZON_BAR_LIMITS.get(horizon, HORIZON_BAR_LIMITS["medium"])
