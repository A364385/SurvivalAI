from typing import List, Optional
from app.core.models.market import (
    Bar, MarketTrend, MarketRegime, MovingAverages, VolatilityMetrics, ReturnFeatures
)
from app.agents.market_research.config import MarketAnalysisConfig


class MarketRegimeClassifier:
    """Deterministic trend and regime rules. Not a forecast of future prices.

    Trend rules (evaluated in order):
    1. Fewer than 20 bars or missing SMA20 -> INSUFFICIENT_DATA
    2. SMA20 and SMA50 available:
       - close > SMA20 > SMA50 and 5-day return > 0 -> UPTREND
       - close < SMA20 < SMA50 and 5-day return < 0 -> DOWNTREND
       - MA alignment disagrees with 5-day return sign -> MIXED
    3. |close - SMA20| / SMA20 <= 1.5% and |5-day return| <= 2% -> SIDEWAYS
    4. close > SMA20 -> UPTREND; close < SMA20 -> DOWNTREND; else MIXED

    Regime rules (evaluated in order):
    1. Fewer than 20 bars -> UNKNOWN
    2. Short-term historical vol >= 40% annualized -> HIGH_VOLATILITY
    3. Trend UPTREND / DOWNTREND -> TRENDING_UP / TRENDING_DOWN
    4. Short-term historical vol <= 12% annualized -> LOW_VOLATILITY
    5. Trend SIDEWAYS -> RANGE_BOUND
    6. Otherwise TRANSITION
    """

    def __init__(self, config: Optional[MarketAnalysisConfig] = None):
        self.config = config or MarketAnalysisConfig()

    def classify_trend(self, bars: List[Bar], mas: MovingAverages, returns: ReturnFeatures) -> MarketTrend:
        if len(bars) < 20 or mas.sma_20 is None:
            return MarketTrend.INSUFFICIENT_DATA

        curr = bars[-1].close
        sma_20 = mas.sma_20
        sma_50 = mas.sma_50
        r_5d = returns.return_5d

        if sma_50 is not None:
            bullish_stack = curr > sma_20 > sma_50
            bearish_stack = curr < sma_20 < sma_50
            if bullish_stack and (r_5d is not None and r_5d > 0.0):
                return MarketTrend.UPTREND
            if bearish_stack and (r_5d is not None and r_5d < 0.0):
                return MarketTrend.DOWNTREND
            if (bullish_stack and r_5d is not None and r_5d < 0.0) or (
                bearish_stack and r_5d is not None and r_5d > 0.0
            ):
                return MarketTrend.MIXED

        if sma_20 > 0:
            pct_diff_sma20 = abs(curr - sma_20) / sma_20
            if pct_diff_sma20 <= self.config.sideways_sma_band and (
                r_5d is None or abs(r_5d) <= self.config.sideways_return_band
            ):
                return MarketTrend.SIDEWAYS

        if curr > sma_20:
            return MarketTrend.UPTREND
        if curr < sma_20:
            return MarketTrend.DOWNTREND
        return MarketTrend.MIXED

    def classify_regime(
        self,
        bars: List[Bar],
        trend: MarketTrend,
        volatility: VolatilityMetrics
    ) -> MarketRegime:
        if len(bars) < 20 or trend == MarketTrend.INSUFFICIENT_DATA:
            return MarketRegime.UNKNOWN

        st_vol = volatility.short_term_volatility
        if st_vol is not None and st_vol >= self.config.high_volatility_threshold:
            return MarketRegime.HIGH_VOLATILITY

        if trend == MarketTrend.UPTREND:
            return MarketRegime.TRENDING_UP
        if trend == MarketTrend.DOWNTREND:
            return MarketRegime.TRENDING_DOWN

        if st_vol is not None and st_vol <= self.config.low_volatility_threshold:
            return MarketRegime.LOW_VOLATILITY

        if trend == MarketTrend.SIDEWAYS:
            return MarketRegime.RANGE_BOUND

        if trend == MarketTrend.MIXED:
            return MarketRegime.TRANSITION

        return MarketRegime.TRANSITION
