import math
from typing import List, Optional
from app.core.models.market import (
    Bar, PriceFeatures, ReturnFeatures, MovingAverages,
    VolatilityMetrics, VolumeMetrics, MomentumMetrics
)
from app.agents.market_research.config import MarketAnalysisConfig


class MarketFeatureCalculator:
    """Deterministic market features. No LLM arithmetic.

    Volatility: sample stdev of log returns, annualized with sqrt(trading_days_per_year).
    This is historical volatility only — never implied volatility.
    """

    def __init__(self, config: Optional[MarketAnalysisConfig] = None):
        self.config = config or MarketAnalysisConfig()

    def calculate_price_features(self, bars: List[Bar]) -> PriceFeatures:
        if not bars:
            return PriceFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        latest = bars[-1]
        current_price = latest.close
        open_price = latest.open
        price_change = current_price - open_price
        pct_change = (price_change / open_price * 100.0) if open_price > 0 else 0.0

        daily_high = latest.high
        daily_low = latest.low
        return PriceFeatures(
            current_price=current_price,
            price_change=price_change,
            percentage_change=pct_change,
            daily_high=daily_high,
            daily_low=daily_low,
            distance_from_high=daily_high - current_price,
            distance_from_low=current_price - daily_low
        )

    def calculate_returns(self, bars: List[Bar]) -> ReturnFeatures:
        n = len(bars)
        if n < 2:
            return ReturnFeatures()

        curr_close = bars[-1].close

        def period_return(offset: int) -> Optional[float]:
            # offset bars ago: 1-day needs 2 closes (index -2), 5-day needs 6 closes.
            idx = -(offset + 1)
            if n < offset + 1:
                return None
            base = bars[idx].close
            if base <= 0:
                return None
            return (curr_close - base) / base

        return ReturnFeatures(
            return_1d=period_return(1),
            return_5d=period_return(5),
            return_20d=period_return(20)
        )

    def calculate_moving_averages(self, bars: List[Bar]) -> MovingAverages:
        closes = [b.close for b in bars]
        n = len(closes)

        sma_20 = (sum(closes[-20:]) / 20.0) if n >= 20 else None
        sma_50 = (sum(closes[-50:]) / 50.0) if n >= 50 else None
        sma_200 = (sum(closes[-200:]) / 200.0) if n >= 200 else None

        status = "VALID" if sma_20 is not None else "INSUFFICIENT_DATA"
        return MovingAverages(
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            status=status
        )

    def calculate_volatility(self, bars: List[Bar]) -> VolatilityMetrics:
        """Historical volatility: stdev(log returns, sample, ddof=1) * sqrt(252).

        Short-term uses `volatility_short_window` log returns (needs window+1 bars).
        Medium-term uses `volatility_medium_window` log returns.
        Insufficient observations yield None — values are never fabricated.
        """
        closes = [b.close for b in bars]
        n = len(closes)
        if n < 2:
            return VolatilityMetrics()

        log_returns = []
        for i in range(1, n):
            if closes[i - 1] > 0 and closes[i] > 0:
                log_returns.append(math.log(closes[i] / closes[i - 1]))

        def _calc_vol(returns_subset: List[float]) -> Optional[float]:
            k = len(returns_subset)
            if k < 2:
                return None
            mean_ret = sum(returns_subset) / k
            variance = sum((r - mean_ret) ** 2 for r in returns_subset) / (k - 1)
            daily_std = math.sqrt(variance)
            return daily_std * math.sqrt(self.config.trading_days_per_year)

        st_n = self.config.volatility_short_window
        mt_n = self.config.volatility_medium_window
        st_vol = _calc_vol(log_returns[-st_n:]) if len(log_returns) >= st_n else None
        mt_vol = _calc_vol(log_returns[-mt_n:]) if len(log_returns) >= mt_n else None

        return VolatilityMetrics(
            short_term_volatility=st_vol,
            medium_term_volatility=mt_vol,
            calculation_method="ANNUALIZED_STD_OF_LOG_RETURNS"
        )

    def calculate_volume_metrics(self, bars: List[Bar]) -> VolumeMetrics:
        if not bars:
            return VolumeMetrics(0, 0.0, 1.0, False)

        curr_vol = bars[-1].volume
        prior = [b.volume for b in bars[:-1]]
        window = min(len(prior), self.config.volume_average_window) if prior else 0
        if window > 0:
            avg_vol = sum(prior[-window:]) / window
        else:
            avg_vol = float(curr_vol) if curr_vol > 0 else 0.0

        if avg_vol > 0:
            ratio = curr_vol / avg_vol
        elif curr_vol == 0:
            ratio = 1.0
        else:
            ratio = 1.0

        is_unusual = avg_vol > 0 and (
            ratio >= self.config.unusual_volume_ratio or ratio <= 0.2
        )
        return VolumeMetrics(
            current_volume=curr_vol,
            average_volume=avg_vol,
            volume_ratio=ratio,
            is_unusual_volume=is_unusual
        )

    def calculate_rsi(self, closes: List[float], period: Optional[int] = None) -> Optional[float]:
        """Wilder's RSI. Requires period+1 closes. Never estimated by an LLM."""
        period = period or self.config.rsi_period
        n = len(closes)
        if n < period + 1:
            return None

        changes = [closes[i] - closes[i - 1] for i in range(1, n)]
        gains = [max(c, 0.0) for c in changes]
        losses = [max(-c, 0.0) for c in changes]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(changes)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def calculate_momentum(self, bars: List[Bar], mas: MovingAverages) -> MomentumMetrics:
        closes = [b.close for b in bars]
        n = len(closes)
        rsi_14 = self.calculate_rsi(closes)

        roc: Optional[float] = None
        period = self.config.roc_period
        if n >= period + 1 and closes[-(period + 1)] > 0:
            roc = ((closes[-1] - closes[-(period + 1)]) / closes[-(period + 1)]) * 100.0

        ma_alignment = "INSUFFICIENT_DATA"
        if mas.sma_20 is not None and mas.sma_50 is not None and n > 0:
            if closes[-1] > mas.sma_20 > mas.sma_50:
                ma_alignment = "BULLISH_ALIGNED"
            elif closes[-1] < mas.sma_20 < mas.sma_50:
                ma_alignment = "BEARISH_ALIGNED"
            else:
                ma_alignment = "MIXED"

        return MomentumMetrics(
            rsi_14=rsi_14,
            rate_of_change=roc,
            ma_alignment=ma_alignment
        )
