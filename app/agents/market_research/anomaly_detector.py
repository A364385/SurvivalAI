from typing import List, Optional
from datetime import datetime
from app.core.models.market import (
    Bar, Quote, MarketAnomaly, AnomalyType, PriceFeatures, ReturnFeatures,
    VolumeMetrics, VolatilityMetrics
)
from app.agents.market_research.config import MarketAnalysisConfig
from app.utils.ids import generate_id


class MarketAnomalyDetector:
    """Objective anomaly flags. An anomaly is not a bullish or bearish recommendation."""

    def __init__(self, config: Optional[MarketAnalysisConfig] = None):
        self.config = config or MarketAnalysisConfig()

    def detect_anomalies(
        self,
        symbol: str,
        bars: List[Bar],
        quote: Optional[Quote],
        price_feat: PriceFeatures,
        returns: ReturnFeatures,
        vol: VolatilityMetrics,
        vol_metrics: VolumeMetrics,
        now: datetime
    ) -> List[MarketAnomaly]:
        anomalies: List[MarketAnomaly] = []
        if not bars:
            return anomalies

        threshold = self.config.price_anomaly_threshold
        if returns.return_1d is not None:
            r1d = returns.return_1d
            if r1d >= threshold:
                anomalies.append(MarketAnomaly(
                    anomaly_id=generate_id("anom_pspike"),
                    anomaly_type=AnomalyType.PRICE_SPIKE,
                    symbol=symbol,
                    detected_at=now,
                    severity=min(1.0, r1d / (threshold * 2.0)),
                    supporting_measurements={
                        "1d_return": r1d,
                        "percentage_change": price_feat.percentage_change,
                    },
                    confidence=0.95,
                    description=f"Unusually large upward close-to-close move of {r1d * 100:.2f}%.",
                ))
            elif r1d <= -threshold:
                anomalies.append(MarketAnomaly(
                    anomaly_id=generate_id("anom_pdrop"),
                    anomaly_type=AnomalyType.PRICE_DROP,
                    symbol=symbol,
                    detected_at=now,
                    severity=min(1.0, abs(r1d) / (threshold * 2.0)),
                    supporting_measurements={
                        "1d_return": r1d,
                        "percentage_change": price_feat.percentage_change,
                    },
                    confidence=0.95,
                    description=f"Unusually large downward close-to-close move of {r1d * 100:.2f}%.",
                ))

        if vol_metrics.volume_ratio >= self.config.anomaly_volume_ratio:
            anomalies.append(MarketAnomaly(
                anomaly_id=generate_id("anom_vspike"),
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                symbol=symbol,
                detected_at=now,
                severity=min(1.0, vol_metrics.volume_ratio / 5.0),
                supporting_measurements={
                    "volume_ratio": vol_metrics.volume_ratio,
                    "current_volume": vol_metrics.current_volume,
                    "average_volume": vol_metrics.average_volume,
                },
                confidence=0.90,
                description=f"Unusual volume relative to recent average: {vol_metrics.volume_ratio:.2f}x.",
            ))

        if (
            vol.short_term_volatility
            and vol.medium_term_volatility
            and vol.medium_term_volatility >= 0.15
        ):
            ratio = vol.short_term_volatility / vol.medium_term_volatility
            if ratio >= 2.0:
                anomalies.append(MarketAnomaly(
                    anomaly_id=generate_id("anom_volspike"),
                    anomaly_type=AnomalyType.VOLATILITY_SPIKE,
                    symbol=symbol,
                    detected_at=now,
                    severity=min(1.0, ratio / 3.0),
                    supporting_measurements={
                        "short_term_vol": vol.short_term_volatility,
                        "medium_term_vol": vol.medium_term_volatility,
                        "vol_ratio": ratio,
                    },
                    confidence=0.85,
                    description=(
                        "Short-term historical volatility is elevated versus the medium-term baseline."
                    ),
                ))

        if len(bars) >= 2:
            prev_close = bars[-2].close
            curr_open = bars[-1].open
            if prev_close > 0:
                gap_ratio = (curr_open - prev_close) / prev_close
                if gap_ratio >= self.config.gap_threshold:
                    anomalies.append(MarketAnomaly(
                        anomaly_id=generate_id("anom_gapup"),
                        anomaly_type=AnomalyType.GAP_UP,
                        symbol=symbol,
                        detected_at=now,
                        severity=min(1.0, gap_ratio / 0.05),
                        supporting_measurements={
                            "gap_percentage": gap_ratio * 100.0,
                            "previous_close": prev_close,
                            "open": curr_open,
                        },
                        confidence=0.90,
                        description=f"Open is {gap_ratio * 100:.2f}% above prior close.",
                    ))
                elif gap_ratio <= -self.config.gap_threshold:
                    anomalies.append(MarketAnomaly(
                        anomaly_id=generate_id("anom_gapdown"),
                        anomaly_type=AnomalyType.GAP_DOWN,
                        symbol=symbol,
                        detected_at=now,
                        severity=min(1.0, abs(gap_ratio) / 0.05),
                        supporting_measurements={
                            "gap_percentage": gap_ratio * 100.0,
                            "previous_close": prev_close,
                            "open": curr_open,
                        },
                        confidence=0.90,
                        description=f"Open is {gap_ratio * 100:.2f}% below prior close.",
                    ))

        if quote and quote.bid_price > 0:
            spread = quote.ask_price - quote.bid_price
            spread_pct = spread / quote.bid_price
            if spread_pct >= self.config.spread_anomaly_pct:
                anomalies.append(MarketAnomaly(
                    anomaly_id=generate_id("anom_spread"),
                    anomaly_type=AnomalyType.ABNORMAL_SPREAD,
                    symbol=symbol,
                    detected_at=now,
                    severity=min(1.0, spread_pct / 0.03),
                    supporting_measurements={
                        "bid": quote.bid_price,
                        "ask": quote.ask_price,
                        "spread_pct": spread_pct * 100.0,
                    },
                    confidence=0.95,
                    description=f"Bid-ask spread is {spread_pct * 100:.2f}% of bid.",
                ))

        return anomalies
