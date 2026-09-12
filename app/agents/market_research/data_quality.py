import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from app.core.models.market import Bar, Quote, DataQualityIssue, DataQualityReport
from app.agents.market_research.config import MarketAnalysisConfig


class DataQualityChecker:
    """Validates raw market data integrity before numerical indicator calculations."""

    def __init__(self, config: Optional[MarketAnalysisConfig] = None):
        self.config = config or MarketAnalysisConfig()

    @staticmethod
    def _is_invalid_price(value: float) -> bool:
        return (not isinstance(value, (int, float))) or math.isnan(value) or math.isinf(value)

    def validate_bars(self, symbol: str, bars: List[Bar]) -> DataQualityReport:
        issues: List[str] = []
        issue_types: List[DataQualityIssue] = []
        warnings: List[str] = []

        if not bars:
            return DataQualityReport(
                symbol=symbol,
                is_valid=False,
                total_bars=0,
                issues=["Bar series is completely empty."],
                issue_types=[DataQualityIssue.MISSING_BARS]
            )

        seen_timestamps = set()
        prev_timestamp = None

        for idx, bar in enumerate(bars):
            prices = (bar.open, bar.high, bar.low, bar.close)
            if any(self._is_invalid_price(p) for p in prices):
                issues.append(f"Bar {idx} has invalid (NaN/Inf) price values.")
                if DataQualityIssue.INVALID_PRICE not in issue_types:
                    issue_types.append(DataQualityIssue.INVALID_PRICE)
            elif any(p <= 0 for p in prices):
                issues.append(
                    f"Bar {idx} has non-positive price values "
                    f"(O:{bar.open}, H:{bar.high}, L:{bar.low}, C:{bar.close})"
                )
                if DataQualityIssue.NEGATIVE_PRICE not in issue_types:
                    issue_types.append(DataQualityIssue.NEGATIVE_PRICE)
            elif bar.high < bar.low or bar.open < bar.low or bar.open > bar.high or bar.close < bar.low or bar.close > bar.high:
                issues.append(
                    f"Bar {idx} violates OHLC boundary constraints "
                    f"(O:{bar.open}, H:{bar.high}, L:{bar.low}, C:{bar.close})"
                )
                if DataQualityIssue.IMPOSSIBLE_OHLC not in issue_types:
                    issue_types.append(DataQualityIssue.IMPOSSIBLE_OHLC)

            if bar.volume < 0:
                issues.append(f"Bar {idx} has negative volume: {bar.volume}")
                if DataQualityIssue.MISSING_VOLUME not in issue_types:
                    issue_types.append(DataQualityIssue.MISSING_VOLUME)
            elif bar.volume == 0:
                warnings.append(f"Bar {idx} has zero volume.")
                if DataQualityIssue.MISSING_VOLUME not in issue_types:
                    issue_types.append(DataQualityIssue.MISSING_VOLUME)

            if bar.timestamp in seen_timestamps:
                issues.append(f"Duplicate timestamp detected: {bar.timestamp}")
                if DataQualityIssue.DUPLICATE_TIMESTAMPS not in issue_types:
                    issue_types.append(DataQualityIssue.DUPLICATE_TIMESTAMPS)
            seen_timestamps.add(bar.timestamp)

            if prev_timestamp and bar.timestamp < prev_timestamp:
                issues.append(
                    f"Non-chronological timestamp at bar {idx}: {bar.timestamp} < {prev_timestamp}"
                )
                if DataQualityIssue.INCONSISTENT_TIMESTAMPS not in issue_types:
                    issue_types.append(DataQualityIssue.INCONSISTENT_TIMESTAMPS)
            prev_timestamp = bar.timestamp

        # Missing bars: unusually large gaps in a daily-like series (weekends excluded loosely).
        if len(bars) >= 2:
            spans = sorted(bars, key=lambda b: b.timestamp)
            for i in range(1, len(spans)):
                gap = spans[i].timestamp - spans[i - 1].timestamp
                if gap > timedelta(days=5):
                    warnings.append(
                        f"Possible missing bars between {spans[i - 1].timestamp} and {spans[i].timestamp}."
                    )
                    if DataQualityIssue.MISSING_BARS not in issue_types:
                        issue_types.append(DataQualityIssue.MISSING_BARS)
                    break

        # Zero volume and calendar gaps are warnings. Negative/invalid prices and
        # duplicate/out-of-order timestamps must not enter calculations.
        has_negative_volume = any("negative volume" in i for i in issues)
        blocking_types = [
            t for t in issue_types
            if t in (
                DataQualityIssue.INVALID_PRICE,
                DataQualityIssue.NEGATIVE_PRICE,
                DataQualityIssue.IMPOSSIBLE_OHLC,
                DataQualityIssue.DUPLICATE_TIMESTAMPS,
                DataQualityIssue.INCONSISTENT_TIMESTAMPS,
            )
            or (t == DataQualityIssue.MISSING_VOLUME and has_negative_volume)
        ]
        is_valid = len(blocking_types) == 0

        return DataQualityReport(
            symbol=symbol,
            is_valid=is_valid,
            total_bars=len(bars),
            issues=issues,
            issue_types=issue_types,
            warnings=warnings
        )

    def validate_quote(
        self,
        symbol: str,
        quote: Optional[Quote],
        now: datetime,
        market_is_open: Optional[bool] = None
    ) -> Tuple[List[str], List[DataQualityIssue]]:
        issues: List[str] = []
        types: List[DataQualityIssue] = []
        if quote is None:
            return issues, types

        if quote.bid_price <= 0 or quote.ask_price <= 0 or quote.ask_price < quote.bid_price:
            issues.append(
                f"Invalid quote prices for {symbol}: bid={quote.bid_price} ask={quote.ask_price}"
            )
            types.append(DataQualityIssue.INVALID_PRICE)

        age = now - quote.timestamp
        if market_is_open and age.total_seconds() > self.config.stale_quote_seconds:
            issues.append(
                f"Stale quote for {symbol}: age={age.total_seconds():.0f}s exceeds "
                f"{self.config.stale_quote_seconds}s."
            )
            types.append(DataQualityIssue.STALE_QUOTE)
        return issues, types

    def filter_valid_bars(self, bars: List[Bar]) -> List[Bar]:
        """Drop bars that would corrupt calculations. Duplicates keep the first occurrence."""
        cleaned: List[Bar] = []
        seen = set()
        for bar in sorted(bars, key=lambda b: b.timestamp):
            if bar.timestamp in seen:
                continue
            prices = (bar.open, bar.high, bar.low, bar.close)
            if any(self._is_invalid_price(p) or p <= 0 for p in prices):
                continue
            if bar.high < bar.low or bar.open < bar.low or bar.open > bar.high or bar.close < bar.low or bar.close > bar.high:
                continue
            if bar.volume < 0:
                continue
            seen.add(bar.timestamp)
            cleaned.append(bar)
        return cleaned
