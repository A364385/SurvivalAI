from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class Quote:
    """Best bid/ask quote for a symbol."""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    timestamp: datetime


@dataclass(frozen=True)
class Trade:
    """Executed trade print for a symbol."""
    symbol: str
    price: float
    size: int
    timestamp: datetime


@dataclass(frozen=True)
class Bar:
    """Aggregated OHLCV price bar."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class MarketClock:
    """Current market operational status and next open/close transitions."""
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


class MarketTrend(str, Enum):
    """Directional trend classification."""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"
    MIXED = "MIXED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class MarketRegime(str, Enum):
    """Broad market structural state."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class AnomalyType(str, Enum):
    """Identified market anomalies."""
    PRICE_SPIKE = "PRICE_SPIKE"
    PRICE_DROP = "PRICE_DROP"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    GAP_UP = "GAP_UP"
    GAP_DOWN = "GAP_DOWN"
    ABNORMAL_SPREAD = "ABNORMAL_SPREAD"


@dataclass
class MarketAnomaly:
    """Unusual market behavior detection."""
    anomaly_id: str
    anomaly_type: AnomalyType
    symbol: str
    detected_at: datetime
    severity: float  # 0.0 to 1.0
    supporting_measurements: Dict[str, Any]
    confidence: float
    description: str = ""


class DataQualityIssue(str, Enum):
    """Specific data quality defects."""
    MISSING_BARS = "MISSING_BARS"
    DUPLICATE_TIMESTAMPS = "DUPLICATE_TIMESTAMPS"
    INVALID_PRICE = "INVALID_PRICE"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    IMPOSSIBLE_OHLC = "IMPOSSIBLE_OHLC"
    MISSING_VOLUME = "MISSING_VOLUME"
    STALE_QUOTE = "STALE_QUOTE"
    INCONSISTENT_TIMESTAMPS = "INCONSISTENT_TIMESTAMPS"


@dataclass
class DataQualityReport:
    """Audit report of raw market data cleanliness."""
    symbol: str
    is_valid: bool
    total_bars: int
    issues: List[str] = field(default_factory=list)
    issue_types: List[DataQualityIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PriceFeatures:
    """Core price and distance metrics."""
    current_price: float
    price_change: float
    percentage_change: float
    daily_high: float
    daily_low: float
    distance_from_high: float
    distance_from_low: float


@dataclass
class ReturnFeatures:
    """Period returns."""
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_20d: Optional[float] = None


@dataclass
class MovingAverages:
    """Simple moving averages."""
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    status: str = "VALID"  # VALID or INSUFFICIENT_DATA


@dataclass
class VolatilityMetrics:
    """Historical annualized volatility measurements."""
    short_term_volatility: Optional[float] = None   # 5-period
    medium_term_volatility: Optional[float] = None  # 20-period
    calculation_method: str = "ANNUALIZED_STD_OF_LOG_RETURNS"


@dataclass
class VolumeMetrics:
    """Volume and liquidity measurements."""
    current_volume: int
    average_volume: float
    volume_ratio: float
    is_unusual_volume: bool


@dataclass
class MomentumMetrics:
    """Technical momentum indicators."""
    rsi_14: Optional[float] = None
    rate_of_change: Optional[float] = None
    ma_alignment: Optional[str] = None


@dataclass
class AssetCapabilities:
    """Explicit capability flags so the agent never assumes earnings, dividends, or statements exist."""
    asset_type: str = "UNKNOWN"  # EQUITY, ETF, CRYPTO, OTHER, UNKNOWN
    has_earnings: bool = False
    has_dividends: bool = False
    has_market_cap: bool = False
    has_financial_statements: bool = False
    has_options_data: bool = False


def capabilities_from_asset_type(asset_type: str) -> AssetCapabilities:
    """Map a declared asset type to conservative capability flags. Never fabricates fundamentals."""
    normalized = (asset_type or "UNKNOWN").upper()
    if normalized in ("EQUITY", "STOCK"):
        return AssetCapabilities(
            asset_type="EQUITY",
            has_earnings=True,
            has_dividends=True,
            has_market_cap=True,
            has_financial_statements=True,
            has_options_data=False,
        )
    if normalized == "ETF":
        return AssetCapabilities(
            asset_type="ETF",
            has_earnings=False,
            has_dividends=True,
            has_market_cap=True,
            has_financial_statements=False,
            has_options_data=False,
        )
    if normalized in ("CRYPTO", "CRYPTOCURRENCY"):
        return AssetCapabilities(
            asset_type="CRYPTO",
            has_earnings=False,
            has_dividends=False,
            has_market_cap=False,
            has_financial_statements=False,
            has_options_data=False,
        )
    if normalized in ("COMMODITY", "COMMODITIES"):
        return AssetCapabilities(
            asset_type="COMMODITY",
            has_earnings=False,
            has_dividends=False,
            has_market_cap=False,
            has_financial_statements=False,
            has_options_data=False,
        )
    return AssetCapabilities(asset_type="UNKNOWN")


@dataclass
class MarketSnapshot:
    """Unified, normalized market state container for downstream agents."""
    symbol: str
    timestamp: datetime
    current_price: float
    price_features: PriceFeatures
    returns: ReturnFeatures
    moving_averages: MovingAverages
    volatility: VolatilityMetrics
    volume: VolumeMetrics
    momentum: MomentumMetrics
    trend: MarketTrend
    regime: MarketRegime
    anomalies: List[MarketAnomaly] = field(default_factory=list)
    data_quality: Optional[DataQualityReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieved_at: Optional[datetime] = None
    data_timestamp: Optional[datetime] = None
    asset_capabilities: Optional[AssetCapabilities] = None
    market_is_open: Optional[bool] = None
