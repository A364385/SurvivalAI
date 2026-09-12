"""Crypto-specific data models for SurvivalAI.

These models extend the general market models with crypto-specific
metrics like on-chain data, tokenomics, and crypto risk factors.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class CryptoRiskType(str, Enum):
    """Specific risk factors for cryptocurrency assets."""
    EXTREME_VOLATILITY = "EXTREME_VOLATILITY"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
    EXCHANGE_COUNTERPARTY_RISK = "EXCHANGE_COUNTERPARTY_RISK"
    SMART_CONTRACT_RISK = "SMART_CONTRACT_RISK"
    PROTOCOL_RISK = "PROTOCOL_RISK"
    REGULATORY_RISK = "REGULATORY_RISK"
    CUSTODY_RISK = "CUSTODY_RISK"
    CONCENTRATION_RISK = "CONCENTRATION_RISK"
    TOKEN_UNLOCK_RISK = "TOKEN_UNLOCK_RISK"
    STABLECOIN_RISK = "STABLECOIN_RISK"
    NETWORK_OUTAGE_RISK = "NETWORK_OUTAGE_RISK"
    SECURITY_INCIDENT_RISK = "SECURITY_INCIDENT_RISK"
    MARKET_MANIPULATION_RISK = "MARKET_MANIPULATION_RISK"
    CORRELATION_RISK = "CORRELATION_RISK"
    DEPEG_RISK = "DEPEG_RISK"


class CryptoRegime(str, Enum):
    """Crypto-specific market regimes."""
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    ALTCOIN_SEASON = "ALTCOIN_SEASON"
    BITCOIN_DOMINANCE = "BITCOIN_DOMINANCE"
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    UNKNOWN = "UNKNOWN"


@dataclass
class CryptoTokenomics:
    """Token economic metrics."""
    token_id: str
    symbol: str
    circulating_supply: Optional[float] = None
    maximum_supply: Optional[float] = None
    total_supply: Optional[float] = None
    inflation_rate: Optional[float] = None
    emission_schedule: Optional[str] = None
    market_capitalization: Optional[float] = None
    fully_diluted_valuation: Optional[float] = None
    burn_mechanism: Optional[bool] = None
    staking_enabled: Optional[bool] = None
    staking_rate: Optional[float] = None
    governance_token: Optional[bool] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnChainMetrics:
    """On-chain network activity metrics."""
    token_id: str
    symbol: str
    active_addresses_24h: Optional[int] = None
    transaction_count_24h: Optional[int] = None
    transaction_volume_24h: Optional[float] = None
    average_transaction_value: Optional[float] = None
    exchange_inflow_24h: Optional[float] = None
    exchange_outflow_24h: Optional[float] = None
    net_exchange_flow: Optional[float] = None
    whale_transactions_24h: Optional[int] = None
    large_holders_count: Optional[int] = None
    hashrate: Optional[float] = None  # For PoW coins
    staked_amount: Optional[float] = None
    tvl: Optional[float] = None  # Total Value Locked for DeFi
    protocol_revenue_24h: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUnlock:
    """Scheduled token unlock event."""
    unlock_id: str
    symbol: str
    unlock_date: datetime
    amount: float
    percentage_of_supply: float
    source: Optional[str] = None  # Team, investors, ecosystem, etc.
    cliff_vesting: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CryptoRiskFactor:
    """Specific crypto risk factor with evidence."""
    risk_id: str
    risk_type: CryptoRiskType
    symbol: str
    severity: str  # LOW, MODERATE, HIGH, CRITICAL
    confidence: float
    evidence: List[str] = field(default_factory=list)
    time_horizon: Optional[str] = None  # SHORT_TERM, MEDIUM_TERM, LONG_TERM
    affected_assets: List[str] = field(default_factory=list)
    description: str = ""
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CryptoMarketStructure:
    """Crypto market structure analysis."""
    symbol: str
    trend: str
    momentum: str
    volatility: str
    liquidity: str
    drawdown: float
    breakout_condition: Optional[str] = None
    breakdown_condition: Optional[str] = None
    regime: CryptoRegime = CryptoRegime.UNKNOWN
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CryptoMarketWideConditions:
    """Market-wide crypto conditions."""
    total_crypto_market_cap: Optional[float] = None
    btc_dominance: Optional[float] = None
    eth_dominance: Optional[float] = None
    stablecoin_market_cap: Optional[float] = None
    total_market_trend: Optional[str] = None
    fear_greed_index: Optional[int] = None
    funding_rates: Dict[str, float] = field(default_factory=dict)
    open_interest: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CryptoAssetCorrelation:
    """Correlation between crypto assets."""
    asset_pair: str  # e.g., "BTC-ETH"
    correlation_30d: Optional[float] = None
    correlation_90d: Optional[float] = None
    beta: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CryptoSnapshot:
    """Unified crypto snapshot combining market, tokenomics, and on-chain data."""
    symbol: str
    timestamp: datetime
    current_price: float
    market_structure: CryptoMarketStructure
    tokenomics: Optional[CryptoTokenomics] = None
    on_chain_metrics: Optional[OnChainMetrics] = None
    risk_factors: List[CryptoRiskFactor] = field(default_factory=list)
    market_wide_conditions: Optional[CryptoMarketWideConditions] = None
    correlations: List[CryptoAssetCorrelation] = field(default_factory=list)
    upcoming_unlocks: List[TokenUnlock] = field(default_factory=list)
    data_quality: Optional[str] = None  # VALID, INSUFFICIENT_DATA, STALE
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
