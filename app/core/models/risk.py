from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.models.knowledge import Evidence


class RiskDecision(str, Enum):
    APPROVED = "APPROVED"
    APPROVED_WITH_WARNINGS = "APPROVED_WITH_WARNINGS"
    BLOCKED = "BLOCKED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RiskSeverity(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class RiskRuleType(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


class RiskDimension(str, Enum):
    POSITION_SIZE = "POSITION_SIZE"
    CASH_RESERVE = "CASH_RESERVE"
    CONCENTRATION = "CONCENTRATION"
    SECTOR = "SECTOR"
    ASSET_CLASS = "ASSET_CLASS"
    GEOGRAPHIC = "GEOGRAPHIC"
    POSITION_COUNT = "POSITION_COUNT"
    VOLATILITY = "VOLATILITY"
    DRAWDOWN = "DRAWDOWN"
    LIQUIDITY = "LIQUIDITY"
    CRISIS = "CRISIS"
    FUNDAMENTAL = "FUNDAMENTAL"
    THESIS = "THESIS"
    CORRELATION = "CORRELATION"
    DATA_QUALITY = "DATA_QUALITY"
    PORTFOLIO_STATE = "PORTFOLIO_STATE"


@dataclass
class InvestmentProposal:
    proposal_id: str
    asset: str
    proposed_position_value: float
    asset_class: Optional[str] = None
    sector: Optional[str] = None
    geography: Optional[str] = None
    investment_id: Optional[str] = None
    quantity: Optional[float] = None
    limit_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionExposure:
    symbol: str
    market_value: float
    asset_class: Optional[str] = None
    sector: Optional[str] = None
    geography: Optional[str] = None
    correlation_group: Optional[str] = None


@dataclass
class PortfolioRiskState:
    portfolio_value: float
    available_cash: float
    positions: List[PositionExposure] = field(default_factory=list)
    historical_peak_value: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskPolicy:
    max_single_position_pct: float = 0.10
    max_portfolio_concentration_pct: float = 0.15
    max_sector_exposure_pct: float = 0.30
    max_asset_class_exposure_pct: float = 0.60
    max_geographic_exposure_pct: float = 0.60
    min_cash_reserve_pct: float = 0.20
    max_portfolio_drawdown_pct: float = 0.25
    volatility_warning_threshold: float = 0.35
    volatility_block_threshold: float = 0.60
    crisis_block_severity: RiskSeverity = RiskSeverity.CRITICAL
    crisis_warning_severity: RiskSeverity = RiskSeverity.HIGH
    max_correlated_exposure_pct: float = 0.35
    max_positions: int = 10
    min_data_confidence: float = 0.50
    stale_market_data_minutes: int = 60
    require_market_data: bool = True
    require_deep_research: bool = False
    block_on_severe_crisis: bool = True


@dataclass
class RiskRuleResult:
    rule_id: str
    dimension: RiskDimension
    rule_type: RiskRuleType
    passed: bool
    description: str
    observed_value: Optional[float] = None
    limit_value: Optional[float] = None
    severity: RiskSeverity = RiskSeverity.LOW
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    assessment_id: str
    generation_id: str
    proposal_id: str
    timestamp: datetime
    asset: str
    proposed_position_size: Optional[float]
    proposed_position_value: float
    portfolio_value: float
    available_cash: float
    resulting_cash: float
    portfolio_exposure: float
    concentration_exposure: float
    sector_exposure: Optional[float]
    geographic_exposure: Optional[float]
    asset_class_exposure: Optional[float]
    volatility_risk: RiskSeverity
    drawdown_risk: RiskSeverity
    liquidity_risk: RiskSeverity
    crisis_risk: RiskSeverity
    fundamental_risk: RiskSeverity
    thesis_risk: RiskSeverity
    correlation_risk: RiskSeverity
    data_quality_risk: RiskSeverity
    warnings: List[str]
    violated_rules: List[RiskRuleResult]
    passed_rules: List[RiskRuleResult]
    required_actions: List[str]
    decision: RiskDecision
    confidence: float
    reasoning: str
    evidence: List[Evidence] = field(default_factory=list)
    recommended_max_position_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def execution_allowed(self) -> bool:
        return self.decision in (RiskDecision.APPROVED, RiskDecision.APPROVED_WITH_WARNINGS)
