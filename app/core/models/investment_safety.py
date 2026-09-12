from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.models.knowledge import Evidence, Source


class SafetyRecommendation(str, Enum):
    HOLD = "HOLD"
    REVIEW = "REVIEW"
    EXIT_CANDIDATE = "EXIT_CANDIDATE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ThesisStatus(str, Enum):
    STRONGLY_SUPPORTED = "STRONGLY_SUPPORTED"
    SUPPORTED = "SUPPORTED"
    MIXED = "MIXED"
    WEAKENING = "WEAKENING"
    INVALIDATED = "INVALIDATED"
    UNKNOWN = "UNKNOWN"


class ThesisConditionCategory(str, Enum):
    FUNDAMENTAL = "FUNDAMENTAL"
    MARKET = "MARKET"
    VALUATION = "VALUATION"
    BUSINESS = "BUSINESS"
    COMPETITIVE = "COMPETITIVE"
    MACRO = "MACRO"
    GEOPOLITICAL = "GEOPOLITICAL"
    REGULATORY = "REGULATORY"
    MANAGEMENT = "MANAGEMENT"
    LIQUIDITY = "LIQUIDITY"
    OTHER = "OTHER"


class ThesisConditionStatus(str, Enum):
    VALID = "VALID"
    WEAKENING = "WEAKENING"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"


class TimeHorizon(str, Enum):
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ThesisCondition:
    condition_id: str
    description: str
    category: ThesisConditionCategory
    status: ThesisConditionStatus
    severity: str
    source: str
    observed_value: Optional[float] = None
    expected_value: Optional[float] = None
    threshold_operator: Optional[str] = None
    detected_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Contradiction:
    contradiction_id: str
    original_statement: str
    contradictory_evidence: str
    source: str
    severity: str
    confidence: float
    affected_thesis_component: str
    detected_at: datetime
    evidence_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssessmentDimension:
    dimension_name: str
    status: str
    confidence: float
    key_findings: List[str]
    warnings: List[str]
    evidence: List[Evidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvestmentSafetyAssessment:
    assessment_id: str
    investment_id: str
    generation_id: str
    asset: str
    timestamp: datetime
    original_thesis: str
    original_entry_price: float
    current_price: float
    current_position_value: float
    unrealized_profit_loss: float
    unrealized_return_percentage: float
    original_time_horizon: TimeHorizon
    original_risk_level: RiskLevel
    thesis_status: ThesisStatus
    recommendation: SafetyRecommendation
    thesis_supporting_factors: List[str]
    thesis_invalidating_factors: List[str]
    new_risks: List[str]
    changed_conditions: List[str]
    market_assessment: Optional[AssessmentDimension]
    fundamental_assessment: Optional[AssessmentDimension]
    news_assessment: Optional[AssessmentDimension]
    crisis_assessment: Optional[AssessmentDimension]
    valuation_assessment: Optional[AssessmentDimension]
    drawdown_assessment: Optional[AssessmentDimension]
    contradiction_findings: List[Contradiction]
    catalysts: List[str]
    warnings: List[str]
    confidence: float
    evidence: List[Evidence]
    sources: List[Source]
    required_follow_up: List[str]
    previous_recommendation: Optional[SafetyRecommendation] = None
    previous_thesis_status: Optional[ThesisStatus] = None
    assessment_comparison: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def requires_higher_level_review(self) -> bool:
        return self.recommendation in (SafetyRecommendation.REVIEW, SafetyRecommendation.EXIT_CANDIDATE)

    def is_critical(self) -> bool:
        return self.recommendation == SafetyRecommendation.EXIT_CANDIDATE

    def has_contradictions(self) -> bool:
        return len(self.contradiction_findings) > 0