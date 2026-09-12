from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.models.knowledge import Evidence, Source
from app.core.models.market import AssetCapabilities


class ResearchDepth(str, Enum):
    STANDARD = "STANDARD"
    DEEP = "DEEP"
    COMPREHENSIVE = "COMPREHENSIVE"


class CompletenessLevel(str, Enum):
    COMPLETE = "COMPLETE"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    STALE = "STALE"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT = "INSUFFICIENT"


class ValuationBasis(str, Enum):
    HISTORICAL = "HISTORICAL"
    FORWARD_ESTIMATED = "FORWARD_ESTIMATED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ValuationContextLabel(str, Enum):
    BELOW_OWN_HISTORY = "BELOW_OWN_HISTORY"
    IN_LINE_WITH_OWN_HISTORY = "IN_LINE_WITH_OWN_HISTORY"
    ABOVE_OWN_HISTORY = "ABOVE_OWN_HISTORY"
    BELOW_PEER = "BELOW_PEER"
    IN_LINE_WITH_PEER = "IN_LINE_WITH_PEER"
    ABOVE_PEER = "ABOVE_PEER"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class CatalystDirection(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"
    UNCERTAIN = "UNCERTAIN"


class ThesisStatus(str, Enum):
    STRONG_SUPPORT = "STRONG_SUPPORT"
    MODERATE_SUPPORT = "MODERATE_SUPPORT"
    MIXED = "MIXED"
    WEAK_SUPPORT = "WEAK_SUPPORT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ContradictionStatus(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    RESOLVED = "RESOLVED"


class RiskCategory(str, Enum):
    BUSINESS = "BUSINESS"
    FINANCIAL = "FINANCIAL"
    VALUATION = "VALUATION"
    MARKET = "MARKET"
    LIQUIDITY = "LIQUIDITY"
    GEOPOLITICAL = "GEOPOLITICAL"
    REGULATORY = "REGULATORY"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    CURRENCY = "CURRENCY"
    CONCENTRATION = "CONCENTRATION"
    TECHNOLOGY = "TECHNOLOGY"
    EXECUTION = "EXECUTION"


class ProbabilityAssessment(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass
class DeepResearchRequest:
    symbol: str
    analysis_timestamp: datetime
    requested_depth: ResearchDepth = ResearchDepth.DEEP
    historical_window: str = "medium"
    included_sources: List[str] = field(default_factory=list)
    generation_id: str = "gen_current"
    originating_task_id: str = ""
    optional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatedValue:
    value: Optional[float]
    event_time: Optional[datetime] = None
    publication_time: Optional[datetime] = None
    data_period: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    status: str = "OK"  # OK | INSUFFICIENT_DATA | STALE


@dataclass
class FinancialHealthMetrics:
    """Deterministic health metrics. None means INSUFFICIENT_DATA, never a fabricated zero."""
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_to_cash: Optional[float] = None
    free_cash_flow_growth: Optional[float] = None
    interest_coverage: Optional[float] = None
    formulas: Dict[str, str] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)


@dataclass
class ComputedValuation:
    pe_historical: Optional[float] = None
    pe_forward: Optional[float] = None
    ps: Optional[float] = None
    pb: Optional[float] = None
    ev_ebitda: Optional[float] = None
    fcf_yield: Optional[float] = None
    basis_notes: Dict[str, ValuationBasis] = field(default_factory=dict)
    context_labels: List[ValuationContextLabel] = field(default_factory=list)
    formulas: Dict[str, str] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)


@dataclass
class AssetIdentity:
    symbol: str
    name: str
    asset_type: str
    sector: str
    industry: str
    country: str
    capabilities: AssetCapabilities
    description: str = ""


@dataclass
class Catalyst:
    catalyst: str
    expected_timeframe: str
    direction: CatalystDirection
    evidence_ids: List[str]
    uncertainty: str
    confidence: float


@dataclass
class IdentifiedRisk:
    category: RiskCategory
    description: str
    severity: str
    probability: ProbabilityAssessment
    potential_impact: str
    time_horizon: str
    evidence_ids: List[str]
    confidence: float


@dataclass
class Contradiction:
    claim_a: str
    claim_b: str
    evidence_a: str
    evidence_b: str
    severity: str
    status: ContradictionStatus
    confidence: float


@dataclass
class ThesisAssumption:
    assumption: str
    evidence: str
    confidence: float
    invalidation_condition: str


@dataclass
class ScenarioCase:
    name: str  # BULL | BASE | BEAR
    assumptions: List[str]
    supporting_evidence: List[str]
    risks: List[str]
    time_horizon: str
    confidence: float


@dataclass
class InvestmentThesis:
    story: str
    supporting_factors: List[str]
    invalidation_factors: List[str]
    what_would_strengthen: List[str]
    what_would_weaken: List[str]
    status: ThesisStatus
    assumptions: List[ThesisAssumption] = field(default_factory=list)
    scenarios: List[ScenarioCase] = field(default_factory=list)
    invalidation_conditions: List[str] = field(default_factory=list)


@dataclass
class DataCompleteness:
    fundamentals: CompletenessLevel = CompletenessLevel.MISSING
    market_data: CompletenessLevel = CompletenessLevel.MISSING
    news: CompletenessLevel = CompletenessLevel.MISSING
    geopolitical: CompletenessLevel = CompletenessLevel.MISSING
    valuation: CompletenessLevel = CompletenessLevel.MISSING
    notes: List[str] = field(default_factory=list)


@dataclass
class DeepResearchDossier:
    research_id: str
    symbol: str
    timestamp: datetime
    request: DeepResearchRequest
    asset_identity: AssetIdentity
    business_or_asset_model: Dict[str, Any]
    market_analysis: Dict[str, Any]
    fundamental_analysis: Dict[str, Any]
    valuation_analysis: Dict[str, Any]
    competitive_analysis: Dict[str, Any]
    news_analysis: Dict[str, Any]
    geopolitical_analysis: Dict[str, Any]
    regulatory_analysis: Dict[str, Any]
    risk_analysis: List[IdentifiedRisk]
    catalysts: List[Catalyst]
    contradictions: List[Contradiction]
    unknowns: List[str]
    thesis: InvestmentThesis
    evidence: List[Evidence]
    confidence: float
    sources: List[Source]
    completeness: DataCompleteness
    health_metrics: FinancialHealthMetrics
    computed_valuation: ComputedValuation
    governance_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_decision_context(self) -> Dict[str, Any]:
        """Payload later DecisionRecord/InvestmentRecord can reference. Does not create an investment."""
        return {
            "research_id": self.research_id,
            "symbol": self.symbol,
            "thesis_status": self.thesis.status.value,
            "confidence": self.confidence,
            "invalidation_conditions": list(self.thesis.invalidation_conditions),
            "assumptions": [a.assumption for a in self.thesis.assumptions],
            "contradiction_count": len(self.contradictions),
            "timestamp": self.timestamp.isoformat(),
        }
