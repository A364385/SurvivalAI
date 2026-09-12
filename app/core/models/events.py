from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List

@dataclass
class BaseEvent:
    event_id: str
    timestamp: datetime
    event_type: str

# Agent Events
@dataclass
class AgentStarted(BaseEvent):
    agent_id: str
    task_id: str

@dataclass
class AgentFinished(BaseEvent):
    agent_id: str
    task_id: str
    status: str

@dataclass
class AgentFailed(BaseEvent):
    agent_id: str
    task_id: str
    error_message: str

# World Events
@dataclass
class MarketEvent(BaseEvent):
    asset_id: str
    price_change: float
    details: Dict[str, Any]

@dataclass
class NewsEvent(BaseEvent):
    news_id: str
    headline: str
    impact_score: float
    cluster_id: Optional[str] = None
    affected_symbols: List[str] = field(default_factory=list)
    news_category: Optional[str] = None

@dataclass
class CrisisEvent(BaseEvent):
    crisis_id: str
    severity: int
    description: str
    title: str = ""
    severity_label: str = "UNKNOWN"
    escalation_status: str = "UNKNOWN"
    geographic_scope: str = "UNKNOWN"
    countries_involved: List[str] = field(default_factory=list)
    is_update: bool = False
    agent_id: str = ""
    task_id: str = ""

# Investment Events
@dataclass
class InvestmentCreated(BaseEvent):
    investment_id: str
    asset_id: str
    amount: float

@dataclass
class InvestmentReviewed(BaseEvent):
    investment_id: str
    decision: str

# Generational Events
@dataclass
class GenerationStarted(BaseEvent):
    generation_id: str
    starting_capital: float

@dataclass
class GenerationDied(BaseEvent):
    generation_id: str
    reason: str
    survival_time_days: int

# Strategy Events
@dataclass
class StrategyUpdated(BaseEvent):
    strategy_id: str
    new_rules: Dict[str, Any]

# Execution & Order Lifecycle Events
@dataclass
class OrderSubmitted(BaseEvent):
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float

@dataclass
class OrderAccepted(BaseEvent):
    order_id: str
    client_order_id: str
    symbol: str

@dataclass
class OrderPartiallyFilled(BaseEvent):
    order_id: str
    symbol: str
    filled_quantity: float
    average_fill_price: float

@dataclass
class OrderFilled(BaseEvent):
    order_id: str
    symbol: str
    filled_quantity: float
    average_fill_price: float

@dataclass
class OrderRejected(BaseEvent):
    order_id: str
    reason: str

@dataclass
class OrderCancelled(BaseEvent):
    order_id: str
    reason: str

# Portfolio Synchronization Events
@dataclass
class PositionUpdated(BaseEvent):
    symbol: str
    quantity: float
    market_value: float
    unrealized_pl: float

@dataclass
class AccountUpdated(BaseEvent):
    equity: float
    cash: float
    buying_power: float

@dataclass
class MarketDataReceived(BaseEvent):
    symbol: str
    data_type: str


@dataclass
class MarketAnalysisCompleted(BaseEvent):
    """Published after a Market Research Agent run. Research-only; not a trade signal."""
    agent_id: str
    task_id: str
    symbols: List[str]
    primary_symbol: str
    trend: str
    regime: str
    anomaly_count: int


@dataclass
class DeepResearchCompleted(BaseEvent):
    """Published after Deep Looker research. Not an investment approval."""
    research_id: str
    asset: str
    agent_id: str
    task_id: str
    confidence: float
    thesis_status: str
    key_findings: List[str] = field(default_factory=list)
    risk_summary: str = ""


@dataclass
class RiskAssessmentStarted(BaseEvent):
    agent_id: str
    task_id: str
    proposal_id: str
    asset: str


@dataclass
class RiskAssessmentCompleted(BaseEvent):
    agent_id: str
    task_id: str
    assessment_id: str
    proposal_id: str
    asset: str
    decision: str
    confidence: float
    violated_rule_count: int = 0


@dataclass
class RiskViolationEvent(BaseEvent):
    agent_id: str
    task_id: str
    assessment_id: str
    proposal_id: str
    asset: str
    rule_id: str
    severity: str
    description: str


# Investment Safety Events
@dataclass
class SafetyAssessmentStarted(BaseEvent):
    agent_id: str
    task_id: str
    investment_id: str
    asset: str


@dataclass
class SafetyAssessmentCompleted(BaseEvent):
    agent_id: str
    task_id: str
    assessment_id: str
    investment_id: str
    asset: str
    recommendation: str
    thesis_status: str
    contradiction_count: int


@dataclass
class ThesisWeakenedEvent(BaseEvent):
    agent_id: str
    task_id: str
    assessment_id: str
    investment_id: str
    asset: str
    previous_status: str
    current_status: str
    reason: str


@dataclass
class ThesisInvalidatedEvent(BaseEvent):
    agent_id: str
    task_id: str
    assessment_id: str
    investment_id: str
    asset: str
    invalidation_reason: str
    severity: str


@dataclass
class ReviewRequiredEvent(BaseEvent):
    agent_id: str
    task_id: str
    assessment_id: str
    investment_id: str
    asset: str
    reason: str
    urgency: str


@dataclass
class ExitCandidateDetectedEvent(BaseEvent):
    agent_id: str
    task_id: str
    assessment_id: str
    investment_id: str
    asset: str
    primary_reason: str
    contradiction_count: int
    drawdown_percentage: Optional[float] = None


# Generation Lifecycle Events
@dataclass
class GenerationCreated(BaseEvent):
    generation_id: str
    parent_generation_id: Optional[str]
    generation_number: int
    strategy_version: str
    starting_capital: float


@dataclass
class GenerationInitializing(BaseEvent):
    generation_id: str
    strategy_version: str
    infrastructure_check_status: str


@dataclass
class GenerationPaused(BaseEvent):
    generation_id: str
    reason: str
    current_capital: float


@dataclass
class GenerationDying(BaseEvent):
    generation_id: str
    trigger: str
    reason: str
    current_capital: float
    maximum_drawdown: float


@dataclass
class SuccessorGenerationRequested(BaseEvent):
    parent_generation_id: str
    reason: str
    strategy_candidate_id: Optional[str] = None


@dataclass
class SuccessorGenerationCreated(BaseEvent):
    parent_generation_id: str
    successor_generation_id: str
    successor_generation_number: int
    strategy_version: str


@dataclass
class StrategyCandidateCreated(BaseEvent):
    parent_generation_id: str
    candidate_strategy_id: str
    parent_strategy_id: str
    motivation: str


@dataclass
class StrategyCandidateApproved(BaseEvent):
    candidate_strategy_id: str
    generation_id: str
    approval_timestamp: datetime
    confidence: float


@dataclass
class StrategyCandidateRejected(BaseEvent):
    candidate_strategy_id: str
    generation_id: str
    rejection_reason: str
    rejection_stage: str

@dataclass
class CostUpdateEvent(BaseEvent):
    cost_id: str
    cost_type: str
    amount: float
    currency: str
    provider: str
    description: str
