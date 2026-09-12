from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.models.agent import AgentResult
from app.core.models.risk import RiskAssessment
from app.core.models.investment_safety import InvestmentSafetyAssessment


class RequestType(str, Enum):
    """Types of orchestration requests."""
    INVESTMENT_RESEARCH = "INVESTMENT_RESEARCH"
    INVESTMENT_PROPOSAL = "INVESTMENT_PROPOSAL"
    EXISTING_INVESTMENT_REVIEW = "EXISTING_INVESTMENT_REVIEW"
    MARKET_EVENT = "MARKET_EVENT"
    CRISIS_REASSESSMENT = "CRISIS_REASSESSMENT"
    PORTFOLIO_REVIEW = "PORTFOLIO_REVIEW"


class OrchestrationStage(str, Enum):
    """Stages in the orchestration pipeline."""
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    RESEARCHING = "RESEARCHING"
    DEEP_ANALYSIS = "DEEP_ANALYSIS"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    SAFETY_ASSESSMENT = "SAFETY_ASSESSMENT"
    DECISION_SYNTHESIS = "DECISION_SYNTHESIS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    DEFERRED = "DEFERRED"


class DecisionType(str, Enum):
    """Decision types for new investments."""
    INVEST = "INVEST"
    DO_NOT_INVEST = "DO_NOT_INVEST"
    DEFER = "DEFER"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ExistingInvestmentDecision(str, Enum):
    """Decision types for existing investments."""
    HOLD = "HOLD"
    REVIEW = "REVIEW"
    EXIT_CANDIDATE = "EXIT_CANDIDATE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class DecisionStatus(str, Enum):
    """Status of a decision proposal."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"
    BLOCKED = "BLOCKED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ConflictCategory(str, Enum):
    """Categories of conflicts between agent outputs."""
    MARKET_VS_FUNDAMENTAL = "MARKET_VS_FUNDAMENTAL"
    NEWS_VS_DEEP_ANALYSIS = "NEWS_VS_DEEP_ANALYSIS"
    CRISIS_VS_THESIS = "CRISIS_VS_THESIS"
    RISK_VS_REWARD = "RISK_VS_REWARD"
    DATA_QUALITY = "DATA_QUALITY"
    TIMELINESS = "TIMELINESS"
    OTHER = "OTHER"


class ConflictSeverity(str, Enum):
    """Severity of conflicts."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConflictResolutionStatus(str, Enum):
    """Status of conflict resolution."""
    UNRESOLVED = "UNRESOLVED"
    DEFERRED_FOR_ANALYSIS = "DEFERRED_FOR_ANALYSIS"
    RESOLVED_IN_FAVOR_OF_EVIDENCE = "RESOLVED_IN_FAVOR_OF_EVIDENCE"
    RESOLVED_BY_RISK_GATE = "RESOLVED_BY_RISK_GATE"
    ACKNOWLEDGED_WITH_WARNINGS = "ACKNOWLEDGED_WITH_WARNINGS"


@dataclass
class OrchestrationRequest:
    """Request for CEO orchestration."""
    request_id: str
    generation_id: str
    request_type: RequestType
    asset: str
    objective: str
    timestamp: datetime
    priority: int = 1
    requested_position_size: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentExecution:
    """Record of an agent execution within orchestration."""
    agent_id: str
    task_id: str
    stage: OrchestrationStage
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "PENDING"
    result: Optional[AgentResult] = None
    error: Optional[str] = None
    timeout: bool = False


@dataclass
class Conflict:
    """Conflict between agent outputs."""
    conflict_id: str
    category: ConflictCategory
    agents_involved: List[str]
    statements: List[str]
    evidence: List[str]
    severity: ConflictSeverity
    resolution_status: ConflictResolutionStatus = ConflictResolutionStatus.UNRESOLVED
    detected_at: datetime = field(default_factory=lambda: datetime.utcnow())
    notes: Optional[str] = None


@dataclass
class Evidence:
    """Structured evidence with traceability."""
    fact: str
    source_agent: str
    source_id: str
    timestamp: datetime
    confidence: float
    interpretation: Optional[str] = None
    supporting_for_decision: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationRun:
    """A complete orchestration run."""
    run_id: str
    request_id: str
    generation_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: OrchestrationStage = OrchestrationStage.CREATED
    current_stage: OrchestrationStage = OrchestrationStage.CREATED
    agent_executions: List[AgentExecution] = field(default_factory=list)
    risk_assessment: Optional[RiskAssessment] = None
    safety_assessment: Optional[InvestmentSafetyAssessment] = None
    decision: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionProposal:
    """Formal investment decision proposal."""
    decision_id: str
    generation_id: str
    asset: str
    decision_type: str  # DecisionType or ExistingInvestmentDecision
    proposed_action: str
    position_size: Optional[float]
    thesis: str
    supporting_evidence: List[str] = field(default_factory=list)
    opposing_evidence: List[str] = field(default_factory=list)
    market_context: Optional[str] = None
    news_context: Optional[str] = None
    crisis_context: Optional[str] = None
    deep_analysis: Optional[str] = None
    risk_assessment: Optional[str] = None
    confidence: float = 0.0
    expected_outcome: Optional[str] = None
    invalidation_conditions: List[str] = field(default_factory=list)
    decision_reason: str = ""
    status: DecisionStatus = DecisionStatus.PENDING
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    metadata: Dict[str, Any] = field(default_factory=dict)
