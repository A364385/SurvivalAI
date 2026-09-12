from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.models.events import BaseEvent


@dataclass
class OrchestrationStarted(BaseEvent):
    """Published when a CEO orchestration run begins."""
    run_id: str
    request_id: str
    generation_id: str
    request_type: str
    asset: str


@dataclass
class AgentDispatched(BaseEvent):
    """Published when the CEO dispatches an agent."""
    run_id: str
    agent_id: str
    task_id: str
    stage: str


@dataclass
class AgentCompleted(BaseEvent):
    """Published when an agent completes successfully."""
    run_id: str
    agent_id: str
    task_id: str
    stage: str
    duration_ms: float


@dataclass
class AgentFailed(BaseEvent):
    """Published when an agent fails."""
    run_id: str
    agent_id: str
    task_id: str
    stage: str
    error: str
    duration_ms: float


@dataclass
class DeepAnalysisStarted(BaseEvent):
    """Published when Deep Looker analysis begins."""
    run_id: str
    asset: str


@dataclass
class RiskAssessmentStarted(BaseEvent):
    """Published when Risk Manager assessment begins."""
    run_id: str
    asset: str
    proposal_id: str


@dataclass
class DecisionProposalCreated(BaseEvent):
    """Published when CEO creates a decision proposal."""
    run_id: str
    decision_id: str
    asset: str
    decision_type: str
    confidence: float


@dataclass
class InvestmentBlocked(BaseEvent):
    """Published when an investment is blocked by Risk Manager."""
    run_id: str
    asset: str
    reason: str
    violated_rules: List[str] = field(default_factory=list)


@dataclass
class OrchestrationCompleted(BaseEvent):
    """Published when orchestration completes successfully."""
    run_id: str
    request_id: str
    asset: str
    decision_type: str
    duration_ms: float
    agent_count: int


@dataclass
class OrchestrationFailed(BaseEvent):
    """Published when orchestration fails."""
    run_id: str
    request_id: str
    asset: str
    error: str
    stage: str
    duration_ms: float


@dataclass
class ConflictDetected(BaseEvent):
    """Published when conflicts between agent outputs are detected."""
    run_id: str
    conflict_id: str
    category: str
    severity: str
    agents_involved: List[str] = field(default_factory=list)
