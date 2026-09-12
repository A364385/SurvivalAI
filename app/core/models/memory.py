from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

class MemoryType(Enum):
    FACT = "FACT"
    ANALYSIS = "ANALYSIS"
    DECISION = "DECISION"
    INVESTMENT = "INVESTMENT"
    OUTCOME = "OUTCOME"
    EXPERIENCE = "EXPERIENCE"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DEATH = "DEATH"
    STRATEGY = "STRATEGY"
    AGENT_PERFORMANCE = "AGENT_PERFORMANCE"
    GENERATION = "GENERATION"

class GenerationStatus(Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    DEAD = "DEAD"

class StrategyStatus(Enum):
    ACTIVE = "ACTIVE"
    TESTING = "TESTING"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"

class InvestmentStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"

@dataclass
class MemoryRecord:
    memory_id: str
    memory_type: MemoryType
    generation_id: str
    timestamp: datetime
    source_agent: str
    importance: int
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Experience:
    experience_id: str
    generation_id: str
    event_type: str
    context: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    result: str
    reward: float
    loss: float
    lessons: List[str]
    contributing_agents: List[str]
    timestamp: datetime

@dataclass
class DecisionRecord:
    decision_id: str
    generation_id: str
    decision_type: str
    asset: str
    timestamp: datetime
    reasoning_reference: str
    supporting_agent_results: List[str]
    risk_assessment: Dict[str, Any]
    crisis_assessment: Dict[str, Any]
    decision: Dict[str, Any]
    confidence: float
    expected_outcome: Dict[str, Any]
    actual_outcome: Optional[Dict[str, Any]] = None

@dataclass
class InvestmentRecord:
    investment_id: str
    asset: str
    entry_price: float
    entry_timestamp: datetime
    position_size: float
    investment_thesis: str
    time_horizon: str
    risk_level: str
    originating_generation: str
    status: InvestmentStatus
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    realized_profit_loss: Optional[float] = None

@dataclass
class AgentPerformanceRecord:
    agent_id: str
    generation_id: str
    tasks_completed: int
    tasks_failed: int
    useful_predictions: int
    incorrect_predictions: int
    contribution_score: float
    cost: float
    performance_notes: List[str] = field(default_factory=list)

@dataclass
class GenerationRecord:
    generation_id: str
    parent_generation_id: Optional[str]
    generation_number: int
    strategy_version: str
    starting_capital: float
    ending_capital: float
    start_timestamp: datetime
    end_timestamp: Optional[datetime]
    lifespan: Optional[int]
    return_percentage: Optional[float]
    maximum_drawdown: Optional[float]
    status: GenerationStatus
    cause_of_death: Optional[str] = None

@dataclass
class DeathReport:
    generation_id: str
    starting_capital: float
    final_capital: float
    lifespan: int
    total_return: float
    maximum_drawdown: float
    largest_losses: List[Dict[str, Any]]
    major_decisions: List[str]
    failed_decisions: List[str]
    successful_decisions: List[str]
    agent_performance: List[AgentPerformanceRecord]
    market_conditions: Dict[str, Any]
    crisis_conditions: Dict[str, Any]
    active_strategy: str
    suspected_causes: List[str]
    lessons_for_future_analysis: List[str]

@dataclass
class StrategyVersion:
    strategy_id: str
    version: str
    parent_strategy_id: Optional[str]
    created_at: datetime
    generation_id: str
    parameters: Dict[str, Any]
    rules: Dict[str, Any]
    description: str
    status: StrategyStatus
