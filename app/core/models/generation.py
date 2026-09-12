from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.core.models.memory import GenerationStatus as BaseGenerationStatus
from app.core.models.memory import GenerationRecord, Experience, StrategyVersion
from app.core.models.strategy import ProposalStatus
from app.utils.ids import generate_id


class GenerationLifecycleState(Enum):
    """Extended generation lifecycle states."""
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    DYING = "DYING"
    DEAD = "DEAD"
    SUCCESSOR_PENDING = "SUCCESSOR_PENDING"
    SUCCESSOR_CREATED = "SUCCESSOR_CREATED"
    ARCHIVED = "ARCHIVED"


class DeathTrigger(Enum):
    """Configurable death triggers."""
    CAPITAL_DEPLETED = "CAPITAL_DEPLETED"
    MINIMUM_SURVIVAL_THRESHOLD = "MINIMUM_SURVIVAL_THRESHOLD"
    MAXIMUM_DRAWDOWN_EXCEEDED = "MAXIMUM_DRAWDOWN_EXCEEDED"
    UNRECOVERABLE_PORTFOLIO_STATE = "UNRECOVERABLE_PORTFOLIO_STATE"
    OPERATING_COSTS_EXCEEDED = "OPERATING_COSTS_EXCEEDED"
    CONFIGURED_CONDITION_VIOLATED = "CONFIGURED_CONDITION_VIOLATED"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"


@dataclass
class OperatingCost:
    """Operating cost abstraction for generations."""
    cost_id: str
    cost_type: str  # API, DATA, MODEL, INFRASTRUCTURE, PAPER_TRADING, RECURRING
    amount: float
    currency: str
    frequency: str  # ONE_TIME, DAILY, WEEKLY, MONTHLY
    description: str
    provider: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class GenerationState:
    """Extended generation state with lifecycle information."""
    generation_id: str
    parent_generation_id: Optional[str]
    generation_number: int
    strategy_version: str
    lifecycle_state: GenerationLifecycleState
    creation_timestamp: datetime
    start_timestamp: Optional[datetime]
    end_timestamp: Optional[datetime]
    starting_capital: float
    current_capital: float
    ending_capital: Optional[float]
    return_percentage: Optional[float]
    maximum_drawdown: float
    lifespan_days: Optional[int]
    cause_of_death: Optional[str]
    death_trigger: Optional[DeathTrigger]
    death_timestamp: Optional[datetime]
    death_details: Dict[str, Any] = field(default_factory=dict)
    inherited_experience_refs: List[str] = field(default_factory=list)
    inherited_strategy_refs: List[str] = field(default_factory=list)
    successor_generation_id: Optional[str] = None
    operating_costs: List[OperatingCost] = field(default_factory=list)
    total_operating_costs: float = 0.0
    active_investments: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InheritedKnowledge:
    """Knowledge inherited from parent generation."""
    knowledge_id: str
    source_generation_id: str
    source_type: str  # EXPERIENCE, STRATEGY, BACKTEST, AGENT_PERFORMANCE
    relevance: float
    confidence: float
    timestamp: datetime
    validation_status: str  # VALIDATED, UNVALIDATED, REJECTED
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationComparisonResult:
    """Comparison between two generations."""
    generation_a_id: str
    generation_b_id: str
    comparison_timestamp: datetime
    total_return_difference: float
    maximum_drawdown_difference: float
    survival_duration_difference: int
    ending_capital_difference: float
    number_of_investments_difference: int
    win_rate_difference: Optional[float]
    transaction_cost_difference: float
    operating_cost_difference: float
    crisis_performance_difference: Dict[str, Any]
    market_regime_performance_difference: Dict[str, Any]
    benchmark_performance_difference: Dict[str, Any]
    strategy_version_difference: str
    risk_violation_count_difference: int
    blocked_investment_count_difference: int
    major_failures: List[str]
    fitness_scores: Dict[str, float]
    recommended_generation: Optional[str]


@dataclass
class DeathCondition:
    """Configurable death condition."""
    condition_id: str
    trigger: DeathTrigger
    threshold: float
    is_fatal: bool
    description: str
    enabled: bool = True


@dataclass
class ExperienceSelectionCriteria:
    """Criteria for selecting relevant experiences."""
    min_relevance: float = 0.5
    min_confidence: float = 0.6
    max_age_days: Optional[int] = None
    required_outcomes: Optional[List[str]] = None
    required_asset_classes: Optional[List[str]] = None
    required_regimes: Optional[List[str]] = None
    require_validated: bool = True
    max_selection_count: Optional[int] = None
