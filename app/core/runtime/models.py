from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.utils.ids import generate_id


class RuntimeState(Enum):
    """Runtime state machine states."""
    STARTING = "STARTING"
    HEALTH_CHECK = "HEALTH_CHECK"
    INITIALIZING_GENERATION = "INITIALIZING_GENERATION"
    OBSERVING = "OBSERVING"
    RESEARCHING = "RESEARCHING"
    ANALYZING = "ANALYZING"
    RISK_CHECK = "RISK_CHECK"
    DECIDING = "DECIDING"
    EXECUTING = "EXECUTING"
    MONITORING = "MONITORING"
    LEARNING = "LEARNING"
    SURVIVAL_CHECK = "SURVIVAL_CHECK"
    GENERATION_TRANSITION = "GENERATION_TRANSITION"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class FailureType(Enum):
    """Failure classification."""
    TRANSIENT = "TRANSIENT"
    RECOVERABLE = "RECOVERABLE"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"


class CyclePhase(Enum):
    """Autonomous cycle phases."""
    OBSERVE = "OBSERVE"
    RESEARCH = "RESEARCH"
    ANALYZE = "ANALYZE"
    RISK_CHECK = "RISK_CHECK"
    DECIDE = "DECIDE"
    EXECUTE = "EXECUTE"
    MONITOR = "MONITOR"
    LEARN = "LEARN"
    SURVIVAL_CHECK = "SURVIVAL_CHECK"


@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    cycle_interval_seconds: int = 60
    market_data_interval_seconds: int = 30
    news_interval_seconds: int = 300
    crisis_interval_seconds: int = 60
    portfolio_sync_interval_seconds: int = 60
    investment_monitor_interval_seconds: int = 300
    learning_interval_seconds: int = 3600
    health_check_interval_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: int = 5
    stale_data_threshold_seconds: int = 300
    paper_mode_required: bool = True
    test_mode: bool = False
    api_timeout_seconds: int = 30
    decision_cooldown_seconds: int = 60
    max_concurrent_research: int = 3

    # --- Step 15 additions -------------------------------------------------
    # Symbols the runtime observes and may invest in.
    watched_symbols: List[str] = field(default_factory=lambda: ["AAPL"])
    # Fraction of portfolio equity proposed per new investment.
    proposed_position_pct: float = 0.05
    # Gate limits for decision frequency and exposure.
    max_open_positions: int = 10
    max_open_orders: int = 5
    min_research_confidence: float = 0.30
    # Minimum cash fraction that must remain after a proposed investment.
    min_cash_fraction_after_investment: float = 0.05
    # Maximum consecutive critical failures before the runtime stops.
    max_consecutive_critical_failures: int = 5
    # Bounded number of cycles (0 = unlimited). Useful for tests/simulation.
    max_cycles: int = 0
    # Configured operating costs (transparent; never fabricated).
    operating_costs: List[Dict[str, Any]] = field(default_factory=list)
    # Deep-analysis trigger configuration.
    deep_analysis_on_new_candidate: bool = True
    deep_analysis_on_major_news: bool = True
    deep_analysis_on_major_crisis: bool = True
    deep_analysis_on_unusual_market: bool = True
    # Paper-trading safety: runtime refuses execution unless verified.
    paper_only_enforced: bool = True


@dataclass
class RuntimeStateSnapshot:
    """Snapshot of runtime state."""
    runtime_id: str
    generation_id: Optional[str]
    current_state: RuntimeState
    previous_state: Optional[RuntimeState]
    timestamp: datetime
    cycle_id: Optional[str]
    cycle_phase: Optional[CyclePhase]
    health_status: Dict[str, bool]
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleRecord:
    """Record of an autonomous cycle."""
    cycle_id: str
    generation_id: str
    start_timestamp: datetime
    end_timestamp: Optional[datetime]
    phase: CyclePhase
    triggered_actions: List[str] = field(default_factory=list)
    agent_results: Dict[str, Any] = field(default_factory=dict)
    decision_result: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    portfolio_state: Optional[Dict[str, Any]] = None
    survival_state: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "IN_PROGRESS"


@dataclass
class FailureRecord:
    """Record of a failure."""
    failure_id: str
    failure_type: FailureType
    component: str
    error_message: str
    timestamp: datetime
    cycle_id: Optional[str] = None
    retry_count: int = 0
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    available: bool
    latency_ms: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IdempotencyKey:
    """Idempotency key for operations."""
    operation_type: str
    operation_id: str
    generation_id: str
    timestamp: datetime
    expires_at: Optional[datetime] = None
