from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


class ProposalStatus(Enum):
    PROPOSED = "PROPOSED"
    BACKTESTING = "BACKTESTING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    APPROVED_FOR_SIMULATION = "APPROVED_FOR_SIMULATION"
    ACTIVE = "ACTIVE"


@dataclass
class StrategyChangeProposal:
    """Proposal for changing the investment strategy."""
    proposal_id: str
    parent_strategy_id: str
    proposed_parameters: Dict[str, Any]
    changed_rules: Dict[str, Any]
    unchanged_rules: Dict[str, Any]
    motivation: str
    supporting_experiences: List[str]
    supporting_decisions: List[str]
    supporting_outcomes: List[str]
    expected_effect: str
    risks: List[str]
    assumptions: List[str]
    confidence: float
    backtest_required: bool
    status: ProposalStatus
    created_at: datetime
    generation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results of a strategy backtest."""
    backtest_id: str
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: Optional[float]
    maximum_drawdown: float
    volatility: Optional[float]
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    win_rate: float
    number_of_trades: int
    average_trade_return: Optional[float]
    largest_gain: float
    largest_loss: float
    transaction_costs: float
    slippage_costs: float
    exposure_statistics: Dict[str, Any]
    risk_violations: List[Dict[str, Any]]
    benchmark_comparison: Optional[Dict[str, Any]]
    market_regime_results: Dict[str, Any]
    crisis_period_results: Dict[str, Any]
    warnings: List[str]
    data_quality: Dict[str, Any]
    assumptions: List[str]
    timestamp: datetime


@dataclass
class StrategyComparison:
    """Comparison between two strategies."""
    strategy_a_id: str
    strategy_b_id: str
    comparison_date: datetime
    total_return_difference: float
    drawdown_difference: float
    risk_adjusted_return_difference: Optional[float]
    volatility_difference: Optional[float]
    survival_characteristics: Dict[str, Any]
    transaction_cost_difference: float
    crisis_performance_difference: Dict[str, Any]
    regime_performance_difference: Dict[str, Any]
    number_of_trades_difference: int
    concentration_difference: Dict[str, Any]
    consistency_metrics: Dict[str, Any]
    recommended_strategy: Optional[str]
    confidence: float
    reasoning: str


@dataclass
class SurvivalFitness:
    """Survival-aware fitness evaluation of a strategy."""
    strategy_id: str
    evaluation_date: datetime
    capital_preservation_score: float
    probability_of_ruin: float
    maximum_drawdown_score: float
    return_score: float
    risk_adjusted_return_score: float
    cost_score: float
    stability_score: float
    diversification_score: float
    overall_fitness: float
    survival_weight: float
    growth_weight: float
    risk_control_weight: float
    never_invested_penalty: float
    metrics: Dict[str, Any]
