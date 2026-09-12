from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

from app.core.models.strategy import StrategyComparison, SurvivalFitness
from app.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyComparator:
    """Compares two strategies and determines which is superior."""

    def __init__(
        self,
        survival_weight: float = 0.5,
        growth_weight: float = 0.3,
        risk_control_weight: float = 0.2,
    ):
        self.survival_weight = survival_weight
        self.growth_weight = growth_weight
        self.risk_control_weight = risk_control_weight

    def compare_strategies(
        self,
        strategy_a: Dict[str, Any],
        strategy_b: Dict[str, Any],
        comparison_date: datetime,
    ) -> StrategyComparison:
        """Compare two strategies based on their backtest results."""
        # Extract metrics
        return_a = strategy_a.get("total_return", 0)
        return_b = strategy_b.get("total_return", 0)
        drawdown_a = strategy_a.get("maximum_drawdown", 0)
        drawdown_b = strategy_b.get("maximum_drawdown", 0)
        trades_a = strategy_a.get("number_of_trades", 0)
        trades_b = strategy_b.get("number_of_trades", 0)

        # Calculate differences
        return_diff = return_b - return_a
        drawdown_diff = drawdown_a - drawdown_b  # Lower drawdown is better
        trades_diff = trades_b - trades_a

        # Calculate survival fitness
        fitness_a = self._calculate_survival_fitness(strategy_a)
        fitness_b = self._calculate_survival_fitness(strategy_b)

        # Determine recommendation
        recommended = None
        if fitness_b.overall_fitness > fitness_a.overall_fitness:
            recommended = strategy_b.get("strategy_id")
        elif fitness_a.overall_fitness > fitness_b.overall_fitness:
            recommended = strategy_a.get("strategy_id")

        comparison = StrategyComparison(
            strategy_a_id=strategy_a.get("strategy_id", "unknown"),
            strategy_b_id=strategy_b.get("strategy_id", "unknown"),
            comparison_date=comparison_date,
            total_return_difference=return_diff,
            drawdown_difference=drawdown_diff,
            risk_adjusted_return_difference=None,  # Would need more data
            volatility_difference=None,
            survival_characteristics={
                "fitness_a": fitness_a.overall_fitness,
                "fitness_b": fitness_b.overall_fitness,
            },
            transaction_cost_difference=0,  # Would need transaction data
            crisis_performance_difference={},
            regime_performance_difference={},
            number_of_trades_difference=trades_diff,
            concentration_difference={},
            consistency_metrics={},
            recommended_strategy=recommended,
            confidence=abs(fitness_b.overall_fitness - fitness_a.overall_fitness),
            reasoning=f"Strategy comparison based on return ({return_diff:.2%}), drawdown ({drawdown_diff:.2%}), and survival fitness",
        )

        return comparison

    def _calculate_survival_fitness(self, strategy_result: Dict[str, Any]) -> SurvivalFitness:
        """Calculate survival-aware fitness for a strategy."""
        total_return = strategy_result.get("total_return", 0)
        max_drawdown = strategy_result.get("maximum_drawdown", 0)
        final_capital = strategy_result.get("final_capital", 0)
        initial_capital = strategy_result.get("initial_capital", 10000)

        # Calculate individual scores
        capital_preservation_score = min(1.0, final_capital / initial_capital)
        max_drawdown_score = max(0.0, 1.0 - max_drawdown)
        return_score = min(1.0, max(0.0, total_return))

        # Never-invested penalty
        trades = strategy_result.get("number_of_trades", 0)
        never_invested_penalty = 0.0 if trades > 0 else 0.5

        # Calculate overall fitness with weights
        overall_fitness = (
            (self.survival_weight * capital_preservation_score)
            + (self.survival_weight * max_drawdown_score)
            + (self.growth_weight * return_score)
            - never_invested_penalty
        )

        fitness = SurvivalFitness(
            strategy_id=strategy_result.get("strategy_id", "unknown"),
            evaluation_date=datetime.now(),
            capital_preservation_score=capital_preservation_score,
            probability_of_ruin=max_drawdown,
            maximum_drawdown_score=max_drawdown_score,
            return_score=return_score,
            risk_adjusted_return_score=return_score / (max_drawdown + 0.01),
            cost_score=0.9,  # Placeholder
            stability_score=0.8,  # Placeholder
            diversification_score=0.8,  # Placeholder
            overall_fitness=max(0.0, overall_fitness),
            survival_weight=self.survival_weight,
            growth_weight=self.growth_weight,
            risk_control_weight=self.risk_control_weight,
            never_invested_penalty=never_invested_penalty,
            metrics=strategy_result,
        )

        return fitness
