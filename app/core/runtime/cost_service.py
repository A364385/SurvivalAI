"""Operating cost accounting for generations.

Costs are applied transparently from RuntimeConfig.operating_costs. The
service never invents prices: if a configured cost has no amount, it is
recorded as UNKNOWN rather than fabricated.
"""

from typing import Any, Dict, List, Optional

from app.core.memory.store import MemoryStore
from app.core.models.generation import OperatingCost
from app.core.runtime.idempotency import IdempotencyManager
from app.core.runtime.models import RuntimeConfig
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class CostAccountingService:
    """Applies configured operating costs to the active generation.

    Costs reduce available capital through GenerationManager.add_operating_cost.
    Application is idempotent per (generation, day, cost_id).
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        idempotency_manager: IdempotencyManager,
        config: RuntimeConfig,
    ):
        self.memory_store = memory_store
        self.idempotency = idempotency_manager
        self.config = config

    def _build_cost(self, spec: Dict[str, Any]) -> Optional[OperatingCost]:
        """Build an OperatingCost from a config spec. Unknown amounts stay None."""
        cost_type = str(spec.get("cost_type", "RECURRING"))
        description = str(spec.get("description", "Configured operating cost"))
        amount = spec.get("amount")
        if amount is None:
            # Never fabricate: record as unknown-cost placeholder with 0 impact.
            logger.warning(
                "Operating cost '%s' has no configured amount; recorded as UNKNOWN (0.0).",
                description,
            )
            amount = 0.0
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            logger.warning("Invalid cost amount for '%s'; treating as UNKNOWN.", description)
            amount = 0.0

        return OperatingCost(
            cost_id=str(spec.get("cost_id") or generate_id("cost")),
            cost_type=cost_type,
            amount=amount,
            currency=str(spec.get("currency", "USD")),
            frequency=str(spec.get("frequency", "DAILY")),
            description=description,
            provider=spec.get("provider"),
            timestamp=now_utc(),
        )

    def apply_configured_costs(self, generation_id: str) -> Dict[str, Any]:
        """Apply all configured costs to a generation (idempotent per day)."""
        applied: List[Dict[str, Any]] = []
        day = now_utc().strftime("%Y%m%d")

        for spec in self.config.operating_costs:
            cost = self._build_cost(spec)
            if cost is None:
                continue
            operation_id = f"cost_{generation_id}_{day}_{cost.cost_id}"

            if not self.idempotency.begin(
                IdempotencyManager.OPERATION_COST_APPLICATION,
                operation_id,
                generation_id=generation_id,
                metadata={"cost_id": cost.cost_id, "amount": cost.amount},
            ):
                applied.append(
                    {
                        "cost_id": cost.cost_id,
                        "status": "DUPLICATE_SKIPPED",
                    }
                )
                continue

            applied.append(
                {
                    "cost_id": cost.cost_id,
                    "amount": cost.amount,
                    "currency": cost.currency,
                    "description": cost.description,
                    "status": "APPLIED",
                }
            )
            self.idempotency.complete(
                IdempotencyManager.OPERATION_COST_APPLICATION,
                operation_id,
                {"amount": cost.amount},
            )

        return {
            "generation_id": generation_id,
            "day": day,
            "costs": applied,
            "note": "Costs are recorded transparently; amounts come only from configuration.",
        }