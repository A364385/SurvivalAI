from typing import Dict, List, Optional
from datetime import datetime
from app.core.models.generation import OperatingCost
from app.core.models.memory import MemoryRecord
from app.core.memory.store import MemoryStore
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.core.models.events import CostUpdateEvent

logger = get_logger(__name__)

class CostAccountingService:
    """Handles tracking and subtraction of operating costs for generations."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def record_cost(self, generation_id: str, cost: OperatingCost) -> bool:
        """Record a new cost and update the generation's total operating costs and capital."""
        generation = self.memory_store.get_generation(generation_id)
        if not generation:
            return False

        # Update total costs
        generation.total_operating_costs += cost.amount
        
        # Update current capital
        generation.current_capital -= cost.amount

        # Persist updated state
        from app.core.models.generation import GenerationState
        self.memory_store.save(generation)

        # Emit event
        self.memory_store.publish(
            CostUpdateEvent(
                event_id=generate_id("evt_cost"),
                timestamp=datetime.now(),
                event_type="CostUpdateEvent",
                generation_id=generation_id,
                cost_id=cost.cost_id,
                cost_type=cost.cost_type,
                amount=cost.amount,
                currency=cost.currency,
                provider=cost.provider,
                description=cost.description
            )
        )

        logger.info(f"Recorded cost {cost.cost_type} ({cost.amount} {cost.currency}) for gen {generation_id}")
        return True

    def get_cost_summary(self, generation_id: str) -> Dict[str, Any]:
        """Retrieve a summary of costs for a generation."""
        generation = self.memory_store.get_generation(generation_id)
        if not generation:
            return {}

        return {
            "total_operating_costs": generation.total_operating_costs,
            "costs": generation.operating_costs,
            "current_capital": generation.current_capital,
            "starting_capital": generation.starting_capital
        }
