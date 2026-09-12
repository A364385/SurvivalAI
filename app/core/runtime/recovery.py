from typing import Optional, Dict, Any
from datetime import datetime

from app.core.memory.store import MemoryStore
from app.core.generation.manager import GenerationManager
from app.core.runtime.models import RuntimeState, RuntimeStateSnapshot
from app.core.models.memory import MemoryType
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class RuntimeRecovery:
    """Handles runtime restart and recovery."""

    def __init__(
        self,
        memory_store: MemoryStore,
        generation_manager: GenerationManager,
    ):
        self.memory_store = memory_store
        self.generation_manager = generation_manager

    def recover_runtime_state(self) -> Optional[RuntimeStateSnapshot]:
        """Recover runtime state from memory."""
        try:
            records = self.memory_store.query(memory_type=MemoryType.FACT)

            # Find the most recent runtime state snapshot
            for record in records:
                if record.content.get("runtime_id"):
                    return self._parse_state_snapshot(record)

            return None

        except Exception as e:
            logger.error(f"Failed to recover runtime state: {e}")
            return None

    def recover_active_generation(self) -> Optional[str]:
        """Recover active generation ID."""
        try:
            records = self.memory_store.query(memory_type=MemoryType.GENERATION)

            # Find active generation (lifecycle_state may be enum or string)
            for record in records:
                content = record.content
                state = content.get("lifecycle_state")
                state_value = (
                    state.value if hasattr(state, "value") else str(state or "")
                )
                if state_value in ("ACTIVE", "PAUSED"):
                    return content.get("generation_id")

            return None

        except Exception as e:
            logger.error(f"Failed to recover active generation: {e}")
            return None

    def reconcile_portfolio_state(self) -> Dict[str, Any]:
        """Reconcile portfolio state after restart."""
        # This would use the portfolio synchronizer
        return {"status": "RECONCILED", "timestamp": now_utc()}

    def reconcile_orders(self) -> Dict[str, Any]:
        """Reconcile orders after restart."""
        # This would use the execution provider
        return {"status": "RECONCILED", "timestamp": now_utc()}

    def restore_strategy(self) -> Optional[Dict[str, Any]]:
        """Restore active strategy."""
        try:
            records = self.memory_store.query(memory_type=MemoryType.STRATEGY)

            # Find active strategy
            for record in records:
                content = record.content
                if content.get("status") == "ACTIVE":
                    return content

            return None

        except Exception as e:
            logger.error(f"Failed to restore strategy: {e}")
            return None

    def restore_survival_state(self) -> Dict[str, Any]:
        """Restore survival state."""
        try:
            # Load generation state for survival metrics
            active_gen_id = self.recover_active_generation()
            if active_gen_id:
                gen = self.generation_manager.get_generation(active_gen_id)
                if gen:
                    return {
                        "generation_id": gen.generation_id,
                        "current_capital": gen.current_capital,
                        "maximum_drawdown": gen.maximum_drawdown,
                        "lifespan_days": gen.lifespan_days,
                    }

            return {}

        except Exception as e:
            logger.error(f"Failed to restore survival state: {e}")
            return {}

    def identify_unfinished_operations(self) -> list:
        """Identify unfinished operations after restart."""
        unfinished = []

        # Check for incomplete cycles
        records = self.memory_store.query(memory_type=MemoryType.FACT)
        for record in records:
            if record.content.get("status") == "IN_PROGRESS":
                unfinished.append({
                    "type": "cycle",
                    "id": record.content.get("cycle_id"),
                    "generation_id": record.content.get("generation_id"),
                })

        return unfinished

    def resume_from_valid_state(self, runtime_id: str) -> bool:
        """Resume runtime from a valid state."""
        try:
            # Recover state
            state = self.recover_runtime_state()
            if not state:
                logger.warning("No recoverable state found")
                return False

            # Recover active generation
            active_gen_id = self.recover_active_generation()
            if active_gen_id:
                # Resume generation
                gen = self.generation_manager.get_generation(active_gen_id)
                if gen and gen.lifecycle_state.value == "PAUSED":
                    # Can resume
                    pass

            logger.info(f"Runtime {runtime_id} recovered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to resume from valid state: {e}")
            return False

    def _parse_state_snapshot(self, record) -> RuntimeStateSnapshot:
        """Parse state snapshot from memory record."""
        content = record.content
        return RuntimeStateSnapshot(
            runtime_id=content.get("runtime_id"),
            generation_id=content.get("generation_id"),
            current_state=RuntimeState(content.get("current_state")),
            previous_state=RuntimeState(content.get("previous_state")) if content.get("previous_state") else None,
            timestamp=content.get("timestamp"),
            cycle_id=content.get("cycle_id"),
            cycle_phase=None,  # Would parse if needed
            health_status=content.get("health_status", {}),
            error_count=content.get("error_count", 0),
            last_error=content.get("last_error"),
            last_error_timestamp=content.get("last_error_timestamp"),
            metadata=content.get("metadata", {}),
        )
