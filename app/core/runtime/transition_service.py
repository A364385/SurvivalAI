"""Generation transition service.

When a generation dies, this service coordinates the documented evolution
pipeline:

DEATH -> cancel/reconcile paper orders -> freeze generation -> death report
      -> experience extraction -> learning loop -> strategy evaluation
      -> successor request (SUCCESSOR_PENDING if no strategy passes).

The service never invents a valid strategy: if no proposal passes
validation, the generation stays SUCCESSOR_PENDING and the runtime stops
new investment activity.
"""

from typing import Any, Dict, List, Optional

from app.core.memory.store import MemoryStore
from app.core.generation.manager import GenerationManager
from app.core.models.generation import DeathTrigger
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.runtime.execution_service import PaperExecutionService
from app.core.runtime.idempotency import IdempotencyManager
from app.core.runtime.learning_service import ExperienceCollector, LearningCycle
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class GenerationTransitionService:
    """Coordinates the full generation death -> successor pipeline."""

    def __init__(
        self,
        memory_store: MemoryStore,
        generation_manager: GenerationManager,
        execution_service: PaperExecutionService,
        experience_collector: ExperienceCollector,
        learning_cycle: LearningCycle,
        idempotency_manager: IdempotencyManager,
    ):
        self.memory_store = memory_store
        self.generation_manager = generation_manager
        self.execution_service = execution_service
        self.experience_collector = experience_collector
        self.learning_cycle = learning_cycle
        self.idempotency = idempotency_manager

    def handle_generation_death(
        self,
        generation_id: str,
        trigger: DeathTrigger,
        reason: str,
    ) -> Dict[str, Any]:
        """Run the full death -> transition pipeline. Idempotent per generation."""
        operation_id = f"transition_{generation_id}"

        if not self.idempotency.begin(
            IdempotencyManager.OPERATION_GENERATION_TRANSITION,
            operation_id,
            generation_id=generation_id,
        ):
            return {
                "status": "DUPLICATE_SKIPPED",
                "operation_id": operation_id,
                "previous": self.idempotency.get_operation(
                    IdempotencyManager.OPERATION_GENERATION_TRANSITION, operation_id
                ),
            }

        pipeline: Dict[str, Any] = {
            "generation_id": generation_id,
            "trigger": trigger.value,
            "reason": reason,
            "steps": [],
        }

        try:
            # 1. Stop new investment activity: cancel open paper orders.
            cancelled = self.execution_service.cancel_all_open_orders(
                reason=f"generation death: {trigger.value}"
            )
            pipeline["steps"].append({"step": "CANCEL_OPEN_ORDERS", "result": cancelled})

            # 2. Reconcile portfolio state from the provider.
            account = self.execution_service.get_account_snapshot()
            pipeline["steps"].append({"step": "RECONCILE_PORTFOLIO", "result": account})

            # 3. Freeze the generation (kill + death report via GenerationManager).
            killed = self.generation_manager.kill_generation(
                generation_id, trigger, reason
            )
            pipeline["steps"].append({"step": "FREEZE_GENERATION", "result": killed})

            # 4. Experience extraction from remaining open investments.
            experiences = self.experience_collector.collect_experiences(generation_id)
            pipeline["steps"].append({"step": "EXPERIENCE_EXTRACTION", "result": experiences})

            # 5. Learning loop: strategy analysis and proposals.
            learning = self.learning_cycle.run_learning_cycle(generation_id)
            pipeline["steps"].append({"step": "LEARNING_LOOP", "result": learning})

            # 6. Strategy evaluation: record proposal outcomes. Proposals are
            #    only marked PASSED if the validation pipeline approves them;
            #    here we record the learning outcome without inventing validity.
            proposal_id = learning.get("proposal_id") if isinstance(learning, dict) else None
            if proposal_id:
                self.learning_cycle.record_strategy_proposal_outcome(
                    proposal_id,
                    backtest_passed=False,
                    reason="Awaiting backtest/walk-forward validation pipeline (Step 12).",
                )
                pipeline["steps"].append(
                    {"step": "STRATEGY_EVALUATION", "result": {"proposal_id": proposal_id, "status": "PENDING_VALIDATION"}}
                )
            else:
                pipeline["steps"].append(
                    {"step": "STRATEGY_EVALUATION", "result": {"status": "NO_PROPOSAL"}}
                )

            # 7. Request successor. If no validated strategy exists the
            #    generation remains SUCCESSOR_PENDING (safe state).
            successor_requested = self.generation_manager.request_successor(generation_id)
            pipeline["steps"].append(
                {"step": "SUCCESSOR_REQUEST", "result": successor_requested}
            )
            pipeline["status"] = (
                "SUCCESSOR_PENDING" if successor_requested else "TRANSITION_FAILED"
            )

            self.idempotency.complete(
                IdempotencyManager.OPERATION_GENERATION_TRANSITION,
                operation_id,
                {"status": pipeline["status"]},
            )
            return pipeline

        except Exception as e:
            self.idempotency.fail(
                IdempotencyManager.OPERATION_GENERATION_TRANSITION,
                operation_id,
                str(e),
            )
            logger.error("Generation transition failed: %s", e)
            pipeline["status"] = "FAILED"
            pipeline["error"] = str(e)
            return pipeline

    def create_successor_generation(
        self,
        parent_generation_id: str,
        strategy_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a successor generation after validation approved a strategy.

        Idempotent: duplicate calls for the same parent return the existing
        successor instead of creating a second one.
        """
        parent = self.generation_manager.get_generation(parent_generation_id)
        if parent is None:
            return None

        if parent.lifecycle_state.value not in ("SUCCESSOR_PENDING", "SUCCESSOR_CREATED"):
            logger.warning(
                "Parent %s is not SUCCESSOR_PENDING (state=%s); refusing successor creation.",
                parent_generation_id,
                parent.lifecycle_state.value,
            )
            return None

        # Idempotency: only one successor per parent.
        operation_id = f"successor_{parent_generation_id}"
        if not self.idempotency.begin(
            IdempotencyManager.OPERATION_GENERATION_CREATION,
            operation_id,
            generation_id=parent_generation_id,
        ):
            existing = self.idempotency.get_operation(
                IdempotencyManager.OPERATION_GENERATION_CREATION, operation_id
            )
            # Return the recorded successor result (nested under "result").
            if existing and isinstance(existing.get("result"), dict):
                return existing["result"]
            return existing

        try:
            successor = self.generation_manager.create_generation(
                parent_generation_id=parent_generation_id,
                strategy_version=strategy_version,
            )
            started = self.generation_manager.start_generation(successor.generation_id)

            result = {
                "parent_generation_id": parent_generation_id,
                "successor_generation_id": successor.generation_id,
                "generation_number": successor.generation_number,
                "started": bool(started),
            }
            self.idempotency.complete(
                IdempotencyManager.OPERATION_GENERATION_CREATION,
                operation_id,
                result,
            )
            return result

        except Exception as e:
            self.idempotency.fail(
                IdempotencyManager.OPERATION_GENERATION_CREATION,
                operation_id,
                str(e),
            )
            logger.error("Successor creation failed: %s", e)
            return None