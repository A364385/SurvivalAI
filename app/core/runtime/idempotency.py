"""Idempotency management for critical runtime operations.

Every critical operation (order submission, generation creation, learning
cycle, portfolio synchronization) is recorded with a unique operation key
before it executes. A retry after a crash or timeout can never duplicate
the operation because the key is checked first.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from app.core.memory.store import MemoryStore
from app.core.models.memory import MemoryRecord, MemoryType
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class IdempotencyManager:
    """Tracks critical operations so retries never duplicate side effects."""

    OPERATION_ORDER_SUBMISSION = "order_submission"
    OPERATION_GENERATION_CREATION = "generation_creation"
    OPERATION_GENERATION_TRANSITION = "generation_transition"
    OPERATION_LEARNING_CYCLE = "learning_cycle"
    OPERATION_COST_APPLICATION = "cost_application"
    OPERATION_PORTFOLIO_SYNC = "portfolio_sync"

    def __init__(self, memory_store: MemoryStore, default_ttl_hours: int = 24):
        self.memory_store = memory_store
        self.default_ttl_hours = default_ttl_hours

    def _record_id(self, operation_type: str, operation_id: str) -> str:
        return f"idem_{operation_type}_{operation_id}"

    def begin(
        self,
        operation_type: str,
        operation_id: str,
        generation_id: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Claim an operation key. Returns False if the operation was already
        claimed (duplicate attempt) — callers must not re-execute side effects.
        """
        record_id = self._record_id(operation_type, operation_id)
        existing = self.memory_store.get(record_id)
        if existing is not None:
            status = existing.content.get("status")
            if status == "COMPLETED":
                logger.warning(
                    "Duplicate operation blocked: %s/%s (completed)",
                    operation_type,
                    operation_id,
                )
                return False
            if status == "IN_PROGRESS" and not self.is_stale(operation_type, operation_id):
                logger.warning(
                    "Duplicate operation blocked: %s/%s (in progress)",
                    operation_type,
                    operation_id,
                )
                return False
            # FAILED or abandoned IN_PROGRESS operations may be retried.
            logger.info(
                "Retrying operation %s/%s (previous status=%s)",
                operation_type,
                operation_id,
                status,
            )

        record = MemoryRecord(
            memory_id=record_id,
            memory_type=MemoryType.FACT,
            generation_id=generation_id,
            timestamp=now_utc(),
            source_agent="IdempotencyManager",
            importance=8,
            content={
                "operation_type": operation_type,
                "operation_id": operation_id,
                "status": "IN_PROGRESS",
                "started_at": now_utc().isoformat(),
                "metadata": metadata or {},
            },
            metadata={},
        )
        self.memory_store.save(record)
        return True

    def complete(
        self,
        operation_type: str,
        operation_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark an operation as completed with its result summary."""
        record_id = self._record_id(operation_type, operation_id)
        existing = self.memory_store.get(record_id)
        if existing is None:
            logger.warning(
                "Completing unknown operation: %s/%s", operation_type, operation_id
            )
            return
        existing.content["status"] = "COMPLETED"
        existing.content["completed_at"] = now_utc().isoformat()
        existing.content["result"] = result or {}
        self.memory_store.save(existing)

    def fail(
        self,
        operation_type: str,
        operation_id: str,
        error: str,
    ) -> None:
        """Mark an operation as failed so it can be retried safely."""
        record_id = self._record_id(operation_type, operation_id)
        existing = self.memory_store.get(record_id)
        if existing is None:
            return
        existing.content["status"] = "FAILED"
        existing.content["failed_at"] = now_utc().isoformat()
        existing.content["error"] = error
        self.memory_store.save(existing)

    def get_operation(self, operation_type: str, operation_id: str) -> Optional[Dict[str, Any]]:
        """Return the recorded operation content, or None."""
        record = self.memory_store.get(self._record_id(operation_type, operation_id))
        return record.content if record else None

    def is_completed(self, operation_type: str, operation_id: str) -> bool:
        op = self.get_operation(operation_type, operation_id)
        return bool(op and op.get("status") == "COMPLETED")

    def is_in_progress(self, operation_type: str, operation_id: str) -> bool:
        op = self.get_operation(operation_type, operation_id)
        return bool(op and op.get("status") == "IN_PROGRESS")

    def is_stale(self, operation_type: str, operation_id: str) -> bool:
        """An IN_PROGRESS operation older than the TTL is considered abandoned
        (e.g. process crash) and may be retried."""
        op = self.get_operation(operation_type, operation_id)
        if not op or op.get("status") != "IN_PROGRESS":
            return False
        started_at = op.get("started_at")
        if not started_at:
            return True
        try:
            started = datetime.fromisoformat(started_at)
        except (TypeError, ValueError):
            return True
        return now_utc() - started > timedelta(hours=self.default_ttl_hours)