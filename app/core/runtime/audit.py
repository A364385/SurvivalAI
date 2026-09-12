"""Structured audit trail.

Every consequential action is recorded so the question
"Why did SurvivalAI make this decision?" is always answerable:

    timestamp | generation | component | task | input refs | output |
    confidence | sources | decision | risk result | capital protection
    result | execution result | outcome

Entries are stored through the MemoryStore (SQLite-backed in production) with
memory_type=FACT and the `audit` marker, making them queryable and immutable
in practice (records are never deleted by the runtime).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.core.models.memory import MemoryRecord, MemoryType
from app.core.memory.store import MemoryStore
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


@dataclass
class AuditEntry:
    timestamp: str
    generation_id: str
    component: str
    task: str
    decision: Optional[str] = None
    confidence: Optional[float] = None
    input_refs: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    risk_result: Optional[str] = None
    capital_protection_result: Optional[str] = None
    execution_result: Optional[str] = None
    outcome: Optional[str] = None
    latency_ms: Optional[float] = None
    audit_id: str = field(default_factory=lambda: generate_id("audit"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditTrail:
    """Append-only audit trail backed by the MemoryStore."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def record(self, entry: AuditEntry) -> str:
        record = MemoryRecord(
            memory_id=f"audit_{entry.audit_id}",
            memory_type=MemoryType.FACT,
            generation_id=entry.generation_id or "unknown",
            timestamp=now_utc(),
            source_agent=f"AuditTrail:{entry.component}",
            importance=8,
            content=entry.to_dict(),
            metadata={"audit": True, "component": entry.component},
        )
        try:
            self.memory_store.save(record)
        except Exception as e:
            # Audit failures must never break the pipeline, but must be loud.
            logger.error("AUDIT WRITE FAILED: %s", e)
        return entry.audit_id

    def record_decision(
        self,
        generation_id: str,
        decision_id: str,
        decision_type: str,
        confidence: float,
        risk_result: str,
        capital_protection_result: str,
        execution_result: Optional[str],
        sources: Optional[List[str]] = None,
        output: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.record(AuditEntry(
            timestamp=now_utc().isoformat(),
            generation_id=generation_id,
            component="decision_pipeline",
            task=f"decision:{decision_id}",
            decision=decision_type,
            confidence=confidence,
            sources=sources or [],
            output=output or {},
            risk_result=risk_result,
            capital_protection_result=capital_protection_result,
            execution_result=execution_result,
        ))

    def record_event(self, generation_id: str, component: str, task: str,
                     outcome: str, output: Optional[Dict[str, Any]] = None) -> str:
        return self.record(AuditEntry(
            timestamp=now_utc().isoformat(),
            generation_id=generation_id,
            component=component,
            task=task,
            outcome=outcome,
            output=output or {},
        ))

    def query_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        records = self.memory_store.query(memory_type=MemoryType.FACT)
        entries = []
        for record in records:
            content = getattr(record, "content", {})
            if isinstance(content, dict) and content.get("audit_id"):
                entries.append(content)
        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return entries[:limit]
