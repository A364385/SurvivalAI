"""Local model registry: role-specialized model versions with lifecycle states.

States: DISCOVERED, DOWNLOADING, READY, TRAINING, EVALUATING, VALIDATED,
ACTIVE, FAILED, ARCHIVED.

Rules enforced here:
- A model is NEVER auto-activated; activation is explicit.
- Only VALIDATED models can be activated.
- A role has at most one ACTIVE model at a time.
- Every state change is appended to an immutable audit history.

Storage is a local SQLite database (stdlib only) under the data directory.
"""

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class ModelStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    DOWNLOADING = "DOWNLOADING"
    READY = "READY"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


@dataclass
class ModelRecord:
    model_id: str
    role: str
    base_model: str
    model_version: str
    adapter_version: str = "v0"
    training_dataset: Optional[str] = None
    training_date: Optional[str] = None
    training_config: Dict[str, Any] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    quantization: str = "none"
    context_length: int = 2048
    parameter_count: Optional[str] = None
    vram_estimate_gb: float = 0.0
    artifact_path: Optional[str] = None
    provider: str = "transformers"
    status: str = ModelStatus.DISCOVERED.value
    notes: str = ""
    created_at: str = field(default_factory=lambda: now_utc().isoformat())
    updated_at: str = field(default_factory=lambda: now_utc().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    role     TEXT NOT NULL,
    status   TEXT NOT NULL,
    json     TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_models_role ON models(role);
CREATE TABLE IF NOT EXISTS model_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    event    TEXT NOT NULL,
    details  TEXT,
    ts       TEXT NOT NULL
);
"""


class ModelRegistry:
    """SQLite-backed registry with lifecycle enforcement and audit history."""

    def __init__(self, db_path: str = "data/models_registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    def _log_event(self, model_id: str, event: str, details: Any = None) -> None:
        self._conn.execute(
            "INSERT INTO model_events (model_id, event, details, ts) VALUES (?, ?, ?, ?)",
            (model_id, event,
             json.dumps(details, default=str) if details is not None else None,
             now_utc().isoformat()),
        )

    def register(self, record: ModelRecord) -> ModelRecord:
        with self._lock:
            existing = self.get(record.model_id)
            if existing is not None:
                raise ValueError(f"Model already registered: {record.model_id}")
            self._upsert(record)
            self._log_event(record.model_id, "REGISTERED", {"status": record.status})
            self._conn.commit()
        logger.info("Registered model %s (role=%s, status=%s)",
                    record.model_id, record.role, record.status)
        return record

    def _upsert(self, record: ModelRecord) -> None:
        record.updated_at = now_utc().isoformat()
        self._conn.execute(
            "INSERT INTO models (model_id, role, status, json, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(model_id) DO UPDATE SET status=excluded.status, "
            "json=excluded.json, updated_at=excluded.updated_at",
            (record.model_id, record.role, record.status,
             json.dumps(record.to_dict(), default=str), record.updated_at),
        )

    def get(self, model_id: str) -> Optional[ModelRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT json FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        return ModelRecord(**data)

    def list_models(self, role: Optional[str] = None) -> List[ModelRecord]:
        with self._lock:
            if role:
                rows = self._conn.execute(
                    "SELECT json FROM models WHERE role = ? ORDER BY updated_at DESC",
                    (role,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT json FROM models ORDER BY role, updated_at DESC"
                ).fetchall()
        return [ModelRecord(**json.loads(r[0])) for r in rows]

    def get_active_model(self, role: str) -> Optional[ModelRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT json FROM models WHERE role = ? AND status = 'ACTIVE'",
                (role,),
            ).fetchall()
        return ModelRecord(**json.loads(rows[0][0])) if rows else None

    def update_status(self, model_id: str, status: ModelStatus,
                      notes: str = "") -> Optional[ModelRecord]:
        with self._lock:
            record = self.get(model_id)
            if record is None:
                return None
            # Activation rules
            if status == ModelStatus.ACTIVE:
                if record.status != ModelStatus.VALIDATED.value:
                    raise ValueError(
                        f"Model {model_id} cannot be activated from state "
                        f"{record.status}; only VALIDATED models may activate"
                    )
                # Deactivate any other ACTIVE model for this role.
                for other in self.list_models(role=record.role):
                    if other.model_id != model_id and other.status == ModelStatus.ACTIVE.value:
                        other.status = ModelStatus.ARCHIVED.value
                        self._upsert(other)
                        self._log_event(other.model_id, "ARCHIVED",
                                        {"reason": f"superseded by {model_id}"})
            record.status = status.value if isinstance(status, ModelStatus) else str(status)
            if notes:
                record.notes = notes
            self._upsert(record)
            self._log_event(model_id, f"STATUS->{record.status}", {"notes": notes})
            self._conn.commit()
        return record

    def update_fields(self, model_id: str, **fields: Any) -> Optional[ModelRecord]:
        with self._lock:
            record = self.get(model_id)
            if record is None:
                return None
            for key, value in fields.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            self._upsert(record)
            self._log_event(model_id, "UPDATED", fields)
            self._conn.commit()
        return record

    def history(self, model_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT event, details, ts FROM model_events WHERE model_id = ? ORDER BY id ASC",
                (model_id,),
            ).fetchall()
        return [{"event": r[0], "details": json.loads(r[1]) if r[1] else None, "ts": r[2]}
                for r in rows]

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass
