"""SQLite-backed MemoryStore.

A durable, thread-safe implementation of the MemoryStore interface and a
drop-in replacement for InMemoryStore: `query(**kwargs)` matches object
attributes exactly like the in-memory version, and records are serialized as
JSON so every dataclass record type (MemoryRecord, Experience, decisions,
generation state, ...) survives process restarts.

Design:
- One JSON row per record, keyed by its primary id (memory_id, decision_id, ...)
- WAL journal mode for safe concurrent readers/writers on Windows
- stdlib only (sqlite3)
- Enum members nested anywhere in record content are revived from their
  "ClassName:VALUE" markers (never silently dropped or mangled)
"""

import json
import sqlite3
import threading
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.memory.store import MemoryStore
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id   TEXT PRIMARY KEY,
    record_type TEXT NOT NULL,
    json        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_type ON records(record_type);
"""

# Enum classes that may appear as "ClassName:VALUE" markers in record content.
_KNOWN_ENUM_NAMES = {
    "MemoryType", "GenerationStatus", "StrategyStatus", "InvestmentStatus",
    "GenerationLifecycleState", "DeathTrigger", "AgentStatus",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return {"__dt__": obj.isoformat()}
    if isinstance(obj, date):
        return {"__date__": obj.isoformat()}
    if isinstance(obj, Enum):
        return {"__enum__": f"{type(obj).__name__}:{obj.value}"}
    if isinstance(obj, set):
        return {"__set__": sorted(obj)}
    if is_dataclass(obj) and not isinstance(obj, type):
        # Nested dataclasses (e.g. OperatingCost inside generation state) must
        # survive as structured data, not be flattened to an opaque str().
        return {
            "__dataclass__": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "fields": {f.name: getattr(obj, f.name) for f in fields(obj)},
        }
    return str(obj)


def _import_dotted(path: str) -> Optional[type]:
    """Resolve 'package.module.ClassName' to the class, or None."""
    if not isinstance(path, str) or "." not in path:
        return None
    module_path, _, class_name = path.rpartition(".")
    try:
        module = __import__(module_path, fromlist=[class_name])
    except ImportError:
        return None
    candidate = getattr(module, class_name, None)
    return candidate if isinstance(candidate, type) else None


def _to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    return dict(getattr(obj, "__dict__", {}))


def _revive_enum_marker(value: str) -> Any:
    """Revive a 'ClassName:VALUE' marker to a real Enum member, or return
    the original string when the enum type cannot be resolved."""
    if not isinstance(value, str) or ":" not in value:
        return value
    prefix, _, raw = value.partition(":")
    if prefix not in _KNOWN_ENUM_NAMES:
        return value
    for module_name in ("app.core.models.memory", "app.core.models.generation"):
        try:
            module = __import__(module_name, fromlist=["*"])
        except ImportError:
            continue
        enum_cls = getattr(module, prefix, None)
        if enum_cls is None or not isinstance(enum_cls, type) or not issubclass(enum_cls, Enum):
            continue
        try:
            return enum_cls(raw)
        except ValueError:
            continue
    return value


def _revive_container(data: Any, depth: int = 0) -> Any:
    """Recursively revive datetime/enum/set markers inside structures."""
    if depth > 12:
        return data
    if isinstance(data, dict):
        keys = set(data.keys())
        if keys == {"__dt__"}:
            try:
                return datetime.fromisoformat(data["__dt__"])
            except (TypeError, ValueError):
                return data["__dt__"]
        if keys == {"__date__"}:
            try:
                return date.fromisoformat(data["__date__"])
            except (TypeError, ValueError):
                return data["__date__"]
        if keys == {"__enum__"} and isinstance(data["__enum__"], str):
            return _revive_enum_marker(data["__enum__"])
        if keys == {"__set__"}:
            return set(data["__set__"])
        if keys == {"__dataclass__", "fields"}:
            cls = _import_dotted(data["__dataclass__"])
            if cls is None or not is_dataclass(cls):
                return data
            return cls(**{
                k: _revive_container(v, depth + 1)
                for k, v in data["fields"].items()
            })
        return {k: _revive_container(v, depth + 1) for k, v in data.items()}
    if isinstance(data, list):
        return [_revive_container(v, depth + 1) for v in data]
    if isinstance(data, str) and ":" in data:
        # Handle bare enum markers stored as plain strings.
        prefix = data.partition(":")[0]
        if prefix in _KNOWN_ENUM_NAMES:
            return _revive_enum_marker(data)
    return data


def _enum_for_field(cls: type, field_name: str) -> Optional[type]:
    try:
        for f in fields(cls):
            if f.name == field_name:
                ann = f.type
                if isinstance(ann, str):
                    module = __import__(cls.__module__, fromlist=["*"])
                    enum_cls = getattr(module, ann.split(".")[-1], None)
                    if enum_cls is not None and isinstance(enum_cls, type) and issubclass(enum_cls, Enum):
                        return enum_cls
                elif isinstance(ann, type) and issubclass(ann, Enum):
                    return ann
    except Exception:
        return None
    return None


class SqliteMemoryStore(MemoryStore):
    """Durable MemoryStore backed by a local SQLite database file."""

    def __init__(self, db_path: str = "data/survivalai.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("SQLite memory store initialized at %s", self.db_path)

    _KNOWN_CLASSES = {
        "MemoryRecord": "app.core.models.memory",
        "GenerationRecord": "app.core.models.memory",
        "InvestmentRecord": "app.core.models.memory",
        "StrategyVersion": "app.core.models.memory",
        "DecisionRecord": "app.core.models.memory",
        "DeathReport": "app.core.models.memory",
        "Experience": "app.core.models.memory",
        "AgentPerformanceRecord": "app.core.models.memory",
    }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @staticmethod
    def _record_id(record: Any) -> Any:
        return (
            getattr(record, "memory_id", None)
            or getattr(record, "experience_id", None)
            or getattr(record, "decision_id", None)
            or getattr(record, "investment_id", None)
            or getattr(record, "generation_id", None)
            or getattr(record, "strategy_id", None)
            or id(record)
        )

    @staticmethod
    def _record_type(record: Any) -> str:
        return type(record).__name__

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def save(self, record: Any) -> None:
        record_id = str(self._record_id(record))
        record_type = self._record_type(record)
        payload = json.dumps(_to_dict(record), default=_json_default)
        with self._lock:
            self._conn.execute(
                "INSERT INTO records (record_id, record_type, json, created_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(record_id) DO UPDATE SET json=excluded.json",
                (record_id, record_type, payload, now_utc().isoformat()),
            )
            self._conn.commit()

    def _resolve_class(self, name: str) -> Optional[type]:
        module_path = self._KNOWN_CLASSES.get(name)
        if not module_path:
            return None
        try:
            module = __import__(module_path, fromlist=["*"])
            return getattr(module, name, None)
        except ImportError:
            return None

    def _revive(self, record_type: str, data: Dict[str, Any]) -> Any:
        """Rebuild a record object of `record_type` from its dict."""
        revived = _revive_container(data)
        cls = self._resolve_class(record_type)
        if cls is None:
            return revived
        try:
            # Enum-typed dataclass fields: revive "ClassName:VALUE" strings.
            for field in fields(cls):
                value = revived.get(field.name)
                if isinstance(value, str) and ":" in value:
                    enum_cls = _enum_for_field(cls, field.name)
                    if enum_cls is not None:
                        prefix, _, raw = value.partition(":")
                        if prefix == enum_cls.__name__:
                            try:
                                revived[field.name] = enum_cls(raw)
                            except ValueError:
                                pass
            return cls(**revived)
        except TypeError:
            # Be liberal: if construction fails, return the dict (no data loss).
            return revived

    def get(self, record_id: str) -> Optional[Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT record_type, json FROM records WHERE record_id = ?",
                (str(record_id),),
            ).fetchone()
        if not row:
            return None
        try:
            return self._revive(row[0], json.loads(row[1]))
        except Exception as e:
            logger.error("Failed to revive record %s: %s", record_id, e)
            return None

    def query(self, **kwargs) -> List[Any]:
        results: List[Any] = []
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_type, json FROM records ORDER BY created_at ASC"
            ).fetchall()
        for record_type, raw in rows:
            try:
                record = self._revive(record_type, json.loads(raw))
            except Exception as e:
                logger.error("Failed to revive record: %s", e)
                continue
            match = True
            for k, v in kwargs.items():
                if getattr(record, k, None) != v:
                    match = False
                    break
            if match:
                results.append(record)
        return results

    def delete(self, record_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM records WHERE record_id = ?", (str(record_id),)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def list(self) -> List[Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_type, json FROM records ORDER BY created_at ASC"
            ).fetchall()
        results = []
        for record_type, raw in rows:
            try:
                results.append(self._revive(record_type, json.loads(raw)))
            except Exception as e:
                logger.error("Failed to revive record: %s", e)
        return results

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM records").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass

    def integrity_check(self) -> bool:
        try:
            with self._lock:
                row = self._conn.execute("PRAGMA integrity_check").fetchone()
            return bool(row) and str(row[0]).lower() == "ok"
        except sqlite3.DatabaseError:
            return False
