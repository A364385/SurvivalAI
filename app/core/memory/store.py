from abc import ABC, abstractmethod
from typing import List, Optional, Any

from app.core.models.memory import MemoryRecord, MemoryType


class MemoryStore(ABC):
    @abstractmethod
    def save(self, record: Any) -> None:
        pass

    @abstractmethod
    def get(self, record_id: str) -> Optional[Any]:
        pass

    @abstractmethod
    def query(self, **kwargs) -> List[Any]:
        pass

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        pass

    @abstractmethod
    def list(self) -> List[Any]:
        pass


class InMemoryStore(MemoryStore):
    """Thread-safe in-memory implementation of the MemoryStore interface."""

    def __init__(self):
        self._store = {}
        import threading
        self._lock = threading.RLock()

    def save(self, record: Any) -> None:
        record_id = (
            getattr(record, "memory_id", None)
            or getattr(record, "experience_id", None)
            or getattr(record, "decision_id", None)
            or getattr(record, "investment_id", None)
            or getattr(record, "generation_id", None)
            or getattr(record, "strategy_id", None)
            or id(record)
        )
        with self._lock:
            self._store[record_id] = record

    def get(self, record_id: str) -> Optional[Any]:
        with self._lock:
            return self._store.get(record_id)

    def query(self, **kwargs) -> List[Any]:
        results = []
        with self._lock:
            for item in self._store.values():
                match = True
                for k, v in kwargs.items():
                    if getattr(item, k, None) != v:
                        match = False
                        break
                if match:
                    results.append(item)
        return results

    def delete(self, record_id: str) -> bool:
        with self._lock:
            if record_id in self._store:
                del self._store[record_id]
                return True
        return False

    def list(self) -> List[Any]:
        with self._lock:
            return list(self._store.values())