from typing import Callable, Dict, List, Any
from app.core.models.events import BaseEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EventBus:
    """Simple in-memory event bus for publishing and subscribing to events."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[BaseEvent] = []

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event: BaseEvent) -> None:
        """Publish an event to all subscribers."""
        self._event_history.append(event)

        subscribers = self._subscribers.get(event.event_type, [])
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def get_history(self, event_type: str = None) -> List[BaseEvent]:
        """Get event history, optionally filtered by type."""
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type]
        return self._event_history.copy()

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
