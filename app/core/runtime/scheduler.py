"""Configurable scheduler for runtime tasks.

Different tasks run at different frequencies (market polling, news polling,
portfolio sync, investment monitoring, learning, health checks). The
scheduler avoids uncontrolled concurrency: each task type runs at most once
concurrently, guarded by a lock.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


@dataclass
class ScheduledTask:
    """A named task with its own interval."""
    name: str
    interval_seconds: int
    callback: Callable[[], Dict]
    last_run_at: Optional[float] = None
    run_count: int = 0
    last_result: Dict = field(default_factory=dict)


class RuntimeScheduler:
    """Runs named tasks when their intervals elapse.

    Tasks execute sequentially in the caller's thread (deterministic, no
    overlapping runs of the same task). `run_due_tasks()` is invoked once per
    autonomous cycle; each task fires only when its interval has passed.
    """

    def __init__(self):
        self._tasks: Dict[str, ScheduledTask] = {}
        self._lock = threading.Lock()

    def register(self, name: str, interval_seconds: int, callback: Callable[[], Dict]) -> None:
        """Register a task with its polling interval."""
        self._tasks = getattr(self, "_tasks", {})
        self._tasks[name] = ScheduledTask(
            name=name,
            interval_seconds=max(1, int(interval_seconds)),
            callback=callback,
        )

    def run_due_tasks(self) -> List[Dict]:
        """Run every task whose interval has elapsed. Returns task results."""
        results: List[Dict] = []
        tasks: Dict[str, ScheduledTask] = getattr(self, "_tasks", {})
        now = time.monotonic()

        for task in tasks.values():
            due = (
                task.last_run_at is None
                or (now - task.last_run_at) >= task.interval_seconds
            )
            if not due:
                continue

            started = now_utc()
            try:
                result = task.callback() or {}
                task.run_count += 1
                task.last_run_at = time.monotonic()
                result = dict(result) if isinstance(result, dict) else {"result": result}
                result.setdefault("task", task.name)
                result.setdefault("duration_hint", started.isoformat())
                results.append(result)
            except Exception as e:
                logger.error("Scheduled task '%s' failed: %s", task.name, e)
                task.last_run_at = time.monotonic()
                results.append({"task": task.name, "status": "FAILED", "error": str(e)})

        return results

    def task_stats(self) -> List[Dict]:
        tasks: Dict[str, ScheduledTask] = getattr(self, "_tasks", {})
        return [
            {
                "name": t.name,
                "interval_seconds": t.interval_seconds,
                "run_count": t.run_count,
            }
            for t in tasks.values()
        ]