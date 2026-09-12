from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class Task:
    task_id: str
    requesting_agent: str
    target_agent: str
    task_type: str
    priority: int
    input_data: Dict[str, Any]
    created_at: datetime
    status: TaskStatus = TaskStatus.PENDING
