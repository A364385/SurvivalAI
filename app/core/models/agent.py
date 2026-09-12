from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from .knowledge import Fact, Source
from .error import ErrorInfo

class AgentStatus(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DISABLED = "DISABLED"

@dataclass
class AgentConfig:
    enabled: bool = True
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1000
    timeout: int = 60
    budget: float = 0.0
    version: str = "1.0.0"

@dataclass
class AgentResult:
    agent_id: str
    task_id: str
    timestamp: datetime
    status: AgentStatus
    summary: str
    confidence: float
    facts: List[Fact] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    impact: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[ErrorInfo] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
