from abc import ABC, abstractmethod
from typing import List, Optional
from app.core.models.agent import AgentConfig, AgentStatus, AgentResult
from app.core.models.task import Task

class BaseAgent(ABC):
    def __init__(self, agent_id: str, agent_name: str, role: str, description: str, version: str, configuration: AgentConfig, capabilities: List[str]):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.role = role
        self.description = description
        self.version = version
        self.configuration = configuration
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE

    @abstractmethod
    def process_task(self, task: Task) -> AgentResult:
        pass
