from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.core.models.agent import AgentConfig


class AgentRegistry:
    """Registry for available agents in the system.

    The CEO uses this registry to discover and dispatch agents rather than
    instantiating them directly throughout its code.
    """

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._configurations: Dict[str, AgentConfig] = {}
        self._enabled: Dict[str, bool] = {}
        self._agent_types: Dict[str, str] = {}  # Track agent type for dispatch logic

    def register(
        self,
        agent: BaseAgent,
        configuration: Optional[AgentConfig] = None,
        enabled: bool = True,
        agent_type: str = "general",
    ) -> None:
        """Register an agent in the registry."""
        self._agents[agent.agent_id] = agent
        self._configurations[agent.agent_id] = configuration or AgentConfig(
            version="1.0.0", model=None, max_tokens=1200
        )
        self._enabled[agent.agent_id] = enabled
        self._agent_types[agent.agent_id] = agent_type

    def retrieve(self, agent_id: str) -> Optional[BaseAgent]:
        """Retrieve an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self._agents.keys())

    def list_agents_by_type(self, agent_type: str) -> List[str]:
        """List all registered agent IDs of a specific type."""
        return [aid for aid, atype in self._agent_types.items() if atype == agent_type]

    def check_enabled(self, agent_id: str) -> bool:
        """Check if an agent is enabled."""
        return self._enabled.get(agent_id, False)

    def get_configuration(self, agent_id: str) -> Optional[AgentConfig]:
        """Retrieve an agent's configuration."""
        return self._configurations.get(agent_id)

    def get_agent_type(self, agent_id: str) -> Optional[str]:
        """Get the type of an agent."""
        return self._agent_types.get(agent_id)

    def enable(self, agent_id: str) -> None:
        """Enable an agent."""
        if agent_id in self._enabled:
            self._enabled[agent_id] = True

    def disable(self, agent_id: str) -> None:
        """Disable an agent."""
        if agent_id in self._enabled:
            self._enabled[agent_id] = False

    def is_registered(self, agent_id: str) -> bool:
        """Check if an agent is registered."""
        return agent_id in self._agents
