import unittest
from app.agents.base import BaseAgent
from app.core.models.agent import AgentConfig, AgentStatus, AgentResult
from app.core.models.task import Task
from app.utils.time import now_utc

class DummyAgent(BaseAgent):
    def process_task(self, task: Task) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            timestamp=now_utc(),
            status=AgentStatus.SUCCESS,
            summary="Dummy task processed",
            confidence=1.0
        )

class TestBaseAgent(unittest.TestCase):
    def test_base_agent_creation(self):
        config = AgentConfig()
        agent = DummyAgent(
            agent_id="dummy_1",
            agent_name="Dummy",
            role="tester",
            description="A dummy agent",
            version="1.0.0",
            configuration=config,
            capabilities=["test"]
        )
        self.assertEqual(agent.status, AgentStatus.IDLE)
        self.assertEqual(agent.agent_id, "dummy_1")

if __name__ == '__main__':
    unittest.main()
