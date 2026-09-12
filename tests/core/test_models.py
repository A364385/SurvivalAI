import unittest
from datetime import datetime, timezone
from app.core.models.agent import AgentConfig, AgentStatus, AgentResult
from app.core.models.error import ErrorInfo
from app.core.models.knowledge import Fact, Source, Evidence
from app.core.models.task import Task, TaskStatus
from app.core.models.events import AgentStarted
from app.utils.ids import generate_id
from app.utils.time import now_utc

class TestCoreModels(unittest.TestCase):
    def test_ids_and_time(self):
        id1 = generate_id("test")
        self.assertTrue(id1.startswith("test_"))
        
        now = now_utc()
        self.assertIsInstance(now, datetime)
        self.assertEqual(now.tzinfo, timezone.utc)

    def test_agent_models(self):
        config = AgentConfig(enabled=True, model="test-model")
        self.assertEqual(config.model, "test-model")
        
        result = AgentResult(
            agent_id="a1",
            task_id="t1",
            timestamp=now_utc(),
            status=AgentStatus.SUCCESS,
            summary="Done",
            confidence=0.95
        )
        self.assertEqual(result.status, AgentStatus.SUCCESS)

    def test_knowledge_models(self):
        src = Source(
            source_id="s1",
            title="Test Source",
            publisher="News Corp",
            url="http://example.com"
        )
        fact = Fact(
            fact_id="f1",
            statement="Test Fact",
            source_ids=["s1"],
            timestamp=now_utc(),
            confidence=0.9,
            data_type="financial"
        )
        self.assertEqual(fact.source_ids[0], "s1")

    def test_task_models(self):
        task = Task(
            task_id="t1",
            requesting_agent="a1",
            target_agent="a2",
            task_type="analysis",
            priority=1,
            input_data={},
            created_at=now_utc()
        )
        self.assertEqual(task.status, TaskStatus.PENDING)

    def test_events(self):
        event = AgentStarted(
            event_id="e1",
            timestamp=now_utc(),
            event_type="AgentStarted",
            agent_id="a1",
            task_id="t1"
        )
        self.assertEqual(event.event_type, "AgentStarted")

if __name__ == '__main__':
    unittest.main()
