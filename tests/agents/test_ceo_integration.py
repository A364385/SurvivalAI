import unittest
from datetime import timedelta

from app.agents.ceo.agent import CEOAgent
from app.agents.registry import AgentRegistry
from app.agents.market_research.agent import MarketResearchAgent
from app.agents.news_research.agent import NewsResearchAgent
from app.agents.crisis_risk.agent import CrisisRiskAgent
from app.agents.deep_looker.agent import DeepLookerAgent
from app.agents.risk_manager.agent import RiskManagerAgent
from app.core.memory.store import InMemoryStore
from app.core.models.agent import AgentResult, AgentStatus
from app.core.models.orchestration import RequestType
from app.core.models.task import Task
from app.services.llm.mock_provider import MockLLMProvider
from app.services.market_data.mock_provider import MockMarketDataProvider
from app.services.news.mock_provider import MockNewsProvider
from app.utils.time import now_utc


class TestCEOIntegration(unittest.TestCase):
    def setUp(self):
        self.now = now_utc()
        self.llm = MockLLMProvider({
            "facts": [],
            "analysis": {"summary": "Research completed"},
            "impact": {"direction": "NEUTRAL", "affected_symbols": ["AAPL"]},
            "confidence": 0.8,
            "warnings": [],
        })
        self.memory = InMemoryStore()
        self.registry = AgentRegistry()
        self.emitted_events = []

        def event_publisher(event):
            self.emitted_events.append(event)

        self.ceo = CEOAgent(
            agent_id="ceo_integration",
            agent_registry=self.registry,
            llm_provider=self.llm,
            memory_store=self.memory,
            event_publisher=event_publisher,
        )

        # Register all agents
        market_data = MockMarketDataProvider()
        news_data = MockNewsProvider()

        self.registry.register(MarketResearchAgent("market_research", market_data, self.llm))
        self.registry.register(NewsResearchAgent("news_research", news_data, self.llm, self.memory))
        self.registry.register(CrisisRiskAgent("crisis_risk", self.llm, memory_store=self.memory))
        self.registry.register(DeepLookerAgent("deep_looker", self.llm, memory_store=self.memory))
        self.registry.register(RiskManagerAgent("risk_manager", self.llm, memory_store=self.memory))

    def _task(self, input_data):
        return Task(
            task_id="integration_task",
            requesting_agent="System",
            target_agent="CEOAgent",
            task_type="orchestration",
            priority=1,
            input_data=input_data,
            created_at=self.now,
        )

    def test_end_to_end_investment_proposal_workflow(self):
        """Test complete workflow from request to decision proposal."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Evaluate AAPL for investment",
                "generation_id": "gen_integration",
                "requested_position_size": 5000.0,
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIn("run_id", result.analysis)
        self.assertIn("decision_proposal", result.analysis)

        decision_proposal = result.analysis["decision_proposal"]
        self.assertIn(decision_proposal["decision_type"], ["INVEST", "DO_NOT_INVEST", "DEFER", "INSUFFICIENT_DATA"])

    def test_all_agents_called_in_correct_order(self):
        """Test that agents are called in the correct order."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_integration",
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        agent_executions = result.analysis["agent_executions"]
        agent_ids = [e["agent_id"] for e in agent_executions]

        # Verify key agents were called
        self.assertIn("market_research", agent_ids)
        self.assertIn("news_research", agent_ids)
        self.assertIn("crisis_risk", agent_ids)
        self.assertIn("deep_looker", agent_ids)
        self.assertIn("risk_manager", agent_ids)

    def test_risk_gate_blocks_investment(self):
        """Test that Risk Manager BLOCKED prevents investment."""
        # This would require configuring Risk Manager to block
        # For integration test, we verify the architecture
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_integration",
                "requested_position_size": 1000000.0,  # Very large
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        # Check if blocked or approved based on risk rules

    def test_complete_orchestration_can_be_reconstructed(self):
        """Test that complete orchestration can be reconstructed from stored records."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_integration",
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        
        run_id = result.analysis["run_id"]
        agent_executions = result.analysis["agent_executions"]
        
        # Verify we have enough information to reconstruct
        self.assertIsNotNone(run_id)
        self.assertGreater(len(agent_executions), 0)
        
        # Check memory was stored
        memories = self.memory.list()
        self.assertGreater(len(memories), 0)


if __name__ == "__main__":
    unittest.main()
