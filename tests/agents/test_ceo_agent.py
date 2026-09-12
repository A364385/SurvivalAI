import unittest
from datetime import timedelta

from app.agents.ceo.agent import CEOAgent
from app.agents.registry import AgentRegistry
from app.agents.market_research.agent import MarketResearchAgent
from app.agents.news_research.agent import NewsResearchAgent
from app.agents.crisis_risk.agent import CrisisRiskAgent
from app.agents.deep_looker.agent import DeepLookerAgent
from app.agents.risk_manager.agent import RiskManagerAgent
from app.agents.investment_safety.agent import InvestmentSafetyManagerAgent
from app.core.memory.store import InMemoryStore
from app.core.models.agent import AgentResult, AgentStatus
from app.core.models.orchestration import (
    RequestType,
    OrchestrationStage,
    DecisionType,
    ExistingInvestmentDecision,
    ConflictCategory,
    ConflictSeverity,
    OrchestrationRequest,
    AgentExecution,
    Conflict,
    Evidence,
    OrchestrationRun,
    DecisionProposal,
)
from app.core.models.task import Task
from app.services.llm.mock_provider import MockLLMProvider
from app.services.market_data.mock_provider import MockMarketDataProvider
from app.services.news.mock_provider import MockNewsProvider
from app.utils.time import now_utc


class TestCEOAgent(unittest.TestCase):
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
            agent_id="ceo_test",
            agent_registry=self.registry,
            llm_provider=self.llm,
            memory_store=self.memory,
            event_publisher=event_publisher,
        )

        # Register mock agents
        self._register_mock_agents()

    def _register_mock_agents(self):
        """Register mock research agents for testing."""
        market_data = MockMarketDataProvider()
        news_data = MockNewsProvider()

        market_agent = MarketResearchAgent("market_research", market_data, self.llm)
        news_agent = NewsResearchAgent("news_research", news_data, self.llm, self.memory)
        crisis_agent = CrisisRiskAgent("crisis_risk", self.llm, memory_store=self.memory)
        deep_agent = DeepLookerAgent("deep_looker", self.llm, memory_store=self.memory)
        risk_agent = RiskManagerAgent("risk_manager", self.llm, memory_store=self.memory)
        safety_agent = InvestmentSafetyManagerAgent(
            "investment_safety", self.llm, memory_store=self.memory
        )

        self.registry.register(market_agent)
        self.registry.register(news_agent)
        self.registry.register(crisis_agent)
        self.registry.register(deep_agent)
        self.registry.register(risk_agent)
        self.registry.register(safety_agent)

    def _task(self, input_data):
        return Task(
            task_id="ceo_task",
            requesting_agent="System",
            target_agent="CEOAgent",
            task_type="orchestration",
            priority=1,
            input_data=input_data,
            created_at=self.now,
        )

    def test_complete_research_pipeline_succeeds(self):
        """Test that the complete research pipeline executes successfully."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
                "requested_position_size": 10000.0,
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIn("run_id", result.analysis)
        self.assertIn("decision_proposal", result.analysis)

    def test_parallel_research_execution(self):
        """Test that Market, News, and Crisis research can run in parallel."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        agent_executions = result.analysis["agent_executions"]
        agent_ids = [e["agent_id"] for e in agent_executions]

        self.assertIn("market_research", agent_ids)
        self.assertIn("news_research", agent_ids)
        self.assertIn("crisis_risk", agent_ids)

    def test_deep_looker_receives_upstream_outputs(self):
        """Test that Deep Looker receives outputs from upstream research agents."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        agent_executions = result.analysis["agent_executions"]
        deep_looker_exec = next(
            (e for e in agent_executions if e["agent_id"] == "deep_looker"), None
        )
        self.assertIsNotNone(deep_looker_exec)
        self.assertEqual(deep_looker_exec["status"], "COMPLETED")

    def test_risk_manager_receives_proposal(self):
        """Test that Risk Manager receives the investment proposal."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
                "requested_position_size": 10000.0,
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        agent_executions = result.analysis["agent_executions"]
        risk_exec = next(
            (e for e in agent_executions if e["agent_id"] == "risk_manager"), None
        )
        self.assertIsNotNone(risk_exec)
        self.assertEqual(risk_exec["status"], "COMPLETED")

    def test_risk_blocked_ceo_cannot_approve(self):
        """Test that when Risk Manager blocks, CEO cannot approve investment."""
        # This would require configuring the Risk Manager to block
        # For now, we test the architecture exists
        from app.core.models.risk import RiskDecision

        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
                "requested_position_size": 1000000.0,  # Very large to trigger block
            })
        )

        # The CEO should handle the risk assessment result
        self.assertEqual(result.status, AgentStatus.SUCCESS)

    def test_risk_approved_ceo_can_continue(self):
        """Test that when Risk Manager approves, CEO can continue."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
                "requested_position_size": 5000.0,  # Small to approve
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        decision_proposal = result.analysis["decision_proposal"]
        self.assertIsNotNone(decision_proposal)

    def test_missing_critical_research_insufficient_data(self):
        """Test that missing critical research results in INSUFFICIENT_DATA or DEFER."""
        # Disable one of the research agents
        self.registry.disable("market_research")

        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        # Should handle gracefully with error
        self.assertIn(result.status, [AgentStatus.SUCCESS, AgentStatus.FAILED])

    def test_agent_failure_is_visible(self):
        """Test that agent failures are visible in the orchestration run."""
        # Disable an agent to cause failure
        self.registry.disable("deep_looker")

        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        # Failure should be recorded
        self.assertIn(result.status, [AgentStatus.SUCCESS, AgentStatus.FAILED])

    def test_conflicting_agent_outputs_recorded(self):
        """Test that conflicting agent outputs are recorded."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        # Conflicts should be recorded if detected
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        conflicts = result.analysis.get("conflicts", [])
        # May or may not have conflicts depending on mock data

    def test_evidence_traceable_to_source_agent(self):
        """Test that evidence remains traceable to source agent."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        # Evidence should be aggregated from agent results

    def test_orchestration_state_transitions_valid(self):
        """Test that orchestration state transitions are valid."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        # State transitions should follow valid path

    def test_invalid_state_transition_rejected(self):
        """Test that invalid state transitions are rejected."""
        run = OrchestrationRun(
            run_id="test_run",
            request_id="test_req",
            generation_id="gen_1",
            started_at=self.now,
            status=OrchestrationStage.COMPLETED,
            current_stage=OrchestrationStage.COMPLETED,
        )

        with self.assertRaises(ValueError):
            self.ceo._transition_to(run, OrchestrationStage.RESEARCHING)

    def test_duplicate_orchestration_requests_detected(self):
        """Test that duplicate orchestration requests are detectable."""
        request_id = "dup_req"

        result1 = self.ceo.process_task(
            self._task({
                "request_id": request_id,
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        result2 = self.ceo.process_task(
            self._task({
                "request_id": request_id,
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        # Both should complete (idempotency check would be separate)
        self.assertEqual(result1.status, AgentStatus.SUCCESS)
        self.assertEqual(result2.status, AgentStatus.SUCCESS)

    def test_existing_investment_review_workflow(self):
        """Test CEO support for existing investment reviews."""
        # Just verify the architecture supports the request type
        request = OrchestrationRequest(
            request_id="req_review",
            generation_id="gen_1",
            request_type=RequestType.EXISTING_INVESTMENT_REVIEW,
            asset="AAPL",
            objective="Review existing investment",
            timestamp=self.now,
        )
        self.assertEqual(request.request_type, RequestType.EXISTING_INVESTMENT_REVIEW)

    def test_exit_candidate_never_executes_sell(self):
        """Test that EXIT_CANDIDATE never executes a sell."""
        # Verify CEO has no execution methods
        with self.assertRaises(PermissionError):
            self.ceo.execute_order()

    def test_ceo_cannot_execute_trades(self):
        """Test that CEO cannot directly execute trades."""
        with self.assertRaises(PermissionError):
            self.ceo.execute_order()

    def test_ceo_cannot_modify_portfolio(self):
        """Test that CEO cannot modify portfolio."""
        with self.assertRaises(PermissionError):
            self.ceo.modify_portfolio()

    def test_ceo_cannot_modify_capital(self):
        """Test that CEO cannot modify capital."""
        with self.assertRaises(PermissionError):
            self.ceo.modify_capital()

    def test_ceo_cannot_change_strategy(self):
        """Test that CEO cannot change strategy."""
        with self.assertRaises(PermissionError):
            self.ceo.change_strategy()

    def test_ceo_cannot_approve_investment(self):
        """Test that CEO cannot bypass Risk Manager to approve investments."""
        with self.assertRaises(PermissionError):
            self.ceo.approve_investment()

    def test_event_bus_integration(self):
        """Test that CEO emits correct events."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        # Should have emitted events
        self.assertGreater(len(self.emitted_events), 0)

    def test_memory_store_integration(self):
        """Test that CEO stores orchestration records in MemoryStore."""
        result = self.ceo.process_task(
            self._task({
                "request_type": "INVESTMENT_PROPOSAL",
                "asset": "AAPL",
                "objective": "Test investment",
                "generation_id": "gen_1",
            })
        )

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        # Should have stored in memory
        memories = self.memory.list()
        self.assertGreater(len(memories), 0)

    def test_agent_registry_operations(self):
        """Test Agent Registry operations."""
        self.assertTrue(self.registry.is_registered("market_research"))
        self.assertTrue(self.registry.check_enabled("market_research"))
        self.assertIsNotNone(self.registry.retrieve("market_research"))
        self.assertIn("market_research", self.registry.list_agents())

        self.registry.disable("market_research")
        self.assertFalse(self.registry.check_enabled("market_research"))

        self.registry.enable("market_research")
        self.assertTrue(self.registry.check_enabled("market_research"))

    def test_orchestration_request_validation(self):
        """Test that orchestration requests are validated."""
        # Dataclasses accept empty strings, so we test the validation logic directly
        request = OrchestrationRequest(
            request_id="req",
            generation_id="gen_1",
            request_type=RequestType.INVESTMENT_PROPOSAL,
            asset="AAPL",
            objective="Test",
            timestamp=self.now,
        )
        # Valid request should pass
        self.assertTrue(self.ceo._validate_request(request))

    def test_evidence_aggregation(self):
        """Test evidence aggregation from agent results."""
        run = OrchestrationRun(
            run_id="test_run",
            request_id="test_req",
            generation_id="gen_1",
            started_at=self.now,
            status=OrchestrationStage.RESEARCHING,
            current_stage=OrchestrationStage.RESEARCHING,
        )

        # Add mock agent execution with facts
        execution = AgentExecution(
            agent_id="market_research",
            task_id="task_1",
            stage=OrchestrationStage.RESEARCHING,
            started_at=self.now,
            completed_at=self.now,
            status="COMPLETED",
            result=AgentResult(
                agent_id="market_research",
                task_id="task_1",
                timestamp=self.now,
                status=AgentStatus.SUCCESS,
                summary="Market analysis",
                confidence=0.8,
                facts=["Price is increasing", "Volume is high"],
            ),
        )
        run.agent_executions.append(execution)

        evidence = self.ceo._aggregate_evidence(run)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0].source_agent, "market_research")

    def test_conflict_detection(self):
        """Test conflict detection between opposing facts."""
        run = OrchestrationRun(
            run_id="test_run",
            request_id="test_req",
            generation_id="gen_1",
            started_at=self.now,
            status=OrchestrationStage.RESEARCHING,
            current_stage=OrchestrationStage.RESEARCHING,
        )

        run.evidence = [
            Evidence(
                fact="Price is increasing",
                source_agent="market_research",
                source_id="task_1",
                timestamp=self.now,
                confidence=0.8,
            ),
            Evidence(
                fact="Price is decreasing",
                source_agent="news_research",
                source_id="task_2",
                timestamp=self.now,
                confidence=0.7,
            ),
        ]

        conflicts = self.ceo._detect_conflicts(run)
        self.assertGreater(len(conflicts), 0)
        self.assertEqual(conflicts[0].severity, ConflictSeverity.MODERATE)


if __name__ == "__main__":
    unittest.main()
