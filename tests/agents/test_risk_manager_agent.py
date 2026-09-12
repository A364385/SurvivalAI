import unittest
from datetime import timedelta

from app.agents.deep_looker.agent import DeepLookerAgent
from app.agents.market_research.agent import MarketResearchAgent
from app.agents.risk_manager.agent import RiskManagerAgent
from app.core.memory.store import InMemoryStore
from app.core.models.agent import AgentResult, AgentStatus
from app.core.models.events import RiskAssessmentCompleted, RiskAssessmentStarted, RiskViolationEvent
from app.core.models.execution import ExecutionEnvironment, OrderSide, OrderType, TimeInForce, OrderRequest
from app.core.models.fundamental import CompanyProfile, FinancialStatements, ValuationMetrics
from app.core.models.market import Bar, MarketSnapshot
from app.core.models.memory import MemoryType
from app.core.models.risk import (
    InvestmentProposal,
    PortfolioRiskState,
    PositionExposure,
    RiskDecision,
    RiskPolicy,
)
from app.core.models.task import Task
from app.services.execution.mock_provider import MockExecutionProvider
from app.services.execution.safety import PaperOnlyExecutionProvider
from app.services.fundamental.mock_provider import MockFundamentalDataProvider
from app.services.llm.mock_provider import MockLLMProvider
from app.services.market_data.mock_provider import MockMarketDataProvider
from app.utils.time import now_utc


class TestRiskManagerAgent(unittest.TestCase):
    def setUp(self):
        self.now = now_utc()
        self.llm = MockLLMProvider({
            "facts": [],
            "analysis": {"summary": "LLM says approve, but deterministic engine must remain authoritative."},
            "impact": {"direction": "POSITIVE", "affected_symbols": ["AAPL"]},
            "confidence": 0.99,
            "warnings": [],
        })
        self.memory = InMemoryStore()
        self.policy = RiskPolicy(
            max_single_position_pct=0.10,
            max_portfolio_concentration_pct=0.15,
            max_sector_exposure_pct=0.35,
            max_asset_class_exposure_pct=0.70,
            max_geographic_exposure_pct=0.80,
            min_cash_reserve_pct=0.20,
            max_positions=3,
            require_market_data=False,
            require_deep_research=False,
        )
        self.agent = RiskManagerAgent(
            agent_id="risk_1",
            llm_provider=self.llm,
            memory_store=self.memory,
            policy=self.policy,
        )

    def _proposal(self, value=5000.0, symbol="AAPL", sector="Technology", asset_class="EQUITY", geography="US", metadata=None):
        return InvestmentProposal(
            proposal_id="prop_1",
            asset=symbol,
            proposed_position_value=value,
            sector=sector,
            asset_class=asset_class,
            geography=geography,
            metadata=metadata or {},
        )

    def _portfolio(self, value=100000.0, cash=50000.0, positions=None, peak=None):
        if positions is None:
            positions = [
                PositionExposure("MSFT", 10000.0, asset_class="EQUITY", sector="Technology", geography="US", correlation_group="mega_tech"),
                PositionExposure("GLD", 5000.0, asset_class="COMMODITY", sector="Metals", geography="GLOBAL", correlation_group="gold"),
            ]
        return PortfolioRiskState(
            portfolio_value=value,
            available_cash=cash,
            positions=positions,
            historical_peak_value=peak,
            timestamp=self.now,
        )

    def _task(self, input_data):
        return Task(
            task_id="risk_task",
            requesting_agent="CEOAgent",
            target_agent="RiskManagerAgent",
            task_type="risk_assessment",
            priority=1,
            input_data=input_data,
            created_at=self.now,
        )

    def _bars(self, prices, symbol="AAPL"):
        return [
            Bar(symbol, self.now - timedelta(days=len(prices) - i), p, p * 1.02, p * 0.98, p, 1000 + i)
            for i, p in enumerate(prices)
        ]

    def test_valid_low_risk_investment_approved(self):
        assessment = self.agent.evaluate_investment(self._proposal(), self._portfolio(), self.policy)
        self.assertEqual(assessment.decision, RiskDecision.APPROVED)
        self.assertTrue(assessment.execution_allowed())
        self.assertAlmostEqual(assessment.portfolio_exposure, 0.05)

    def test_position_too_large_blocked(self):
        assessment = self.agent.evaluate_investment(self._proposal(value=20000.0), self._portfolio(), self.policy)
        self.assertEqual(assessment.decision, RiskDecision.BLOCKED)
        self.assertTrue(any(r.rule_id == "max_single_position_pct" for r in assessment.violated_rules))

    def test_cash_reserve_violation_blocked(self):
        assessment = self.agent.evaluate_investment(self._proposal(value=35000.0), self._portfolio(cash=50000.0), self.policy)
        self.assertEqual(assessment.decision, RiskDecision.BLOCKED)
        self.assertTrue(any(r.rule_id == "minimum_cash_reserve" for r in assessment.violated_rules))

    def test_sector_concentration_blocked(self):
        policy = RiskPolicy(max_sector_exposure_pct=0.12, require_market_data=False)
        assessment = self.agent.evaluate_investment(self._proposal(value=5000.0), self._portfolio(), policy)
        self.assertEqual(assessment.decision, RiskDecision.BLOCKED)
        self.assertTrue(any(r.rule_id == "max_sector_exposure_pct" for r in assessment.violated_rules))

    def test_asset_class_concentration_blocked(self):
        policy = RiskPolicy(max_asset_class_exposure_pct=0.12, require_market_data=False)
        assessment = self.agent.evaluate_investment(self._proposal(value=5000.0), self._portfolio(), policy)
        self.assertEqual(assessment.decision, RiskDecision.BLOCKED)
        self.assertTrue(any(r.rule_id == "max_asset_class_exposure_pct" for r in assessment.violated_rules))

    def test_geographic_concentration_blocked(self):
        policy = RiskPolicy(max_geographic_exposure_pct=0.12, require_market_data=False)
        assessment = self.agent.evaluate_investment(self._proposal(value=5000.0), self._portfolio(), policy)
        self.assertEqual(assessment.decision, RiskDecision.BLOCKED)
        self.assertTrue(any(r.rule_id == "max_geographic_exposure_pct" for r in assessment.violated_rules))

    def test_too_many_positions_blocked(self):
        positions = [
            PositionExposure("A", 1000.0),
            PositionExposure("B", 1000.0),
            PositionExposure("C", 1000.0),
        ]
        assessment = self.agent.evaluate_investment(self._proposal(symbol="D", sector=None, asset_class=None, geography=None), self._portfolio(positions=positions), self.policy)
        self.assertEqual(assessment.decision, RiskDecision.BLOCKED)
        self.assertTrue(any(r.rule_id == "max_positions" for r in assessment.violated_rules))

    def test_elevated_volatility_warning(self):
        snapshot = self._snapshot_with_volatility(0.40)
        assessment = self.agent.evaluate_investment(self._proposal(), self._portfolio(), self.policy, market_snapshot=snapshot)
        self.assertEqual(assessment.decision, RiskDecision.APPROVED_WITH_WARNINGS)
        self.assertTrue(any("volatility" in w.lower() for w in assessment.warnings))

    def test_severe_crisis_exposure_policy_block(self):
        crisis = AgentResult(
            agent_id="crisis",
            task_id="crisis_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Severe sanctions exposure.",
            confidence=0.8,
            analysis={"events": [{"event_id": "c1", "severity": "CRITICAL"}]},
        )
        assessment = self.agent.evaluate_investment(self._proposal(), self._portfolio(), self.policy, crisis_result=crisis)
        self.assertEqual(assessment.decision, RiskDecision.BLOCKED)
        self.assertTrue(any(r.rule_id == "crisis_block_severity" for r in assessment.violated_rules))

    def test_missing_critical_data_insufficient(self):
        policy = RiskPolicy(require_market_data=True, require_deep_research=True)
        assessment = self.agent.evaluate_investment(self._proposal(), self._portfolio(), policy)
        self.assertEqual(assessment.decision, RiskDecision.INSUFFICIENT_DATA)
        self.assertTrue(any(r.dimension.value == "DATA_QUALITY" for r in assessment.violated_rules))

    def test_stale_market_data_warning(self):
        snapshot = self._snapshot_with_volatility(0.10)
        snapshot.retrieved_at = self.now - timedelta(hours=3)
        assessment = self.agent.evaluate_investment(self._proposal(), self._portfolio(), self.policy, market_snapshot=snapshot)
        self.assertEqual(assessment.decision, RiskDecision.APPROVED_WITH_WARNINGS)
        self.assertTrue(any("stale" in w.lower() for w in assessment.warnings))

    def test_zero_and_negative_portfolio_value_safe_failure(self):
        with self.assertRaises(ValueError):
            self.agent.evaluate_investment(self._proposal(), self._portfolio(value=0), self.policy)
        with self.assertRaises(ValueError):
            self.agent.evaluate_investment(self._proposal(), self._portfolio(value=-1), self.policy)

    def test_correlated_positions_warning(self):
        proposal = self._proposal(metadata={"correlation_group": "mega_tech"})
        policy = RiskPolicy(max_correlated_exposure_pct=0.12, require_market_data=False)
        assessment = self.agent.evaluate_investment(proposal, self._portfolio(), policy)
        self.assertEqual(assessment.decision, RiskDecision.APPROVED_WITH_WARNINGS)
        self.assertTrue(any("correlated" in w.lower() for w in assessment.warnings))

    def test_multiple_rules_violated_all_reported(self):
        policy = RiskPolicy(max_single_position_pct=0.05, min_cash_reserve_pct=0.90, max_positions=1, require_market_data=False)
        assessment = self.agent.evaluate_investment(self._proposal(value=20000), self._portfolio(), policy)
        rule_ids = {r.rule_id for r in assessment.violated_rules}
        self.assertIn("max_single_position_pct", rule_ids)
        self.assertIn("minimum_cash_reserve", rule_ids)
        self.assertIn("max_positions", rule_ids)

    def test_hard_rule_cannot_be_overridden_by_llm(self):
        result = self.agent.process_task(self._task({
            "proposal": self._proposal(value=25000.0),
            "portfolio_state": self._portfolio(),
            "risk_policy": self.policy,
            "generation_id": "gen_1",
        }))
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertEqual(result.analysis["decision"], RiskDecision.BLOCKED.value)
        self.assertFalse(result.analysis["execution_allowed"])

    def test_memory_and_events_emitted(self):
        result = self.agent.process_task(self._task({
            "proposal": self._proposal(),
            "portfolio_state": self._portfolio(),
            "generation_id": "gen_events",
        }))
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertTrue(self.memory.query(generation_id="gen_events", memory_type=MemoryType.ANALYSIS))
        self.assertTrue(any(isinstance(e, RiskAssessmentStarted) for e in self.agent.emitted_events))
        self.assertTrue(any(isinstance(e, RiskAssessmentCompleted) for e in self.agent.emitted_events))

    def test_violation_event_emitted_for_block(self):
        self.agent.process_task(self._task({
            "proposal": self._proposal(value=25000.0),
            "portfolio_state": self._portfolio(),
        }))
        self.assertTrue(any(isinstance(e, RiskViolationEvent) for e in self.agent.emitted_events))

    def test_risk_manager_permissions(self):
        for method in (
            self.agent.execute_order,
            self.agent.modify_portfolio,
            self.agent.modify_capital,
            self.agent.change_strategy,
            self.agent.approve_investment,
        ):
            with self.assertRaises(PermissionError):
                method()

    def test_paper_only_safety_boundary_remains_intact(self):
        provider = PaperOnlyExecutionProvider(MockExecutionProvider())
        request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id="safe",
            environment=ExecutionEnvironment.PAPER,
        )
        response = provider.submit_order(request)
        self.assertEqual(response.symbol, "AAPL")

    def test_integration_market_deep_risk(self):
        market = MockMarketDataProvider()
        market.set_bars("AAPL", self._bars([100.0 + i for i in range(30)]))
        fundamentals = MockFundamentalDataProvider()
        fundamentals.set_company_profile(CompanyProfile("AAPL", "Apple", "Technology", "Hardware", "US", "EQUITY"))
        fundamentals.set_financial_statements(FinancialStatements(
            symbol="AAPL",
            period_end=self.now - timedelta(days=30),
            revenue=120.0,
            prior_revenue=100.0,
            net_income=20.0,
            prior_net_income=18.0,
            cash=80.0,
            total_debt=30.0,
            total_equity=100.0,
            free_cash_flow=25.0,
            prior_free_cash_flow=20.0,
        ))
        fundamentals.set_valuation_metrics(ValuationMetrics(symbol="AAPL", market_price=100.0, market_cap=1000.0))
        llm = MockLLMProvider({
            "facts": [],
            "analysis": {"synthesis": "Research context available."},
            "impact": {"direction": "UNCERTAIN", "affected_symbols": ["AAPL"]},
            "confidence": 0.7,
            "warnings": [],
        })
        market_agent = MarketResearchAgent("mkt", market, llm, fundamental_provider=fundamentals)
        snapshot = market_agent.build_market_snapshot("AAPL", limit=30, now=self.now)
        deep_agent = DeepLookerAgent("deep", llm, fundamental_provider=fundamentals)
        deep_result = deep_agent.process_task(self._task({
            "symbol": "AAPL",
            "market_snapshot": snapshot,
            "generation_id": "gen_int",
        }))
        risk_result = self.agent.process_task(self._task({
            "proposal": self._proposal(value=5000.0, symbol="AAPL"),
            "portfolio_state": self._portfolio(),
            "market_snapshot": snapshot,
            "deep_research_result": deep_result,
            "generation_id": "gen_int",
        }))
        self.assertEqual(risk_result.status, AgentStatus.SUCCESS)
        self.assertIn(risk_result.analysis["decision"], (RiskDecision.APPROVED.value, RiskDecision.APPROVED_WITH_WARNINGS.value))
        self.assertTrue(risk_result.analysis["risk_assessment"].evidence)

    def _snapshot_with_volatility(self, vol: float) -> MarketSnapshot:
        market = MockMarketDataProvider()
        market.set_bars("AAPL", self._bars([100.0 + i for i in range(30)]))
        agent = MarketResearchAgent("mkt", market, self.llm)
        snapshot = agent.build_market_snapshot("AAPL", limit=30, now=self.now)
        snapshot.volatility.short_term_volatility = vol
        return snapshot


if __name__ == "__main__":
    unittest.main()
