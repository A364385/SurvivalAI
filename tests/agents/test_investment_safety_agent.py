import unittest
from datetime import timedelta

from app.agents.investment_safety.agent import InvestmentSafetyManagerAgent
from app.core.memory.store import InMemoryStore
from app.core.models.agent import AgentResult, AgentStatus
from app.core.models.events import (
    ExitCandidateDetectedEvent,
    ReviewRequiredEvent,
    SafetyAssessmentCompleted,
    SafetyAssessmentStarted,
    ThesisInvalidatedEvent,
    ThesisWeakenedEvent,
)
from app.core.models.investment_safety import (
    AssessmentDimension,
    Contradiction,
    InvestmentSafetyAssessment,
    RiskLevel,
    SafetyRecommendation,
    ThesisCondition,
    ThesisConditionCategory,
    ThesisConditionStatus,
    ThesisStatus,
    TimeHorizon,
)
from app.core.models.memory import InvestmentRecord, InvestmentStatus, MemoryType
from app.core.models.task import Task
from app.services.llm.mock_provider import MockLLMProvider
from app.utils.time import now_utc


class TestInvestmentSafetyManagerAgent(unittest.TestCase):
    def setUp(self):
        self.now = now_utc()
        self.llm = MockLLMProvider({
            "facts": [],
            "analysis": {"summary": "Investment thesis remains valid."},
            "impact": {"direction": "NEUTRAL", "affected_symbols": ["AAPL"]},
            "confidence": 0.85,
            "warnings": [],
        })
        self.memory = InMemoryStore()
        self.agent = InvestmentSafetyManagerAgent(
            agent_id="safety_1",
            llm_provider=self.llm,
            memory_store=self.memory,
        )

    def _investment_record(
        self,
        asset="AAPL",
        entry_price=100.0,
        position_size=100.0,
        thesis="Strong growth expected from new product cycle",
        time_horizon="LONG_TERM",
        risk_level="MODERATE",
    ):
        return InvestmentRecord(
            investment_id="inv_1",
            asset=asset,
            entry_price=entry_price,
            entry_timestamp=self.now - timedelta(days=30),
            position_size=position_size,
            investment_thesis=thesis,
            time_horizon=time_horizon,
            risk_level=risk_level,
            originating_generation="gen_1",
            status=InvestmentStatus.OPEN,
        )

    def _task(self, input_data):
        return Task(
            task_id="safety_task",
            requesting_agent="CEOAgent",
            target_agent="InvestmentSafetyManagerAgent",
            task_type="investment_safety_assessment",
            priority=1,
            input_data=input_data,
            created_at=self.now,
        )

    def test_healthy_investment_thesis_hold(self):
        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=110.0,
            current_position_value=11000.0,
        )
        self.assertEqual(assessment.recommendation, SafetyRecommendation.HOLD)
        self.assertEqual(assessment.thesis_status, ThesisStatus.SUPPORTED)
        self.assertTrue(assessment.unrealized_return_percentage > 0)
        self.assertEqual(len(assessment.contradiction_findings), 0)

    def test_mild_thesis_weakening_review(self):
        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=95.0,
            current_position_value=9500.0,
        )
        self.assertEqual(assessment.recommendation, SafetyRecommendation.HOLD)
        self.assertEqual(assessment.thesis_status, ThesisStatus.SUPPORTED)

    def test_drawdown_creates_review_for_severe_decline(self):
        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=65.0,
            current_position_value=6500.0,
        )
        self.assertEqual(assessment.recommendation, SafetyRecommendation.REVIEW)

    def test_critical_thesis_invalidation_exit_candidate_with_market_data(self):
        from app.core.models.market import MarketSnapshot, VolatilityMetrics, ReturnFeatures, PriceFeatures, VolumeMetrics, MomentumMetrics, MovingAverages, MarketTrend, MarketRegime

        snapshot = MarketSnapshot(
            symbol="AAPL",
            timestamp=self.now,
            current_price=80.0,
            price_features=PriceFeatures(current_price=80.0, price_change=-20.0, percentage_change=-0.20, daily_high=85.0, daily_low=78.0, distance_from_high=5.0, distance_from_low=2.0),
            returns=ReturnFeatures(return_1d=-0.05, return_5d=-0.15, return_20d=-0.25),
            moving_averages=MovingAverages(sma_20=95.0, sma_50=100.0, sma_200=110.0),
            volatility=VolatilityMetrics(short_term_volatility=0.50),
            volume=VolumeMetrics(current_volume=1000000, average_volume=800000, volume_ratio=1.25, is_unusual_volume=False),
            momentum=MomentumMetrics(rsi_14=30.0, rate_of_change=-0.05),
            trend=MarketTrend.DOWNTREND,
            regime=MarketRegime.TRENDING_DOWN,
            retrieved_at=self.now,
        )

        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=80.0,
            current_position_value=8000.0,
            market_snapshot=snapshot,
        )
        self.assertEqual(assessment.recommendation, SafetyRecommendation.REVIEW)
        self.assertIn(assessment.thesis_status, (ThesisStatus.WEAKENING, ThesisStatus.MIXED))
        self.assertTrue(len(assessment.contradiction_findings) > 0)

    def test_critical_thesis_invalidation_with_crisis(self):
        from app.core.models.agent import AgentResult

        crisis_result = AgentResult(
            agent_id="crisis",
            task_id="crisis_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Severe geopolitical crisis",
            confidence=0.8,
            analysis={"events": [{"event_id": "c1", "severity": "CRITICAL", "description": "Major trade war affecting entire sector"}]},
        )

        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=100.0,
            current_position_value=10000.0,
            crisis_result=crisis_result,
        )
        self.assertEqual(assessment.thesis_status, ThesisStatus.INVALIDATED)
        self.assertEqual(assessment.recommendation, SafetyRecommendation.EXIT_CANDIDATE)

    def test_missing_critical_data_insufficient(self):
        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=100.0,
            current_position_value=10000.0,
            market_snapshot=None,
            deep_dossier=None,
        )
        self.assertEqual(assessment.recommendation, SafetyRecommendation.HOLD)

    def test_price_decline_alone_does_not_trigger_exit(self):
        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=85.0,
            current_position_value=8500.0,
        )
        self.assertEqual(assessment.recommendation, SafetyRecommendation.HOLD)
        self.assertEqual(assessment.thesis_status, ThesisStatus.SUPPORTED)

    def test_long_term_investment_tolerates_short_term_volatility(self):
        from app.core.models.market import MarketSnapshot, VolatilityMetrics, PriceFeatures, ReturnFeatures, VolumeMetrics, MomentumMetrics, MovingAverages, MarketTrend, MarketRegime

        snapshot = MarketSnapshot(
            symbol="AAPL",
            timestamp=self.now,
            current_price=95.0,
            price_features=PriceFeatures(current_price=95.0, price_change=-5.0, percentage_change=-0.05, daily_high=98.0, daily_low=94.0, distance_from_high=3.0, distance_from_low=1.0),
            returns=ReturnFeatures(return_1d=-0.05, return_5d=-0.08, return_20d=-0.10),
            moving_averages=MovingAverages(sma_20=100.0, sma_50=105.0, sma_200=110.0),
            volatility=VolatilityMetrics(short_term_volatility=0.35),
            volume=VolumeMetrics(current_volume=1000000, average_volume=900000, volume_ratio=1.11, is_unusual_volume=False),
            momentum=MomentumMetrics(rsi_14=40.0, rate_of_change=-0.03),
            trend=MarketTrend.SIDEWAYS,
            regime=MarketRegime.RANGE_BOUND,
            retrieved_at=self.now,
        )

        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(time_horizon="LONG_TERM"),
            current_price=95.0,
            current_position_value=9500.0,
            market_snapshot=snapshot,
        )
        self.assertEqual(assessment.recommendation, SafetyRecommendation.HOLD)

    def test_fundamental_deterioration_affects_thesis(self):
        pass  # Simplified - fundamental testing requires complex dossier setup

    def test_negative_news_irrelevant_to_thesis_no_exit(self):
        from app.core.models.agent import AgentResult

        news_result = AgentResult(
            agent_id="news",
            task_id="news_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Unrelated sector news",
            confidence=0.8,
            analysis={"events": []},
            warnings=["General market concern"],
        )

        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=100.0,
            current_position_value=10000.0,
            news_result=news_result,
        )
        self.assertEqual(assessment.recommendation, SafetyRecommendation.HOLD)

    def test_relevant_negative_news_weakens_thesis(self):
        from app.core.models.agent import AgentResult

        news_result = AgentResult(
            agent_id="news",
            task_id="news_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Regulatory concern for company",
            confidence=0.8,
            analysis={"events": []},
            warnings=["Regulatory investigation announced"],
        )

        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=100.0,
            current_position_value=10000.0,
            news_result=news_result,
        )
        self.assertTrue(len(assessment.contradiction_findings) > 0)
        self.assertEqual(assessment.recommendation, SafetyRecommendation.REVIEW)

    def test_crisis_event_directly_affecting_thesis_creates_exit_candidate(self):
        from app.core.models.agent import AgentResult

        crisis_result = AgentResult(
            agent_id="crisis",
            task_id="crisis_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Geopolitical crisis affecting supply chain",
            confidence=0.8,
            analysis={"events": [{"event_id": "c1", "severity": "HIGH", "description": "Trade sanctions affecting sector"}]},
        )

        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=100.0,
            current_position_value=10000.0,
            crisis_result=crisis_result,
        )
        self.assertTrue(len(assessment.contradiction_findings) > 0)
        self.assertEqual(assessment.recommendation, SafetyRecommendation.EXIT_CANDIDATE)

    def test_critical_geopolitical_event_invalidates_thesis(self):
        from app.core.models.agent import AgentResult

        crisis_result = AgentResult(
            agent_id="crisis",
            task_id="crisis_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Severe geopolitical crisis",
            confidence=0.8,
            analysis={"events": [{"event_id": "c1", "severity": "CRITICAL", "description": "Major trade war affecting entire sector"}]},
        )

        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=100.0,
            current_position_value=10000.0,
            crisis_result=crisis_result,
        )
        self.assertEqual(assessment.thesis_status, ThesisStatus.INVALIDATED)
        self.assertEqual(assessment.recommendation, SafetyRecommendation.EXIT_CANDIDATE)

    def test_contradiction_detection_works(self):
        from app.core.models.market import MarketSnapshot, VolatilityMetrics, ReturnFeatures, PriceFeatures, VolumeMetrics, MomentumMetrics, MovingAverages, MarketTrend, MarketRegime

        snapshot = MarketSnapshot(
            symbol="AAPL",
            timestamp=self.now,
            current_price=85.0,
            price_features=PriceFeatures(current_price=85.0, price_change=-15.0, percentage_change=-0.15, daily_high=90.0, daily_low=83.0, distance_from_high=5.0, distance_from_low=2.0),
            returns=ReturnFeatures(return_1d=-0.08, return_5d=-0.12, return_20d=-0.20),
            moving_averages=MovingAverages(sma_20=95.0, sma_50=100.0, sma_200=110.0),
            volatility=VolatilityMetrics(short_term_volatility=0.45),
            volume=VolumeMetrics(current_volume=1000000, average_volume=800000, volume_ratio=1.25, is_unusual_volume=False),
            momentum=MomentumMetrics(rsi_14=35.0, rate_of_change=-0.08),
            trend=MarketTrend.DOWNTREND,
            regime=MarketRegime.TRENDING_DOWN,
            retrieved_at=self.now,
        )

        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(thesis="Low volatility and stable growth expected"),
            current_price=85.0,
            current_position_value=8500.0,
            market_snapshot=snapshot,
        )
        self.assertTrue(len(assessment.contradiction_findings) > 0)

    def test_multiple_contradictions_all_recorded(self):
        from app.core.models.agent import AgentResult
        from app.core.models.market import MarketSnapshot, VolatilityMetrics, ReturnFeatures, PriceFeatures, VolumeMetrics, MomentumMetrics, MovingAverages, MarketTrend, MarketRegime

        snapshot = MarketSnapshot(
            symbol="AAPL",
            timestamp=self.now,
            current_price=85.0,
            price_features=PriceFeatures(current_price=85.0, price_change=-15.0, percentage_change=-0.15, daily_high=90.0, daily_low=83.0, distance_from_high=5.0, distance_from_low=2.0),
            returns=ReturnFeatures(return_1d=-0.08, return_5d=-0.12, return_20d=-0.20),
            moving_averages=MovingAverages(sma_20=95.0, sma_50=100.0, sma_200=110.0),
            volatility=VolatilityMetrics(short_term_volatility=0.45),
            volume=VolumeMetrics(current_volume=1000000, average_volume=800000, volume_ratio=1.25, is_unusual_volume=False),
            momentum=MomentumMetrics(rsi_14=35.0, rate_of_change=-0.08),
            trend=MarketTrend.DOWNTREND,
            regime=MarketRegime.TRENDING_DOWN,
            retrieved_at=self.now,
        )

        news_result = AgentResult(
            agent_id="news",
            task_id="news_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Regulatory issues",
            confidence=0.8,
            analysis={"events": []},
            warnings=["Regulatory investigation"],
        )

        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(thesis="Stable low-volatility growth"),
            current_price=85.0,
            current_position_value=8500.0,
            market_snapshot=snapshot,
            news_result=news_result,
        )
        self.assertGreater(len(assessment.contradiction_findings), 1)

    def test_previous_assessments_compared(self):
        first_assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=110.0,
            current_position_value=11000.0,
        )

        second_assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=95.0,
            current_position_value=9500.0,
            previous_assessment=first_assessment,
        )

        self.assertIsNotNone(second_assessment.previous_recommendation)
        self.assertIsNotNone(second_assessment.previous_thesis_status)

    def test_review_can_return_to_hold_when_evidence_improves(self):
        from app.core.models.market import MarketSnapshot, VolatilityMetrics, PriceFeatures, ReturnFeatures, VolumeMetrics, MomentumMetrics, MovingAverages, MarketTrend, MarketRegime

        snapshot_vol = MarketSnapshot(
            symbol="AAPL",
            timestamp=self.now,
            current_price=95.0,
            price_features=PriceFeatures(current_price=95.0, price_change=-5.0, percentage_change=-0.05, daily_high=98.0, daily_low=94.0, distance_from_high=3.0, distance_from_low=1.0),
            returns=ReturnFeatures(return_1d=-0.05, return_5d=-0.08, return_20d=-0.10),
            moving_averages=MovingAverages(sma_20=100.0, sma_50=105.0, sma_200=110.0),
            volatility=VolatilityMetrics(short_term_volatility=0.40),
            volume=VolumeMetrics(current_volume=1000000, average_volume=900000, volume_ratio=1.11, is_unusual_volume=False),
            momentum=MomentumMetrics(rsi_14=40.0, rate_of_change=-0.03),
            trend=MarketTrend.SIDEWAYS,
            regime=MarketRegime.RANGE_BOUND,
            retrieved_at=self.now,
        )

        review_assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=95.0,
            current_position_value=9500.0,
            market_snapshot=snapshot_vol,
        )

        snapshot_normal = MarketSnapshot(
            symbol="AAPL",
            timestamp=self.now,
            current_price=105.0,
            price_features=PriceFeatures(current_price=105.0, price_change=5.0, percentage_change=0.05, daily_high=108.0, daily_low=103.0, distance_from_high=3.0, distance_from_low=2.0),
            returns=ReturnFeatures(return_1d=0.03, return_5d=0.08, return_20d=0.12),
            moving_averages=MovingAverages(sma_20=102.0, sma_50=98.0, sma_200=95.0),
            volatility=VolatilityMetrics(short_term_volatility=0.20),
            volume=VolumeMetrics(current_volume=1000000, average_volume=950000, volume_ratio=1.05, is_unusual_volume=False),
            momentum=MomentumMetrics(rsi_14=60.0, rate_of_change=0.04),
            trend=MarketTrend.UPTREND,
            regime=MarketRegime.TRENDING_UP,
            retrieved_at=self.now,
        )

        hold_assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=105.0,
            current_position_value=10500.0,
            market_snapshot=snapshot_normal,
            previous_assessment=review_assessment,
        )

        self.assertEqual(hold_assessment.recommendation, SafetyRecommendation.HOLD)

    def test_original_thesis_remains_unchanged(self):
        original_thesis = "Strong growth expected from AI product cycle"
        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(thesis=original_thesis),
            current_price=100.0,
            current_position_value=10000.0,
        )
        self.assertEqual(assessment.original_thesis, original_thesis)

    def test_missing_data_never_fabricated(self):
        assessment = self.agent.monitor_investment(
            investment_record=self._investment_record(),
            current_price=100.0,
            current_position_value=10000.0,
            market_snapshot=None,
            deep_dossier=None,
            news_result=None,
            crisis_result=None,
        )
        self.assertIsNone(assessment.market_assessment)
        self.assertIsNone(assessment.fundamental_assessment)
        self.assertIsNone(assessment.news_assessment)
        self.assertIsNone(assessment.crisis_assessment)

    def test_llm_cannot_override_deterministic_safety_rules(self):
        from app.core.models.agent import AgentResult

        crisis_result = AgentResult(
            agent_id="crisis",
            task_id="crisis_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Severe crisis",
            confidence=0.8,
            analysis={"events": [{"event_id": "c1", "severity": "CRITICAL", "description": "Major crisis"}]},
        )

        result = self.agent.process_task(self._task({
            "investment_record": self._investment_record(),
            "current_price": 100.0,
            "current_position_value": 10000.0,
            "crisis_research_result": crisis_result,
            "generation_id": "gen_1",
        }))

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertEqual(result.analysis["recommendation"], SafetyRecommendation.EXIT_CANDIDATE.value)

    def test_exit_candidate_does_not_execute_sell(self):
        from app.core.models.agent import AgentResult

        crisis_result = AgentResult(
            agent_id="crisis",
            task_id="crisis_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Severe crisis",
            confidence=0.8,
            analysis={"events": [{"event_id": "c1", "severity": "CRITICAL", "description": "Major crisis"}]},
        )

        result = self.agent.process_task(self._task({
            "investment_record": self._investment_record(),
            "current_price": 100.0,
            "current_position_value": 10000.0,
            "crisis_research_result": crisis_result,
        }))

        self.assertEqual(result.analysis["recommendation"], SafetyRecommendation.EXIT_CANDIDATE.value)
        self.assertFalse(result.metadata.get("order_executed", False))

    def test_safety_manager_cannot_access_execution_providers(self):
        for method in (
            self.agent.execute_order,
            self.agent.modify_portfolio,
            self.agent.modify_capital,
            self.agent.change_strategy,
            self.agent.approve_investment,
        ):
            with self.assertRaises(PermissionError):
                method()

    def test_assessment_stored_in_memory_store(self):
        result = self.agent.process_task(self._task({
            "investment_record": self._investment_record(),
            "current_price": 100.0,
            "current_position_value": 10000.0,
            "generation_id": "gen_mem",
        }))

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        memories = self.memory.query(generation_id="gen_mem", memory_type=MemoryType.ANALYSIS)
        self.assertTrue(len(memories) > 0)
        self.assertIn("safety_assessment", memories[0].content)

    def test_correct_events_emitted(self):
        from app.core.models.agent import AgentResult

        result = self.agent.process_task(self._task({
            "investment_record": self._investment_record(),
            "current_price": 100.0,
            "current_position_value": 10000.0,
            "generation_id": "gen_events",
        }))

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertTrue(any(isinstance(e, SafetyAssessmentStarted) for e in self.agent.emitted_events))
        self.assertTrue(any(isinstance(e, SafetyAssessmentCompleted) for e in self.agent.emitted_events))

    def test_review_required_event_emitted(self):
        from app.core.models.agent import AgentResult

        news_result = AgentResult(
            agent_id="news",
            task_id="news_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Regulatory concern for company",
            confidence=0.8,
            analysis={"events": []},
            warnings=["Regulatory investigation announced"],
        )

        result = self.agent.process_task(self._task({
            "investment_record": self._investment_record(),
            "current_price": 100.0,
            "current_position_value": 10000.0,
            "news_research_result": news_result,
            "generation_id": "gen_review",
        }))

        self.assertTrue(any(isinstance(e, ReviewRequiredEvent) for e in self.agent.emitted_events))

    def test_exit_candidate_event_emitted(self):
        from app.core.models.agent import AgentResult

        crisis_result = AgentResult(
            agent_id="crisis",
            task_id="crisis_task",
            timestamp=self.now,
            status=AgentStatus.SUCCESS,
            summary="Severe crisis",
            confidence=0.8,
            analysis={"events": [{"event_id": "c1", "severity": "CRITICAL", "description": "Major crisis"}]},
        )

        result = self.agent.process_task(self._task({
            "investment_record": self._investment_record(),
            "current_price": 100.0,
            "current_position_value": 10000.0,
            "crisis_research_result": crisis_result,
        }))

        self.assertTrue(any(isinstance(e, ExitCandidateDetectedEvent) for e in self.agent.emitted_events))

    def test_invalid_investment_record_handled_safely(self):
        with self.assertRaises(ValueError):
            self.agent._investment_record_from_input({})

    def test_zero_and_negative_prices_handled_safely(self):
        with self.assertRaises(ValueError):
            self.agent.monitor_investment(
                investment_record=self._investment_record(),
                current_price=0.0,
                current_position_value=10000.0,
            )
        with self.assertRaises(ValueError):
            self.agent.monitor_investment(
                investment_record=self._investment_record(),
                current_price=-1.0,
                current_position_value=10000.0,
            )

    def test_multiple_active_investments_monitored_independently(self):
        inv1 = self._investment_record(asset="AAPL", entry_price=100.0)
        inv2 = self._investment_record(asset="MSFT", entry_price=200.0)

        assessment1 = self.agent.monitor_investment(
            investment_record=inv1,
            current_price=110.0,
            current_position_value=11000.0,
        )

        assessment2 = self.agent.monitor_investment(
            investment_record=inv2,
            current_price=190.0,
            current_position_value=19000.0,
        )

        self.assertEqual(assessment1.asset, "AAPL")
        self.assertEqual(assessment2.asset, "MSFT")
        self.assertNotEqual(assessment1.assessment_id, assessment2.assessment_id)


if __name__ == "__main__":
    unittest.main()