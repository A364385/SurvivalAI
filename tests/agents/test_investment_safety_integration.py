import unittest
from datetime import timedelta

from app.agents.investment_safety.agent import InvestmentSafetyManagerAgent
from app.agents.market_research.agent import MarketResearchAgent
from app.agents.news_research.agent import NewsResearchAgent
from app.agents.crisis_risk.agent import CrisisRiskAgent
from app.core.memory.store import InMemoryStore
from app.core.models.agent import AgentResult, AgentStatus
from app.core.models.memory import InvestmentRecord, InvestmentStatus
from app.core.models.task import Task
from app.services.llm.mock_provider import MockLLMProvider
from app.services.market_data.mock_provider import MockMarketDataProvider
from app.core.models.market import Bar
from app.services.news.mock_provider import MockNewsProvider
from app.core.models.news import NewsItem
from app.utils.time import now_utc
from app.utils.ids import generate_id


class TestInvestmentSafetyIntegration(unittest.TestCase):
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
        self.safety_agent = InvestmentSafetyManagerAgent(
            agent_id="safety_int",
            llm_provider=self.llm,
            memory_store=self.memory,
        )

    def _investment_record(self, asset="AAPL", entry_price=100.0, thesis="Strong growth expected from AI product cycle"):
        return InvestmentRecord(
            investment_id="inv_int_1",
            asset=asset,
            entry_price=entry_price,
            entry_timestamp=self.now - timedelta(days=30),
            position_size=100.0,
            investment_thesis=thesis,
            time_horizon="LONG_TERM",
            risk_level="MODERATE",
            originating_generation="gen_1",
            status=InvestmentStatus.OPEN,
        )

    def _task(self, input_data):
        return Task(
            task_id="safety_int_task",
            requesting_agent="CEOAgent",
            target_agent="InvestmentSafetyManagerAgent",
            task_type="investment_safety_assessment",
            priority=1,
            input_data=input_data,
            created_at=self.now,
        )

    def test_integration_market_research_safety_manager(self):
        market_data = MockMarketDataProvider()
        market_data.set_bars("AAPL", [
            Bar("AAPL", self.now - timedelta(days=i), 100.0 + i, 102.0 + i, 98.0 + i, 100.0 + i, 1000000 + i)
            for i in range(30)
        ])

        market_agent = MarketResearchAgent("mkt_int", market_data, self.llm)
        market_result = market_agent.process_task(self._task({
            "symbol": "AAPL",
            "generation_id": "gen_int",
        }))

        safety_result = self.safety_agent.process_task(self._task({
            "investment_record": self._investment_record(),
            "current_price": 110.0,
            "current_position_value": 11000.0,
            "market_research_result": market_result,
            "generation_id": "gen_int",
        }))

        self.assertEqual(safety_result.status, AgentStatus.SUCCESS)
        self.assertIn("safety_assessment", safety_result.analysis)

    def test_integration_news_research_safety_manager(self):
        news_data = MockNewsProvider()
        news_item = NewsItem(
            news_id=generate_id("news"),
            timestamp=self.now - timedelta(hours=2),
            headline="Apple announces strong quarterly earnings",
            summary="Strong earnings beat expectations",
            source="Mock Wire",
            url="https://example.com/aapl",
            related_symbols=["AAPL"]
        )
        news_data.add_news_item(news_item)

        news_agent = NewsResearchAgent("news_int", news_data, self.llm, self.memory)
        news_result = news_agent.process_task(self._task({
            "symbol": "AAPL",
            "limit": 5,
            "generation_id": "gen_int",
        }))

        safety_result = self.safety_agent.process_task(self._task({
            "investment_record": self._investment_record(),
            "current_price": 105.0,
            "current_position_value": 10500.0,
            "news_research_result": news_result,
            "generation_id": "gen_int",
        }))

        self.assertEqual(safety_result.status, AgentStatus.SUCCESS)
        self.assertIn("safety_assessment", safety_result.analysis)

    def test_integration_crisis_research_safety_manager(self):
        crisis_agent = CrisisRiskAgent("crisis_int", self.llm, memory_store=self.memory)
        crisis_result = crisis_agent.process_task(self._task({
            "news_items": [],
            "generation_id": "gen_int",
        }))

        safety_result = self.safety_agent.process_task(self._task({
            "investment_record": self._investment_record(),
            "current_price": 100.0,
            "current_position_value": 10000.0,
            "crisis_research_result": crisis_result,
            "generation_id": "gen_int",
        }))

        self.assertEqual(safety_result.status, AgentStatus.SUCCESS)
        self.assertIn("safety_assessment", safety_result.analysis)

    def test_integration_multiple_research_sources_safety_manager(self):
        market_data = MockMarketDataProvider()
        market_data.set_bars("AAPL", [
            Bar("AAPL", self.now - timedelta(days=i), 100.0 + i, 102.0 + i, 98.0 + i, 100.0 + i, 1000000 + i)
            for i in range(30)
        ])

        news_data = MockNewsProvider()
        news_item = NewsItem(
            news_id=generate_id("news"),
            timestamp=self.now - timedelta(hours=1),
            headline="Apple announces new product launch",
            summary="New AI product announced",
            source="Mock Wire",
            url="https://example.com/aapl",
            related_symbols=["AAPL"]
        )
        news_data.add_news_item(news_item)

        market_agent = MarketResearchAgent("mkt_int", market_data, self.llm)
        news_agent = NewsResearchAgent("news_int", news_data, self.llm, self.memory)

        market_result = market_agent.process_task(self._task({
            "symbol": "AAPL",
            "generation_id": "gen_int",
        }))

        news_result = news_agent.process_task(self._task({
            "symbol": "AAPL",
            "limit": 5,
            "generation_id": "gen_int",
        }))

        safety_result = self.safety_agent.process_task(self._task({
            "investment_record": self._investment_record(),
            "current_price": 115.0,
            "current_position_value": 11500.0,
            "market_research_result": market_result,
            "news_research_result": news_result,
            "generation_id": "gen_int",
        }))

        self.assertEqual(safety_result.status, AgentStatus.SUCCESS)
        self.assertIn("safety_assessment", safety_result.analysis)

    def test_original_thesis_preserved_across_integration(self):
        original_thesis = "Strong growth expected from AI product cycle"
        inv_record = self._investment_record(thesis=original_thesis)

        safety_result = self.safety_agent.process_task(self._task({
            "investment_record": inv_record,
            "current_price": 100.0,
            "current_position_value": 10000.0,
            "generation_id": "gen_int",
        }))

        assessment = safety_result.analysis["safety_assessment"]
        self.assertEqual(assessment.original_thesis, original_thesis)

    def test_safety_manager_cannot_execute_orders_in_integration(self):
        for method in (
            self.safety_agent.execute_order,
            self.safety_agent.modify_portfolio,
            self.safety_agent.modify_capital,
        ):
            with self.assertRaises(PermissionError):
                method()


if __name__ == "__main__":
    unittest.main()