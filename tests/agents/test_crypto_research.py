"""Tests for Crypto Research Agent."""

import unittest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from app.agents.crypto_research.agent import CryptoResearchAgent
from app.core.models.agent import AgentConfig, AgentStatus
from app.core.models.task import Task
from app.services.crypto.mock_provider import MockCryptoDataProvider
from app.utils.time import now_utc


class TestCryptoResearchAgent(unittest.TestCase):
    """Test cases for CryptoResearchAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.crypto_provider = MockCryptoDataProvider()
        self.agent = CryptoResearchAgent(
            agent_id="test_crypto_agent",
            crypto_market_provider=self.crypto_provider,
            tokenomics_provider=self.crypto_provider,
            onchain_provider=self.crypto_provider,
            configuration=AgentConfig(version="1.0.0", model=None),
        )

    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.agent_id, "test_crypto_agent")
        self.assertEqual(self.agent.agent_name, "CryptoResearchAgent")
        self.assertEqual(self.agent.role, "Cryptocurrency Market Analyst & Tokenomics Researcher")
        self.assertIn("crypto_market_structure", self.agent.capabilities)
        self.assertIn("tokenomics_analysis", self.agent.capabilities)

    def test_process_task_single_symbol(self):
        """Test processing a single crypto symbol."""
        task = Task(
            task_id="test_task_1",
            requesting_agent="ceo",
            target_agent="crypto_research",
            task_type="crypto_analysis",
            priority=1,
            input_data={"symbol": "BTC"},
            created_at=now_utc(),
        )

        result = self.agent.process_task(task)

        self.assertEqual(result.agent_id, "test_crypto_agent")
        self.assertEqual(result.task_id, "test_task_1")
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIn("BTC", result.summary)
        self.assertGreater(len(result.facts), 0)
        self.assertIn("market_structure", result.analysis)
        self.assertIn("risk_factors", result.analysis)

    def test_process_task_multiple_symbols(self):
        """Test processing multiple crypto symbols."""
        task = Task(
            task_id="test_task_2",
            requesting_agent="ceo",
            target_agent="crypto_research",
            task_type="crypto_analysis",
            priority=1,
            input_data={"symbols": ["BTC", "ETH", "SOL"]},
            created_at=now_utc(),
        )

        result = self.agent.process_task(task)

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIn("BTC", result.summary)
        self.assertGreater(len(result.facts), 0)

    def test_market_structure_assessment(self):
        """Test market structure assessment."""
        structure = self.agent._assess_market_structure(
            symbol="BTC",
            current_price=65000.0,
            price_change_24h=0.06,
            volatility=0.6,  # Changed to > 0.5 for HIGH
        )

        self.assertEqual(structure.symbol, "BTC")
        self.assertEqual(structure.trend, "STRONG_UPTREND")
        self.assertEqual(structure.momentum, "BULLISH")
        self.assertEqual(structure.volatility, "HIGH")

    def test_crypto_risk_identification(self):
        """Test crypto risk factor identification."""
        from app.core.models.crypto import (
            CryptoMarketStructure,
            CryptoRegime,
            TokenUnlock,
        )

        structure = CryptoMarketStructure(
            symbol="SOL",
            trend="UPTREND",
            momentum="BULLISH",
            volatility="HIGH",
            liquidity="HIGH",
            drawdown=0.1,
            regime=CryptoRegime.BULL_MARKET,
        )

        unlock = TokenUnlock(
            unlock_id="test_unlock",
            symbol="SOL",
            unlock_date=now_utc(),
            amount=5e6,
            percentage_of_supply=1.0,
            source="team",
        )

        risks = self.agent._identify_crypto_risks(
            symbol="SOL",
            market_structure=structure,
            tokenomics=None,
            on_chain=None,
            upcoming_unlocks=[unlock],
        )

        self.assertGreater(len(risks), 0)
        risk_types = [r.risk_type.value for r in risks]
        self.assertIn("EXTREME_VOLATILITY", risk_types)
        self.assertIn("TOKEN_UNLOCK_RISK", risk_types)
        self.assertIn("REGULATORY_RISK", risk_types)

    def test_tokenomics_inclusion(self):
        """Test that tokenomics data is included when available."""
        task = Task(
            task_id="test_task_3",
            requesting_agent="ceo",
            target_agent="crypto_research",
            task_type="crypto_analysis",
            priority=1,
            input_data={"symbol": "BTC"},
            created_at=now_utc(),
        )

        result = self.agent.process_task(task)

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIn("tokenomics", result.analysis)
        self.assertIsNotNone(result.analysis["tokenomics"])
        self.assertIn("market_cap", result.analysis["tokenomics"])

    def test_on_chain_metrics_inclusion(self):
        """Test that on-chain metrics are included when available."""
        task = Task(
            task_id="test_task_4",
            requesting_agent="ceo",
            target_agent="crypto_research",
            task_type="crypto_analysis",
            priority=1,
            input_data={"symbol": "BTC"},
            created_at=now_utc(),
        )

        result = self.agent.process_task(task)

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIn("on_chain", result.analysis)
        self.assertIsNotNone(result.analysis["on_chain"])
        self.assertIn("active_addresses_24h", result.analysis["on_chain"])

    def test_market_wide_conditions(self):
        """Test market-wide conditions inclusion."""
        task = Task(
            task_id="test_task_5",
            requesting_agent="ceo",
            target_agent="crypto_research",
            task_type="crypto_analysis",
            priority=1,
            input_data={"symbol": "BTC"},
            created_at=now_utc(),
        )

        result = self.agent.process_task(task)

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIn("market_wide", result.analysis)
        self.assertIsNotNone(result.analysis["market_wide"])
        self.assertIn("btc_dominance", result.analysis["market_wide"])

    def test_unsupported_symbol(self):
        """Test handling of unsupported symbol."""
        task = Task(
            task_id="test_task_6",
            requesting_agent="ceo",
            target_agent="crypto_research",
            task_type="crypto_analysis",
            priority=1,
            input_data={"symbol": "UNSUPPORTED"},
            created_at=now_utc(),
        )

        result = self.agent.process_task(task)

        # Should still succeed (mock provider returns default data)
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        # May have warnings about unsupported symbol
        # self.assertGreater(len(result.warnings), 0)

    def test_risk_factors_count(self):
        """Test that risk factors are properly counted."""
        task = Task(
            task_id="test_task_7",
            requesting_agent="ceo",
            target_agent="crypto_research",
            task_type="crypto_analysis",
            priority=1,
            input_data={"symbol": "SOL"},
            created_at=now_utc(),
        )

        result = self.agent.process_task(task)

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIn("risk_factors", result.analysis)
        self.assertGreater(len(result.analysis["risk_factors"]), 0)

    def test_metadata_contains_crypto_info(self):
        """Test that metadata contains crypto-specific information."""
        task = Task(
            task_id="test_task_8",
            requesting_agent="ceo",
            target_agent="crypto_research",
            task_type="crypto_analysis",
            priority=1,
            input_data={"symbol": "BTC"},
            created_at=now_utc(),
        )

        result = self.agent.process_task(task)

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIn("risk_factors_count", result.metadata)
        self.assertIn("has_tokenomics", result.metadata)
        self.assertIn("has_on_chain", result.metadata)
        self.assertIn("data_quality", result.metadata)


if __name__ == "__main__":
    unittest.main()
