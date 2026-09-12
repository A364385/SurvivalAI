"""Tests for Dashboard API."""

import unittest
from unittest.mock import Mock

from app.dashboard.api import DashboardAPI, DashboardHandler


class TestDashboardAPI(unittest.TestCase):
    """Test cases for Dashboard API."""

    def setUp(self):
        """Set up test fixtures."""
        self.api = DashboardAPI(
            host="127.0.0.1",
            port=8080,
            memory_store=None,
            generation_manager=None,
            runtime=None,
            portfolio_sync=None,
            agent_registry=None,
        )

    def test_api_initialization(self):
        """Test API initialization."""
        self.assertEqual(self.api.host, "127.0.0.1")
        self.assertEqual(self.api.port, 8080)
        self.assertIsNone(self.api.server)


class TestDashboardHandler(unittest.TestCase):
    """Test cases for Dashboard Handler methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory_store = Mock()
        self.generation_manager = Mock()
        self.runtime = Mock()
        self.portfolio_sync = Mock()
        self.agent_registry = Mock()

    def _create_handler(self):
        """Create a handler for testing."""
        handler = DashboardHandler.__new__(DashboardHandler)
        handler.memory_store = self.memory_store
        handler.generation_manager = self.generation_manager
        handler.runtime = self.runtime
        handler.portfolio_sync = self.portfolio_sync
        handler.agent_registry = self.agent_registry
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        return handler

    def test_paper_only_warning(self):
        """Test paper-only warning is included."""
        handler = self._create_handler()
        warning = handler._get_paper_only_warning()
        self.assertEqual(warning["mode"], "PAPER_TRADING_ONLY")
        self.assertIn("enforced_by", warning)

    def test_runtime_info_without_runtime(self):
        """Test runtime info when runtime is not configured."""
        handler = self._create_handler()
        handler.runtime = None
        info = handler._get_runtime_info()
        self.assertEqual(info["state"], "UNKNOWN")
        self.assertIsNone(info["generation_id"])

    def test_runtime_info_with_runtime(self):
        """Test runtime info when runtime is configured."""
        handler = self._create_handler()
        mock_runtime = Mock()
        mock_runtime.current_state = Mock(value="ACTIVE")
        mock_runtime.active_generation_id = "gen_123"
        handler.runtime = mock_runtime

        info = handler._get_runtime_info()

        self.assertEqual(info["state"], "ACTIVE")
        self.assertEqual(info["generation_id"], "gen_123")

    def test_html_serving(self):
        """Test HTML serving."""
        handler = self._create_handler()
        html = handler._get_dashboard_html()

        self.assertIn("SurvivalAI Dashboard", html)
        self.assertIn("PAPER TRADING", html)
        self.assertIn("paper-warning", html)


if __name__ == "__main__":
    unittest.main()
