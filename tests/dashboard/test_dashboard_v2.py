"""Tests for Dashboard v2: state hub, controls, provider masking, endpoints."""

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from app.dashboard.state import DashboardState, PROTECTED_COMPONENTS, _mask


class TestDashboardState(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()

    def test_portfolio_points_and_drawdown(self):
        self.state.record_portfolio_point(equity=100000, cash=100000)
        self.state.record_portfolio_point(equity=120000, cash=90000)
        self.state.record_portfolio_point(equity=90000, cash=90000)
        snap = self.state.portfolio_snapshot()
        self.assertEqual(len(snap["equity"]), 3)
        self.assertEqual(self.state.peak_equity, 120000)
        dd = snap["drawdown"][-1]["value"]
        self.assertAlmostEqual(dd, (120000 - 90000) / 120000, places=4)

    def test_flags(self):
        self.assertTrue(self.state.set_flag("fast_loop", False))
        self.assertFalse(self.state.controls_state()["fast_loop"])
        # protected components refused
        self.assertFalse(self.state.set_flag("risk_manager", False))
        self.assertFalse(self.state.set_flag("paper_trading_safety", False))

    def test_agent_toggle_protection(self):
        self.assertTrue(self.state.set_agent_enabled("news_research", False))
        self.assertFalse(self.state.agent_toggles["news_research"])
        # safety-critical cannot be toggled
        self.assertFalse(self.state.set_agent_enabled("risk_manager", False))
        self.assertFalse(self.state.set_agent_enabled("capital_protection", False))

    def test_secret_masking(self):
        self.assertEqual(_mask("abcd"), "****")
        masked = _mask("SK1234567890END")
        self.assertTrue(masked.startswith("SK"))
        self.assertTrue(masked.endswith("ND"))
        self.assertIn("*", masked)
        self.assertNotIn("1234567890", masked)

    def test_provider_settings_masked(self):
        self.state.set_provider_setting("alpaca_paper", "api_key", "SECRETVALUE123", secret=True)
        self.state.set_provider_setting("alpaca_paper", "endpoint", "https://paper-api.alpaca.markets/v2")
        masked = self.state.provider_settings_masked()
        self.assertNotIn("SECRETVALUE123", json.dumps(masked))
        self.assertTrue(masked["alpaca_paper"]["api_key__set"])
        self.assertEqual(masked["alpaca_paper"]["endpoint"], "https://paper-api.alpaca.markets/v2")

    def test_broadcast_and_subscribe(self):
        q = self.state.subscribe()
        self.state.broadcast("test_event", {"x": 1})
        payload = q.get(timeout=2)
        self.assertEqual(payload["event"], "test_event")
        self.state.unsubscribe(q)
        self.state.broadcast("no_listeners", {})
        self.assertTrue(q.empty())


class TestDashboardV2Server(unittest.TestCase):
    """Live HTTP integration test on an ephemeral port."""

    @classmethod
    def setUpClass(cls):
        from app.dashboard.api_v2 import DashboardV2
        from app.services.llm.model_router import RouterLLMProvider
        # Pass a router as a component so LLM endpoints are exercised for real.
        cls.server = DashboardV2(host="127.0.0.1", port=0, state=DashboardState(),
                                 llm_router=RouterLLMProvider())
        cls.server.httpd = ThreadingHTTPServer(("127.0.0.1", 0), cls.server.handler_cls)
        cls.port = cls.server.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.server.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.httpd.shutdown()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())

    def _post(self, path, body, expect_status=200):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def test_index_served(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=5) as resp:
            html = resp.read().decode()
        self.assertIn("SurvivalAI", html)
        self.assertIn("PAPER ONLY", html)

    def test_state_endpoint(self):
        status, data = self._get("/api/v2/state")
        self.assertEqual(status, 200)
        self.assertEqual(data["paper_only"]["mode"], "PAPER_TRADING_ONLY")
        self.assertIn("charts", data)

    def test_controls_roundtrip(self):
        status, data = self._post("/api/v2/controls", {"flag": "fast_loop", "value": False})
        self.assertTrue(data["ok"])
        self.assertFalse(data["controls"]["fast_loop"])

    def test_protected_agent_refused(self):
        status, data = self._post("/api/v2/controls", {"agent": "risk_manager", "enabled": False})
        self.assertEqual(status, 403)

    def test_provider_secrets_masked_over_http(self):
        self._post("/api/v2/providers", {"provider": "alpaca_paper",
                                         "settings": {"api_key": "TOPSECRET99"}})
        status, data = self._get("/api/v2/providers")
        self.assertNotIn("TOPSECRET99", json.dumps(data))

    def test_health_endpoint(self):
        status, data = self._get("/api/v2/health")
        self.assertEqual(status, 200)
        self.assertIn("database", data["checks"])
        self.assertIn("capital_protection", data["checks"])

    def test_models_endpoint_empty(self):
        status, data = self._get("/api/v2/models")
        self.assertEqual(status, 200)
        self.assertEqual(data["models"], [])

    def test_training_endpoint_shape(self):
        status, data = self._get("/api/v2/training")
        self.assertEqual(status, 200)
        self.assertIn("queue", data)

    def test_llm_usage_endpoint_shape(self):
        status, data = self._get("/api/v2/llm/usage")
        self.assertEqual(status, 200)
        self.assertIn("summary", data)
        self.assertIn("overall", data["summary"])
        self.assertIn("recent", data)
        self.assertIn("available_remote", data)

    def test_llm_configure_remote_without_key_fails_softly(self):
        status, data = self._post("/api/v2/llm/configure", {"provider": "gemini"})
        self.assertEqual(status, 200)
        self.assertFalse(data["ok"])
        self.assertIn("error", data)

    def test_llm_configure_mock_is_noop(self):
        status, data = self._post("/api/v2/llm/configure", {"provider": "mock"})
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])

    def test_llm_configure_unknown_provider_fails_softly(self):
        status, data = self._post("/api/v2/llm/configure", {"provider": "skynet"})
        self.assertEqual(status, 200)
        self.assertFalse(data["ok"])

    def test_llm_configure_unreachable_local_provider(self):
        # Port 1 is never listening: must report failure, not crash.
        status, data = self._post("/api/v2/llm/configure", {
            "provider": "lm_studio", "endpoint": "http://127.0.0.1:1"})
        self.assertEqual(status, 200)
        self.assertFalse(data["ok"])

    def test_llm_secrets_never_in_usage_response(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "TOPSECRETKEY"}):
            status, data = self._get("/api/v2/llm/usage")
        self.assertNotIn("TOPSECRETKEY", json.dumps(data))


if __name__ == "__main__":
    unittest.main()
