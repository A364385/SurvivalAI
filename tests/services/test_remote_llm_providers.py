"""Tests for remote LLM providers (Gemini/Claude): payloads, retries,
schema-guided structured output, usage/cost accounting, credential fail-safe.

All HTTP is mocked — no network in unit tests.
"""

import json
import time
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from app.core.models.provider_errors import InvalidResponseError
from app.services.llm.remote_providers import (
    ClaudeProvider, GeminiProvider, RemoteLLMError,
    available_remote_providers, build_remote_provider,
)
from app.services.llm.schema_validator import validate_against_schema
from app.services.llm.usage_tracker import (
    LLMUsageTracker, UsageRecord, estimate_cost,
)


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, code: int, body: str = "error"):
        super().__init__("http://test", code, "err", hdrs=None, fp=None)
        self._body = body

    def read(self):
        return self._body.encode()



class TestEstimateCost(unittest.TestCase):
    def test_known_model_priced(self):
        cost = estimate_cost("gemini", "gemini-2.0-flash", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.10 + 0.40)

    def test_unknown_model_free(self):
        self.assertEqual(estimate_cost("gemini", "unknown-model", 1000, 1000), 0.0)

    def test_proportional(self):
        self.assertAlmostEqual(
            estimate_cost("claude", "claude-3-5-haiku-20241022", 500_000, 0),
            0.40, places=6,
        )


class TestUsageTracker(unittest.TestCase):
    def test_record_and_summary(self):
        tracker = LLMUsageTracker()
        tracker.record(UsageRecord(
            provider="gemini", model="gemini-2.0-flash", role="news_research",
            input_tokens=1000, output_tokens=500, estimated_cost_usd=0.0003,
            latency_ms=120.0, success=True,
        ))
        tracker.record(UsageRecord(
            provider="gemini", model="gemini-2.0-flash", role="news_research",
            input_tokens=2000, output_tokens=100, estimated_cost_usd=0.0006,
            latency_ms=150.0, success=False, error="HTTP 429",
        ))
        summary = tracker.summary()
        self.assertEqual(summary["overall"]["calls"], 2)
        self.assertEqual(summary["overall"]["errors"], 1)
        self.assertEqual(summary["overall"]["input_tokens"], 3000)
        self.assertAlmostEqual(summary["overall"]["estimated_cost_usd"], 0.0009)

    def test_cost_attributed_to_active_generation(self):
        """Estimated API cost must land in the generation's operating costs."""

        class _FakeManager:
            def __init__(self):
                self.calls = []

            def add_operating_cost(self, generation_id, cost):
                self.calls.append((generation_id, cost))

        manager = _FakeManager()
        tracker = LLMUsageTracker(
            generation_manager=manager,
            generation_id_resolver=lambda: "gen_42",
        )
        tracker.record(UsageRecord(
            provider="gemini", model="gemini-2.0-flash", role="ceo",
            input_tokens=1_000_000, output_tokens=0,
            estimated_cost_usd=estimate_cost("gemini", "gemini-2.0-flash",
                                             1_000_000, 0),
            latency_ms=10.0, success=True,
        ))
        self.assertEqual(len(manager.calls), 1)
        generation_id, cost = manager.calls[0]
        self.assertEqual(generation_id, "gen_42")
        self.assertAlmostEqual(cost.amount, 0.10, places=6)
        self.assertEqual(cost.cost_type, "API")

    def test_zero_cost_records_are_not_attached(self):
        class _FakeManager:
            def __init__(self):
                self.calls = []

            def add_operating_cost(self, generation_id, cost):
                self.calls.append((generation_id, cost))

        manager = _FakeManager()
        tracker = LLMUsageTracker(generation_manager=manager,
                                  generation_id_resolver=lambda: "gen_1")
        tracker.record(UsageRecord(
            provider="gemini", model="unknown-model", role="ceo",
            input_tokens=1000, output_tokens=10, estimated_cost_usd=0.0,
            latency_ms=10.0, success=True,
        ))
        self.assertEqual(manager.calls, [])

    def test_no_generation_id_means_no_attachment(self):
        class _FakeManager:
            def __init__(self):
                self.calls = []

            def add_operating_cost(self, generation_id, cost):
                self.calls.append((generation_id, cost))

        manager = _FakeManager()
        tracker = LLMUsageTracker(generation_manager=manager)
        tracker.record(UsageRecord(
            provider="gemini", model="gemini-2.0-flash", role="ceo",
            input_tokens=1000, output_tokens=10, estimated_cost_usd=0.001,
            latency_ms=10.0, success=True,
        ))
        self.assertEqual(manager.calls, [])

    def test_persists_to_memory_store(self):
        from app.core.memory.store import InMemoryStore
        from app.core.models.memory import MemoryType
        store = InMemoryStore()
        tracker = LLMUsageTracker(memory_store=store)
        tracker.record(UsageRecord(
            provider="claude", model="claude-3-5-haiku-20241022", role="ceo",
            input_tokens=100, output_tokens=50, estimated_cost_usd=0.00028,
            latency_ms=90.0, success=True,
        ))
        records = store.query(memory_type=MemoryType.FACT)
        self.assertTrue(any(
            getattr(r, "content", {}).get("usage_id") for r in records
        ))


class TestGeminiProvider(unittest.TestCase):
    def _provider(self, tracker=None):
        return GeminiProvider(api_key="fake-key", usage_tracker=tracker)

    def test_missing_key_refused(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RemoteLLMError):
                GeminiProvider()

    def test_payload_and_endpoint(self):
        provider = self._provider()
        payload = provider._request_payload("hello", "system", 100, 0.5)
        self.assertIn("system_instruction", payload)
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 100)
        endpoint = provider._endpoint()
        self.assertIn("generativelanguage.googleapis.com", endpoint)
        # Credentials must never be carried in the URL.
        self.assertNotIn("fake-key", endpoint)
        self.assertNotIn("key=", endpoint)
        self.assertEqual(provider._headers()["x-goog-api-key"], "fake-key")

    def test_generate_parses_and_tracks_usage(self):
        tracker = LLMUsageTracker()
        provider = self._provider(tracker)
        response = {
            "candidates": [{"content": {"parts": [{"text": "hello world"}]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5},
        }
        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            text = provider.generate("hi", system_prompt="sys")
        self.assertEqual(text, "hello world")
        summary = tracker.summary()
        self.assertEqual(summary["overall"]["calls"], 1)
        self.assertEqual(summary["overall"]["input_tokens"], 12)
        self.assertEqual(summary["overall"]["output_tokens"], 5)
        self.assertGreater(summary["overall"]["estimated_cost_usd"], 0.0)

    def test_structured_output_schema_guided(self):
        provider = self._provider()
        good = {"facts": [], "analysis": {}, "impact": {}, "confidence": 0.5, "warnings": []}
        response = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(good)}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 40},
        }
        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            result = provider.generate_structured("analyze", schema={"type": "object"})
        self.assertEqual(result["confidence"], 0.5)

    def test_transient_429_retries_then_succeeds(self):
        tracker = LLMUsageTracker()
        provider = self._provider(tracker)
        provider.max_retries = 3
        good = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
        }
        responses = [
            _FakeHTTPError(429, "rate limited"),
            _FakeHTTPError(429, "rate limited"),
            _FakeResponse(good),
        ]
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(1)
            item = responses.pop(0)
            if isinstance(item, _FakeHTTPError):
                raise item
            return item

        with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("time.sleep") as fake_sleep:
            text = provider.generate("hi")
        self.assertEqual(text, "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(fake_sleep.call_count, 2)
        summary = tracker.summary()
        # Every attempt is recorded: 2 failed retries + 1 success.
        self.assertEqual(summary["overall"]["calls"], 3)
        self.assertEqual(summary["overall"]["errors"], 2)

    def test_non_retryable_400_fails_fast(self):
        provider = self._provider()
        provider.max_retries = 3
        with patch("urllib.request.urlopen", side_effect=_FakeHTTPError(400, "bad request")), \
             patch("time.sleep") as fake_sleep:
            with self.assertRaises(RemoteLLMError):
                provider.generate("hi")
        fake_sleep.assert_not_called()

    def test_exhausted_retries_raises(self):
        provider = self._provider()
        provider.max_retries = 2
        with patch("urllib.request.urlopen", side_effect=_FakeHTTPError(503, "down")), \
             patch("time.sleep"):
            with self.assertRaises(RemoteLLMError):
                provider.generate("hi")

    def test_blocked_prompt_raises_clean_error(self):
        provider = self._provider()
        response = {"promptFeedback": {"blockReason": "SAFETY"}}
        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            with self.assertRaises(RemoteLLMError):
                provider.generate("hi")


class TestClaudeProvider(unittest.TestCase):
    def _provider(self, tracker=None):
        return ClaudeProvider(api_key="fake-key", usage_tracker=tracker)

    def test_missing_key_refused(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RemoteLLMError):
                ClaudeProvider()

    def test_payload_headers_and_endpoint(self):
        provider = self._provider()
        payload = provider._request_payload("hello", "system", 100, 0.5)
        self.assertEqual(payload["model"], "claude-3-5-haiku-20241022")
        self.assertEqual(payload["system"], "system")
        self.assertEqual(payload["max_tokens"], 100)
        self.assertIn("api.anthropic.com", provider._endpoint())
        headers = provider._headers()
        self.assertEqual(headers["x-api-key"], "fake-key")
        self.assertIn("anthropic-version", headers)

    def test_generate_parses_and_tracks_usage(self):
        tracker = LLMUsageTracker()
        provider = self._provider(tracker)
        response = {
            "content": [{"type": "text", "text": "answer"}],
            "usage": {"input_tokens": 21, "output_tokens": 9},
        }
        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            text = provider.generate("hi")
        self.assertEqual(text, "answer")
        summary = tracker.summary()
        self.assertEqual(summary["overall"]["input_tokens"], 21)
        self.assertEqual(summary["overall"]["output_tokens"], 9)

    def test_structured_output(self):
        provider = self._provider()
        good = {"facts": [], "analysis": {}, "impact": {}, "confidence": 0.8, "warnings": []}
        response = {
            "content": [{"type": "text", "text": json.dumps(good)}],
            "usage": {"input_tokens": 10, "output_tokens": 50},
        }
        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            result = provider.generate_structured("analyze", schema={"type": "object"})
        self.assertEqual(result["confidence"], 0.8)

    def test_retry_on_500_then_success(self):
        tracker = LLMUsageTracker()
        provider = self._provider(tracker)
        provider.max_retries = 2
        good = {
            "content": [{"type": "text", "text": "recovered"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        responses = [_FakeHTTPError(500, "server error"), _FakeResponse(good)]

        def fake_urlopen(req, timeout=None):
            item = responses.pop(0)
            if isinstance(item, _FakeHTTPError):
                raise item
            return item

        with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("time.sleep"):
            text = provider.generate("hi")
        self.assertEqual(text, "recovered")
        self.assertEqual(tracker.summary()["overall"]["errors"], 1)


class TestCredentialsNeverInUrl(unittest.TestCase):
    """The outgoing request must carry auth in headers, not the query string.

    URLs are far more likely to end up in logs, proxies and tracebacks, so the
    key must not be observable there for either provider.
    """

    def _capture_request(self, provider):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["request"] = req
            raise urllib.error.URLError("stop after capture")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RemoteLLMError):
                provider.generate("hi")
        return captured["request"]

    def test_gemini_key_is_a_header_not_a_query_param(self):
        provider = GeminiProvider(api_key="SUPERSECRET")
        request = self._capture_request(provider)
        self.assertNotIn("SUPERSECRET", request.full_url)
        self.assertNotIn("key=", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "SUPERSECRET")

    def test_claude_key_is_a_header_not_a_query_param(self):
        provider = ClaudeProvider(api_key="SUPERSECRET")
        request = self._capture_request(provider)
        self.assertNotIn("SUPERSECRET", request.full_url)
        self.assertEqual(request.get_header("X-api-key"), "SUPERSECRET")
        self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")

    def test_error_text_never_contains_the_key(self):
        provider = GeminiProvider(api_key="SUPERSECRET", max_retries=1)
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("boom")):
            with self.assertRaises(RemoteLLMError) as ctx:
                provider.generate("hi")
        self.assertNotIn("SUPERSECRET", str(ctx.exception))


class TestFactoryAndEnv(unittest.TestCase):
    def test_factory_unknown_type(self):
        with self.assertRaises(RemoteLLMError):
            build_remote_provider("openai_skynet")

    def test_factory_gemini_without_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RemoteLLMError):
                build_remote_provider("gemini")

    def test_available_remote_providers_masks_keys(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "supersecret"}):
            available = available_remote_providers()
            self.assertIn("gemini", available)
            self.assertEqual(available["gemini"], "configured")
            self.assertNotIn("supersecret", json.dumps(available))


class TestSchemaValidation(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "required": ["facts", "analysis", "impact", "confidence", "warnings"],
        "properties": {
            "facts": {"type": "array"},
            "analysis": {"type": "object"},
            "impact": {"type": "object"},
            "confidence": {"type": "number"},
            "warnings": {"type": "array"},
        },
    }

    def test_missing_required_field_rejected(self):
        with self.assertRaises(InvalidResponseError):
            validate_against_schema({"facts": []}, self.SCHEMA)

    def test_wrong_type_rejected(self):
        data = {"facts": "not-a-list", "analysis": {}, "impact": {},
                "confidence": 0.5, "warnings": []}
        with self.assertRaises(InvalidResponseError):
            validate_against_schema(data, self.SCHEMA)

    def test_bool_not_accepted_as_number(self):
        data = {"facts": [], "analysis": {}, "impact": {},
                "confidence": True, "warnings": []}
        with self.assertRaises(InvalidResponseError):
            validate_against_schema(data, self.SCHEMA)

    def test_valid_payload_passes(self):
        data = {"facts": [], "analysis": {}, "impact": {},
                "confidence": 0.5, "warnings": []}
        self.assertIs(validate_against_schema(data, self.SCHEMA), data)

    def test_empty_schema_is_permissive(self):
        data = {"anything": 1}
        self.assertIs(validate_against_schema(data, {}), data)


class TestStructuredValidationRetry(unittest.TestCase):
    """The remote provider must not hand malformed output to agents."""

    SCHEMA = TestSchemaValidation.SCHEMA

    def _gemini(self, tracker=None):
        return GeminiProvider(api_key="fake-key", usage_tracker=tracker)

    def _response(self, payload_text: str, in_tokens=10, out_tokens=20):
        return {
            "candidates": [{"content": {"parts": [{"text": payload_text}]}}],
            "usageMetadata": {"promptTokenCount": in_tokens,
                              "candidatesTokenCount": out_tokens},
        }

    def test_invalid_then_repaired_on_retry(self):
        provider = self._gemini()
        bad = json.dumps({"facts": []})  # missing required fields
        good = json.dumps({"facts": [], "analysis": {}, "impact": {},
                           "confidence": 0.4, "warnings": []})
        responses = [_FakeResponse(self._response(bad)),
                     _FakeResponse(self._response(good))]

        def fake_urlopen(req, timeout=None):
            return responses.pop(0)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = provider.generate_structured("analyze", self.SCHEMA)
        self.assertEqual(result["confidence"], 0.4)

    def test_persistently_invalid_raises(self):
        provider = self._gemini()
        bad = json.dumps({"facts": []})

        def fake_urlopen(req, timeout=None):
            return _FakeResponse(self._response(bad))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RemoteLLMError):
                provider.generate_structured("analyze", self.SCHEMA)

    def test_non_json_raises(self):
        provider = self._gemini()

        def fake_urlopen(req, timeout=None):
            return _FakeResponse(self._response("I cannot help with that."))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RemoteLLMError):
                provider.generate_structured("analyze", self.SCHEMA)

    def test_custom_validator_is_used(self):
        provider = self._gemini()
        payload = {"ok": True}
        seen = {}

        def validator(data):
            seen["called"] = True
            return data

        def fake_urlopen(req, timeout=None):
            return _FakeResponse(self._response(json.dumps(payload)))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = provider.generate_structured("x", {}, validator=validator)
        self.assertTrue(seen.get("called"))
        self.assertEqual(result, payload)


class TestRouterRemoteIntegration(unittest.TestCase):
    """Router must expose remote providers as first-class with fallback."""

    def test_configure_remote_without_key_fails_softly(self):
        from app.services.llm.model_router import RouterLLMProvider
        with patch.dict("os.environ", {}, clear=True):
            router = RouterLLMProvider()
            result = router.configure_remote_provider("gemini")
            self.assertFalse(result["ok"])
            self.assertIn("error", result)
            # Still usable: falls back to the deterministic mock.
            output = router.generate("hi")
            self.assertIsInstance(output, str)
            self.assertTrue(output)

    def test_configure_unknown_remote_type_fails_softly(self):
        from app.services.llm.model_router import RouterLLMProvider
        router = RouterLLMProvider()
        result = router.configure_remote_provider("skynet")
        self.assertFalse(result["ok"])

    def test_gemini_routed_and_cost_tracked(self):
        from app.services.llm.model_router import RouterLLMProvider
        tracker = LLMUsageTracker()
        router = RouterLLMProvider(usage_tracker=tracker)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
            self.assertTrue(router.configure_remote_provider("gemini")["ok"])
        response = {
            "candidates": [{"content": {"parts": [{"text": "routed"}]}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50},
        }
        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            text = router.generate("hello")
        self.assertEqual(text, "routed")
        summary = tracker.summary()
        self.assertEqual(summary["overall"]["calls"], 1)
        self.assertGreater(summary["overall"]["estimated_cost_usd"], 0.0)

    def test_router_falls_back_when_remote_errors(self):
        from app.services.llm.model_router import RouterLLMProvider
        router = RouterLLMProvider()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}):
            router.configure_remote_provider("claude")
        with patch("urllib.request.urlopen",
                   side_effect=_FakeHTTPError(400, "bad request")):
            text = router.generate("hi")
        # Fallback produced text instead of propagating the failure.
        self.assertTrue(isinstance(text, str) and len(text) > 0)
        self.assertGreaterEqual(router.stats["fallbacks"], 1)

    def test_mock_calls_recorded_for_observability(self):
        """Offline (mock) traffic must still be observable in the usage log."""
        from app.services.llm.model_router import RouterLLMProvider
        tracker = LLMUsageTracker()
        router = RouterLLMProvider(usage_tracker=tracker)
        bound = router.for_role("market_research")
        output = bound.generate("hi")
        self.assertTrue(output)
        summary = tracker.summary()
        self.assertEqual(summary["overall"]["calls"], 1)
        self.assertEqual(summary["overall"]["errors"], 0)
        entry = next(iter(summary["by_model"].values()))
        self.assertGreaterEqual(entry["calls"], 1)

    def test_remote_calls_not_double_counted(self):
        from app.services.llm.model_router import RouterLLMProvider
        tracker = LLMUsageTracker()
        router = RouterLLMProvider(usage_tracker=tracker)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
            router.configure_remote_provider("gemini")
        response = {
            "candidates": [{"content": {"parts": [{"text": "x"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
        }
        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            router.generate("hi", role="ceo")
        # Exactly one record: the remote provider's own token-level record.
        self.assertEqual(tracker.summary()["overall"]["calls"], 1)

    def test_concurrent_calls_keep_their_own_role(self):
        """Regression: the CEO dispatches agents in parallel (4 workers).

        Role/generation must be per-call (thread-local), not written onto the
        shared provider instance — otherwise every concurrent call is
        attributed to whichever role stamped last.
        """
        import threading as _threading

        from app.services.llm.model_router import RouterLLMProvider

        tracker = LLMUsageTracker()
        captured: list = []
        original = tracker.record

        def spy(usage):
            captured.append((usage.role, usage.generation_id))
            return original(usage)

        tracker.record = spy
        router = RouterLLMProvider(usage_tracker=tracker,
                                   generation_id_resolver=lambda: "gen_race")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}):
            router.configure_remote_provider("claude")
        provider = router._default
        response = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

        roles = ["news_research", "market_research", "crisis_risk",
                 "deep_looker", "ceo", "crypto_research",
                 "strategy_updater", "investment_safety"]

        def slow_post(payload):
            time.sleep(0.05)  # widen the window a real network call creates
            return response

        with patch.object(provider, "_post", side_effect=slow_post):
            threads = [
                _threading.Thread(target=lambda r=r: router.generate("hi", role=r))
                for r in roles
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        recorded_roles = sorted(role for role, _ in captured)
        self.assertEqual(recorded_roles, sorted(roles))
        self.assertEqual({gen for _, gen in captured}, {"gen_race"})

    def test_remote_test_connection_returns_bool_false(self):
        """Regression: 'false' as a string is truthy and inverts the result."""
        provider = GeminiProvider(api_key="k", max_retries=1)
        with patch("urllib.request.urlopen",
                   side_effect=_FakeHTTPError(400, "bad request")):
            result = provider.test_connection()
        self.assertIs(result["ok"], False)
        self.assertFalse(result["ok"])

    def test_bind_to_agents_rebinds_every_agent_with_role_view(self):
        from app.agents.registry import AgentRegistry
        from app.services.llm.model_router import RouterLLMProvider, RoleBoundProvider

        class _Agent:
            def __init__(self, agent_id):
                self.agent_id = agent_id
                self.llm_provider = "raw"

        registry = AgentRegistry()
        for role in ("news_research", "market_research", "ceo"):
            registry.register(_Agent(role), agent_type="general")

        router = RouterLLMProvider()
        rebound = router.bind_to_agents(registry)
        self.assertEqual(rebound, 3)
        for role in ("news_research", "market_research", "ceo"):
            provider = registry.retrieve(role).llm_provider
            self.assertIsInstance(provider, RoleBoundProvider)
            self.assertEqual(provider.role, role)

    def test_role_view_routes_usage_to_real_role(self):
        """A role-pinned call must record the real role, not 'general'."""
        from app.services.llm.model_router import RouterLLMProvider
        tracker = LLMUsageTracker()
        router = RouterLLMProvider(usage_tracker=tracker,
                                   generation_id_resolver=lambda: "gen_7")
        with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
            router.configure_remote_provider("gemini")
        captured = {}

        def fake_record(usage):
            captured["role"] = usage.role
            captured["generation_id"] = usage.generation_id
            return LLMUsageTracker.record(tracker, usage)

        response = {
            "candidates": [{"content": {"parts": [{"text": "x"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
        }
        bound = router.for_role("news_research")
        with patch.object(tracker, "record", side_effect=fake_record), \
             patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            bound.generate("hi")
        self.assertEqual(captured.get("role"), "news_research")
        self.assertEqual(captured.get("generation_id"), "gen_7")

    def test_role_view_structured_passes_role(self):
        from app.services.llm.model_router import RouterLLMProvider
        router = RouterLLMProvider()
        bound = router.for_role("ceo")
        schema = {"type": "object", "required": ["confidence"],
                  "properties": {"confidence": {"type": "number"}}}
        payload = {"confidence": 0.9}
        response = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5},
        }
        with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
            router.configure_remote_provider("gemini")
        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            result = bound.generate_structured("x", schema)
        self.assertEqual(result, payload)

    def test_usage_is_persisted_to_store(self):
        from app.core.memory.store import InMemoryStore
        from app.core.models.memory import MemoryType
        from app.services.llm.model_router import RouterLLMProvider
        store = InMemoryStore()
        tracker = LLMUsageTracker(memory_store=store)
        router = RouterLLMProvider(usage_tracker=tracker)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
            router.configure_remote_provider("gemini")
        response = {
            "candidates": [{"content": {"parts": [{"text": "x"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
        }
        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            router.generate("hi")
        records = store.query(memory_type=MemoryType.FACT)
        self.assertTrue(any(getattr(r, "content", {}).get("usage_id") for r in records))

class TestOfflineAttributionAndEstimation(unittest.TestCase):
    """Regression: the default offline (mock/local) path must name the real
    provider and report honest length-derived token ESTIMATES.

    Previously these records were `router:unknown` with 0/0 tokens, which made
    by_model attribution and the usage panel useless exactly where the system
    normally runs.
    """

    def _usage_contents(self, store):
        from app.core.models.memory import MemoryType
        return [r.content for r in store.query(memory_type=MemoryType.FACT)
                if isinstance(getattr(r, "content", None), dict)
                and r.content.get("usage_id")]

    def test_mock_usage_names_provider_and_estimates_tokens(self):
        from app.core.memory.store import InMemoryStore
        from app.services.llm.model_router import RouterLLMProvider

        tracker = LLMUsageTracker(memory_store=InMemoryStore())
        router = RouterLLMProvider(usage_tracker=tracker)
        router.for_role("news_research").generate("p" * 400)

        summary = tracker.summary()
        keys = list(summary["by_model"])
        self.assertEqual(keys, ["mock:deterministic"])
        self.assertNotIn("unknown", keys[0])
        self.assertGreater(summary["overall"]["input_tokens"], 0)
        self.assertGreater(summary["overall"]["output_tokens"], 0)
        # Flagged as estimated so it is never mistaken for reported counts.
        self.assertEqual(summary["overall"]["estimated_token_calls"], 1)

    def test_persisted_mock_record_is_marked_estimated(self):
        from app.core.memory.store import InMemoryStore
        from app.services.llm.model_router import RouterLLMProvider

        store = InMemoryStore()
        router = RouterLLMProvider(usage_tracker=LLMUsageTracker(memory_store=store))
        router.for_role("risk_manager").generate("q" * 200)

        contents = self._usage_contents(store)
        self.assertEqual(len(contents), 1)
        record = contents[0]
        self.assertEqual((record["provider"], record["model"]),
                         ("mock", "deterministic"))
        self.assertEqual(record["role"], "risk_manager")
        self.assertTrue(record["tokens_estimated"])
        self.assertGreater(record["input_tokens"], 0)
        self.assertGreater(record["output_tokens"], 0)
        # Self-hosted/mock inference bills nothing, and that stays honest.
        self.assertEqual(record["estimated_cost_usd"], 0.0)

    def test_local_provider_is_named_in_usage(self):
        from app.services.llm.local_providers import LMStudioProvider
        from app.services.llm.model_router import RouterLLMProvider

        tracker = LLMUsageTracker()
        router = RouterLLMProvider(usage_tracker=tracker)
        router.register_role_provider(
            "general",
            LMStudioProvider(endpoint="http://127.0.0.1:1", model="qwen2.5-7b"),
        )
        # The call fails (nothing listening) and falls back, but the attempt
        # must still be attributed to the real local provider/model.
        router.generate("hi")
        self.assertIn("lm_studio:qwen2.5-7b", tracker.summary()["by_model"])

    def test_estimate_tokens_never_reports_a_misleading_zero(self):
        from app.services.llm.usage_tracker import estimate_tokens

        self.assertEqual(estimate_tokens(None), 0)
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("a" * 400), 100)
        self.assertEqual(estimate_tokens("ab"), 1)


class TestRoleContractDrivesValidation(unittest.TestCase):
    """Regression: the role's expected output contract must reach the provider
    so it drives validation AND the self-repair attempt.

    Previously the provider only ever applied the generic required-keys subset,
    so role-semantic violations went undetected and were never repaired.
    """

    ANALYSIS_KEYS = {"facts": [], "analysis": {"interpretation": "x",
                     "event_type": "y", "importance": 1.0},
                     "impact": {"direction": "neutral",
                                "horizon": "short_term"}, "warnings": []}

    def _payload(self, confidence):
        body = dict(self.ANALYSIS_KEYS)
        body["confidence"] = confidence
        return body

    def _router(self, tracker=None):
        from app.services.llm.model_router import RouterLLMProvider
        router = RouterLLMProvider(usage_tracker=tracker or LLMUsageTracker())
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}):
            self.assertTrue(router.configure_remote_provider("claude")["ok"])
        return router

    def _claude(self, body):
        return {
            "content": [{"type": "text", "text": json.dumps(body)}],
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }

    def test_role_violation_repairs(self):
        router = self._router()
        responses = [_FakeResponse(self._claude(self._payload(5.0))),
                     _FakeResponse(self._claude(self._payload(0.6)))]
        sent = []

        def fake_urlopen(request, timeout=None):
            sent.append(json.loads(request.data.decode("utf-8")))
            return responses.pop(0)

        # Permissive schema on purpose: only the role contract can catch this.
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = router.for_role("market_research").generate_structured(
                "analyse AAPL", schema={})

        self.assertEqual(result["confidence"], 0.6)
        self.assertEqual(len(sent), 2, "role violation did not trigger a repair")
        repair_prompt = sent[1]["messages"][0]["content"]
        self.assertIn("between 0.0 and 1.0", repair_prompt)
        self.assertIn("analyse AAPL", repair_prompt)

    def test_role_without_a_contract_accepts_the_same_payload(self):
        """Control: proves the role validator is the discriminator."""
        router = self._router()
        with patch("urllib.request.urlopen",
                   return_value=_FakeResponse(self._claude(self._payload(5.0)))):
            result = router.for_role("ceo").generate_structured(
                "orchestrate", schema={})
        self.assertEqual(result["confidence"], 5.0)

    def test_persistently_role_invalid_never_escapes_through_the_router(self):
        """After repair is exhausted the router must not hand the agent the
        role-invalid payload — it falls back to the deterministic mock."""
        router = self._router()
        with patch("urllib.request.urlopen",
                   return_value=_FakeResponse(self._claude(self._payload(9.0)))):
            result = router.for_role("market_research").generate_structured(
                "analyse", schema={})
        self.assertNotEqual(result.get("confidence"), 9.0)
        self.assertGreaterEqual(router.stats["fallbacks"], 1)

    def test_provider_raises_on_persistent_role_violation(self):
        from app.services.llm.remote_providers import ClaudeProvider
        from app.services.llm.schema_validator import ROLE_VALIDATORS

        provider = ClaudeProvider(api_key="k")
        with patch("urllib.request.urlopen",
                   return_value=_FakeResponse(self._claude(self._payload(9.0)))):
            with self.assertRaises(RemoteLLMError):
                provider.generate_structured(
                    "analyse", {}, validator=ROLE_VALIDATORS["market_research"])

    def test_role_validator_mapping_covers_analysis_roles(self):
        from app.services.llm.schema_validator import ROLE_VALIDATORS
        for role in ("news_research", "market_research", "crisis_risk"):
            self.assertIn(role, ROLE_VALIDATORS)


class TestLocalProvidersTestConnection(unittest.TestCase):
    """Round out test_connection paths on local providers."""

    def test_lm_studio_test_connection_failure_shape(self):
        from app.services.llm.local_providers import LMStudioProvider
        provider = LMStudioProvider(endpoint="http://127.0.0.1:1")
        result = provider.test_connection()
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_ollama_test_connection_failure_shape(self):
        from app.services.llm.local_providers import OllamaProvider
        provider = OllamaProvider(endpoint="http://127.0.0.1:1")
        result = provider.test_connection()
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
