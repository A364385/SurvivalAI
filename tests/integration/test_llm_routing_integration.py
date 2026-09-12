"""Integration test: real agent traffic must flow through the router.

This is the end-to-end proof that the router is not just a dashboard
decoration: it boots the actual test system, binds the router (as
`run_local.py` does), runs a real agent task, and asserts the call was routed
with the correct role and recorded in usage/cost accounting.
"""

import json
import os
import unittest
from unittest.mock import patch

from app.core.models.task import Task
from app.services.llm.model_router import RoleBoundProvider, RouterLLMProvider
from app.services.llm.usage_tracker import LLMUsageTracker
from app.utils.time import now_utc


def _claude_response(payload_text: str, input_tokens: int = 800,
                     output_tokens: int = 200):
    return {
        "content": [{"type": "text", "text": payload_text}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


class TestRouterBindsToRealAgents(unittest.TestCase):
    def test_system_agents_are_rebound_to_router(self):
        from app.core.runtime.bootstrap import build_test_system

        result = build_test_system()
        runtime = result.runtime
        registry = runtime.agent_registry
        agent_ids = registry.list_agents()
        self.assertGreaterEqual(len(agent_ids), 8)

        tracker = LLMUsageTracker()
        router = RouterLLMProvider(usage_tracker=tracker,
                                   generation_id_resolver=lambda: "gen_e2e")
        rebound = router.bind_to_agents(registry)
        self.assertEqual(rebound, len(agent_ids))

        # Every agent now holds a role-pinned router view, not a raw provider.
        for agent_id in agent_ids:
            provider = registry.retrieve(agent_id).llm_provider
            self.assertIsInstance(provider, RoleBoundProvider, agent_id)
            self.assertEqual(provider.role, agent_id)

    def test_market_agent_task_routes_and_records_usage(self):
        from app.core.runtime.bootstrap import build_test_system

        result = build_test_system()
        runtime = result.runtime
        registry = runtime.agent_registry

        tracker = LLMUsageTracker()
        router = RouterLLMProvider(usage_tracker=tracker,
                                   generation_id_resolver=lambda: "gen_e2e")
        router.bind_to_agents(registry)

        agent = registry.retrieve("market_research")
        self.assertIsNotNone(agent)

        task = Task(
            task_id="task_router_e2e",
            requesting_agent="CEOAgent",
            target_agent="MarketResearchAgent",
            task_type="market_research",
            priority=1,
            input_data={"symbol": "AAPL", "generation_id": "gen_e2e"},
            created_at=now_utc(),
        )
        agent.process_task(task)

        summary = tracker.summary()
        self.assertGreaterEqual(
            summary["overall"]["calls"], 1,
            "agent traffic never reached the router",
        )
        # Attribution must name the real provider/model, never an anonymous
        # placeholder (this assertion previously encoded the old defect).
        providers = " ".join(summary["by_model"].keys())
        self.assertIn("mock:deterministic", providers)
        self.assertNotIn("unknown", providers)

    def test_usage_records_carry_role_and_generation(self):
        from app.core.runtime.bootstrap import build_test_system

        from app.core.memory.store import InMemoryStore

        result = build_test_system()
        registry = result.runtime.agent_registry

        tracker = LLMUsageTracker(memory_store=InMemoryStore())
        router = RouterLLMProvider(usage_tracker=tracker,
                                   generation_id_resolver=lambda: "gen_e2e")
        router.bind_to_agents(registry)

        agent = registry.retrieve("market_research")
        task = Task(
            task_id="task_router_role",
            requesting_agent="CEOAgent",
            target_agent="MarketResearchAgent",
            task_type="market_research",
            priority=1,
            input_data={"symbol": "AAPL"},
            created_at=now_utc(),
        )
        agent.process_task(task)

        records = tracker.query_recent(limit=10)
        self.assertTrue(records)
        roles = {r.get("role") for r in records}
        self.assertIn("market_research", roles)
        generations = {r.get("generation_id") for r in records}
        self.assertIn("gen_e2e", generations)


class TestBuildLlmStackIsTheRealPath(unittest.TestCase):
    """`bootstrap.build_llm_stack` is what `scripts/run_local.py` calls, so
    these tests exercise the shipping configuration rather than a copy of it.
    """

    def _system(self):
        from app.core.memory.store import InMemoryStore
        from app.core.runtime.bootstrap import build_test_system

        result = build_test_system()
        store = InMemoryStore()
        result.runtime.memory_store = store
        return result, store

    def test_default_config_binds_agents_and_names_usage(self):
        from app.core.runtime.bootstrap import build_llm_stack

        result, store = self._system()
        runtime = result.runtime
        stack = build_llm_stack(runtime, store, runtime.generation_manager,
                                result.components)

        # Default environment: no provider configured -> deterministic mock.
        self.assertEqual(stack.provider_type, "")
        self.assertEqual(stack.provider_state, "fallback")
        self.assertIn("mock", stack.provider_detail)

        agent_ids = runtime.agent_registry.list_agents()
        self.assertEqual(stack.rebound_agents, len(agent_ids))
        for agent_id in agent_ids:
            provider = runtime.agent_registry.retrieve(agent_id).llm_provider
            self.assertIsInstance(provider, RoleBoundProvider, agent_id)

        # The components dict is repointed too, so nothing keeps the raw mock.
        self.assertIs(result.components["llm_provider"], stack.router)
        self.assertIs(runtime.llm_provider, stack.router)

        # Real agent traffic through the real factory records named usage.
        runtime.agent_registry.retrieve("news_research").llm_provider.generate("hi")
        self.assertIn("mock:deterministic", stack.usage_tracker.summary()["by_model"])

    def test_remote_provider_without_credentials_degrades_safely(self):
        from app.core.runtime.bootstrap import build_llm_stack

        result, store = self._system()
        runtime = result.runtime
        with patch.dict("os.environ", {"SURVIVALAI_LLM_PROVIDER": "gemini"},
                        clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            stack = build_llm_stack(runtime, store, runtime.generation_manager,
                                    result.components)
        self.assertEqual(stack.provider_state, "failed")
        self.assertIn("fallback", stack.provider_detail)
        # Still usable: the deterministic mock answers.
        self.assertTrue(stack.router.generate("hi"))

    def test_unreachable_local_provider_degrades_safely(self):
        from app.core.runtime.bootstrap import build_llm_stack

        result, store = self._system()
        runtime = result.runtime
        with patch.dict("os.environ", {
            "SURVIVALAI_LLM_PROVIDER": "lm_studio",
            "SURVIVALAI_LLM_ENDPOINT": "http://127.0.0.1:1",
        }, clear=False):
            stack = build_llm_stack(runtime, store, runtime.generation_manager,
                                    result.components)
        self.assertEqual(stack.provider_state, "unavailable")
        self.assertTrue(stack.router.generate("hi"))


class TestLlmCostReachesGeneration(unittest.TestCase):
    """The headline integration: a real agent LLM call must show up as an
    operating cost on the active generation (i.e. inside the survival math).
    """

    ANALYSIS = {
        "facts": [],
        "analysis": {"interpretation": "x", "event_type": "y",
                     "importance": 0.5},
        "impact": {"direction": "neutral", "horizon": "short_term"},
        "confidence": 0.5,
        "warnings": [],
    }

    def test_agent_call_is_attributed_to_generation(self):
        from app.core.memory.store import InMemoryStore
        from app.core.runtime.bootstrap import build_llm_stack, build_test_system

        bootstrap = build_test_system()
        runtime = bootstrap.runtime
        store = InMemoryStore()
        runtime.memory_store = store
        gm = runtime.generation_manager
        generation = gm.create_generation()
        gid = generation.generation_id
        runtime.active_generation_id = gid

        # Build the stack exactly as the launcher does, then point it at a
        # remote provider (the offline default is covered elsewhere).
        stack = build_llm_stack(runtime, store, gm, bootstrap.components)
        tracker = stack.usage_tracker
        router = stack.router
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            self.assertTrue(router.configure_remote_provider("claude")["ok"])

        agent = runtime.agent_registry.retrieve("market_research")
        task = Task(
            task_id="task_cost_attribution",
            requesting_agent="CEOAgent",
            target_agent="MarketResearchAgent",
            task_type="market_research",
            priority=1,
            input_data={"symbol": "AAPL", "generation_id": gid},
            created_at=now_utc(),
        )
        response = _claude_response(json.dumps(self.ANALYSIS))
        with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
            agent.process_task(task)

        reread = gm.get_generation(gid)
        self.assertIsNotNone(reread, "generation state was lost")
        self.assertGreater(reread.total_operating_costs, 0.0)
        self.assertEqual(len(reread.operating_costs), 1)
        cost = reread.operating_costs[0]
        self.assertEqual(cost.cost_type, "API")
        self.assertIn("claude", cost.provider)
        self.assertLess(reread.current_capital, generation.starting_capital)


if __name__ == "__main__":
    unittest.main()
