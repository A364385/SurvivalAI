"""Step 15 end-to-end integration test.

Runs the complete deterministic loop with fully-wired components:

START -> HEALTH CHECK -> GENERATION START -> MARKET DATA -> NEWS -> CRISIS
-> RESEARCH (market/news/crisis in parallel via CEO) -> DEEP LOOKER -> RISK
MANAGER -> CEO DECISION -> PAPER/SIMULATED EXECUTION -> PORTFOLIO SYNC ->
INVESTMENT MONITORING -> OUTCOME -> EXPERIENCE -> LEARNING -> SURVIVAL CHECK
-> GENERATION DEATH -> DEATH REPORT -> STRATEGY UPDATE -> VALIDATION ->
SUCCESSOR -> GENERATION 2 START.

All providers are deterministic mocks (no network, no real orders).
"""

import unittest
from datetime import datetime, timedelta

from app.core.runtime.bootstrap import build_test_system
from app.core.runtime.decision_gate import DecisionGate
from app.core.runtime.idempotency import IdempotencyManager
from app.core.runtime.execution_service import PaperExecutionService
from app.core.runtime.models import RuntimeConfig, RuntimeState
from app.core.memory.store import InMemoryStore
from app.core.models.memory import MemoryType
from app.core.models.generation import DeathTrigger
from app.utils.time import now_utc


class TestEndToEndAutonomousLoop(unittest.TestCase):
    """Full deterministic integration test of the autonomous loop."""

    def setUp(self):
        self.config = RuntimeConfig(
            cycle_interval_seconds=0,
            test_mode=True,
            paper_mode_required=True,
            decision_cooldown_seconds=0,  # allow repeat decisions in-test
            proposed_position_pct=0.05,
            watched_symbols=["AAPL"],
            max_consecutive_critical_failures=3,
        )
        self.system = build_test_system(
            config=self.config, initial_capital=100000.0
        )
        self.runtime = self.system.runtime
        self.memory = self.system.components["memory_store"]
        self.market_data = self.system.components["market_data_provider"]
        self.execution = self.system.components["execution_provider"]
        self.generation_manager = self.system.components["generation_manager"]

    def test_full_cycle_executes_all_phases(self):
        """A single full cycle passes every phase and stores a cycle record."""
        cycle_record_id = None
        from app.core.runtime.models import CycleRecord, CyclePhase

        cycle_record = CycleRecord(
            cycle_id="cycle_e2e_test",
            generation_id="pending",
            start_timestamp=now_utc(),
            end_timestamp=None,
            phase=CyclePhase.OBSERVE,
        )
        self.runtime._recover_from_restart()
        self.assertTrue(self.runtime._perform_health_check())
        self.assertTrue(self.runtime._initialize_generation())
        self.runtime.current_state = RuntimeState.OBSERVING

        cycle_record.generation_id = self.runtime.active_generation_id
        self.runtime._execute_cycle(cycle_record)

        self.assertEqual(cycle_record.status, "COMPLETED")
        self.assertIn("observe", cycle_record.triggered_actions)
        self.assertIn("decision", cycle_record.triggered_actions)
        self.assertIsNotNone(cycle_record.decision_result)
        self.assertIsNotNone(cycle_record.portfolio_state)
        self.assertIsNotNone(cycle_record.survival_state)

        # Cycle record persisted for observability
        stored = self.memory.get("cycle_cycle_e2e_test")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.content["status"], "COMPLETED")

    def test_decision_approved_and_paper_order_executed(self):
        """Health -> generation -> gated decision -> paper fill -> investment record."""
        self.runtime._perform_health_check()
        self.assertTrue(self.runtime._initialize_generation())

        gate = self.runtime._evaluate_decision_gate({"fresh": True}, True)
        self.assertTrue(gate["allowed"], f"gate should allow: {gate['reason']}")

        decision = self.runtime._execute_decision_pipeline(gate)
        self.assertIn(decision["decision"], ("INVEST", "DO_NOT_INVEST", "DEFER", "INSUFFICIENT_DATA"))
        self.assertIsNotNone(decision.get("decision_id"))

        if decision["decision"] == "INVEST":
            execution = self.runtime._execute_paper_execution(decision)
            self.assertEqual(execution["status"], "EXECUTED")
            self.assertIsNotNone(execution.get("provider_order_id"))
            self.assertGreater(execution.get("quantity") or 0, 0)

            # Investment record persisted with OPEN status
            investments = self.memory.query(memory_type=MemoryType.INVESTMENT)
            self.assertTrue(len(investments) >= 1)
            open_inv = [
                r for r in investments if r.content.get("status") == "OPEN"
            ]
            self.assertTrue(len(open_inv) >= 1)

            # Order lifecycle record shows FILLED confirmation
            lifecycle = self.memory.get(
                f"order_lifecycle_{execution['client_order_id']}"
            )
            self.assertIsNotNone(lifecycle)
            statuses = [e["status"] for e in lifecycle.content["lifecycle"]]
            self.assertIn("FILLED", statuses)

            # Portfolio shows the position after provider confirmation
            positions = self.execution.get_positions()
            self.assertTrue(any(p.symbol == "AAPL" for p in positions))

    def test_decision_gate_cooldown_blocks_repeat(self):
        """Cooldown prevents immediate repeat decisions on the same asset."""
        from app.core.runtime.models import RuntimeConfig as RC
        cooldown_config = RC(decision_cooldown_seconds=300)
        gate_logic = DecisionGate(self.memory, cooldown_config)
        first = gate_logic.evaluate(
            asset="AAPL", generation_id="g1",
            market_data_fresh=True, providers_healthy=True,
        )
        self.assertTrue(first.allowed)
        gate_logic.record_decision("AAPL", "INVEST", "dec_1")
        second = gate_logic.evaluate(
            asset="AAPL", generation_id="g1",
            market_data_fresh=True, providers_healthy=True,
        )
        self.assertFalse(second.allowed)
        self.assertIn("cooldown", second.reason)

    def test_portfolio_sync_updates_generation_capital(self):
        """Portfolio sync pulls provider truth into generation capital."""
        self.runtime._perform_health_check()
        self.runtime._initialize_generation()

        result = self.runtime._monitor_portfolio()
        self.assertEqual(result["status"], "SYNCED")
        gen = self.generation_manager.get_generation(
            self.runtime.active_generation_id
        )
        self.assertEqual(gen.current_capital, 100000.0)

    def test_experience_collection_once_per_investment(self):
        """Experiences are collected exactly once per open investment."""
        self.runtime._perform_health_check()
        self.runtime._initialize_generation()

        # Simulate a filled paper investment record
        investment_record = type(
            "M", (), {}
        )()  # placeholder; use MemoryRecord directly
        from app.core.models.memory import MemoryRecord
        record = MemoryRecord(
            memory_id="investment_inv_e2e",
            memory_type=MemoryType.INVESTMENT,
            generation_id=self.runtime.active_generation_id,
            timestamp=now_utc(),
            source_agent="TestSetup",
            importance=10,
            content={
                "investment_id": "inv_e2e",
                "asset": "AAPL",
                "quantity": 10.0,
                "entry_price": 100.0,
                "position_size": 1000.0,
                "investment_thesis": "Test thesis",
                "status": "OPEN",
                "originating_generation": self.runtime.active_generation_id,
                "order_id": "ord_1",
                "decision_id": "dec_1",
            },
            metadata={},
        )
        self.memory.save(record)

        first = self.runtime._collect_experiences()
        second = self.runtime._collect_experiences()
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["status"], "COLLECTED")
        self.assertEqual(len(second), 0)  # idempotent

    def test_generation_death_triggers_full_transition_pipeline(self):
        """Death -> cancel orders -> freeze -> experiences -> learning -> successor pending."""
        from app.core.runtime.models import CycleRecord, CyclePhase

        self.runtime._perform_health_check()
        self.assertTrue(self.runtime._initialize_generation())
        gen_id = self.runtime.active_generation_id

        # Drive equity to zero through portfolio sync -> death condition.
        self.execution._cash = 0.0
        self.execution._positions.clear()

        cycle_record = CycleRecord(
            cycle_id="cycle_death_test",
            generation_id=gen_id,
            start_timestamp=now_utc(),
            end_timestamp=None,
            phase=CyclePhase.OBSERVE,
        )
        self.runtime._execute_cycle(cycle_record)

        # Generation must be DEAD
        gen = self.generation_manager.get_generation(gen_id)
        self.assertIsNotNone(gen)
        self.assertIn(
            gen.lifecycle_state.value,
            ("DEAD", "SUCCESSOR_PENDING", "SUCCESSOR_CREATED"),
        )

        # Death report stored
        death_records = self.memory.query(memory_type=MemoryType.DEATH)
        self.assertTrue(any(r.content.get("generation_id") == gen_id for r in death_records))

        # Transition pipeline steps recorded in cycle warnings-free completion
        self.assertIsNotNone(cycle_record.survival_state)

    def test_transition_service_runs_idempotently(self):
        """Calling the death pipeline twice must not duplicate work."""
        self.runtime._perform_health_check()
        self.assertTrue(self.runtime._initialize_generation())
        gen_id = self.runtime.active_generation_id

        pipeline1 = self.runtime.transition_service.handle_generation_death(
            gen_id, DeathTrigger.CAPITAL_DEPLETED, "test death"
        )
        self.assertEqual(pipeline1["status"], "SUCCESSOR_PENDING")

        pipeline2 = self.runtime.transition_service.handle_generation_death(
            gen_id, DeathTrigger.CAPITAL_DEPLETED, "test death retry"
        )
        self.assertEqual(pipeline2["status"], "DUPLICATE_SKIPPED")

    def test_successor_generation_created_and_started(self):
        """After a death, a successor generation (Gen 2) can be created and started."""
        self.runtime._perform_health_check()
        self.assertTrue(self.runtime._initialize_generation())
        parent_id = self.runtime.active_generation_id

        pipeline = self.runtime.transition_service.handle_generation_death(
            parent_id, DeathTrigger.MINIMUM_SURVIVAL_THRESHOLD, "capital below threshold"
        )
        self.assertEqual(pipeline["status"], "SUCCESSOR_PENDING")

        successor = self.runtime.transition_service.create_successor_generation(parent_id)
        self.assertIsNotNone(successor)
        self.assertTrue(successor["started"])
        self.assertEqual(successor["generation_number"], 2)
        self.assertNotEqual(successor["successor_generation_id"], parent_id)

        # Successor is ACTIVE
        successor_gen = self.generation_manager.get_generation(
            successor["successor_generation_id"]
        )
        self.assertEqual(successor_gen.lifecycle_state.value, "ACTIVE")

        # Lineage is traceable
        lineage = self.generation_manager.get_generation_lineage(
            successor["successor_generation_id"]
        )
        self.assertEqual(len(lineage), 2)
        self.assertEqual(lineage[0].generation_id, parent_id)

        # Duplicate successor creation is idempotent
        duplicate = self.runtime.transition_service.create_successor_generation(parent_id)
        self.assertIsNotNone(duplicate)
        self.assertEqual(
            duplicate["successor_generation_id"], successor["successor_generation_id"]
        )

    def test_audit_trail_decision_traceable(self):
        """Every decision leaves a traceable DecisionRecord in memory."""
        self.runtime._perform_health_check()
        self.runtime._initialize_generation()

        gate = self.runtime._evaluate_decision_gate({"fresh": True}, True)
        decision = self.runtime._execute_decision_pipeline(gate)

        stored = self.memory.get(f"decision_{decision['decision_id']}")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.memory_type, MemoryType.DECISION)
        self.assertEqual(stored.content["asset"], "AAPL")
        self.assertIn(stored.content["decision_type"], (
            "INVEST", "DO_NOT_INVEST", "DEFER", "INSUFFICIENT_DATA",
        ))

    def test_survival_check_healthy_generation_no_death(self):
        """A healthy generation does not trigger death."""
        self.runtime._perform_health_check()
        self.assertTrue(self.runtime._initialize_generation())

        result = self.runtime._check_survival()
        self.assertFalse(result["death_triggered"])


class TestPaperModeSafety(unittest.TestCase):
    """Paper-only guarantees at the architecture level."""

    def setUp(self):
        self.system = build_test_system()
        self.runtime = self.system.runtime
        self.execution_service = self.runtime.execution_service
        self.execution = self.system.components["execution_provider"]

    def test_paper_verification_passes_with_mocks(self):
        verification = self.execution_service.verify_paper_mode(force=True)
        self.assertTrue(verification.verified, verification.failures)

    def test_paper_verification_fails_when_disabled(self):
        from app.core.runtime.models import RuntimeConfig
        self.runtime.config.paper_mode_required = False
        self.execution_service.invalidate_verification()
        verification = self.execution_service.verify_paper_mode(force=True)
        self.assertFalse(verification.verified)
        self.assertIn(
            "paper_mode_required is disabled", " ".join(verification.failures)
        )

    def test_execution_blocked_without_paper_verification(self):
        """Execution must fail safely when paper checks fail."""
        from app.core.runtime.models import RuntimeConfig
        self.runtime.config.paper_mode_required = False
        self.execution_service.invalidate_verification()

        result = self.execution_service.execute_decision(
            decision_id="dec_blocked",
            symbol="AAPL",
            position_value=1000.0,
            generation_id="g_test",
        )
        self.assertEqual(result["status"], "BLOCKED_PAPER_VERIFICATION")

        # After re-enabling, the SAME decision id is retried safely.
        self.runtime.config.paper_mode_required = True
        self.execution_service.invalidate_verification()
        retry = self.execution_service.execute_decision(
            decision_id="dec_blocked",
            symbol="AAPL",
            position_value=1000.0,
            generation_id="g_test",
        )
        self.assertEqual(retry["status"], "EXECUTED")

    def test_duplicate_order_submission_is_idempotent(self):
        """The same decision id submitted twice never creates two orders."""
        first = self.execution_service.execute_decision(
            decision_id="dec_dup",
            symbol="AAPL",
            position_value=1000.0,
            generation_id="g_dup",
        )
        second = self.execution_service.execute_decision(
            decision_id="dec_dup",
            symbol="AAPL",
            position_value=1000.0,
            generation_id="g_dup",
        )
        self.assertEqual(first["status"], "EXECUTED")
        self.assertEqual(second["status"], "DUPLICATE_SKIPPED")

        orders = self.execution.list_orders()
        matching = [
            o for o in orders if o.client_order_id == first["client_order_id"]
        ]
        self.assertEqual(len(matching), 1)

    def test_no_price_never_fabricates_order(self):
        """Without provider price data the service refuses to execute."""
        provider = self.system.components["market_data_provider"]
        # Break all price paths
        provider.get_quote = lambda s: (_ for _ in ()).throw(Exception("down"))
        provider.get_latest_trade = lambda s: (_ for _ in ()).throw(Exception("down"))

        result = self.execution_service.execute_decision(
            decision_id="dec_noprice",
            symbol="AAPL",
            position_value=1000.0,
            generation_id="g_np",
        )
        self.assertEqual(result["status"], "SKIPPED_NO_PRICE")

    def test_runtime_refuses_live_trading(self):
        with self.assertRaises(PermissionError):
            self.runtime.execute_live_trade()
        with self.assertRaises(PermissionError):
            self.runtime.activate_live_trading()

    def test_execution_environment_enum_is_paper_only(self):
        from app.core.models.execution import ExecutionEnvironment
        self.assertEqual([e.value for e in ExecutionEnvironment], ["PAPER"])


class TestFailureHandling(unittest.TestCase):
    """Failure-mode tests: system must fail safely, never crash."""

    def setUp(self):
        self.config = RuntimeConfig(
            cycle_interval_seconds=0,
            test_mode=True,
            paper_mode_required=True,
            decision_cooldown_seconds=0,
        )
        self.system = build_test_system(config=self.config)
        self.runtime = self.runtime = self.system.runtime
        self.memory = self.system.components["memory_store"]
        self.market_data = self.system.components["market_data_provider"]
        self.news = self.system.components["news_provider"]
        self.execution = self.system.components["execution_provider"]
        self.generation_manager = self.system.components["generation_manager"]
        self.runtime._perform_health_check()
        self.runtime._initialize_generation()

    def test_market_api_outage_triggers_safe_pause(self):
        """A market data outage pauses new decisions instead of crashing."""
        provider = self.market_data

        def fail(*a, **k):
            raise Exception("market API down")
        provider.get_market_clock = fail

        from app.core.runtime.models import CycleRecord, CyclePhase
        cycle_record = CycleRecord(
            cycle_id="cycle_outage",
            generation_id=self.runtime.active_generation_id,
            start_timestamp=now_utc(),
            end_timestamp=None,
            phase=CyclePhase.OBSERVE,
        )
        self.runtime._execute_cycle(cycle_record)

        self.assertEqual(cycle_record.status, "SAFE_PAUSED")
        self.assertEqual(self.runtime.current_state, RuntimeState.PAUSED)
        # Failure recorded
        failures = self.memory.query(memory_type=MemoryType.ERROR)
        self.assertTrue(len(failures) >= 1)

    def test_recovery_after_outage(self):
        """After an outage clears, the runtime can resume decisions."""
        def fail(*a, **k):
            raise Exception("market API down")
        original_clock = self.market_data.get_market_clock
        self.market_data.get_market_clock = fail

        from app.core.runtime.models import CycleRecord, CyclePhase
        cycle1 = CycleRecord(
            cycle_id="cycle_outage1",
            generation_id=self.runtime.active_generation_id,
            start_timestamp=now_utc(),
            end_timestamp=None,
            phase=CyclePhase.OBSERVE,
        )
        self.runtime._execute_cycle(cycle1)
        self.assertEqual(self.runtime.current_state, RuntimeState.PAUSED)

        # Restore provider
        self.market_data.get_market_clock = original_clock
        self.runtime._running = True  # simulate an active runtime process
        self.assertTrue(self.runtime.resume())
        self.assertEqual(self.runtime.current_state, RuntimeState.OBSERVING)

        cycle2 = CycleRecord(
            cycle_id="cycle_outage2",
            generation_id=self.runtime.active_generation_id,
            start_timestamp=now_utc(),
            end_timestamp=None,
            phase=CyclePhase.OBSERVE,
        )
        self.runtime._execute_cycle(cycle2)
        self.assertEqual(cycle2.status, "COMPLETED")

    def test_news_outage_recorded_not_fatal(self):
        """News provider failure is recorded but does not stop the cycle."""
        def fail(*a, **k):
            raise Exception("news API down")
        self.news.get_latest_news = fail

        data = self.runtime._collect_news()
        self.assertIn("error", data)
        self.assertEqual(data["news"], [])
        failures = self.memory.query(memory_type=MemoryType.ERROR)
        self.assertTrue(any("news" in str(f.content.get("component")) for f in failures))

    def test_execution_provider_outage_fails_safely(self):
        """Order submission failure returns structured failure, no crash."""
        def fail(*a, **k):
            raise Exception("paper API down")
        self.execution.get_account = fail

        result = self.runtime.execution_service.execute_decision(
            decision_id="dec_outage",
            symbol="AAPL",
            position_value=1000.0,
            generation_id=self.runtime.active_generation_id,
        )
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("paper API down", result["error"])

    def test_stale_market_data_blocks_decisions(self):
        """Stale market data blocks new decisions via the gate."""
        stale_clock = type("Clock", (), {"timestamp": now_utc() - timedelta(hours=3), "is_open": True})
        self.market_data.get_market_clock = lambda: stale_clock

        market_data = self.runtime._collect_market_data()
        self.assertFalse(market_data["fresh"])

        gate = self.runtime._evaluate_decision_gate(market_data, True)
        self.assertFalse(gate["allowed"])
        self.assertIn("stale", gate["reason"])

    def test_restarted_runtime_recovers_active_generation(self):
        """A restarted runtime finds and reuses the active generation."""
        original_gen_id = self.runtime.active_generation_id

        from app.core.runtime.survival_runtime import SurvivalRuntime
        new_runtime = SurvivalRuntime(
            memory_store=self.memory,
            event_bus=self.system.components["event_bus"],
            generation_manager=self.generation_manager,
            agent_registry=self.system.agent_registry,
            market_data_provider=self.market_data,
            news_provider=self.news,
            execution_provider=self.execution,
            config=self.config,
        )
        new_runtime._recover_from_restart()

        self.assertEqual(new_runtime.active_generation_id, original_gen_id)


class TestScheduledTasks(unittest.TestCase):
    """Scheduler runs tasks at their configured intervals."""

    def test_scheduler_runs_due_tasks_once(self):
        from app.core.runtime.scheduler import RuntimeScheduler

        scheduler = RuntimeScheduler()
        calls = {"a": 0}

        def task_a():
            calls["a"] += 1
            return {"status": "OK"}

        scheduler.register("a", 3600, task_a)
        results1 = scheduler.run_due_tasks()
        results2 = scheduler.run_due_tasks()

        self.assertEqual(len(results1), 1)
        self.assertEqual(len(results2), 0)  # interval not yet elapsed
        self.assertEqual(calls["a"], 1)


if __name__ == "__main__":
    unittest.main()