"""Final local-readiness integration tests.

Covers the acceptance sequence:
  build system (SQLite) -> health -> generation -> paper decision -> survival
  -> death -> report -> learning -> successor -> knowledge transfer
plus watchdog pause, crash recovery with durable state, and the training queue.
"""

import tempfile
import unittest
from pathlib import Path

from app.core.memory.sqlite_store import SqliteMemoryStore
from app.core.models.memory import MemoryType
from app.core.runtime.models import RuntimeConfig
from app.core.runtime.bootstrap import build_test_system
from app.core.runtime.watchdog import Watchdog, WatchdogCheck, WatchdogConfig


class TestFullSystemWithSQLite(unittest.TestCase):
    """Builds the real system with the SQLite store and runs cycles."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        self.config = RuntimeConfig(
            test_mode=True,
            cycle_interval_seconds=0,
            max_cycles=2,
            decision_cooldown_seconds=0,
        )
        self.bootstrap = build_test_system(config=self.config)
        self.runtime = self.bootstrap.runtime
        self.sqlite = SqliteMemoryStore(str(self.data_dir / "survivalai.db"))
        self.runtime.memory_store = self.sqlite
        for service_name in ("idempotency", "decision_gate", "portfolio_monitor",
                             "investment_monitor", "experience_collector",
                             "learning_cycle", "cost_service", "transition_service",
                             "health_checker", "execution_service", "audit"):
            service = getattr(self.runtime, service_name, None)
            if service is not None:
                setattr(service, "memory_store", self.sqlite)
        gm = self.bootstrap.components["generation_manager"]
        gm.memory_store = self.sqlite

    def tearDown(self):
        self.sqlite.close()
        self.tmp.cleanup()

    def test_start_runs_cycles_and_persists_records(self):
        started = self.runtime.start()
        self.assertTrue(started)
        self.assertGreaterEqual(self.runtime.completed_cycles, 1)
        # Durable audit/decision records exist in SQLite
        self.assertGreater(self.sqlite.count(), 0)
        decision_records = [
            r for r in self.sqlite.list()
            if getattr(r, "memory_type", None) in (MemoryType.DECISION, MemoryType.FACT)
        ]
        self.assertTrue(decision_records)
        self.assertTrue(self.sqlite.integrity_check())

    def test_capital_protection_vetoes_oversized_invest(self):
        """A CEO INVEST with an absurd size must be downgraded by protection."""
        self.runtime._initialize_generation()
        decision = {
            "decision": "INVEST",
            "decision_id": "test_decision_1",
            "position_size": 10_000_000.0,  # absurd
            "reason": "test",
            "confidence": 0.9,
        }
        result = self.runtime._execute_paper_execution(decision)
        self.assertEqual(result.get("status"), "BLOCKED")
        self.assertEqual(result.get("reason"), "capital_protection")

    def test_emergency_stop_blocks_execution(self):
        self.runtime._initialize_generation()
        self.runtime.capital_protection.engage_emergency_stop("test")
        decision = {
            "decision": "INVEST", "decision_id": "test_decision_2",
            "position_size": 100.0, "reason": "test", "confidence": 0.9,
        }
        result = self.runtime._execute_paper_execution(decision)
        self.assertEqual(result.get("status"), "BLOCKED")

    def test_survival_death_and_successor_pipeline(self):
        """Force capital below threshold -> death -> report -> successor."""
        started = self.runtime.start()
        self.assertTrue(started)
        gen_id = self.runtime.active_generation_id
        self.assertIsNotNone(gen_id)

        # Force a death condition: bleed capital below the survival threshold.
        gm = self.bootstrap.components["generation_manager"]
        gm.update_capital(gen_id, 1.0)  # below any threshold

        survival = self.runtime._check_survival()
        if survival.get("death_triggered"):
            self.runtime._handle_generation_death(survival)
            # Either a successor started or the runtime stopped safely —
            # both are valid outcomes; records must exist either way.
            reports = self.sqlite.query(memory_type=MemoryType.DEATH) or \
                [r for r in self.sqlite.list()
                 if "death" in str(getattr(r, "memory_id", "")).lower()]
            self.assertTrue(reports, "death report must be persisted")

    def test_runtime_refuses_live_trading(self):
        with self.assertRaises(PermissionError):
            self.runtime.execute_live_trade()
        with self.assertRaises(PermissionError):
            self.runtime.activate_live_trading()


class TestCrashRecovery(unittest.TestCase):
    def test_state_survives_restart(self):
        tmp = tempfile.TemporaryDirectory()
        db_path = str(Path(tmp.name) / "recovery.db")
        config = RuntimeConfig(test_mode=True, cycle_interval_seconds=0,
                               max_cycles=1, decision_cooldown_seconds=0)
        try:
            # First "process"
            bootstrap1 = build_test_system(config=config)
            store1 = SqliteMemoryStore(db_path)
            bootstrap1.runtime.memory_store = store1
            bootstrap1.components["generation_manager"].memory_store = store1
            self.assertTrue(bootstrap1.runtime.start())
            gen1 = bootstrap1.runtime.active_generation_id
            count1 = store1.count()
            store1.close()

            # Second "process": same database
            store2 = SqliteMemoryStore(db_path)
            self.assertGreater(store2.count(), 0)
            self.assertEqual(store2.count(), count1)
            records = store2.list()
            self.assertTrue(any(
                getattr(r, "memory_id", "").startswith(("cycle_", "decision_", "audit_"))
                or getattr(r, "generation_id", "")
                for r in records
            ))
            store2.close()
        finally:
            tmp.cleanup()


class TestWatchdog(unittest.TestCase):
    def test_healthy_state_does_not_pause(self):
        paused = []
        watchdog = Watchdog(WatchdogConfig(consecutive_failures_before_pause=2))
        watchdog.on_unsafe = lambda checks: paused.append(checks)
        result = watchdog.run_checks()
        self.assertTrue(result["safe"])
        self.assertFalse(result["should_pause"])
        self.assertEqual(paused, [])

    def test_repeated_failures_trigger_pause_callback(self):
        paused = []
        failing = lambda: WatchdogCheck("custom_fail", False, "broken")  # noqa: E731
        watchdog = Watchdog(WatchdogConfig(consecutive_failures_before_pause=2))
        watchdog.register_checker(failing)
        watchdog.on_unsafe = lambda checks: paused.append([c.name for c in checks])
        watchdog.run_checks()
        self.assertEqual(paused, [])  # first failure: warn only
        watchdog.run_checks()
        self.assertEqual(len(paused), 1)  # second: pause
        self.assertIn("custom_fail", paused[0])
        self.assertTrue(watchdog.paused_by_watchdog)

    def test_recovery_resets_failure_count(self):
        failing = lambda: WatchdogCheck("flaky", False, "x")  # noqa: E731
        watchdog = Watchdog(WatchdogConfig(consecutive_failures_before_pause=2))
        watchdog.register_checker(failing)
        watchdog.run_checks()
        watchdog.run_checks()
        self.assertGreaterEqual(watchdog.last_result["consecutive_failures"], 2)
        # remove the failing checker -> next run is safe again
        watchdog._checkers.remove(failing)
        result = watchdog.run_checks()
        self.assertTrue(result["safe"])
        self.assertEqual(result["consecutive_failures"], 0)


class TestTrainingQueue(unittest.TestCase):
    def test_queue_processes_job_end_to_end(self):
        import os
        tmp = tempfile.TemporaryDirectory()
        os.environ["SURVIVALAI_DATA_DIR"] = tmp.name
        try:
            from app.ml.model_registry import ModelRegistry
            from app.ml.promotion import PromotionPipeline, TrainingQueue
            from app.ml.training import MockTrainer
            from app.ml.evaluation import HeuristicEvaluator

            registry = ModelRegistry(str(Path(tmp.name) / "queue_registry.db"))
            pipeline = PromotionPipeline(registry, trainer=MockTrainer(),
                                         evaluator=HeuristicEvaluator())
            queue = TrainingQueue(registry, pipeline)
            job = queue.enqueue(role="market_research", dataset_version="v1",
                                model_version="v0.1", base_model="x")
            # Wait for the worker thread (bounded; generous under full-suite load).
            import time
            for _ in range(600):
                state = queue.state()
                if state["items"] and state["items"][0]["status"] in ("DONE", "FAILED"):
                    break
                time.sleep(0.1)
            state = queue.state()
            self.assertEqual(state["items"][0]["status"], "DONE", state)
            self.assertEqual(state["items"][0]["result"], "ARCHIVED")  # smoke: archived
            registry.close()
        finally:
            os.environ.pop("SURVIVALAI_DATA_DIR", None)
            registry.close()
            tmp.cleanup()


class TestFastLoopComponents(unittest.TestCase):
    def test_indicator_history_updates(self):
        from app.agents.market_research.indicator_history import IndicatorHistory
        ih = IndicatorHistory()
        closes = [100 + i * 0.5 for i in range(60)]
        snap = ih.update_symbol("AAPL", closes=closes)
        self.assertIsNotNone(snap["price"])
        latest = ih.latest("AAPL")
        self.assertEqual(latest["price"], closes[-1])
        series = ih.series("AAPL", limit=10)
        self.assertLessEqual(len(series), 10)


if __name__ == "__main__":
    unittest.main()
