"""Regression tests: operating costs must survive a round-trip through the
generation manager on BOTH storage backends.

These exist because the LLM cost-accounting integration previously appeared to
work (add_operating_cost returned True) while the cost was silently discarded:
in-memory lost the whole generation state, and SQLite dropped the cost record
so the survival math never saw the spend.
"""

import tempfile
import unittest
from pathlib import Path

from app.core.memory.sqlite_store import SqliteMemoryStore
from app.core.models.generation import OperatingCost
from app.core.runtime.bootstrap import build_test_system
from app.utils.time import now_utc


def _cost(amount: float = 0.10, cost_id: str = "c1") -> OperatingCost:
    return OperatingCost(
        cost_id=cost_id,
        cost_type="API",
        amount=amount,
        currency="USD",
        frequency="ONE_TIME",
        description="llm call",
        provider="gemini:gemini-2.0-flash",
        timestamp=now_utc(),
    )


class _RoundTripMixin:
    """Shared assertions, run against each storage backend."""

    generation_manager = None

    def test_cost_is_persisted_and_capital_reduced(self):
        state = self.generation_manager.create_generation()
        gid = state.generation_id
        starting = self.generation_manager.get_generation(gid).current_capital

        self.assertTrue(self.generation_manager.add_operating_cost(gid, _cost(0.10)))

        reread = self.generation_manager.get_generation(gid)
        self.assertIsNotNone(reread, "generation state was lost after adding a cost")
        self.assertEqual(len(reread.operating_costs), 1)
        self.assertAlmostEqual(reread.total_operating_costs, 0.10, places=6)
        self.assertAlmostEqual(reread.current_capital, starting - 0.10, places=6)
        self.assertIsInstance(reread.operating_costs[0], OperatingCost)
        self.assertEqual(reread.operating_costs[0].cost_id, "c1")

    def test_multiple_costs_accumulate(self):
        state = self.generation_manager.create_generation()
        gid = state.generation_id
        starting = self.generation_manager.get_generation(gid).current_capital

        for i in range(3):
            self.generation_manager.add_operating_cost(gid, _cost(0.05, f"c{i}"))

        reread = self.generation_manager.get_generation(gid)
        self.assertEqual(len(reread.operating_costs), 3)
        self.assertAlmostEqual(reread.total_operating_costs, 0.15, places=6)
        self.assertAlmostEqual(reread.current_capital, starting - 0.15, places=6)

    def test_unknown_generation_is_rejected(self):
        self.assertFalse(self.generation_manager.add_operating_cost("gen_missing", _cost()))


class TestInMemoryOperatingCosts(_RoundTripMixin, unittest.TestCase):
    def setUp(self):
        self.generation_manager = build_test_system().runtime.generation_manager


class TestSqliteOperatingCosts(_RoundTripMixin, unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SqliteMemoryStore(
            str(Path(self._tmp.name) / "gen_costs.db")
        )
        self.generation_manager = build_test_system().runtime.generation_manager
        self.generation_manager.memory_store = self.store

    def tearDown(self):
        # Windows keeps the file locked while the connection is open.
        self.store.close()
        self._tmp.cleanup()


class TestOperatingCostRevivalRobustness(unittest.TestCase):
    """The parser must accept both dicts and already-built cost objects."""

    def test_parser_accepts_constructed_object(self):
        gm = build_test_system().runtime.generation_manager
        state = gm.create_generation()
        state.operating_costs = [_cost(0.25)]
        state.total_operating_costs = 0.25
        gm._store_generation_state(state)

        reread = gm.get_generation(state.generation_id)
        self.assertIsNotNone(reread)
        self.assertEqual(len(reread.operating_costs), 1)
        self.assertAlmostEqual(reread.total_operating_costs, 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
