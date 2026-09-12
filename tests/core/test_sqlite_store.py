"""Tests for the SQLite MemoryStore: durability, queries, revival, integrity."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from app.core.memory.sqlite_store import SqliteMemoryStore
from app.core.models.memory import (
    DecisionRecord, Experience, GenerationStatus, InvestmentRecord,
    InvestmentStatus, MemoryRecord, MemoryType,
)


class TestSqliteStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test.db")
        self.store = SqliteMemoryStore(self.db_path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _record(self, memory_id="m1", mtype=MemoryType.FACT):
        return MemoryRecord(
            memory_id=memory_id, memory_type=mtype, generation_id="gen_1",
            timestamp=datetime(2026, 1, 1, 12, 0, 0), source_agent="test",
            importance=5, content={"key": "value", "n": 42},
            metadata={"tag": "x"},
        )

    def test_save_get_roundtrip(self):
        record = self._record()
        self.store.save(record)
        loaded = self.store.get("m1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.memory_id, "m1")
        self.assertEqual(loaded.memory_type, MemoryType.FACT)
        self.assertEqual(loaded.content["n"], 42)
        self.assertEqual(loaded.timestamp, datetime(2026, 1, 1, 12, 0, 0))

    def test_persistence_across_reopen(self):
        self.store.save(self._record("persist_1"))
        self.store.close()
        store2 = SqliteMemoryStore(self.db_path)
        try:
            loaded = store2.get("persist_1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.content["key"], "value")
        finally:
            store2.close()

    def test_query_by_attribute(self):
        self.store.save(self._record("a", MemoryType.DECISION))
        self.store.save(self._record("b", MemoryType.FACT))
        self.store.save(self._record("c", MemoryType.DECISION))
        decisions = self.store.query(memory_type=MemoryType.DECISION)
        self.assertEqual({r.memory_id for r in decisions}, {"a", "c"})
        all_m = self.store.query(source_agent="test")
        self.assertEqual(len(all_m), 3)

    def test_decision_record_roundtrip(self):
        record = DecisionRecord(
            decision_id="d1", generation_id="gen_1", decision_type="INVEST",
            asset="AAPL", timestamp=datetime(2026, 1, 2),
            reasoning_reference="run_1", supporting_agent_results=["a", "b"],
            risk_assessment={"APPROVED": True}, crisis_assessment={},
            decision={"size": 1000}, confidence=0.7,
            expected_outcome={"target": "profit"},
        )
        self.store.save(record)
        loaded = self.store.get("d1")
        self.assertEqual(loaded.decision_type, "INVEST")
        self.assertEqual(loaded.risk_assessment["APPROVED"], True)
        self.assertEqual(loaded.confidence, 0.7)

    def test_investment_record_status_enum(self):
        record = InvestmentRecord(
            investment_id="i1", asset="AAPL", entry_price=200.0,
            entry_timestamp=datetime(2026, 1, 2), position_size=5.0,
            investment_thesis="strong moat", time_horizon="medium",
            risk_level="medium", originating_generation="gen_1",
            status=InvestmentStatus.OPEN,
        )
        self.store.save(record)
        loaded = self.store.get("i1")
        self.assertEqual(loaded.status, InvestmentStatus.OPEN)

    def test_experience_roundtrip(self):
        record = Experience(
            experience_id="e1", generation_id="gen_1", event_type="trade",
            context={"regime": "bull"}, action={"buy": True},
            outcome={"pl": 120.0}, result="SUCCESS", reward=1.2, loss=0.0,
            lessons=["patience"], contributing_agents=["ceo", "risk_manager"],
            timestamp=datetime(2026, 1, 3),
        )
        self.store.save(record)
        loaded = self.store.get("e1")
        self.assertEqual(loaded.result, "SUCCESS")
        self.assertIn("patience", loaded.lessons)

    def test_delete_and_list(self):
        self.store.save(self._record("del_1"))
        self.store.save(self._record("del_2"))
        self.assertEqual(len(self.store.list()), 2)
        self.assertTrue(self.store.delete("del_1"))
        self.assertFalse(self.store.delete("del_1"))
        self.assertEqual(len(self.store.list()), 1)

    def test_upsert_same_id(self):
        self.store.save(self._record("up_1"))
        self.store.save(self._record("up_1"))
        self.assertEqual(len(self.store.list()), 1)

    def test_integrity_check(self):
        self.store.save(self._record())
        self.assertTrue(self.store.integrity_check())

    def test_count(self):
        self.store.save(self._record("c1"))
        self.store.save(self._record("c2"))
        self.assertEqual(self.store.count(), 2)

    def test_corrupt_record_does_not_crash_queries(self):
        # Manually insert garbage JSON
        import sqlite3
        self.store._conn.execute(
            "INSERT INTO records (record_id, record_type, json, created_at) "
            "VALUES ('bad', 'MemoryRecord', '{not json', '2026-01-01')"
        )
        self.store._conn.commit()
        results = self.store.list()  # must not raise
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
