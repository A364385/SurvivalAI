import unittest
from datetime import datetime
from app.utils.time import now_utc
from app.core.models.memory import (
    MemoryType, GenerationStatus, StrategyStatus, InvestmentStatus,
    MemoryRecord, Experience, DecisionRecord, InvestmentRecord,
    AgentPerformanceRecord, GenerationRecord, DeathReport, StrategyVersion
)
from app.core.memory.store import InMemoryStore

class TestMemoryArchitecture(unittest.TestCase):
    def test_memory_record(self):
        record = MemoryRecord(
            memory_id="m1",
            memory_type=MemoryType.FACT,
            generation_id="gen1",
            timestamp=now_utc(),
            source_agent="agent1",
            importance=5,
            content={"data": "test"},
            metadata={}
        )
        self.assertEqual(record.memory_type, MemoryType.FACT)

    def test_generation_record(self):
        gen = GenerationRecord(
            generation_id="gen2",
            parent_generation_id="gen1",
            generation_number=2,
            strategy_version="1.0",
            starting_capital=1000.0,
            ending_capital=1200.0,
            start_timestamp=now_utc(),
            end_timestamp=now_utc(),
            lifespan=10,
            return_percentage=20.0,
            maximum_drawdown=5.0,
            status=GenerationStatus.ACTIVE
        )
        self.assertEqual(gen.parent_generation_id, "gen1")

    def test_in_memory_store(self):
        store = InMemoryStore()
        record = MemoryRecord(
            memory_id="m1",
            memory_type=MemoryType.DECISION,
            generation_id="gen1",
            timestamp=now_utc(),
            source_agent="agent1",
            importance=5,
            content={"data": "test"}
        )
        store.save(record)
        fetched = store.get("m1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.memory_id, "m1")
        
        results = store.query(generation_id="gen1")
        self.assertEqual(len(results), 1)
        
        deleted = store.delete("m1")
        self.assertTrue(deleted)
        self.assertIsNone(store.get("m1"))

if __name__ == '__main__':
    unittest.main()
