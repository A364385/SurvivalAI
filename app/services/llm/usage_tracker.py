"""LLM usage & cost tracking.

Records every LLM call's token usage and estimated USD cost through the
MemoryStore (SQLite-backed in production) so costs land in the same durable
audit surface as the rest of the system. Cost rates come from an explicit
price table — never fabricated at runtime.
"""

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.core.memory.store import MemoryStore
from app.core.models.memory import MemoryRecord, MemoryType
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)

# Explicit price table (USD per 1M tokens). Update as pricing changes; costs
# are estimates computed ONLY from this table.
PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "gemini:gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini:gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini:gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "claude:claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    "claude:claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
}


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD cost from the explicit price table (0.0 if unknown model)."""
    entry = PRICE_TABLE.get(f"{provider}:{model}")
    if not entry:
        return 0.0
    return (input_tokens / 1e6) * entry["input"] + (output_tokens / 1e6) * entry["output"]


# Tokenizers are provider-specific and unavailable for mock/local servers, so
# token counts for those callers are a documented character-length estimate.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: Optional[str]) -> int:
    """Length-derived token ESTIMATE — never a tokenizer count.

    Callers must mark records built from this with ``tokens_estimated=True``
    so an estimate is never mistaken for a provider-reported total.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class UsageRecord:
    provider: str
    model: str
    role: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    success: bool
    error: Optional[str] = None
    generation_id: str = "unknown"
    # True when the token counts are length-derived estimates rather than
    # counts reported by the provider (mock / local / any self-hosted model).
    tokens_estimated: bool = False
    timestamp: str = field(default_factory=lambda: now_utc().isoformat())
    usage_id: str = field(default_factory=lambda: generate_id("llm_usage"))


class LLMUsageTracker:
    """Records LLM usage and estimated costs; aggregates summaries."""

    def __init__(self, memory_store: Optional[MemoryStore] = None,
                 generation_manager: Optional[Any] = None,
                 generation_id_resolver: Optional[Any] = None):
        self.memory_store = memory_store
        # Optional cost attribution: `generation_manager` receives the cost and
        # `generation_id_resolver` (a zero-arg callable) supplies the current
        # generation id when a call did not carry one.
        self.generation_manager = generation_manager
        self.generation_id_resolver = generation_id_resolver
        self._lock = threading.Lock()
        self.totals: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    def record(self, usage: UsageRecord) -> None:
        with self._lock:
            bucket = self.totals.setdefault(
                f"{usage.provider}:{usage.model}",
                {"calls": 0.0, "input_tokens": 0.0, "output_tokens": 0.0,
                 "cost_usd": 0.0, "errors": 0.0, "estimated_calls": 0.0},
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += usage.input_tokens
            bucket["output_tokens"] += usage.output_tokens
            bucket["cost_usd"] += usage.estimated_cost_usd
            if usage.tokens_estimated:
                bucket["estimated_calls"] += 1
            if not usage.success:
                bucket["errors"] += 1

        if self.memory_store is not None:
            try:
                record = MemoryRecord(
                    memory_id=f"llm_usage_{usage.usage_id}",
                    memory_type=MemoryType.FACT,
                    generation_id=usage.generation_id,
                    timestamp=now_utc(),
                    source_agent="LLMUsageTracker",
                    importance=4,
                    content=asdict(usage),
                    metadata={"llm_usage": True, "provider": usage.provider,
                              "model": usage.model},
                )
                self.memory_store.save(record)
            except Exception as e:
                logger.error("Failed to persist LLM usage: %s", e)

        if usage.estimated_cost_usd > 0:
            self._apply_generation_cost(usage)

    def _resolve_generation_id(self, usage: UsageRecord) -> Optional[str]:
        if usage.generation_id and usage.generation_id != "unknown":
            return usage.generation_id
        if self.generation_id_resolver is not None:
            try:
                return self.generation_id_resolver() or None
            except Exception as e:
                logger.debug("Generation id resolver failed: %s", e)
        return None

    def _apply_generation_cost(self, usage: UsageRecord) -> None:
        """Push the estimated cost into the generation's operating costs.

        The manager is injected explicitly — it is the single owner of
        generation state, so there is no fallback lookup path here.
        """
        manager = self.generation_manager
        if manager is None:
            return
        generation_id = self._resolve_generation_id(usage)
        if not generation_id:
            return
        try:
            from app.core.models.generation import OperatingCost  # local: avoids cycle
            cost = OperatingCost(
                cost_id=usage.usage_id,
                cost_type="API",
                amount=round(usage.estimated_cost_usd, 6),
                currency="USD",
                frequency="ONE_TIME",
                description=f"LLM {usage.provider}:{usage.model} "
                            f"({usage.input_tokens}in/{usage.output_tokens}out tokens)",
                provider=f"{usage.provider}:{usage.model}",
                timestamp=now_utc(),
            )
            manager.add_operating_cost(generation_id, cost)
        except Exception as e:
            # Audible on purpose: silently losing cost attribution would let
            # cloud spend escape the survival math.
            logger.warning("Failed to attribute LLM cost to generation %s: %s",
                           generation_id, e)

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            totals = {
                key: dict(value) for key, value in self.totals.items()
            }
        overall = {
            "calls": sum(b["calls"] for b in totals.values()),
            "input_tokens": sum(b["input_tokens"] for b in totals.values()),
            "output_tokens": sum(b["output_tokens"] for b in totals.values()),
            "estimated_cost_usd": round(
                sum(b["cost_usd"] for b in totals.values()), 6),
            "errors": sum(b["errors"] for b in totals.values()),
            # How many of the calls above carry estimated (not reported)
            # token counts, so the dashboard can label them honestly.
            "estimated_token_calls": sum(
                b.get("estimated_calls", 0.0) for b in totals.values()),
        }
        return {"by_model": totals, "overall": overall}

    def query_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        if self.memory_store is None:
            return []
        records = self.memory_store.query(memory_type=MemoryType.FACT)
        entries = []
        for record in records:
            content = getattr(record, "content", {})
            if isinstance(content, dict) and content.get("usage_id"):
                entries.append(content)
        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return entries[:limit]
