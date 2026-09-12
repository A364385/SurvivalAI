"""Paper-only execution service.

Wraps the ExecutionProvider abstraction with:
- hard paper-mode verification before any submission,
- pre-flight OrderValidator checks,
- idempotent order submission (client_order_id keyed),
- explicit order lifecycle tracking (SUBMITTED -> ACCEPTED -> FILLED ...),
- confirmation from the provider before an order is ever marked FILLED.

Live trading is architecturally impossible: the ExecutionEnvironment enum
contains only PAPER, OrderRequest.__post_init__ rejects any other value, and
this service refuses to run unless the paper-only guarantee is verified.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.memory.store import MemoryStore
from app.core.models.execution import (
    ExecutionEnvironment,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.runtime.idempotency import IdempotencyManager
from app.core.runtime.models import RuntimeConfig
from app.services.execution.provider import ExecutionProvider
from app.services.execution.safety import PaperOnlyExecutionProvider
from app.services.execution.validator import OrderValidator
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


@dataclass
class PaperModeVerification:
    """Result of verifying that execution is paper-only."""
    verified: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)


class PaperExecutionService:
    """Executes approved decisions against the paper execution provider.

    Safety properties:
    - refuses to operate unless paper mode is verified,
    - validates every order with OrderValidator before submission,
    - submits through PaperOnlyExecutionProvider (hard PAPER boundary),
    - idempotent per client_order_id (retries never duplicate orders),
    - never marks an investment filled without provider confirmation.
    """

    def __init__(
        self,
        execution_provider: ExecutionProvider,
        market_data_provider,
        memory_store: MemoryStore,
        idempotency_manager: IdempotencyManager,
        config: RuntimeConfig,
    ):
        self.config = config
        self.memory_store = memory_store
        self.idempotency = idempotency_manager
        self.market_data_provider = market_data_provider

        # Hard safety: always wrap the provider in the paper-only shield.
        if isinstance(execution_provider, PaperOnlyExecutionProvider):
            self._provider = execution_provider
        else:
            self._provider = PaperOnlyExecutionProvider(execution_provider)
        self._validator = OrderValidator(self._provider)

        self._last_verification: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Paper-mode verification
    # ------------------------------------------------------------------
    def verify_paper_mode(self, force: bool = False) -> PaperModeVerification:
        """Verify the execution path is strictly paper trading.

        Checks (all must pass):
        1. config.paper_mode_required is enabled
        2. ExecutionEnvironment enum contains only PAPER
        3. provider account is reachable
        4. order endpoint is reachable
        """
        if not force and self._last_verification.get("fresh"):
            return self._last_verification["result"]

        checks: Dict[str, bool] = {}
        failures: List[str] = []

        checks["paper_mode_required"] = bool(self.config.paper_mode_required)
        if not checks["paper_mode_required"]:
            failures.append("paper_mode_required is disabled in configuration")

        checks["environment_enum_paper_only"] = (
            [e.value for e in ExecutionEnvironment] == ["PAPER"]
        )
        if not checks["environment_enum_paper_only"]:
            failures.append("ExecutionEnvironment enum must contain only PAPER")

        try:
            account = self._provider.get_account()
            checks["account_reachable"] = account is not None
        except Exception as e:
            checks["account_reachable"] = False
            failures.append(f"account unreachable: {e}")

        try:
            self._provider.list_orders(limit=1)
            checks["order_endpoint_reachable"] = True
        except Exception as e:
            checks["order_endpoint_reachable"] = False
            failures.append(f"order endpoint unreachable: {e}")

        result = PaperModeVerification(
            verified=all(checks.values()),
            checks=checks,
            failures=failures,
        )
        self._last_verification = {"fresh": True, "result": result, "at": now_utc()}
        return result

    def invalidate_verification(self) -> None:
        self._last_verification = {}

    # ------------------------------------------------------------------
    # Price resolution (never fabricated)
    # ------------------------------------------------------------------
    def _resolve_price(self, symbol: str) -> Dict[str, Any]:
        """Resolve a reference price from the market data provider.

        Returns {"price": float, "source": str} or {"price": None, "error": str}.
        """
        try:
            quote = self.market_data_provider.get_quote(symbol)
            bid = getattr(quote, "bid_price", None)
            ask = getattr(quote, "ask_price", None)
            if bid and ask and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                if math.isfinite(mid) and mid > 0:
                    return {"price": mid, "source": "quote_mid"}
            last = getattr(quote, "last_price", None) or getattr(quote, "last_price", None)
            if last and math.isfinite(last) and last > 0:
                return {"price": float(last), "source": "quote_last"}
        except Exception:
            pass
        try:
            trade = self.market_data_provider.get_latest_trade(symbol)
            price = getattr(trade, "price", None)
            if price and math.isfinite(price) and price > 0:
                return {"price": float(price), "source": "latest_trade"}
        except Exception:
            pass
        return {"price": None, "error": "no reliable price available"}

    # ------------------------------------------------------------------
    # Order lifecycle records
    # ------------------------------------------------------------------
    def _lifecycle_record_id(self, client_order_id: str) -> str:
        return f"order_lifecycle_{client_order_id}"

    def _save_lifecycle(self, record: MemoryRecord) -> None:
        self.memory_store.save(record)

    def _load_lifecycle(self, client_order_id: str) -> Optional[MemoryRecord]:
        return self.memory_store.get(self._lifecycle_record_id(client_order_id))

    def _append_lifecycle_event(
        self,
        record: MemoryRecord,
        status: str,
        detail: str = "",
    ) -> None:
        record.content["lifecycle"].append(
            {
                "status": status,
                "timestamp": now_utc().isoformat(),
                "detail": detail,
            }
        )
        record.content["current_status"] = status
        self._save_lifecycle(record)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute_decision(
        self,
        decision_id: str,
        symbol: str,
        position_value: float,
        generation_id: str,
        thesis: str = "",
    ) -> Dict[str, Any]:
        """Execute an approved INVEST decision as a paper market order.

        Returns a structured result dict; never raises for business-level
        failures (they are returned as status=FAILED / SKIPPED with reasons).
        """
        client_order_id = f"survival_{generation_id}_{decision_id}_{symbol}"

        # Idempotency: never submit the same decision twice.
        if not self.idempotency.begin(
            IdempotencyManager.OPERATION_ORDER_SUBMISSION,
            client_order_id,
            generation_id=generation_id,
            metadata={"decision_id": decision_id, "symbol": symbol},
        ):
            existing = self.idempotency.get_operation(
                IdempotencyManager.OPERATION_ORDER_SUBMISSION, client_order_id
            )
            return {
                "status": "DUPLICATE_SKIPPED",
                "client_order_id": client_order_id,
                "previous": existing,
            }

        try:
            # Paper-mode verification is mandatory before any submission.
            verification = self.verify_paper_mode()
            if not verification.verified:
                self.idempotency.fail(
                    IdempotencyManager.OPERATION_ORDER_SUBMISSION,
                    client_order_id,
                    f"paper verification failed: {verification.failures}",
                )
                return {
                    "status": "BLOCKED_PAPER_VERIFICATION",
                    "client_order_id": client_order_id,
                    "failures": verification.failures,
                }

            # Price must come from the provider — never fabricated.
            price_info = self._resolve_price(symbol)
            if price_info.get("price") is None:
                self.idempotency.fail(
                    IdempotencyManager.OPERATION_ORDER_SUBMISSION,
                    client_order_id,
                    price_info.get("error", "price unavailable"),
                )
                return {
                    "status": "SKIPPED_NO_PRICE",
                    "client_order_id": client_order_id,
                    "reason": price_info.get("error", "price unavailable"),
                }

            price = price_info["price"]
            quantity = math.floor((position_value / price) * 100) / 100.0
            if quantity <= 0:
                self.idempotency.fail(
                    IdempotencyManager.OPERATION_ORDER_SUBMISSION,
                    client_order_id,
                    f"quantity below minimum for value {position_value} at price {price}",
                )
                return {
                    "status": "SKIPPED_QUANTITY_TOO_SMALL",
                    "client_order_id": client_order_id,
                    "price": price,
                }

            request = OrderRequest(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
                environment=ExecutionEnvironment.PAPER,
            )

            # Pre-flight validation (symbol, qty, buying power, positions).
            validator = OrderValidator(self._provider)
            validator.validate(request)

            lifecycle = MemoryRecord(
                memory_id=self._lifecycle_record_id(client_order_id),
                memory_type=MemoryType.FACT,
                generation_id=generation_id,
                timestamp=now_utc(),
                source_agent="PaperExecutionService",
                importance=9,
                content={
                    "client_order_id": client_order_id,
                    "decision_id": decision_id,
                    "symbol": symbol,
                    "side": OrderSide.BUY.value,
                    "quantity": quantity,
                    "reference_price": price,
                    "price_source": price_info["source"],
                    "thesis": thesis,
                    "lifecycle": [],
                    "current_status": "CREATED",
                },
                metadata={},
            )
            self._append_lifecycle_event(lifecycle, "CREATED", "order request built")

            response = self._provider.submit_order(request)
            self._append_lifecycle_event(
                lifecycle,
                response.status.value,
                f"provider order_id={response.order_id}",
            )
            lifecycle.content["provider_order_id"] = response.order_id
            self._save_lifecycle(lifecycle)

            # Confirm actual provider state — never trust the submit response alone.
            confirmed = self._provider.get_order(response.order_id)
            self._append_lifecycle_event(
                lifecycle, confirmed.status.value, "provider confirmation"
            )
            lifecycle.content["confirmed_status"] = confirmed.status.value
            lifecycle.content["filled_quantity"] = confirmed.filled_quantity
            lifecycle.content["average_fill_price"] = confirmed.average_fill_price
            self._save_lifecycle(lifecycle)

            filled = (
                confirmed.status == OrderStatus.FILLED
                and confirmed.filled_quantity
                and confirmed.filled_quantity > 0
            )

            result = {
                "status": "EXECUTED" if filled else "SUBMITTED",
                "client_order_id": client_order_id,
                "provider_order_id": confirmed.order_id,
                "symbol": symbol,
                "quantity": confirmed.filled_quantity or confirmed.quantity,
                "average_fill_price": confirmed.average_fill_price,
                "order_status": confirmed.status.value,
                "generation_id": generation_id,
                "decision_id": decision_id,
            }

            if filled:
                self._create_investment_record(
                    decision_id=decision_id,
                    generation_id=generation_id,
                    symbol=symbol,
                    quantity=confirmed.filled_quantity,
                    entry_price=confirmed.average_fill_price or price,
                    thesis=thesis,
                    order_id=confirmed.order_id,
                )

            self.idempotency.complete(
                IdempotencyManager.OPERATION_ORDER_SUBMISSION,
                client_order_id,
                {"status": result["status"], "order_id": confirmed.order_id},
            )
            return result

        except Exception as e:
            self.idempotency.fail(
                IdempotencyManager.OPERATION_ORDER_SUBMISSION,
                client_order_id,
                str(e),
            )
            logger.error("Paper execution failed for %s: %s", symbol, e)
            return {
                "status": "FAILED",
                "client_order_id": client_order_id,
                "error": str(e),
            }

    def _create_investment_record(
        self,
        decision_id: str,
        generation_id: str,
        symbol: str,
        quantity: float,
        entry_price: float,
        thesis: str,
        order_id: str,
    ) -> str:
        investment_id = generate_id("inv")
        record = MemoryRecord(
            memory_id=f"investment_{investment_id}",
            memory_type=MemoryType.INVESTMENT,
            generation_id=generation_id,
            timestamp=now_utc(),
            source_agent="PaperExecutionService",
            importance=10,
            content={
                "investment_id": investment_id,
                "decision_id": decision_id,
                "order_id": order_id,
                "asset": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "position_size": quantity * entry_price,
                "investment_thesis": thesis or "Executed approved decision",
                "time_horizon": "MEDIUM_TERM",
                "risk_level": "MODERATE",
                "originating_generation": generation_id,
                "status": "OPEN",
                "entry_timestamp": now_utc().isoformat(),
            },
            metadata={},
        )
        self.memory_store.save(record)
        return investment_id

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------
    def reconcile_open_orders(self) -> List[Dict[str, Any]]:
        """Reconcile tracked open orders against provider state."""
        updates: List[Dict[str, Any]] = []
        try:
            open_orders: List[OrderResponse] = self._provider.get_open_orders()
        except Exception as e:
            logger.error("Failed to list open orders: %s", e)
            return [{"status": "FAILED", "error": str(e)}]

        tracked_ids = set()
        for order in open_orders:
            tracked_ids.add(order.order_id)
            record = self._load_lifecycle(order.client_order_id)
            if record is None:
                # Order exists at provider but was never tracked (e.g. crash
                # between submit and record). Adopt it for reconciliation.
                record = MemoryRecord(
                    memory_id=self._lifecycle_record_id(order.client_order_id),
                    memory_type=MemoryType.FACT,
                    generation_id="recovered",
                    timestamp=now_utc(),
                    source_agent="PaperExecutionService",
                    importance=9,
                    content={
                        "client_order_id": order.client_order_id,
                        "provider_order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": order.quantity,
                        "lifecycle": [],
                        "current_status": order.status.value,
                        "recovered": True,
                    },
                    metadata={},
                )
            self._append_lifecycle_event(record, order.status.value, "reconciliation")
            updates.append(
                {
                    "client_order_id": order.client_order_id,
                    "status": order.status.value,
                }
            )
        return updates

    def cancel_all_open_orders(self, reason: str = "") -> List[Dict[str, Any]]:
        """Cancel all open paper orders (used on generation death)."""
        results: List[Dict[str, Any]] = []
        try:
            open_orders = self._provider.get_open_orders()
        except Exception as e:
            return [{"status": "FAILED", "error": str(e)}]
        for order in open_orders:
            try:
                cancelled = self._provider.cancel_order(order.order_id)
                results.append(
                    {
                        "order_id": order.order_id,
                        "cancelled": bool(cancelled),
                        "reason": reason,
                    }
                )
            except Exception as e:
                results.append(
                    {"order_id": order.order_id, "cancelled": False, "error": str(e)}
                )
        return results

    def get_account_snapshot(self) -> Dict[str, Any]:
        """Fetch account state through the paper-only provider."""
        account = self._provider.get_account()
        return {
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "currency": account.currency,
            "status": account.status,
            "timestamp": account.timestamp.isoformat() if account.timestamp else None,
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        positions = self._provider.get_positions()
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "average_entry_price": p.average_entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_profit_loss": p.unrealized_profit_loss,
            }
            for p in positions
        ]