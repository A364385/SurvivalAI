#!/usr/bin/env python
"""SurvivalAI one-command local startup.

Usage:
    python scripts/run_local.py --mode test
    python scripts/run_local.py --mode paper          (requires ALPACA paper keys)
    python scripts/run_local.py --port 3000

Starts:
- SQLite-backed persistence (data/survivalai.db)
- model registry + training queue (data/models_registry.db)
- local LLM router (LM Studio / Ollama / Transformers / mock fallback)
- Dashboard v2 on http://127.0.0.1:<port> (SSE live charts, controls)
- the autonomous runtime loop in a background thread

Never starts anything but paper execution. Fails loudly on config errors.
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logging import get_logger  # noqa: E402

logger = get_logger("run_local")


def parse_args():
    parser = argparse.ArgumentParser(description="SurvivalAI local launcher")
    parser.add_argument("--mode", choices=["test", "paper"], default="test",
                        help="test = deterministic offline mocks; paper = Alpaca PAPER (never live)")
    parser.add_argument("--port", type=int, default=8080, help="dashboard port")
    parser.add_argument("--cycles", type=int, default=0,
                        help="max runtime cycles (0 = unlimited)")
    parser.add_argument("--no-runtime", action="store_true",
                        help="start dashboard only (no autonomous loop)")
    parser.add_argument("--data-dir", default=None,
                        help="override data directory (SURVIVALAI_DATA_DIR)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_dir:
        os.environ["SURVIVALAI_DATA_DIR"] = args.data_dir
    data_dir = Path(os.environ.get("SURVIVALAI_DATA_DIR", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("SurvivalAI — LOCAL PAPER-TRADING SYSTEM")
    print(f"mode={args.mode}  data={data_dir.resolve()}")
    print("=" * 64)

    # ---- System assembly -------------------------------------------------
    from app.core.runtime.bootstrap import build_llm_stack, build_test_system
    from app.core.runtime.models import RuntimeConfig
    from app.core.memory.sqlite_store import SqliteMemoryStore
    from app.ml.model_registry import ModelRegistry
    from app.ml.promotion import PromotionPipeline, TrainingQueue
    from app.dashboard.api_v2 import DashboardV2
    from app.dashboard.state import DashboardState

    config = RuntimeConfig(test_mode=(args.mode == "test"), max_cycles=args.cycles)
    bootstrap = build_test_system(config=config)

    runtime = bootstrap.runtime
    agent_registry = bootstrap.agent_registry
    components = bootstrap.components

    # Durable store: SQLite-backed MemoryStore replaces the in-memory one so
    # ALL records persist across restarts (same interface, drop-in).
    sqlite_store = SqliteMemoryStore(str(data_dir / "survivalai.db"))
    runtime.memory_store = sqlite_store
    for service_name in ("idempotency", "decision_gate", "portfolio_monitor",
                         "investment_monitor", "experience_collector",
                         "learning_cycle", "cost_service", "transition_service",
                         "health_checker", "execution_service", "audit"):
        service = getattr(runtime, service_name, None)
        if service is not None:
            setattr(service, "memory_store", sqlite_store)
    generation_manager = components["generation_manager"]
    generation_manager.memory_store = sqlite_store

    # Model registry + training queue
    model_registry = ModelRegistry(str(data_dir / "models_registry.db"))
    pipeline = PromotionPipeline(model_registry)
    training_queue = TrainingQueue(model_registry, pipeline)

    # LLM stack: router + usage tracker + provider selection + agent binding.
    # Single owner lives in bootstrap.build_llm_stack so the tests exercise the
    # exact configuration path this launcher uses.
    llm_stack = build_llm_stack(runtime, sqlite_store, generation_manager, components)
    llm_router = llm_stack.router
    print(llm_stack.describe())

    # Dashboard state + v2 server. The state hub is attached to the runtime
    # so every portfolio sync feeds the live charts (equity/P/L/drawdown).
    state = DashboardState()
    runtime.dashboard_state = state
    dashboard = DashboardV2(
        host="127.0.0.1", port=args.port, state=state,
        runtime=runtime,
        generation_manager=generation_manager,
        memory_store=sqlite_store,
        agent_registry=agent_registry,
        portfolio_sync=components.get("portfolio_sync"),
        model_registry=model_registry,
        training_queue=training_queue,
        llm_router=llm_router,
    )

    # Runtime watchdog pause hook is already wired in SurvivalRuntime.
    runtime_start_errors = []

    def run_runtime():
        # runtime.start() runs the full autonomous cycle loop (blocking)
        # inside this dedicated thread and returns on stop().
        try:
            if not args.no_runtime:
                ok = runtime.start()
                if not ok:
                    runtime_start_errors.append("runtime.start() returned False")
        except Exception as e:
            logger.exception("runtime thread failed")
            runtime_start_errors.append(str(e))

    # ---- Health checks before start --------------------------------------
    print("\n[health] pre-flight checks:")
    db_ok = sqlite_store.integrity_check()
    print(f"  database: {'OK' if db_ok else 'FAIL'}")
    print(f"  dashboard: http://127.0.0.1:{args.port}")
    print(f"  mode: PAPER-ONLY (live trading architecturally impossible)")
    if not db_ok:
        print("FATAL: database integrity check failed")
        return 1

    # ---- Start ------------------------------------------------------------
    dashboard.start_background()
    print(f"\n[dashboard] serving on http://127.0.0.1:{args.port}")

    if args.no_runtime:
        print("[runtime] disabled (--no-runtime); dashboard only")
    else:
        t = threading.Thread(target=run_runtime, daemon=True, name="survivalai-loop")
        t.start()
        print("[runtime] autonomous paper-trading loop started")

    print("\nPress Ctrl+C to stop safely.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[shutdown] stopping safely...")
        try:
            if not args.no_runtime:
                runtime.stop()
        except Exception:
            pass
        try:
            training_queue.stop()
        except Exception:
            pass
        try:
            dashboard.stop()
        except Exception:
            pass
        sqlite_store.close()
        model_registry.close()
        print("[shutdown] complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
