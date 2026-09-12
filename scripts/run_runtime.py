#!/usr/bin/env python
"""SurvivalAI autonomous runtime entry point.

Usage:
    python scripts/run_runtime.py --mode test --cycles 3
    python scripts/run_runtime.py --mode production

Modes:
    test        Deterministic offline simulation with mock providers
                (no network, no real orders).
    production  Internet-connected, PAPER-TRADING-ONLY operation via the
                Alpaca paper API. Requires ALPACA_API_KEY and
                ALPACA_API_SECRET environment variables. Live trading is
                architecturally impossible in this code path.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.runtime.bootstrap import build_production_system, build_test_system
from app.core.runtime.models import RuntimeConfig
from app.utils.logging import get_logger

logger = get_logger("run_runtime")


def parse_args():
    parser = argparse.ArgumentParser(description="SurvivalAI autonomous runtime")
    parser.add_argument(
        "--mode",
        choices=["test", "production"],
        default="test",
        help="test = deterministic mocks; production = Alpaca PAPER trading only",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Maximum number of autonomous cycles (0 = unlimited)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Cycle interval in seconds",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL",
        help="Comma-separated watched symbols",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial generation capital",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = RuntimeConfig(
        cycle_interval_seconds=args.interval,
        max_cycles=args.cycles,
        test_mode=(args.mode == "test"),
        paper_mode_required=True,
        watched_symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
    )

    if args.mode == "test":
        print("=== SurvivalAI TEST MODE (deterministic mocks, no network) ===")
        system = build_test_system(config=config, initial_capital=args.capital)
    else:
        print("=== SurvivalAI PRODUCTION MODE (PAPER TRADING ONLY) ===")
        try:
            system = build_production_system(config=config, initial_capital=args.capital)
        except RuntimeError as e:
            print(f"FAIL-SAFE: {e}")
            return 2

    runtime = system.runtime
    print(f"Runtime {runtime.runtime_id} starting (paper-only enforcement: ON)")
    try:
        runtime.start()
    except KeyboardInterrupt:
        print("Interrupted; stopping runtime gracefully...")
        runtime.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())