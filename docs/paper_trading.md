# Paper Trading (Safety Architecture)

## Absolute boundary

SurvivalAI is **paper-trading only**. There is no live mode:

- `ExecutionEnvironment` enum contains only `PAPER`
- `PaperOnlyExecutionProvider` wraps every execution path and raises
  `InvalidEnvironmentError` on any non-PAPER request
- `PaperExecutionService` re-verifies paper mode before every order
- `SurvivalRuntime.execute_live_trade()` / `activate_live_trading()` always
  raise `PermissionError` (tested)
- The Alpaca integration hardcodes the paper endpoint
  (`https://paper-api.alpaca.markets/v2`); there is no code path to the live
  endpoint, and tests prove live endpoints cannot be used

## Decision flow (gates in order)

```
DATA → RESEARCH AGENTS → DEEP LOOKER → RISK MANAGER (deterministic policy)
  → CAPITAL PROTECTION (deterministic, portfolio-level)
  → CEO DECISION → EXECUTION GATE (re-checks protection with live numbers)
  → PAPER ORDER (idempotent)
```

Any gate failure = **no order**. AI output can request; only deterministic
layers dispose.

## Capital Protection Layer

`app/core/safety/capital_protection.py` — pure deterministic, no LLM/network:
max position fraction, max portfolio exposure, min cash reserve, max
drawdown (emergency stop), correlated-group and crypto exposure caps,
emergency stop + trading pause. Every verdict lists failed checks with
numbers. The runtime applies it twice: after the CEO decision (veto) and at
execution time (defense in depth).

## Idempotency & crash recovery

Orders carry unique decision/execution IDs with an idempotency manager; a
repeated cycle can never double-submit. On restart the runtime recovers the
active generation, reconciles portfolio and open orders with the paper
broker, and never blindly repeats an order (tested in
`tests/integration/test_end_to_end_runtime.py`).

## Emergency stop

Dashboard → Controls → **EMERGENCY PAPER STOP** engages
`CapitalProtectionLayer.engage_emergency_stop()`: all new paper orders are
rejected until explicitly lifted. Pause/stop controls are graceful; the
watchdog also auto-pauses on unsafe conditions.
