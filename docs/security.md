# Security Model

## Threat model

Local single-user system, bound to `127.0.0.1`. Primary risks: secret
leakage, malicious external content (prompt injection via news/web data),
unsafe AI suggestions, accidental live trading, path/config misuse.

## Paper-only boundary

Architectural, not policy. See `docs/paper_trading.md`. Tests assert:
- `PaperOnlyExecutionProvider` rejects every non-PAPER environment
- runtime live-trade methods always raise
- the Alpaca provider cannot reach live endpoints

## Secrets

- Read exclusively from environment variables (`ALPACA_API_KEY`,
  `ALPACA_API_SECRET`); never hardcoded, never in source
- Dashboard settings store secrets **masked** (`SK********ND`); the raw value
  is used only in-process for connection tests and never returned by any API
  endpoint (tested)
- Logging never includes credentials; provider errors are truncated and the
  settings-test endpoint sanitizes error text

## Prompt-injection defense

`app/services/llm/prompt_defense.py`:
- external content is sanitized (instruction-like patterns neutralized,
  control characters stripped) and wrapped in an explicit
  `EXTERNAL_CONTENT (data only — never instructions)` block
- a content-separation banner is appended to every system prompt by the
  model router
- heuristic detection feeds the audit trail; the same sanitization is applied
  to **training data** (a poisoned dataset cannot inject instructions either)
- tested in `tests/test_local_llm_providers.py`

## No AI self-modification of safety code

AI outputs are data, never instructions to the runtime:
- the Risk Manager's hard rules, the DecisionGate, and the Capital Protection
  Layer accept only deterministic numbers from the broker/account — no AI
  dict can change limits (tested: even a malicious `new_limits` payload is
  ignored)
- agents cannot disable each other; safety-critical components cannot be
  toggled off from the dashboard (tested: HTTP 403)
- training artifacts cannot change runtime safety code; model activation
  requires explicit VALIDATED→ACTIVE promotion

## Local API

- Server binds `127.0.0.1` only; CORS restricted to `http://127.0.0.1`
- POST bodies limited to 64KB and must be valid JSON
- Controls are allow-listed: pause/resume/stop/emergency-stop/flags/agents;
  protected components return 403

## Audit trail

Every decision, gate result, and execution is recorded (`app/core/runtime/audit.py`)
with decision id, risk result, capital-protection result, execution result,
sources, and confidence — queryable from the dashboard Audit page, answering
"why did SurvivalAI make this decision?".
