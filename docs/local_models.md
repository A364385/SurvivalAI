# Local Model Architecture

## Overview

SurvivalAI uses **role-specialized local models**. Each role (News, Market,
Deep Looker, Crisis, Crypto, Risk, Safety, Strategy, Learning, CEO) has its
own model identity: system prompt, dataset, training run, version, evaluation,
and registry record. Roles without a trained specialized model fall back to:

```
TRAINED SPECIALIZED MODEL → GENERAL LOCAL MODEL → DETERMINISTIC MOCK → INSUFFICIENT_DATA
```

Nothing ever invents results silently.

## Providers

| Provider | Use | Endpoint |
|---|---|---|
| `LMStudioProvider` | served GGUF models via LM Studio | `http://127.0.0.1:1234` |
| `OllamaProvider` | served models via Ollama | `http://127.0.0.1:11434` |
| `TransformersLocalProvider` | in-process CUDA inference, trained adapters | — |
| `MockLLMProvider` | deterministic offline fallback | — |
| `GeminiProvider` (optional) | Google Gemini HTTP API | `generativelanguage.googleapis.com` |
| `ClaudeProvider` (optional) | Anthropic Claude HTTP API | `api.anthropic.com` |

All implement the same `generate` / `generate_structured` interface.
Selection: set `SURVIVALAI_LLM_PROVIDER`
(`lm_studio`|`ollama`|`gemini`|`claude`) + endpoint/model, or use the dashboard
**API & Providers** page (Test Connection included).

### Optional remote providers (Gemini / Claude)

Local providers are the default and the system never requires a cloud LLM.
Remote providers exist as an *optional* quality upgrade:

- Credentials come **only** from environment variables (`GEMINI_API_KEY` /
  `GOOGLE_API_KEY`, or `ANTHROPIC_API_KEY`). They are never accepted from the
  UI, never written to disk, never logged, and never returned by the API.
- Missing credentials fail **softly**: the router keeps serving and nothing
  crashes.
- **Retries**: exponential backoff (1s, 2s, 4s) on transient failures
  (HTTP 429/500/502/503/504) behind a circuit breaker. Non-retryable errors
  (4xx like 400/401) fail fast.
- **Schema validation**: `generate_structured` validates the returned JSON
  against the requested schema (required keys + top-level types) **and** the
  calling role's output contract (`ROLE_VALIDATORS` in
  `schema_validator.py`, e.g. "`confidence` must be between 0.0 and 1.0").
  Validation is cumulative, and on failure the model gets exactly one
  self-repair retry whose prompt carries the failing check's error text before
  we raise. Malformed or role-invalid output is never handed to an agent.
- **Cost accounting**: every call records tokens, latency, success/failure and
  an estimated USD cost from an explicit price table
  (`app/services/llm/usage_tracker.py`) into the SQLite-backed memory store.
  Costs above zero are attributed to the **active generation** as an `API`
  operating cost, so cloud spend counts against survival math. Inspect at
  `GET /api/v2/llm/usage` or the dashboard "LLM Usage & Cost" panel.

  Usage always names the real `provider:model`, including the default offline
  path (`mock:deterministic`) and local servers (`lm_studio:qwen2.5-7b`).
  Self-hosted and mock inference bills nothing, so `cost_usd` stays `0.0` —
  but token counts for those callers are **length-derived estimates**, flagged
  `tokens_estimated: true` and counted in `estimated_token_calls`, rather than
  being reported as a misleading zero. Only remote providers report true
  tokenizer counts.

## Model Router

`app/services/llm/model_router.py` picks the provider per role:
registered role providers → model registry ACTIVE record → configured default
→ fallback chain → deterministic mock. The router appends the
content-separation banner (prompt-injection defense) to every system prompt.

## Model Registry

`app/ml/model_registry.py` — SQLite-backed (`data/models_registry.db`).

States: `DISCOVERED → READY → TRAINING → EVALUATING → VALIDATED → ACTIVE`
(plus `FAILED`, `ARCHIVED`).

Rules enforced by code (not convention):
- only `VALIDATED` models can be activated
- one `ACTIVE` model per role (activation archives the previous)
- every state change is appended to an immutable event history
- activation is always explicit (dashboard button or `--activate`)

## Promotion Pipeline

`app/ml/promotion.py` runs, in order, with no step skippable:

```
BUILD DATASET → TRAIN → EVALUATE → COMPARE AGAINST CURRENT MODEL
  → PASS  → REGISTER (VALIDATED)   [activation separate + explicit]
  → FAIL  → REGISTER (ARCHIVED, with reasons)
```

Training failure preserves logs/checkpoints and marks the model FAILED —
it can never become VALIDATED by a shortcut. Smoke artifacts (pipeline tests
without the ML stack) are permanently barred from VALIDATED.

## Roles

`app/ml/roles.py` defines all ten roles with output schemas and
role-boundary rules (research roles cannot emit trade decisions; the CEO
chooses only among INVEST / DO_NOT_INVEST / DEFER / INSUFFICIENT_DATA; the
Investment Safety role only HOLD / REVIEW / EXIT_CANDIDATE / INSUFFICIENT_DATA).
