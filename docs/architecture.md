# SurvivalAI Technical Architecture

## 1. Architectural Philosophy
SurvivalAI is built on an evolutionary, generational simulation framework where autonomous agents collaborate to navigate financial markets, preserve capital, and survive.

Key Tenets:
- **Strict Paper Trading Only**: Absolute safety boundary. No live trading credentials, endpoints, or execution paths exist.
- **Provider Abstraction**: Core logic depends strictly on abstract interfaces (`MarketDataProvider`, `ExecutionProvider`, `NewsProvider`, `LLMProvider`, `FundamentalDataProvider`), never directly on external SDKs like Alpaca or Gemini/OpenAI.
- **Structured Contracts**: All inter-component and external communication uses strictly-typed domain models rather than unstructured JSON or chat strings.
- **Asynchronous Order Lifecycle**: Orders progress explicitly from `PROPOSED` -> `VALIDATED` -> `SUBMITTED` -> `ACCEPTED` -> `PARTIALLY_FILLED` -> `FILLED` (or `REJECTED`/`CANCELLED`/`EXPIRED`).
- **External Source of Truth**: The paper broker account is the canonical source of truth, reconciled locally via `PortfolioSynchronizer`.
- **Untrusted External Data Boundary**: Real-world news and web data are treated as untrusted input; prompt injection protection is enforced at the prompt boundary.

---

## 2. Component Layers

```text
┌────────────────────────────────────────────────────────┐
│                   REAL INTERNET DATA                   │
│   (Alpaca Market Data v2 / Real Financial News APIs)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               PROVIDER ADAPTER & NORMALIZATION         │
│   MarketDataProvider ──► Quote / Trade / Bar           │
│   NewsProvider       ──► NewsItem                      │
│   FundamentalDataProvider (interface only / future)    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               NEWS RESEARCH PIPELINE                   │
│   Source Validation (Primary vs Secondary)             │
│   Deduplication & Event Clustering                     │
│   Contradiction & Staleness Detection                  │
│   LLM Analysis with Prompt Injection Isolation         │
│   Structured Fact Extraction & Traceable Citations     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│              MARKET RESEARCH PIPELINE                  │
│   Data Validation → Feature Calculation (no LLM math)  │
│   Trend / Regime / Anomalies → Optional News Context   │
│   LLM Interpretation → MarketSnapshot + AgentResult    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│           CRISIS & GEOPOLITICAL RISK PIPELINE          │
│   Normalized news/official claims → classify/severity  │
│   Transmission, exposure, risk dimensions, updates     │
│   LLM interpretation → GeopoliticalCrisis + AgentResult│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                 AUTONOMOUS AGENT SYSTEM                │
│  NewsResearchAgent     MarketResearchAgent  DeepLooker │
│  CrisisRiskAgent       RiskManagerAgent                │
│  InvestmentSafetyManagerAgent                           │
│                         ▼                              │
│                    CEOAgent (Orchestrator)              │
│                 (Decision Proposal Only)               │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             ORDER VALIDATION & SAFETY SHIELD           │
│  OrderValidator (Symbol, Qty, Balance, Positions)      │
│  PaperOnlyExecutionProvider (Hard PAPER boundary)      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               PAPER EXECUTION & SYNC                   │
│  ExecutionProvider (Alpaca Paper API / MockExecution)  │
│  PortfolioSynchronizer (Equity, Cash, Positions Sync)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             GENERATIONAL MEMORY & EVOLUTION            │
│  Generations (Capital, Drawdown, Survival/Death)       │
│  MemoryStore (FACT, ANALYSIS, DECISION, EXPERIENCE)    │
│  StrategyUpdaterAgent (Iterative Parameter Tuning)     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             AUTONOMOUS RUNTIME (Step 15)               │
│  SurvivalRuntime (health checks, decision gating,      │
│  paper execution, investment monitoring, learning,     │
│  cost accounting, generation transition, scheduler)    │
│  PaperExecutionService (idempotent order submission,   │
│  order lifecycle tracking, provider-confirmed fills)   │
│  DecisionGate (cooldowns, exposure limits, crisis)     │
│  IdempotencyManager (critical operations, retries)     │
│  RuntimeScheduler (configurable task intervals)        │
│  RuntimeHealthChecker (layered health checks)          │
│  GenerationTransitionService (death → successor)       │
└────────────────────────────────────────────────────────┘
```

---

## 3. Providers & Data Contracts

### Market Data Layer (`app/services/market_data/`)
- `MarketDataProvider`: Abstract interface for `get_quote()`, `get_quotes()`, `get_latest_trade()`, `get_historical_bars()`, `get_market_clock()`.
- `AlpacaMarketDataProvider`: Real HTTP REST adapter reading `ALPACA_API_KEY` and `ALPACA_API_SECRET` from environment variables.
- `MockMarketDataProvider`: In-memory implementation for offline testing and backtesting.

### Execution & Safety Layer (`app/services/execution/`)
- `ExecutionProvider`: Abstract interface for `submit_order()`, `cancel_order()`, `get_order()`, `list_orders()`, `get_open_orders()`, `get_positions()`, `get_account()`.
- `PaperOnlyExecutionProvider`: Safety wrapper raising `InvalidEnvironmentError` if any request attempts execution outside `ExecutionEnvironment.PAPER`.
- `OrderValidator`: Pre-flight validation verifying symbol formatting, positive quantities, valid types, buying power for buys, and position quantities for sells.
- `AlpacaPaperExecutionProvider`: Connects strictly to `https://paper-api.alpaca.markets/v2`. Rejects any non-paper base URL.
- `MockExecutionProvider`: Fully deterministic paper execution engine with configurable fill models for unit testing.

### News Layer (`app/services/news/`)
- `NewsProvider`: Abstract interface for `get_latest_news()` and `get_news_for_symbol()`.
- `AlpacaNewsProvider`: Real HTTP REST adapter connecting to `https://data.alpaca.markets/v1beta1/news`.
- `MockNewsProvider`: In-memory provider for synthetic scenarios and offline testing.

### LLM Abstraction Layer (`app/services/llm/`)
- `LLMProvider`: Abstract interface for LLM calls (`generate` and `generate_structured`).
- `MockLLMProvider`: Deterministic mock for unit testing and offline development.
- `validate_news_analysis_schema` / `validate_market_analysis_schema`: Validates that model responses conform to expected analytical schemas.

### Fundamental Data Layer (`app/services/fundamental/`)
- `FundamentalDataProvider`: Future interface for `CompanyProfile`, `ValuationMetrics`, `FinancialStatements`, and `EarningsData`.
- `MockFundamentalDataProvider`: Returns only explicitly configured records; never invents filings or ratios.
- Live fundamental APIs are **not** implemented in this step because existing market-data adapters do not expose them.

### Market Research Pipeline (`app/agents/market_research/`)
- `DataQualityChecker`: Blocks NaN/Inf/negative prices, impossible OHLC, duplicates, unordered timestamps, and negative volume from calculations; warns on zero volume, stale quotes, and large gaps.
- `MarketFeatureCalculator` (deterministic Python):
  - **Price**: last close, session change vs open, % change, high/low, distance from high/low.
  - **Returns**: close-to-close 1-day, 5-day, 20-day when enough bars exist.
  - **SMA**: 20 / 50 / 200; otherwise `MovingAverages.status = INSUFFICIENT_DATA`.
  - **Historical volatility**: sample standard deviation of log returns (`ddof=1`) × `sqrt(252)`. Short window default 5 periods, medium 20. This is **not** implied volatility (no options feed).
  - **Volume**: `volume_ratio = current_volume / average_prior_volume` (zero averages do not divide).
  - **Momentum**: Wilder RSI(14), 14-period rate of change, SMA alignment.
- `MarketRegimeClassifier` documented rules: UPTREND/DOWNTREND/SIDEWAYS/MIXED/INSUFFICIENT_DATA and TRENDING_UP/DOWN, HIGH/LOW_VOLATILITY, TRANSITION, UNKNOWN.
- `MarketAnomalyDetector`: price spike/drop, volume spike, volatility spike, gap up/down, abnormal spread. Severity and measurements only — not a trade direction.
- `MarketDataCache`: in-process TTL cache. Expired entries are discarded so stale history is never presented as current. `retrieved_at` is fetch time; `data_timestamp` is the last bar's market time.
- `MarketPromptBuilder`: untrusted market/news payloads in delimited tags; forbids BUY/SELL/HOLD and causation-without-evidence.
- `MarketResearchAgent`: orchestrates the pipeline, stores ANALYSIS/FACT/ERROR in `MemoryStore`, emits `MarketAnalysisCompleted`.

### News Research Pipeline (`app/agents/news_research/`)
- `SourceValidator`: Distinguishes primary sources (SEC, central banks, company press releases) from secondary sources (news media).
- `NewsDeduplicator`: Clusters related articles into `NewsEventCluster` without discarding sources.
- `ConflictDetector`: Identifies discrepancies between reporting outlets.
- `RecencyTracker`: Tracks whether newer statements clarify or supersede earlier reports.
- `NewsPromptBuilder`: Enforces strict prompt boundaries with `<untrusted_news_data>` delimiters to defeat prompt injection attempts.
- `NewsResearchAgent`: Orchestrates the complete pipeline, storing structured analysis in `MemoryStore` and emitting `NewsEvent`.

### Official & Macro Interfaces (`app/services/official/`, `app/services/macro/`)
- `OfficialDataProvider`: Future adapter for government / central-bank / IGO statements normalized as `NewsItem`. No in-agent web scraping.
- `MacroDataProvider`: Future adapter for inflation/rates series. Returns `None` unless a real feed is configured; never fabricates prints.
- Mock implementations store only explicitly provided records.

### Crisis & Geopolitical Risk Pipeline (`app/agents/crisis_risk/`)
- Consumes normalized news (and optional official statements / market snapshots). Reuses news clustering, source classes, and conflict/recency tools.
- Deterministic `CrisisClassifier`: event types (primary + secondary), severity, escalation, geographic scope, horizon, transmission channels, exposures, risk dimensions.
- `CrisisRegistry` / `CrisisLinker`: updates and related-event links without a second event bus.
- `CrisisPromptBuilder`: untrusted external content delimiters; forbids investment advice and false causation.
- `CrisisRiskAgent`: emits `CrisisEvent` on the existing event types when severity is at least MODERATE. Downstream Risk Manager should inspect dimensions, not a single opaque score.

### Deep Looker Pipeline (`app/agents/deep_looker/`)
- `DeepLookerAgent` is a research-only due-diligence integrator. It consumes structured outputs from News Research, Market Research, Crisis Risk, and `FundamentalDataProvider`; it does not import or rely on those agents' private internals.
- `DeepResearchRequest` captures symbol, timestamp, depth, history window, source selection, generation id, originating task id, and optional context.
- `DeepResearchDossier` is the structured output contract: identity, business/asset model, market analysis, fundamentals, valuation, competition, news, geopolitical/regulatory analysis, risks, catalysts, unknowns, thesis, scenarios, assumptions, evidence, sources, confidence, timestamps, and data completeness.
- Deterministic calculations happen in `FundamentalMetricsCalculator` before synthesis. The LLM may interpret relationships and uncertainty but cannot calculate financial ratios, approve investments, place orders, or enforce risk limits.
- Valuation metrics distinguish historical values from forward/estimated values. Forward metrics are used only when real estimates are supplied by a provider. Historical or peer valuation context returns `INSUFFICIENT_DATA` when comparison data is unavailable.
- Contradiction detection is a first-class step. The dossier keeps unresolved opposing evidence visible instead of smoothing it into a cleaner narrative.
- Thesis construction is decision-support only. It uses support/invalidation factors, observable invalidation conditions, key assumptions, and bull/base/bear scenarios without price targets or BUY/SELL/HOLD labels.
- Evidence traceability follows `Conclusion -> Evidence -> Source -> URL/identifier -> Timestamp`. Missing data remains explicit in `DataCompleteness` and `unknowns`.
- `DeepResearchCompleted` is emitted on the existing event model with research id, asset, confidence, thesis status, key findings, and risk summary.

### Risk Manager Pipeline (`app/agents/risk_manager/`)
- `RiskManagerAgent` is a deterministic-first risk gate that evaluates investment proposals against configured risk limits. It is NOT a strategist and has no execution or portfolio mutation authority.
- `InvestmentProposal` captures asset, proposed_position_value, classifications (sector, asset_class, geography), and optional metadata.
- `PortfolioRiskState` captures portfolio_value, available_cash, positions (with classifications), historical_peak_value, and timestamp.
- `RiskPolicy` provides configurable limits: max_single_position_pct, max_portfolio_concentration_pct, max_sector_exposure_pct, max_asset_class_exposure_pct, max_geographic_exposure_pct, min_cash_reserve_pct, max_portfolio_drawdown_pct, volatility thresholds, crisis severity thresholds, max_correlated_exposure_pct, max_positions, min_data_confidence, stale_market_data_minutes, and data requirements.
- `RiskAssessment` is the structured output: assessment_id, generation_id, proposal_id, timestamp, asset, position metrics, exposure metrics, risk severity levels (volatility, drawdown, liquidity, crisis, fundamental, thesis, correlation, data_quality), warnings, violated_rules, passed_rules, required_actions, decision (APPROVED/APPROVED_WITH_WARNINGS/BLOCKED/INSUFFICIENT_DATA), confidence, reasoning, evidence, and recommended_max_position_size.
- Deterministic risk calculations in Python: position percentages, cash after investment, concentration exposure, sector/asset-class/geographic exposure, drawdown, all with safe handling of zero/negative/missing values.
- Hard vs soft rule distinction: HARD rules (position size, cash reserve, concentration, portfolio state, data quality) cause BLOCKED violations; SOFT warnings (elevated volatility, moderate crisis risk, high valuation uncertainty) produce APPROVED_WITH_WARNINGS.
- Research integration: Consumes `MarketSnapshot` (volatility, regime, anomalies), `AgentResult` from News Research (negative events, regulatory changes), `AgentResult` from Crisis Risk (severity, escalation, exposure), and `DeepResearchDossier` (thesis confidence, contradictions, fundamental risk).
- LLM usage limited to summarization and explanation only; LLM cannot calculate risk limits, override hard rules, approve blocked investments, or invent missing data.
- Safety hierarchy: Research Agents → Deep Looker → Risk Manager → Future CEO/Orchestrator → Future Paper Execution. The Risk Manager is a safety gate that future components cannot bypass.
- Event emission: `RiskAssessmentStarted`, `RiskAssessmentCompleted`, and `RiskViolationEvent` on the existing event bus.
- Memory integration: Stores important risk assessments as `MemoryRecord` type `ANALYSIS` in `MemoryStore`.
- Permission boundaries: `execute_order`, `modify_portfolio`, `modify_capital`, `change_strategy`, and `approve_investment` raise `PermissionError`. The Risk Manager cannot execute trades, directly modify portfolio state, or bypass paper-trading safety boundaries.

### Investment Safety Pipeline (`app/agents/investment_safety/`)
- `InvestmentSafetyManagerAgent` monitors existing investments and evaluates whether original theses remain valid. It is NOT the same as the Risk Manager.
- `InvestmentSafetyAssessment` is the structured output: assessment_id, investment_id, generation_id, asset, timestamp, original_thesis (immutable), current price, position metrics, unrealized P/L, thesis status, recommendation (HOLD/REVIEW/EXIT_CANDIDATE/INSUFFICIENT_DATA), thesis status, supporting/invalidating factors, new risks, changed conditions, market/fundamental/news/crisis/drawdown assessments, contradictions, findings, confidence, evidence, sources, required follow-up, and previous assessment comparison.
- `SafetyRecommendation` enum: HOLD (thesis supported, no critical invalidation), REVIEW (thesis weakened, mixed evidence, new risks), EXIT_CANDIDATE (critical thesis invalidation or severe deterioration), INSUFFICIENT_DATA (not enough reliable information).
- `ThesisStatus` enum: STRONGLY_SUPPORTED, SUPPORTED, MIXED, WEAKENING, INVALIDATED, UNKNOWN. The system explains WHY the thesis has its current state.
- `ThesisCondition` represents thesis invalidation conditions with category (FUNDAMENTAL, MARKET, VALUATION, BUSINESS, COMPETITIVE, MACRO, GEOPOLITICAL, REGULATORY, MANAGEMENT, LIQUIDITY, OTHER), status (VALID, WEAKENING, VIOLATED, UNKNOWN), severity, source, and observed/expected values.
- `Contradiction` captures contradictions between original thesis and current evidence with original_statement, contradictory_evidence, source, severity, confidence, affected_thesis_component, and detection time.
- Deterministic calculations in Python: position percentages, cash after investment, concentration exposure, sector/asset-class/geographic exposure, drawdown. Missing data returns None rather than fabricated values.
- Hard vs soft rules distinction: Critical contradictions and severe thesis invalidations cause EXIT_CANDIDATE; mixed evidence and moderate risks cause REVIEW; supporting evidence with no issues causes HOLD.
- Research integration: Consumes `MarketSnapshot` (volatility, regime, anomalies), `AgentResult` from News Research (negative events, regulatory changes), `AgentResult` from Crisis Risk (severity, escalation, exposure), and `DeepResearchDossier` (thesis confidence, contradictions, fundamental risk).
- Previous assessment tracking: Compares current assessment with previous assessments to detect trends in recommendation, thesis status, risks, and contradictions.
- Time horizon awareness: Respects SHORT_TERM, MEDIUM_TERM, LONG_TERM investment horizons. Long-term investments emphasize fundamentals and business conditions; shorter horizons may weigh market conditions more heavily.
- Price vs thesis damage separation: Distinguishes between PRICE DAMAGE (normal volatility, expected drawdown) and THESIS DAMAGE (growing contradiction, business deterioration). Price decline alone does not automatically trigger EXIT_CANDIDATE.
- LLM usage limited to explanation only: LLM may synthesize evidence, explain contradictions, summarize thesis changes, and provide qualitative interpretation. LLM must NOT calculate metrics, rewrite original thesis, approve investments, or override deterministic safety rules.
- Event emission: `SafetyAssessmentStarted`, `SafetyAssessmentCompleted`, `ThesisWeakenedEvent`, `ThesisInvalidatedEvent`, `ReviewRequiredEvent`, and `ExitCandidateDetectedEvent` on the existing event bus.
- Memory integration: Stores important safety assessments as `MemoryRecord` type `ANALYSIS` in `MemoryStore` for future CEO, Strategy Updater, and generation learning.
- Permission boundaries: `execute_order`, `modify_portfolio`, `modify_capital`, `change_strategy`, and `approve_investment` raise `PermissionError`. The Safety Manager cannot execute trades, directly modify portfolio state, or bypass paper-trading safety boundaries.

### CEO / Orchestrator Pipeline (`app/agents/ceo/`)
- `CEOAgent` is the central coordination layer that turns specialized-agent research into controlled, explainable investment workflows. It coordinates agents but does NOT execute trades.
- `AgentRegistry` provides a dependency-injected registry for available agents. The CEO resolves agents by stable identifier rather than hard-coding dependencies. Agents can be enabled/disabled dynamically.
- `OrchestrationRequest` captures the coordination request: request_id, generation_id, request_type (INVESTMENT_PROPOSAL, EXISTING_INVESTMENT_REVIEW, MARKET_EVENT, CRISIS_REASSESSMENT, PORTFOLIO_REVIEW), asset, objective, timestamp, priority, requested_position_size, context, and metadata.
- `OrchestrationRun` tracks the complete execution: run_id, request_id, generation_id, started_at, completed_at, status, current_stage, stages, agent_executions, risk_assessment, safety_assessment, decision, errors, warnings, evidence, and metadata.
- `DecisionProposal` is the final output: decision_id, generation_id, asset, decision_type (INVEST/DO_NOT_INVEST/DEFER/INSUFFICIENT_DATA for new investments; HOLD/REVIEW/EXIT_CANDIDATE/INSUFFICIENT_DATA for existing investments), proposed_action, position_size, thesis, supporting_evidence, opposing_evidence, market/news/crisis/deep_analysis context, risk_assessment, confidence, expected_outcome, invalidation_conditions, decision_reason, and status.
- **New Investment Pipeline**: REQUEST → VALIDATION → RESEARCHING (Market Research, News Research, Crisis Risk in parallel) → DEEP ANALYSIS (Deep Looker receives upstream outputs) → EVIDENCE AGGREGATION → RISK ASSESSMENT (always required) → DECISION SYNTHESIS → MEMORY + EVENTS.
- **Existing Investment Review Pipeline**: REQUEST → VALIDATION → SAFETY ASSESSMENT (Investment Safety Manager) → EVIDENCE AGGREGATION → DECISION SYNTHESIS → MEMORY + EVENTS.
- **State Machine**: Explicit orchestration stages with valid transitions: CREATED → VALIDATING → RESEARCHING → DEEP_ANALYSIS → RISK_ASSESSMENT/SAFETY_ASSESSMENT → DECISION_SYNTHESIS → COMPLETED/FAILED/BLOCKED/DEFERRED. Invalid transitions are rejected.
- **Evidence Aggregation**: Preserves source agent identity, timestamps, and identifiers. Distinguishes facts from interpretations. Tracks supporting and opposing evidence. Makes evidence traceable to originating agent/result. Records missing data and warnings.
- **Conflict Detection**: Detects conflicting outputs between agents (e.g., Market Research says favorable while Crisis Risk reports severe exposure). Conflicts are recorded in the orchestration run and decision proposal rather than silently resolved.
- **Risk Manager Gate**: The CEO always routes new-investment proposals through Risk Manager. BLOCKED prevents INVEST. INSUFFICIENT_DATA prevents approval and results in DEFER or INSUFFICIENT_DATA. APPROVED/APPROVED_WITH_WARNINGS may allow INVEST with preserved warnings. The CEO cannot override hard risk rules.
- **Deep Looker Integration**: Deep Looker receives upstream outputs from Market Research, News Research, Crisis Risk, and the original request/context. The CEO does not fabricate missing research.
- **MemoryStore Integration**: Persists orchestration run records, decision proposals, and important stage results as `MemoryRecord` with type `DECISION`. Makes it possible to reconstruct complete orchestration from stored records.
- **Event Bus Integration**: Emits typed orchestration events: `OrchestrationStarted`, `AgentDispatched`, `AgentCompleted`, `AgentFailed`, `DeepAnalysisStarted`, `RiskAssessmentStarted`, `DecisionProposalCreated`, `InvestmentBlocked`, `OrchestrationCompleted`, `OrchestrationFailed`, `ConflictDetected`.
- **Failure Handling**: Explicitly handles missing critical research, agent failures, agent timeouts, agent cancellation, missing registry entries, invalid agent results, conflicting outputs, and duplicate requests. Failures are represented in typed run fields and structured errors.
- **Permission Boundaries**: `execute_order`, `modify_portfolio`, `modify_capital`, `change_strategy`, and `approve_investment` raise `PermissionError`. The CEO cannot execute trades, access live trading, bypass Risk Manager, or treat decision proposals as orders.
- **LLM Constraints**: If an LLM is used, it may summarize or explain but cannot override deterministic safety rules, approve a Risk Manager-blocked proposal, invent missing data, convert EXIT_CANDIDATE into an execution command, change state transitions, or manufacture evidence or source attribution.
- **STOP BEFORE EXECUTION**: The CEO produces decision proposals only. It does not execute orders, access live broker functionality, or interact with execution providers. Future Step 12+ will handle paper execution separately.

### Strategy Updater Pipeline (`app/agents/strategy_updater/`)
- `StrategyUpdaterAgent` analyzes historical performance and proposes strategy improvements. It NEVER directly modifies the active strategy.
- **Strategy Versioning**: Uses existing `StrategyVersion` architecture with strategy_id, version, parent_strategy_id, generation_id, created_at, parameters, rules, description, and status (ACTIVE, TESTING, REJECTED, ARCHIVED). The currently active strategy is always explicitly identifiable.
- **Strategy Structure**: Strongly typed strategy representation including asset universe, allowed asset classes, research requirements, minimum thesis strength, position sizing rules, diversification rules, cash reserve requirements, risk tolerance, maximum drawdown tolerance, market regime rules, crisis response rules, entry conditions, exit/review conditions, holding periods, and portfolio constraints. Strategy configuration is centralized, not hardcoded.
- **StrategyChangeProposal**: Proposal for changing strategy with proposal_id, parent_strategy_id, proposed_parameters, changed_rules, unchanged_rules, motivation, supporting_experiences, supporting_decisions, supporting_outcomes, expected_effect, risks, assumptions, confidence, backtest_required, and status (PROPOSED, BACKTESTING, PASSED, FAILED, REJECTED, APPROVED_FOR_SIMULATION, ACTIVE). ACTIVE status is not directly reachable from PROPOSED.
- **Learning from Experience**: Consumes DecisionRecord, InvestmentRecord, Experience, DeathReport, AgentPerformanceRecord, historical market data, historical news/event data, previous strategy versions, backtest results, risk assessments, and investment safety assessments. Identifies repeated losses, repeated successful patterns, excessive concentration, poor market-regime performance, excessive drawdowns, poor crisis handling, weak thesis selection, excessive trading, underinvestment, overinvestment, strategy weaknesses, and agent weaknesses.
- **No Automatic Overfitting**: Avoids changing strategy based on single random results. Uses minimum evidence requirements: minimum number of observations, minimum number of trades, minimum confidence, minimum performance difference, and statistical significance where practical.
- **Memory Integration**: Stores proposals as `MemoryRecord` with type `STRATEGY` for future evaluation and backtesting.
- **Permission Boundaries**: `activate_strategy` and `modify_active_strategy` raise `PermissionError`. The Strategy Updater cannot directly activate strategies or modify the active strategy.
- **LLM Constraints**: LLM may be used for identifying patterns, proposing hypotheses, explaining failures, summarizing backtest findings, and generating candidate rule changes. LLM must NOT directly activate strategies, modify active strategy state, fabricate historical data, perform financial calculations, override risk constraints, access broker execution, or invent performance results. All calculations are deterministic Python.

### Backtesting & Evaluation Pipeline (`app/backtesting/`)
- `BacktestEngine` provides robust backtesting framework for strategy evaluation. Supports historical market data, simulated portfolio, simulated positions, cash, transaction costs, slippage, position sizing, strategy rules, risk limits, entry decisions, exit/review decisions, portfolio value, and drawdown.
- **TransactionCostConfig**: Configurable transaction costs including commission_per_trade, commission_per_share, spread_percentage, slippage_percentage, and market_impact_factor. Conservative defaults ensure realistic cost modeling.
- **SimulatedPortfolio**: Tracks cash, positions, average entry prices, realized P&L, unrealized P&L, portfolio value, exposure, drawdown, and transactions. Reuses existing portfolio models where possible.
- **SimulatedTrade**: Records each executed trade with trade_id, symbol, side, quantity, execution_price, timestamp, commission, slippage, and total_cost.
- **DataLeakageProtection**: Protects against look-ahead bias and data leakage. Filters future data based on timestamps with configurable lookahead window. Historical information is only available at the time it would actually have been known.
- **Risk Manager Integration**: Backtester respects Risk Manager rules. Simulated investments violating hard risk rules are rejected. Strategy performance reflects actual SurvivalAI safety constraints.
- **No Look-Ahead Bias**: Engine prevents access to future prices, future fundamental data, future news, or future events. Timestamps are strictly enforced.
- **Provider Agnostic**: Backtesting engine uses `MarketDataProvider` abstraction, supporting both mock and real external API data. Strategy/backtesting remains provider-agnostic.
- **StrategyComparison**: Compares two strategies based on total return, drawdown, risk-adjusted return, volatility, survival characteristics, transaction costs, crisis performance, regime performance, number of trades, concentration, and consistency.
- **SurvivalFitness**: Survival-aware fitness evaluation considering capital preservation, probability of ruin, maximum drawdown, return, risk-adjusted return, costs, stability, and diversification. Balances SURVIVAL + GROWTH + RISK CONTROL. Penalizes "never invest" strategies to avoid loopholes.
- **External API Architecture**: Supports Internet → External APIs → Provider layer → Normalized SurvivalAI data models → Agents → CEO → Risk Manager → Paper Trading. API credentials from environment variables, request timeouts, retries, rate-limit handling, API failures, stale data detection, source timestamps, and caching where appropriate. No hardcoded API keys.
- **Provider Health Check**: `ProviderHealthChecker` provides health check architecture for external providers. Reports provider available, unavailable, authentication failure, rate limited, stale data, malformed response, and timeout. Separate optional integration/health tests for real providers (unit tests do not depend on internet access).
- **Paper Trading Boundary**: Existing paper-trading provider from Step 4 remains the ONLY execution path. Architecture preserves Research → CEO proposal → Risk Manager → approved proposal → paper execution. Strategy Updater and Backtester NEVER directly place paper orders; they only evaluate strategies. No live broker credentials or real-money execution introduced.

### Generation & Evolution System (`app/core/generation/`)
- `GenerationManager` manages generation lifecycle, inheritance, and evolution. This is the first component that allows SurvivalAI to operate across multiple generations, learning from past performance and creating successor generations.
- **GenerationLifecycleState**: Explicit lifecycle states: CREATED, INITIALIZING, ACTIVE, PAUSED, DYING, DEAD, SUCCESSOR_PENDING, SUCCESSOR_CREATED, ARCHIVED. State transitions are controlled and observable.
- **GenerationState**: Extended generation state with generation_id, parent_generation_id, generation_number, strategy_version, lifecycle_state, creation/start/end/death timestamps, starting/current/ending capital, return percentage, maximum drawdown, lifespan, cause_of_death, death_trigger, death_details, inherited knowledge refs, successor_generation_id, operating costs, and metrics.
- **DeathTrigger**: Configurable death triggers: CAPITAL_DEPLETED, MINIMUM_SURVIVAL_THRESHOLD, MAXIMUM_DRAWDOWN_EXCEEDED, UNRECOVERABLE_PORTFOLIO_STATE, OPERATING_COSTS_EXCEEDED, CONFIGURED_CONDITION_VIOLATED, INFRASTRUCTURE_FAILURE. Death conditions are explicit and configurable.
- **OperatingCost**: Cost abstraction for generations with cost_id, cost_type (API, DATA, MODEL, INFRASTRUCTURE, PAPER_TRADING, RECURRING), amount, currency, frequency, description, provider, and timestamp. Costs reduce available capital.
- **DeathCondition**: Configurable death condition with condition_id, trigger, threshold, is_fatal, description, and enabled flag.
- **InheritedKnowledge**: Knowledge inherited from parent generation with knowledge_id, source_generation_id, source_type (EXPERIENCE, STRATEGY, BACKTEST, AGENT_PERFORMANCE), relevance, confidence, timestamp, validation_status, content, and metadata. Only validated knowledge becomes active.
- **GenerationComparison**: Comparison between two generations with generation_a_id, generation_b_id, comparison_timestamp, total_return_difference, maximum_drawdown_difference, survival_duration_difference, ending_capital_difference, number_of_investments_difference, win_rate_difference, transaction_cost_difference, operating_cost_difference, crisis/market_regime/benchmark performance differences, strategy_version_difference, risk_violation_count_difference, blocked_investment_count_difference, major_failures, fitness_scores, and recommended_generation.
- **ExperienceSelectionCriteria**: Criteria for selecting relevant previous experiences with min_relevance, min_confidence, max_age_days, required_outcomes, required_asset_classes, required_regimes, require_validated, and max_selection_count.
- **Generation Creation Pipeline**: CREATE GENERATION → SELECT APPROVED STRATEGY → LOAD INHERITED KNOWLEDGE → LOAD RELEVANT EXPERIENCES → LOAD PREVIOUS FAILURE LESSONS → LOAD APPROVED STRATEGY PARAMETERS → INITIALIZE CAPITAL → INITIALIZE PORTFOLIO → INITIALIZE RISK LIMITS → INITIALIZE AGENT CONFIGURATION → HEALTH CHECK PROVIDERS → START GENERATION. Generation fails to start if critical infrastructure is unavailable.
- **Infrastructure Health Check**: Checks MarketDataProvider, NewsProvider, LLMProvider, MemoryStore, Risk Manager, CEO/Orchestrator, and paper-trading provider configuration. Distinguishes infrastructure unavailable, data unavailable, strategy unavailable, and risk configuration unavailable.
- **Inheritance Mechanism**: New generations may inherit successful experiences, failed experiences, DeathReport lessons, validated strategy parameters, validated strategy rules, useful agent-performance information, known market/crisis patterns, previous backtesting results, and known failure modes. Each inherited item has source generation, source type, relevance, confidence, timestamp, and validation status. Historical experiences remain immutable.
- **Evolution Pipeline**: GENERATION DIES → DEATH REPORT → EXPERIENCE EXTRACTION → FAILURE ANALYSIS → SUCCESS ANALYSIS → STRATEGY CHANGE PROPOSALS → BACKTEST → WALK-FORWARD VALIDATION → RISK EVALUATION → STRATEGY COMPARISON → APPROVAL/REJECTION → SUCCESSOR GENERATION. System never bypasses Step 12's validation pipeline.
- **Mutation/Improvement**: Controlled strategy evolution for position sizing, diversification limits, cash reserve, asset allocation, entry thresholds, exit/review thresholds, risk tolerance, market-regime rules, crisis-response rules, research requirements, minimum thesis strength, maximum exposure, and holding period rules. Every change has reason, source experience, expected benefit, potential downside, affected parameters, and validation status.
- **Anti-Overfitting**: Requires minimum sample size, multiple market regimes, bull/bear/sideways conditions, crisis periods, out-of-sample validation, walk-forward testing, transaction costs, slippage, drawdown evaluation, survival evaluation, and benchmark comparison. Rejects changes that only improve narrow historical periods.
- **Survival Fitness**: Considers survival, return, risk-adjusted performance, maximum drawdown, volatility, capital preservation, diversification, crisis resilience, transaction costs, operating costs, and strategy robustness. Prevents "never invest = perfect survival" loophole.
- **Death Conditions**: Capital <= 0, equity <= minimum survival threshold, unrecoverable portfolio state, maximum allowed drawdown exceeded, configured survival condition violated, or critical infrastructure condition (if explicitly configured as fatal). Every death records exact trigger, timestamp, capital/equity, portfolio state, active strategy, open investments, recent decisions, risk state, market conditions, crisis conditions, operating costs, and suspected causes.
- **DeathReport Integration**: Uses existing DeathReport model. When generation dies: freeze state, collect final portfolio, collect decisions/investments/agent_performance/market/crisis conditions, calculate final metrics, identify failures/successes/causes/lessons, store DeathReport, emit GenerationDied event. DeathReport is immutable after finalization.
- **Lineage Tracking**: Every successor generation is traceable. Implements get_generation(), get_parent_generation(), get_child_generations(), get_generation_lineage(), get_generation_history(), and get_generation_performance(). Preserves historical lineage.
- **Event System Integration**: Emits GenerationCreated, GenerationInitializing, GenerationStarted, GenerationPaused, GenerationDying, GenerationDied, SuccessorGenerationRequested, SuccessorGenerationCreated, StrategyCandidateCreated, StrategyCandidateApproved, and StrategyCandidateRejected events. Reuses existing event models where possible.
- **Safety Boundaries**: NEVER places live trades, connects to live broker, bypasses Risk Manager, bypasses CEO/Orchestrator, bypasses Strategy Backtesting, activates unvalidated strategy, rewrites historical records, fabricates experiences/market data/death causes/performance. Remains simulation/paper-trading only.
- **Paper Trading Boundary**: May create generations operating against existing simulated/paper portfolio. Uses ONLY configured Alpaca paper-trading environment, never live trading endpoint, no configuration switch to silently redirect to live trading. Generation creation fails safely if paper-trading configuration is invalid.
- **LLM Constraints**: LLMs may help analyze DeathReports, extract lessons, propose hypotheses, explain why strategy might improve, and identify patterns. LLMs must NOT decide generation death, calculate financial metrics, override death conditions, activate strategies, bypass backtesting, override Risk Manager hard rules, fabricate historical data/experiences, or rewrite immutable historical records. Deterministic code remains authoritative.

### Autonomous Runtime (`app/core/runtime/`)
- `SurvivalRuntime` provides central runtime service for autonomous SurvivalAI operation. Coordinates the entire system including startup, health checks, generation initialization, market-data cycles, research cycles, decision cycles, execution cycles, monitoring cycles, learning cycles, survival checks, generation transitions, and graceful shutdown.
- **RuntimeState**: Typed runtime state machine with states: STARTING, HEALTH_CHECK, INITIALIZING_GENERATION, OBSERVING, RESEARCHING, ANALYZING, RISK_CHECK, DECIDING, EXECUTING, MONITORING, LEARNING, SURVIVAL_CHECK, GENERATION_TRANSITION, PAUSED, STOPPING, STOPPED, ERROR. Invalid state transitions are rejected.
- **RuntimeConfig**: Configurable runtime parameters including cycle_interval_seconds, market_data_interval_seconds, news_interval_seconds, crisis_interval_seconds, portfolio_sync_interval_seconds, investment_monitor_interval_seconds, learning_interval_seconds, health_check_interval_seconds, max_retries, retry_delay_seconds, stale_data_threshold_seconds, paper_mode_required, test_mode, api_timeout_seconds, decision_cooldown_seconds, and max_concurrent_research.
- **CyclePhase**: Autonomous cycle phases: OBSERVE, RESEARCH, ANALYZE, RISK_CHECK, DECIDE, EXECUTE, MONITOR, LEARN, SURVIVAL_CHECK.
- **RuntimeStateSnapshot**: Snapshot of runtime state with runtime_id, generation_id, current_state, previous_state, timestamp, cycle_id, cycle_phase, health_status, error_count, last_error, last_error_timestamp, and metadata.
- **CycleRecord**: Record of an autonomous cycle with cycle_id, generation_id, start_timestamp, end_timestamp, phase, triggered_actions, agent_results, decision_result, execution_result, portfolio_state, survival_state, errors, warnings, and status.
- **FailureRecord**: Record of a failure with failure_id, failure_type (TRANSIENT, RECOVERABLE, CRITICAL, FATAL), component, error_message, timestamp, cycle_id, retry_count, resolved, resolution_timestamp, and metadata.
- **HealthCheckResult**: Result of a health check with component, available, latency_ms, error_message, timestamp, and metadata.
- **IdempotencyKey**: Idempotency key for operations with operation_type, operation_id, generation_id, timestamp, and expires_at.
- **Autonomous Cycle**: Controlled recurring cycle: OBSERVE → RESEARCH → ANALYZE → RISK CHECK → DECIDE → PAPER EXECUTE IF APPROVED → MONITOR → LEARN → SURVIVAL CHECK → NEXT CYCLE. Cycle interval is configurable.
- **Market Data Collection**: Uses existing MarketDataProvider to retrieve current market data, validate timestamps, detect stale data, detect provider failures, handle rate limits, handle temporary outages, and cache where appropriate. Never fabricates missing market data.
- **News Collection**: Uses existing NewsProvider and News Research Agent to periodically update relevant news, event clusters, market-moving events, company events, macro events, and geopolitical events. Respects source quality, freshness, deduplication, conflicts, and prompt-injection protection.
- **Crisis Data Collection**: Uses existing Crisis & Geopolitical Risk Agent to monitor wars, military conflicts, sanctions, elections, government changes, tariffs, trade restrictions, energy disruptions, supply-chain disruptions, central-bank events, inflation, recession risks, regulatory changes, major infrastructure failures, cyber incidents, and other configured crisis events.
- **Research Pipeline**: Uses existing architecture with Market Research, News Research, and Crisis Risk running in parallel where appropriate, followed by Deep Looker. Uses configurable triggers such as new investment candidate, major news event, major crisis event, thesis contradiction, unusual market movement, existing investment deterioration.
- **Investment Decision Pipeline**: For new investments: REQUEST → INPUT VALIDATION → MARKET RESEARCH → NEWS RESEARCH → CRISIS RISK → DEEP LOOKER → RISK MANAGER → CEO/ORCHESTRATOR → DECISION. CEO cannot bypass Risk Manager. If Risk Manager returns BLOCKED, investment must not execute. If INSUFFICIENT_DATA, system must not pretend investment is safe.
- **Paper Execution**: If decision is approved, uses existing ExecutionProvider. Execution remains behind existing abstraction. If Alpaca is configured, ONLY uses Alpaca paper-trading environment. Runtime verifies paper mode before execution with explicit safety check.
- **Live Trading Safety**: Runtime makes live trading architecturally impossible through SurvivalAI code path. Does not add live broker credentials, live endpoints, "switch to live" button, automatic live environment selection, live mode inference from missing configuration, or silent fallback to live trading. If paper-trading configuration is missing, FAIL SAFE.
- **Portfolio Synchronization**: After paper execution, synchronizes portfolio state tracking cash, equity, positions, average entry price, current price, unrealized P/L, realized P/L, exposure, available buying power, open orders, filled orders, cancelled orders, and rejected orders. Uses provider abstraction.
- **Order Lifecycle**: Implements/reuses CREATED → SUBMITTED → ACCEPTED → PARTIALLY_FILLED → FILLED and failure paths: REJECTED, CANCELLED, EXPIRED, FAILED. Runtime reconciles actual provider state. Never marks investment as filled without confirmation.
- **Existing Investment Monitoring**: Periodically runs Investment Safety Manager for every active investment, monitoring thesis, fundamentals, valuation, market conditions, news, crisis conditions, drawdown, contradictions, and time horizon. Investment Safety Manager may return HOLD, REVIEW, EXIT_CANDIDATE, or INSUFFICIENT_DATA. Does not automatically sell simply because EXIT_CANDIDATE is returned unless existing architecture explicitly permits it.
- **Decision Frequency Gating**: Implements decision gating to avoid blindly trading every cycle. Reasons to avoid new decision: no meaningful market change, insufficient data, no candidate, existing portfolio already sufficiently exposed, risk limits, crisis conditions, cooldown, duplicate decision, provider outage, recent decision on same asset, or strategy constraints.
- **Idempotency**: Critical operations are idempotent including investment request, order submission, portfolio synchronization, generation transition, and learning cycle. Uses unique operation/request IDs to prevent duplicate investments or duplicate generations on temporary retry.
- **Failure Handling**: Implements graceful failure handling classifying failures as TRANSIENT, RECOVERABLE, CRITICAL, or FATAL. Possible failures: API outage, network timeout, rate limit, malformed API response, stale data, provider authentication failure, paper-trading outage, LLM failure, agent timeout, invalid model output, database/memory failure, or event failure. Does not crash entire system because of temporary provider outage.
- **Safe Pause**: Implements safe pause state when market data is stale, critical API unavailable, paper provider unavailable, Risk Manager unavailable, required research unavailable, configuration invalid, or system integrity check fails. Existing positions can continue to be monitored where data is available. Does not create new investments while critical safety infrastructure is unavailable.
- **Survival Check**: After each meaningful cycle evaluates capital, equity, peak equity, current drawdown, maximum drawdown, operating costs, realized P/L, unrealized P/L, risk violations, and configured survival conditions. If death condition is reached: STOP NEW INVESTMENT ACTIVITY → CANCEL/RECONCILE APPROPRIATE PAPER ORDERS → FREEZE GENERATION → CREATE DEATH REPORT → START LEARNING LOOP → STRATEGY EVALUATION → VALIDATION → SUCCESSOR GENERATION. Does not immediately continue operating the dead generation.
- **Operating Costs**: Accounts for configured operating costs including model/API usage, market-data costs, news-data costs, infrastructure, and other configured expenses. Costs are represented transparently. If costs are unavailable, marked as unknown rather than fabricated.
- **Generation Transition**: When generation dies, uses existing Step 13 + Step 14 architecture: DEATH → DEATH REPORT → EXPERIENCE EXTRACTION → LESSON VALIDATION → PATTERN ANALYSIS → STRATEGY PROPOSAL → BACKTEST → WALK-FORWARD VALIDATION → RISK EVALUATION → STRATEGY COMPARISON → APPROVAL/REJECTION → SUCCESSOR GENERATION. If no strategy passes validation, enters SUCCESSOR_PENDING or another safe configured state. Does not invent a valid strategy.
- **Internet Connectivity**: Designed for real internet-connected operation using Internet → External APIs → Provider Layer → Normalized Data → Agents → Orchestrator → Risk Manager → Paper Trading. Does not replace real external providers with hardcoded demo data. Tests may use mock providers. Production configuration uses real providers with provider health checks.
- **API Configuration**: Uses environment variables for market API credentials, news API credentials, LLM credentials, and paper-trading credentials. NEVER hardcodes secrets. Creates/updates `.env.example` with placeholder names only. Never logs credentials or exposes secrets through logs, events, memory, agent output, or dashboard APIs.
- **Scheduling**: Configurable scheduler/orchestrator supporting different task frequencies: market polling, news polling, crisis monitoring, portfolio synchronization, investment monitoring, learning, cost accounting, and health checks. Does not make every component run at the same interval. Avoids uncontrolled concurrent executions.
- **Concurrency**: Uses controlled concurrency. Parallelizes independent research tasks where safe. Prevents concurrent conflicting operations such as two investments against the same decision, duplicate generation creation, simultaneous strategy activation, or conflicting portfolio synchronization. Uses locks or idempotency mechanisms where necessary.
- **Observability**: Adds structured runtime logging where every cycle has cycle_id, generation_id, timestamp, runtime state, triggered actions, agent results, decision result, execution result, portfolio state, survival state, errors, and warnings. Does not log secrets. Creates structured event records rather than relying only on plain text logs.
- **Audit Trail**: Every investment decision is traceable: market/news/crisis evidence → research outputs → Deep Looker → Risk Manager → CEO → DecisionRecord → execution → portfolio outcome → Experience → Learning. A user can later answer "Why did SurvivalAI make this investment?" with evidence-backed records.
- **No Hallucinations**: LLMs must never fabricate market prices, news, API responses, portfolio values, investment outcomes, strategy results, backtest results, or crisis events. If information is unavailable, returns INSUFFICIENT_DATA or appropriate failure state.
- **Prompt Injection Defense**: External news/web/API content is untrusted data. Never allows external content to modify system instructions, execute commands, change risk limits, change strategy, bypass agents, request secrets, place orders, or modify generation state. Treats external content strictly as data.
- **Test Mode**: Implements controlled TEST/SIMULATION mode using deterministic mock market provider, deterministic mock news provider, deterministic mock crisis provider, and simulated execution provider. Allows entire autonomous loop to be tested without internet or real paper orders. Production mode remains internet-connected and paper-trading-only.
- **Paper Mode Health Check**: Before autonomous execution, verifies provider reachable, authentication valid, environment is paper, account accessible, portfolio accessible, order endpoint available, and no live endpoint configured. If any critical check fails, DOES NOT EXECUTE.
- **RuntimeRecovery**: Handles runtime restart and recovery including recovering runtime state from memory, recovering active generation, reconciling portfolio state, reconciling orders, restoring strategy, restoring survival state, identifying unfinished operations, and resuming from valid state. Never assumes previous process completed an operation.
- **Safety Boundaries**: `execute_live_trade` and `activate_live_trading` raise `PermissionError`. NEVER places live trades, connects to live broker, bypasses Risk Manager, bypasses CEO/Orchestrator, bypasses Strategy Backtesting, activates unvalidated strategy, rewrites historical records, or fabricates experiences/market data/death causes/performance. Remains simulation/paper-trading only.
- Important: EXIT_CANDIDATE is ONLY a recommendation/state. Future CEO/orchestration will decide what to do with that recommendation. The Safety Manager never executes orders or directly sells positions.

### Portfolio Synchronization (`app/core/portfolio/`)
- `PortfolioSynchronizer`: Reconciles external account state, cash balances, open positions, and active orders with local portfolio representations, emitting `AccountUpdated` and `PositionUpdated` domain events.

---

## 4. Autonomous Runtime (Step 15)

### SurvivalRuntime (`app/core/runtime/survival_runtime.py`)
The central runtime service that coordinates the full end-to-end paper-trading loop:

```
START -> HEALTH CHECK -> GENERATION START -> OBSERVE (market/news/crisis)
-> RESEARCH (market/news/crisis in parallel via CEO) -> DEEP ANALYSIS
-> RISK CHECK -> DECIDE -> PAPER EXECUTE (if approved) -> PORTFOLIO SYNC
-> INVESTMENT MONITORING -> LEARN -> SURVIVAL CHECK -> REPEAT
-> GENERATION DEATH -> TRANSITION -> SUCCESSOR -> GENERATION 2 START
```

- **State Machine**: Explicit runtime states with valid transitions: STARTING → HEALTH_CHECK → INITIALIZING_GENERATION → OBSERVING → RESEARCHING → ANALYZING → RISK_CHECK → DECIDING → EXECUTING → MONITORING → LEARNING → SURVIVAL_CHECK → GENERATION_TRANSITION → INITIALIZING_GENERATION (successor) → STOPPING → STOPPED. Invalid transitions are rejected.
- **Health Checks**: Layered health verification before autonomous operation:
  1. Infrastructure health: memory store, generation manager, agent registry.
  2. Internet/API health: market data, news, LLM providers.
  3. Paper-trading health: execution provider reachable, account accessible, order endpoint available, environment strictly PAPER.
  Any critical failure puts the runtime into a safe state (no new decisions).
- **Decision Gating**: The runtime does not blindly trade every cycle. `DecisionGate` evaluates configurable reasons to skip a new investment decision: cooldowns, duplicate decisions, insufficient data, exposure limits, crisis conditions, provider outages, and stale market data.
- **Paper Execution**: `PaperExecutionService` wraps the ExecutionProvider abstraction with:
  - Hard paper-mode verification before any submission,
  - Pre-flight OrderValidator checks,
  - Idempotent order submission (client_order_id keyed),
  - Explicit order lifecycle tracking (SUBMITTED → ACCEPTED → FILLED ...),
  - Confirmation from the provider before an order is ever marked FILLED.
  Live trading is architecturally impossible: the ExecutionEnvironment enum contains only PAPER, OrderRequest.__post_init__ rejects any other value, and this service refuses to run unless the paper-only guarantee is verified.
- **Investment Monitoring**: `InvestmentMonitor` runs the Investment Safety Manager for every open investment. EXIT_CANDIDATE recommendations are recorded and flagged for higher-level review — the monitor never executes sells automatically.
- **Learning**: `ExperienceCollector` evaluates outcomes of executed decisions and writes Experience records for the learning loop. `LearningCycle` runs the StrategyUpdaterAgent periodically and records strategy proposals (never activates strategies directly).
- **Cost Accounting**: `CostAccountingService` applies configured operating costs to the active generation. Costs are applied transparently from RuntimeConfig.operating_costs. The service never invents prices: if a configured cost has no amount, it is recorded as UNKNOWN rather than fabricated.
- **Generation Transition**: `GenerationTransitionService` coordinates the full death → successor pipeline:
  DEATH → cancel/reconcile paper orders → freeze generation → death report → experience extraction → learning loop → strategy evaluation → successor request (SUCCESSOR_PENDING if no strategy passes).
  The service never invents a valid strategy: if no proposal passes validation, the generation stays SUCCESSOR_PENDING and the runtime stops new investment activity.
- **Idempotency**: `IdempotencyManager` tracks critical operations (order submission, generation creation, learning cycle, cost application, portfolio sync) so retries never duplicate side effects. A retry after a crash or timeout can never duplicate the operation because the key is checked first.
- **Scheduler**: `RuntimeScheduler` runs named tasks at their configured intervals (portfolio sync, investment monitoring, learning, health checks, cost accounting). Tasks execute sequentially in the caller's thread (deterministic, no overlapping runs of the same task).
- **Restart/Recovery**: `RuntimeRecovery` handles process restarts. On startup: recover runtime state, recover active generation, reconcile portfolio state, reconcile orders, restore strategy, restore survival state, identify unfinished operations, and resume from a valid state. Never assumes the previous process completed anything.
- **Audit Trail**: Every decision leaves a traceable DecisionRecord in memory. Every cycle stores a CycleRecord with triggered actions, agent results, decision result, execution result, portfolio state, and survival state. Every failure is recorded with classification (TRANSIENT/RECOVERABLE/CRITICAL/FATAL).
- **Permission Boundaries**: `execute_live_trade` and `activate_live_trading` raise `PermissionError`. The runtime cannot execute live trades, access live broker functionality, or interact with live trading endpoints.

### Safety Guarantees (Step 15)
1. **Typed Environment**: The `ExecutionEnvironment` enum contains exclusively `PAPER`.
2. **URL Guard**: `AlpacaPaperExecutionProvider` raises `InvalidEnvironmentError` if configured with a live URL.
3. **Double Verification**: Both `OrderValidator` and `PaperOnlyExecutionProvider` check environment validity before network transmission.
4. **Secret Protection**: `SafeFormatter` and `mask_sensitive_data()` sanitize logs to ensure credentials and headers are never written to logs or error outputs.
5. **Prompt Injection Protection**: News, market-data, and crisis source text is wrapped in data delimiters and guarded by system instructions preventing command execution.
6. **Strict Agent Permissions**: Research agents have no order execution or portfolio modification capabilities.
7. **Deterministic numerics**: Market indicators are computed in Python. The LLM may interpret features but must not replace arithmetic, RSI, or moving averages.
8. **Crisis evidence model**: Confirmed facts, reported claims, and interpretations are labeled separately. Severity is evidence-based, not an unconstrained LLM score.
9. **Deep research boundary**: Deep Looker can prepare a dossier for later decision records but cannot approve an investment, modify capital, override Risk Manager, or issue trade instructions.
10. **Risk Manager authority**: The Risk Manager uses deterministic calculations for all risk metrics (position sizes, percentages, exposure, drawdown). Hard risk rules cannot be overridden by LLM output. BLOCKED assessments must never reach execution. The Risk Manager cannot execute orders, modify portfolio state, or bypass paper-trading safety boundaries.
11. **Investment Safety authority**: The Investment Safety Manager uses deterministic calculations for all thesis evaluation metrics (position P/L, drawdown, exposure). Hard thesis invalidations cannot be overridden by LLM output. EXIT_CANDIDATE assessments never reach execution. The Safety Manager cannot execute orders, modify portfolio state, or bypass paper-trading safety boundaries.
12. **Paper-Only Execution**: `PaperExecutionService` verifies paper mode before any submission, validates every order with `OrderValidator`, submits through `PaperOnlyExecutionProvider` (hard PAPER boundary), and never marks an investment filled without provider confirmation.
13. **Idempotent Operations**: `IdempotencyManager` ensures critical operations (order submission, generation creation, learning cycle, cost application, portfolio sync) are never duplicated by retries after crashes or timeouts.
14. **Decision Gating**: `DecisionGate` prevents blind trading every cycle. Cooldowns, exposure limits, crisis conditions, provider outages, and stale market data block new decisions.
15. **Safe Pause**: Any critical health check failure puts the runtime into a safe state (no new decisions). Monitoring of existing positions continues where data is available.
16. **Fail-Safe Configuration**: If paper-trading configuration is missing (no ALPACA_API_KEY/ALPACA_API_SECRET), the production build fails safely. No runtime is returned and nothing executes.
17. **No Fabricated Prices**: `PaperExecutionService` never invents prices. If no reliable price is available from the provider, the order is skipped (SKIPPED_NO_PRICE) rather than executed at a fabricated price.
18. **No Automatic Sells**: EXIT_CANDIDATE recommendations are recorded and flagged for higher-level review. The Investment Monitor never executes sells automatically.
19. **No Strategy Activation**: `LearningCycle` records strategy proposals with status PROPOSED. Activation requires the validation pipeline (backtest, walk-forward, risk evaluation) from Step 12/13.
20. **No Live Trading**: `SurvivalRuntime.execute_live_trade()` and `activate_live_trading()` raise `PermissionError`. Live trading is architecturally impossible in this code path.

---

## 5. Safety Guarantees
1. **Typed Environment**: The `ExecutionEnvironment` enum contains exclusively `PAPER`.
2. **URL Guard**: `AlpacaPaperExecutionProvider` raises `InvalidEnvironmentError` if configured with a live URL.
3. **Double Verification**: Both `OrderValidator` and `PaperOnlyExecutionProvider` check environment validity before network transmission.
4. **Secret Protection**: `SafeFormatter` and `mask_sensitive_data()` sanitize logs to ensure credentials and headers are never written to logs or error outputs.
5. **Prompt Injection Protection**: News, market-data, and crisis source text is wrapped in data delimiters and guarded by system instructions preventing command execution.
6. **Strict Agent Permissions**: Research agents have no order execution or portfolio modification capabilities.
7. **Deterministic numerics**: Market indicators are computed in Python. The LLM may interpret features but must not replace arithmetic, RSI, or moving averages.
8. **Crisis evidence model**: Confirmed facts, reported claims, and interpretations are labeled separately. Severity is evidence-based, not an unconstrained LLM score.
9. **Deep research boundary**: Deep Looker can prepare a dossier for later decision records but cannot approve an investment, modify capital, override Risk Manager, or issue trade instructions.
10. **Risk Manager authority**: The Risk Manager uses deterministic calculations for all risk metrics (position sizes, percentages, exposure, drawdown). Hard risk rules cannot be overridden by LLM output. BLOCKED assessments must never reach execution. The Risk Manager cannot execute orders, modify portfolio state, or bypass paper-trading safety boundaries.
11. **Investment Safety authority**: The Investment Safety Manager uses deterministic calculations for all thesis evaluation metrics (position P/L, drawdown, exposure). Hard thesis invalidations cannot be overridden by LLM output. EXIT_CANDIDATE assessments never reach execution. The Safety Manager cannot execute orders, modify portfolio state, or bypass paper-trading safety boundaries.
12. **Paper-Only Execution**: `PaperExecutionService` verifies paper mode before any submission, validates every order with `OrderValidator`, submits through `PaperOnlyExecutionProvider` (hard PAPER boundary), and never marks an investment filled without provider confirmation.
13. **Idempotent Operations**: `IdempotencyManager` ensures critical operations (order submission, generation creation, learning cycle, cost application, portfolio sync) are never duplicated by retries after crashes or timeouts.
14. **Decision Gating**: `DecisionGate` prevents blind trading every cycle. Cooldowns, exposure limits, crisis conditions, provider outages, and stale market data block new decisions.
15. **Safe Pause**: Any critical health check failure puts the runtime into a safe state (no new decisions). Monitoring of existing positions continues where data is available.
16. **Fail-Safe Configuration**: If paper-trading configuration is missing (no ALPACA_API_KEY/ALPACA_API_SECRET), the production build fails safely. No runtime is returned and nothing executes.
17. **No Fabricated Prices**: `PaperExecutionService` never invents prices. If no reliable price is available from the provider, the order is skipped (SKIPPED_NO_PRICE) rather than executed at a fabricated price.
18. **No Automatic Sells**: EXIT_CANDIDATE recommendations are recorded and flagged for higher-level review. The Investment Monitor never executes sells automatically.
19. **No Strategy Activation**: `LearningCycle` records strategy proposals with status PROPOSED. Activation requires the validation pipeline (backtest, walk-forward, risk evaluation) from Step 12/13.
20. **No Live Trading**: `SurvivalRuntime.execute_live_trade()` and `activate_live_trading()` raise `PermissionError`. Live trading is architecturally impossible in this code path.