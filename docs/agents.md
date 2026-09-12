# Agents

This document describes the agent architecture for SurvivalAI. All agents inherit from a shared `BaseAgent` and communicate via structured data models (`Task` and `AgentResult`).

## BaseAgent
The foundation for all agents. Defines common properties like `agent_id`, `role`, `status`, and `capabilities`. Agents are NOT chatbots; they return structured insights.

---

## NewsResearchAgent (Implemented)
* **Purpose**: Ingests, normalizes, validates, deduplicates, and analyzes market and macro news reports. Extracts verified factual claims and analyzes potential impacts without making investment decisions.
* **Architecture & Pipeline**:
  1. **News Provider Abstraction**: Ingests `NewsItem` records via `NewsProvider` (e.g. `AlpacaNewsProvider`, `MockNewsProvider`).
  2. **Source Validation**: Categorizes sources into `PRIMARY` (regulatory filings, central bank statements, official press releases) vs `SECONDARY` (financial news wires, analyst reports). Computes reliability scores and validates timestamps/URLs.
  3. **Duplicate Detection & Event Clustering**: Groups related reports across multiple outlets into `NewsEventCluster` objects using Jaccard token similarity and symbol association while preserving all independent source citations.
  4. **Contradiction Detection**: Identifies conflicting claims across reporting outlets, flags cluster-level conflicts, and lowers analytical confidence accordingly.
  5. **Recency & Staleness Tracking**: Detects when subsequent reports correct, revise, or supersede earlier reporting, and flags reports older than 7 days as stale.
  6. **LLM Prompt Injection Shield**: Isolates all external text inside `<untrusted_news_data>` delimiters with strict system prompts prohibiting execution of embedded commands.
  7. **Schema Validation & Error Recovery**: Enforces strict JSON output schemas on LLM analysis with automated fallback and structured error logging.
  8. **Memory & Event Integration**: Persists structured facts and analysis to `MemoryStore` as `MemoryRecord` (type `ANALYSIS`), and emits `NewsEvent` domain events.
* **Strict Permission Boundaries**: Research-only agent. Has no authorization or interfaces to place orders, execute trades, modify portfolios, or change strategies.
* **Expected Input**: `Task` with `input_data` containing `symbol` (optional) and `limit`.
* **Expected Output**: Structured `AgentResult` containing:
  - `facts`: `List[Fact]` linked to traceable `source_ids`.
  - `analysis`: Event classification, contextual significance, and interpretation.
  - `impact`: Structured `ImpactDirection` (`POSITIVE`, `NEGATIVE`, `MIXED`, `UNCERTAIN`) and `ImpactHorizon` (`IMMEDIATE`, `SHORT_TERM`, `MEDIUM_TERM`, `LONG_TERM`).
  - `confidence`: Calibrated float from `0.0` to `1.0`.
  - `warnings`: Discrepancies, unconfirmed claims, and superseding updates.
  - `sources`: Complete list of `Source` metadata including URLs and reliability rankings.

---

## MarketResearchAgent (Implemented)
* **Purpose**: Produces structured, objective market research from `MarketDataProvider` data. Research-only: it does not recommend BUY/SELL/HOLD, place orders, modify the portfolio, or override Risk Manager / CEO.
* **Architecture & Pipeline**:
  1. **MarketDataProvider** retrieval of `Quote`, `Trade`, `Bar`, historical bars, and market clock (cached with `retrieved_at` vs `data_timestamp`).
  2. **Data validation** (`DataQualityChecker`): invalid/negative prices, impossible OHLC, duplicate or inconsistent timestamps, missing/zero volume, stale quotes, gaps. Invalid bars are excluded from calculations.
  3. **Historical preparation** using configurable `short` / `medium` / `long` windows (does not pull unbounded history).
  4. **Deterministic features** (`MarketFeatureCalculator`): price, 1/5/20-day returns, SMA 20/50/200, historical volatility, volume ratio, Wilder RSI, rate of change. Missing history yields `None` / `INSUFFICIENT_DATA` — never fabricated values.
  5. **Trend & regime** (`MarketRegimeClassifier`): documented rule-based `MarketTrend` and `MarketRegime`. Not a price forecast.
  6. **Anomaly detection** (`MarketAnomalyDetector`): large moves, volume/volatility spikes, gaps, abnormal spreads — without labeling them bullish or bearish.
  7. **Optional news context** via shared `NewsItem` / `NewsEventCluster` models (no import of `NewsResearchAgent`). Correlation is not treated as causation.
  8. **Optional `FundamentalDataProvider`** interface for future statements/valuation/profile/earnings. No fabricated fundamentals. `AssetCapabilities` records whether earnings, dividends, or statements even apply (stocks vs ETFs vs crypto).
  9. **LLM interpretation** through `LLMProvider` only: summarization and uncertainty. Arithmetic stays in Python. Output is schema-validated and stripped of BUY/SELL/HOLD language.
* **Expected Input**: `Task.input_data` with `symbol` or `symbols`, optional `horizon` (`short`/`medium`/`long`), `timeframe`, `limit`, `news_context`, `generation_id`.
* **Expected Output**: `AgentResult` with typed `Fact`s, analysis (trend, regime, anomalies, comparisons), impact, confidence, warnings, and sources. Also a reusable `MarketSnapshot`. Emits `MarketAnalysisCompleted` on the existing event types — not a second bus.
* **Strict Permission Boundaries**: `execute_order`, `modify_portfolio`, and `change_strategy` raise `PermissionError`.

## CrisisRiskAgent (Implemented)
* **Purpose**: Monitors geopolitical, political, trade, sanctions, energy, supply-chain, macroeconomic, and regulatory developments and produces **structured risk intelligence**. Research-only: no BUY/SELL/HOLD, orders, portfolio/capital changes, or overrides of Risk Manager / CEO.
* **Inputs**: Normalized `NewsItem` / `NewsEventCluster` (preferred path: News Provider → News Research → Crisis Agent). Optional `OfficialDataProvider` statements, optional `MarketSnapshot` for temporal consistency notes, optional `NewsProvider` fallback. Does not scrape websites inside the agent.
* **Pipeline**:
  1. Reuse Step 5 clustering (`NewsDeduplicator`), conflict detection, recency/staleness, and primary vs secondary source weighting (official/IGO sources weighted higher, still treated as **claims**).
  2. Deterministic classification of `CrisisEventType` (primary + secondary), `SeverityLevel`, `EscalationStatus`, `GeographicScope`, `CrisisTimeHorizon`.
  3. Structured **transmission paths** (energy, commodities, inflation, rates, FX, trade, supply chains, etc.) — hypotheses, not proven causation.
  4. **Exposure** by country/region/sector/company/commodity/currency/asset — not trade advice.
  5. **Risk dimensions** (`geopolitical_risk`, `political_risk`, `trade_risk`, `energy_risk`, `supply_chain_risk`, `macroeconomic_risk`, `regulatory_risk`) plus separate `severity` / `probability` / `exposure` / `horizon`. No opaque “Risk = 83”.
  6. Timeline built only from supplied articles. Event **updates** reuse an in-agent registry rather than spawning a duplicate crisis. Related events may be **linked**.
  7. LLM (`LLMProvider`) interprets and summarizes; it must not invent arithmetic, severity without evidence, or investment decisions. External text is wrapped in `<untrusted_external_content>`.
  8. Facts labeled `CONFIRMED_FACT` / `REPORTED_CLAIM` / `ANALYTICAL_INTERPRETATION`. Conflicts are retained. Market notes use “consistent with”, never “definitely caused”.
* **Output**: `AgentResult` plus `GeopoliticalCrisis` payloads in `analysis.events`. Significant events emit existing-bus `CrisisEvent`. Memory stores ANALYSIS/FACT/ERROR only for meaningful intelligence.
* **Downstream**: A future Risk Manager should read risk dimensions and evidence, not a single AI score.
* **Permissions**: `execute_order`, `modify_portfolio`, `modify_capital`, and `change_strategy` raise `PermissionError`.

## DeepLookerAgent
* **Purpose**: Performs high-depth due diligence on an asset or investment candidate before it can reach later decision or risk layers. It combines existing market, news, crisis/geopolitical, and fundamental research into a structured `DeepResearchDossier`.
* **Scope**: Asset-agnostic. Equities may include statements, earnings, margins, leverage, and valuation. ETFs, crypto, commodities, and unknown assets expose only capabilities that are actually available; missing sections remain explicit `INSUFFICIENT_DATA` / `MISSING` rather than fabricated values.
* **Pipeline**:
  1. `DeepResearchRequest` validation (`STANDARD`, `DEEP`, `COMPREHENSIVE`; default `DEEP`).
  2. Asset identification from `CompanyProfile` and `AssetCapabilities`.
  3. Data collection from provider abstractions and supplied `AgentResult`s.
  4. Deterministic financial-health and valuation calculations before LLM synthesis.
  5. Data completeness and freshness flags.
  6. Existing-agent integration from News Research, Market Research, and Crisis Risk results without depending on their internals.
  7. Risk, catalyst, contradiction, thesis, invalidation conditions, and bull/base/bear scenario construction.
  8. LLM synthesis through `LLMProvider` only for contextual reasoning; arithmetic stays in Python.
  9. Evidence/source validation and structured `DeepResearchCompleted` event emission.
* **Deterministic Metrics**: Revenue growth, earnings growth, gross/operating/net margins, debt-to-equity, debt-to-cash, free-cash-flow growth, interest coverage, historical P/E, forward P/E only when an estimate exists, P/S, P/B, EV/EBITDA, and free-cash-flow yield. Formulas are documented in `app/agents/deep_looker/metrics.py`.
* **Contradictions**: The agent actively preserves unresolved tensions, such as positive revenue growth alongside declining free cash flow, or positive market trend alongside elevated crisis risk.
* **Thesis System**: Produces `STRONG_SUPPORT`, `MODERATE_SUPPORT`, `MIXED`, `WEAK_SUPPORT`, or `INSUFFICIENT_DATA`. It never returns BUY, SELL, or HOLD.
* **Source Traceability**: Important conclusions link from dossier section to `Evidence`, then to `Source` with URL/identifier and timestamps where available. Source type distinguishes primary, secondary, and tertiary evidence.
* **Expected Input**: `Task.input_data` with `symbol`, optional `requested_depth`, `historical_window`, `generation_id`, optional `market_snapshot`, `news_research_result`, `market_research_result`, `crisis_research_result`, and optional context.
* **Expected Output**: `AgentResult.analysis["dossier"]` containing the full `DeepResearchDossier`, plus a compact `decision_context` that future `InvestmentRecord` and `DecisionRecord` flows can reference. This does not create an investment.
* **Security**: External data is wrapped as untrusted content in the prompt. Prompt injection in news, descriptions, web content, or metadata is treated as data, not instruction.
* **Permissions**: `execute_order`, `modify_portfolio`, `modify_capital`, `approve_investment`, `override_risk_manager`, and `change_strategy` raise `PermissionError`.

## RiskManagerAgent (Implemented)
* **Purpose**: Deterministic-first risk gate that evaluates whether a proposed investment fits within SurvivalAI's configured risk limits. It is NOT an investment strategist and has no execution or portfolio mutation authority.
* **Core Responsibility**: The Risk Manager receives proposed investments, current portfolio state, available cash, positions, and research outputs from other agents. It evaluates the proposal against hard and soft risk constraints and produces a structured `RiskAssessment`.
* **Decision Types**: The Risk Manager produces one of four decisions:
  - `APPROVED`: Proposal passes all mandatory risk rules
  - `APPROVED_WITH_WARNINGS`: Proposal passes hard limits but has meaningful risk concerns
  - `BLOCKED`: At least one mandatory risk rule is violated
  - `INSUFFICIENT_DATA`: Not enough trustworthy data to make a safe assessment
* **Risk Dimensions Evaluated**:
  - Position size and portfolio concentration
  - Cash reserve protection
  - Sector, asset-class, and geographic diversification
  - Position count limits
  - Volatility and drawdown risk
  - Crisis and geopolitical risk
  - Fundamental and thesis risk (from Deep Looker)
  - Correlation risk
  - Data quality and freshness
* **Deterministic Calculations**: All risk calculations are performed in Python, not by LLMs:
  - Position percentages: `position_value / portfolio_value`
  - Cash after investment: `available_cash - position_value`
  - Concentration exposure: `asset_value / portfolio_value`
  - Sector/asset-class/geographic exposure: `sum(position values in category) / portfolio_value`
  - Drawdown: `(peak - current) / peak`
* **Hard vs Soft Rules**:
  - **HARD RULES**: Violation automatically causes `BLOCKED` (e.g., cash reserve violation, maximum position violation, maximum concentration violation, invalid portfolio state, insufficient critical data)
  - **SOFT WARNINGS**: Does not block but produces `APPROVED_WITH_WARNINGS` (e.g., elevated volatility, moderate geopolitical risk, high valuation uncertainty, elevated correlation)
* **Policy Configuration**: Configurable via `RiskPolicy` model with limits such as:
  - `max_single_position_pct`: Maximum percentage of portfolio for single position
  - `max_portfolio_concentration_pct`: Maximum concentration in one asset
  - `max_sector_exposure_pct`: Maximum exposure to any sector
  - `max_asset_class_exposure_pct`: Maximum exposure to any asset class
  - `max_geographic_exposure_pct`: Maximum exposure to any geography
  - `min_cash_reserve_pct`: Minimum cash reserve percentage
  - `max_portfolio_drawdown_pct`: Maximum allowed portfolio drawdown
  - `volatility_warning_threshold`: Volatility level that triggers warning
  - `volatility_block_threshold`: Volatility level that blocks investment
  - `crisis_block_severity`: Crisis severity that triggers block
  - `max_correlated_exposure_pct`: Maximum exposure to correlated assets
  - `max_positions`: Maximum number of simultaneous positions
  - `min_data_confidence`: Minimum required data confidence
  - `stale_market_data_minutes`: Age threshold for stale market data
  - `require_market_data`: Whether market data is mandatory
  - `require_deep_research`: Whether deep research is mandatory
  - `block_on_severe_crisis`: Whether to block on severe crisis exposure
* **Research Integration**: Consumes structured outputs from:
  - **Market Research**: Volatility, regime, anomalies, market snapshot
  - **News Research**: Negative events, regulatory changes, legal events
  - **Crisis Risk**: Severity, escalation, geographic exposure, transmission mechanisms
  - **Deep Looker**: Thesis confidence, data sufficiency, fundamental risk, contradictions, thesis status
* **LLM Usage**: The LLM is used only for:
  - Summarizing risk findings in natural language
  - Explaining warnings
  - Identifying qualitative contradictions
  - The LLM must NOT calculate risk limits, override hard rules, approve blocked investments, or invent missing data
* **Safety Hierarchy**: Research Agents → Deep Looker → Risk Manager → Future CEO/Orchestrator → Future Paper Execution. The Risk Manager is a safety gate that future components cannot bypass.
* **Expected Input**: `Task.input_data` containing:
  - `proposal` or `investment_proposal`: `InvestmentProposal` with asset, proposed_position_value, classifications
  - `portfolio_state`: `PortfolioRiskState` with portfolio_value, available_cash, positions
  - `risk_policy` (optional): `RiskPolicy` configuration override
  - `market_snapshot` (optional): `MarketSnapshot` from Market Research
  - `news_research_result` (optional): `AgentResult` from News Research
  - `crisis_research_result` (optional): `AgentResult` from Crisis Risk
  - `deep_research_result` or `deep_looker_result` (optional): `AgentResult` from Deep Looker
  - `generation_id`: Current generation identifier
* **Expected Output**: `AgentResult` with:
  - `analysis["risk_assessment"]`: Full `RiskAssessment` object
  - `analysis["decision"]`: Risk decision string
  - `analysis["execution_allowed"]`: Boolean indicating if execution is permitted
  - `analysis["llm_explanation"]`: Optional LLM-generated explanation
  - `warnings`: List of risk warnings
  - Emits `RiskAssessmentStarted`, `RiskAssessmentCompleted`, and `RiskViolationEvent` events
  - Stores assessment in `MemoryStore` as `MemoryRecord` type `ANALYSIS`
* **Strict Permission Boundaries**: `execute_order`, `modify_portfolio`, `modify_capital`, `change_strategy`, and `approve_investment` raise `PermissionError`. The Risk Manager cannot execute trades, directly modify portfolio state, or bypass paper-trading safety boundaries.

## InvestmentSafetyAgent (Implemented)
* **Purpose**: Monitors existing investments and evaluates whether their original investment theses remain valid. It is NOT the same as the Risk Manager.
* **Difference from Risk Manager**: 
  - Risk Manager: "Can we safely ENTER this position?" (evaluates new proposed investments)
  - Investment Safety Manager: "Does the thesis for this EXISTING position still hold?" (monitors existing positions)
* **Core Responsibility**: For every active investment, evaluates original thesis, current market conditions, news, crisis conditions, drawdown, contradictions, and time horizon.
* **Decision Types**: Produces one of four decisions:
  - `HOLD`: Original thesis remains sufficiently supported, no critical invalidation triggered
  - `REVIEW`: Thesis has weakened, evidence mixed, risk increased, or important conditions changed
  - `EXIT_CANDIDATE`: Significant thesis invalidation or severe deterioration detected
  - `INSUFFICIENT_DATA`: Not enough reliable information to safely evaluate
* **Important**: `EXIT_CANDIDATE` does NOT mean execute a sell. It means the investment requires higher-level review. Future CEO/orchestration will decide what to do with that recommendation.
* **Thesis Status**: Explicit thesis status enum with `STRONGLY_SUPPORTED`, `SUPPORTED`, `MIXED`, `WEAKENING`, `INVALIDATED`, `UNKNOWN`. The system explains WHY the thesis has its current state.
* **Thesis Preservation**: The original investment thesis remains immutable. Current assessment is separate from original thesis. The system never silently rewrites the original thesis based on current information.
* **Thesis Invalidation Framework**: Structured representation of thesis invalidation conditions with `ThesisCondition` containing condition_id, category (FUNDAMENTAL, MARKET, VALUATION, BUSINESS, COMPETITIVE, MACRO, GEOPOLITICAL, REGULATORY, MANAGEMENT, LIQUIDITY, OTHER), status (VALID, WEAKENING, VIOLATED, UNKNOWN), severity, source, and observed/expected values.
* **Contradiction Detection**: Explicit detection of contradictions between original thesis and current evidence. Examples: "Strong revenue growth expected" vs "Revenue growth has materially deteriorated", or "Company benefits from declining interest rates" vs "Interest rates rising."
* **Deterministic Calculations**: All calculations performed in Python: position percentages, cash after investment, concentration exposure, sector/asset-class/geographic exposure, drawdown. LLM may interpret but must not calculate.
* **Research Integration**: Consumes structured outputs from Market Research (volatility, regime, anomalies), News Research (negative events, regulatory changes), Crisis Risk (severity, escalation, exposure), and Deep Looker (thesis confidence, contradictions, fundamental risk).
* **Previous Assessment Tracking**: Compares current assessment with previous assessments to detect trends. Tracks previous recommendation, thesis status, new/resolved/worsening/improving risks, and contradictions.
* **Time Horizon Awareness**: Respects SHORT_TERM, MEDIUM_TERM, LONG_TERM investment horizons. Long-term investments place greater emphasis on fundamentals and business conditions; shorter horizons may weigh market conditions and catalysts more heavily.
* **Price vs Thesis Damage**: Separates PRICE DAMAGE from THESIS DAMAGE. A declining price alone must NOT automatically trigger EXIT_CANDIDATE. Price decline + thesis invalidation may be appropriate.
* **LLM Usage**: LLM used only for synthesizing structured evidence, explaining contradictions, summarizing thesis changes, and qualitative interpretation. LLM must NOT invent facts, rewrite original thesis, directly sell positions, place orders, override deterministic safety rules, or fabricate missing data.
* **Safety Hierarchy**: Research Agents → Deep Looker → Risk Manager → Investment Safety Manager → Future CEO/Orchestrator → Future Paper Execution. The Safety Manager is a secondary safety gate for existing positions.
* **Expected Input**: `Task.input_data` containing:
  - `investment_record`: `InvestmentRecord` with asset, entry_price, position_size, investment_thesis, time_horizon, risk_level
  - `current_price`: Current market price
  - `current_position_value`: Current position value
  - `market_snapshot` (optional): `MarketSnapshot` from Market Research
  - `news_research_result` (optional): `AgentResult` from News Research
  - `crisis_research_result` (optional): `AgentResult` from Crisis Risk
  - `deep_research_result` or `deep_looker_result` (optional): `AgentResult` from Deep Looker
  - `previous_assessment_id` (optional): ID of previous safety assessment for comparison
  - `generation_id`: Current generation identifier
* **Expected Output**: `AgentResult` with:
  - `analysis["safety_assessment"]`: Full `InvestmentSafetyAssessment` object
  - `analysis["recommendation"]`: Safety recommendation string
  - `analysis["thesis_status"]`: Current thesis status
  - `analysis["requires_review"]`: Boolean indicating if higher-level review needed
  - `analysis["is_critical"]`: Boolean indicating if critical (EXIT_CANDIDATE)
  - Emits `SafetyAssessmentStarted`, `SafetyAssessmentCompleted`, `ThesisWeakenedEvent`, `ThesisInvalidatedEvent`, `ReviewRequiredEvent`, and `ExitCandidateDetectedEvent` events
  - Stores assessment in `MemoryStore` as `MemoryRecord` type `ANALYSIS`
* **Strict Permission Boundaries**: `execute_order`, `modify_portfolio`, `modify_capital`, `change_strategy`, and `approve_investment` raise `PermissionError`. The Safety Manager cannot execute trades, directly modify portfolio state, or bypass paper-trading safety boundaries.

## CEOAgent (Implemented)
* **Purpose**: Central coordination layer that turns specialized-agent research into controlled, explainable investment workflows. Coordinates agents but does NOT execute trades. Produces decision proposals only.
* **Architecture & Pipeline**:
  1. **Agent Registry**: Dependency-injected registry for available agents. CEO resolves agents by stable identifier rather than hard-coding dependencies. Agents can be enabled/disabled dynamically.
  2. **Orchestration Request**: `OrchestrationRequest` captures request_id, generation_id, request_type (INVESTMENT_PROPOSAL, EXISTING_INVESTMENT_REVIEW, MARKET_EVENT, CRISIS_REASSESSMENT, PORTFOLIO_REVIEW), asset, objective, timestamp, priority, requested_position_size, context, and metadata.
  3. **State Machine**: Explicit orchestration stages with valid transitions: CREATED → VALIDATING → RESEARCHING → DEEP_ANALYSIS → RISK_ASSESSMENT/SAFETY_ASSESSMENT → DECISION_SYNTHESIS → COMPLETED/FAILED/BLOCKED/DEFERRED. Invalid transitions are rejected.
  4. **New Investment Pipeline**: REQUEST → VALIDATION → RESEARCHING (Market Research, News Research, Crisis Risk in parallel) → DEEP ANALYSIS (Deep Looker receives upstream outputs) → EVIDENCE AGGREGATION → RISK ASSESSMENT (always required) → DECISION SYNTHESIS → MEMORY + EVENTS.
  5. **Existing Investment Review Pipeline**: REQUEST → VALIDATION → SAFETY ASSESSMENT (Investment Safety Manager) → EVIDENCE AGGREGATION → DECISION SYNTHESIS → MEMORY + EVENTS.
  6. **Evidence Aggregation**: Preserves source agent identity, timestamps, and identifiers. Distinguishes facts from interpretations. Tracks supporting and opposing evidence. Makes evidence traceable to originating agent/result. Records missing data and warnings.
  7. **Conflict Detection**: Detects conflicting outputs between agents (e.g., Market Research says favorable while Crisis Risk reports severe exposure). Conflicts are recorded in the orchestration run and decision proposal rather than silently resolved.
  8. **Risk Manager Gate**: Always routes new-investment proposals through Risk Manager. BLOCKED prevents INVEST. INSUFFICIENT_DATA prevents approval and results in DEFER or INSUFFICIENT_DATA. APPROVED/APPROVED_WITH_WARNINGS may allow INVEST with preserved warnings. Cannot override hard risk rules.
  9. **Deep Looker Integration**: Deep Looker receives upstream outputs from Market Research, News Research, Crisis Risk, and the original request/context. Does not fabricate missing research.
  10. **MemoryStore Integration**: Persists orchestration run records, decision proposals, and important stage results as `MemoryRecord` with type `DECISION`. Makes it possible to reconstruct complete orchestration from stored records.
  11. **Event Bus Integration**: Emits typed orchestration events: `OrchestrationStarted`, `AgentDispatched`, `AgentCompleted`, `AgentFailed`, `DeepAnalysisStarted`, `RiskAssessmentStarted`, `DecisionProposalCreated`, `InvestmentBlocked`, `OrchestrationCompleted`, `OrchestrationFailed`, `ConflictDetected`.
  12. **Failure Handling**: Explicitly handles missing critical research, agent failures, agent timeouts, agent cancellation, missing registry entries, invalid agent results, conflicting outputs, and duplicate requests. Failures are represented in typed run fields and structured errors.
* **Decision Types**: For new investments: INVEST, DO_NOT_INVEST, DEFER, INSUFFICIENT_DATA. For existing investments: HOLD, REVIEW, EXIT_CANDIDATE, INSUFFICIENT_DATA. These are decision proposals and never broker orders.
* **Permission Boundaries**: `execute_order`, `modify_portfolio`, `modify_capital`, `change_strategy`, and `approve_investment` raise `PermissionError`. The CEO cannot execute trades, access live trading, bypass Risk Manager, or treat decision proposals as orders.
* **LLM Constraints**: If an LLM is used, it may summarize or explain but cannot override deterministic safety rules, approve a Risk Manager-blocked proposal, invent missing data, convert EXIT_CANDIDATE into an execution command, change state transitions, or manufacture evidence or source attribution.
* **STOP BEFORE EXECUTION**: The CEO produces decision proposals only. It does not execute orders, access live broker functionality, or interact with execution providers. Future Step 12+ will handle paper execution separately.
* **Expected Input**: `Task` with `input_data` containing request_type, asset, objective, generation_id, optional requested_position_size, and context.
* **Expected Output**: Structured `AgentResult` containing:
  - `decision_proposal`: Full `DecisionProposal` with decision_type, thesis, evidence, risk_assessment, confidence, and decision_reason.
  - `agent_executions`: List of all agent executions with status, duration, and results.
  - `conflicts`: List of detected conflicts between agent outputs.
  - `summary`: Human-readable summary of the orchestration and decision.

## StrategyUpdaterAgent (Implemented)
* **Purpose**: Analyzes historical performance and proposes strategy improvements based on evidence. NEVER directly modifies the active strategy.
* **Architecture & Pipeline**:
  1. **Strategy Versioning**: Uses existing `StrategyVersion` architecture with strategy_id, version, parent_strategy_id, generation_id, created_at, parameters, rules, description, and status (ACTIVE, TESTING, REJECTED, ARCHIVED). The currently active strategy is always explicitly identifiable.
  2. **Strategy Structure**: Strongly typed strategy representation including asset universe, allowed asset classes, research requirements, minimum thesis strength, position sizing rules, diversification rules, cash reserve requirements, risk tolerance, maximum drawdown tolerance, market regime rules, crisis response rules, entry conditions, exit/review conditions, holding periods, and portfolio constraints. Strategy configuration is centralized, not hardcoded.
  3. **StrategyChangeProposal**: Proposal for changing strategy with proposal_id, parent_strategy_id, proposed_parameters, changed_rules, unchanged_rules, motivation, supporting_experiences, supporting_decisions, supporting_outcomes, expected_effect, risks, assumptions, confidence, backtest_required, and status (PROPOSED, BACKTESTING, PASSED, FAILED, REJECTED, APPROVED_FOR_SIMULATION, ACTIVE). ACTIVE status is not directly reachable from PROPOSED.
  4. **Learning from Experience**: Consumes DecisionRecord, InvestmentRecord, Experience, DeathReport, AgentPerformanceRecord, historical market data, historical news/event data, previous strategy versions, backtest results, risk assessments, and investment safety assessments. Identifies repeated losses, repeated successful patterns, excessive concentration, poor market-regime performance, excessive drawdowns, poor crisis handling, weak thesis selection, excessive trading, underinvestment, overinvestment, strategy weaknesses, and agent weaknesses.
  5. **No Automatic Overfitting**: Avoids changing strategy based on single random results. Uses minimum evidence requirements: minimum number of observations, minimum number of trades, minimum confidence, minimum performance difference, and statistical significance where practical.
  6. **Memory Integration**: Stores proposals as `MemoryRecord` with type `STRATEGY` for future evaluation and backtesting.
  7. **Proposal Creation**: Creates proposals based on identified weaknesses with supporting evidence, expected effects, risks, and assumptions. Confidence and backtest_required flag ensure proposals are validated before activation.
* **Permission Boundaries**: `activate_strategy` and `modify_active_strategy` raise `PermissionError`. The Strategy Updater cannot directly activate strategies or modify the active strategy.
* **LLM Constraints**: LLM may be used for identifying patterns, proposing hypotheses, explaining failures, summarizing backtest findings, and generating candidate rule changes. LLM must NOT directly activate strategies, modify active strategy state, fabricate historical data, perform financial calculations, override risk constraints, access broker execution, or invent performance results. All calculations are deterministic Python.
* **Expected Input**: `Task` with `input_data` containing generation_id and optional force_proposal flag.
* **Expected Output**: Structured `AgentResult` containing:
  - `proposal`: Full `StrategyChangeProposal` with proposed parameters, changed rules, motivation, supporting evidence, expected effects, risks, assumptions, confidence, and status.
  - `weaknesses`: List of identified strategy weaknesses.
  - `supporting_evidence_count`: Number of supporting experiences/decisions.
  - `summary`: Human-readable summary of the analysis and proposal.

## SurvivalRuntime (Implemented)
* **Purpose**: Central runtime service for autonomous SurvivalAI operation. Coordinates the entire system including startup, health checks, generation initialization, market-data cycles, research cycles, decision cycles, execution cycles, monitoring cycles, learning cycles, survival checks, generation transitions, and graceful shutdown.
* **Architecture & Pipeline**:
  1. **RuntimeState**: Typed runtime state machine with states: STARTING, HEALTH_CHECK, INITIALIZING_GENERATION, OBSERVING, RESEARCHING, ANALYZING, RISK_CHECK, DECIDING, EXECUTING, MONITORING, LEARNING, SURVIVAL_CHECK, GENERATION_TRANSITION, PAUSED, STOPPING, STOPPED, ERROR. Invalid state transitions are rejected.
  2. **Autonomous Cycle**: Controlled recurring cycle: OBSERVE → RESEARCH → ANALYZE → RISK CHECK → DECIDE → PAPER EXECUTE IF APPROVED → MONITOR → LEARN → SURVIVAL CHECK → NEXT CYCLE. Cycle interval is configurable.
  3. **Market Data Collection**: Uses existing MarketDataProvider to retrieve current market data, validate timestamps, detect stale data, detect provider failures, handle rate limits, handle temporary outages, and cache where appropriate. Never fabricates missing market data.
  4. **News Collection**: Uses existing NewsProvider and News Research Agent to periodically update relevant news, event clusters, market-moving events, company events, macro events, and geopolitical events. Respects source quality, freshness, deduplication, conflicts, and prompt-injection protection.
  5. **Crisis Data Collection**: Uses existing Crisis & Geopolitical Risk Agent to monitor wars, military conflicts, sanctions, elections, government changes, tariffs, trade restrictions, energy disruptions, supply-chain disruptions, central-bank events, inflation, recession risks, regulatory changes, major infrastructure failures, cyber incidents, and other configured crisis events.
  6. **Research Pipeline**: Uses existing architecture with Market Research, News Research, and Crisis Risk running in parallel where appropriate, followed by Deep Looker. Uses configurable triggers such as new investment candidate, major news event, major crisis event, thesis contradiction, unusual market movement, existing investment deterioration.
  7. **Investment Decision Pipeline**: For new investments: REQUEST → INPUT VALIDATION → MARKET RESEARCH → NEWS RESEARCH → CRISIS RISK → DEEP LOOKER → RISK MANAGER → CEO/ORCHESTRATOR → DECISION. CEO cannot bypass Risk Manager. If Risk Manager returns BLOCKED, investment must not execute. If INSUFFICIENT_DATA, system must not pretend investment is safe.
  8. **Paper Execution**: If decision is approved, uses existing ExecutionProvider. Execution remains behind existing abstraction. If Alpaca is configured, ONLY uses Alpaca paper-trading environment. Runtime verifies paper mode before execution with explicit safety check.
  9. **Portfolio Synchronization**: After paper execution, synchronizes portfolio state tracking cash, equity, positions, average entry price, current price, unrealized P/L, realized P/L, exposure, available buying power, open orders, filled orders, cancelled orders, and rejected orders. Uses provider abstraction.
  10. **Order Lifecycle**: Implements/reuses CREATED → SUBMITTED → ACCEPTED → PARTIALLY_FILLED → FILLED and failure paths: REJECTED, CANCELLED, EXPIRED, FAILED. Runtime reconciles actual provider state. Never marks investment as filled without confirmation.
  11. **Existing Investment Monitoring**: Periodically runs Investment Safety Manager for every active investment, monitoring thesis, fundamentals, valuation, market conditions, news, crisis conditions, drawdown, contradictions, and time horizon. Investment Safety Manager may return HOLD, REVIEW, EXIT_CANDIDATE, or INSUFFICIENT_DATA. Does not automatically sell simply because EXIT_CANDIDATE is returned unless existing architecture explicitly permits it.
  12. **Decision Frequency Gating**: Implements decision gating to avoid blindly trading every cycle. Reasons to avoid new decision: no meaningful market change, insufficient data, no candidate, existing portfolio already sufficiently exposed, risk limits, crisis conditions, cooldown, duplicate decision, provider outage, recent decision on same asset, or strategy constraints.
  13. **Idempotency**: Critical operations are idempotent including investment request, order submission, portfolio synchronization, generation transition, and learning cycle. Uses unique operation/request IDs to prevent duplicate investments or duplicate generations on temporary retry.
  14. **Failure Handling**: Implements graceful failure handling classifying failures as TRANSIENT, RECOVERABLE, CRITICAL, or FATAL. Possible failures: API outage, network timeout, rate limit, malformed API response, stale data, provider authentication failure, paper-trading outage, LLM failure, agent timeout, invalid model output, database/memory failure, or event failure. Does not crash entire system because of temporary provider outage.
  15. **Safe Pause**: Implements safe pause state when market data is stale, critical API unavailable, paper provider unavailable, Risk Manager unavailable, required research unavailable, configuration invalid, or system integrity check fails. Existing positions can continue to be monitored where data is available. Does not create new investments while critical safety infrastructure is unavailable.
  16. **Survival Check**: After each meaningful cycle evaluates capital, equity, peak equity, current drawdown, maximum drawdown, operating costs, realized P/L, unrealized P/L, risk violations, and configured survival conditions. If death condition is reached: STOP NEW INVESTMENT ACTIVITY → CANCEL/RECONCILE APPROPRIATE PAPER ORDERS → FREEZE GENERATION → CREATE DEATH REPORT → START LEARNING LOOP → STRATEGY EVALUATION → VALIDATION → SUCCESSOR GENERATION. Does not immediately continue operating the dead generation.
  17. **Operating Costs**: Accounts for configured operating costs including model/API usage, market-data costs, news-data costs, infrastructure, and other configured expenses. Costs are represented transparently. If costs are unavailable, marked as unknown rather than fabricated.
  18. **Generation Transition**: When generation dies, uses existing Step 13 + Step 14 architecture: DEATH → DEATH REPORT → EXPERIENCE EXTRACTION → LESSON VALIDATION → PATTERN ANALYSIS → STRATEGY PROPOSAL → BACKTEST → WALK-FORWARD VALIDATION → RISK EVALUATION → STRATEGY COMPARISON → APPROVAL/REJECTION → SUCCESSOR GENERATION. If no strategy passes validation, enters SUCCESSOR_PENDING or another safe configured state. Does not invent a valid strategy.
  19. **Internet Connectivity**: Designed for real internet-connected operation using Internet → External APIs → Provider Layer → Normalized Data → Agents → Orchestrator → Risk Manager → Paper Trading. Does not replace real external providers with hardcoded demo data. Tests may use mock providers. Production configuration uses real providers with provider health checks.
  20. **API Configuration**: Uses environment variables for market API credentials, news API credentials, LLM credentials, and paper-trading credentials. NEVER hardcodes secrets. Creates/updates `.env.example` with placeholder names only. Never logs credentials or exposes secrets through logs, events, memory, agent output, or dashboard APIs.
  21. **Scheduling**: Configurable scheduler/orchestrator supporting different task frequencies: market polling, news polling, crisis monitoring, portfolio synchronization, investment monitoring, learning, cost accounting, and health checks. Does not make every component run at the same interval. Avoids uncontrolled concurrent executions.
  22. **Concurrency**: Uses controlled concurrency. Parallelizes independent research tasks where safe. Prevents concurrent conflicting operations such as two investments against the same decision, duplicate generation creation, simultaneous strategy activation, or conflicting portfolio synchronization. Uses locks or idempotency mechanisms where necessary.
  23. **Observability**: Adds structured runtime logging where every cycle has cycle_id, generation_id, timestamp, runtime state, triggered actions, agent results, decision result, execution result, portfolio state, survival state, errors, and warnings. Does not log secrets. Creates structured event records rather than relying only on plain text logs.
  24. **Audit Trail**: Every investment decision is traceable: market/news/crisis evidence → research outputs → Deep Looker → Risk Manager → CEO → DecisionRecord → execution → portfolio outcome → Experience → Learning. A user can later answer "Why did SurvivalAI make this investment?" with evidence-backed records.
  25. **No Hallucinations**: LLMs must never fabricate market prices, news, API responses, portfolio values, investment outcomes, strategy results, backtest results, or crisis events. If information is unavailable, returns INSUFFICIENT_DATA or appropriate failure state.
  26. **Prompt Injection Defense**: External news/web/API content is untrusted data. Never allows external content to modify system instructions, execute commands, change risk limits, change strategy, bypass agents, request secrets, place orders, or modify generation state. Treats external content strictly as data.
  27. **Test Mode**: Implements controlled TEST/SIMULATION mode using deterministic mock market provider, deterministic mock news provider, deterministic mock crisis provider, and simulated execution provider. Allows entire autonomous loop to be tested without internet or real paper orders. Production mode remains internet-connected and paper-trading-only.
  28. **Paper Mode Health Check**: Before autonomous execution, verifies provider reachable, authentication valid, environment is paper, account accessible, portfolio accessible, order endpoint available, and no live endpoint configured. If any critical check fails, DOES NOT EXECUTE.
  29. **RuntimeRecovery**: Handles runtime restart and recovery including recovering runtime state from memory, recovering active generation, reconciling portfolio state, reconciling orders, restoring strategy, restoring survival state, identifying unfinished operations, and resuming from valid state. Never assumes previous process completed an operation.
* **Permission Boundaries**: `execute_live_trade` and `activate_live_trading` raise `PermissionError`. NEVER places live trades, connects to live broker, bypasses Risk Manager, bypasses CEO/Orchestrator, bypasses Strategy Backtesting, activates unvalidated strategy, rewrites historical records, or fabricates experiences/market data/death causes/performance.
* **Paper Trading Boundary**: May create generations operating against existing simulated/paper portfolio. Uses ONLY configured Alpaca paper-trading environment, never live trading endpoint, no configuration switch to silently redirect to live trading. Generation creation fails safely if paper-trading configuration is invalid.
* **LLM Constraints**: LLMs may help analyze DeathReports, extract lessons, propose hypotheses, explain why strategy might improve, and identify patterns. LLMs must NOT decide generation death, calculate financial metrics, override death conditions, activate strategies, bypass backtesting, override Risk Manager hard rules, fabricate historical data/experiences, or rewrite immutable historical records. Deterministic code remains authoritative.

## SurvivalRuntime (Step 15)
* **Purpose**: Central runtime service that coordinates the full end-to-end paper-trading loop. It is NOT a chatbot, NOT a trading bot, and NOT a live-trading system. It is the autonomous orchestrator that connects all agents, providers, and services into one continuously operating system.
* **Architecture & Pipeline**:
  1. **State Machine**: Explicit runtime states with valid transitions: STARTING → HEALTH_CHECK → INITIALIZING_GENERATION → OBSERVING → RESEARCHING → ANALYZING → RISK_CHECK → DECIDING → EXECUTING → MONITORING → LEARNING → SURVIVAL_CHECK → GENERATION_TRANSITION → INITIALIZING_GENERATION (successor) → STOPPING → STOPPED. Invalid transitions are rejected.
  2. **Health Checks**: Layered health verification before autonomous operation:
     - Infrastructure health: memory store, generation manager, agent registry.
     - Internet/API health: market data, news, LLM providers.
     - Paper-trading health: execution provider reachable, account accessible, order endpoint available, environment strictly PAPER.
     Any critical failure puts the runtime into a safe state (no new decisions).
  3. **Decision Gating**: The runtime does not blindly trade every cycle. `DecisionGate` evaluates configurable reasons to skip a new investment decision: cooldowns, duplicate decisions, insufficient data, exposure limits, crisis conditions, provider outages, and stale market data.
  4. **Paper Execution**: `PaperExecutionService` wraps the ExecutionProvider abstraction with:
     - Hard paper-mode verification before any submission,
     - Pre-flight OrderValidator checks,
     - Idempotent order submission (client_order_id keyed),
     - Explicit order lifecycle tracking (SUBMITTED → ACCEPTED → FILLED ...),
     - Confirmation from the provider before an order is ever marked FILLED.
     Live trading is architecturally impossible: the ExecutionEnvironment enum contains only PAPER, OrderRequest.__post_init__ rejects any other value, and this service refuses to run unless the paper-only guarantee is verified.
  5. **Investment Monitoring**: `InvestmentMonitor` runs the Investment Safety Manager for every open investment. EXIT_CANDIDATE recommendations are recorded and flagged for higher-level review — the monitor never executes sells automatically.
  6. **Learning**: `ExperienceCollector` evaluates outcomes of executed decisions and writes Experience records for the learning loop. `LearningCycle` runs the StrategyUpdaterAgent periodically and records strategy proposals (never activates strategies directly).
  7. **Cost Accounting**: `CostAccountingService` applies configured operating costs to the active generation. Costs are applied transparently from RuntimeConfig.operating_costs. The service never invents prices: if a configured cost has no amount, it is recorded as UNKNOWN rather than fabricated.
  8. **Generation Transition**: `GenerationTransitionService` coordinates the full death → successor pipeline:
     DEATH → cancel/reconcile paper orders → freeze generation → death report → experience extraction → learning loop → strategy evaluation → successor request (SUCCESSOR_PENDING if no strategy passes).
     The service never invents a valid strategy: if no proposal passes validation, the generation stays SUCCESSOR_PENDING and the runtime stops new investment activity.
  9. **Idempotency**: `IdempotencyManager` tracks critical operations (order submission, generation creation, learning cycle, cost application, portfolio sync) so retries never duplicate side effects. A retry after a crash or timeout can never duplicate the operation because the key is checked first.
  10. **Scheduler**: `RuntimeScheduler` runs named tasks at their configured intervals (portfolio sync, investment monitoring, learning, health checks, cost accounting). Tasks execute sequentially in the caller's thread (deterministic, no overlapping runs of the same task).
  11. **Restart/Recovery**: `RuntimeRecovery` handles process restarts. On startup: recover runtime state, recover active generation, reconcile portfolio state, reconcile orders, restore strategy, restore survival state, identify unfinished operations, and resume from a valid state. Never assumes the previous process completed anything.
  12. **Audit Trail**: Every decision leaves a traceable DecisionRecord in memory. Every cycle stores a CycleRecord with triggered actions, agent results, decision result, execution result, portfolio state, and survival state. Every failure is recorded with classification (TRANSIENT/RECOVERABLE/CRITICAL/FATAL).
* **Permission Boundaries**: `execute_live_trade` and `activate_live_trading` raise `PermissionError`. The runtime cannot execute live trades, access live broker functionality, or interact with live trading endpoints.
* **Expected Input**: `RuntimeConfig` with cycle_interval_seconds, watched_symbols, proposed_position_pct, decision_cooldown_seconds, max_open_positions, max_open_orders, min_research_confidence, max_consecutive_critical_failures, max_cycles, operating_costs, paper_mode_required, and test_mode.
* **Expected Output**: Continuous autonomous operation with:
  - Health check results (layered: infrastructure, API, paper-trading)
  - Decision gate results (allowed/blocked with reasons)
  - Paper execution results (idempotent, provider-confirmed fills)
  - Investment monitoring results (HOLD/REVIEW/EXIT_CANDIDATE recommendations)
  - Learning results (experiences collected, strategy proposals recorded)
  - Cost accounting results (transparent operating costs)
  - Generation transition results (death → successor pipeline)
  - Audit trail (decision records, cycle records, failure records)

---

## CryptoResearchAgent (Implemented)
* **Purpose**: Specialized research agent for cryptocurrency markets, focusing on crypto-specific metrics, tokenomics, on-chain data, and crypto risk factors. Research-only: does NOT independently buy, sell, execute orders, modify risk limits, or activate strategies.
* **Architecture & Pipeline**:
  1. **Crypto-Specific Data Models**: Extended models for crypto including CryptoTokenomics (supply, inflation, market cap), OnChainMetrics (active addresses, transaction volume, exchange flows), CryptoRiskFactor (extreme volatility, liquidity risk, smart contract risk, regulatory risk, etc.), CryptoMarketStructure (trend, momentum, volatility, regime), CryptoMarketWideConditions (total market cap, BTC dominance, fear/greed index), and CryptoAssetCorrelation.
  2. **Crypto Provider Abstractions**: CryptoMarketDataProvider extends MarketDataProvider with crypto-specific capabilities (24/7 markets, market-wide conditions, correlations). CryptoFundamentalDataProvider provides tokenomics and unlock schedules. OnChainDataProvider provides blockchain metrics. MockCryptoDataProvider provides deterministic test data.
  3. **Market Structure Assessment**: Deterministic classification of crypto market structure including trend (UPTREND, DOWNTREND, SIDEWAYS), momentum (BULLISH, BEARISH), volatility (EXTREME, HIGH, MODERATE, LOW), liquidity, drawdown, and regime (BULL_MARKET, BEAR_MARKET, ACCUMULATION, etc.).
  4. **Crypto Risk Factors**: Structured risk factor identification including EXTREME_VOLATILITY, LIQUIDITY_RISK, EXCHANGE_COUNTERPARTY_RISK, SMART_CONTRACT_RISK, PROTOCOL_RISK, REGULATORY_RISK, CUSTODY_RISK, CONCENTRATION_RISK, TOKEN_UNLOCK_RISK, STABLECOIN_RISK, NETWORK_OUTAGE_RISK, SECURITY_INCIDENT_RISK, MARKET_MANIPULATION_RISK, CORRELATION_RISK, DEPEG_RISK. Each risk has type, evidence, severity, confidence, time horizon, and affected assets.
  5. **Tokenomics Analysis**: When available, analyzes circulating supply, maximum supply, inflation rate, emission schedule, market capitalization, fully diluted valuation, burn mechanism, staking enabled/rate, and governance token status.
  6. **On-Chain Metrics**: When available, analyzes active addresses 24h, transaction count 24h, transaction volume 24h, average transaction value, exchange inflow/outflow, net exchange flow, whale transactions, large holders count, hashrate (PoW), staked amount, TVL (DeFi), and protocol revenue.
  7. **Market-Wide Conditions**: Tracks total crypto market cap, BTC dominance, ETH dominance, stablecoin market cap, total market trend, fear/greed index, funding rates, and open interest.
  8. **Token Unlock Risk**: Identifies upcoming token unlock events (team, investors, ecosystem) with amount, percentage of supply, unlock date, and cliff vesting. Flags unlocks within 30 days as risk factors.
  9. **Asset Correlations**: Tracks correlations between crypto assets (e.g., BTC-ETH) with 30d and 90d correlation metrics and beta.
  10. **Integration with General Architecture**: Uses existing BaseAgent, AgentConfig, AgentResult, Task, Fact, Source models. Integrates with CEO/Orchestrator through AgentRegistry. Registered with agent_type="crypto" for dispatch logic.
* **Permission Boundaries**: Does NOT independently buy, sell, execute orders, modify risk limits, or activate strategies. Output is research information for Deep Looker / Risk Manager / CEO. Does NOT output BUY or SELL as executable commands.
* **Paper Trading Boundary**: Research-only agent. Uses crypto market data providers but does not execute trades. Paper-trading enforcement handled by execution layer.
* **LLM Constraints**: Optional LLM provider for interpretation. Deterministic calculations remain authoritative for market structure, risk factors, and data quality. LLMs must NOT fabricate tokenomics, on-chain metrics, or crypto risk data.
* **Data Quality**: Returns INSUFFICIENT_DATA when crypto metrics are unavailable. Does not fabricate missing tokenomics or on-chain data. Warns when symbol is not in supported crypto list.
* **Expected Input**: `Task.input_data` with `symbol` or `symbols`, optional `generation_id`.
* **Expected Output**: `AgentResult` with:
  - `analysis["market_structure"]`: Crypto market structure (trend, momentum, volatility, regime)
  - `analysis["risk_factors"]`: List of crypto-specific risk factors with type, severity, evidence
  - `analysis["tokenomics"]`: Tokenomics data when available (market cap, supply, inflation)
  - `analysis["on_chain"]`: On-chain metrics when available (active addresses, transactions)
  - `analysis["market_wide"]`: Market-wide conditions (BTC dominance, total market cap)
  - `facts`: Structured facts about price, market structure, tokenomics, on-chain activity
  - `warnings`: Data quality warnings, unsupported symbols, missing metrics
  - `metadata`: Risk factors count, tokenomics availability, on-chain availability, data quality

---

## Dashboard API (Implemented)
* **Purpose**: HTTP-based monitoring dashboard for SurvivalAI. Provides real-time visibility into system state, generations, portfolio, risk, agents, learning, and decisions without enabling live trading.
* **Architecture & Pipeline**:
  1. **HTTP Server**: Uses Python's built-in http.server (no external dependencies). Serves on configurable host/port (default 127.0.0.1:8080).
  2. **Paper-Only Enforcement**: Every API response includes paper-only warning. Backend independently enforces paper-trading boundary. Frontend cannot override safety checks.
  3. **System Status Endpoint** (`/api/system`): Runtime state, generation ID, uptime, paper-only warning.
  4. **Generation Status Endpoint** (`/api/generation`): Generation number, ID, status, starting/current/ending capital, return, max drawdown.
  5. **Portfolio Endpoint** (`/api/portfolio`): Cash, equity, positions, exposure, realized/unrealized P/L, open orders.
  6. **Strategy Endpoint** (`/api/strategy`): Active strategy, previous strategies, candidate strategies, validation status, backtest summary.
  7. **Agents Endpoint** (`/api/agents`): All registered agents with status, latest run, success/failure, model, version, enabled state.
  8. **Risk Endpoint** (`/api/risk`): Position concentration, asset-class exposure, sector exposure, geographic exposure, cash reserve, current/max drawdown, volatility, liquidity, crisis exposure, risk-manager blocks, warnings, active risk limits.
  9. **Learning Endpoint** (`/api/learning`): Recent experiences, successful/failed decisions, lessons, lesson confidence, validated/rejected lessons, detected patterns, strategy proposals, backtest results, generation improvements.
  10. **Decisions Endpoint** (`/api/decisions`): Decision audit trail with evidence → research → Deep Looker → Risk Manager → CEO → Decision → Execution → Outcome.
  11. **Health Endpoint** (`/api/health`): System health checks for memory, generation manager, runtime, portfolio sync, agent registry, providers.
  12. **Safe Controls**: POST endpoints for pause, resume, health check, portfolio refresh. NO live trading controls.
  13. **Logs Endpoint** (`/api/logs`): System logs with filtering by timestamp, generation, agent, severity, event type, cycle ID.
  14. **Real-Time Updates**: JavaScript auto-refresh every 5 seconds. Polls API endpoints for live data.
  15. **Security**: Never exposes API keys, broker credentials, LLM credentials, or internal secrets. Sanitizes logs and errors. Validates all dashboard inputs. Frontend requests cannot bypass backend authorization/safety checks.
  16. **HTML Dashboard**: Single-page HTML dashboard with sections for system status, generation status, portfolio, agents, and recent decisions. Displays paper-trading-only warning prominently.
* **Permission Boundaries**: NO live trading controls. NO strategy activation without validation. NO Risk Manager bypass. Frontend cannot override backend safety.
* **Expected Input**: HTTP GET/POST requests with optional query parameters for filtering.
* **Expected Output**: JSON responses for API endpoints, HTML for main dashboard page. All responses include paper-only warning.
* **Paper Trading Boundary**: Dashboard clearly indicates "PAPER TRADING ONLY". Backend independently enforces this. No configuration switch to live trading through dashboard.