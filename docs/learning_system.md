# Learning System

This document describes the learning and evolution system for SurvivalAI. The system learns from experiences, evaluates outcomes, and proposes strategy improvements across generations.

## Learning Loop (Step 15)

The autonomous runtime (Step 15) closes the learning loop:

```
OBSERVE (market/news/crisis)
-> RESEARCH (market/news/crisis in parallel via CEO)
-> DEEP ANALYSIS (Deep Looker)
-> RISK CHECK (Risk Manager)
-> DECIDE (CEO)
-> PAPER EXECUTE (if approved)
-> PORTFOLIO SYNC
-> INVESTMENT MONITORING (Investment Safety Manager)
-> LEARN (ExperienceCollector + LearningCycle)
-> SURVIVAL CHECK
-> GENERATION TRANSITION (if death)
-> SUCCESSOR
```

### ExperienceCollector (`app/core/runtime/learning_service.py`)

Evaluates outcomes of executed decisions and writes Experience records for the learning loop.

- **Outcome Evaluation**: Deterministic: current position value vs entry cost, using provider-confirmed data only.
- **Idempotent**: Experiences are only recorded once per investment (idempotent).
- **Never Fabricates**: If no reliable price is available from the provider, the outcome is not evaluated (INSUFFICIENT_DATA) rather than fabricated.
- **Memory Integration**: Stores experiences as `MemoryRecord` type `EXPERIENCE` in `MemoryStore`.

### LearningCycle (`app/core/runtime/learning_service.py`)

Runs the StrategyUpdaterAgent periodically and records strategy proposals.

- **Periodic**: Runs on a configurable interval (learning_interval_seconds).
- **Idempotent**: One learning cycle per generation + timestamp hour.
- **Never Activates Strategies**: Proposals are recorded with status PROPOSED. Activation requires the validation pipeline (backtest, walk-forward, risk evaluation) from Step 12/13.
- **Memory Integration**: Stores proposals as `MemoryRecord` type `STRATEGY` in `MemoryStore`.

### CostAccountingService (`app/core/runtime/cost_service.py`)

Applies configured operating costs to the active generation.

- **Transparent**: Costs are applied transparently from RuntimeConfig.operating_costs.
- **Never Fabricates**: If a configured cost has no amount, it is recorded as UNKNOWN rather than fabricated.
- **Idempotent**: Application is idempotent per (generation, day, cost_id).
- **Memory Integration**: Stores cost applications as `MemoryRecord` type `FACT` in `MemoryStore`.

### GenerationTransitionService (`app/core/runtime/transition_service.py`)

Coordinates the full death → successor pipeline.

```
DEATH -> cancel/reconcile paper orders -> freeze generation -> death report
      -> experience extraction -> learning loop -> strategy evaluation
      -> successor request (SUCCESSOR_PENDING if no strategy passes)
```

- **Never Invents a Valid Strategy**: If no proposal passes validation, the generation stays SUCCESSOR_PENDING and the runtime stops new investment activity.
- **Idempotent**: The transition pipeline runs once per generation.
- **Successor Creation**: Only one successor per parent generation (idempotent).

### IdempotencyManager (`app/core/runtime/idempotency.py`)

Tracks critical operations so retries never duplicate side effects.

- **Critical Operations**: Order submission, generation creation, learning cycle, cost application, portfolio sync.
- **Retry Safety**: A retry after a crash or timeout can never duplicate the operation because the key is checked first.
- **Status Tracking**: Operations progress from IN_PROGRESS → COMPLETED or FAILED.
- **Stale Detection**: Abandoned IN_PROGRESS operations (e.g., process crash) older than the TTL may be retried.

### RuntimeScheduler (`app/core/runtime/scheduler.py`)

Runs named tasks at their configured intervals.

- **Configurable Intervals**: Portfolio sync, investment monitoring, learning, health checks, cost accounting.
- **Deterministic**: Tasks execute sequentially in the caller's thread (no overlapping runs of the same task).
- **No Uncontrolled Concurrency**: Each task type runs at most once concurrently, guarded by a lock.

### RuntimeHealthChecker (`app/core/runtime/health_service.py`)

Performs layered health checks for the autonomous runtime.

1. **Infrastructure Health**: Memory store, generation manager, agent registry.
2. **Internet/API Health**: Market data, news, LLM providers.
3. **Paper-Trading Health**: Execution provider reachable, account accessible, order endpoint available, environment strictly PAPER.

Any critical failure puts the runtime into a safe state (no new decisions).

### RuntimeRecovery (`app/core/runtime/recovery.py`)

Handles process restarts.

- **Recover Runtime State**: Restores the last known runtime state snapshot.
- **Recover Active Generation**: Restores the active generation ID.
- **Reconcile Portfolio State**: Reconciles local portfolio against the paper provider (source of truth).
- **Reconcile Orders**: Reconciles tracked orders against the provider.
- **Restore Strategy**: Restores the active strategy.
- **Restore Survival State**: Restores survival metrics.
- **Identify Unfinished Operations**: Identifies operations that were in progress when the process crashed.
- **Resume From Valid State**: Resumes the runtime from a valid state. Never assumes the previous process completed anything.

## Strategy Updater (Step 12)

### StrategyUpdaterAgent (`app/agents/strategy_updater/`)

Analyzes historical performance and proposes strategy improvements based on evidence. NEVER directly modifies the active strategy.

- **Strategy Versioning**: Uses existing `StrategyVersion` architecture with strategy_id, version, parent_strategy_id, generation_id, created_at, parameters, rules, description, and status (ACTIVE, TESTING, REJECTED, ARCHIVED). The currently active strategy is always explicitly identifiable.
- **Strategy Structure**: Strongly typed strategy representation including asset universe, allowed asset classes, research requirements, minimum thesis strength, position sizing rules, diversification rules, cash reserve requirements, risk tolerance, maximum drawdown tolerance, market regime rules, crisis response rules, entry conditions, exit/review conditions, holding periods, and portfolio constraints. Strategy configuration is centralized, not hardcoded.
- **StrategyChangeProposal**: Proposal for changing strategy with proposal_id, parent_strategy_id, proposed_parameters, changed_rules, unchanged_rules, motivation, supporting_experiences, supporting_decisions, supporting_outcomes, expected_effect, risks, assumptions, confidence, backtest_required, and status (PROPOSED, BACKTESTING, PASSED, FAILED, REJECTED, APPROVED_FOR_SIMULATION, ACTIVE). ACTIVE status is not directly reachable from PROPOSED.
- **Learning from Experience**: Consumes DecisionRecord, InvestmentRecord, Experience, DeathReport, AgentPerformanceRecord, historical market data, historical news/event data, previous strategy versions, backtest results, risk assessments, and investment safety assessments. Identifies repeated losses, repeated successful patterns, excessive concentration, poor market-regime performance, excessive drawdowns, poor crisis handling, weak thesis selection, excessive trading, underinvestment, overinvestment, strategy weaknesses, and agent weaknesses.
- **No Automatic Overfitting**: Avoids changing strategy based on single random results. Uses minimum evidence requirements: minimum number of observations, minimum number of trades, minimum confidence, minimum performance difference, and statistical significance where practical.
- **Memory Integration**: Stores proposals as `MemoryRecord` with type `STRATEGY` for future evaluation and backtesting.
- **Permission Boundaries**: `activate_strategy` and `modify_active_strategy` raise `PermissionError`. The Strategy Updater cannot directly activate strategies or modify the active strategy.
- **LLM Constraints**: LLM may be used for identifying patterns, proposing hypotheses, explaining failures, summarizing backtest findings, and generating candidate rule changes. LLM must NOT directly activate strategies, modify active strategy state, fabricate historical data, perform financial calculations, override risk constraints, access broker execution, or invent performance results. All calculations are deterministic Python.

## Backtesting & Evaluation (Step 12)

### BacktestEngine (`app/backtesting/`)

Provides robust backtesting framework for strategy evaluation. Supports historical market data, simulated portfolio, simulated positions, cash, transaction costs, slippage, position sizing, strategy rules, risk limits, entry decisions, exit/review decisions, portfolio value, and drawdown.

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

## Generation & Evolution (Step 13)

### GenerationManager (`app/core/generation/`)

Manages generation lifecycle, inheritance, and evolution. This is the first component that allows SurvivalAI to operate across multiple generations, learning from past performance and creating successor generations.

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

## Autonomous Runtime (Step 15)

The Autonomous Runtime provides a central runtime service that connects all existing components into a continuously operating SurvivalAI system.

### Runtime Lifecycle

The runtime operates through a controlled recurring cycle:

OBSERVE
→ RESEARCH
→ ANALYZE
→ RISK CHECK
→ DECIDE
→ PAPER EXECUTE IF APPROVED
→ MONITOR
→ LEARN
→ SURVIVAL CHECK
→ NEXT CYCLE

### Runtime States

Typed runtime state machine with states:
- STARTING
- HEALTH_CHECK
- INITIALIZING_GENERATION
- OBSERVING
- RESEARCHING
- ANALYZING
- RISK_CHECK
- DECIDING
- EXECUTING
- MONITORING
- LEARNING
- SURVIVAL_CHECK
- GENERATION_TRANSITION
- PAUSED
- STOPPING
- STOPPED
- ERROR

Invalid state transitions are rejected.

### Market Data Collection

Uses existing MarketDataProvider to:
- Retrieve current market data
- Validate timestamps
- Detect stale data
- Detect provider failures
- Handle rate limits
- Handle temporary outages
- Cache where appropriate

Never fabricates missing market data. If market data is unavailable, do not invent values, do not make a normal investment decision, record the failure, retry according to configuration, and enter a safe state if necessary.

### News Collection

Uses existing NewsProvider and News Research Agent to periodically update:
- Relevant news
- Event clusters
- Market-moving events
- Company events
- Macro events
- Geopolitical events

Respects source quality, freshness, deduplication, conflicts, and prompt-injection protection. External text is never treated as executable instructions.

### Crisis Data Collection

Uses existing Crisis & Geopolitical Risk Agent to monitor:
- Wars
- Military conflicts
- Sanctions
- Elections
- Government changes
- Tariffs
- Trade restrictions
- Energy disruptions
- Supply-chain disruptions
- Central-bank events
- Inflation
- Recession risks
- Regulatory changes
- Major infrastructure failures
- Cyber incidents
- Other configured crisis events

Crisis Risk contributes evidence and risk analysis but must NOT independently execute trades.

### Research Pipeline

Uses existing architecture with Market Research, News Research, and Crisis Risk running in parallel where appropriate, followed by Deep Looker. Uses configurable triggers such as:
- New investment candidate
- Major news event
- Major crisis event
- Thesis contradiction
- Unusual market movement
- Existing investment deterioration

### Investment Decision Pipeline

For new investments:
REQUEST → INPUT VALIDATION → MARKET RESEARCH → NEWS RESEARCH → CRISIS RISK → DEEP LOOKER → RISK MANAGER → CEO/ORCHESTRATOR → DECISION

The CEO cannot bypass Risk Manager. If Risk Manager returns BLOCKED, the investment must not execute. If INSUFFICIENT_DATA, the system must not pretend the investment is safe.

### Paper Execution

If a decision is approved, uses the existing ExecutionProvider. Execution remains behind the existing abstraction. If Alpaca is configured, ONLY uses the Alpaca paper-trading environment. The runtime verifies paper mode before execution with an explicit safety check.

### Live Trading Safety

The runtime makes live trading architecturally impossible through the SurvivalAI code path. Does not:
- Add live broker credentials
- Add live endpoints
- Create a "switch to live" button
- Automatically select a live environment
- Infer live mode from missing configuration
- Silently fall back to live trading

If paper-trading configuration is missing, FAIL SAFE. Do not execute.

### Portfolio Synchronization

After paper execution, synchronizes portfolio state tracking:
- Cash
- Equity
- Positions
- Average entry price
- Current price
- Unrealized P/L
- Realized P/L
- Exposure
- Available buying power
- Open orders
- Filled orders
- Cancelled orders
- Rejected orders

Uses the provider abstraction. Does not assume an order succeeded simply because an order request was sent. Tracks actual order lifecycle.

### Order Lifecycle

Implements/reuses:
CREATED → SUBMITTED → ACCEPTED → PARTIALLY_FILLED → FILLED

and failure paths:
→ REJECTED
→ CANCELLED
→ EXPIRED
→ FAILED

The runtime reconciles actual provider state. Never marks an investment as filled without confirmation.

### Existing Investment Monitoring

Periodically runs Investment Safety Manager for every active investment, monitoring:
- Thesis
- Fundamentals
- Valuation
- Market conditions
- News
- Crisis conditions
- Drawdown
- Contradictions
- Time horizon

The Investment Safety Manager may return HOLD, REVIEW, EXIT_CANDIDATE, or INSUFFICIENT_DATA. Does not automatically sell simply because EXIT_CANDIDATE is returned unless the existing architecture explicitly permits it. Preserves the original investment thesis.

### Decision Frequency Gating

Implements decision gating to avoid blindly trading every cycle. Reasons to avoid a new decision:
- No meaningful market change
- Insufficient data
- No candidate
- Existing portfolio already sufficiently exposed
- Risk limits
- Crisis conditions
- Cooldown
- Duplicate decision
- Provider outage
- Recent decision on same asset
- Strategy constraints

The system is autonomous without becoming unnecessarily active.

### Idempotency

Critical operations are idempotent:
- Investment request
- Order submission
- Portfolio synchronization
- Generation transition
- Learning cycle

Uses unique operation/request IDs. A temporary retry must not create duplicate investments or duplicate generations.

### Failure Handling

Implements graceful failure handling classifying failures as:
- TRANSIENT
- RECOVERABLE
- CRITICAL
- FATAL

Possible failures:
- API outage
- Network timeout
- Rate limit
- Malformed API response
- Stale data
- Provider authentication failure
- Paper-trading outage
- LLM failure
- Agent timeout
- Invalid model output
- Database/memory failure
- Event failure

Does not crash the entire system because of a temporary provider outage.

### Safe Pause

Implements a safe pause state when:
- Market data is stale
- Critical API unavailable
- Paper provider unavailable
- Risk Manager unavailable
- Required research unavailable
- Configuration invalid
- System integrity check fails

Existing positions can continue to be monitored where data is available. Does not create new investments while critical safety infrastructure is unavailable.

### Survival Check

After each meaningful cycle evaluates:
- Capital
- Equity
- Peak equity
- Current drawdown
- Maximum drawdown
- Operating costs
- Realized P/L
- Unrealized P/L
- Risk violations
- Configured survival conditions

If death condition is reached:
STOP NEW INVESTMENT ACTIVITY → CANCEL/RECONCILE APPROPRIATE PAPER ORDERS → FREEZE GENERATION → CREATE DEATH REPORT → START LEARNING LOOP → STRATEGY EVALUATION → VALIDATION → SUCCESSOR GENERATION

Does not immediately continue operating the dead generation.

### Operating Costs

Accounts for configured operating costs:
- Model/API usage
- Market-data costs
- News-data costs
- Infrastructure
- Other configured expenses

Costs are represented transparently. If costs are unavailable, marked as unknown rather than fabricated.

### Generation Transition

When a generation dies, uses existing Step 13 + Step 14 architecture:
DEATH → DEATH REPORT → EXPERIENCE EXTRACTION → LESSON VALIDATION → PATTERN ANALYSIS → STRATEGY PROPOSAL → BACKTEST → WALK-FORWARD VALIDATION → RISK EVALUATION → STRATEGY COMPARISON → APPROVAL/REJECTION → SUCCESSOR GENERATION

If no strategy passes validation, enters SUCCESSOR_PENDING or another safe configured state. Does not invent a valid strategy.

### Internet Connectivity

Designed for real internet-connected operation:
Internet → External APIs → Provider Layer → Normalized Data → Agents → Orchestrator → Risk Manager → Paper Trading

Does not replace real external providers with hardcoded demo data. Tests may use mock providers. Production configuration uses real providers with provider health checks.

### API Configuration

Uses environment variables for:
- Market API credentials
- News API credentials
- LLM credentials
- Paper-trading credentials

NEVER hardcodes secrets. Creates/updates `.env.example` with placeholder names only. Never logs credentials or exposes secrets through logs, events, memory, agent output, or dashboard APIs.

### Scheduling

Configurable scheduler/orchestrator supporting different task frequencies:
- Market polling
- News polling
- Crisis monitoring
- Portfolio synchronization
- Investment monitoring
- Learning
- Cost accounting
- Health checks

Does not make every component run at the same interval. Avoids uncontrolled concurrent executions.

### Concurrency

Uses controlled concurrency. Parallelizes independent research tasks where safe. Prevents concurrent conflicting operations such as:
- Two investments against the same decision
- Duplicate generation creation
- Simultaneous strategy activation
- Conflicting portfolio synchronization

Uses locks or idempotency mechanisms where necessary.

### Observability

Adds structured runtime logging where every cycle has:
- Cycle_id
- Generation_id
- Timestamp
- Runtime state
- Triggered actions
- Agent results
- Decision result
- Execution result
- Portfolio state
- Survival state
- Errors/warnings

Does not log secrets. Creates structured event records rather than relying only on plain text logs.

### Audit Trail

Every investment decision is traceable:
Market/news/crisis evidence → Research outputs → Deep Looker → Risk Manager → CEO → DecisionRecord → Execution → Portfolio outcome → Experience → Learning

A user can later answer "Why did SurvivalAI make this investment?" with evidence-backed records.

### No Hallucinations

LLMs must never fabricate:
- Market prices
- News
- API responses
- Portfolio values
- Investment outcomes
- Strategy results
- Backtest results
- Crisis events

If information is unavailable, returns INSUFFICIENT_DATA or the appropriate failure state.

### Prompt Injection Defense

External news/web/API content is untrusted data. Never allows external content to:
- Modify system instructions
- Execute commands
- Change risk limits
- Change strategy
- Bypass agents
- Request secrets
- Place orders
- Modify generation state

Treat external content strictly as data.

### Test Mode

Implements controlled TEST/SIMULATION mode using:
- Deterministic mock market provider
- Deterministic mock news provider
- Deterministic mock crisis provider
- Simulated execution provider

This mode allows the entire autonomous loop to be tested without internet or real paper orders. Production mode remains internet-connected and paper-trading-only.

### Paper Mode Health Check

Before autonomous execution, verifies:
- Provider reachable
- Authentication valid
- Environment is paper
- Account accessible
- Portfolio accessible
- Order endpoint available
- No live endpoint configured

If any critical check fails, DO NOT EXECUTE.

### Runtime Recovery

Handles runtime restart and recovery:
- Recover runtime state from memory
- Recover active generation
- Reconcile portfolio state
- Reconcile orders
- Restore strategy
- Restore survival state
- Identify unfinished operations
- Resume from valid state

Never assumes the previous process completed an operation.

### Safety Boundaries

The Autonomous Runtime:
- NEVER places live trades
- NEVER connects to live broker
- NEVER bypasses Risk Manager
- NEVER bypasses CEO/Orchestrator
- NEVER bypasses Strategy Backtesting
- NEVER activates unvalidated strategy
- NEVER rewrites historical records
- NEVER fabricates experiences/market data/death causes/performance

Remains simulation/paper-trading only.
- **Paper Trading Boundary**: May create generations operating against existing simulated/paper portfolio. Uses ONLY configured Alpaca paper-trading environment, never live trading endpoint, no configuration switch to silently redirect to live trading. Generation creation fails safely if paper-trading configuration is invalid.
- **LLM Constraints**: LLMs may help analyze DeathReports, extract lessons, propose hypotheses, explain why strategy might improve, and identify patterns. LLMs must NOT decide generation death, calculate financial metrics, override death conditions, activate strategies, bypass backtesting, override Risk Manager hard rules, fabricate historical data/experiences, or rewrite immutable historical records. Deterministic code remains authoritative.
- Important: EXIT_CANDIDATE is ONLY a recommendation/state. Future CEO/orchestration will decide what to do with that recommendation. The Safety Manager never executes orders or directly sells positions.