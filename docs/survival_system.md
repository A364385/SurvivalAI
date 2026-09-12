# Survival System

Das Konzept basiert auf Kapital, laufenden Kosten und der Bedingung Überleben oder Tod.

## Risk Manager Role in Survival

The Risk Manager Agent serves as a critical survival mechanism by preventing catastrophic losses that could lead to generational death. It enforces hard limits on:

- **Position Sizing**: Prevents over-concentration in single assets that could cause rapid capital depletion
- **Cash Reserves**: Ensures sufficient liquidity is maintained for opportunities and margin of safety
- **Diversification**: Enforces sector, asset-class, and geographic diversification to reduce systemic risk
- **Drawdown Limits**: Protects against excessive portfolio drawdown that could trigger generational death conditions
- **Crisis Exposure**: Blocks investments during severe geopolitical or macroeconomic crises
- **Data Quality**: Prevents investment decisions based on insufficient or stale data

The Risk Manager's deterministic calculations ensure that survival decisions are based on mathematical limits rather than subjective LLM judgments. When an investment proposal violates hard risk rules, it is BLOCKED before it can reach execution, protecting the generation's capital from preventable losses.

The Risk Manager maintains the safety hierarchy: Research → Deep Looker → Risk Manager → CEO → Execution. This ensures that no investment can bypass risk controls, preserving the generational survival mandate.

## Investment Safety Manager Role in Survival

The Investment Safety Manager Agent provides the second layer of survival protection by continuously monitoring existing investments. While the Risk Manager prevents dangerous new entries, the Investment Safety Manager prevents existing positions from deteriorating into capital-destroying losses.

It protects survival by:

- **Thesis Preservation**: Continuously validates that the original investment thesis remains supported by current evidence
- **Contradiction Detection**: Identifies when current market conditions, news, or events contradict original investment assumptions
- **Early Warning System**: Detects thesis weakening before it becomes critical, allowing for proactive position management
- **Drawdown Monitoring**: Separates normal price volatility from thesis damage, preventing panic selling during normal market fluctuations
- **Geopolitical Awareness**: Evaluates how new crisis events affect existing investment theses and risk profiles
- **Memory and Learning**: Stores assessment history to detect trends in thesis deterioration over time, supporting generational learning

The Investment Safety Manager's decisions (HOLD/REVIEW/EXIT_CANDIDATE) provide structured recommendations to the CEO/orchestrator. This two-layer safety system (Risk Manager for entries, Safety Manager for holdings) creates robust protection against both entry and exit risks, significantly improving generational survival probability.

## CEO / Orchestrator Role in Survival

The CEO Agent serves as the central coordination layer that ensures all survival mechanisms are properly integrated before any investment decision is made. It does not bypass safety controls—it enforces them.

The CEO protects survival by:

- **Safety Gate Enforcement**: Always routes new investment proposals through Risk Manager before any decision. BLOCKED status prevents INVEST decisions, ensuring hard risk limits are never violated.
- **Evidence Preservation**: Aggregates and preserves all evidence from research agents, ensuring decisions are based on complete and traceable information rather than incomplete summaries.
- **Conflict Detection**: Identifies when specialized agents produce conflicting outputs (e.g., Market Research says favorable while Crisis Risk reports severe exposure), preventing decisions based on contradictory intelligence.
- **Thesis Integrity**: For existing investments, routes through Investment Safety Manager and preserves the distinction between HOLD (thesis supported), REVIEW (thesis weakened), and EXIT_CANDIDATE (critical thesis invalidation). EXIT_CANDIDATE never directly executes a sell.
- **Observable Orchestration**: Maintains explicit state machines and audit trails for all decisions, making it possible to reconstruct why any decision was made and verify that all safety controls were respected.
- **Failure Handling**: Explicitly handles agent failures, timeouts, and missing research rather than fabricating success, preventing decisions based on incomplete intelligence.
- **STOP BEFORE EXECUTION**: Produces decision proposals only and never executes trades. This separation ensures that the CEO's coordination logic cannot accidentally bypass paper-trading safety boundaries.

The CEO maintains the safety hierarchy: Research → Deep Looker → Risk Manager → CEO → Decision Proposal → (Future) Paper Execution. This ensures that no investment decision can bypass the coordinated safety controls, preserving the generational survival mandate through multiple layers of protection.

## Strategy Updater Role in Survival

The Strategy Updater Agent provides the third layer of survival protection by learning from historical performance and proposing strategy improvements that address identified weaknesses. While the Risk Manager prevents dangerous entries and the Investment Safety Manager prevents holding deterioration, the Strategy Updater improves the system's overall survival fitness over time.

It protects survival by:

- **Historical Pattern Learning**: Analyzes past decisions, investments, and outcomes to identify repeated losses, successful patterns, and systematic weaknesses that may not be obvious from individual cases.
- **Evidence-Based Proposals**: Only proposes strategy changes with supporting historical evidence (DecisionRecord, InvestmentRecord, Experience, DeathReport, AgentPerformanceRecord). Never invents lessons without data.
- **Risk Rule Optimization**: Identifies when existing risk rules are too loose (allowing dangerous investments) or too tight (preventing reasonable opportunities) and proposes calibrated adjustments.
- **Regime Adaptation**: Detects poor performance during specific market regimes (high volatility, crisis, bear markets) and proposes regime-specific rule adjustments.
- **Drawdown Analysis**: Learns from generation deaths and severe drawdowns to identify which strategy parameters contributed to capital destruction and proposes corrections.
- **No Overfitting Protection**: Uses minimum evidence requirements (minimum observations, trades, confidence, performance difference) to prevent strategy changes based on random noise or tiny samples.
- **Safe Proposal Pipeline**: Proposals follow PROPOSED → BACKTESTING → VALIDATION → APPROVED_FOR_SIMULATION pipeline. Cannot directly activate strategies. ACTIVE status is not reachable from PROPOSED.
- **Survival-Aware Fitness**: When comparing strategies, evaluates capital preservation, probability of ruin, maximum drawdown, return, risk-adjusted return, costs, stability, and diversification. Balances SURVIVAL + GROWTH + RISK CONTROL. Penalizes "never invest" loopholes.

The Strategy Updater's proposals provide candidate strategies for future evaluation and backtesting. This three-layer safety system (Risk Manager for entries, Safety Manager for holdings, Strategy Updater for learning) creates robust protection against entry risks, holding risks, and systemic strategy weaknesses, significantly improving generational survival probability across multiple generations.

## Generation & Evolution System Role in Survival

The Generation & Evolution System provides the fourth layer of survival protection by creating a controlled evolutionary lifecycle that improves the system's survival fitness across multiple generations while preserving all safety boundaries.

It protects survival by:

- **Controlled Lifecycle**: Generations operate with explicit lifecycle states (CREATED, INITIALIZING, ACTIVE, PAUSED, DYING, DEAD, SUCCESSOR_PENDING, SUCCESSOR_CREATED, ARCHIVED). No generation dies without explicit trigger and comprehensive death reporting.
- **Configurable Death Conditions**: Death triggers (CAPITAL_DEPLETED, MINIMUM_SURVIVAL_THRESHOLD, MAXIMUM_DRAWDOWN_EXCEEDED, UNRECOVERABLE_PORTFOLIO_STATE, OPERATING_COSTS_EXCEEDED, CONFIGURED_CONDITION_VIOLATED, INFRASTRUCTURE_FAILURE) are explicit and configurable. No automatic death from temporary losses.
- **Operating Costs**: Generations account for API costs, data costs, model usage costs, infrastructure costs, and paper-trading/data service costs. Costs reduce available capital, so generations can die from combination of losses and costs, not just investment losses.
- **Infrastructure Health Checks**: Before starting, generations verify MarketDataProvider, NewsProvider, LLMProvider, MemoryStore, Risk Manager, CEO/Orchestrator, and paper-trading provider are available. Generation fails to start if critical infrastructure is unavailable.
- **Inheritance with Validation**: New generations may inherit knowledge from parents (successful experiences, failed experiences, DeathReport lessons, validated strategy parameters, validated strategy rules, agent performance, market/crisis patterns, backtesting results, failure modes). Only validated knowledge becomes active. Historical experiences remain immutable.
- **Evolution Pipeline**: GENERATION DIES → DEATH REPORT → EXPERIENCE EXTRACTION → FAILURE ANALYSIS → SUCCESS ANALYSIS → STRATEGY CHANGE PROPOSALS → BACKTEST → WALK-FORWARD VALIDATION → RISK EVALUATION → STRATEGY COMPARISON → APPROVAL/REJECTION → SUCCESSOR GENERATION. System never bypasses Step 12's validation pipeline; unvalidated strategies cannot become active.
- **Anti-Overfitting**: Requires minimum sample size, multiple market regimes, bull/bear/sideways conditions, crisis periods, out-of-sample validation, walk-forward testing, transaction costs, slippage, drawdown evaluation, survival evaluation, and benchmark comparison. Rejects changes that only improve narrow historical periods.
- **Survival-Aware Fitness**: Considers survival, return, risk-adjusted performance, maximum drawdown, volatility, capital preservation, diversification, crisis resilience, transaction costs, operating costs, and strategy robustness. Prevents "never invest = perfect survival" loophole.
- **Lineage Tracking**: Every successor generation is traceable (get_generation, get_parent_generation, get_child_generations, get_generation_lineage, get_generation_history, get_generation_performance). Preserves historical lineage for debugging and analysis.
- **DeathReport Integration**: Uses existing DeathReport model. When generation dies: freeze state, collect final portfolio, collect decisions/investments/agent_performance/market/crisis conditions, calculate final metrics, identify failures/successes/causes/lessons, store DeathReport, emit GenerationDied event. DeathReport is immutable after finalization.
- **Safety Boundaries**: NEVER places live trades, connects to live broker, bypasses Risk Manager, bypasses CEO/Orchestrator, bypasses Strategy Backtesting, activates unvalidated strategy, rewrites historical records, fabricates experiences/market data/death causes/performance. Remains simulation/paper-trading only.
- **Paper Trading Boundary**: May create generations operating against existing simulated/paper portfolio. Uses ONLY configured Alpaca paper-trading environment, never live trading endpoint, no configuration switch to silently redirect to live trading. Generation creation fails safely if paper-trading configuration is invalid.

The Generation & Evolution System's controlled lifecycle provides a fourth layer of protection (after Risk Manager for entries, Safety Manager for holdings, Strategy Updater for learning) that improves generational survival probability across multiple generations while maintaining all existing safety boundaries and paper-trading-only execution.

## Autonomous Runtime Role in Survival (Step 15)

The Autonomous Runtime (SurvivalRuntime) provides the fifth layer of survival protection by coordinating the full end-to-end paper-trading loop. It connects all agents, providers, and services into one continuously operating system that monitors, learns, and evolves across generations.

It protects survival by:

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

The Autonomous Runtime's coordinated loop provides a fifth layer of protection (after Risk Manager for entries, Safety Manager for holdings, Strategy Updater for learning, Generation & Evolution for generational survival) that improves generational survival probability across multiple generations while maintaining all existing safety boundaries and paper-trading-only execution.