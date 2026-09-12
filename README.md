# SurvivalAI

Ein System von KI-Agenten, die in einem simulierten Finanzmarkt investieren und überleben müssen.

## Architecture

SurvivalAI is a safety-first, generational agent ecosystem for simulated financial-market investing. The system coordinates research, risk management, investment monitoring, paper-trading decisions, and evolutionary learning across multiple generations.

### Key Components

- **Research Agents**: News Research, Market Research, Crisis & Geopolitical Risk, Deep Looker (Deep Research), Crypto Research (Step 16)
- **Risk Management**: Risk Manager (entry risk) and Investment Safety Manager (holding risk)
- **Orchestration**: CEO / Orchestrator coordinates agent workflows and produces decision proposals
- **Strategy & Learning**: Strategy Updater (Step 12) analyzes historical performance and proposes strategy improvements
- **Backtesting**: Robust backtesting framework with transaction costs, slippage, data leakage protection, and survival-aware fitness
- **Generation & Evolution**: GenerationManager (Step 13) manages generational lifecycle, inheritance, and controlled evolution
- **Autonomous Runtime**: SurvivalRuntime (Step 15) provides central runtime service for autonomous operation, coordinating all components into a continuously operating system
- **Dashboard**: Monitoring Dashboard (Step 16) provides HTTP-based monitoring of system state, generations, portfolio, risk, and decisions

### Safety Guarantees

- **Paper Trading Only**: No live trading, no real-money execution, no live broker credentials
- **Provider Abstraction**: External APIs accessed through provider layer with proper authentication and error handling
- **Typed Models**: Strongly typed models for inter-agent communication
- **Risk Manager Authority**: Hard risk rules cannot be overridden by LLM output
- **Deterministic Safety Controls**: Critical calculations in Python, not LLM
- **Evidence Traceability**: All decisions reference supporting evidence and sources
- **No Look-Ahead Bias**: Backtesting prevents access to future data
- **Architectural Paper-Only Enforcement**: `ExecutionEnvironment` enum contains only `PAPER`; `PaperOnlyExecutionProvider` wraps every execution path; `PaperExecutionService` verifies paper mode before any order submission; live trading is architecturally impossible

### Steps Implemented

1. Project architecture
2. Agent foundation and structured agent results
3. Memory and experience architecture
4. Provider-based market/news/execution architecture
5. News Research Agent
6. Market Research Agent
7. Crisis & Geopolitical Risk Agent
8. Deep Looker / Deep Research Agent
9. Risk Manager
10. Investment Safety Manager
11. CEO / Orchestrator
12. Strategy Updater + Backtesting & Evaluation
13. Generation & Evolution System
14. Learning Loop
15. End-to-End Autonomous Paper-Trading Loop
16. Crypto Research Agent + Monitoring Dashboard

### Current Status

Step 16 (Crypto Research Agent + Monitoring Dashboard) is implemented and tested. The system can now:

- Run a complete end-to-end autonomous loop: health checks → generation start → observe (market/news/crisis) → research → deep analysis → risk assessment → decision → paper execution → portfolio sync → investment monitoring → learning → survival check → generation transition
- Execute paper orders through `PaperExecutionService` with idempotent order submission, explicit order lifecycle tracking, and provider-confirmed fills
- Gate decisions to prevent blind trading every cycle (cooldowns, exposure limits, crisis conditions, stale data)
- Monitor existing investments via the Investment Safety Manager (HOLD/REVIEW/EXIT_CANDIDATE recommendations, never automatic sells)
- Apply operating costs transparently from configuration
- Recover from process restarts (restores active generation, reconciles portfolio and orders)
- Handle generation death through the full transition pipeline (death report → experience extraction → learning → strategy evaluation → successor creation)
- Run in two modes: `test` (deterministic mocks, no network) and `production` (internet-connected, Alpaca paper-trading only)
- **Analyze cryptocurrency markets with specialized Crypto Research Agent**
- **Track crypto-specific metrics: tokenomics, on-chain data, crypto risk factors**
- **Monitor system state through HTTP-based dashboard at http://127.0.0.1:8080**
- **View generation history, portfolio, risk, and decisions in real-time**

### Running the Dashboard

```bash
# Start the dashboard (paper-trading only, no live trading)
python -m app.dashboard.api
```

Then open http://127.0.0.1:8080 in your browser.

### Documentation

- [Architecture](docs/architecture.md) - System architecture and pipeline documentation
- [Agents](docs/agents.md) - Agent architecture and responsibilities
- [Learning System](docs/learning_system.md) - Strategy learning, generational evolution, and autonomous runtime
- [Survival System](docs/survival_system.md) - Safety boundaries and survival guarantees