from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable
from app.agents.base import BaseAgent
from app.core.models.agent import AgentConfig, AgentStatus, AgentResult
from app.core.models.task import Task
from app.core.models.knowledge import Fact, Source
from app.core.models.error import ErrorInfo
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.models.events import BaseEvent, MarketAnalysisCompleted
from app.core.models.market import (
    Quote, MarketSnapshot, DataQualityIssue, capabilities_from_asset_type,
)
from app.core.models.provider_errors import (
    ProviderError, RateLimitError, AuthenticationError,
    APIUnavailableError, InvalidResponseError,
)
from app.core.memory.store import MemoryStore
from app.services.market_data.provider import MarketDataProvider
from app.services.llm.provider import LLMProvider
from app.services.llm.schema_validator import validate_market_analysis_schema
from app.services.fundamental.provider import FundamentalDataProvider
from app.agents.market_research.config import MarketAnalysisConfig
from app.agents.market_research.data_quality import DataQualityChecker
from app.agents.market_research.indicators import MarketFeatureCalculator
from app.agents.market_research.regime_classifier import MarketRegimeClassifier
from app.agents.market_research.anomaly_detector import MarketAnomalyDetector
from app.agents.market_research.cache import MarketDataCache
from app.agents.market_research.prompt_builder import MarketPromptBuilder
from app.agents.market_research.recommendation_guard import sanitize_llm_payload
from app.utils.ids import generate_id
from app.utils.time import now_utc
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MarketResearchAgent(BaseAgent):
    """Research-only agent: deterministic market features plus optional LLM interpretation.

    Does not decide investments, place orders, modify portfolios, or override risk/CEO.
    """

    def __init__(
        self,
        agent_id: str,
        market_data_provider: MarketDataProvider,
        llm_provider: LLMProvider,
        memory_store: Optional[MemoryStore] = None,
        configuration: Optional[AgentConfig] = None,
        analysis_config: Optional[MarketAnalysisConfig] = None,
        fundamental_provider: Optional[FundamentalDataProvider] = None,
        event_publisher: Optional[Callable[[BaseEvent], None]] = None,
    ):
        config = configuration or AgentConfig(version="1.0.0", model=None)
        super().__init__(
            agent_id=agent_id,
            agent_name="MarketResearchAgent",
            role="Quantitative Market Analyst & Technical Pattern Researcher",
            description="Analyzes market prices, returns, volatility, volume, and technical regimes deterministically.",
            version="1.0.0",
            configuration=config,
            capabilities=[
                "market_data_validation",
                "feature_calculation",
                "trend_classification",
                "regime_analysis",
                "anomaly_detection",
            ],
        )
        self.market_data_provider = market_data_provider
        self.llm_provider = llm_provider
        self.memory_store = memory_store
        self.fundamental_provider = fundamental_provider
        self.event_publisher = event_publisher
        self.emitted_events: List[BaseEvent] = []
        self.analysis_config = analysis_config or MarketAnalysisConfig()

        self.quality_checker = DataQualityChecker(self.analysis_config)
        self.calculator = MarketFeatureCalculator(self.analysis_config)
        self.regime_classifier = MarketRegimeClassifier(self.analysis_config)
        self.anomaly_detector = MarketAnomalyDetector(self.analysis_config)
        self.cache = MarketDataCache(ttl_seconds=self.analysis_config.cache_ttl_seconds)
        self.prompt_builder = MarketPromptBuilder()

    def _emit(self, event: BaseEvent) -> None:
        self.emitted_events.append(event)
        if self.event_publisher:
            self.event_publisher(event)

    def _horizon_limit(self, task_input: Dict[str, Any]) -> int:
        if "limit" in task_input:
            return int(task_input["limit"])
        horizon = str(task_input.get("horizon", self.analysis_config.default_horizon))
        return self.analysis_config.bar_limit_for_horizon(horizon)

    def _lookback_start(self, now: datetime, limit: int) -> datetime:
        # Extra calendar days cover weekends/holidays without pulling unbounded history.
        return now - timedelta(days=int(limit * 1.7) + 5)

    def build_market_snapshot(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
        now: Optional[datetime] = None,
        horizon: str = "medium",
    ) -> MarketSnapshot:
        """Constructs a deterministic MarketSnapshot without invoking the LLM."""
        ref_time = now or now_utc()
        timeframe = timeframe or self.analysis_config.default_timeframe
        limit = limit if limit is not None else self.analysis_config.bar_limit_for_horizon(horizon)

        cached = self.cache.get(symbol, timeframe, limit)
        retrieved_at = ref_time
        if cached is not None:
            bars = cached.bars
            retrieved_at = cached.retrieved_at
        else:
            start = self._lookback_start(ref_time, limit)
            bars = self.market_data_provider.get_historical_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=ref_time,
                limit=limit,
            )
            cached_entry = self.cache.set(symbol, timeframe, bars, limit)
            retrieved_at = cached_entry.retrieved_at

        raw_report = self.quality_checker.validate_bars(symbol, bars)
        clean_bars = self.quality_checker.filter_valid_bars(bars)
        if len(clean_bars) != len(bars):
            raw_report.warnings.append(
                f"Excluded {len(bars) - len(clean_bars)} invalid/duplicate bars from calculations."
            )

        quote: Optional[Quote] = None
        try:
            quote = self.market_data_provider.get_quote(symbol)
        except ProviderError as e:
            logger.warning("Could not retrieve quote for %s: %s", symbol, e)
            raw_report.warnings.append(f"Quote unavailable: {e.message}")
        except Exception as e:
            logger.warning("Could not retrieve quote for %s: %s", symbol, e)

        market_is_open: Optional[bool] = None
        try:
            market_is_open = self.market_data_provider.get_market_status()
        except Exception as e:
            logger.warning("Could not retrieve market clock: %s", e)

        quote_issues, quote_types = self.quality_checker.validate_quote(
            symbol, quote, ref_time, market_is_open
        )
        raw_report.issues.extend(quote_issues)
        for qt in quote_types:
            if qt not in raw_report.issue_types:
                raw_report.issue_types.append(qt)
        if DataQualityIssue.STALE_QUOTE in quote_types:
            raw_report.warnings.append("Quote is stale relative to the analysis clock.")

        price_feat = self.calculator.calculate_price_features(clean_bars)
        returns = self.calculator.calculate_returns(clean_bars)
        mas = self.calculator.calculate_moving_averages(clean_bars)
        vol = self.calculator.calculate_volatility(clean_bars)
        vol_metrics = self.calculator.calculate_volume_metrics(clean_bars)
        momentum = self.calculator.calculate_momentum(clean_bars, mas)
        trend = self.regime_classifier.classify_trend(clean_bars, mas, returns)
        regime = self.regime_classifier.classify_regime(clean_bars, trend, vol)
        anomalies = self.anomaly_detector.detect_anomalies(
            symbol=symbol,
            bars=clean_bars,
            quote=quote,
            price_feat=price_feat,
            returns=returns,
            vol=vol,
            vol_metrics=vol_metrics,
            now=ref_time,
        )

        asset_caps = capabilities_from_asset_type("UNKNOWN")
        if self.fundamental_provider is not None:
            try:
                profile = self.fundamental_provider.get_company_profile(symbol)
                if profile is not None:
                    asset_caps = capabilities_from_asset_type(profile.asset_type)
            except Exception as e:
                logger.warning("Fundamental profile unavailable for %s: %s", symbol, e)

        data_timestamp = clean_bars[-1].timestamp if clean_bars else None
        return MarketSnapshot(
            symbol=symbol,
            timestamp=ref_time,
            current_price=price_feat.current_price,
            price_features=price_feat,
            returns=returns,
            moving_averages=mas,
            volatility=vol,
            volume=vol_metrics,
            momentum=momentum,
            trend=trend,
            regime=regime,
            anomalies=anomalies,
            data_quality=raw_report,
            retrieved_at=retrieved_at,
            data_timestamp=data_timestamp,
            asset_capabilities=asset_caps,
            market_is_open=market_is_open,
            metadata={
                "bars_analyzed": len(clean_bars),
                "bars_received": len(bars),
                "timeframe": timeframe,
                "horizon_limit": limit,
                "retrieved_at": retrieved_at.isoformat(),
                "data_timestamp": data_timestamp.isoformat() if data_timestamp else None,
            },
        )

    def _objective_comparisons(self, snapshots: List[MarketSnapshot]) -> List[str]:
        statements: List[str] = []
        if len(snapshots) < 2:
            return statements

        with_vol = [
            s for s in snapshots
            if s.volatility.short_term_volatility is not None
        ]
        if len(with_vol) >= 2:
            ranked = sorted(with_vol, key=lambda s: s.volatility.short_term_volatility or 0.0)
            statements.append(
                f"{ranked[-1].symbol} has higher short-term historical volatility than {ranked[0].symbol}."
            )

        with_rsi = [s for s in snapshots if s.momentum.rsi_14 is not None]
        if len(with_rsi) >= 2:
            ranked = sorted(with_rsi, key=lambda s: s.momentum.rsi_14 or 0.0)
            statements.append(
                f"{ranked[-1].symbol} has stronger recent RSI momentum than {ranked[0].symbol}."
            )
        return statements

    def _deterministic_facts(self, snap: MarketSnapshot, now: datetime) -> List[Fact]:
        facts: List[Fact] = [
            Fact(
                fact_id=generate_id("mfact"),
                statement=(
                    f"{snap.symbol} current price is {snap.current_price} "
                    f"with 1d return of {snap.returns.return_1d}."
                ),
                source_ids=["MarketDataProvider"],
                timestamp=now,
                confidence=1.0,
                data_type="market_price",
            ),
            Fact(
                fact_id=generate_id("mfact"),
                statement=(
                    f"{snap.symbol} trend is classified as {snap.trend.value} "
                    f"and regime as {snap.regime.value}."
                ),
                source_ids=["MarketDataProvider"],
                timestamp=now,
                confidence=1.0,
                data_type="market_regime",
            ),
        ]
        if snap.momentum.rsi_14 is not None:
            facts.append(Fact(
                fact_id=generate_id("mfact"),
                statement=f"{snap.symbol} 14-period RSI is {snap.momentum.rsi_14}.",
                source_ids=["MarketDataProvider"],
                timestamp=now,
                confidence=1.0,
                data_type="technical_indicator",
            ))
        return facts

    def _fallback_llm_result(self, snap: MarketSnapshot) -> Dict[str, Any]:
        return {
            "facts": [],
            "analysis": {
                "technical_summary": (
                    f"{snap.symbol} is in {snap.regime.value} regime with {snap.trend.value} trend."
                ),
                "market_regime": snap.regime.value,
                "market_trend": snap.trend.value,
                "risk_factors": [a.description for a in snap.anomalies],
            },
            "impact": {
                "direction": "UNCERTAIN",
                "horizon": "SHORT_TERM",
                "affected_symbols": [snap.symbol],
            },
            "confidence": 0.7,
            "warnings": ["LLM interpretation unavailable; returning deterministic snapshot only."],
        }

    def process_task(self, task: Task) -> AgentResult:
        self.status = AgentStatus.RUNNING
        now = now_utc()
        symbol = task.input_data.get("symbol", "SPY")
        symbols = task.input_data.get("symbols", [symbol])
        news_context = task.input_data.get("news_context")
        timeframe = task.input_data.get("timeframe", self.analysis_config.default_timeframe)
        limit = self._horizon_limit(task.input_data)
        horizon = str(task.input_data.get("horizon", self.analysis_config.default_horizon))

        logger.info(
            "MarketResearchAgent processing task %s for %s symbol(s): %s",
            task.task_id,
            len(symbols),
            symbols,
        )

        try:
            snapshots: List[MarketSnapshot] = []
            for sym in symbols:
                snapshots.append(
                    self.build_market_snapshot(
                        symbol=sym,
                        timeframe=timeframe,
                        limit=limit,
                        now=now,
                        horizon=horizon,
                    )
                )

            primary_snap = snapshots[0]
            warnings: List[str] = []
            if primary_snap.data_quality:
                warnings.extend(primary_snap.data_quality.warnings)
                if not primary_snap.data_quality.is_valid:
                    warnings.extend(primary_snap.data_quality.issues)
            if primary_snap.moving_averages.status == "INSUFFICIENT_DATA":
                warnings.append("Insufficient historical bars for SMA20; status=INSUFFICIENT_DATA.")

            llm_result = self._fallback_llm_result(primary_snap)
            try:
                prompt = self.prompt_builder.build_analysis_prompt(
                    primary_snap, news_context=news_context
                )
                raw_llm = self.llm_provider.generate_structured(
                    prompt=prompt,
                    schema={},
                    system_prompt=self.prompt_builder.SYSTEM_PROMPT,
                )
                llm_result = validate_market_analysis_schema(raw_llm)
                llm_result, rec_warnings = sanitize_llm_payload(llm_result)
                warnings.extend(rec_warnings)
            except (InvalidResponseError, ProviderError, RuntimeError, TypeError, ValueError) as e:
                logger.warning("LLM interpretation failed; using deterministic fallback: %s", e)
                warnings.append(f"LLM interpretation unavailable: {e}")

            facts = self._deterministic_facts(primary_snap, now)
            for extra in self._objective_comparisons(snapshots):
                facts.append(Fact(
                    fact_id=generate_id("mfact"),
                    statement=extra,
                    source_ids=["MarketDataProvider"],
                    timestamp=now,
                    confidence=1.0,
                    data_type="multi_asset_comparison",
                ))

            for f in llm_result.get("facts", []):
                stmt = f.get("statement", "") if isinstance(f, dict) else ""
                if stmt and not any(stmt == existing.statement for existing in facts):
                    facts.append(Fact(
                        fact_id=generate_id("mfact"),
                        statement=stmt,
                        source_ids=["MarketDataProvider"],
                        timestamp=now,
                        confidence=float(f.get("confidence", 0.9)),
                        data_type=f.get("data_type", "market_metric"),
                    ))

            for anom in primary_snap.anomalies:
                warnings.append(f"Anomaly detected [{anom.anomaly_type.value}]: {anom.description}")
            warnings.extend(llm_result.get("warnings", []))

            sources = [
                Source(
                    source_id="MarketDataProvider",
                    title=f"Market Data Feed ({primary_snap.symbol})",
                    publisher="MarketDataProvider",
                    url="internal://market-data",
                    published_at=primary_snap.data_timestamp or now,
                    retrieved_at=primary_snap.retrieved_at or now,
                    is_primary=True,
                    reliability_score=1.0,
                )
            ]

            summary = ""
            analysis = llm_result.get("analysis", {})
            if isinstance(analysis, dict):
                summary = analysis.get("technical_summary", "") or ""
            if not summary:
                summary = (
                    f"{primary_snap.symbol} is in {primary_snap.regime.value} regime "
                    f"with {primary_snap.trend.value} trend."
                )
            if len(snapshots) > 1:
                multi_summary = " | ".join(f"{s.symbol}: {s.regime.value}" for s in snapshots)
                summary = f"Multi-Asset Overview ({multi_summary}). {summary}"

            analysis_out = analysis if isinstance(analysis, dict) else {}
            analysis_out = {
                **analysis_out,
                "snapshots": {
                    s.symbol: {
                        "trend": s.trend.value,
                        "regime": s.regime.value,
                        "current_price": s.current_price,
                        "return_1d": s.returns.return_1d,
                        "rsi_14": s.momentum.rsi_14,
                    }
                    for s in snapshots
                },
            }

            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now,
                status=AgentStatus.SUCCESS,
                summary=summary,
                confidence=float(llm_result.get("confidence", 0.90)),
                facts=facts,
                analysis=analysis_out,
                impact=llm_result.get("impact", {}),
                warnings=warnings,
                sources=sources,
                metadata={
                    "symbol": primary_snap.symbol,
                    "trend": primary_snap.trend.value,
                    "regime": primary_snap.regime.value,
                    "anomalies_count": len(primary_snap.anomalies),
                    "total_symbols_analyzed": len(snapshots),
                    "retrieved_at": (primary_snap.retrieved_at or now).isoformat(),
                    "data_timestamp": (
                        primary_snap.data_timestamp.isoformat()
                        if primary_snap.data_timestamp
                        else None
                    ),
                    "horizon": horizon,
                },
            )

            self._emit(MarketAnalysisCompleted(
                event_id=generate_id("evt_mkt"),
                timestamp=now,
                event_type="MarketAnalysisCompleted",
                agent_id=self.agent_id,
                task_id=task.task_id,
                symbols=list(symbols),
                primary_symbol=primary_snap.symbol,
                trend=primary_snap.trend.value,
                regime=primary_snap.regime.value,
                anomaly_count=len(primary_snap.anomalies),
            ))

            if self.memory_store:
                gen_id = task.input_data.get("generation_id", "gen_current")
                self.memory_store.save(MemoryRecord(
                    memory_id=generate_id("mem_market"),
                    memory_type=MemoryType.ANALYSIS,
                    generation_id=gen_id,
                    timestamp=now,
                    source_agent=self.agent_id,
                    importance=7,
                    content={
                        "symbol": primary_snap.symbol,
                        "current_price": primary_snap.current_price,
                        "trend": primary_snap.trend.value,
                        "regime": primary_snap.regime.value,
                        "summary": result.summary,
                    },
                    metadata=result.metadata,
                ))
                if primary_snap.anomalies:
                    self.memory_store.save(MemoryRecord(
                        memory_id=generate_id("mem_market_fact"),
                        memory_type=MemoryType.FACT,
                        generation_id=gen_id,
                        timestamp=now,
                        source_agent=self.agent_id,
                        importance=8,
                        content={
                            "symbol": primary_snap.symbol,
                            "anomalies": [a.anomaly_type.value for a in primary_snap.anomalies],
                        },
                        metadata={},
                    ))

            self.status = AgentStatus.SUCCESS
            return result

        except RateLimitError as e:
            return self._fail(task, now, "MARKET_DATA_RATE_LIMIT", e)
        except AuthenticationError as e:
            return self._fail(task, now, "MARKET_DATA_AUTH_ERROR", e)
        except APIUnavailableError as e:
            return self._fail(task, now, "MARKET_DATA_UNAVAILABLE", e)
        except InvalidResponseError as e:
            return self._fail(task, now, "MARKET_DATA_INVALID_RESPONSE", e)
        except ProviderError as e:
            return self._fail(task, now, "MARKET_DATA_PROVIDER_ERROR", e)
        except Exception as e:
            logger.error("Error in MarketResearchAgent: %s", e)
            return self._fail(task, now, "MARKET_RESEARCH_FAILURE", e)

    def _fail(self, task: Task, now: datetime, code: str, error: Exception) -> AgentResult:
        self.status = AgentStatus.FAILED
        if self.memory_store:
            gen_id = task.input_data.get("generation_id", "gen_current")
            self.memory_store.save(MemoryRecord(
                memory_id=generate_id("mem_market_err"),
                memory_type=MemoryType.ERROR,
                generation_id=gen_id,
                timestamp=now,
                source_agent=self.agent_id,
                importance=5,
                content={"error_code": code, "message": str(error)},
                metadata={},
            ))
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            timestamp=now,
            status=AgentStatus.FAILED,
            summary="Failed to process market research task.",
            confidence=0.0,
            errors=[ErrorInfo(error_code=code, message=str(error), details=type(error).__name__)],
        )

    def execute_order(self, *args, **kwargs):
        raise PermissionError("MarketResearchAgent is a research-only agent and cannot execute orders.")

    def modify_portfolio(self, *args, **kwargs):
        raise PermissionError("MarketResearchAgent cannot modify portfolio state.")

    def change_strategy(self, *args, **kwargs):
        raise PermissionError("MarketResearchAgent cannot alter investment strategies.")
