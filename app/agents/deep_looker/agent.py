from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from app.agents.base import BaseAgent
from app.agents.deep_looker.contradictions import DossierContradictionDetector, completeness_from_presence
from app.agents.deep_looker.metrics import FundamentalMetricsCalculator
from app.agents.deep_looker.prompt_builder import DeepLookerPromptBuilder
from app.agents.deep_looker.thesis import ThesisBuilder, collect_risks
from app.agents.market_research.recommendation_guard import sanitize_llm_payload
from app.core.memory.store import MemoryStore
from app.core.models.agent import AgentConfig, AgentResult, AgentStatus
from app.core.models.deep_research import (
    AssetIdentity,
    Catalyst,
    CatalystDirection,
    CompletenessLevel,
    DataCompleteness,
    DeepResearchDossier,
    DeepResearchRequest,
    ResearchDepth,
)
from app.core.models.error import ErrorInfo
from app.core.models.events import BaseEvent, DeepResearchCompleted
from app.core.models.knowledge import Evidence, Fact, Source, SourceType
from app.core.models.market import MarketSnapshot, capabilities_from_asset_type
from app.core.models.memory import MemoryRecord, MemoryType
from app.core.models.task import Task
from app.services.fundamental.provider import FundamentalDataProvider
from app.services.llm.provider import LLMProvider
from app.services.market_data.provider import MarketDataProvider
from app.services.news.provider import NewsProvider
from app.utils.ids import generate_id
from app.utils.logging import get_logger
from app.utils.time import now_utc

logger = get_logger(__name__)


class DeepLookerAgent(BaseAgent):
    """High-depth due-diligence agent.

    Research-only: it cannot approve investments, place orders, alter strategy, or mutate portfolio state.
    """

    def __init__(
        self,
        agent_id: str,
        llm_provider: LLMProvider,
        fundamental_provider: Optional[FundamentalDataProvider] = None,
        market_data_provider: Optional[MarketDataProvider] = None,
        news_provider: Optional[NewsProvider] = None,
        memory_store: Optional[MemoryStore] = None,
        configuration: Optional[AgentConfig] = None,
        event_publisher: Optional[Callable[[BaseEvent], None]] = None,
    ):
        config = configuration or AgentConfig(version="1.0.0", model=None, max_tokens=2000)
        super().__init__(
            agent_id=agent_id,
            agent_name="DeepLookerAgent",
            role="Deep Research and Investment Due-Diligence Analyst",
            description="Combines market, news, crisis, and fundamental research into a traceable dossier.",
            version="1.0.0",
            configuration=config,
            capabilities=[
                "asset_identification",
                "fundamental_research",
                "valuation_analysis",
                "contradiction_detection",
                "thesis_construction",
                "scenario_analysis",
                "source_traceability",
            ],
        )
        self.llm_provider = llm_provider
        self.fundamental_provider = fundamental_provider
        self.market_data_provider = market_data_provider
        self.news_provider = news_provider
        self.memory_store = memory_store
        self.event_publisher = event_publisher
        self.emitted_events: List[BaseEvent] = []
        self.metrics = FundamentalMetricsCalculator()
        self.contradictions = DossierContradictionDetector()
        self.thesis_builder = ThesisBuilder()
        self.prompt_builder = DeepLookerPromptBuilder()

    def _emit(self, event: BaseEvent) -> None:
        self.emitted_events.append(event)
        if self.event_publisher:
            self.event_publisher(event)

    def _request_from_task(self, task: Task) -> DeepResearchRequest:
        symbol = str(task.input_data.get("symbol", "")).strip().upper()
        if not symbol:
            raise ValueError("Deep research requires a non-empty asset symbol.")
        depth = task.input_data.get("requested_depth", ResearchDepth.DEEP)
        if isinstance(depth, str):
            depth = ResearchDepth(depth.upper())
        return DeepResearchRequest(
            symbol=symbol,
            analysis_timestamp=task.input_data.get("analysis_timestamp") or now_utc(),
            requested_depth=depth,
            historical_window=str(task.input_data.get("historical_window", "medium")),
            included_sources=list(task.input_data.get("included_sources", [])),
            generation_id=str(task.input_data.get("generation_id", "gen_current")),
            originating_task_id=str(task.input_data.get("originating_task_id", task.task_id)),
            optional_context=dict(task.input_data.get("optional_context", {})),
        )

    def _market_snapshot(self, symbol: str, task: Task, now) -> Optional[MarketSnapshot]:
        snapshot = task.input_data.get("market_snapshot")
        if isinstance(snapshot, MarketSnapshot):
            return snapshot
        market_result = task.input_data.get("market_research_result")
        if isinstance(market_result, AgentResult):
            snap = market_result.analysis.get("market_snapshot")
            if isinstance(snap, MarketSnapshot):
                return snap
        return None

    def _safe_provider_call(self, label: str, func, warnings: List[str]):
        try:
            return func()
        except Exception as e:
            logger.warning("%s unavailable: %s", label, e)
            warnings.append(f"{label} unavailable: {e}")
            return None

    def _sources_and_evidence(
        self,
        symbol: str,
        profile,
        statements,
        valuation,
        earnings,
        snapshot: Optional[MarketSnapshot],
        news_result: Optional[AgentResult],
        crisis_result: Optional[AgentResult],
    ) -> tuple[List[Source], List[Evidence]]:
        now = now_utc()
        sources: List[Source] = []
        evidence: List[Evidence] = []

        def add_source(src: Source) -> None:
            if not any(existing.source_id == src.source_id for existing in sources):
                sources.append(src)

        if any(x is not None for x in (profile, statements, valuation, earnings)):
            add_source(Source(
                source_id="FundamentalDataProvider",
                title=f"Fundamental Data Feed ({symbol})",
                publisher="FundamentalDataProvider",
                url="internal://fundamentals",
                published_at=getattr(statements, "period_end", None),
                retrieved_at=getattr(statements, "retrieved_at", None) or getattr(valuation, "retrieved_at", None) or now,
                source_type=SourceType.PRIMARY,
                is_primary=True,
                reliability_score=0.9,
            ))
            evidence.append(Evidence(
                evidence_id="financial_statements",
                description="Provider-supplied financial statements and valuation inputs.",
                source_id="FundamentalDataProvider",
                relevance_score=0.95,
            ))
        if snapshot is not None:
            add_source(Source(
                source_id="MarketDataProvider",
                title=f"Market Data Feed ({symbol})",
                publisher="MarketDataProvider",
                url="internal://market-data",
                published_at=snapshot.data_timestamp,
                retrieved_at=snapshot.retrieved_at or now,
                source_type=SourceType.PRIMARY,
                is_primary=True,
                reliability_score=1.0,
            ))
            evidence.append(Evidence(
                evidence_id="market_snapshot",
                description="Deterministic market snapshot from normalized market data.",
                source_id="MarketDataProvider",
                relevance_score=0.9,
            ))
        for result, evidence_id, desc in (
            (news_result, "news_research", "Structured output from News Research Agent."),
            (crisis_result, "crisis_research", "Structured output from Crisis Risk Agent."),
        ):
            if not isinstance(result, AgentResult):
                continue
            for src in result.sources:
                add_source(src)
            evidence.append(Evidence(
                evidence_id=evidence_id,
                description=desc,
                source_id=result.sources[0].source_id if result.sources else result.agent_id,
                relevance_score=0.8,
            ))
        return sources, evidence

    def _facts_from_llm(self, raw_facts: List[Dict[str, Any]], now) -> List[Fact]:
        facts: List[Fact] = []
        for f in raw_facts:
            if not isinstance(f, dict) or not f.get("statement"):
                continue
            source_id = f.get("source_id") or "deep_research_context"
            facts.append(Fact(
                fact_id=generate_id("dfact"),
                statement=str(f["statement"]),
                source_ids=[str(source_id)],
                timestamp=now,
                confidence=float(f.get("confidence", 0.4)),
                data_type=str(f.get("data_type", "due_diligence")),
            ))
        return facts

    def process_task(self, task: Task) -> AgentResult:
        self.status = AgentStatus.RUNNING
        now = now_utc()
        warnings: List[str] = []

        try:
            request = self._request_from_task(task)
            symbol = request.symbol
            if self.market_data_provider is not None:
                try:
                    self.market_data_provider.get_quote(symbol)
                except Exception as e:
                    warnings.append(f"Market quote unavailable: {e}")

            profile = statements = valuation = earnings = None
            if self.fundamental_provider is not None:
                profile = self._safe_provider_call("Company profile", lambda: self.fundamental_provider.get_company_profile(symbol), warnings)
                statements = self._safe_provider_call("Financial statements", lambda: self.fundamental_provider.get_financial_statements(symbol), warnings)
                valuation = self._safe_provider_call("Valuation metrics", lambda: self.fundamental_provider.get_valuation_metrics(symbol), warnings)
                earnings = self._safe_provider_call("Earnings data", lambda: self.fundamental_provider.get_earnings_data(symbol), warnings)

            snapshot = self._market_snapshot(symbol, task, now)
            news_result = task.input_data.get("news_research_result")
            crisis_result = task.input_data.get("crisis_research_result")
            if not isinstance(news_result, AgentResult):
                news_result = None
            if not isinstance(crisis_result, AgentResult):
                crisis_result = None

            capabilities = capabilities_from_asset_type(profile.asset_type if profile else task.input_data.get("asset_type", "UNKNOWN"))
            asset_identity = AssetIdentity(
                symbol=symbol,
                name=profile.company_name if profile else symbol,
                asset_type=capabilities.asset_type,
                sector=profile.sector if profile else "INSUFFICIENT_DATA",
                industry=profile.industry if profile else "INSUFFICIENT_DATA",
                country=profile.country if profile else "INSUFFICIENT_DATA",
                capabilities=capabilities,
                description=profile.description if profile else "",
            )

            market_price = snapshot.current_price if snapshot is not None else (valuation.market_price if valuation else None)
            health = self.metrics.calculate_health(statements)
            computed_valuation = self.metrics.calculate_valuation(statements, valuation, earnings, market_price)
            crisis_events = (crisis_result.analysis.get("events") if crisis_result else []) or []
            risks = collect_risks(
                capabilities,
                health,
                snapshot,
                crisis_events,
                (news_result.impact or {}).get("direction") if news_result else None,
            )
            contradictions = self.contradictions.detect(health, snapshot, news_result, crisis_result)

            completeness = DataCompleteness(
                fundamentals=completeness_from_presence(statements is not None, partial=bool(health.missing_fields)),
                market_data=completeness_from_presence(snapshot is not None),
                news=completeness_from_presence(news_result is not None, conflicting=bool(news_result and news_result.warnings)),
                geopolitical=completeness_from_presence(crisis_result is not None),
                valuation=completeness_from_presence(
                    any(v is not None for v in (
                        computed_valuation.pe_historical,
                        computed_valuation.ps,
                        computed_valuation.pb,
                        computed_valuation.ev_ebitda,
                        computed_valuation.fcf_yield,
                    )),
                    partial=bool(computed_valuation.missing_fields),
                ),
                notes=[],
            )
            if capabilities.asset_type == "UNKNOWN":
                completeness.notes.append("Asset type is UNKNOWN; sections are intentionally conservative.")
                warnings.append("Unsupported or unknown asset type; using conservative capabilities.")
            if health.missing_fields:
                completeness.notes.append(f"Missing deterministic metric inputs: {', '.join(health.missing_fields)}")
            if computed_valuation.context_labels and all(x.value == "INSUFFICIENT_DATA" for x in computed_valuation.context_labels):
                completeness.notes.append("Valuation context versus history or peers is INSUFFICIENT_DATA.")

            supporting: List[str] = []
            invalidating: List[str] = []
            if health.revenue_growth is not None and health.revenue_growth > 0:
                supporting.append("Revenue growth is positive in available statements.")
            if health.free_cash_flow_growth is not None and health.free_cash_flow_growth < 0:
                invalidating.append("Free cash flow growth is negative in available statements.")
            if snapshot is not None:
                supporting.append(f"Market trend is {snapshot.trend.value}.")
            if risks:
                invalidating.extend(r.description for r in risks[:3])

            thesis = self.thesis_builder.build(
                symbol,
                capabilities,
                health,
                snapshot,
                contradictions,
                risks,
                completeness,
                supporting,
                invalidating,
            )

            catalysts = []
            if earnings is not None:
                catalysts.append(Catalyst(
                    catalyst="Upcoming or recent earnings information is available for review.",
                    expected_timeframe="MEDIUM_TERM",
                    direction=CatalystDirection.UNCERTAIN,
                    evidence_ids=["financial_statements"],
                    uncertainty="Earnings outcome is not an investment decision.",
                    confidence=0.55,
                ))
            if news_result is not None:
                catalysts.append(Catalyst(
                    catalyst=news_result.summary,
                    expected_timeframe=str((news_result.impact or {}).get("horizon", "UNKNOWN")),
                    direction=CatalystDirection(str((news_result.impact or {}).get("direction", "UNCERTAIN")).upper())
                    if str((news_result.impact or {}).get("direction", "UNCERTAIN")).upper() in CatalystDirection.__members__
                    else CatalystDirection.UNCERTAIN,
                    evidence_ids=["news_research"],
                    uncertainty="News impact is contextual and may change with later reporting.",
                    confidence=news_result.confidence,
                ))

            sources, evidence = self._sources_and_evidence(
                symbol, profile, statements, valuation, earnings, snapshot, news_result, crisis_result
            )
            unknowns = []
            if statements is None:
                unknowns.append("Financial statements are unavailable or not applicable.")
            if valuation is None:
                unknowns.append("Provider valuation inputs are unavailable.")
            if news_result is None:
                unknowns.append("No structured News Research Agent result was supplied.")
            if crisis_result is None:
                unknowns.append("No structured Crisis Risk Agent result was supplied.")

            context = {
                "asset_identity": asdict(asset_identity),
                "market_snapshot": None if snapshot is None else {
                    "symbol": snapshot.symbol,
                    "current_price": snapshot.current_price,
                    "trend": snapshot.trend.value,
                    "regime": snapshot.regime.value,
                    "data_timestamp": snapshot.data_timestamp,
                    "retrieved_at": snapshot.retrieved_at,
                },
                "health_metrics": asdict(health),
                "valuation": asdict(computed_valuation),
                "news_result": None if news_result is None else {
                    "summary": news_result.summary,
                    "analysis": news_result.analysis,
                    "impact": news_result.impact,
                    "warnings": news_result.warnings,
                },
                "crisis_result": None if crisis_result is None else {
                    "summary": crisis_result.summary,
                    "analysis": crisis_result.analysis,
                    "impact": crisis_result.impact,
                    "warnings": crisis_result.warnings,
                },
                "risks": [asdict(r) for r in risks],
                "contradictions": [asdict(c) for c in contradictions],
                "unknowns": unknowns,
                "data_completeness": asdict(completeness),
            }

            llm_result: Dict[str, Any] = {
                "facts": [],
                "analysis": {
                    "synthesis": "Deterministic deep research dossier constructed; LLM synthesis unavailable.",
                    "thesis_commentary": thesis.story,
                    "contradiction_commentary": f"{len(contradictions)} unresolved contradiction(s) detected.",
                    "scenario_commentary": "Bull, base, and bear scenarios are represented structurally in the thesis.",
                },
                "impact": {"direction": "UNCERTAIN", "horizon": "MEDIUM_TERM", "affected_symbols": [symbol]},
                "confidence": 0.45,
                "warnings": ["LLM synthesis unavailable; deterministic dossier returned."],
            }
            try:
                raw = self.llm_provider.generate_structured(
                    prompt=self.prompt_builder.build(context),
                    schema={},
                    system_prompt=self.prompt_builder.SYSTEM_PROMPT,
                )
                llm_result, rec_warnings = sanitize_llm_payload(raw)
                warnings.extend(rec_warnings)
            except Exception as e:
                logger.warning("Deep Looker LLM synthesis failed: %s", e)
                warnings.append(f"LLM synthesis unavailable: {e}")

            facts = self._facts_from_llm(llm_result.get("facts", []), now)
            confidence_components = [
                float(llm_result.get("confidence", 0.45)),
                0.8 if statements else 0.25,
                0.8 if snapshot else 0.25,
                0.7 if sources else 0.2,
            ]
            if contradictions:
                confidence_components.append(max(0.1, 0.7 - 0.1 * len(contradictions)))
            confidence = max(0.0, min(1.0, sum(confidence_components) / len(confidence_components)))

            research_id = generate_id("deep")
            dossier = DeepResearchDossier(
                research_id=research_id,
                symbol=symbol,
                timestamp=now,
                request=request,
                asset_identity=asset_identity,
                business_or_asset_model={
                    "description": asset_identity.description,
                    "products": getattr(profile, "products", []) if profile else [],
                    "geographic_exposure": getattr(profile, "geographic_exposure", []) if profile else [],
                    "business_model_notes": getattr(profile, "business_model_notes", "") if profile else "INSUFFICIENT_DATA",
                },
                market_analysis={} if snapshot is None else {
                    "trend": snapshot.trend.value,
                    "regime": snapshot.regime.value,
                    "current_price": snapshot.current_price,
                    "data_timestamp": snapshot.data_timestamp.isoformat() if snapshot.data_timestamp else None,
                    "retrieved_at": snapshot.retrieved_at.isoformat() if snapshot.retrieved_at else None,
                },
                fundamental_analysis={
                    "statements_available": statements is not None,
                    "period_end": statements.period_end.isoformat() if statements else None,
                    "metrics": asdict(health),
                },
                valuation_analysis={
                    "metrics": asdict(computed_valuation),
                    "context": [x.value for x in computed_valuation.context_labels],
                },
                competitive_analysis={
                    "competitors": getattr(profile, "competitors", []) if profile else [],
                    "status": "INSUFFICIENT_DATA" if not profile or not profile.competitors else "AVAILABLE",
                },
                news_analysis={} if news_result is None else {
                    "summary": news_result.summary,
                    "impact": news_result.impact,
                    "warnings": news_result.warnings,
                },
                geopolitical_analysis={} if crisis_result is None else {
                    "summary": crisis_result.summary,
                    "events": crisis_events,
                    "impact": crisis_result.impact,
                },
                regulatory_analysis={
                    "status": "AVAILABLE_FROM_CRISIS_OR_NEWS" if crisis_result or news_result else "INSUFFICIENT_DATA"
                },
                risk_analysis=risks,
                catalysts=catalysts,
                contradictions=contradictions,
                unknowns=unknowns,
                thesis=thesis,
                evidence=evidence,
                confidence=confidence,
                sources=sources,
                completeness=completeness,
                health_metrics=health,
                computed_valuation=computed_valuation,
                governance_notes=[],
                metadata={
                    "pipeline": [
                        "asset_identification",
                        "data_collection",
                        "data_validation",
                        "existing_agent_results",
                        "fundamental_research",
                        "valuation_analysis",
                        "risk_analysis",
                        "contradiction_detection",
                        "catalyst_analysis",
                        "thesis_construction",
                        "llm_synthesis",
                        "evidence_validation",
                    ],
                    "llm_analysis": llm_result.get("analysis", {}),
                },
            )

            summary = str((llm_result.get("analysis") or {}).get("synthesis") or thesis.story)
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now,
                status=AgentStatus.SUCCESS,
                summary=summary,
                confidence=confidence,
                facts=facts,
                analysis={
                    "dossier": dossier,
                    "research_id": research_id,
                    "thesis_status": thesis.status.value,
                    "contradiction_count": len(contradictions),
                    "data_completeness": asdict(completeness),
                    "decision_context": dossier.to_decision_context(),
                },
                impact=llm_result.get("impact", {}),
                warnings=list(dict.fromkeys(warnings + llm_result.get("warnings", []))),
                sources=sources,
                metadata={
                    "symbol": symbol,
                    "research_id": research_id,
                    "thesis_status": thesis.status.value,
                    "requested_depth": request.requested_depth.value,
                    "no_investment_decision": True,
                },
            )

            self._emit(DeepResearchCompleted(
                event_id=generate_id("evt_deep"),
                timestamp=now,
                event_type="DeepResearchCompleted",
                research_id=research_id,
                asset=symbol,
                agent_id=self.agent_id,
                task_id=task.task_id,
                confidence=confidence,
                thesis_status=thesis.status.value,
                key_findings=[f.statement for f in facts[:3]] or supporting[:3],
                risk_summary="; ".join(r.description for r in risks[:3]),
            ))

            if self.memory_store:
                for content, mtype, importance in (
                    ({"symbol": symbol, "research_id": research_id, "thesis_status": thesis.status.value, "assumptions": [a.assumption for a in thesis.assumptions]}, MemoryType.ANALYSIS, 7),
                    ({"symbol": symbol, "invalidation_conditions": thesis.invalidation_conditions}, MemoryType.FACT, 8),
                    ({"symbol": symbol, "contradictions": [asdict(c) for c in contradictions]}, MemoryType.FACT, 8),
                ):
                    if content.get("contradictions") == []:
                        continue
                    self.memory_store.save(MemoryRecord(
                        memory_id=generate_id("mem_deep"),
                        memory_type=mtype,
                        generation_id=request.generation_id,
                        timestamp=now,
                        source_agent=self.agent_id,
                        importance=importance,
                        content=content,
                        metadata={"research_id": research_id, "symbol": symbol},
                    ))

            self.status = AgentStatus.SUCCESS
            return result

        except Exception as e:
            logger.error("DeepLookerAgent failure: %s", e)
            self.status = AgentStatus.FAILED
            return AgentResult(
                agent_id=self.agent_id,
                task_id=task.task_id,
                timestamp=now,
                status=AgentStatus.FAILED,
                summary="Failed to process deep research task.",
                confidence=0.0,
                errors=[ErrorInfo(error_code="DEEP_RESEARCH_FAILURE", message=str(e), details=type(e).__name__)],
                warnings=warnings,
            )

    def execute_order(self, *args, **kwargs):
        raise PermissionError("DeepLookerAgent is research-only and cannot execute orders.")

    def modify_portfolio(self, *args, **kwargs):
        raise PermissionError("DeepLookerAgent cannot modify portfolio state.")

    def modify_capital(self, *args, **kwargs):
        raise PermissionError("DeepLookerAgent cannot modify capital.")

    def approve_investment(self, *args, **kwargs):
        raise PermissionError("DeepLookerAgent cannot approve investments.")

    def override_risk_manager(self, *args, **kwargs):
        raise PermissionError("DeepLookerAgent cannot override the Risk Manager.")

    def change_strategy(self, *args, **kwargs):
        raise PermissionError("DeepLookerAgent cannot alter investment strategies.")
